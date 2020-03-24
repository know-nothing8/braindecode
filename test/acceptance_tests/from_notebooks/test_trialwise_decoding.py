from braindecode.datautil.signal_target import SignalAndTarget
import numpy as np
import mne
from mne.io import concatenate_raws
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from torch import nn
from braindecode.torch_ext.util import set_random_seeds
from braindecode.torch_ext.util import np_to_var, var_to_np
from braindecode.datautil.iterators import get_balanced_batches
import torch.nn.functional as F
from numpy.random import RandomState
from torch import optim


def test_trialwise_decoding():
    # 5,6,7,10,13,14 are codes for executed and imagined hands/feet
    subject_id = 1
    event_codes = [5, 6, 9, 10, 13, 14]
    # event_codes = [6]

    # This will download the files if you don't have them yet,
    # and then return the paths to the files.
    physionet_paths = mne.datasets.eegbci.load_data(subject_id, event_codes)

    # Load each of the files
    parts = [mne.io.read_raw_edf(path, preload=True, stim_channel='auto', verbose='WARNING') for path in physionet_paths]

    # Concatenate them
    raw = concatenate_raws(parts)

    # Find the events in this dataset
    # events = mne.find_events(raw, shortest_event=0, stim_channel='STI 014')
    events, _ = mne.events_from_annotations(raw)

    # Extract trials, only using EEG channels
    eeg_channel_inds = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    # Extract trials, only using EEG channels
    epoched = mne.Epochs(raw, events, dict(hands=2, feet=3), tmin=1, tmax=4.1, proj=False, picks=eeg_channel_inds, baseline=None, preload=True)

    # Convert data from volt to millivolt
    # Pytorch expects float32 for input and int64 for labels.
    # X:[90,64,497]
    X = (epoched.get_data() * 1e6).astype(np.float32)
    # y:[90]
    y = (epoched.events[:, 2] - 2).astype(np.int64)  # 2,3 -> 0,1

    # X_train:[60,64,497], y_train:[60]
    train_set = SignalAndTarget(X[:60], y=y[:60])
    # X_test:[30,64,497], y_test:[30]
    test_set = SignalAndTarget(X[60:], y=y[60:])

    # Set if you want to use GPU
    # You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
    cuda = False
    set_random_seeds(seed=20170629, cuda=cuda)
    n_classes = 2
    in_chans = train_set.X.shape[1]
    # final_conv_length = auto ensures we only get a single output in the time dimension
    # def __init__(self, in_chans=64, n_classes=2, input_time_length=497, n_filters_time=40, filter_time_length=25, n_filters_spat=40, pool_time_length=75, pool_time_stride=15, final_conv_length='auto, conv_nonlin=square, pool_mode="mean", pool_nonlin=safe_log, split_first_layer=True, batch_norm=True, batch_norm_alpha=0.1, drop_prob=0.5, ):
    # 感觉create_network()就是__init__的一部分, 现在改成用self.model调用了, 还是感觉不优雅, 主要是forward集成在nn.Sequential里面了
    # 然后这个model的实际__init__不是ShallowFBCSPNet, 而是nn.Sequential, 感觉我更喜欢原来的定义方式, 这种方式看不到中间输出
    # model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes, input_time_length=train_set.X.shape[2], final_conv_length='auto').create_network() #原来的
    model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes, input_time_length=train_set.X.shape[2], final_conv_length='auto').model
    if cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters())

    rng = RandomState((2017, 6, 30))
    losses = []
    accuracies = []
    for i_epoch in range(6):
        i_trials_in_batch = get_balanced_batches(len(train_set.X), rng, shuffle=True, batch_size=10)
        # Set model to training mode
        model.train()
        for i_trials in i_trials_in_batch:
            # Have to add empty fourth dimension to X
            batch_X = train_set.X[i_trials][:, :, :, None]
            batch_y = train_set.y[i_trials]
            net_in = np_to_var(batch_X)
            if cuda:
                net_in = net_in.cuda()
            net_target = np_to_var(batch_y)
            if cuda:
                net_target = net_target.cuda()
            # Remove gradients of last backward pass from all parameters
            optimizer.zero_grad()
            # Compute outputs of the network
            #net_in: [10, 64, 497, 1]=[bsz, H_im, W_im, C_im]
            #
            outputs = model.forward(net_in)
            # model=Sequential(
            #                   (dimshuffle): Expression(expression=_transpose_time_to_spat)
            #                   (conv_time): Conv2d(1, 40, kernel_size=(25, 1), stride=(1, 1))
            #                   (conv_spat): Conv2d(40, 40, kernel_size=(1, 64), stride=(1, 1), bias=False)
            #                   (bnorm): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            #                   (conv_nonlin): Expression(expression=square)
            #                   (pool): AvgPool2d(kernel_size=(75, 1), stride=(15, 1), padding=0)
            #                   (pool_nonlin): Expression(expression=safe_log)
            #                   (drop): Dropout(p=0.5)
            #                   (conv_classifier): Conv2d(40, 2, kernel_size=(27, 1), stride=(1, 1))
            #                   (softmax): LogSoftmax()
            #                   (squeeze): Expression(expression=_squeeze_final_output)
            #                 )
            # Compute the loss
            loss = F.nll_loss(outputs, net_target)
            # Do the backpropagation
            loss.backward()
            # Update parameters with the optimizer
            optimizer.step()

        # Print some statistics each epoch
        model.eval()
        print("Epoch {:d}".format(i_epoch))
        for setname, dataset in (('Train', train_set), ('Test', test_set)):
            # Here, we will use the entire dataset at once, which is still possible
            # for such smaller datasets. Otherwise we would have to use batches.
            net_in = np_to_var(dataset.X[:, :, :, None])
            if cuda:
                net_in = net_in.cuda()
            net_target = np_to_var(dataset.y)
            if cuda:
                net_target = net_target.cuda()
            outputs = model(net_in)
            loss = F.nll_loss(outputs, net_target)
            losses.append(float(var_to_np(loss)))
            print("{:6s} Loss: {:.5f}".format(
                setname, float(var_to_np(loss))))
            predicted_labels = np.argmax(var_to_np(outputs), axis=1)
            accuracy = np.mean(dataset.y == predicted_labels)
            accuracies.append(accuracy * 100)
            print("{:6s} Accuracy: {:.1f}%".format(
                setname, accuracy * 100))

    np.testing.assert_allclose(
        np.array(losses),
        np.array([1.1775966882705688,
                  1.2602351903915405,
                  0.7068756818771362,
                  0.9367912411689758,
                  0.394258975982666,
                  0.6598362326622009,
                  0.3359280526638031,
                  0.656258761882782,
                  0.2790488004684448,
                  0.6104397177696228,
                  0.27319177985191345,
                  0.5949864983558655]),
        rtol=1e-4, atol=1e-5)

    np.testing.assert_allclose(
        np.array(accuracies),
        np.array(
            [51.666666666666671,
             53.333333333333336,
             63.333333333333329,
             56.666666666666664,
             86.666666666666671,
             66.666666666666657,
             90.0,
             63.333333333333329,
             96.666666666666671,
             56.666666666666664,
             96.666666666666671,
             66.666666666666657]),
        rtol=1e-4, atol=1e-5)


test_trialwise_decoding()
