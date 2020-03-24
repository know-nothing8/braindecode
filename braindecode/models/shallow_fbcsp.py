import numpy as np
from torch import nn
from torch.nn import init
from braindecode.models.base import BaseModel
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.functions import safe_log, square
from braindecode.torch_ext.util import np_to_var


class ShallowFBCSPNet(BaseModel):
    """
    Shallow ConvNet model from [2]_.
    References
    ----------
     .. [2] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017). Deep learning with convolutional neural networks for EEG decoding and        visualization. Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(self, in_chans=64, n_classes=2, input_time_length=497, n_filters_time=40, filter_time_length=25, n_filters_spat=40, pool_time_length=75, pool_time_stride=15, final_conv_length='auto', conv_nonlin=square, pool_mode="mean", pool_nonlin=safe_log, split_first_layer=True, batch_norm=True, batch_norm_alpha=0.1, drop_prob=0.5, ):
        if final_conv_length == "auto":
            assert input_time_length is not None
        # self.__dict__.update(locals())
        # del self.self

        # net_in: [10, 64, 497, 1]=[bsz, H_im, W_im, C_im]
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]
        model = nn.Sequential()
        if split_first_layer:
            # [10,64,497,1]-->[10,1,497,64]
            model.add_module("dimshuffle", Expression(_transpose_time_to_spat))
            # C_in=1, C_o=40, H_f=25, W_f=1, s=1
            # in:[10,1,497,64]=[N,C_in,H_in,W_in]
            # f: [C_f, H_f, W_f]=[1,25,1];  n_f=40; s=1
            # out: [N, C_o,H_o,W_o]=[10,40,473,64]
            # 宽过滤器: W像时间轴, 所以每次移动1步, 然后H像一个时间步的数据, 用于分析
            model.add_module("conv_time", nn.Conv2d(in_channels=1, out_channels=n_filters_time, kernel_size=(filter_time_length, 1), stride=1, ), )
            # C_in=40, C_o=40, H_f=1, W_f=64, s=1
            # in: [N, C_i,H_i,W_i]=[10,40,473,64]
            # f: [C_f, H_f, W_f]=[40,1,64];  n_f=40; s=1
            # out: [N, C_o,H_o,W_o]=[10,40,473,1]
            # 长过滤器: 这次换到在W轴移动了
            model.add_module("conv_spat",nn.Conv2d(in_channels=n_filters_time, out_channels=n_filters_spat, kernel_size=(1, in_chans), stride=1,bias=not batch_norm, ), )
            n_filters_conv = n_filters_spat
        else:
            # C_in=64, C_o=n_f=40, H_f=25, W_f=1, s=1
            # in: [10,64,497,1]=[N, C_in, H_in, W_in]
            # f:  [C_f, H_f, W_f]=[64,25,1];  n_f=40; s=1
            # out:[N, C_o,H_o,W_o]=[10,40,473,1]
            # 结果和上面竟完全一致, 什么情况???
            model.add_module("conv_time",nn.Conv2d(in_channels=in_chans, out_channels=n_filters_time, kernel_size=(filter_time_length, 1), stride=1,bias=not batch_norm, ), )
            n_filters_conv = n_filters_time
        if batch_norm:
            model.add_module("bnorm", nn.BatchNorm2d(num_features=n_filters_conv, momentum=batch_norm_alpha, affine=True), )
        # conv_nonlin=square, return x*x
        tmp=Expression(conv_nonlin)
        model.add_module("conv_nonlin", Expression(conv_nonlin))
        # in:[N, C_in, H_in, W_in]=[10,40,473,1]
        # f: C_f=40, H_f=75, W_f=1, s_H=15, s_W=1
        # out:[N, C_o,H_o,W_o]=[10,40,398,1]
        model.add_module("pool",pool_class(kernel_size=(pool_time_length, 1), stride=(pool_time_stride, 1), ), )
        model.add_module("pool_nonlin", Expression(pool_nonlin))
        model.add_module("drop", nn.Dropout(p=drop_prob))
        model.eval()
        if final_conv_length == "auto":
            out = model(np_to_var(np.ones((1, in_chans, input_time_length, 1), dtype=np.float32, )))
            n_out_time = out.cpu().data.numpy().shape[2]
            final_conv_length = n_out_time
        model.add_module("conv_classifier",nn.Conv2d(n_filters_conv, n_classes, (final_conv_length, 1), bias=True, ), )
        model.add_module("softmax", nn.LogSoftmax(dim=1))
        model.add_module("squeeze", Expression(_squeeze_final_output))

        # Initialization, xavier is same as in paper...
        init.xavier_uniform_(model.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        if split_first_layer or (not batch_norm):
            init.constant_(model.conv_time.bias, 0)
        if split_first_layer:
            init.xavier_uniform_(model.conv_spat.weight, gain=1)
            if not batch_norm:
                init.constant_(model.conv_spat.bias, 0)
        if batch_norm:
            init.constant_(model.bnorm.weight, 1)
            init.constant_(model.bnorm.bias, 0)
        init.xavier_uniform_(model.conv_classifier.weight, gain=1)
        init.constant_(model.conv_classifier.bias, 0)
        self.model=model


# remove empty dim at end and potentially remove empty time dim
# do not just use squeeze as we never want to remove first dim
def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


def _transpose_time_to_spat(x):
    # [10,64,497,1]-->[10,1,497,64]
    return x.permute(0, 3, 2, 1)
