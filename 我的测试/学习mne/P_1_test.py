import mne
import numpy as np
import matplotlib.pyplot as plt

"""
生成一个大小为5x1000的二维随机数据
其中5代表5个通道，1000代表times
"""
# data: [5, 1000]=[n_channels, n_times]
data = np.random.randn(5, 1000)

"""
创建info结构,
内容包括：通道名称和通道类型
设置采样频率为:sfreq=100
"""
info = mne.create_info(
    ch_names=['MEG1', 'MEG2', 'EEG1', 'EEG2', 'EOG'],
    ch_types=['grad', 'grad', 'eeg', 'eeg', 'eog'],
    sfreq=100
)
"""
利用mne.io.RawArray类创建Raw对象
"""
custom_raw = mne.io.RawArray(data, info)
print(custom_raw)