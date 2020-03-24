import mne
from mne.datasets import sample
import os
import matplotlib.pyplot as plt

# sample的存放地址
data_path = sample.data_path()
# 该fif文件存放地址
fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

"""
如果上述给定的地址中存在该文件，则直接加载本地文件，
如果不存在则在网上下载改数据
"""
raw = mne.io.read_raw_fif(fname)
print(raw)
print(raw.info)
# todo ....
