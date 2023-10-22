import hyperparams as hp
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
from text import text_to_sequence
import collections
from scipy import signal
import torch as t
import math

class LJDatasets(Dataset):
    def __init__(self, csv_file, root_dir):
        # csv_file (string) 文本数据的路径
        # root_dir (string) 音频数据的路径
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir

    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sample_rate)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0]) + '.wav' # idx对应的音频文件路径
        text = self.landmarks_frame.iloc[idx, 1] # idx对应的文本内容
        # 将英文文本转为序列，相当于字符级别的分词，在最后都会加上一个1
        text = np.asarray(text_to_sequence(text, [hp.cleaners]), dtype=np.int32)
        mel = np.load(wav_name[:-4] + '.pt.npy') # 加载 mel 谱图
        # 将 [[0] * 80] 与 mel 中的前 n-1 行在垂直方向 concat，即去掉 mel 的最后一行，并且在最前面添加全为 0 的一行，作为输入
        mel_input = np.concatenate([np.zeros([1,hp.num_mels], np.float32), mel[:-1,:]], axis=0)
        text_length = len(text) # 序列长度
        pos_text = np.arange(1, text_length + 1) # 位置编码
        pos_mel = np.arange(1, mel.shape[0] + 1)
        sample = {'text': text, 'mel': mel, 'text_length':text_length, 'mel_input':mel_input, 'pos_mel':pos_mel, 'pos_text':pos_text}
        return sample

# 用于加载 mel 图谱和 mag 谱图数据
class PostDatasets(Dataset):
    def __init__(self, csv_file, root_dir):
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0]) + '.wav'
        mel = np.load(wav_name[:-4] + '.pt.npy')
        mag = np.load(wav_name[:-4] + '.mag.npy')
        sample = {'mel':mel, 'mag':mag}

        return sample

# 用于对 LJDatasets 类构建的数据进行 batch 中的转换处理
def collate_fn_transformer(batch):
    if isinstance(batch[0], collections.Mapping):
        text = [d['text'] for d in batch] # batch 中所有的文本数据
        mel = [d['mel'] for d in batch]  # batch 中所有的 mel 数据
        mel_input = [d['mel_input'] for d in batch] # batch 中所有的 mel_input
        text_length = [d['text_length'] for d in batch] # batch 中所有的 test_length
        pos_mel = [d['pos_mel'] for d in batch] # batch 中所有的 pos_mel
        pos_text= [d['pos_text'] for d in batch] # batch 中所有的 pos_text

        # 将每个 text 与其对应的长度 text_length 匹配，以长度为标准对 text 进行降序排序，最后的列表中只取 text
        text = [i for i,_ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
        # 将每个 mel 与其对应的长度 text_length 匹配，以长度为标准对 mel 进行降序排序，最后的列表中只取 mel
        mel = [i for i, _ in sorted(zip(mel, text_length), key=lambda x: x[1], reverse=True)]
        # 下面几项也是如此，就是以 text_length 的大小进行降序排序
        mel_input = [i for i, _ in sorted(zip(mel_input, text_length), key=lambda x: x[1], reverse=True)]
        pos_text = [i for i, _ in sorted(zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
        pos_mel = [i for i, _ in sorted(zip(pos_mel, text_length), key=lambda x: x[1], reverse=True)]
        text_length = sorted(text_length, reverse=True)
        text = _prepare_data(text).astype(np.int32)# 用 0 将 text 中的每个文本序列都 pad 到最长的文本序列的长度
        mel = _pad_mel(mel)# 对 mel 进行 pad
        mel_input = _pad_mel(mel_input)# 对 mel_input 进行 pad
        pos_mel = _prepare_data(pos_mel).astype(np.int32)# 用 0 将 pos_mel 中的每个序列都 pad 到最长的序列的长度
        pos_text = _prepare_data(pos_text).astype(np.int32)# 用 0 将 pos_text 中的每个序列都 pad 到最长的序列的长度
        return t.LongTensor(text), t.FloatTensor(mel), t.FloatTensor(mel_input), t.LongTensor(pos_text), t.LongTensor(pos_mel), t.LongTensor(text_length)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

# 用于对 PostDatasets 类构建的数据进行 batch 中的转换处理
def collate_fn_postnet(batch):
    if isinstance(batch[0], collections.Mapping):
        mel = [d['mel'] for d in batch]
        mag = [d['mag'] for d in batch]
        mel = _pad_mel(mel)
        mag = _pad_mel(mag)
        return t.FloatTensor(mel), t.FloatTensor(mag)
    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

# 使用0对输出的x进行pad到指定长度length
def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

# 将inputs中所有的序列用0pad到其中最长序列的长度
def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])

# 将一个batch中所有的mel用0pad到其中最大长度的大小
def _pad_mel(inputs):
    _pad = 0
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0,max_len - mel_len],[0,0]], mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])

# 计算模型的参数大小
def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params

def get_dataset():
    return LJDatasets(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))

def get_post_dataset():
    return PostDatasets(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))


def _pad_per_step(inputs):
    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0,0],[0,0],[0, hp.outputs_per_step - (timesteps % hp.outputs_per_step)]], mode='constant', constant_values=0.0)