from torch.utils.data import Dataset
import glob
import soundfile as sf
import os
import torch.nn as nn
import random
import torch
import numpy as np
import librosa


class ToTensor(object):
    def __call__(self, x, type='float'):
        if type == 'float':
            return torch.FloatTensor(x)
        elif type == 'int':
            return torch.IntTensor(x)


class BatchInfo(object):
    def __init__(self, noisy, clean, frame_num_list, wav_len_list):
        self.feats = noisy
        self.labels = clean
        self.frame_num_list = frame_num_list
        self.wav_len_list = wav_len_list


class Collate(object):
    def __init__(self, config):
        self.win_size = config.train.win_size
        self.fft_num = config.train.fft_num
        self.win_shift = config.train.win_shift

    @staticmethod
    def normalize(x):
        return x / np.max(abs(x))

    def collate_fn(self, batch):
        noisy_list, clean_list, frame_num_list, wav_len_list = [], [], [], []
        to_tensor = ToTensor()
        for sample in batch:
            '''normalize in DBAIAT and DARCN'''
            c = np.sqrt(len(sample[0]) / np.sum(sample[0] ** 2.0))
            noisy_list.append(to_tensor(sample[0] * c))
            clean_list.append(to_tensor(sample[1] * c))
            '''normalize in DiffWave'''
            # noisy_list.append(to_tensor(self.normalize(sample[0])))
            # clean_list.append(to_tensor(self.normalize(sample[1])))
            '''No normalize'''
            # noisy_list.append(to_tensor(sample[0]))
            # clean_list.append(to_tensor(sample[1]))
            frame_num_list.append(sample[2])
            wav_len_list.append(sample[3])
        noisy_list = nn.utils.rnn.pad_sequence(noisy_list, batch_first=True)  # [B, chunk_length]
        clean_list = nn.utils.rnn.pad_sequence(clean_list, batch_first=True)  # [B, chunk_length]
        noisy_list = torch.stft(
            noisy_list,
            n_fft=self.fft_num,
            hop_length=self.win_shift,
            win_length=self.win_size,
            window=torch.hann_window(self.fft_num)
        ).permute(0, 3, 2, 1)  # [B, 2, T, F]
        clean_list = torch.stft(
            clean_list,
            n_fft=self.fft_num,
            hop_length=self.win_shift,
            win_length=self.win_size,
            window=torch.hann_window(self.fft_num)
        ).permute(0, 3, 2, 1)  # [B, 2, T, F]

        return BatchInfo(noisy_list, clean_list, frame_num_list, wav_len_list)


class VBDataset(Dataset):
    def __init__(self, noisy_root, clean_root, config):
        super(VBDataset, self).__init__()
        self.noisy_root = noisy_root
        self.clean_root = clean_root
        self.chunk_length = config.train.chunk_length
        self.win_size = config.train.win_size
        self.fft_num = config.train.fft_num
        self.win_shift = config.train.win_shift
        self.raw_paths = [x.split('/')[-1] for x in glob.glob(noisy_root + '/*.wav')]

    def __len__(self):
        return len(self.raw_paths)

    def __getitem__(self, index):
        noisy, _ = librosa.load(os.path.join(self.noisy_root, self.raw_paths[index]), sr=16000)
        clean, _ = librosa.load(os.path.join(self.clean_root, self.raw_paths[index]), sr=16000)
        if len(noisy) > self.chunk_length:
            wav_start = random.randint(0, len(noisy) - self.chunk_length)
            noisy = noisy[wav_start:wav_start + self.chunk_length]
            clean = clean[wav_start:wav_start + self.chunk_length]
        wav_len = len(noisy)
        frame_num = (len(noisy) - self.win_size + self.fft_num) // self.win_shift + 1
        return noisy, clean, frame_num, wav_len


class VBTrDataset(Dataset):
    def __init__(self, noisy_root, clean_root, config):
        super(VBTrDataset, self).__init__()
        self.noisy_root = noisy_root
        self.clean_root = clean_root
        self.chunk_length = config.train.chunk_length
        self.win_size = config.train.win_size
        self.fft_num = config.train.fft_num
        self.win_shift = config.train.win_shift
        self.raw_paths = [x.split('/')[-1] for x in glob.glob(noisy_root + '/*.wav')]

    def __len__(self):
        return len(self.raw_paths)

    def __getitem__(self, index):
        noisy, _ = librosa.load(os.path.join(self.noisy_root, self.raw_paths[index]), sr=16000)
        clean, _ = librosa.load(os.path.join(self.clean_root, self.raw_paths[index]), sr=16000)
        if len(noisy) > self.chunk_length:
            wav_start = random.randint(0, len(noisy) - self.chunk_length)
            noisy = noisy[wav_start:wav_start + self.chunk_length]
            clean = clean[wav_start:wav_start + self.chunk_length]
        wav_len = len(noisy)
        frame_num = (len(noisy) - self.win_size + self.fft_num) // self.win_shift + 1
        return noisy, clean, frame_num, wav_len


class VBCvDataset(Dataset):
    def __init__(self, noisy_root, clean_root, config):
        super(VBCvDataset, self).__init__()
        self.noisy_root = noisy_root
        self.clean_root = clean_root
        self.chunk_length = config.train.chunk_length
        self.win_size = config.train.win_size
        self.fft_num = config.train.fft_num
        self.win_shift = config.train.win_shift
        self.raw_paths = [x.split('/')[-1] for x in glob.glob(noisy_root + '/*.wav')]

    def __len__(self):
        return len(self.raw_paths)

    def __getitem__(self, index):
        noisy, _ = librosa.load(os.path.join(self.noisy_root, self.raw_paths[index]), sr=16000)
        clean, _ = librosa.load(os.path.join(self.clean_root, self.raw_paths[index]), sr=16000)
        wav_len = len(noisy)
        frame_num = (len(noisy) - self.win_size + self.fft_num) // self.win_shift + 1
        return noisy, clean, frame_num, wav_len
