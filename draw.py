import torch
import matplotlib.pyplot as plt
import librosa
import os
import glob
import numpy as np
from utils.metrics import compareone

def draw_spec(self, esti_list, label_list, frame_list, feat_type='sqrt'):
    with torch.no_grad():
        esti_mag, esti_phase = torch.norm(esti_list, dim=1), torch.atan2(esti_list[:, -1, :, :],
                                                                         esti_list[:, 0, :, :])
        label_mag, label_phase = torch.norm(label_list, dim=1), torch.atan2(label_list[:, -1, :, :],
                                                                            label_list[:, 0, :, :])
        if feat_type == 'sqrt':
            esti_mag = esti_mag ** 2
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            label_mag = label_mag ** 2
            label_com = torch.stack((label_mag * torch.cos(label_phase), label_mag * torch.sin(label_phase)), dim=1)
        elif feat_type == 'cubic':
            esti_mag = esti_mag ** (10 / 3)
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            label_mag = label_mag ** (10 / 3)
            label_com = torch.stack((label_mag * torch.cos(label_phase), label_mag * torch.sin(label_phase)), dim=1)
        elif feat_type == 'log_1x':
            esti_mag = torch.exp(esti_mag) - 1
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            label_mag = torch.exp(label_mag) - 1
            label_com = torch.stack((label_mag * torch.cos(label_phase), label_mag * torch.sin(label_phase)), dim=1)
        else:
            esti_com = esti_list
            label_com = label_list
        clean_utts, esti_utts = [], []
        utt_num = label_list.size()[0]
        for i in range(utt_num):
            # print("utt_num: ", i)
            tf_esti = esti_com[i, :, :, :].unsqueeze(dim=0).permute(0, 3, 2, 1).cpu()
            t_esti = torch.istft(tf_esti, n_fft=320, hop_length=160, win_length=320,
                                 window=torch.hann_window(320)).transpose(1, 0).squeeze(dim=-1).numpy()
            tf_label = label_com[i, :, :, :].unsqueeze(dim=0).permute(0, 3, 2, 1).cpu()
            t_label = torch.istft(tf_label, n_fft=320, hop_length=160, win_length=320,
                                  window=torch.hann_window(320)).transpose(1, 0).squeeze(dim=-1).numpy()
            t_len = (frame_list[i] - 1) * 160
            t_esti, t_label = t_esti[:t_len], t_label[:t_len]
            # draw one test

            print(t_esti.shape)
            compareone((t_esti, t_label))
            # plt setting
            dynamicRange = 100
            vmin = 20 * np.log10(np.max(t_label)) - dynamicRange

            f, ax = plt.subplots(1, figsize=[24, 4])

            _, _, _, pcm = ax.specgram(t_esti, NFFT=512, Fs=16000, vmin=vmin, cmap='inferno')
            # plt.colorbar(pcm, ax=ax[0], format='%+2.0f dB')
            ax.set_title("noisy_audio")
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

            esti_utts.append(t_esti)
            clean_utts.append(t_label)

def draw_wav(index):
    # wav_root = '/home/kevin/code/CDiffuSE/Output/Enhanced/cdiffuse/model507600/test'
    noisy_root = '/home/kevin/code/nips2022/data/noisy_testset_wav'
    clean_root = '/home/kevin/code/nips2022/data/clean_testset_wav'
    enhance_cdiffuse_root = '/home/kevin/code/CDiffuSE/Output/Enhanced/cdiffuse/model507600/test'
    enhance_priorDiffuse_root = '/home/kevin/code/nips2022/asset_priorDiffuse_sigma/wav/diff'

    raw_paths = [x.split('/')[-1] for x in glob.glob(noisy_root + '/*.wav')]
    print(raw_paths)
    file_name = raw_paths[index]
    data_no, _ = librosa.load(os.path.join(noisy_root, file_name), sr=16000)
    data_cl, _ = librosa.load(os.path.join(clean_root, file_name), sr=16000)
    data_en_c, _ = librosa.load(os.path.join(enhance_cdiffuse_root, file_name), sr=16000)
    data_en_p, _ = librosa.load(os.path.join(enhance_priorDiffuse_root, file_name), sr=16000)

    # batch_result = compareone((data_en, data_cl))
    # print(batch_result)
    dynamicRange = 100
    vmin = 20*np.log10(np.max(data_cl)) - dynamicRange
    print(raw_paths[index]) # *.wav

    default_font = {'family': 'Times New Roman', 'weight': 'normal', 'size':30}
    tick_font = 16
    y=-0.19
    f,ax=plt.subplots(1,4,figsize=[20,4], dpi=300)
    # f.subplots_adjust(top=0.97, bottom=0.2, left=0.05, right=0.995, hspace=0.1, wspace=0.2)
    f.subplots_adjust( top=0.9, bottom=0.15, left=0.03, right=0.97, hspace=0.4, wspace=0.1)

    _, _, _, pcm = ax[0].specgram(data_no,NFFT=512, Fs=16000, vmin=vmin, cmap='inferno')
    # plt.colorbar(pcm, ax=ax[1], format='%+2.0f dB')
    ax[0].set_title("(a) noisy input",default_font, y=y)
    ax[0].axes.xaxis.set_visible(False)
    ax[0].axes.yaxis.set_visible(False)

    _,_,_,pcm =  ax[1].specgram(data_cl, NFFT=512, Fs=16000,vmin=vmin, cmap='inferno')
    # plt.colorbar(pcm, ax=ax[0], format='%+2.0f dB')
    ax[1].set_title("(b) clean target",default_font, y=y)
    ax[1].axes.xaxis.set_visible(False)
    ax[1].axes.yaxis.set_visible(False)

    _, _, _, pcm = ax[2].specgram(data_en_c, NFFT=512, Fs=16000, vmin=vmin, cmap='inferno')
    # plt.colorbar(pcm, ax=ax[2], format='%+2.0f dB')
    ax[2].set_title("(c) CDiffuSE",default_font, y=y)
    ax[2].axes.xaxis.set_visible(False)
    ax[2].axes.yaxis.set_visible(False)

    _, _, _, pcm = ax[3].specgram(data_en_p, NFFT=512, Fs=16000, vmin=vmin, cmap='inferno')
    # plt.colorbar(pcm, ax=ax[2], format='%+2.0f dB')
    ax[3].set_title("(d) PriorDiffuse",default_font, y=y)
    ax[3].axes.xaxis.set_visible(False)
    ax[3].axes.yaxis.set_visible(False)

    plt.savefig('vsCDiffuse_6.pdf', bbox_inches='tight')
    plt.show()
if __name__ == '__main__':
    draw_wav(6)
    # great example: 4, 6, 26