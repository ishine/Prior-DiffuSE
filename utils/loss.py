import torch
import torch.nn as nn
import numpy as np
from pesq import pesq
from pesq import PesqError
from utils import *
from joblib import Parallel, delayed


def mag_mse_loss(esti, label, frame_list):
    mask_for_loss = []
    utt_num = len(frame_list)
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], esti.size()[-1]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mag_mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
    loss = (((esti - label) * mag_mask_for_loss) ** 2).sum() / mag_mask_for_loss.sum()
    return loss


def mag_mae_loss(esti, label, frame_list):
    mask_for_loss = []
    utt_num = len(frame_list)
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], esti.size()[-1]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mag_mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
    loss = (torch.abs((esti - label) * mag_mask_for_loss)).sum() / mag_mask_for_loss.sum()
    return loss


def com_mse_loss(esti, label, frame_list):
    mask_for_loss = []
    utt_num = len(frame_list)
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], esti.size()[-1]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    loss = (((esti - label) * com_mask_for_loss) ** 2).sum() / com_mask_for_loss.sum()
    return loss

def com_mse_sigma_loss(esti, label, frame_list, mask):
    mask_for_loss = []
    utt_num = len(frame_list)
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], esti.size()[-1]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    loss = ((esti - label) * com_mask_for_loss / mask ** 2 * (esti - label) * com_mask_for_loss).sum() / com_mask_for_loss.sum()
    return loss


def com_mag_mse_loss(esti, label, frame_list):
    mask_for_loss = []
    utt_num = esti.size()[0]
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], esti.size()[-1]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    mag_esti, mag_label = torch.norm(esti, dim=1), torch.norm(label, dim=1)
    loss1 = (((esti - label) * com_mask_for_loss) ** 2).sum() / com_mask_for_loss.sum()
    loss2 = (((mag_esti - mag_label) * mask_for_loss) ** 2).sum() / mask_for_loss.sum()
    return 0.5 * (loss1 + loss2)


def pesq_loss(esti_list, label_list, frame_list, feat_type='sqrt'):
    with torch.no_grad():
        esti_mag, esti_phase = torch.norm(esti_list, dim=1), torch.atan2(esti_list[:, -1, :, :], esti_list[:, 0, :, :])
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
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            label_com = torch.stack((label_mag * torch.cos(label_phase), label_mag * torch.sin(label_phase)), dim=1)
        clean_utts, esti_utts = [], []
        utt_num = label_list.size()[0]
        for i in range(utt_num):
            tf_esti = esti_com[i, :, :, :].unsqueeze(dim=0).permute(0, 3, 2, 1).cpu()
            t_esti = torch.istft(tf_esti, n_fft=320, hop_length=160, win_length=320,
                                 window=torch.hann_window(320)).transpose(1, 0).squeeze(dim=-1).numpy()
            tf_label = label_com[i, :, :, :].unsqueeze(dim=0).permute(0, 3, 2, 1).cpu()
            t_label = torch.istft(tf_label, n_fft=320, hop_length=160, win_length=320,
                                  window=torch.hann_window(320)).transpose(1, 0).squeeze(dim=-1).numpy()
            t_len = (frame_list[i] - 1) * 160
            t_esti, t_label = t_esti[:t_len], t_label[:t_len]
            esti_utts.append(t_esti)
            clean_utts.append(t_label)

        # cv_pesq_score = Parallel(n_jobs=8)(delayed(eval_pesq)(id, esti_utts, clean_utts) for id in range(utt_num))
        cv_pesq_score = eval_pesq(esti_utts, clean_utts)
    return 4.50 - cv_pesq_score


def eval_pesq(esti_utts, clean_utts):
    pesq_score_list = []
    for clean_utt, esti_utt in zip(clean_utts, esti_utts):
        try:
            pesq_score = pesq(fs=16000, ref=clean_utt, deg=esti_utt,
                              mode='wb')  # https://github.com/samedii/python-pesq
            pesq_score_list.append(pesq_score)
        except PesqError as e:
            print(e)
    return np.mean(pesq_score_list)
