import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from ptflops import get_model_complexity_info

import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM, GRU
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.container import ModuleList
import copy


def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, bidirectional=True, dropout=0, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.gru = GRU(d_model, d_model * 2, 1, bidirectional=bidirectional)
        self.dropout = Dropout(dropout)
        if bidirectional:
            self.linear2 = Linear(d_model * 2 * 2, d_model)
        else:
            self.linear2 = Linear(d_model * 2, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src_norm = self.norm3(src)
        # src_norm = src
        src2 = self.self_attn(src_norm, src_norm, src_norm, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        self.gru.flatten_parameters()
        out, h_n = self.gru(src)
        del h_n
        src2 = self.linear2(self.dropout(self.activation(out)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class AIA_Transformer(nn.Module):
    """
    Adaptive time-frequency attention Transformer without interaction on maginitude path and complex path.
    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
    """

    def __init__(self, input_size, output_size, dropout=0, num_layers=1):
        super(AIA_Transformer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.k1 = Parameter(torch.ones(1))
        self.k2 = Parameter(torch.ones(1))

        self.input = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=1),
            nn.PReLU()
        )

        # dual-path RNN
        self.row_trans = nn.ModuleList([])
        self.col_trans = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_trans.append(
                TransformerEncoderLayer(d_model=input_size // 2, nhead=4, dropout=dropout, bidirectional=True))
            self.col_trans.append(
                TransformerEncoderLayer(d_model=input_size // 2, nhead=4, dropout=dropout, bidirectional=True))
            self.row_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(input_size // 2, output_size, 1)
                                    )

    def forward(self, input):
        #  input --- [b,  c,  num_frames, frame_size]  --- [b, c, dim2, dim1]
        b, c, dim2, dim1 = input.shape
        output_list = []
        output = self.input(input)
        for i in range(len(self.row_trans)):
            row_input = output.permute(3, 0, 2, 1).contiguous().view(dim1, b * dim2, -1)  # [F, B*T, c]
            row_output = self.row_trans[i](row_input)  # [F, B*T, c]
            row_output = row_output.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()  # [B, C, T, F]
            row_output = self.row_norm[i](row_output)  # [B, C, T, F]

            col_input = output.permute(2, 0, 3, 1).contiguous().view(dim2, b * dim1, -1)
            col_output = self.col_trans[i](col_input)
            col_output = col_output.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()
            col_output = self.col_norm[i](col_output)
            output = output + self.k1 * row_output + self.k2 * col_output
            output_i = self.output(output)
            output_list.append(output_i)
        del row_input, row_output, col_input, col_output

        return output_i, output_list


class AIA_Transformer_merge(nn.Module):
    """
    Adaptive time-frequency attention Transformer with interaction on two branch
    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
    """

    def __init__(self, input_size, output_size, dropout=0, num_layers=1):
        super(AIA_Transformer_merge, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.k1 = Parameter(torch.ones(1))
        self.k2 = Parameter(torch.ones(1))

        self.input = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=1),
            nn.PReLU()
        )

        self.row_trans = nn.ModuleList([])
        self.col_trans = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_trans.append(
                TransformerEncoderLayer(d_model=input_size // 2, nhead=4, dropout=dropout, bidirectional=True))
            self.col_trans.append(
                TransformerEncoderLayer(d_model=input_size // 2, nhead=4, dropout=dropout, bidirectional=True))
            self.row_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(input_size // 2, output_size, 1)
                                    )

    def forward(self, input1, input2):
        #  input --- [B,  C,  T, F]  --- [b, c, dim2, dim1]
        b, c, dim2, dim1 = input1.shape
        output_list_mag = []
        output_list_ri = []
        input_merge = torch.cat((input1, input2), dim=1)
        input_mag = self.input(input_merge)
        input_ri = self.input(input_merge)
        for i in range(len(self.row_trans)):
            if i >= 1:
                output_mag_i = output_list_mag[-1] + output_list_ri[-1]
            else:
                output_mag_i = input_mag
            AFA_input_mag = output_mag_i.permute(3, 0, 2, 1).contiguous().view(dim1, b * dim2, -1)  # [F, B*T, c]
            AFA_output_mag = self.row_trans[i](AFA_input_mag)  # [F, B*T, c]
            AFA_output_mag = AFA_output_mag.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()  # [B, C, T, F]
            AFA_output_mag = self.row_norm[i](AFA_output_mag)  # [B, C, T, F]

            ATA_input_mag = output_mag_i.permute(2, 0, 3, 1).contiguous().view(dim2, b * dim1, -1)  # [T, B*F, C]
            ATA_output_mag = self.col_trans[i](ATA_input_mag)  # [T, B*F, C]
            ATA_output_mag = ATA_output_mag.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()  # [B, C, T, F]
            ATA_output_mag = self.col_norm[i](ATA_output_mag)  # [B, C, T, F]
            output_mag_i = input_mag + self.k1 * AFA_output_mag + self.k2 * ATA_output_mag  # [B, C, T, F]
            output_mag_i = self.output(output_mag_i)
            output_list_mag.append(output_mag_i)

            if i >= 1:
                output_ri_i = output_list_ri[-1] + output_list_mag[-2]
            else:
                output_ri_i = input_ri
            # input_ri_mag = output_ri_i + output_list_mag[-1]
            AFA_input_ri = output_ri_i.permute(3, 0, 2, 1).contiguous().view(dim1, b * dim2, -1)  # [F, B*T, c]
            AFA_output_ri = self.row_trans[i](AFA_input_ri)  # [F, B*T, c]
            AFA_output_ri = AFA_output_ri.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()  # [B, C, T, F]
            AFA_output_ri = self.row_norm[i](AFA_output_ri)  # [B, C, T, F]

            ATA_input_ri = output_ri_i.permute(2, 0, 3, 1).contiguous().view(dim2, b * dim1, -1)  # [T, B*F, C]
            ATA_output_ri = self.col_trans[i](ATA_input_ri)  # [T, B*F, C]
            ATA_output_ri = ATA_output_ri.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()  # [B, C, T, F]
            ATA_output_ri = self.col_norm[i](ATA_output_ri)  # [B, C, T, F]
            output_ri_i = input_ri + self.k1 * AFA_output_ri + self.k2 * ATA_output_ri  # [B, C, T, F]
            output_ri_i = self.output(output_ri_i)
            output_list_ri.append(output_ri_i)

        del AFA_input_mag, AFA_output_mag, ATA_input_mag, ATA_output_mag, AFA_input_ri, AFA_output_ri, ATA_input_ri, ATA_output_ri
        # [b, c, dim2, dim1]

        return output_mag_i, output_list_mag, output_ri_i, output_list_ri


class AHAM(nn.Module):  # aham merge
    def __init__(self, input_channel=64, kernel_size=(1, 1), bias=True):
        super(AHAM, self).__init__()

        self.k3 = Parameter(torch.zeros(1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-2)
        self.conv1 = nn.Conv2d(input_channel, 1, kernel_size, (1, 1), bias=bias)

    def merge(self, x, y):
        batch, channel, height, width, blocks = x.size()
        input_x = x
        y = self.softmax(y)
        context = torch.matmul(input_x, y)
        context = context.view(batch, channel, height, width)

        return context

    def forward(self, input_list):  # X:BCTFG Y:B11G1
        batch, channel, frames, frequency = input_list[-1].size()
        x_list = []
        y_list = []
        for i in range(len(input_list)):
            input = self.avg_pool(input_list[i])
            y = self.conv1(input)
            x = input_list[i].unsqueeze(-1)
            y = y.unsqueeze(-2)
            x_list.append(x)
            y_list.append(y)

        x_merge = torch.cat((x_list[0], x_list[1], x_list[2], x_list[3]), dim=-1)
        y_merge = torch.cat((y_list[0], y_list[1], y_list[2], y_list[3]), dim=-2)

        y_softmax = self.softmax(y_merge)

        aham = torch.matmul(x_merge, y_softmax)
        aham = aham.view(batch, channel, frames, frequency)
        aham_output = input_list[-1] + aham
        return aham_output


class AHAM_ori(nn.Module):  # aham merge, share the weights on 1D CNN get better performance and lower parameter
    def __init__(self, input_channel=64, kernel_size=(1, 1), bias=True, act=nn.ReLU(True)):
        super(AHAM_ori, self).__init__()

        self.k3 = Parameter(torch.zeros(1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-2)
        self.conv1 = nn.Conv2d(input_channel, 1, kernel_size, (1, 1), bias=bias)

    def merge(self, x, y):
        batch, channel, height, width, blocks = x.size()
        input_x = x  # B*C*T*F*G
        y = self.softmax(y)
        context = torch.matmul(input_x, y)
        context = context.view(batch, channel, height, width)  # B*C*T*F

        return context

    def forward(self, input_list):  # X:BCTFG Y:B11G1
        batch, channel, frames, frequency = input_list[-1].size()
        x_list = []
        y_list = []
        for i in range(len(input_list)):
            input = self.avg_pool(input_list[i])
            y = self.conv1(input)
            x = input_list[i].unsqueeze(-1)
            y = y.unsqueeze(-2)
            x_list.append(x)
            y_list.append(y)

        x_merge = torch.cat((x_list[0], x_list[1], x_list[2], x_list[3]), dim=-1)
        y_merge = torch.cat((y_list[0], y_list[1], y_list[2], y_list[3]), dim=-2)

        y_softmax = self.softmax(y_merge)
        aham = torch.matmul(x_merge, y_softmax)
        aham = aham.view(batch, channel, frames, frequency)
        aham_output = input_list[-1] + aham
        # print(str(aham_output.shape))
        return aham_output


class dual_aia_complex_trans(nn.Module):
    def __init__(self):
        super(dual_aia_complex_trans, self).__init__()
        self.en_ri = dense_encoder()
        self.en_mag = dense_encoder_mag()
        self.dual_trans = AIA_Transformer(64, 64, num_layers=4)
        self.aham = AHAM(input_channel=64)
        self.dual_trans_mag = AIA_Transformer(64, 64, num_layers=4)
        self.aham_mag = AHAM(input_channel=64)

        self.de1 = dense_decoder()
        self.de2 = dense_decoder()
        self.de_mag_mask = dense_decoder_masking()

    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape
        x_mag_ori = torch.norm(x, dim=1)
        x_mag = x_mag_ori.unsqueeze(dim=1)
        x_ri = self.en_ri(x)  # BCTF
        x_last, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF
        x_mag_en = self.en_mag(x_mag)
        x_last_mag, x_outputlist_mag = self.dual_trans_mag(x_mag_en)  # BCTF, #BCTFG
        x_mag_en = self.aham_mag(x_outputlist_mag)  # BCTF
        x_mag_mask = self.de_mag_mask(x_mag_en)
        x_mag = x_mag_mask * x_mag
        x_mag = x_mag.squeeze(dim=1)
        x_real = self.de1(x_ri)
        x_imag = self.de2(x_ri)
        x_real = x_real.squeeze(dim=1)
        x_imag = x_imag.squeeze(dim=1)
        x_com = torch.stack((x_real, x_imag), dim=1)
        pre_mag, pre_phase = torch.norm(x_com, dim=1), torch.atan2(x_com[:, -1, :, :], x_com[:, 0, :, :])
        x_mag_out = (x_mag + pre_mag) / 2
        x_r_out, x_i_out = x_mag_out * torch.cos(pre_phase), x_mag_out * torch.sin(pre_phase)
        x_com_out = torch.stack((x_r_out, x_i_out), dim=1)

        return x_com_out


class dual_aia_trans_merge_crm(nn.Module):
    def __init__(self):
        super(dual_aia_trans_merge_crm, self).__init__()
        self.en_ri = dense_encoder()
        self.en_mag = dense_encoder_mag()
        self.aia_trans_merge = AIA_Transformer_merge(128, 64, num_layers=4)
        self.aham = AHAM_ori(input_channel=64)
        self.aham_mag = AHAM_ori(input_channel=64)

        self.de1 = dense_decoder()
        self.de2 = dense_decoder()
        self.de_mag_mask = dense_decoder_masking()

    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape
        x_r_input, x_i_input = x[:, 0, :, :], x[:, 1, :, :]
        x_mag_ori, x_phase_ori = torch.norm(x, dim=1), torch.atan2(x[:, -1, :, :], x[:, 0, :, :])
        x_mag = x_mag_ori.unsqueeze(dim=1)
        # ri/mag components enconde+ aia_transformer_merge
        x_ri = self.en_ri(x)  # BCTF
        x_mag_en = self.en_mag(x_mag)
        x_last_mag, x_outputlist_mag, x_last_ri, x_outputlist_ri = self.aia_trans_merge(x_mag_en, x_ri)  # BCTF, #BCTFG

        x_ri = self.aham(x_outputlist_ri)  # BCT
        x_mag_en = self.aham_mag(x_outputlist_mag)  # BCTF
        x_mag_mask = self.de_mag_mask(x_mag_en)
        x_mag_mask = x_mag_mask.squeeze(dim=1)

        # real and imag decode
        x_real = self.de1(x_ri)
        x_imag = self.de2(x_ri)
        x_real = x_real.squeeze(dim=1)
        x_imag = x_imag.squeeze(dim=1)
        # magnitude and ri components interaction

        x_mag_out = x_mag_mask * x_mag_ori
        x_r_out, x_i_out = (x_mag_out * torch.cos(x_phase_ori) + x_real), (x_mag_out * torch.sin(x_phase_ori) + x_imag)

        x_com_out = torch.stack((x_r_out, x_i_out), dim=1)

        return x_com_out


class aia_complex_trans_mag(nn.Module):
    def __init__(self):
        super(aia_complex_trans_mag, self).__init__()
        self.en_mag = dense_encoder_mag()

        self.dual_trans_mag = AIA_Transformer(64, 64, num_layers=4)
        self.aham_mag = AHAM(input_channel=64)

        self.de_mag_mask = dense_decoder_masking()

    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape
        x_r_input, x_i_input = x[:, 0, :, :], x[:, 1, :, :]
        x_mag_ori, x_phase_ori = torch.norm(x, dim=1), torch.atan2(x[:, -1, :, :], x[:, 0, :, :])
        x_mag = x_mag_ori.unsqueeze(dim=1)

        # magnitude enconde+ aia_transformer +  masking module
        x_mag_en = self.en_mag(x_mag)
        x_last_mag, x_outputlist_mag = self.dual_trans_mag(x_mag_en)  # BCTF, #BCTFG
        x_mag_en = self.aham_mag(x_outputlist_mag)  # BCTF
        x_mag_mask = self.de_mag_mask(x_mag_en)
        x_mag_mask = x_mag_mask.squeeze(dim=1)

        # real and imag decode
        # magnitude and ri components interaction

        x_mag_out = x_mag_mask * x_mag_ori
        x_r_out, x_i_out = (x_mag_out * torch.cos(x_phase_ori)), (x_mag_out * torch.sin(x_phase_ori))

        x_com_out = torch.stack((x_r_out, x_i_out), dim=1)

        return x_com_out


class aia_complex_trans_ri(nn.Module):
    def __init__(self):
        super(aia_complex_trans_ri, self).__init__()
        self.en_ri = dense_encoder()

        self.dual_trans = AIA_Transformer(64, 64, num_layers=4)
        self.aham = AHAM(input_channel=64)

        self.de1 = dense_decoder()
        self.de2 = dense_decoder()

    def forward(self, x):
        batch_size, _, seq_len, _ = x.shape
        x_r_input, x_i_input = x[:, 0, :, :], x[:, 1, :, :]
        x_mag_ori, x_phase_ori = torch.norm(x, dim=1), torch.atan2(x[:, -1, :, :], x[:, 0, :, :])
        x_mag = x_mag_ori.unsqueeze(dim=1)
        # ri components enconde+ aia_transformer
        x_ri = self.en_ri(x)  # BCTF
        x_last, x_outputlist = self.dual_trans(x_ri)  # BCTF, #BCTFG
        x_ri = self.aham(x_outputlist)  # BCTF

        # real and imag decode
        x_real = self.de1(x_ri)
        x_imag = self.de2(x_ri)
        x_real = x_real.squeeze(dim=1)
        x_imag = x_imag.squeeze(dim=1)
        x_com = torch.stack((x_real, x_imag), dim=1)

        return x_com


class dense_encoder(nn.Module):
    def __init__(self, width=64):
        super(dense_encoder, self).__init__()
        self.in_channels = 2
        self.out_channels = 1
        self.width = width
        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width,
                                  kernel_size=(1, 1))  # [b, 64, nframes, 512]
        self.inp_norm = nn.LayerNorm(161)
        self.inp_prelu = nn.PReLU(self.width)
        self.enc_dense1 = DenseBlock(161, 4, self.width)  # [b, 64, nframes, 512]
        self.enc_conv1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3),
                                   stride=(1, 2))  # [b, 64, nframes, 256]
        self.enc_norm1 = nn.LayerNorm(80)
        self.enc_prelu1 = nn.PReLU(self.width)

    def forward(self, x):
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))  # [b, 64, T, F]
        out = self.enc_dense1(out)  # [b, 64, T, F]
        x = self.enc_prelu1(self.enc_norm1(self.enc_conv1(out)))  # [b, 64, T, F]
        return x


class dense_encoder_mag(nn.Module):
    def __init__(self, width=64):
        super(dense_encoder_mag, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.width = width
        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width,
                                  kernel_size=(1, 1))  # [b, 64, nframes, 512]
        self.inp_norm = nn.LayerNorm(161)
        self.inp_prelu = nn.PReLU(self.width)
        self.enc_dense1 = DenseBlock(161, 4, self.width)  # [b, 64, nframes, 512]
        self.enc_conv1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3),
                                   stride=(1, 2))  # [b, 64, nframes, 256]
        self.enc_norm1 = nn.LayerNorm(80)
        self.enc_prelu1 = nn.PReLU(self.width)

    def forward(self, x):
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))  # [b, 64, T, F]
        out = self.enc_dense1(out)  # [b, 64, T, F]
        x = self.enc_prelu1(self.enc_norm1(self.enc_conv1(out)))  # [b, 64, T, F]
        return x


class dense_decoder(nn.Module):
    def __init__(self, width=64):
        super(dense_decoder, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.)
        self.width = width
        self.dec_dense1 = DenseBlock(80, 4, self.width)
        self.dec_conv1 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm1 = nn.LayerNorm(161)
        self.dec_prelu1 = nn.PReLU(self.width)

        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channels, kernel_size=(1, 1))

    def forward(self, x):
        out = self.dec_dense1(x)
        out = self.dec_prelu1(self.dec_norm1(self.pad1(self.dec_conv1(self.pad(out)))))

        out = self.out_conv(out)
        out.squeeze(dim=1)
        return out


class dense_decoder_masking(nn.Module):
    def __init__(self, width=64):
        super(dense_decoder_masking, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.)
        self.width = width
        self.dec_dense1 = DenseBlock(80, 4, self.width)
        self.dec_conv1 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm1 = nn.LayerNorm(161)
        self.dec_prelu1 = nn.PReLU(self.width)
        self.mask1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1)),
            nn.Sigmoid()
        )
        self.mask2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1)),
            nn.Tanh()
        )
        self.maskconv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
        # self.maskrelu = nn.ReLU(inplace=True)
        self.maskrelu = nn.Sigmoid()
        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channels, kernel_size=(1, 1))

    def forward(self, x):
        out = self.dec_dense1(x)
        out = self.dec_prelu1(self.dec_norm1(self.pad1(self.dec_conv1(self.pad(out)))))

        out = self.out_conv(out)
        out.squeeze(dim=1)
        out = self.mask1(out) * self.mask2(out)
        out = self.maskrelu(self.maskconv(out))  # mask
        return out


class SPConvTranspose2d(nn.Module):  # sub-pixel convolution
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class DenseBlock(nn.Module):  # dilated dense block
    def __init__(self, input_size, depth=5, in_channels=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.LayerNorm(input_size))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


if __name__ == '__main__':
    model = dual_aia_trans_merge_crm()
    model.eval()
    x = torch.FloatTensor(4, 2, 10, 161)
    # output = model(x)
    x = model(x)
    print(str(x.shape))

    print('The number of parameters of the model is:%.5d' % numParams(model))
    macs, params = get_model_complexity_info(model, (2, 100, 161), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
