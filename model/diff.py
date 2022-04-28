import torch
import torch.nn as nn
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''
A UNet-typed model for Diffusion-SE.
'''


class DiffUNet(nn.Module):
    def __init__(self):
        super(DiffUNet, self).__init__()
        self.en = Encoder()
        self.de_real = Decoder()
        self.de_imag = Decoder()
        self.TCMs = nn.Sequential(TCM(),
                                  TCM(),
                                  TCM())

    def forward(self, x):
        x, en_list = self.en(x)  # [b,c,t,f]
        x = x.permute(0, 2, 1, 3)  # [b,t,c,f]
        x = x.reshape(x.size()[0], x.size()[1], -1).permute(0, 2, 1)  # [b, cf, t]
        x = self.TCMs(x).permute(0, 2, 1)
        x = x.reshape(x.size()[0], x.size()[1], 64, 4)  # [b,t,c,f]
        x = x.permute(0, 2, 1, 3)
        x_real = self.de_real(x, en_list)
        x_imag = self.de_imag(x, en_list)
        out = torch.cat((x_real, x_imag), dim=1)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pad1 = nn.ConstantPad2d((0, 0, 1, 0), value=0.)  # left right up down

        # convGLU
        self.conv1 = BiConvGLU(in_channels=2, out_channels=64, kernel_size=(2, 5), stride=(1, 2))
        self.conv2 = BiConvGLU(in_channels=64, out_channels=64, kernel_size=(2, 3), stride=(1, 2))
        self.conv3 = BiConvGLU(in_channels=64, out_channels=64, kernel_size=(2, 3), stride=(1, 2))
        self.conv4 = BiConvGLU(in_channels=64, out_channels=64, kernel_size=(2, 3), stride=(1, 2))
        self.conv5 = BiConvGLU(in_channels=64, out_channels=64, kernel_size=(2, 3), stride=(1, 2))
        self.en1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.en2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.en3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.en4 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.en5 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

    def forward(self, x):
        en_list = []
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.en1(x)
        en_list.append(x)
        x = self.pad1(x)
        x = self.conv2(x)
        x = self.en2(x)
        en_list.append(x)
        x = self.pad1(x)
        x = self.conv3(x)
        x = self.en3(x)
        en_list.append(x)
        x = self.pad1(x)
        x = self.conv4(x)
        x = self.en4(x)
        en_list.append(x)
        x = self.pad1(x)
        x = self.conv5(x)
        x = self.en5(x)
        en_list.append(x)
        return x, en_list


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up_f = up_Chomp_F(1)
        self.down_f = down_Chomp_F(1)
        self.chomp_t = Chomp_T(1)
        self.de5 = nn.Sequential(
            BiConvTransGLU(in_channels=128, out_channels=64, kernel_size=(2, 3), stride=(1, 2)),
            self.chomp_t,
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.de4 = nn.Sequential(
            BiConvTransGLU(in_channels=128, out_channels=64, kernel_size=(2, 3), stride=(1, 2)),
            self.chomp_t,
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.de3 = nn.Sequential(
            BiConvTransGLU(in_channels=128, out_channels=64, kernel_size=(2, 3), stride=(1, 2)),
            self.chomp_t,
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.de2 = nn.Sequential(
            BiConvTransGLU(in_channels=128, out_channels=64, kernel_size=(2, 3), stride=(1, 2)),
            self.chomp_t,
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.de1 = nn.Sequential(
            BiConvTransGLU(in_channels=128, out_channels=1, kernel_size=(2, 5), stride=(1, 2)),
            self.chomp_t,
            # nn.BatchNorm2d(1),
            # nn.PReLU()
        )

    def forward(self, x, x_list):
        x = self.de5(torch.cat((x, x_list[-1]), dim=1))
        x = self.de4(torch.cat((x, x_list[-2]), dim=1))
        x = self.de3(torch.cat((x, x_list[-3]), dim=1))
        x = self.de2(torch.cat((x, x_list[-4]), dim=1))
        x = self.de1(torch.cat((x, x_list[-5]), dim=1))
        return x


class Residual(nn.Module):
    def __init__(self, dilation):
        super(Residual, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1, stride=1)

        self.mainbranch = nn.Sequential(
            nn.PReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2 * dilation,
                dilation=dilation)
        )
        self.maskbranch = nn.Sequential(
            nn.PReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2 * dilation,
                dilation=dilation),
            nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.PReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=256, stride=1, kernel_size=1)
        )

    def forward(self, x):
        t = x
        x = self.conv1(x)
        x = self.mainbranch(x) * self.maskbranch(x)
        x = self.conv2(x)
        out = x + t
        return out


class TCM(nn.Module):
    def __init__(self):
        super(TCM, self).__init__()
        self.residual1 = Residual(dilation=1)
        self.residual2 = Residual(dilation=2)
        self.residual3 = Residual(dilation=4)
        self.residual4 = Residual(dilation=8)
        self.residual5 = Residual(dilation=16)
        self.residual6 = Residual(dilation=32)

    def forward(self, x):
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.residual4(x)
        x = self.residual5(x)
        x = self.residual6(x)
        return x


class up_Chomp_F(nn.Module):
    def __init__(self, chomp_f):
        super(up_Chomp_F, self).__init__()
        self.chomp_f = chomp_f

    def forward(self, x):
        return x[:, :, :, self.chomp_f:]


class down_Chomp_F(nn.Module):
    def __init__(self, chomp_f):
        super(down_Chomp_F, self).__init__()
        self.chomp_f = chomp_f

    def forward(self, x):
        return x[:, :, :, :-self.chomp_f]


class Chomp_T(nn.Module):
    def __init__(self, chomp_t):
        super(Chomp_T, self).__init__()
        self.chomp_t = chomp_t

    def forward(self, x):
        return x[:, :, :-self.chomp_t, :]


class BiConvGLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(BiConvGLU, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(1, 1), stride=(1, 1))
        self.l = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=stride)
        self.l_conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=(1, 1))
        self.r = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=stride)
        self.r_conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=(1, 1))
        self.Sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.conv1(x)
        left = self.l(x)
        right = self.r(x)
        left_mask = self.Sigmoid(self.l_conv(left))
        right_mask = self.Sigmoid(self.r_conv(right))
        left = left * right_mask
        right = right * left_mask
        return self.conv2(left + right)


class BiConvTransGLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(BiConvTransGLU, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=32, kernel_size=(1, 1), stride=(1, 1))
        self.l = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=stride)
        self.l_conv = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=(1, 1))
        self.r_conv = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=(1, 1))
        self.r = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=stride)
        self.Sigmoid = nn.Sigmoid()
        self.conv2 = nn.ConvTranspose2d(in_channels=32, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.conv1(x)
        left = self.l(x)
        right = self.r(x)
        left_mask = self.Sigmoid(self.l_conv(left))
        right_mask = self.Sigmoid(self.r_conv(right))
        left = left * right_mask
        right = right * left_mask
        return self.conv2(left + right)
