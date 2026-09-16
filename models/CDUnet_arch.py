import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from thop import profile
import time
import numpy as np
import torch.nn.functional as F
from pytorch_wavelets import DWTForward


class CDUnetv(nn.Module):
    def __init__(self):
        super(CDUnetv, self).__init__()

        self.inc_r = OutConv(3, 64)
        self.down1_r = Down(64, 128)
        self.down2_r = Down(128, 256)
        self.down3_r = Down(256, 512)

        self.inc_l = OutConv(1, 64)
        self.down1_l = Down(64, 128)
        self.down2_l = Down(128, 256)
        self.down3_l = Down(256, 512)

        self.up1_r = Up(512, 256)
        self.up2_r = Up(256, 128)
        self.up3_r = Up(128, 64)
        self.outc_r = OutConv(64, 3)

        self.up1_l = Up(512, 256)
        self.up2_l = Up(256, 128)
        self.up3_l = Up(128, 64)
        self.outc_l = OutConv(64, 3)

    def forward(self, x_r ,x_l):
    # encoder

        x1_r = self.inc_r(x_r)
        x2_r = self.down1_r(x1_r)
        x3_r = self.down2_r(x2_r)
        x4_r = self.down3_r(x3_r)

        x1_l = self.inc_l(x_l)
        x2_l = self.down1_l(x1_l)
        x3_l = self.down2_l(x2_l)
        x4_l = self.down3_l(x3_l)


    # mix





    # decoder
        x_r = self.up1_r(x4_r, x3_r)
        x_r = self.up2_r(x_r, x2_r)
        x_r = self.up3_r(x_r, x1_r)
        our_r = self.outc_r(x_r)

        x_l = self.up1_l(x4_l, x3_l)
        x_l = self.up2_l(x_l, x2_l)
        x_l = self.up3_l(x_l, x1_l)
        our_l = self.outc_l(x_l)

        out = torch.mul(our_r,our_l)

        return out





class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        self.wtdown = nn.Sequential(WaveLetDown(in_channels),
                                    DoubleConv(out_channels, out_channels)
                                    )
    def forward(self, x):
        return self.wtdown(x)




class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = UPDWCA(in_channels, in_channels // 2)
            self.resconv = DoubleConv(out_channels, out_channels)
        self.ca = CALayer(in_channels)
        self.act = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3 // 2, groups=out_channels)
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1,x2)
        # [N, C, H, W]
        x = torch.cat([x2, x1], dim=1)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.ca(x)
        x = self.conv1(x)
        x = self.resconv(x)
        return x


class UPDWCA(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(UPDWCA, self).__init__()
        self.scale = scale

        self.pacomp = nn.Conv2d(c, 4 , kernel_size=3,padding=1)
        self.pix_shf = nn.PixelShuffle(scale)
        self.pacomp2 = nn.Conv2d(c//2, 1, kernel_size =3, padding=1)
        self.upsmp = nn.Upsample(scale_factor=scale, mode='bilinear')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)
        self.cout = nn.Conv2d(c, int(c/2), kernel_size=3, padding=1)
        self.ca =  nn.Sequential(
                nn.Conv2d(c, c // 8, 1, padding=0, bias=True),
                nn.GELU(),
                nn.Conv2d(c // 8, c, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, X, Y):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W1 = self.pacomp(X)
        W1 = self.pix_shf(W1)
        W2 = self.pacomp2(Y)
        W = W1 + W2
        W = torch.softmax(W, dim=1)

        X = self.upsmp(X)  # b * c * h_ * w_
        #X = self.unfold(X)  # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        X = self.cout(X)
        return X


class WaveLetDown(nn.Module):
    def __init__(self, inchanel):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()

        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        self.wtconv = nn.Conv2d(int(inchanel*2), inchanel, kernel_size=1, padding=0, )
        self.conv = nn.Conv2d( int(inchanel/2), inchanel, kernel_size=3, padding=1, )
        self.conv2 = nn.Conv2d( int(inchanel/2), inchanel, kernel_size=3, padding=1, )

    def forward(self, x):
        x1, x2 = x.chunk(2, 1)

        yL, yH = self.wt(x1)
        HL = yH[0][:, :, 0, ::]
        LH = yH[0][:, :, 1, ::]
        HH = yH[0][:, :, 2, ::]
        wtx = torch.cat([yL, HL, LH, HH], dim=1)
        x1 = self.wtconv(wtx)

        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.conv2(x2)
        return torch.cat((x1, x2), 1)


class HWD(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p):
        super(HWD, self).__init__()

        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv = Conv(in_ch * 4, out_ch, k, s, p)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)




class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU()
        )
        self.frdb = FRDB(in_channels)

    def forward(self, x):
        return self.frdb(x)

class ChannelAggregationFFN(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size=3, act_fuc=nn.GELU):
        super().__init__()

        self.conv1 = nn.Conv2d(inchannel, outchannel, 1)
        self.dwconv = nn.Conv2d(outchannel, outchannel, kernel_size, padding=kernel_size // 2, groups=outchannel)
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size,padding=1)
        self.convout = nn.Conv2d(inchannel, outchannel, 1)
        self.act = act_fuc()
        self.decompose = nn.Conv2d(
            outchannel,  # C -> 1
            1, kernel_size=1,
        )
        self.sigma = ElementScale(
            outchannel, init_value=1e-5, requires_grad=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.dwconv(x1)
        x1 = self.act(x1)
        decompose = self.act(self.decompose(x1))
        x1 = x1 + self.sigma(x1 - decompose)
        x1 = self.conv2(x1)
        x = self.convout(x)
        x = x + x1
        return x

class ElementScale(nn.Module):
    #A learnable element-wise scaler.

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class make_fdense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=1):
        super(make_fdense, self).__init__()
        # self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
        # bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                      bias=False), nn.BatchNorm2d(growthRate)
        )
        self.bat = nn.BatchNorm2d(growthRate),
        self.leaky = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.leaky(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class FRDB(nn.Module):
    def __init__(self, nChannels, nDenselayer=1, growthRate=32):
        super(FRDB, self).__init__()
        nChannels_1 = nChannels
        nChannels_2 = nChannels
        modules1 = []
        for i in range(nDenselayer):
            modules1.append(make_fdense(nChannels_1, growthRate))
            nChannels_1 += growthRate
        self.dense_layers1 = nn.Sequential(*modules1)
        modules2 = []
        for i in range(nDenselayer):
            modules2.append(make_fdense(nChannels_2, growthRate))
            nChannels_2 += growthRate
        self.dense_layers2 = nn.Sequential(*modules2)
        self.conv_1 = nn.Conv2d(nChannels_1, nChannels, kernel_size=1, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(nChannels_2, nChannels, kernel_size=1, padding=0, bias=False)
        self.SRDB = SRDB(nChannels)
        # self.patch_embed = PatchEmbed(img_size=224, patch_size=7, stride=4, in_chans=nChannels,
        # embed_dim=embed_dims[0])

    def forward(self, x):
        x = self.SRDB(x)
        _, _, H, W = x.shape
        # print(x.shape)
        x_freq = torch.fft.rfft2(x, norm='backward')
        # print(x_freq.shape)
        mag = torch.abs(x_freq)
        # print(mag.shape)
        pha = torch.angle(x_freq)
        mag = self.dense_layers1(mag)
        # print(mag.shape)
        mag = self.conv_1(mag)
        # print(mag.shape)
        pha = self.dense_layers2(pha)
        pha = self.conv_2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        out = out + x
        return out


class SRDB(nn.Module):
    def __init__(self, nChannels, growthRate=64):
        super(SRDB, self).__init__()
        nChannels_ = nChannels
        modules1 = []
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=1, padding=(1 - 1) // 2,
                               bias=False)
        self.conv2 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=(3 - 1) // 2,
                               bias=False)
        self.conv3 = nn.Conv2d(nChannels, growthRate, kernel_size=5, padding=(5 - 1) // 2,
                               bias=False)

        self.conv6 = nn.Conv2d(growthRate * 3, nChannels, kernel_size=1, padding=(1 - 1) // 2,
                               bias=False)
        self.leaky1 = nn.LeakyReLU(0.1, inplace=True)
        self.leaky2 = nn.LeakyReLU(0.1, inplace=True)
        self.leaky3 = nn.LeakyReLU(0.1, inplace=True)


    def forward(self, x):
        # x_1=self.bat1(self.conv1(x))
        x_1 = self.leaky1(self.conv1(x))
        x_2 = self.leaky2(self.conv2(x))
        x_3 = self.leaky3(self.conv3(x))
        x_0 = torch.cat((x_1, x_2, x_3), dim=1)

        x_0 = self.conv6(x_0)

        out = x_0 + x
        return out