import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from thop import profile
import time
import numpy as np
from models.DCN_layer import DCN_layer
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

class CDUnet(nn.Module):
    def __init__(self):
        super(CDUnet, self).__init__()

        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, 3)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)


        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
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
            self.up = CARAFE(in_channels, in_channels // 2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class CARAFE(nn.Module):
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
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = nn.Conv2d(c, c_mid,kernel_size=3,padding=1)
        self.enc = nn.Conv2d(c_mid, (scale * k_up) ** 2, kernel_size=3,padding=1)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='bilinear')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)
        self.cout = nn.Conv2d(c, int(c/2), kernel_size=3, padding=1)
    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * m * h * w
        W = self.enc(W)  # b * 100 * h * w
        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = torch.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * 25c * h_ * w_
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
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
