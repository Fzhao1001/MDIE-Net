import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from thop import profile
import time
import numpy as np
import torch.nn.functional as F
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

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



class Block(nn.Module):  # origin
    def __init__(self, conv, dim, kernel_size, ):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, 3, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, 3, bias=True)
        self.conv3 = conv(dim, dim, 5, bias=True)
        self.ca = CALayer(dim)
        self.pa = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.ca(res)
        res = self.pa(res)

        res += x
        return res



class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        self.conv= conv(dim, dim, kernel_size)
        self.gp = nn.Sequential(*modules)

        self.gdfusion = IGM(dim, dim, kernel_size)


    def forward(self, x, xgd):
        res = self.gp(x)
        res1 = self.gdfusion(res,xgd)
        return res1+x


class GetgradientBlock(nn.Module):
    def __init__(self,dim):
        super(GetgradientBlock, self).__init__()
        self.gradientconv = nn.Sequential(*[
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1)])
        self.sft = SFT_layer(dim, dim)
    def forward(self,x,y):
        out = self.sft(x,y)
        out = self.gradientconv(out)
        return out

class LGNet(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(LGNet, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)

        self.gdfusion = IGM(self.dim,self.dim,kernel_size)
        self.gdb = GetgradientBlock(self.dim)
        self.gradientconv = nn.Sequential(*[
            nn.Conv2d(3, self.dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim, self.dim, 3, padding=1)])
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])

        self.gd = GetGradientNopadding()
        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        gd = self.gd(x1)
        x = self.pre(x1)
        gd1 = self.gradientconv(gd)
        res1 = self.g1(x,gd1)
        gd2 = self.gdb(gd1,res1)
        res2 = self.g2(res1,gd2)
        gd3 = self.gdb(gd2, res2)
        res3 = self.g3(res2,gd3)
        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2 + w[:, 2, ::] * res3
        x = self.post(out)

        return x + x1






class SFT_layer(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(SFT_layer, self).__init__()
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )

    def forward(self, x, inter):
        '''
        :param x: degradation representation: B * C
        :param inter: degradation intermediate representation map: B * C * H * W
        '''
        gamma = self.conv_gamma(inter)
        beta = self.conv_beta(inter)

        return x * gamma + beta


class IGM(nn.Module):
    def  __init__(self, channels_in, channels_out, kernel_size):
        super(IGM, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.atf = AFF(self.channels_in)



    def forward(self, x, inter):

        out = self.atf(x,inter)

        return out


class GetGradientNopadding(nn.Module):
    def __init__(self):
        super(GetGradientNopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)

        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, inp_feat):
        x_list = []
        for i in range(inp_feat.shape[1]):
            x_i = inp_feat[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        res = torch.cat(x_list, dim=1)

        return res

# Attention Feature Fusion (AFF)
class AFF(nn.Module):
    def __init__(self, channels, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = torch.sigmoid(xlg)

        xo = x * wei + residual * (1 - wei)
        return xo




