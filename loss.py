import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
from functools import partial
import numpy as np
from numpy import *
import random
# from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY

_reduction_modes = ['none', 'mean', 'sum']


def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='mean')


def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='mean')


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target)


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target)


class AFFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(AFFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        # self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        pred_fft = torch.abs(pred_fft)
        target_fft = torch.abs(target_fft)
        # return self.loss_weight * self.criterion(pred_fft, target_fft)
        return self.loss_weight * l1_loss(pred_fft, target_fft)

class SRN_loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(SRN_loss, self).__init__()
        # self.eps = e
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.first = True
        self.resize = partial(F.interpolate, mode='area', recompute_scale_factor=True)
        # self.scale = 1e2 / np.log(1e2)

    def forward(self, batch_p, batch_l):
        assert batch_p[0].shape[0] == batch_l.shape[0]
        device = batch_p[0].device
        b, c, h, w = batch_p[0].shape
        # self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

        loss = self.loss_weight * self.scale * torch.log(
            ((batch_p[0] - batch_l) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

        loss += 0.5 * self.loss_weight * self.scale * torch.log(
            ((batch_p[1] - self.resize(input=batch_l, scale_factor=0.5)) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        loss += 0.25 * self.loss_weight * self.scale * torch.log(
            ((batch_p[2] - self.resize(input=batch_l, scale_factor=0.25)) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        loss += 0.125 * self.loss_weight * self.scale * torch.log(
            ((batch_p[3] - self.resize(input=batch_l, scale_factor=0.125)) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        """
        loss = self.loss_weight * mse_loss(batch_p[0] , batch_l)

        loss += 0.5 * self.loss_weight  * mse_loss(batch_p[1] ,self.resize(input=batch_l, scale_factor=0.5))
        loss += 0.25 * self.loss_weight  * mse_loss(batch_p[2] , self.resize(input=batch_l, scale_factor=0.25))
        loss += 0.125 * self.loss_weight * mse_loss(batch_p[3] , self.resize(input=batch_l, scale_factor=0.125))
        """
        return loss





class FCLoss(nn.Module):
    def __init__(self, ablation=False):

        super(FCLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.multi_n_num = 2

    def forward(self, a, p, n):
        a_fft = torch.fft.fft2(a)
        p_fft = torch.fft.fft2(p)
        n_fft = torch.fft.fft2(n)

        contrastive = 0
        d_ap = self.l1(a_fft, p_fft)
        d_an = self.l1(a_fft, n_fft)
        contrastive += (d_ap / (d_an + 1e-7))

        return contrastive

class MFCloss(nn.Module):
    def __init__(self, loss_weight=0.5, reduction='mean'):
        super(MFCloss, self).__init__()
        # self.eps = e
        assert reduction == 'mean'
        self.loss_weight = loss_weight

        self.resize = partial(F.interpolate, mode='area', recompute_scale_factor=True)
        # self.scale = 1e2 / np.log(1e2)
        self.fcloss = FCLoss()
    def forward(self, batch_a, batch_p, batch_n):

        loss = self.loss_weight * self.fcloss(batch_a[0], batch_p, batch_n)

        loss += 0.5 * self.loss_weight * self.fcloss(batch_a[1], self.resize(input=batch_p, scale_factor=0.5), self.resize(input=batch_n, scale_factor=0.5))
        loss += 0.25 * self.loss_weight * self.fcloss(batch_a[2], self.resize(input=batch_p, scale_factor=0.25), self.resize(input=batch_n, scale_factor=0.25))
        loss += 0.125 * self.loss_weight * self.fcloss(batch_a[3], self.resize(input=batch_p, scale_factor=0.125), self.resize(input=batch_n, scale_factor=0.125))
        return loss