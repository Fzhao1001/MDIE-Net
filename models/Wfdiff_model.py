import math
import torch
from torch import nn
from basicsr.utils.registry import ARCH_REGISTRY
from models.Padiff_arch.wavelet import DWT, IWT
from models.Padiff_arch.diffx2_arch import GaussianDiffusionx2
from models import CDUnetV2
from models.Padiff_arch.cfc_arch import CFC



class WfDiffx2(nn.Module):
    def __init__(
            self,
            in_channel=6,
            out_channel=3,
            inner_channel=32,
            norm_groups=32,
            with_time_emb=True,
            schedule_opt=None,
            sample_proc='ddim',
            local_ensemble=True,
            feat_unfold=True,
            cell_decode=True,
            ppg_input_channels=3,
    ):
        super().__init__()
        self.denoiser1 = GaussianDiffusionx2(
            in_channel=in_channel,
            out_channel=out_channel,
            inner_channel=inner_channel,
            norm_groups=norm_groups,
            with_time_emb=with_time_emb,
            schedule_opt=schedule_opt,
            sample_proc='sample_proc'
        )
        self.denoiser2 = GaussianDiffusionx2(
            in_channel=in_channel,
            out_channel=out_channel,
            inner_channel=inner_channel,
            norm_groups=norm_groups,
            with_time_emb=with_time_emb,
            schedule_opt=schedule_opt,
            sample_proc='sample_proc'
        )
        self.init_predictor = CDUnetV2()

    def forward(self, condition, gt=None):


        x_ = self.init_predictor(condition)

        input_img = x_[:, :3, :, :]
        n, c, h, w = input_img.shape


        if gt != None:
            residual = gt - input_img
        else:
            residual = None


        x_L = self.denoiser1(input_img, residual,input_img)

        return x_, x_L





