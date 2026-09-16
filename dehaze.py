import argparse
import os

import numpy as np
import torch
import torchvision.transforms as tfs
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

from metrics import psnr, ssim
from models.Padiff_arch.MDIE_arch import MLWNet

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_name', help='name of dataset', choices=[''],
                    default='LOL')
parser.add_argument('--save_dir', type=str, default='', help='dehaze images save path')
parser.add_argument('--save', action='store_true', help='save dehaze images')
opt = parser.parse_args()

dataset = opt.dataset_name


if not os.path.exists(opt.save_dir):
    os.mkdir(opt.save_dir)
output_dir = os.path.join(opt.save_dir, dataset)
print("pred_dir:", output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

haze_dir = ''
clear_dir = ''
model_dir = ''


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
net = MLWNet()
ckp = torch.load(model_dir)
net = net.to(device)
new_state_dict = {}

for key, value in ckp['model'].items():
    new_key = key[7:]
    new_state_dict[new_key] = value

net.load_state_dict(new_state_dict)
net.eval()
psnr_list = []
ssim_list = []

for im in tqdm(os.listdir(haze_dir)):
    haze = Image.open(os.path.join(haze_dir, im)).convert('RGB')
    clear = Image.open(os.path.join(clear_dir, im)).convert('RGB')
    haze1 = tfs.ToTensor()(haze)[None, ::]
    haze1 = haze1.to(device)
    clear_no = tfs.ToTensor()(clear)[None, ::]
    with torch.no_grad():
        pred,_,_,_ = net(haze1)
    ts = torch.squeeze(pred.clamp(0, 1).cpu())
    pp = psnr(pred.cpu(), clear_no)
    ss = ssim(pred.cpu(), clear_no)
    psnr_list.append(pp)
    ssim_list.append(ss)
    vutils.save_image(ts, os.path.join(output_dir, im))

print(f'Average PSNR is {np.mean(psnr_list)}')
print(f'Average SSIM is {np.mean(ssim_list)}')