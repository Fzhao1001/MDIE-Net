import os,argparse
import numpy as np
from PIL import Image
from models import *
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from models.Padiff_arch.MDIE_arch import MLWNet
from metrics import psnr, ssim
abs=os.getcwd()+'/'
def tensorShow(tensors,titles):
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(221+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

parser=argparse.ArgumentParser()
parser.add_argument('--task',type=str,default='ots',help='its or ots')
parser.add_argument('--test_imgs',type=str,default='test_ufo',help='Test imgs folder')
opt=parser.parse_args()
dataset=opt.task
haze_dir = ''
clear_dir = ''
model_dir = ''
output_dir = ''
print("pred_dir:",output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

device='cuda' if torch.cuda.is_available() else 'cpu'
ckp=torch.load(model_dir,map_location=device)
net=MLWNet()
net=nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net.eval()

for im in os.listdir(haze_dir):
    haze = Image.open(os.path.join(haze_dir, im)).convert('RGB')
    clear = Image.open(os.path.join(clear_dir, im)).convert('RGB')
    haze1 = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
    ])(haze)[None, ::]
    haze1 = haze1.to(device)
    clear_no = tfs.ToTensor()(clear)[None, ::]
    with torch.no_grad():
        pred, _, _, _ = net(haze1)
    ts = torch.squeeze(pred.clamp(0, 1).cpu())

    vutils.save_image(ts, os.path.join(output_dir, im))

