import argparse
import torch
import torch.nn as nn
from models.Math_Module import P, Q
from models.decom import Decom
import os
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import time
from utils.Rexutils import *

def one2three(x):
    return torch.cat([x, x, x], dim=1).to(x)


class Inference(nn.Module):
    def __init__(self, ):
        super().__init__()
        # loading decomposition model
        self.model_Decom_low = Decom().cuda()
        self.model_Decom_low = load_initialize(self.model_Decom_low, "")

        self.P = P()
        self.Q = Q()
        transform = [
            transforms.ToTensor(),
        ]
        self.transform = transforms.Compose(transform)

    def RIinit(self, input_low_img):

        P, Q = self.model_Decom_low(input_low_img)

        return P, Q

    def forward(self, input_low_img):
        if torch.cuda.is_available():
            input_low_img = input_low_img.cuda()
        with torch.no_grad():
            R, L = self.RIinit(input_low_img)

        return R, L
'''
    def runm(self, low_img_path):
        file_name = low_img_path
        name = file_name.split('.')[0]
        low_img = self.transform(Image.open(low_img_path)).unsqueeze(0)
        R, L = self.forward(input_low_img=low_img)
        if not os.path.exists(self.opts.output):
            os.makedirs(self.opts.output)
        save_path = os.path.join(self.opts.output+"_r", file_name.replace(name, "%s_R" % (name)))
        save_path2 = os.path.join(self.opts.output+"_l", file_name.replace(name, "%s_L" % (name)))
        np_save_TensorImg(R, save_path)
        np_save_TensorImg(L, save_path2)
        print("================================= time for %s============================" % (file_name))

    def run(self, image_dir):
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                filepath = os.path.join(image_dir, filename)
                self.runm(filepath)

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configure')
    # specify your data path here!
    parser.add_argument('--img_path', type=str, default="")
    parser.add_argument('--output', type=str, default="")
    # ratio are recommended to be 3-5, bigger ratio will lead to over-exposure
    parser.add_argument('--ratio', type=int, default=5)
    # model path
    parser.add_argument('--Decom_model_low_path', type=str, default="")
    parser.add_argument('--unfolding_model_path', type=str, default="")
    parser.add_argument('--adjust_model_path', type=str, default="")
    parser.add_argument('--gpu_id', type=int, default=0)

    opts = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu_id)