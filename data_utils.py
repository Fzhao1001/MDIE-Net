import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt
BS=opt.bs
print(BS)
crop_size='whole_img'
if opt.crop:
    crop_size=opt.crop_size

def tensorShow(tensors,titles=None):
        '''
        t:BCWH
        '''
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(211+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

class RESIDE_Dataset(data.Dataset):
    def __init__(self,path,train,size=crop_size,format='.png'):
        super(RESIDE_Dataset,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,'raw'))
        self.haze_imgs=[os.path.join(path,'raw',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'gt')
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])

        img=self.haze_imgs[index]
        id=img.split('/')[-1]
        # id = img.split('\\')[-1]
        clear_name=id
        # clear_name = id
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)

        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB") )
        return haze,clear
    def augData(self,data,target):
        data=tfs.ToTensor()(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.haze_imgs)

class RESIDE_Dataset2(data.Dataset):
    def __init__(self,path,train,size=crop_size,format='.png'):
        super(RESIDE_Dataset2,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,'raw'))
        self.haze_imgs=[os.path.join(path,'raw',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'gt')
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
        img=self.haze_imgs[index]
        id=img.split('/')[-1]
        clear_name=id
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB") )
        return haze,clear
    def augData(self,data,target):
        data=tfs.ToTensor()(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.haze_imgs)

class RESIDE_Dataset3(data.Dataset):
    def __init__(self,path,train,size=crop_size,format='.png'):
        super(RESIDE_Dataset3,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,'raw'))
        self.haze_imgs=[os.path.join(path,'raw',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'gt')
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
        img=self.haze_imgs[index]
        id=img.split('/')[-1]
        clear_name=id
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB") )
        return haze,clear
    def augData(self,data,target):
        data=tfs.ToTensor()(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):

        return len(self.haze_imgs)

class Unet_Dataset(data.Dataset):
    def __init__(self,path,train,size=crop_size,format='.png'):
        super(Unet_Dataset,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,'raw'))
        self.haze_imgs=[os.path.join(path,'raw',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'gt')
        self.conid = 1
        self.con_dir = ""
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
        img=self.haze_imgs[index]
        id=img.split('/')[-1]
        clear_name=id
        conid = str(self.conid)+".png"
        self.conid = self.conid+1
        if(self.conid==700):
            self.conid=1
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        con = Image.open(os.path.join(self.con_dir, conid))
        con = tfs.CenterCrop(haze.size[::-1])(con)
        haze,clear,con=self.augData(haze.convert("RGB") ,clear.convert("RGB"), con.convert("RGB") )
        return haze,clear,con
    def augData(self,data,target,con):
        data=tfs.ToTensor()(data)
        target=tfs.ToTensor()(target)
        con = tfs.ToTensor()(con)
        return  data ,target,con
    def __len__(self):
        return len(self.haze_imgs)

import os
pwd=os.getcwd()
print(pwd)
path=''#path to your 'data' folder

ITS_train_loader=DataLoader(dataset=RESIDE_Dataset(path+'/train/',train=True,size=crop_size),batch_size=opt.batch_size,shuffle=True)
ITS_test_loader=DataLoader(dataset=RESIDE_Dataset(path+'/test/',train=False,size='whole img'),batch_size=1,shuffle=False)



if __name__ == "__main__":
    pass