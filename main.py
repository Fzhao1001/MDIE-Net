import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
from metrics import psnr,ssim
from models import *
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
#from tensorboardX import SummaryWriter
import torchvision.utils as vutils
warnings.filterwarnings('ignore')
from option import opt,model_name,log_dir
from data_utils import *
from torchvision.models import vgg16
from C2R import C2R
from Retinex_rl import Inference
from loss import *
from models.Padiff_arch.MDIE_arch import MLWNet

print('log_dir :',log_dir)
print('model_name:',model_name)

schedule_opt={
    "schedule": "linear",
    "n_timestep": 2000,
    "linear_start": 1e-6,
    "linear_end": 1e-2,
}

models_={
    'Lgnet':LGNet(gps=opt.gps,blocks=opt.blocks),
    'CDUnetv2':CDUnetV2(),
    'initnet':Inference(),
    'wf': WfDiffx2(schedule_opt=schedule_opt),
    'MFEdiff':MLWNet(),
    'ipt':ipt(),
}
loaders_={
    'its_train':ITS_train_loader,
    'its_test':ITS_test_loader,
}
start_time=time.time()
T=opt.epochs
def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
    lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
    return lr


def train(net,loader_train,loader_test,optim,criterion):
    losses=[]
    start_epoch=0
    max_ssim=0
    max_psnr=0
    ssims=[]
    psnrs=[]
    opt.resume = False
    if opt.resume and os.path.exists(opt.model_dir):
        print(f'resume from {opt.model_dir}')
        ckp=torch.load(opt.model_dir)
        losses=ckp['losses']
        net.load_state_dict(ckp['model'])
        start_epoch=ckp['epoch']
        max_ssim=ckp['max_ssim']
        max_psnr=ckp['max_psnr']
        psnrs=ckp['psnrs']
        ssims=ckp['ssims']
        print(f'start_epoch:{start_epoch} start training ---')
    else :
        print('train from scratch *** ')
    for epoch in range(start_epoch+1,opt.epochs+1):
        net.train()
        lr=opt.lr
        if not opt.no_lr_sche:
            lr=lr_schedule_cosdecay(epoch,T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
        for batch_idx,(x,y) in enumerate(loader_train):
            x=x.to(opt.device);y=y.to(opt.device)
            x_out = net(x)

            n, c, h, w = y.shape

            l_rec=criterion[0](x_out,y)
            l_per=criterion[1](x_out,y)
            l_con=criterion[2](x_out,y,x)

            loss=l_rec +l_per*0.3 + 0.5*l_con

            loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(loss.item())
            print(f'\rtrain loss : {loss.item():.5f}| epoch :{epoch}/{opt.epochs}| batch :{batch_idx+1}/{len(loader_train)}|lr :{lr :.7f} |time_used :{(time.time()-start_time)/60 :.1f}',end='',flush=True)

        if epoch % opt.eval_epoch ==0 :
            with torch.no_grad():
                ssim_eval,psnr_eval=test(net,loader_test, max_psnr,max_ssim,epoch)

            print(f'\nepoch :{epoch} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            if ssim_eval > max_ssim and psnr_eval > max_psnr :
                max_ssim=max(max_ssim,ssim_eval)
                max_psnr=max(max_psnr,psnr_eval)
                torch.save({
                            'epoch':epoch,
                            'max_psnr':max_psnr,
                            'max_ssim':max_ssim,
                            'ssims':ssims,
                            'psnrs':psnrs,
                            'losses':losses,
                            'model':net.state_dict()
                },opt.model_dir)
                print(f'\n model saved at epoch :{epoch}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

    np.save(f'./numpy_files/{model_name}_{opt.epochs}_losses.npy',losses)
    np.save(f'./numpy_files/{model_name}_{opt.epochs}_ssims.npy',ssims)
    np.save(f'./numpy_files/{model_name}_{opt.epochs}_psnrs.npy',psnrs)
"""
def test(net,loader_test,max_psnr,max_ssim,step):
    net.eval()
    torch.cuda.empty_cache()
    dwt, idwt = DWT(), IWT()
    ssims=[]
    psnrs=[]
    #s=True
    for i ,(inputs,targets) in enumerate(loader_test):
        inputs=inputs.to(opt.device);targets=targets.to(opt.device)
        pred=net(inputs)
        # # print(pred)
        # tfs.ToPILImage()(torch.squeeze(targets.cpu())).save('111.png')
        # vutils.save_image(targets.cpu(),'target.png')
        # vutils.save_image(pred.cpu(),'pred.png')
        ssim1=ssim(pred,targets).item()
        psnr1=psnr(pred,targets)
        ssims.append(ssim1)
        psnrs.append(psnr1)
        #if (psnr1>max_psnr or ssim1 > max_ssim) and s :
        #		ts=vutils.make_grid([torch.squeeze(inputs.cpu()),torch.squeeze(targets.cpu()),torch.squeeze(pred.clamp(0,1).cpu())])
        #		vutils.save_image(ts,f'samples/{model_name}/{step}_{psnr1:.4}_{ssim1:.4}.png')
        #		s=False
    return np.mean(ssims) ,np.mean(psnrs)
"""
def test(net,loader_test,max_psnr,max_ssim,step):
    net.eval()
    torch.cuda.empty_cache()

    ssims=[]
    psnrs=[]
    #s=True
    for i ,(inputs,targets) in enumerate(loader_test):
        inputs=inputs.to(opt.device);targets=targets.to(opt.device)
        out1=net(inputs)

        pred = out1[0]

        ssim1=ssim(pred,targets).item()
        psnr1=psnr(pred,targets)
        ssims.append(ssim1)
        psnrs.append(psnr1)
        #if (psnr1>max_psnr or ssim1 > max_ssim) and s :
        #		ts=vutils.make_grid([torch.squeeze(inputs.cpu()),torch.squeeze(targets.cpu()),torch.squeeze(pred.clamp(0,1).cpu())])
        #		vutils.save_image(ts,f'samples/{model_name}/{step}_{psnr1:.4}_{ssim1:.4}.png')
        #		s=False
    return np.mean(ssims) ,np.mean(psnrs)


if __name__ == "__main__":
    loader_train=loaders_['its_train']
    loader_test=loaders_['its_test']
    net=models_[opt.net]
    net=net.to(opt.device)
    if opt.device=='cuda':
        net=torch.nn.DataParallel(net)
        cudnn.benchmark=True
    criterion = []
    criterion.append(SRN_loss().to(opt.device))

    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.to(opt.device)
    for param in vgg_model.parameters():
        param.requires_grad = False

    criterion.append(PerLoss(vgg_model).to(opt.device))
    criterion.append(MFCloss().to(opt.device))

    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()),lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
    optimizer.zero_grad()
    train(net,loader_train,loader_test,optimizer,criterion)