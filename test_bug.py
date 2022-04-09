import argparse
import logging
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset, CarvanaDataset, make_grid
from unet import UNet, UnetResnet50, hrnet48, Unet_p1, hrnet48_p1, UNet_fp16, UNet_fp4
from utils.utils import plot_img_and_mask
from tqdm import tqdm
import cv2
from predict import get_args, predict_img
from utils.miou import IOU, MIOU
from utils.wall_vector import wall_vector
from os import listdir
from os.path import splitext
import time

def process_img(img,device):
    img= BasicDataset.preprocess(img, 1, is_mask=False,is_transforms=False)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    return img

# bug path
dir_img = '../../data/bug/img/'
dir_pre = '../../data/bug/predict/'

# pth
# dir_pth = '../../checkpoints/checkpoint_epoch41_priclean_swf_BCE_unet_iou91.pth'
# dir_pth = '../../checkpoints/checkpoint_epoch16_priclean_sw_BCE_unet_iou83.pth'
dir_pth = '../../checkpoints/sw_unet_93.pth'

# device
cuda_name = 'cuda'  # 'cuda:1'
device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
print('device',device)

# initlization net
net = UNet(n_channels=3, n_classes=2)
net.load_state_dict(torch.load(dir_pth, map_location=device))
net.eval()

# get bug img name list
name_list=[splitext(file)[0] for file in listdir(dir_img) if file.endswith('.png') or file.endswith('.jpg')]
num_data=len(name_list)

# inference
for name in name_list:
    # load img 
    print('name',name)
    img_name=str(dir_img)+name+'.png'
    img=cv2.imread(img_name)
    if img is None:
        img_name=str(dir_img)+name+'.jpg'
        img=cv2.imread(img_name)
    # predict
    mask_pred=torch.zeros(1,net.n_classes,img.shape[0],img.shape[1]).to(device=device)
    weight_mask=torch.zeros(1,1,img.shape[0],img.shape[1]).to(device=device)
    grids=make_grid(img.shape[:2],window=1024)
    for points in grids:
        x1,x2,y1,y2=points
        weight_mask[:,:,y1:y2,x1:x2]+=1
        img1=process_img(img[y1:y2,x1:x2,:],device)
        print('img1.shape',img1.shape)
        with torch.no_grad():
            mask_pred[:,:,y1:y2,x1:x2] += net(img1)
    mask_pred=mask_pred/weight_mask
    print('mask_pred.shape',mask_pred.shape)
    #  save pre
    mask_pred_softmax = F.softmax(mask_pred, dim=1).float()
    print('mask_pred_softmax.shape',mask_pred_softmax.shape)
    probs = mask_pred_softmax[0]
    print('probs.shape',probs.shape)
    tf = transforms.Compose([transforms.ToPILImage(),
                                transforms.ToTensor()])
    full_mask = tf(probs).squeeze()
    full_mask = F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()
    pre_img=np.uint8(full_mask[1]*255)
    print('pre_img.shape',pre_img.shape)
    name_pre=dir_pre+name+'_pre.png'
    cv2.imwrite(name_pre,pre_img)
