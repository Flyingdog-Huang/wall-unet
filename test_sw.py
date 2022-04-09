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

# load img
dir_img = '../../data/bug/img/'
# dir_img = '../../data/bug/'
dir_sw = '../../data/bug/sw/'
name='1'
dir_sw_name = str(Path(dir_sw+name+'/'))
# print('dir_sw_name',dir_sw_name)
if not os.path.exists(dir_sw_name):
    os.makedirs(dir_sw_name)
img_name=dir_img+name+'.jpg'
# print('img_name',img_name)
img=cv2.imread(img_name)
print('img shape',img.shape)

img_i=img/255
img_i=img_i.transpose((2, 0, 1))
img_i = torch.from_numpy(img_i).float()
img_i = img_i.unsqueeze(0)
print('img_i shape',img_i.shape)

# test grids batch 
grids=make_grid(img.shape[:2],window=1024)
len_grids=len(grids)
print('len_grids',len_grids)

tensor_list=[]
tensor_input=None
sw=10
len_grids=7

for i in range(len_grids):
    # x1,x2,y1,y2=grids[i]
    # img_i=img[y1:y2,x1:x2,:]/255
    img_i=np.ones((sw,sw,3))*i
    img_i=img_i.transpose((2, 0, 1))
    img_i = torch.from_numpy(img_i).float()
    img_i = img_i.unsqueeze(0)
    if (i+1)%4==1:
        tensor_input=img_i
        continue
    tensor_input=torch.cat([tensor_input,img_i],0)
    if (i+1)%4==0 or i==len_grids-1:
        tensor_list.append(tensor_input)
        continue

print(len(tensor_list))
print(tensor_list)
print(tensor_list[0].shape)
print(tensor_list[-1].shape)
print(tensor_list[-1][0])
print(tensor_list[-1][0].unsqueeze(0).shape[0])
for i in tensor_list[-1]:
    print(i)
    print(i.shape)



# # save grids
# grids=make_grid(img.shape[:2],window=1024)
# len_grids=len(grids)
# for i in range(len_grids):
#     img_i_name=dir_sw_name+'/'+name+'_'+str(i)+'.png'
#     x1,x2,y1,y2=grids[i]
#     cv2.imwrite(img_i_name,img[y1:y2,x1:x2,:])
