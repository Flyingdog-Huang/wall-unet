from test import test
import torch
from os import listdir
from os.path import splitext
import cv2
from utils.data_loading import BasicDataset
import numpy as np
import torch.nn.functional as F

# pri
dir_img = '../data/test/img/'
dir_mask = '../data/test/mask2/'
dir_pths = '../checkpoints/cosin/'

device = torch.device('cpu')

# get pth
pth_list=[(file) for file in listdir(dir_pths) if file.endswith('.pth')]
# print('pth_list',pth_list)

# cap miou
for pth_name in pth_list:
    dir_pth=dir_pths+pth_name
    miou=test(device,dir_img,dir_mask,dir_pth,is_vector=False)
    print('{} MIOU {}'.format(pth_name,miou))

# ensembles
n_classes=2
# load img and label
name_list=[splitext(file)[0] for file in listdir(dir_img) if file.endswith('.png') or file.endswith('.jpg')]
num_data=len(name_list)
for name in name_list:
    img_name=str(dir_img)+name+'.png'
    mask_name=str(dir_mask)+name+'_mask.png'
    img=cv2.imread(img_name)
    if img is None:
        img_name=str(dir_img)+name+'.jpg'
        img=cv2.imread(img_name)
    mask=cv2.imread(mask_name)
    # process data
    img= BasicDataset.preprocess(img, 1, is_mask=False,is_transforms=False)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    mask_true = BasicDataset.preprocess(mask, 1, is_mask=True,is_transforms=False)
    if mask_true.ndim==2:
        mask_true = mask_true[:,:,np.newaxis]
    mask_true = mask_true.transpose((2, 0, 1))
    mask_true = torch.from_numpy(mask_true)
    mask_true = mask_true.to(device=device, dtype=torch.long)
    mask_true = F.one_hot(torch.squeeze(mask_true,dim=1), n_classes).permute(0, 3, 1, 2).float()
    # emble pre
    pre_embel=torch.zeros_like(mask_true).to(device=device)
    print('pre_embel.shape',pre_embel.shape)