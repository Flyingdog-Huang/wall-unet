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
from utils.data_loading import BasicDataset, CarvanaDataset
from unet import UNet, UnetResnet50, hrnet48, Unet_p1, hrnet48_p1, UNet_fp16, UNet_fp4
from utils.utils import plot_img_and_mask
from tqdm import tqdm
import cv2
from predict import get_args, predict_img
from utils.miou import IOU, MIOU
from utils.wall_vector import wall_vector
from os import listdir
from os.path import splitext

dir_img = '../data/test_vector/img/'
dir_mask = '../data/test_vector/mask/'
dir_pre = '../data/test_vector/predict/'
dir_vec = '../data/test_vector/vector/'
dir_miou = '../data/test_vector/'
dir_pth = './unet.pth'

def test(net,
         device,
         name,
         image,
         gt,
         is_vector:bool=True):
    miou_test=[]
    net.eval()

    # process data
    img= BasicDataset.preprocess(image, 1, is_mask=False,is_transforms=False)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    mask_true = BasicDataset.preprocess(gt, 1, is_mask=True,is_transforms=False)
    if mask_true.ndim==2:
        mask_true = mask_true[:,:,np.newaxis]
    mask_true = mask_true.transpose((2, 0, 1))
    mask_true = torch.from_numpy(mask_true)
    mask_true = mask_true.to(device=device, dtype=torch.long)
    mask_true = F.one_hot(torch.squeeze(mask_true,dim=1), net.n_classes).permute(0, 3, 1, 2).float()
    
    # predict
    with torch.no_grad():
        mask_pred = net(img)
        mask_pred_softmax = F.softmax(mask_pred, dim=1).float()
        mask_pred_onehot = F.one_hot(mask_pred_softmax.argmax(dim=1),net.n_classes).permute(0, 3, 1, 2).float()
        
        # miou - predict and label
        _, miou_pre= MIOU(mask_pred_onehot[:,1:,...], mask_true[:,1:,...])
        miou_test.append(miou_pre)
        
        # save pre
        probs = mask_pred_softmax[0]
        tf = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor()])
        full_mask = tf(probs.cpu()).squeeze()
        full_mask = F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()
        pre_img=np.uint8(full_mask[1]*255)
        img_pre=np.array([pre_img,pre_img,pre_img]).transpose((1,2,0))
        name_pre=dir_pre+name+'_pre.png'
        cv2.imwrite(name_pre,img_pre)
        
        # vector
        if is_vector:
            img_vector=wall_vector(image,img_pre)
            # save vector
            name_vector=dir_vec+name+'_vector.png'
            cv2.imwrite(name_vector,img_vector)
            
            # miou - vector and label
            img_vector=np.array([img_vector,img_vector,img_vector]).transpose((1,2,0))
            mask_vector = BasicDataset.preprocess(img_vector, 1, is_mask=True,is_transforms=False)
            if mask_vector.ndim==2:
                mask_vector = mask_vector[:,:,np.newaxis]
            mask_vector = mask_vector.transpose((2, 0, 1))
            mask_vector = torch.from_numpy(mask_vector)
            mask_vector = mask_vector.to(device=device, dtype=torch.long)
            mask_vector = F.one_hot(torch.squeeze(mask_vector,dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            _, miou_vector= MIOU(mask_vector[:,1:,...], mask_true[:,1:,...])
            miou_test.append(miou_vector)

    return miou_test

if __name__ == '__main__':
    miou=[]
    is_vector=True
    # is_vector=False
    # initlization net
    net = UNet(n_channels=3, n_classes=2)
    # args = get_args()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # logging.info(f'Loading model {args.model}')
    logging.info(f'Loading model {dir_pth}')
    logging.info(f'Using device {device}')
    net.to(device=device)
    # net.load_state_dict(torch.load(args.model, map_location=device))
    net.load_state_dict(torch.load(dir_pth, map_location=device))
    logging.info('Model loaded!')

    # load img and label
    name_list=[splitext(file)[0] for file in listdir(dir_img) if file.endswith('.png') or file.endswith('.jpg')]
    num_data=len(name_list)
    for name in tqdm(name_list,
                      total=num_data,
                      desc='Test round',
                      unit='pic',
                      leave=False):
        img_name=str(dir_img)+name+'.png'
        mask_name=str(dir_mask)+name+'_mask.png'
        img=cv2.imread(img_name)
        mask=cv2.imread(mask_name)
        reduce_rate=max(img.shape[0],img.shape[1])//1000
        img=cv2.resize(img,None,fx=1/2**reduce_rate,fy=1/2**reduce_rate,interpolation=cv2.INTER_LINEAR)
        mask=cv2.resize(mask,None,fx=1/2**reduce_rate,fy=1/2**reduce_rate,interpolation=cv2.INTER_LINEAR)
        # print(img)
        # test IOU
        miou_test=test(net,device,name,img,mask,is_vector)
        miou_dict={}
        miou_dict['name']=name
        miou_dict['shape']=[img.shape[0]//2**reduce_rate,img.shape[1]//2**reduce_rate]
        miou_dict['pre miou']=miou_test[0]
        miou_dict['vector miou']=miou_test[1] if len(miou_test)==2 else None
        miou.append(miou_dict)
    
    # save result as txt
    with open(dir_miou+'MIOU.txt','w') as file:
        [file.write(str(it)+'\n') for it in miou]
        file.close()