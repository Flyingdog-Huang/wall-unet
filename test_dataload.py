from os import listdir
from os.path import splitext
from pathlib import Path
import cv2
import numpy as np

from test import test

dir_img = '../data/test/img/'
dir_mask = '../data/test/mask/'
name_list=[splitext(file)[0] for file in listdir(dir_img) if file.endswith('.png') or file.endswith('.jpg')]
# print(name_list)
for name in name_list:
    img_name=str(dir_img)+name+'.png'
    mask_name=str(dir_mask)+name+'_mask.png'
    img=cv2.imread(img_name)
    mask=cv2.imread(mask_name)
    print('mask shape: ',mask.shape)
    test_merge=np.array([mask[:,:,0],mask[:,:,0],mask[:,:,0]]).transpose((1,2,0))
    print('merge shape: ',test_merge.shape)