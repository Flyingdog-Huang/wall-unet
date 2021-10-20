import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import cv2
from utils.resolveSVG import svg2label

CVC_FP_className_list = [
    'Background', 'Door', 'Parking', 'Room', 'Separation', 'Text', 'Wall',
    'Window'
]


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.') and not file.endswith('.svg')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        # w, h = pil_img.size
        # print()
        # print('***********************************************')
        # print()
        # print('pil_img.shape: ',pil_img.shape)
        h, w = pil_img.shape[:2]
        if h>7000:
            scale=0.1
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # pil_img = pil_img.resize((newW, newH))
        pil_img = cv2.resize(pil_img,(newW,newH))
        # print('pil_img.shape: ',pil_img.shape)
        img_ndarray = np.asarray(pil_img)
        # print()
        # print('***********************************************')
        # print()
        # print('img_ndarray.shape: ',img_ndarray.shape)
        # img_ndarray = img_ndarray.transpose((2, 0, 1))

        if is_mask:
            # CFC svg label
            img_ndarray = img_ndarray[:,:,np.newaxis]
            # print('mask.shape: ',img_ndarray.shape)
            img_ndarray = img_ndarray.transpose((2, 0, 1))
            return img_ndarray
            # self-label by labelme:3-2(no bg)
            # backg,class1,class2=cv2.split(pil_img)
            # he=backg+class1+class2
            # he=255-he
            # pil_img=cv2.merge([he,class1,class2])

        img_ndarray = img_ndarray / 255.0
        # print('data.shape: ',img_ndarray.shape)
        img_ndarray = img_ndarray.transpose((2, 0, 1))

        # if img_ndarray.ndim == 2 and not is_mask:
        #     img_ndarray = img_ndarray[np.newaxis, ...]
        # elif not is_mask:
        #     img_ndarray = img_ndarray.transpose((2, 0, 1))

        # if not is_mask:
        #     img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.svg']:
            return svg2label(filename)
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return cv2.imread(filename)
            # return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        # mask = self.load(mask_file[0])
        # img = self.load(img_file[0])
        mask = self.load(str(mask_file[0]))
        img = self.load(str(img_file[0]))

        # print()
        # print('***********************************************')
        # print()
        # print('img.shape: ',img.shape)
        # print('mask.shape: ',mask.shape)

        assert img.shape[:2] == mask.shape[:2], \
            'Image and mask {name} should be the same shape, but are {img.shape[:2]} and {mask.shape[:2]}'


        # assert img.size == mask.size, \
        #     'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
            # 'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_gt_')
        # super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')


class PimgDataset(Dataset):
    def __init__(self, images_dir: str, pimages_dir: str, masks_dir: str, scale: float = 1.0, pimg_suffix: str = '-DOP', mask_suffix: str = '_mask'):
        self.images_dir = Path(images_dir)
        self.pimages_dir = Path(pimages_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.pimg_suffix = pimg_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        # w, h = pil_img.size
        h, w, chanel = pil_img.shape
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # pil_img = pil_img.resize((newW, newH))
        pil_img = cv2.resize(pil_img,(newW,newH))

        img_ndarray = np.asarray(pil_img)
        img_ndarray = img_ndarray.transpose((2, 0, 1))
        img_ndarray = img_ndarray / 255.0

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return cv2.imread(filename)
            # return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        pname=name.replace('-0',self.pimg_suffix)

        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        pimg_file = list(self.pimages_dir.glob(pname + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(pimg_file) == 1, f'Either no image or multiple images found for the ID {name}: {pimg_file}'

        # mask = self.load(mask_file[0])
        # img = self.load(img_file[0])
        mask = self.load(str(mask_file[0]))
        img = self.load(str(img_file[0]))
        pimg = self.load(str(pimg_file[0]))
        # 处理pimg
        pimg=cv2.cvtColor(pimg,cv2.COLOR_BGR2GRAY)
        h, w, chanel = img.shape
        pimg = cv2.resize(pimg,(w,h))
        # 拼接pimg和img
        img=np.dstack((img,pimg))

        # assert img.size == mask.size, \
        #     'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
            # 'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
