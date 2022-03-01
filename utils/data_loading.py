import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2
from torchvision.transforms import transforms
from utils.resolveSVG import svg2label
from utils.data_augmentation import img_aug, img_mask_aug

CVC_FP_className_list = [
    'Background', 'Door', 'Parking', 'Room', 'Separation', 'Text', 'Wall',
    'Window'
]

window_size=512 
# window_size=1024
thred_ratio=0.01

# 自适应图像滑动分块
def make_grid(shape, window=window_size, min_overlap=4):
    '''
        return - array of grid : [number, coord(x1,x2,y1,y2)]
    '''
    # logging.info('shift window size is: {}'.format(window_size)) 
    grid_list=[]
    h,w=shape
    overlap=int(window//min_overlap)
    not_overlap=window-overlap
    window_h=window if h>window else h
    window_w=window if w>window else w
    if max(shape) <= window:
        # shape smaller than window
        grid=[0,w,0,h]
        grid_list.append(grid)
    else:
        nh=(h-window_h)//not_overlap+2 if (h-window_h)>0 else 1
        y_list=[0 for i in range(nh)]
        for i in range(1,nh):
            y_list[i]=y_list[i-1]+not_overlap
            if y_list[i]+window_h>h:
                y_list[i]=h-window_h
                break

        nw=(w-window_w)//not_overlap+2 if (w-window_w)>0 else 1
        x_list=[0 for i in range(nw)]
        for i in range(1,nw):
            x_list[i]=x_list[i-1]+not_overlap
            if x_list[i]+window_w>w:
                x_list[i]=w-window_w
                break
        
        for xi in x_list:
            for yi in y_list:
                x1=xi
                y1=yi
                x2=x1+window_w
                y2=y1+window_h
                grid=[x1,x2,y1,y2]
                grid_list.append(grid)

    return np.array(grid_list)

class GridDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, is_transforms:bool=False, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.is_transforms=is_transforms
        self.grid_img=[]
        self.grid_mask=[]
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if file.endswith('.png') or file.endswith('.jpg')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Number of imgs is {len(self.ids)} ')
        self.get_grid()
        logging.info(f'Creating dataset with {len(self.grid_img)} examples')

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask,is_transforms):
        h, w = pil_img.shape[:2]
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = cv2.resize(pil_img,(newW,newH))
        img_ndarray = np.asarray(pil_img)
        if is_mask:
            # class=2
            img_ndarray=img_ndarray[:,:,0]/255
            return img_ndarray
        # img augmentation
        if is_transforms :
            img_ndarray = img_aug(img_ndarray)
        else:
            img_ndarray = img_ndarray / 255.0
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

    def get_grid(self):
        logging.info('shift window size is: {}'.format(window_size)) 
        thred_low=thred_ratio*window_size**2
        thred_high=(1-thred_ratio)*window_size**2
        for name in self.ids:
            mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '*'))
            img_file = list(self.images_dir.glob(name + '.*'))
            assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
            assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        
            mask = self.load(str(mask_file[0]))
            img = self.load(str(img_file[0]))
            assert img.shape[:2] == mask.shape[:2], \
            'Image and mask {name} should be the same shape, but are {img.shape[:2]} and {mask.shape[:2]}'
            
            iter_shape = img.shape[:2]
            windows_list=make_grid(iter_shape)

            for win_points in windows_list:
                x1,x2,y1,y2=win_points
                grid_img=img[y1:y2,x1:x2,:]
                grid_mask=mask[y1:y2,x1:x2,:]
                # no filter
                self.grid_img.append(grid_img)
                self.grid_mask.append(grid_mask)
                # data filter
                # if grid_mask.sum()/(255*3)>thred_low and grid_mask.sum()/(255*3)<thred_high:
                #     self.grid_img.append(grid_img)
                #     self.grid_mask.append(grid_mask)
                # else:
                #     if np.random.rand()>0.5:
                #         self.grid_img.append(grid_img)
                #         self.grid_mask.append(grid_mask)

    def __getitem__(self, idx):
        img=self.grid_img[idx]
        mask=self.grid_mask[idx]
        img = self.preprocess(img, self.scale, is_mask=False,is_transforms=self.is_transforms)
        mask = self.preprocess(mask, self.scale, is_mask=True,is_transforms=self.is_transforms)
        if self.is_transforms: 
            img,mask=img_mask_aug(img,mask)
        img = img.transpose((2, 0, 1))
        if mask.ndim==2:
            mask = mask[:,:,np.newaxis]
        mask = mask.transpose((2, 0, 1))
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
        }

    def __len__(self):
        return len(self.grid_img)


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, is_transforms:bool=False, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.is_transforms=is_transforms

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if file.endswith('.png') or file.endswith('.jpg')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask,is_transforms):
        # w, h = pil_img.size
        # print()
        # print('***********************************************')
        # print()
        # print('pil_img.shape: ',pil_img.shape)
        h, w = pil_img.shape[:2]
        # if h>7000:
        #     scale=0.1

        # resize <1000 and keep scale
        # scale = 1/(max(h,w)//1000+1)
        # logging.info('scale: {}'.format(scale))

        newW, newH = int(scale * w), int(scale * h)
        # logging.info('newW: {}, newH: {}'.format(newW, newH))

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
            # img_ndarray = img_ndarray[:,:,np.newaxis]
            # print('mask.shape: ',img_ndarray.shape)
            # img_ndarray = img_ndarray.transpose((2, 0, 1))
            
            # class=2
            img_ndarray=img_ndarray[:,:,0]/255

            # # class=3
            # img_ndarray_nowall=img_ndarray[:,:,0]/255
            # img_ndarray_wall=img_ndarray[:,:,1]/255
            # img_ndarray_mainwall=img_ndarray[:,:,2]/255
            # img_ndarray=img_ndarray_nowall+img_ndarray_wall+img_ndarray_mainwall*2
            return img_ndarray
            # self-label by labelme:3-2(no bg)
            # backg,class1,class2=cv2.split(pil_img)
            # he=backg+class1+class2
            # he=255-he
            # pil_img=cv2.merge([he,class1,class2])

        # img augmentation
        if is_transforms :
            img_ndarray = img_aug(img_ndarray)
        else:
            img_ndarray = img_ndarray / 255.0
        
  

        # print('data.shape: ',img_ndarray.shape)
        # img_ndarray = img_ndarray.transpose((2, 0, 1))

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
        # print()
        # print('*****************************')
        # print('__getitem__ is called')
        # print('*****************************')
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
        # print('after load')
        # print('img.shape: ',img.shape)
        # print('mask.shape: ',mask.shape)

        assert img.shape[:2] == mask.shape[:2], \
            'Image and mask {name} should be the same shape, but are {img.shape[:2]} and {mask.shape[:2]}'


        # assert img.size == mask.size, \
        #     'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False,is_transforms=self.is_transforms)
        mask = self.preprocess(mask, self.scale, is_mask=True,is_transforms=self.is_transforms)

        # print('before Augmentation')
        # print('mask.ndim: ',mask.ndim) 

        # print()
        # print('***********************************************')
        # print('after preprocess')
        # print('img.shape: ',img.shape)
        # print('mask.shape: ',mask.shape)

        # print()
        # print('self.is_transforms: ',self.is_transforms)
        # Augmentation 
             
        if self.is_transforms: 
            img,mask=img_mask_aug(img,mask)
        
        # print()
        # print('***********************************************')
        # print('after Augmentation')
        # print('img.shape: ',img.shape)
        # print('mask.shape: ',mask.shape)
        # print('mask.ndim: ',mask.ndim)
        img = img.transpose((2, 0, 1))
        if mask.ndim==2:
            mask = mask[:,:,np.newaxis]
        mask = mask.transpose((2, 0, 1))

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
            # 'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1, is_transforms=False):
        super().__init__(images_dir, masks_dir, scale, is_transforms, mask_suffix='_mask')
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
