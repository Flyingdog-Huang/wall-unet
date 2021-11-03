import numpy as np
import cv2
from os import listdir
from os.path import splitext
from pathlib import Path
import random

def img_aug(img):
    # in: (w,h,c)-255
    # out: (w,h,c)-1.0
    # gaussian filter(or 双边滤波)
    if  np.random.rand()>0.5:
        img=cv2.GaussianBlur(img,(3,3),0) 
    
    # contrast-对比度
    if  np.random.rand()>0.5:
        contrast=np.random.randint(-50,50)
        img=img*(contrast/127+1)-contrast
    
    img = img / 255.0
    # gaussian noise-高斯噪音
    noise=np.random.normal(0,0.01**0.5,img.shape)
    img+=noise
    img=np.clip(img,0,1)
    return img

def img_mask_aug(img,mask):
    # in: img/mask(w,h,c)-1.0
    # out: img(w,h,c)/mask(w,h)-1.0
    # print('Augmentation')
    # print()
    # para transforms 
    h,w,c=img.shape
    img_shape=(w,h)
    # rotation-旋转——+/-10°
    if np.random.rand()>0.5:
        # print('rotation')
        angle=np.random.uniform(-10,10)
        center=(w//2,h//2)
        M=cv2.getRotationMatrix2D(center,angle,1)
        img=cv2.warpAffine(img,M,img_shape)
        mask=cv2.warpAffine(mask,M,img_shape)
    # shift-平移
    if np.random.rand()>0.5:
        # print('shift')
        x_shift=np.random.randint(-int(w*0.2),int(w*0.2))
        y_shift=np.random.randint(-int(h*0.2),int(h*0.2))
        M=np.float32([[1,0,x_shift],[0,1,y_shift]])
        img=cv2.warpAffine(img,M,img_shape)
        mask=cv2.warpAffine(mask,M,img_shape)
    # inversion -反转:1-水平翻转;0-	垂直翻转;-1-水平垂直翻转
    flip_modes=['水平垂直','垂直','水平']
    for flip_mode in range(len(flip_modes)):
        if np.random.rand()>0.5:
            # print(flip_modes[flip_mode])
            img=cv2.flip(img, flip_mode-1)
            mask=cv2.flip(mask, flip_mode-1)
    # cropping - 剪裁 - 裁剪坐标为[y0:y1, x0:x1]
    if np.random.rand()>0.5:
        # print('cropping')
        x_crop=np.random.randint(int(w*0.3))
        y_crop=np.random.randint(int(h*0.3))
        img=img[y_crop:y_crop+int(h*0.7),x_crop:x_crop+int(w*0.7)]
        mask=mask[y_crop:y_crop+int(h*0.7),x_crop:x_crop+int(w*0.7)]
    return img,mask

def mosaic_load(img_id,all_name_list,images_path,masks_path):
    imgs=[]
    masks=[]
    for id in img_id:
        name=all_name_list[id]
        img_name=list(images_path.glob(name + '.*'))[0]
        mask_name=list(masks_path.glob(name + '_gt_*'))[0]
        img=cv2.imread(str(img_name))
        mask=cv2.imread(str(mask_name))
        # data aug
        # 对每张图像img数据进行随机增广
        img=img_aug(img)
        img=img*255
        img=img.astype(np.uint8)
        # print(img)
        # 对img,mask数据进行随机增广
        img,mask=img_mask_aug(img,mask)
        imgs.append(img)
        masks.append(mask)

    return imgs,masks


def mosaic_aug(images_dir,masks_dir):
    images_path=Path(images_dir)
    masks_path=Path(masks_dir)

    # get all imgs name
    all_name_list=[splitext(file)[0] for file in listdir(images_path) if (file.endswith('.png') or file.endswith('.jpg')) and not file.endswith('_mosaic_mask.png')]
    # print(all_name_list)
    no_aug=int(len(all_name_list)//4)
    mask_suffix='_mosaic_mask.png'
    img_suffix='_mosaic_img.png'

    for i in range(no_aug):
        # random load 4 imgs 
        img_id=random.sample(range(len(all_name_list)),4)
        # print(img_id)
        imgs,masks=mosaic_load(img_id,all_name_list,images_path,masks_path)
        img1,img2,img3,img4=imgs
        mask1,mask2,mask3,mask4=masks

        # find mini weight and hight: img.shape=(hight,weight,channel)
        hight_min=min(imgs[0].shape[0],imgs[1].shape[0],imgs[2].shape[0],imgs[3].shape[0])
        weight_min=min(imgs[0].shape[1],imgs[1].shape[1],imgs[2].shape[1],imgs[3].shape[1])
        # print('weight_min,hight_min: ',weight_min,hight_min)

        # random mosaic point
        x_mosaic=np.random.randint(int(weight_min*0.3),int(weight_min*0.7))
        y_mosaic=np.random.randint(int(hight_min*0.3),int(hight_min*0.7))
        # print('x_mosaic,y_mosaic: ',x_mosaic,y_mosaic)

        # mosaic concatenate
        # 等比例缩放分割点
        x1_mosaic=int(x_mosaic*(img1.shape[1]/weight_min))
        y1_mosaic=int(y_mosaic*(img1.shape[0]/hight_min))
        y2_mosaic=int(y_mosaic*(img2.shape[0]/hight_min))
        # x轴方向拼接
        img11_mosaic=img1[y1_mosaic-y_mosaic:y1_mosaic,x1_mosaic-x_mosaic:x1_mosaic]
        img12_mosaic=img2[y2_mosaic-y_mosaic:y2_mosaic,x_mosaic:weight_min]
        # print('img11_mosaic.shape: ',img11_mosaic.shape)
        # print('img12_mosaic.shape: ',img12_mosaic.shape)
        img1_mosaic=np.concatenate((img11_mosaic,img12_mosaic),axis=1) 
        # print('img1_mosaic.shape: ',img1_mosaic.shape)
        mask11_mosaic=mask1[y1_mosaic-y_mosaic:y1_mosaic,x1_mosaic-x_mosaic:x1_mosaic]
        mask12_mosaic=mask2[y2_mosaic-y_mosaic:y2_mosaic,x_mosaic:weight_min]
        # print('mask11_mosaic.shape: ',mask11_mosaic.shape)
        # print('mask12_mosaic.shape: ',mask12_mosaic.shape)
        mask1_mosaic=np.concatenate((mask11_mosaic,mask12_mosaic),axis=1) 
        # print('mask1_mosaic.shape: ',mask1_mosaic.shape)
        # 等比例缩放分割点
        x3_mosaic=int(x_mosaic*(img3.shape[1]/weight_min))
        # x轴方向拼接
        img21_mosaic=img3[y_mosaic:hight_min,x3_mosaic-x_mosaic:x3_mosaic]
        img22_mosaic=img4[y_mosaic:hight_min,x_mosaic:weight_min]
        # print('img21_mosaic.shape: ',img21_mosaic.shape)
        # print('img22_mosaic.shape: ',img22_mosaic.shape)
        img2_mosaic=np.concatenate((img21_mosaic,img22_mosaic),axis=1) 
        # print('img2_mosaic.shape: ',img2_mosaic.shape)
        mask21_mosaic=mask3[y_mosaic:hight_min,x3_mosaic-x_mosaic:x3_mosaic]
        mask22_mosaic=mask4[y_mosaic:hight_min,x_mosaic:weight_min]
        # print('mask21_mosaic.shape: ',mask21_mosaic.shape)
        # print('mask22_mosaic.shape: ',mask22_mosaic.shape)
        mask2_mosaic=np.concatenate((mask21_mosaic,mask22_mosaic),axis=1) 
        # print('mask2_mosaic.shape: ',mask2_mosaic.shape)
        # Y轴方向拼接
        img_mosaic=np.concatenate((img1_mosaic,img2_mosaic),axis=0)
        # print('img_mosaic.shape: ',img_mosaic.shape)
        mask_mosaic=np.concatenate((mask1_mosaic,mask2_mosaic),axis=0)
        # print('mask_mosaic.shape: ',mask_mosaic.shape)
        # write mosaic
        img_name=str(images_dir)+str(i)+img_suffix
        mask_name=str(masks_dir)+str(i)+mask_suffix
        cv2.imwrite(img_name,img_mosaic)
        cv2.imwrite(mask_name,mask_mosaic)


        


if __name__ == '__main__':
    images_dir='../../../../../data/floorplan/CVC-FP/'
    masks_dir='../../../../../data/floorplan/CVC-FP/masks/'
    mosaic_aug(images_dir,masks_dir)