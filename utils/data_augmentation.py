import numpy as np
import cv2

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

def mosaic_aug():