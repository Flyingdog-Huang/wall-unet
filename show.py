import cv2
from utils.wall_vector import wall_vector

dir_img='../data/show/img/'
dir_mask='../data/show/predict/'
name='1186106754'
img_name=str(dir_img)+name+'.png'
mask_name=str(dir_mask)+name+'_pre.png'
img=cv2.imread(img_name)
train_size=2
img=cv2.resize(img,None,fx=1/train_size,fy=1/train_size,interpolation=cv2.INTER_LINEAR)
mask=cv2.imread(mask_name)
img=cv2.resize(img,None,fx=1/train_size,fy=1/train_size,interpolation=cv2.INTER_LINEAR)
mask=cv2.resize(mask,None,fx=1/train_size,fy=1/train_size,interpolation=cv2.INTER_LINEAR)
print('img , mask shape',img.shape,mask.shape)
img_vector=wall_vector(img,mask)
vector_name=name+'_vec.png'
cv2.imwrite(vector_name,img_vector)