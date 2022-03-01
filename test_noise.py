import numpy as np
import torch
from utils.data_loading import make_grid
import cv2

# img=torch.rand(1,3,2,3)
# print('img.shape',img.shape)
# print('img',img)
# noise=np.random.normal(0,0.01**0.5,img.shape)
# print('noise.shape',noise.shape)
# print('noise',noise)
# img_noise=img.numpy()+noise
# print('img_noise.shape',img_noise.shape)
# print('img_noise',img_noise)
# img=torch.tensor(np.clip(img_noise,0,1))
# print('img.shape',img.shape)
# print('img',img)

# num=3
# l=[0 for i in range(3)]
# print(l)

# shape=(25,25)
# print(make_grid(shape))

# img=cv2.imread('./testData/15902396.png')
# shape=img.shape[:2]
# print('shape:',shape)
# grids_list=make_grid(shape)
# print('grid_list:',grids_list)
# for i in range(len(grids_list)):
#     x1,x2,y1,y2=grids_list[i]
#     img_grid=img[y1:y2,x1:x2,:]
#     cv2.imshow('NO.{}'.format(str(i)),img_grid)
# cv2.waitKey(0)

# a=2
# b=3
# c=4
# f1=b*c**a
# f2=c**a*b
# for i in range(f1):
#     print(np.random.randint(100))
# print(f1,f2)

# img=torch.tensor([[1,2,3],[1,2,3]])
# weight_mask=torch.tensor([1,2,3])
# predict=img/weight_mask
# print('img',img)
# print('weight mask',weight_mask)
# print('predict',predict)
# predict[:,0]=predict[:,0]+1
# print('predict',predict)

# shape=[8,10]
# win=5
# grids=make_grid(shape,win)
# print('grids',grids)
# mask=torch.zeros(shape)
# for points in grids:
#     x1,x2,y1,y2=points
#     mask[y1:y2,x1:x2]+=1
# print('mask',mask)

l=[1,2,3,4,5]
print(l[:2])