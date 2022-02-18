import cv2
import numpy as np
from utils.wall_vector import findMaxMIOU, campIOU, isin_cntflag
import random
import torch
from torch.functional import Tensor

# test tensor
class_num=2
# wall_weight=[1,1]
wall_weight=[0.67,0.33]
loss_weight=torch.ones([class_num,3,3])
# print('loss_weight',loss_weight[0,0,:,:])
for i in range(class_num):
    loss_weight[i,:,:]=loss_weight[i,:,:]*wall_weight[i]
print('loss_weight',loss_weight)
pri=torch.ones([1,class_num,3,3])
print('pri',pri)
label=torch.full([1,class_num,3,3],1.5)
print('label',label)
criterion = torch.nn.BCEWithLogitsLoss()
loss=criterion(pri,label)
print('loss',loss)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=loss_weight)
loss=criterion(pri,label)
print('loss',loss)
s=label.shape
print('s',s[1:])
test_tensor=torch.ones(s[1:])
print('test_tensor',test_tensor)
print('test_tensor.shape',test_tensor.shape)

# wall_weight=Tensor(wall_weight)
# print('wall_weight',wall_weight)

# test bug
# cnt=[[[366,96]],[[366,99]],[[364,101]],[[364,122]],[[363,123]],[[363,146]],[[364,147]],
# [[365,147]],[[366,148]],[[366,150]],[[367,151]],[[367,152]],[[370,149]],[[370,147]],
# [[371,146]],[[370,145]],[[370,143]],[[372,141]],[[373,141]],[[374,142]],[[375,142]],
# [[376,141]],[[376,125]],[[375,124]],[[375,96]]]
# cnt=[[[644,242]],[[644,261]],[[643,262]],[[643,283]],[[646,286]],[[646,436]],[[647,437]],
#  [[647,438]],[[648,438]],[[649,439]],[[652,439]],[[653,438]],[[652,437]],[[652,380]],
#  [[653,379]],[[653,377]],[[652,376]],[[652,286]],[[656,282]],[[657,282]],[[657,251]],
#  [[655,249]],[[655,242]]]
# cnt=[[[100,100]],[[100,700]],[[700,700]],[[700,100]]]
# # cnt=[[[444,336]],[[443,337]],[[431,337]],[[430,338]],[[430,340]],[[431,341]],[[431,343]],[[448,343]],[[448,340]],[[447,340]],[[444,337]]]
# cnt=np.array(cnt)
# print('cnt.shape',cnt.shape)
# rect= cv2.minAreaRect(cnt)
# points=cv2.boxPoints(rect)
# points=np.int0(points) 
# print('cnt.shape',cnt.shape)
# show cnt
# img=np.zeros((400,400),np.uint8)
# cv2.drawContours(img,[cnt],0,255,-1)
# cv2.imshow('img',img)
# cv2.waitKey(0)

# img_test=np.zeros((800,800),np.uint8)
# cnt_test=[[[100,100]],[[100,600]],[[600,600]],[[600,100]]]
# cnt_test=np.array(cnt_test)
# cv2.drawContours(img_test,[cnt_test],0,255,-1)
# cv2.imshow('img_test',img_test)
# print('isin_cntflag(img_test/255,img/255)',isin_cntflag(cnt_test,img//255))

# test cnt seg
# k_test=99999
# k_test=0
# cnts=findMaxMIOU(cnt,k_test)
# print('len(cnts)',len(cnts))
# cnts, min_cnts, miou,cnt_flag=findMaxMIOU(cnt,k_test)
# cv2.imshow('cnt_flag',cnt_flag)

# print('miou',miou)
# print('len(cnts)',len(cnts))
# for cnti in cnts:
#     cv2.drawContours(img,[cnti],0,55,1)
#     print('iou',campIOU(cnti))

# cv2.waitKey(0)

# name='t1.png'
# img=cv2.imread(name)
# img_gray=cv2.imread(name,0)
# # print('img_gray.shape',img_gray.shape)
# contours, _ = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnt=contours[0]
# # rect= cv2.minAreaRect(cnt)
# # points=cv2.boxPoints(rect)
# # points=np.int0(points)[:,np.newaxis,:]
# # print('points',points)
# # img_o=img.copy()
# # cv2.drawContours(img_o,[points],0,[0,0,255],3)
# # cv2.imshow('img_o',img_o)
# # print('contours shape',contours.shape)
# # print('cnt shape',cnt.shape)
# # print(campIOU(cnt))

# # k=[0,1,-1,99999]
# # for ki in k:
# #     imgi=img.copy()
# #     print('ki',ki)
# #     cnts, min_cnts, now_iou,cnt_flag=findMaxMIOU(cnt,ki)
# #     for i in cnts:
# #         cv2.drawContours(imgi,[i],0,[0,0,255],2)
# #         print(campIOU(i))
# #     cv2.imshow('img'+str(ki),imgi)

# # cnts, min_cnts, now_iou,cnt_flag=findMaxMIOU(cnt,99999)
# cnts,min_cnts=findMaxMIOU(cnt,-1)
# # imgi=img.copy()
# print('len(cnts)',len(cnts))
# # i=0
# # cv2.drawContours(imgi,cnts[i],0,[0,0,255],2)
# indeximg=0
# for i in cnts:
#     print('len(i)',len(i))
#     indeximg+=1
#     imgi=img.copy()
#     # print('cnt i',i)
#     # print(campIOU(i))
#     cv2.drawContours(imgi,[i],0,[0,0,255],2)
#     # cv2.drawContours(imgi,[i],0,[random.randint(0,255),random.randint(0,255),random.randint(0,255)],2)
#     cv2.imshow('imgi'+str(indeximg),imgi)

# # print('len(min_cnts)',len(min_cnts))
# # imgj=img.copy()
# # for j in min_cnts:
# #     cv2.drawContours(imgj,[j],0,[0,255,0],1)
# # cv2.imshow('imgj',imgj)

# # cv2.imshow('img_gray',img_gray)

# cv2.waitKey(0)