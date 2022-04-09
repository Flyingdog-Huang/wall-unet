import cv2
import numpy as np
from utils.wall_vector import findMaxMIOU, campIOU, isin_cntflag, wall_vector, imgIOU, is_K_same
import random
import torch
from torch.functional import Tensor
from os import listdir
from os.path import splitext

# path
# vector_path='../vector_test/'
# vector_img_path='../vector_test/img/'
# vector_imgo_path='../vector_test/img_o/'
# vector_mask_path='../vector_test/mask/'
# vector_pre_path='../vector_test/predict/'
# vector_prePost_path='../vector_test/pre_post/'
# vector_vec_path='../vector_test/vector/'
# vector_wallLines_path='../vector_test/wall_lines/'
# vector_allLines_path='../vector_test/all_lines/'

# bug path
vector_path='../vector_test/bug/'
vector_img_path=vector_path+'img/'
vector_wallLines_path=vector_path+'wall_lines/'
vector_pre_path=vector_path+'predict/'
vector_vec_path=vector_path+'vector/'


# get img name
img_name_list=[splitext(file)[0] for file in listdir(vector_img_path) if file.endswith('.png') or file.endswith('.jpg')]
# print('img_name_list',img_name_list)

def post_pre(img_name_list):
    # reduce leaky pix
    for name in img_name_list:
        pre_name=name+'_pre.png'
        pre=cv2.imread(vector_pre_path+pre_name)
        pre_post=pre.copy()
        # close operation 
        pre_post=cv2.cvtColor(pre_post,cv2.COLOR_BGR2GRAY)
        close_kernel=np.ones((3,3),np.uint8)
        pre_post=cv2.morphologyEx(pre_post, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        
        # # fill img
        # h,w=pre.shape[:2]
        # mask=np.zeros((h+2,w+2),np.uint8)

        # write pre post
        cv2.imwrite(vector_prePost_path+pre_name,pre_post)

def wall_line_img_mask_pre(img,mask,pre,wall_lines,color=(0,0,255),thick=1):
    wall_line_img=img.copy()
    wall_line_mask=mask.copy()
    wall_line_pre=pre.copy()
    for wall_line in wall_lines:
        x1,y1,x2,y2=wall_line
        p1=(int(wall_line[0]),int(wall_line[1]))
        p2=(int(wall_line[2]),int(wall_line[3]))
        # print('type(wall_line_img)',type(wall_line_img))
        # print('type(wall_line_pre)',type(wall_line_pre))
        cv2.line(wall_line_img,p1,p2,color,thick)
        cv2.line(wall_line_mask,p1,p2,color,thick)
        cv2.line(wall_line_pre,p1,p2,color,thick)

        # direct line
        detact_length=8
        p3=(int(x1+x2)//2,int(y1+y2)//2)
        dx=0
        dy=detact_length
        if y1!=y2:
            k=(x1-x2)/(y2-y1)
            dx=((detact_length**2)/(1+k**2))**0.5
            dy=dx*k
        x_d=p3[0]-dx
        y_d=p3[1]-dy
        p4=(int(x_d),int(y_d))
        x_d=p3[0]+dx
        y_d=p3[1]+dy
        p5=(int(x_d),int(y_d))
        color_direc=(255,0,0)
        # draw all direc lines
        cv2.line(wall_line_img,p3,p4,color_direc,thick)
        cv2.line(wall_line_mask,p3,p4,color_direc,thick)
        cv2.line(wall_line_pre,p3,p4,color_direc,thick)

        cv2.line(wall_line_img,p3,p5,color_direc,thick)
        cv2.line(wall_line_mask,p3,p5,color_direc,thick)
        cv2.line(wall_line_pre,p3,p5,color_direc,thick)
    return wall_line_img,wall_line_mask, wall_line_pre

def get_img_data(name):
    # img name
    img_name=name+'.png'
    mask_name=name+'_mask.png'
    pre_name=name+'_pre.png'
    # load img data
    img=cv2.imread(vector_img_path+img_name)
    mask=cv2.imread(vector_mask_path+mask_name)
    pre=cv2.imread(vector_pre_path+pre_name)
    return [img_name,img], [mask_name,mask], [pre_name,pre]

def draw_all_lines(name,color=(0,0,255),thick=1):
    # img name
    img_name=name+'.png'
    mask_name=name+'_mask.png'
    pre_name=name+'_pre.png'

    # load img data
    img=cv2.imread(vector_img_path+img_name)
    img_o=cv2.imread(vector_imgo_path+img_name)
    mask=cv2.imread(vector_mask_path+mask_name)
    pre=cv2.imread(vector_pre_path+pre_name)

    # get resize scale
    img_shape=img_o.shape[:2]
    resize_scale=max(img_shape)//500

    # process img
    img_gray=cv2.cvtColor(img_o,cv2.COLOR_BGR2GRAY)

    # Fast LSD
    lsd_dec=cv2.ximgproc.createFastLineDetector()
    lines_fastLSD=lsd_dec.detect(img_gray)

    # draw all lines
    for line in lines_fastLSD:
        x1,y1,x2,y2=line[0]

        x1=x1//resize_scale
        y1=y1//resize_scale
        x2=x2//resize_scale
        y2=y2//resize_scale

        p1=(int(x1),int(y1))
        p2=(int(x2),int(y2))

        # draw all lines
        cv2.line(img,p1,p2,color,thick)
        cv2.line(mask,p1,p2,color,thick)
        cv2.line(pre,p1,p2,color,thick)

        # direct line
        detact_length=8
        p3=(int(x1+x2)//2,int(y1+y2)//2)
        dx=0
        dy=detact_length
        if y1!=y2:
            k=(x1-x2)/(y2-y1)
            dx=((detact_length**2)/(1+k**2))**0.5
            dy=dx*k
        x_d=p3[0]-dx
        y_d=p3[1]-dy
        p4=(int(x_d),int(y_d))
        x_d=p3[0]+dx
        y_d=p3[1]+dy
        p5=(int(x_d),int(y_d))
        color_direc=(255,0,0)
        # draw all direc lines
        cv2.line(img,p3,p4,color_direc,thick)
        cv2.line(mask,p3,p4,color_direc,thick)
        cv2.line(pre,p3,p4,color_direc,thick)
        cv2.line(img,p3,p5,color_direc,thick)
        cv2.line(mask,p3,p5,color_direc,thick)
        cv2.line(pre,p3,p5,color_direc,thick)


    # write all lines
    cv2.imwrite(vector_allLines_path+img_name,img)
    cv2.imwrite(vector_allLines_path+mask_name,mask)
    cv2.imwrite(vector_allLines_path+pre_name,pre)

def vector_img(img_name_list):
    # post vector 
    for name in img_name_list:
        print('name',name)
        # img name
        img_name=name+'.png'
        mask_name=name+'_mask.png'
        pre_name=name+'_pre.png'
        vec_name=name+'_vec.png'

        # load img data
        img=cv2.imread(vector_img_path+img_name)
        img_o=cv2.imread(vector_imgo_path+img_name)
        mask=cv2.imread(vector_mask_path+mask_name)
        pre=cv2.imread(vector_pre_path+pre_name)
        # pre=cv2.imread(vector_prePost_path+pre_name)

        # post vector
        vec,wall_lines=wall_vector(img,img_o,pre)
        cv2.imwrite(vector_vec_path+vec_name,vec)

        # show wall lines
        print('---------------write wall lines img--------------------')
        wall_line_img,wall_line_mask, wall_line_pre=wall_line_img_mask_pre(img,mask,pre,wall_lines)
        cv2.imwrite(vector_wallLines_path+img_name,wall_line_img)
        cv2.imwrite(vector_wallLines_path+mask_name,wall_line_mask)
        cv2.imwrite(vector_wallLines_path+pre_name,wall_line_pre)

    print('---------------finish all--------------------')

def test_dir_iou(img_name_list,dir_path,dir_hz):
    miou=0
    for name in img_name_list:
        print('name',name)
        mask_name=name+'_mask.png'
        dir_name=name+dir_hz
        mask=cv2.imread(vector_mask_path+mask_name)
        dir_img=cv2.imread(dir_path+dir_name)
        iou=imgIOU(mask,dir_img)
        print('IOU',iou)
        miou+=iou
    print('dir_path MIOU:',miou/len(img_name_list))
    print('---------------finish all--------------------')

def test_mask_vector_iou(img_name_list):
    miou=0
    # test imgs iou
    for name in img_name_list:
        print('name',name)
        # img name
        mask_name=name+'_mask.png'
        vec_name=name+'_vec.png'

        # load img data
        mask=cv2.imread(vector_mask_path+mask_name)
        vec=cv2.imread(vector_vec_path+vec_name)

        # iou
        iou=imgIOU(mask,vec)
        print('IOU',iou)
        miou+=iou
    print('vector MIOU:',miou/len(img_name_list))
    print('---------------finish all--------------------')

def test_mask_pre_iou(img_name_list):
    miou=0
    # test imgs iou
    for name in img_name_list:
        print('name',name)
        # img name
        mask_name=name+'_mask.png'
        pre_name=name+'_pre.png'

        # load img data
        mask=cv2.imread(vector_mask_path+mask_name)
        pre=cv2.imread(vector_pre_path+pre_name)

        # iou
        iou=imgIOU(mask,pre)
        print('IOU',iou)
        miou+=iou
    print('pre MIOU:',miou/len(img_name_list))
    print('---------------finish all--------------------')

# draw all img all lines
# for name in img_name_list:
#     draw_all_lines(name)

# post_pre(img_name_list) # post pre
# test_mask_pre_iou(img_name_list) # test pre MIOU
# dir_hz='_pre.png'
# test_dir_iou(img_name_list,vector_prePost_path,dir_hz) # test dir_path MIOU

# vector_img(img_name_list) # genera vector
test_mask_vector_iou(img_name_list) # test vector MIOU

# # test if-for 
# a=10
# b=1
# for i in range(a,b):
#     # if i%3==0 and i%5==0 and i%7==0:
#         print(i)
#         # if i//500>=1:break

# # test is K same func
# k1=1
# k2=-1
# print(is_K_same(k1,k2))

# # test tensor
# class_num=2
# # wall_weight=[1,1]
# wall_weight=[0.67,0.33]
# loss_weight=torch.ones([class_num,3,3])
# # print('loss_weight',loss_weight[0,0,:,:])
# for i in range(class_num):
#     loss_weight[i,:,:]=loss_weight[i,:,:]*wall_weight[i]
# print('loss_weight',loss_weight)
# pri=torch.ones([1,class_num,3,3])
# print('pri',pri)
# label=torch.full([1,class_num,3,3],1.5)
# print('label',label)
# criterion = torch.nn.BCEWithLogitsLoss()
# loss=criterion(pri,label)
# print('loss',loss)
# criterion = torch.nn.BCEWithLogitsLoss(pos_weight=loss_weight)
# loss=criterion(pri,label)
# print('loss',loss)
# s=label.shape
# print('s',s[1:])
# test_tensor=torch.ones(s[1:])
# print('test_tensor',test_tensor)
# print('test_tensor.shape',test_tensor.shape)

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