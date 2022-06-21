# import argparse
import logging
# import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
# from PIL import Image
from torchvision import transforms
# from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset, CarvanaDataset, make_grid
from unet import UNet, UnetResnet50, hrnet48, Unet_p1, hrnet48_p1, UNet_fp16, UNet_fp4
# from utils.utils import plot_img_and_mask
from tqdm import tqdm
import cv2
# from predict import get_args, predict_img
from utils.miou import IOU, MIOU
from utils.wall_vector import wall_vector
from os import listdir
from os.path import splitext
import time
from torchvision import transforms as T
import os

# # base
# dir_img = '../../data/JD_clean/img/'
# dir_mask = '../../data/JD_clean/mask/'
# dir_pre = '../../data/JD_clean/predict/'
# dir_vec = '../../data/JD_clean/vector/'
# dir_miou = '../../data/JD_clean/'

# pth path
# dir_pth = '../checkpoints/checkpoint_epoch41_priclean_swf_BCE_unet_iou91.pth'
# dir_pth = '../checkpoints/checkpoint_epoch16_priclean_sw_BCE_unet_iou83.pth'
# name_pth = 'sw_unet_93'
name_pth = 'unet_sw1024_rot_onoffaug_miou94'
# name_pth='unet_miou90_rotaoff'
# name_pth='unet_89miou89_rotaon'
dir_pth = '../../checkpoints/'+name_pth+'.pth'
# dir_pth = '../test_checkpoints/'+name_pth+'.pth'

# test bug
# dir_img = '../test/img/'
# dir_pre = '../test/predict/'
# dir_vec = '../test/vector/'
# dir_path='../../../FloorPlan/2-Dataset/JD_rotation/test/'
# dir_bug = '20220611_miou90_rot_tta/'
# dir_path='../../../FloorPlan/2-Dataset/JD_rotation/test/'
# dir_path='../../../../data/floorplan/JD_rotation/test/'
# dir_bug = '20220616_'+name_pth+'/'
dir_path = '../../data/bug/'
dir_bug = '20220617_vectorNum/'

# dir_img = dir_path+'img/'
# dir_mask = dir_path+'mask/'
dir_img = dir_path+dir_bug+'img/'
dir_mask = dir_path+dir_bug+'mask/'

dir_pre = dir_path+dir_bug+'pre/'
dir_tta = dir_path+dir_bug+'tta/'

dir_predict = 'predict/'
dir_alllines = 'all_lines/'
dir_walllines = 'wall_lines/'
dir_Kfeatmap = 'K_feature_map/'
dir_vec = 'vector/'
dir_vec_vis = 'vector_vis/'


# cvcr3d
# dir_img = '../data/test_r3dcvc/img/'
# dir_mask = '../data/test_r3dcvc/mask/'
# dir_pre = '../data/test_r3dcvc/predict/'
# dir_vec = '../data/test_r3dcvc/vector/'
# dir_miou = '../data/test_r3dcvc/'
# dir_pth = '../checkpoints/checkpoint_r3dcvc_unet_iou85.pth'

# pri
# dir_img = '../../../../data/floorplan/private/test/img/'
# dir_mask = '../../../../data/floorplan/private/test/mask2/'
# dir_pre = '../../../../data/floorplan/private/test/predict/'
# # dir_vec = '../data/test/vector/'
# dir_miou = '../../../../data/floorplan/private/test/'
# dir_pth = '../checkpoints_pri_unet/checkpoint_epoch41_pri_unet_iou0.7450826168060303.pth'

# show
# dir_img = '../data/show/img/'
# dir_mask = '../data/show/mask/'
# dir_pre = '../data/show/predict/'
# # dir_vec = '../data/test_vector/vector/'
# dir_miou = '../data/show/'
# dir_pth = '../checkpoints/cosin/checkpoint_epoch32_priLRcos_unet_iou72.pth'


def img_grid(img, strategy):
    if strategy == 'sw':
        logging.info(f'predict strategy : shift window')
        return make_grid(img.shape[:2], window=1024)


def process_img(img, device):
    img = BasicDataset.preprocess(img, 1, is_mask=False, is_transforms=False)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    return img


def predict_img_test(net, device, img, is_strategy: bool = False):
    net.eval()
    if is_strategy is False:
        # process data
        img = process_img(img, device)
        # predict
        with torch.no_grad():
            mask_pred = net(img)
            return mask_pred

    else:
        strategy = 'sw'  # 'crop'
        img_grids = img_grid(img, strategy)
        mask_pred = torch.zeros(
            1, net.n_classes, img.shape[0], img.shape[1]).to(device=device)
        weight_mask = torch.zeros(
            1, 1, img.shape[0], img.shape[1]).to(device=device)
        for points in img_grids:
            x1, x2, y1, y2 = points
            weight_mask[:, :, y1:y2, x1:x2] += 1
            img1 = process_img(img[y1:y2, x1:x2, :], device)
            with torch.no_grad():
                mask_pred[:, :, y1:y2, x1:x2] += net(img1)
        mask_pred = mask_pred/weight_mask
        return mask_pred


def test_miou(net,
              device,
              name,
              image,
              gt,
              is_vector: bool = False):
    # start time
    # start_process = time.perf_counter()

    miou_test = []

    # process data
    mask_true = BasicDataset.preprocess(
        gt, 1, is_mask=True, is_transforms=False)
    logging.info(f'mask_true shape {mask_true.shape}')
    if mask_true.ndim == 2:
        mask_true = mask_true[:, :, np.newaxis]
    mask_true = mask_true.transpose((2, 0, 1))
    mask_true = torch.from_numpy(mask_true)
    mask_true = mask_true.to(device=device, dtype=torch.long)
    mask_true = F.one_hot(torch.squeeze(mask_true, dim=1),
                          net.n_classes).permute(0, 3, 1, 2).float()
    # end_process = time.perf_counter()
    # print('process time',end_process-start_process)

    # predict
    logging.info(f'image shape {image.shape}')
    is_strategy = True
    mask_pred = predict_img_test(net, device, image, is_strategy=is_strategy)
    logging.info(f'mask_pred shape {mask_pred.shape}')
    mask_pred_softmax = F.softmax(mask_pred, dim=1).float()
    mask_pred_onehot = F.one_hot(mask_pred_softmax.argmax(
        dim=1), net.n_classes).permute(0, 3, 1, 2).float()

    # miou - predict and label
    _, miou_pre = MIOU(mask_pred_onehot[:, 1:, ...], mask_true[:, 1:, ...])
    miou_test.append(miou_pre)
    print('miou_pre', miou_pre)

    # save pre
    probs = mask_pred_softmax[0]
    logging.info(f'probs shape {probs.shape}')
    tf = transforms.Compose([transforms.ToPILImage(),
                             transforms.ToTensor()])
    full_mask = tf(probs.cpu()).squeeze()
    full_mask = F.one_hot(full_mask.argmax(
        dim=0), net.n_classes).permute(2, 0, 1).numpy()
    pre_img = np.uint8(full_mask[1]*255)
    img_pre = np.array([pre_img, pre_img, pre_img]).transpose((1, 2, 0))
    name_pre = dir_pre+name+'_pre.png'
    logging.info(f'img_pre shape {img_pre.shape}')
    cv2.imwrite(name_pre, img_pre)

    # end_predict = time.perf_counter()
    # print('predict time',end_predict-end_process)

    # vector
    # print('is_vector',is_vector)
    if is_vector:
        img_vector = wall_vector(image, img_pre)

        # save vector
        name_vector = dir_vec+name+'_vector.png'
        cv2.imwrite(name_vector, img_vector)
        # end_vector = time.perf_counter()
        # print('vector time',end_vector-end_predict)

        # miou - vector and label
        img_vector = np.array(
            [img_vector, img_vector, img_vector]).transpose((1, 2, 0))
        mask_vector = BasicDataset.preprocess(
            img_vector, 1, is_mask=True, is_transforms=False)
        if mask_vector.ndim == 2:
            mask_vector = mask_vector[:, :, np.newaxis]
        mask_vector = mask_vector.transpose((2, 0, 1))
        mask_vector = torch.from_numpy(mask_vector)
        mask_vector = mask_vector.to(device=device, dtype=torch.long)
        mask_vector = F.one_hot(torch.squeeze(
            mask_vector, dim=1), net.n_classes).permute(0, 3, 1, 2).float()
        _, miou_vector = MIOU(mask_vector[:, 1:, ...], mask_true[:, 1:, ...])
        miou_test.append(miou_vector)
        print('miou_vector', miou_vector)
        # end_MIOU = time.perf_counter()
        # print('MIOU time',end_MIOU-end_predict)
    return miou_test


def time_test_aug(net, img, device):
    img_list = []
    # create mask and weight
    img_grids = make_grid(img.shape[:2], window=1024)
    img_mask_pred = torch.zeros(
        1, net.n_classes, img.shape[0], img.shape[1]).to(device=device)
    img_weight_mask = torch.zeros(
        1, 1, img.shape[0], img.shape[1]).to(device=device)
    tta_mask_pred = torch.zeros(
        1, net.n_classes, img.shape[0], img.shape[1]).to(device=device)
    tta_weight_mask = torch.zeros(
        1, 1, img.shape[0], img.shape[1]).to(device=device)
    # predict
    for points in img_grids:
        x1, x2, y1, y2 = points
        img_weight_mask[:, :, y1:y2, x1:x2] += 1
        tta_weight_mask[:, :, y1:y2, x1:x2] += 4
        img_i = img[y1:y2, x1:x2, :]/255

        # np rotation
        img_90 = np.rot90(img_i.copy())
        img_180 = np.rot90(img_i.copy(), 2)
        img_270 = np.rot90(img_i.copy(), 3)

        # np to tensor
        img_i = img_i.transpose((2, 0, 1))
        img_i = torch.from_numpy(img_i).float().to(device=device)
        img_i = img_i.unsqueeze(0)

        img_90 = img_90.copy().transpose((2, 0, 1))
        img_90 = torch.from_numpy(img_90).float().to(device=device)
        img_90 = img_90.unsqueeze(0)

        img_180 = img_180.copy().transpose((2, 0, 1))
        img_180 = torch.from_numpy(img_180).float().to(device=device)
        img_180 = img_180.unsqueeze(0)

        img_270 = img_270.copy().transpose((2, 0, 1))
        img_270 = torch.from_numpy(img_270).float().to(device=device)
        img_270 = img_270.unsqueeze(0)

        with torch.no_grad():
            img_i = net.forward(img_i).to(device=device)
            img_mask_pred[:, :, y1:y2, x1:x2] += img_i
            # tta
            img_90 = net.forward(img_90).to(device=device)
            img_180 = net.forward(img_180).to(device=device)
            img_270 = net.forward(img_270).to(device=device)
            img_90 = transforms.RandomRotation(degrees=(-90, -90))(img_90)
            img_180 = transforms.RandomRotation(degrees=(-180, -180))(img_180)
            img_270 = transforms.RandomRotation(degrees=(-270, -270))(img_270)
            tta_mask_pred[:, :, y1:y2, x1:x2] += img_i
            tta_mask_pred[:, :, y1:y2, x1:x2] += img_90
            tta_mask_pred[:, :, y1:y2, x1:x2] += img_180
            tta_mask_pred[:, :, y1:y2, x1:x2] += img_270

    img_mask_pred = img_mask_pred/img_weight_mask
    tta_mask_pred = tta_mask_pred/tta_weight_mask

    # process pridiction data - (h,w,c)
    probs = F.softmax(img_mask_pred, dim=1)[0]
    tf = T.Compose([T.ToPILImage(),
                    T.Resize((img.shape[0], img.shape[1])),
                    T.ToTensor()])
    full_mask = tf(probs.cpu()).squeeze()
    full_mask = F.one_hot(full_mask.argmax(
        dim=0), net.n_classes).permute(2, 0, 1).numpy()
    result_img = np.uint8(full_mask[1]*255)
    img_pre = np.transpose(
        np.array([result_img, result_img, result_img]), (1, 2, 0))
    img_list.append(img_pre)
    probs = F.softmax(tta_mask_pred, dim=1)[0]
    tf = T.Compose([T.ToPILImage(),
                    T.Resize((img.shape[0], img.shape[1])),
                    T.ToTensor()])
    full_mask = tf(probs.cpu()).squeeze()
    full_mask = F.one_hot(full_mask.argmax(
        dim=0), net.n_classes).permute(2, 0, 1).numpy()
    result_img = np.uint8(full_mask[1]*255)
    tta_pre = np.transpose(
        np.array([result_img, result_img, result_img]), (1, 2, 0))
    img_list.append(tta_pre)
    return img_list


def test(device,
         dir_img,
         dir_mask,
         dir_pth,
         is_vector: bool = False,
         is_TTA: bool = False
         ):
    miou = 0
    net = UNet(n_channels=3, n_classes=2)
    net.to(device=device)
    net.load_state_dict(torch.load(dir_pth, map_location=device))
    # load img and label
    name_list = [splitext(file)[0] for file in listdir(
        dir_img) if file.endswith('.png') or file.endswith('.jpg')]
    num_data = len(name_list)
    for name in name_list:
        img_name = str(dir_img)+name+'.png'
        mask_name = str(dir_mask)+name+'_mask.png'
        img = cv2.imread(img_name)
        if img is None:
            img_name = str(dir_img)+name+'.jpg'
            img = cv2.imread(img_name)
        mask = cv2.imread(mask_name)
        reduce_rate = 2
        img = cv2.resize(img, None, fx=1/reduce_rate, fy=1 /
                         reduce_rate, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, None, fx=1/reduce_rate, fy=1 /
                          reduce_rate, interpolation=cv2.INTER_LINEAR)
        # print('img.shape',img.shape)
        # test IOU
        miou_test = test_miou(net, device, name, img, mask, is_vector)
        miou += miou_test[0]
    miou = miou/num_data
    return miou


def test_bug(net, img, device, is_vector):
    img_list = []
    # predict
    img_grids = make_grid(img.shape[:2], window=1024)
    mask_pred = torch.zeros(
        1, net.n_classes, img.shape[0], img.shape[1]).to(device=device)
    weight_mask = torch.zeros(
        1, 1, img.shape[0], img.shape[1]).to(device=device)
    for points in img_grids:
        x1, x2, y1, y2 = points
        weight_mask[:, :, y1:y2, x1:x2] += 1
        img_i = img[y1:y2, x1:x2, :]/255
        img_i = img_i.transpose((2, 0, 1))
        img_i = torch.from_numpy(img_i).float().to(device=device)
        img_i = img_i.unsqueeze(0)
        with torch.no_grad():
            mask_pred[:, :, y1:y2,
                      x1:x2] += net.forward(img_i).to(device=device)
    mask_pred = mask_pred/weight_mask

    # process pridiction data - (h,w,c)
    probs = F.softmax(mask_pred, dim=1)[0]
    tf = T.Compose([T.ToPILImage(),
                    T.Resize((img.shape[0], img.shape[1])),
                    T.ToTensor()])
    full_mask = tf(probs.cpu()).squeeze()
    full_mask = F.one_hot(full_mask.argmax(
        dim=0), net.n_classes).permute(2, 0, 1).numpy()
    result_img = np.uint8(full_mask[1]*255)
    img_pre = np.transpose(
        np.array([result_img, result_img, result_img]), (1, 2, 0))
    img_list.append(img_pre)

    if is_vector:
        #  verctorizaton
        img_all, img_wall, img_kmap, img_vec = wall_vector(img, img_pre)
        img_list.append(img_all)
        img_list.append(img_wall)
        img_list.append(img_kmap)
        img_list.append(img_vec)

    return img_list


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # initlization net
    net = UNet(n_channels=3, n_classes=2)

    # args = get_args()
    cuda_name = 'cuda'  # 'cuda:1'
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # print('device',device)

    # logging.info(f'Loading model {args.model}')
    logging.info(f'Loading model {dir_pth}')
    logging.info(f'Using device {device}')
    net.to(device=device)
    # net.load_state_dict(torch.load(args.model, map_location=device))
    net.load_state_dict(torch.load(dir_pth, map_location=device))
    logging.info('Model loaded!')

    name_list = [splitext(file)[0] for file in listdir(
        dir_img) if file.endswith('.png') or file.endswith('.jpg')]
    num_data = len(name_list)
    logging.info(f'Data number {num_data}')

    is_vector = True
    # is_vector = False

    is_TTA = True
    # is_TTA = False

    # make dir
    dir_pre_predict = dir_pre+dir_predict
    if not os.path.exists(dir_pre_predict):
        os.makedirs(dir_pre_predict)
    if is_vector:
        dir_pre_alllines = dir_pre+dir_alllines
        if not os.path.exists(dir_pre_alllines):
            os.makedirs(dir_pre_alllines)
        dir_pre_walllines = dir_pre+dir_walllines
        if not os.path.exists(dir_pre_walllines):
            os.makedirs(dir_pre_walllines)
        dir_pre_Kfeatmap = dir_pre+dir_Kfeatmap
        if not os.path.exists(dir_pre_Kfeatmap):
            os.makedirs(dir_pre_Kfeatmap)
        dir_pre_vec = dir_pre+dir_vec
        if not os.path.exists(dir_pre_vec):
            os.makedirs(dir_pre_vec)
        dir_pre_vec_vis = dir_pre+dir_vec_vis
        if not os.path.exists(dir_pre_vec_vis):
            os.makedirs(dir_pre_vec_vis)
    # print(os.path.exists(dir_img))
    if is_TTA:
        dir_tta_predict = dir_tta+dir_predict
        if not os.path.exists(dir_tta_predict):
            os.makedirs(dir_tta_predict)
        if is_vector:
            dir_tta_alllines = dir_tta+dir_alllines
            if not os.path.exists(dir_tta_alllines):
                os.makedirs(dir_tta_alllines)
            dir_tta_walllines = dir_tta+dir_walllines
            if not os.path.exists(dir_tta_walllines):
                os.makedirs(dir_tta_walllines)
            dir_tta_Kfeatmap = dir_tta+dir_Kfeatmap
            if not os.path.exists(dir_tta_Kfeatmap):
                os.makedirs(dir_tta_Kfeatmap)
            dir_tta_vec = dir_tta+dir_vec
            if not os.path.exists(dir_tta_vec):
                os.makedirs(dir_tta_vec)
            dir_tta_vec_vis = dir_tta+dir_vec_vis
            if not os.path.exists(dir_tta_vec_vis):
                os.makedirs(dir_tta_vec_vis)

    # initialize MIOU
    miou = []
    pre_pre_miou = []
    pre_vec_miou = []
    tta_pre_miou = []
    tta_vec_miou = []

    # print('compute MIOU')
    # print('generate results')
    for name in tqdm(name_list, total=num_data, desc='Test round', unit='pic', leave=False):
        # load img and label
        # print('name', name)
        img_name = str(dir_img)+name+'.png'
        img = cv2.imread(img_name)
        if img is None:
            img_name = str(dir_img)+name+'.jpg'
            img = cv2.imread(img_name)
        # print('img',img)
        # print('img.shape',img.shape)

        # name img
        pre_predict_name = str(dir_pre_predict)+name+'_pre.png'
        tta_predict_name = str(dir_tta_predict)+name+'_pre.png'
        pre_vector_name = str(dir_pre_vec)+name+'_vec.png'
        tta_vector_name = str(dir_tta_vec)+name+'_vec.png'

        # test

        # 生成图像
        if is_TTA:
            pre, tta_pre = time_test_aug(net, img, device)
            cv2.imwrite(pre_predict_name, pre)
            cv2.imwrite(tta_predict_name, tta_pre)
            print('finish tta, save pre and tta pre')
            if is_vector:
                print('start tta vector')
                img_all, img_wall, img_kmap, img_vec_vis, img_vec = wall_vector(
                    img, tta_pre)
                alllines_name = str(dir_tta_alllines)+name+'_all.png'
                walllines_name = str(dir_tta_walllines)+name+'_wall.png'
                Kfeatmap_name = str(dir_tta_Kfeatmap)+name+'_Kmap.png'
                vector_vis_name = str(dir_tta_vec_vis)+name+'_vec_vis.png'
                # img_all, img_wall, img_kmap, img_vec = img_list[1:]
                cv2.imwrite(alllines_name, img_all)
                cv2.imwrite(walllines_name, img_wall)
                cv2.imwrite(Kfeatmap_name, img_kmap)
                cv2.imwrite(vector_vis_name, img_vec_vis)
                cv2.imwrite(tta_vector_name, img_vec)
                print('finish tta vector, save mid-imgs')
        # # test bug
        # img_list = test_bug(net, img, device, is_vector)
        # img_pre = img_list[0]
        # cv2.imwrite(predict_name, img_pre)
        if is_vector:
            print('start vector')
            img_all, img_wall, img_kmap, img_vec_vis, img_vec = wall_vector(
                img, pre)
            alllines_name = str(dir_pre_alllines)+name+'_all.png'
            walllines_name = str(dir_pre_walllines)+name+'_wall.png'
            Kfeatmap_name = str(dir_pre_Kfeatmap)+name+'_Kmap.png'
            vector_vis_name = str(dir_pre_vec_vis)+name+'_vec_vis.png'
            # img_all, img_wall, img_kmap, img_vec = img_list[1:]
            cv2.imwrite(alllines_name, img_all)
            cv2.imwrite(walllines_name, img_wall)
            cv2.imwrite(Kfeatmap_name, img_kmap)
            cv2.imwrite(vector_vis_name, img_vec_vis)
            cv2.imwrite(pre_vector_name, img_vec)
            print('finish vector, save mid-imgs')

    #     # compute MIOU
    #     # load img
    #     mask_name = str(dir_mask)+name+'_mask.png'
    #     mask = cv2.imread(mask_name)
    #     pre_pre=cv2.imread(pre_predict_name)
    #     pre_vec=cv2.imread(pre_vector_name)
    #     if is_TTA:
    #         tta_pre=cv2.imread(tta_predict_name)
    #         tta_vec=cv2.imread(tta_vector_name)
    #     no_test_list=[]
    #     if mask is None or pre_pre is None or pre_vec is None or tta_pre is None or tta_vec is None:
    #         no_test_list.append(name)
    #         print('there is no predict img: ',name)
    #         continue

    #     mask=cv2.resize(mask,(pre_vec.shape[1],pre_vec.shape[0]),interpolation=cv2.INTER_NEAREST)
    #     pre_pre=cv2.resize(pre_pre,(pre_vec.shape[1],pre_vec.shape[0]),interpolation=cv2.INTER_NEAREST)
    #     tta_pre=cv2.resize(tta_pre,(pre_vec.shape[1],pre_vec.shape[0]),interpolation=cv2.INTER_NEAREST)

    #     # compute MIOU
    #     e_s=0.0000001
    #     mask=mask[:,:,0]/255
    #     pre_pre=pre_pre[:,:,0]/255
    #     pre_vec=pre_vec[:,:,0]/255
    #     tta_pre=tta_pre[:,:,0]/255
    #     tta_vec=tta_vec[:,:,0]/255

    #     miou_pre_pre=np.sum(pre_pre*mask)
    #     miou_pre_pre=miou_pre_pre/(np.sum(mask)+np.sum(pre_pre)-miou_pre_pre+e_s)
    #     pre_pre_miou.append(miou_pre_pre)

    #     miou_pre_vec=np.sum(pre_vec*mask)
    #     miou_pre_vec=miou_pre_vec/(np.sum(mask)+np.sum(pre_vec)-miou_pre_vec+e_s)
    #     pre_vec_miou.append(miou_pre_vec)

    #     miou_tta_pre=np.sum(tta_pre*mask)
    #     miou_tta_pre=miou_tta_pre/(np.sum(mask)+np.sum(tta_pre)-miou_tta_pre+e_s)
    #     tta_pre_miou.append(miou_tta_pre)

    #     miou_tta_vec=np.sum(tta_vec*mask)
    #     miou_tta_vec=miou_tta_vec/(np.sum(mask)+np.sum(tta_vec)-miou_tta_vec+e_s)
    #     tta_vec_miou.append(miou_tta_vec)

    #     # save miou
    #     miou_dict = {}
    #     miou_dict['name'] = name
    #     # miou_dict['shape'] = [img.shape[0], img.shape[1]]
    #     miou_dict['pre_pre miou'] = miou_pre_pre
    #     miou_dict['pre_vec miou'] = miou_pre_vec
    #     miou_dict['tta_pre miou'] = miou_tta_pre
    #     miou_dict['tta_vec miou'] = miou_tta_vec
    #     miou.append(miou_dict)

    # # show MIOU
    # miou_pre_pre=np.sum(np.array(pre_pre_miou))/(len(pre_pre_miou))
    # print('pre_pre_miou',miou_pre_pre )
    # miou_pre_vec=np.sum(np.array(pre_vec_miou))/(len(pre_vec_miou))
    # print('pre_vec_miou', miou_pre_vec)
    # miou_tta_pre=np.sum(np.array(tta_pre_miou))/(len(tta_pre_miou))
    # print('tta_pre_miou', miou_tta_pre)
    # miou_tta_vec=np.sum(np.array(tta_vec_miou))/(len(tta_vec_miou))
    # print('tta_vec_miou', miou_tta_vec)

    # # save result as txt
    # # # txt_name=dir_miou+str(int(float(miou_pre*100)))+'MIOU'+str(int(float(miou_vec*100)))+'_improv'+str(int(float(miou_improv*100)))+'.txt'
    # txt_name = dir_path+dir_bug+'pp'+str(int(miou_pre_pre*100))+'pv'+str(int(miou_pre_vec*100))+'tp'+str(int(miou_tta_pre*100))+'tv'+str(int(miou_tta_vec*100))+'.txt'
    # # if is_vector:
    # #     # MIOU change after vector
    # #     miou_improv = 0
    # #     miou_vec = miou_vec/num_data
    # #     miou_improv = miou_vec-miou_pre
    # #     print('miou improve: ', float(miou_improv))
    # #     # rename txt_name
    # #     txt_name = dir_miou+str(int(float(miou_pre*100))) + \
    # #         '-'+str(int(float(miou_vec*100)))+'MIOU'+'.txt'
    # with open(txt_name, 'w') as file:
    #     [file.write(str(it)+'\n') for it in miou]
    #     file.close()

    print('finish test')

    # test dataset
    # miou = []
    # miou_pre = 0
    # miou_vec = 0
    # num_test = 0
    # for name in tqdm(name_list, total=num_data, desc='Test round', unit='pic', leave=False):
    #     # load img and label
    #     print('name', name)
    #     img_name = str(dir_img)+name+'.png'
    #     mask_name = str(dir_mask)+name+'_mask.png'
    #     img = cv2.imread(img_name)
    #     if img is None:
    #         img_name = str(dir_img)+name+'.jpg'
    #         img = cv2.imread(img_name)
    #     # print('img',img)
    #     mask = cv2.imread(mask_name)
    #     # print('img.shape',img.shape)

    #     # resize img shape
    #     # reduce_rate=max(img.shape[0],img.shape[1])//1000+1
    #     # # print('reduce_rate',reduce_rate)
    #     # img_s=cv2.resize(img,None,fx=1/reduce_rate,fy=1/reduce_rate,interpolation=cv2.INTER_LINEAR)
    #     # mask_s=cv2.resize(mask,None,fx=1/reduce_rate,fy=1/reduce_rate,interpolation=cv2.INTER_LINEAR)
    #     # # print('img.shape',img.shape)
    #     # train_size=2
    #     # img_t=cv2.resize(img,None,fx=1/train_size,fy=1/train_size,interpolation=cv2.INTER_LINEAR)
    #     # mask_t=cv2.resize(mask,None,fx=1/train_size,fy=1/train_size,interpolation=cv2.INTER_LINEAR)

    #     # 统计用时
    #     start_time = time.perf_counter()
    #     # test IOU
    #     # miou_test=test_miou(net,device,name,img,mask,is_vector)
    #     # print('img.shape is {} , MIOU is {}'.format(img.shape,miou_test))
    #     # miou_test_s=test_miou(net,device,name,img_s,mask_s,is_vector)
    #     # print('img.shape is {} , MIOU is {}'.format(img_s.shape,miou_test_s))
    #     # print('img.shape is {} '.format(img_t.shape))
    #     # if max(img_t.shape)>2000:
    #     #     print('img size is too large!')
    #     #     continue
    #     num_test += 1
    #     # logging.info(f'mask shape {mask.shape}')
    #     miou_test_t = test_miou(net, device, name, img, mask, is_vector)
    #     print('MIOU is {}'.format(miou_test_t))

    #     # show MIOU on time
    #     miou_pre = miou_pre+miou_test_t[0]
    #     print('miou_pre', miou_pre/num_test)
    #     if is_vector:
    #         miou_vec = miou_vec+miou_test_t[1]
    #         print('miou_vec', miou_vec/num_test)

    #     # 统计用时
    #     end_time = time.perf_counter()
    #     # print('test time',end_time-start_time)

    #     # save msg
    #     miou_dict = {}
    #     miou_dict['name'] = name
    #     miou_dict['shape'] = [img.shape[0], img.shape[1]]
    #     miou_dict['pre miou'] = miou_test_t[0]
    #     miou_dict['vector miou'] = miou_test_t[1] if len(
    #         miou_test_t) == 2 else None
    #     miou_dict['cost time'] = end_time-start_time
    #     miou.append(miou_dict)
    #     # miou_pre+=miou_test_t[0]
    #     # if is_vector:
    #     #     miou_vec += miou_test_t[1]

    # # compute pre MIOU
    # miou_pre = miou_pre/num_data

    # # save result as txt
    # # txt_name=dir_miou+str(int(float(miou_pre*100)))+'MIOU'+str(int(float(miou_vec*100)))+'_improv'+str(int(float(miou_improv*100)))+'.txt'
    # txt_name = dir_miou+str(int(float(miou_pre*100)))+'MIOU'+'.txt'
    # if is_vector:
    #     # MIOU change after vector
    #     miou_improv = 0
    #     miou_vec = miou_vec/num_data
    #     miou_improv = miou_vec-miou_pre
    #     print('miou improve: ', float(miou_improv))
    #     # rename txt_name
    #     txt_name = dir_miou+str(int(float(miou_pre*100))) + \
    #         '-'+str(int(float(miou_vec*100)))+'MIOU'+'.txt'
    # with open(txt_name, 'w') as file:
    #     [file.write(str(it)+'\n') for it in miou]
    #     file.close()

    # # show MIOU
    # print('miou_pre', miou_pre)
    # if is_vector:
    #     print('miou_vec', miou_vec)
