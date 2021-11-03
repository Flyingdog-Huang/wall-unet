import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff

# add
from PIL import Image
import numpy as np
from torch.functional import Tensor
from utils.miou import IOU, MIOU
'''
def miou(mask_pred,mask_true):
    total_miou=0.0
    return total_miou

def recall(mask_pred,mask_true):
    total_recall=0.0
    return total_recall

def map(mask_pred,mask_true):
    total_map=0.0
    return total_map

def warpingError(mask_pred,mask_true):
    total_warp=0.0
    return total_warp

def randError(mask_pred,mask_true):
    total_rand=0.0
    return total_rand
'''


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_onehot_bg = 0
    dice_onehot_nobg = 0
    dice_softmax_bg = 0
    dice_softmax_nobg = 0
    dice_score = 0
    miou = 0

    # acc compute
    num_correct = 0
    num_pixels = 0

    # iterate over the validation set
    for batch in tqdm(dataloader,
                      total=num_val_batches,
                      desc='Validation round',
                      unit='batch',
                      leave=False):
        # print()
        # print('-------------------------------------------------------')
        # print('Evaluation for batch  ')
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        # mask_true = mask_true.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(torch.squeeze(mask_true,dim=1), net.n_classes).permute(0, 3, 1, 2).float()
        # mask_true = mask_true.float()
        # mask_true = F.one_hot(mask_true.argmax(dim=1),
        #                       net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred,
                                         mask_true,
                                         reduce_batch_first=False)
            else:
                mask_pred_softmax = F.softmax(mask_pred, dim=1).float()
                mask_pred_onehot = F.one_hot(mask_pred_softmax.argmax(dim=1),
                                             net.n_classes).permute(
                                                 0, 3, 1, 2).float()

                # save the eva as pic
                # mask=Tensor.cpu(mask_pred[0])
                # backgrand=Image.fromarray(np.uint8(mask[0]*0))
                # class1=Image.fromarray(np.uint8(mask[1]*255))
                # class2=Image.fromarray(np.uint8(mask[2]*255))
                # result_img=Image.merge('RGB',(class2,class1,backgrand))
                # result_img.save('eva.png')

                # print()
                # print('---------------start evaluate-----------------------------')
                # print('mask_pred: ',Tensor.cpu(mask_pred))
                # print('mask_true: ',Tensor.cpu(mask_true))
                # dice_score += multiclass_dice_coeff(mask_pred, mask_true, reduce_batch_first=True)
                # print('----------------------use softmax to compute dice----------------------')

                #ignoring background
                dice_softmax_nobg += multiclass_dice_coeff(
                    mask_pred_softmax[:, 1:, ...],
                    mask_true[:, 1:, ...],
                    reduce_batch_first=True)

                # consider background
                dice_softmax_bg += multiclass_dice_coeff(
                    mask_pred_softmax, mask_true, reduce_batch_first=True)

                # print('----------------------use onehot to compute dice----------------------')

                #  ignoring background
                dice_onehot_nobg += multiclass_dice_coeff(
                    mask_pred_onehot[:, 1:, ...],
                    mask_true[:, 1:, ...],
                    reduce_batch_first=True)

                # consider background
                dice_onehot_bg += multiclass_dice_coeff(
                    mask_pred_onehot, mask_true, reduce_batch_first=True)
                # print('---------------finish evaluate-----------------------------')

                # compute the acc score, ignoring background
                num_correct += (mask_pred_onehot[:, 1:,
                                                 ...] == mask_true[:, 1:,
                                                                   ...]).sum()
                num_pixels += torch.numel(mask_pred[:, 1:, ...])

                # miou no bg
                miou += MIOU(mask_pred_onehot[:,1:,...], mask_true[:,1:,...])

    net.train()
    return miou / num_val_batches, dice_softmax_nobg / num_val_batches, dice_softmax_bg / num_val_batches, dice_onehot_nobg / num_val_batches, dice_onehot_bg / num_val_batches, num_correct / num_pixels
    return dice_score / num_val_batches, dice_score_softmax / num_val_batches, num_correct / num_pixels
