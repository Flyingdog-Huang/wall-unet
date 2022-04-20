import torch
from torch import Tensor
import numpy as np


def IOU(input: Tensor, target: Tensor, epsilon=1e-6):
    assert input.size() == target.size()
    # print('IOU input.size() size:', input.size())
    # print('IOU input.reshape(-1) size:', input.reshape(-1).size())
    # print('IOU target.reshape(-1) size:', target.reshape(-1).size())
    inter = torch.dot(input.reshape(-1), target.reshape(-1))
    # inter = input*target
    sets_sum = torch.sum(input) + torch.sum(target) - inter
    return inter / (sets_sum + epsilon)


def MIOU(input: Tensor, target: Tensor, epsilon=1e-6):
    assert input.size() == target.size()
    miou = 0
    class_iou = []
    for channel in range(input.shape[1]):
        iou = IOU(input[:, channel, ...], target[:, channel, ...], epsilon)
        miou += iou
        class_iou.append(iou)
    return class_iou, miou / input.shape[1]
