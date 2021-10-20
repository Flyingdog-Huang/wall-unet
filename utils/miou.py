import torch
from torch import Tensor
import numpy as np


def IOU(input: Tensor, target: Tensor, epsilon=1e-6):
    assert input.size() == target.size()
    inter = torch.dot(input.reshape(-1), target.reshape(-1))
    sets_sum = torch.sum(input) + torch.sum(target) - inter
    return inter / (sets_sum + epsilon)


def MIOU(input: Tensor, target: Tensor, epsilon=1e-6):
    assert input.size() == target.size()
    iou = 0
    for channel in range(input.shape[1]):
        iou += IOU(input[:, channel, ...], target[:, channel, ...], epsilon)
    return iou / input.shape[1]
