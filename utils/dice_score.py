import torch
from torch import Tensor
import numpy as np


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # reduce_batch_first 有什么用？？？
    assert input.size() == target.size()
    # print()
    # print('input.dim(): ',input.dim())
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        # print()
        # print('(2 * inter + epsilon) ------------- ',(2 * inter + epsilon))
        # print('(sets_sum + epsilon) ------------- ',(sets_sum + epsilon))
        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor,
                          target: Tensor,
                          reduce_batch_first: bool = False,
                          epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    # print()
    # print('********************************multiclass_dice_coeff*******************************************')
    # compute the Dice score, ignoring background：this depends on the way use dice, in code, loss will add BG, but eva will not
    # print('channel',input.shape[1])
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...],
                           reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    # print()
    # print('***************dice loss*******************')
    # print('')
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

if __name__ == '__main__':
    tensor0=torch.zeros(3,3,3,3)
    pre=torch.ones(3,3,3,3)
    T_mask=torch.ones(3,3,3,3)
    print(multiclass_dice_coeff(pre,T_mask))
    print(dice_loss(pre,T_mask))