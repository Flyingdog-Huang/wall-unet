import torch
import torch.nn as nn

img = torch.rand((1, 3, 3508, 2480))
print('input size', img.size())
aap11 = nn.AdaptiveAvgPool2d((1, 1))
aap1 = nn.AdaptiveAvgPool2d(1)
out11 = aap11(img)
out1 = aap1(img)
print('1*1 size', out11.size())
print('1 size', out1.size())
