import torch.nn.functional as F
import torch
import torch.nn as nn

img=torch.tensor([[1,-1],[1,-1]]).float()
label=torch.tensor([[0,0],[1,1]]).float()
criterion = nn.MSELoss()
# criterion = nn.BCEWithLogitsLoss()
loss = criterion(img, label)
print('loss',loss)