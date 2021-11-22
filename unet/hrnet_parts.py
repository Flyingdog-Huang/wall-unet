""" Parts of the HRNet """

import torch
import torch.nn as nn
import torch.nn.functional as F

# BatchNorm2D 动量参数
BN_MOMENTUM = 0.2

# 占位符
class PlaceHolder(nn.Module):

    def __init__(self):
        super(PlaceHolder,self).__init__()

    def forward(self,inputs):
        return inputs

# 通用3*3conv
class Conv3x3Block(nn.Module):
    """3*3=>BN=>relu"""

    def __init__(self,inchannels, outchannels, stride=1,padding=1):
        super(Conv3x3Block, self).__init__()
        self.conv=nn.Conv2d(inchannels, outchannels, 
                            kernel_size= 3,
                            stride=stride,
                            padding=padding,
                            bias=False)
        self.bn=nn.BatchNorm2d(outchannels,momentum=BN_MOMENTUM)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.relu(x)
        return x

# stem stage
class StemStage(nn.Module):
    """(conv3*3[1/2]=> BN => relu) * 2"""

    def __init__(self,inchannels, outchannels):
        super(StemStage,self).__init__()
        self.conv1=Conv3x3Block(inchannels, outchannels, stride=2,padding=1)
        self.conv2=Conv3x3Block(outchannels, outchannels, stride=2,padding=1)
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        return x

# renet block
class ResBlock(nn.Module):
    """(conv3*3 => BN => relu)*2+1*1"""

    def __init__(self,inchannels, outchannels):
        super(ResBlock,self).__init__()
        self.Func=nn.Sequential(
            Conv3x3Block(inchannels, outchannels),
            nn.Conv2d(outchannels, outchannels, kernel_size= 3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(outchannels,momentum=BN_MOMENTUM)
        )
        self.Identity=nn.Sequential(
            nn.Conv2d(inchannels,outchannels,kernel_size=1,bias=False),
            nn.BatchNorm2d(outchannels,momentum=BN_MOMENTUM)
        )

    def forward(self,x):
        fx=self.Func(x)
        x=self.Identity(x)
        y=fx+x
        return nn.ReLU(inplace=True)(y)

# Down block
class DownBlock(nn.Module):
    """(conv3*3[1/2] => BN =>relu)*(n-1) => conv3*3[1/2] => BN """

    def __init__(self,inchannels, outchannels, num=1):
        super(DownBlock,self).__init__()
        self.num=num
        layers=[]
        if num==1:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(inchannels, outchannels, kernel_size= 3, stride=2, padding=1,bias=False),
                    nn.BatchNorm2d(outchannels,momentum=BN_MOMENTUM)
                    )
            )
        if num>1:
            layers.append(
                Conv3x3Block(inchannels, outchannels, stride=2, padding=1)
            )
            if num>2:
                for _ in range(2,num):
                    layers.append(Conv3x3Block(outchannels, outchannels, stride=2, padding=1))
            layers.append(
                nn.Sequential(
                    nn.Conv2d(outchannels, outchannels, kernel_size= 3, stride=2, padding=1,bias=False),
                    nn.BatchNorm2d(outchannels,momentum=BN_MOMENTUM)
                    )
            )
        self.down=nn.Sequential(*layers)
    
    def forward(self,x):
        x=self.down(x)
        return x

# fusion feature
class FusionBlock(nn.Module):
    """add or connect"""

    def __init__(self,is_add=True):
        super(FusionBlock,self).__init__()
        self.is_add=is_add
    
    def forward(self,x1,x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x=x1+x2 if self.is_add else torch.cat([x2, x1], dim=1)
        return x

# Up block
class UpBlock(nn.Module):
    """(upsample=>1*1conv=>BN + x)=>relu"""

    def __init__(self,inchannels, outchannels, num=1):
        super(UpBlock,self).__init__()
        self.num=num
        layers=[]
        for _ in range(num):
            layers.append(nn.Upsample(scale_factor=2, mode='nearest', align_corners=True))
        layers.append(nn.Conv2d(inchannels,outchannels,kernel_size=1))
        self.up =nn.Sequential(*layers)
    
    def forward(self,x):
        x=self.up(x)
        return x



# layer block
class LayerBlock(nn.Module):
    """(conv3*3 => BN => relu)*2+1*1"""

    def __init__(self,inchannels, outchannels):
        super(LayerBlock,self).__init__()
        self.Layer=nn.Sequential(
            ResBlock(inchannels, outchannels),
            ResBlock(outchannels, outchannels),
            ResBlock(outchannels, outchannels),
            ResBlock(outchannels, outchannels)
        )

    def forward(self,x):
        x=self.Layer(x)
        return x
