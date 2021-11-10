""" Parts of the Resnet backbone """

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """((conv-3*3 => [BN] => Relu => conv-3*3 => [BN]) + x) => Relu"""

    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.Func=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.Identity=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self,x):
        fx=self.Func(x)
        if self.in_channels != self.out_channels:
            x=self.Identity(x)
        y=fx+x
        return nn.ReLU(inplace=True)(y)


class BottleNeck(nn.Module):
    """((conv-1*1 => [BN] => Relu => conv-3*3 => [BN] => Relu => conv-1*1 => [BN] )+x) => Relu"""

    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.Func=nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=1,padding=0,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,mid_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,out_channels,kernel_size=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.Identity=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        fx=self.Func(x)
        if self.in_channels != self.out_channels:
            x=self.Identity(x)
        y=fx+x
        return nn.ReLU(inplace=True)(y)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(mid_channels),
            # nn.PReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DecoderBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, is_doubleConv=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels//2)
        if not is_doubleConv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,bias=False),
                nn.BatchNorm2d(out_channels),
                # nn.PReLU(inplace=True),
                nn.ReLU(inplace=True),
            )
    
    def forward(self,x1,x2):
        x1=self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels,mid_channels=out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class InConv(nn.Module):
    """conv7*7 => [BN] => ReLu => maxpool"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_conv=nn.Sequential(
            # DoubleConv(in_channels,out_channels,mid_channels=out_channels//2),
            # nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

    def forward(self,x):
        return self.in_conv(x)

class OutConv(nn.Module):
    """conv1*1"""

    def __init__(self,in_channels, out_channels):
        super(OutConv, self).__init__()
        self.out_conv=nn.Conv2d(in_channels,out_channels,kernel_size=1)

    def forward(self, x):
        return self.out_conv(x)

class MakeLayerBottleneck(nn.Module):
    
    def __init__(self,in_channels, out_channels, mid_channels,num_bottlenecks=3, is_maxpool=True):
        super().__init__()
        layers=[]
        if is_maxpool:
            layers.append(nn.MaxPool2d(2))
        layers.append(BottleNeck(in_channels,out_channels,mid_channels))
        for _ in range(1,num_bottlenecks):
            layers.append(BottleNeck(out_channels,out_channels,mid_channels))
        self.make_layer_bottleneck=nn.Sequential(*layers)

    def forward(self,x):
        return self.make_layer_bottleneck(x)