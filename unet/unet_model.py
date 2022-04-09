""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet_fp4(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_fp4, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)
        self.cen = nn.Sequential(
            ResBlock(256,256),
            ResBlock(256,512),
            ResBlock(512,1024),
            ResBlock(1024,512),
            ResBlock(512,256),
            ResBlock(256,256)
        ) 
        factor = 2 
        self.up1 = Up(256+128, 256 // factor)
        self.up2 = Up(128+64, 128 // factor)
        self.outc = nn.Sequential(
            DoubleConv(64, 32),
            OutConv(32, n_classes)
        )

    def forward(self, x):
        x1 = self.inc(x) # 64
        # print(x1.shape)
        x2 = self.down1(x1) # 128
        # print(x2.shape)
        x3 = self.down2(x2) # 256
        # print(x3.shape)
        x4 = self.cen(x3) # 256
        # print(x4.shape)
        x = self.up1(x4, x2) # 256+128->128
        # print(x.shape)
        x = self.up2(x, x1) # 128+64->64
        # print(x.shape)
        logits = self.outc(x)
        # print(logits.shape)
        return logits

class UNet_fp16(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_fp16, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)

        self.down3 = DownConv(256, 512)
        self.down4 = DownConv(512, 512)

        self.cen = nn.Sequential(
            ResBlock(512,1024),
            ResBlock(1024,1024),
            ResBlock(1024,1024),
            ResBlock(1024,512),
            ResBlock(512,512),
            ResBlock(512,512)
        )

        self.up1 = Up(512+512, 256 )
        self.up2 = Up(256+256, 128 )

        self.up3 = Up(128+128, 64 )
        self.up4 = Up(64+64, 64 )

        self.outc = nn.Sequential(
            DoubleConv(64, 32),
            OutConv(32, n_classes)
        )

    def forward(self, x):
        x_in = self.inc(x) # 64
        # print(x1.shape)

        x_d1 = self.down1(x_in) # 128
        # print(x2.shape)
        x_d2 = self.down2(x_d1) # 256
        # print(x3.shape)
        x_d3=self.down3(x_d2) # 512
        x_d4=self.down4(x_d3)  # 512

        x = self.cen(x_d4) # 512
        # print(x4.shape)

        x = self.up1(x, x_d3) # 512+512->256
        # print(x.shape)
        x = self.up2(x, x_d2) # 256+256->128
        # print(x.shape)
        x = self.up3(x, x_d1) # 128+128->64
        x = self.up4(x, x_in) # 64+64->64
        logits = self.outc(x)
        # print(logits.shape)
        return logits