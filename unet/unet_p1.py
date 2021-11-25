""" Full assembly of the parts to form the complete network """

from .unet_parts import *

class Unet_p1(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Unet_p1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.factor=2

        self.inc = ResBlock(n_channels, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)
        self.down3 = DownConv(256, 512)
        self.down4 = Down(512, 1024 // self.factor)
        self.up1 = UpRes(1024, 512 // self.factor)
        self.up2 = UpRes(512, 256 // self.factor)
        self.up3 = UpRes(256, 128 // self.factor)
        self.up4 = UpRes(128, 64)
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