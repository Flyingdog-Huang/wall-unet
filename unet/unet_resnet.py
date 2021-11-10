""" Full assembly of the parts to form the complete network """

from .resnet_parts import *

class UnetResnet50(nn.Module):
    def __init__(self,n_channels, n_classes):
        super(UnetResnet50,self).__init__()
        self.n_channels=n_channels
        self.n_classes=n_classes

        self.income=InConv(n_channels,64)
        self.down=Down(64,64)
        
        self.layer1=MakeLayerBottleneck(64,256,64,num_bottlenecks=3)
        self.layer2=MakeLayerBottleneck(256,512,128,num_bottlenecks=4)
        self.layer3=MakeLayerBottleneck(512,1024,256,num_bottlenecks=6)
        self.layer4=MakeLayerBottleneck(1024,2048//2,512,num_bottlenecks=3)

        self.decoder1=DecoderBlock(2048, 1024//2)
        self.decoder2=DecoderBlock(1024,512//2)
        self.decoder3=DecoderBlock(512,128//2)
        self.decoder4=DecoderBlock(128,64)

        self.output=OutConv(64,n_classes)

    def forward(self,x):
        x1=self.income(x)
        x1_down=self.down(x1)
        x2=self.layer1(x1_down)
        x3=self.layer2(x2)
        x4=self.layer3(x3)
        x5=self.layer4(x4)
        x=self.decoder1(x5,x4)
        x=self.decoder2(x,x3)
        x=self.decoder3(x,x2)
        x=self.decoder4(x,x1_down)
        x=self.decoder4(x,x1)
        x=self.output(x)
        return x

