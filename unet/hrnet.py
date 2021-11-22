""" Full assembly of the parts to form the complete network """

from .hrnet_parts import *



class hrnet48(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(hrnet48, self).__init__()

        self.n_channels=n_channels
        self.n_classes=n_classes

        # keep layer
        self.keeplayer=ResBlock(n_channels,32)

        #  stem stage
        self.stemStage=StemStage(n_channels,64)

        # stage 1
        self.stage1layer1=LayerBlock(64,64)

        # branch 1-2
        self.stage1branch1_2=DownBlock(64, 128)

        # stage 2
        self.stage2layer1=LayerBlock(64,64)
        self.stage2layer2=LayerBlock(128,128)

        # branch 1-3 1-2 2-1 2-3 
        self.stage2branch2_3=DownBlock(128, 256)
        self.stage2branch1_3=DownBlock(64, 256,num=2)
        self.stage2branch1_2=DownBlock(64, 128)
        self.stage2branch2_1=UpBlock(128, 64)

        # fusion add
        self.fusion_add=FusionBlock()

        # stage 3
        self.stage3layer1=LayerBlock(64,64)
        self.stage3layer2=LayerBlock(128,128)
        self.stage3layer3=LayerBlock(256,256)

        # branch 1-4 1-3 1-2 2-1 2-3 2-4 3-1 3-2 3-4
        self.stage3branch1_2=DownBlock(64, 128)
        self.stage3branch1_3=DownBlock(64, 256,num=2)
        self.stage3branch1_4=DownBlock(64, 512,num=3)
        self.stage3branch2_1=UpBlock(128, 64)
        self.stage3branch2_3=DownBlock(128, 256)
        self.stage3branch2_4=DownBlock(128, 512,num=2)
        self.stage3branch3_1=UpBlock(256, 64,num=2)
        self.stage3branch3_2=UpBlock(256, 128)
        self.stage3branch3_4=DownBlock(256, 512)

        # stage 4
        self.stage4layer1=LayerBlock(64,64)
        self.stage4layer2=LayerBlock(128,128)
        self.stage4layer3=LayerBlock(256,256)
        self.stage4layer4=LayerBlock(512,512)

        # trans feature
        self.trans1_2=DownBlock(64, 128)
        self.trans1_3=DownBlock(64, 256,num=2)
        self.trans1_4=DownBlock(64, 512,num=3)
        self.trans2_1=UpBlock(128, 64)
        self.trans2_3=DownBlock(128, 256)
        self.trans2_4=DownBlock(128, 512,num=2)
        self.trans3_1=UpBlock(256, 64,num=2)
        self.trans3_2=UpBlock(256, 128)
        self.trans3_4=DownBlock(256, 512)
        self.trans4_1=UpBlock(512, 64,num=3)
        self.trans4_2=UpBlock(512, 128,num=2)
        self.trans4_3=UpBlock(512, 256)

        # contect trans
        self.conTrans2_1=UpBlock(128, 128)
        self.conTrans3_1=UpBlock(256, 256,num=2)
        self.conTrans4_1=UpBlock(512, 512,num=3)

        # hrfeature up to img size
        self.keep_up=UpBlock(64+128+256+512,64+128+256+512,num=2)

        # fusion connect
        self.fusion_cat=FusionBlock(is_add=False)

        # out conv1*1
        self.out=nn.Conv2d(32+64+128+512+256,n_classes,kernel_size=1)

    def forward(self,x):
        # keep layer
        x_keep=self.keeplayer(x)
        #  stem stage
        x1_stem=self.stemStage(x)
        # stage 1
        x1_s1=self.stage1layer1(x1_stem)
        # make branch layer2
        x2_s2=self.stage1branch1_2(x1_s1)
        # stage 2
        x1_s2=self.stage2layer1(x1_s1)
        x2_s2=self.stage2layer2(x2_s2)
        # make branch layer3
        x3_s3=self.stage2branch1_3(x1_s2)
        # add fusion 1,2-1,2,3
        x1_s3=self.fusion_add(self.stage2branch2_1(x2_s2),x1_s2)

        x2_s3=self.fusion_add(self.stage2branch1_2(x1_s2),x2_s2)

        x3_s3=self.fusion_add(self.stage2branch2_3(x2_s2),x3_s3)
        # stage 3
        x1_s3=self.stage3layer1(x1_s3)
        x2_s3=self.stage3layer2(x2_s3)
        x3_s3=self.stage3layer3(x3_s3)
        # make branch layer4
        x4_s4=self.stage3branch1_4(x1_s3)
        # add  fusion 1,2,3-1,2,3,4
        x1_s4=self.fusion_add(self.stage3branch2_1(x2_s3),x1_s3)
        x1_s4=self.fusion_add(self.stage3branch3_1(x3_s3),x1_s4)

        x2_s4=self.fusion_add(self.stage3branch1_2(x1_s3),x2_s3)
        x2_s4=self.fusion_add(self.stage3branch3_2(x3_s3),x2_s4)

        x3_s4=self.fusion_add(self.stage3branch1_3(x1_s3),x3_s3)
        x3_s4=self.fusion_add(self.stage3branch2_3(x2_s3),x3_s4)

        x4_s4=self.fusion_add(self.stage3branch2_4(x2_s3),x4_s4)
        x4_s4=self.fusion_add(self.stage3branch3_4(x3_s3),x4_s4)
        # stage 4
        x1_s4=self.stage4layer1(x1_s4)
        x2_s4=self.stage4layer2(x2_s4)
        x3_s4=self.stage4layer3(x3_s4)
        x4_s4=self.stage4layer4(x4_s4)
        # add fusion 1234-1234
        x1_fus=self.fusion_add(self.trans2_1(x2_s4),x1_s4)
        x1_fus=self.fusion_add(self.trans3_1(x3_s4),x1_fus)
        x1_fus=self.fusion_add(self.trans4_1(x4_s4),x1_fus)

        x2_fus=self.fusion_add(self.trans1_2(x1_s4),x2_s4)
        x2_fus=self.fusion_add(self.trans3_2(x3_s4),x2_fus)
        x2_fus=self.fusion_add(self.trans4_2(x4_s4),x2_fus)

        x3_fus=self.fusion_add(self.trans1_3(x1_s4),x3_s4)
        x3_fus=self.fusion_add(self.trans2_3(x2_s4),x3_fus)
        x3_fus=self.fusion_add(self.trans4_3(x4_s4),x3_fus)
        
        x4_fus=self.fusion_add(self.trans1_4(x1_s4),x4_s4)
        x4_fus=self.fusion_add(self.trans2_4(x2_s4),x4_fus)
        x4_fus=self.fusion_add(self.trans3_4(x3_s4),x4_fus)
        # connect fusion
        x1_hrnet=self.fusion_cat(self.conTrans2_1(x2_fus),x1_fus)
        x1_hrnet=self.fusion_cat(self.conTrans3_1(x3_fus),x1_hrnet)
        x1_hrnet=self.fusion_cat(self.conTrans4_1(x4_fus),x1_hrnet)
        # connect stem
        x_fus=self.fusion_cat(self.keep_up(x1_hrnet),x_keep)
        # change channel
        y=self.out(x_fus)
        return y
