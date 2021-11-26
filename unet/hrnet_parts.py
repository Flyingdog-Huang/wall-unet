""" Parts of the HRNet """

import torch
import torch.nn as nn
import torch.nn.functional as F

# BatchNorm2D 动量参数
BN_MOMENTUM = 0.1

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
    """[(conv3*3 => BN => relu => conv3*3 => BN ) + conv1*1 ] => relu"""

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
            layers.append(nn.Upsample(scale_factor=2))
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

# Stage block
class StageBlock(nn.Module):
    """create stage block"""

    def __init__(self,stage_channels, block):
        super(StageBlock,self).__init__()
        # stage_channels - list[32, 64, 128]
        self.stage_channels=stage_channels
        self.num_layer=len(stage_channels)
        self.block=block
        self.num_block=4

        # func - create layer
        self.stage_layers=self.create_stage_layers()

    def create_stage_layers(self):
        '''create layers '''
        parallel_layers=[]
        for i in range(self.num_layer):
            layers=[]
            for j in range(self.num_block):
                layers.append(self.block(self.stage_channels[i],self.stage_channels[i]))
            layers=nn.Sequential(*layers)
            parallel_layers.append(layers)
        return nn.ModuleList(parallel_layers)

    def forward(self,x):
        outs=[]
        for i in range(len(x)):
            x_i=x[i]
            out_i=self.stage_layers[i](x_i)
            outs.append(out_i)
        return outs

# trans block
class TransBlock(nn.Module):
    """make new branch and fusion feature"""

    def __init__(self,inchannels, outchannels):
        super(StageBlock,self).__init__()
        # in/out channels - [32, 64]/[32, 64, 128]
        self.num_inchannels=len(inchannels)
        self.inchannels=inchannels
        self.num_outchannels=len(outchannels)
        self.outchannels=outchannels

        # fuc - make new branch and fusion feature
        self.trans_layers=self.create_branch_fusion_layers()

    def create_branch_fusion_layers(self):
        '''make new branch and fusion feature'''
        total_trans_layers=[]
        for i in range(self.num_inchannels):
            branch_trans_layers=[]
            for j in range(self.num_outchannels):
                each_trans_layers=[]
                if i<j:
                    down_steps=j-i-1
                    for _ in range(down_steps):
                        each_trans_layers.append(
                            nn.Conv2d(in_channels=self.inchannels[i],out_channels=self.inchannels[i],kernel_size=3,stride=2,padding=1),
                            nn.BatchNorm2d(self.inchannels[i]),
                            nn.ReLU(inplace=True)
                        )
                    each_trans_layers.append(
                        nn.Conv2d(in_channels=self.inchannels[i],out_channels=self.inchannels[i],kernel_size=3,stride=2,padding=1),
                        nn.BatchNorm2d(self.inchannels[i]))

                elif i>j:
                    up_steps=i-j
                    each_trans_layers.append( nn.Upsample(scale_factor=2**up_steps))
                
                each_trans_layers.append(
                    nn.Conv2d(in_channels=self.inchannels[i], out_channels=self.outchannels[j],kernel_size=1))
                
                each_trans_layers=nn.Sequential(*each_trans_layers)
                branch_trans_layers.append(each_trans_layers)
            total_trans_layers.append(nn.ModuleList(branch_trans_layers))
        return nn.ModuleList(total_trans_layers)

    def forward(self,x):
        outs=[]
        for i in range(self.num_inchannels):
            out=[]
            for j in range(self.num_outchannels):
                if i==j:
                    y=x[i]
                else:
                    y=self.trans_layers[i][j](x[i])
                out.append(y)
            if len(outs)==0:
                outs=out
            else:
                for k in range(self.num_outchannels):
                    outs[k]+=out[k]
        return outs

Fusion_model=['keep','fuse','multi']

class FusionBlock2(nn.Module):
    """at last fusion feature"""

    def __init__(self,stage4_channels, model='keep'):
        super(FusionBlock2,self).__init__()

        self.model=model
        self.stage4_channels=stage4_channels

        assert model in Fusion_model,\
            "Please make sure mode({0}) is in ['keep', 'fuse', 'multi'].".format(model)
        
        # 根据模式，构建融合层
        self.fuse_layer=self.create_fuse_layers()

    def create_fuse_layers(self):
        layer=None
        if self.model == 'keep':
            layer=self.create_keep_fuse_layers()
        elif self.model == 'fuse':
            layer=self.create_fuse_fuse_layers()
        else :
            layer=self.create_multi_fuse_layers()
        return layer

    def create_keep_fuse_layers(self):
        self.outchannel = self.stage4_channels[0]
        return PlaceHolder()
    
    def create_fuse_fuse_layers(self):
        '''
        融合各个分辨率，通道保持最大(不同于原论文进行通道扩增)，接着上采样
        '''
        # keep max channel -- 统一目标通道数
        outchannel = self.stage4_channels[3]
        layers=[]

        for i in range(len(self.stage4_channels)):
            inchannel = self.stage4_channels[i]
            layer=[]

            if i!=len(self.stage4_channels)-1:
                layer.append(
                    nn.Conv2d(inchannel,outchannel,1,bias=False)
                )
                layer.append(
                    nn.BatchNorm2d(outchannel, momentum=BN_MOMENTUM)
                )
                layer.append(
                    nn.ReLU(inplace=True)
                )
            
            # 根据当前的index，也对应着离第一个分支的距离(index差)
            # 每一个距离，对应一个上采样，进行大小匹配
            layer.append(
                nn.Upsample(scale_factor=2**i)
            )

            layer=nn.Sequential(*layer)
            layers.append(layer)

        self.outchannel =outchannel 
        return nn.ModuleList(layers)
    
    def create_multi_fuse_layers(self):
        '''c
            多尺度融合，并保持多分辨率输出
        '''
        multi_fuse_layers = []
        # keep max channel -- 统一目标通道数
        outchannel = self.stage4_channels[3]
        layers=[]

        for i in range(len(self.stage4_channels)):
            inchannel = self.stage4_channels[i]
            layer=[]

            if i!=len(self.stage4_channels)-1:
                layer.append(
                    nn.Conv2d(inchannel,outchannel,1,bias=False)
                )
                layer.append(
                    nn.BatchNorm2d(outchannel, momentum=BN_MOMENTUM)
                )
                layer.append(
                    nn.ReLU(inplace=True)
                )
            
            # 根据当前的index，也对应着离第一个分支的距离(index差)
            # 每一个距离，对应一个上采样，进行大小匹配
            layer.append(
                nn.Upsample(scale_factor=2**i)
            )

            layer=nn.Sequential(*layer)
            layers.append(layer)

        # 1st fuse layer
        multi_fuse_layers.append(nn.ModuleList(layers))

        # 2ed layer
        multi_fuse_layers.append(
            nn.Sequential(
                nn.Conv2d(outchannel,outchannel,3,2,1,bias=False),
                nn.BatchNorm2d(outchannel,momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )
        )

        # 3rd layer
        multi_fuse_layers.append(
            nn.Sequential(
                nn.Conv2d(outchannel,outchannel,3,2,1,bias=False),
                nn.BatchNorm2d(outchannel,momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )
        )

        # 4th layer
        multi_fuse_layers.append(
            nn.Sequential(
                nn.Conv2d(outchannel,outchannel,3,2,1,bias=False),
                nn.BatchNorm2d(outchannel,momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )
        )

        self.outchannel =outchannel 
        return nn.ModuleList(multi_fuse_layers)

    def forward(self, inputs):
        x1,x2,x3,x4=inputs
        outs=[]
        if self.model==Fusion_model[0]:
            out=self.fuse_layer(x1)
            outs.append(out)
        elif  self.model==Fusion_model[1]:
            out=self.fuse_layer[0](x1)
            out+=self.fuse_layer[1](x2)
            out+=self.fuse_layer[2](x3)
            out+=self.fuse_layer[3](x4)
            outs.append(out)
        else:
            out=self.fuse_layer[0][0](x1)
            out+=self.fuse_layer[0][1](x2)
            out+=self.fuse_layer[0][2](x3)
            out+=self.fuse_layer[0][3](x4)
            outs.append(out)

            out2=self.fuse_layer[1](out)
            outs.append(out2)

            out3=self.fuse_layer[2](out2)
            outs.append(out3)

            out4=self.fuse_layer[3](out3)
            outs.append(out4)

        return outs

class OutputBlock(nn.Module):
    '''校验完成：完成输出的通道变换，并经过自适应均值池化得到1x1的图像

        params:
            inchannels:  输出层的输入通道
            outchannels: 输出层的变换后的输出通道
    '''

    def __init__(self, inchannels, outchannels):
        super(OutputBlock, self).__init__( )
        self.output=nn.Sequential(
            nn.Conv2d(inchannels,outchannels,1,bias=False),
            nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=1)
        )

    def forward(self, inputs):
        n=len(inputs)
        outs=[]
        for i in range(n):
            out=self.output(inputs[i])
            outs.append(out)
        return outs

class ClassificationBlock(nn.Module):
    '''校验完成：产生预测分类的结果，支持多分辨率预测输出

        params:
            inchannels:  输入大小
            num_classes: 分类数 > 0
    '''

    def __init__(self, inchannels, num_classes):
        super(ClassificationBlock, self).__init__( )
        self.classification=nn.Sequential(
            nn.Flatten(),
            nn.Linear(inchannels,num_classes),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        outs=[]
        for i in range(len(inputs)):
            out=self.classification(inputs[i])
            outs.append(out)
        return outs