from unet import UNet, UnetResnet50
import torch

if __name__ == '__main__':
    # test
    # model=UNet(3,2).cuda()
    model= UnetResnet50(3,2).cuda()
    img=torch.rand((2,3,512,512)).cuda()
    mask=model(img)