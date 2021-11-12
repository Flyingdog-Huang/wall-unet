from unet import UNet, UnetResnet50
import torch

if __name__ == '__main__':
    # test
    cuda_name = 'cuda'# 'cuda:1'
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    # model=UNet(3,2).cuda()
    model= UnetResnet50(3,2).cuda(device=device)
    img=torch.rand((2,3,1024,1024)).cuda(device=device)
    mask=model(img)