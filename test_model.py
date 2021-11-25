from unet import UNet, UnetResnet50, hrnet48, Unet_p1
import torch

if __name__ == '__main__':
    # test
    cuda_name = 'cuda' # 'cuda:1'
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    # model=UNet(3,2).cuda(device=device)
    # model= UnetResnet50(3,2).cuda(device=device)
    model= hrnet48(3,2).cuda(device=device)
    img=torch.rand((1,3,256,256)).cuda(device=device)
    mask=model(img)
    print(mask.shape)