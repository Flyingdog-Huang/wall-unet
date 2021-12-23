from unet import UNet, UnetResnet50, hrnet48, Unet_p1, hrnet48_p1, UNet_fp16, UNet_fp4
import torch

if __name__ == '__main__':
    # test
    cuda_name = 'cuda' # 'cuda:1'
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    model=UNet(3,2).cuda(device=device)
    # model= UnetResnet50(3,2).cuda(device=device)
    # model= hrnet48(3,2).cuda(device=device)
    # model= Unet_p1(3,2).cuda(device=device)
    # model= hrnet48_p1(3,2).cuda(device=device)
    # model = UNet_fp16(n_channels=3, n_classes=2).cuda(device=device)
    # model = UNet_fp4(n_channels=3, n_classes=2).cuda(device=device)

    k=1
    for i in range(500):
        img=torch.rand((1,3,256*k,256*k)).cuda(device=device)
        mask=model(img)
        print(mask.shape)