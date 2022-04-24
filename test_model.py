from unet import UNet, UnetResnet50, hrnet48, Unet_p1, hrnet48_p1, UNet_fp16, UNet_fp4
import torch
import time
from deeplabv3P import DeepLab
from hrnet import seg_hrnet, config

if __name__ == '__main__':
    # test
    cuda_name = 'cuda'  # 'cuda:1'
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    # model = UNet(3, 2).to(device=device).eval()
    # model = DeepLab(backbone='resnet', output_stride=16,
    #                 num_classes=2).to(device=device).eval()
    model = seg_hrnet.get_seg_model(config).to(device=device).eval()
    # model=UNet(3,2).cuda(device=device)
    # model= UnetResnet50(3,2).cuda(device=device)
    # model= hrnet48(3,2).to(device=device).eval()
    # model= Unet_p1(3,2).cuda(device=device)
    # model= hrnet48_p1(3,2).cuda(device=device)
    # model = UNet_fp16(n_channels=3, n_classes=2).cuda(device=device)
    # model = UNet_fp4(n_channels=3, n_classes=2).cuda(device=device)
    model.train()
    k = 4
    time_start = time.perf_counter()
    total_loop = 1
    batch_size = 1
    loop_num = total_loop // batch_size
    torch.cuda.empty_cache()
    img = torch.rand((batch_size, 3, 256*k, 256*k)).to(device=device)
    # img = torch.rand((batch_size, 3, 3508, 2480)).to(device=device)
    print('input size', img.size())
    # mask = None
    for i in range(loop_num):
        # img=torch.rand((1,3,256*k,256*k)).cuda(device=device)
        with torch.no_grad():
            mask = model(img)
            # print(mask.shape)
            print('output size', mask.size())
    # print('output size', mask.size())
    print('cost time:', time.perf_counter() - time_start)
