import argparse
import logging
import sys
from pathlib import Path
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, dataset, random_split
from tqdm import tqdm
from utils.data_loading import BasicDataset, CarvanaDataset, PimgDataset, GridDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet, UnetResnet50, hrnet48, Unet_p1, hrnet48_p1, UNet_fp16, UNet_fp4
from utils.miou import IOU, MIOU
import numpy as np
from torchvision.transforms import transforms
from utils.data_augmentation import tensor_img_mask_aug
from deeplabv3P import DeepLab

# path
# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')

# dir_img = Path('../data/CVC-FP/')
# dir_mask = Path('../data/CVC-FP/')

# selflabel data set
# dir_img = Path('../../../../data/floorplan/selflabel/imgs/')
# dir_mask = Path('../../../../data/floorplan/selflabel/masks/')

# dir_img = Path('../../../../data/floorplan/pimg/imgs/')
# dir_pimg = Path('../../../../data/floorplan/pimg/JPEG-DOP1/')
# dir_mask = Path('../../../../data/floorplan/pimg/masks/')

# dir_img = Path('../../../../data/floorplan/CVC-FP/')
# dir_mask = Path('../../../../data/floorplan/CVC-FP/masks/')

# r3dcvc data set
# dir_img = Path('../../../../data/floorplan/r3d_cvc/imgs/')
# dir_mask = Path('../../../../data/floorplan/r3d_cvc/masks/')

# car data set
# dir_img = Path('../../../../data/floorplan/car/imgs/train/')
# dir_mask = Path('../../../../data/floorplan/car/imgs/masks/')

# JD private data set
# dir_img = Path('../../../../data/floorplan/private/train/img/')
# dir_mask = Path('../../../../data/floorplan/private/train/mask2/')
# dir_img = Path('../../../../data/floorplan/JD_clean/img/')
# dir_mask = Path('../../../../data/floorplan/JD_clean/mask/')
dir_img = Path('../../../../data/floorplan/after_resize/img/')
dir_mask = Path('../../../../data/floorplan/after_resize/mask/')

# merge data set
# dir_img = Path('../../../../data/floorplan/r3d_cvc_pri/imgs/')
# dir_mask = Path('../../../../data/floorplan/r3d_cvc_pri/masks/')

# dir_img = Path('../data/test/img/')
# dir_mask = Path('../data/test/mask/')

dir_checkpoint = Path('../checkpoints/')

# training strategy
# aug on loading 
is_tf = False  
# is_tf = True 

# aug on line
is_aug = False
# is_aug = True  

# shift window
is_sw = False
# is_sw = True  

# multi-GPU
# is_gpus = False
is_gpus = True  


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    # 1. Create dataset
    # print()
    # print('-------------------------------------------------------')
    # print('Create dataset')

    # load data
    if is_sw:
        # shift window crop
        dataset = GridDataset(dir_img, dir_mask, img_scale,is_tf) 
    else:
        try:
            dataset = CarvanaDataset(dir_img, dir_mask, img_scale,is_tf)
        except (AssertionError, RuntimeError):
            dataset = BasicDataset(dir_img, dir_mask, img_scale,is_tf)
    # dataset=PimgDataset(dir_img,dir_pimg,dir_mask,img_scale)

    # 2. Split into train / validation partitions
    # print()
    # print('-------------------------------------------------------')
    # print('Split into train / validation partitions')
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    # print()
    # print('-------------------------------------------------------')
    # print('Create data loaders')
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs,
             batch_size=batch_size,
             learning_rate=learning_rate,
             val_percent=val_percent,
             save_checkpoint=save_checkpoint,
             img_scale=img_scale,
             amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=[0.9,0.99], weight_decay=1e-8) 
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)  # momentum=0.99
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=100)  # goal: maximize Dice score, 2
    # scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=90,gamma=0.5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # loss func
    # wall_weight=[0.33,0.67]
    # wall_weight=[0.67,0.33]
    # wall_weight=[0.1,0.9]
    CE_criterion = nn.CrossEntropyLoss()
    BCE_criterion = nn.BCEWithLogitsLoss()
    MSE_criterion=nn.MSELoss()
    # criterion = BCE_criterion
    global_step = 0

    # 5. Begin training
    # print()
    # print('-------------------------------------------------------')
    # print('Begin training')
    for epoch in range(epochs):
        # print()
        # print('-------------------------------------------------------')
        # print('for epoch ')
        net.train()
        epoch_loss = 0
        miou_epoch=0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}',unit='img') as pbar:
            for batch in train_loader:
                # print()
                # print('-------------------------------------------------------')
                # print('for batch ')
                global_step += 1
                images = batch['image']
                true_masks = batch['mask']

                # print()
                # print('***********************************************')
                # print('true_masks: ',true_masks)
                # print('true_masks.shape: ',true_masks.shape)
                # # print('images.shape: ',images.shape)
                # print()

                # warm up LR
                # start_LR=0.000001
                # end_LR=0.0001
                # warmup_step=50
                # if global_step < warmup_step:
                #     now_LR=start_LR+global_step*(end_LR-start_LR)/warmup_step
                #     print()
                #     print('-------------------------------------------------------')
                #     print('global_step: ',global_step)
                #     print('now_LR: ',now_LR)
                #     optimizer = optim.RMSprop(net.parameters(), lr=now_LR, weight_decay=1e-8,  momentum=0.9)

                # adjust LR
                # reduce_step=55
                # if global_step>reduce_step:
                #     # 直接衰减
                #     # now_LR=0.000001
                #     # optimizer = optim.RMSprop(net.parameters(), lr=now_LR,  weight_decay=1e-8, momentum=0.9)
                #     # 余弦退火调整学习率
                #     cos_a=np.cos((global_step-reduce_step)*np.pi/300)
                #     now_LR=end_LR*cos_a
                #     print()
                #     print('-------------------------------------------------------')
                #     print('global_step: ',global_step)
                #     print('cos_a: ',cos_a)
                #     print('now_LR: ',now_LR)
                #     optimizer = optim.RMSprop(net.parameters(), lr=now_LR,  weight_decay=1e-8, momentum=0.9)
                # if global_step > 5000:
                #     optimizer = optim.RMSprop(net.parameters(),
                #                               lr=0.000001,
                #                               weight_decay=1e-8,
                #                               momentum=0.9)
                    # if global_step>2000:
                    #     optimizer = optim.RMSprop(net.parameters(), lr=0.0000001, weight_decay=1e-8, momentum=0.9)

                input_channels=net.module.n_channels if is_gpus else net.n_channels
                output_classes=net.module.n_classes if is_gpus else net.n_classes
                assert images.shape[1] == input_channels, \
                    f'Network has been defined with {input_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                # augmentation online
                if is_aug:
                    images, true_masks = tensor_img_mask_aug(images, true_masks)
                
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long) # tensor(n,c,w,h)
                # print()
                # print('***********************************************')
                # print()
                # print('true_masks.shape: ',true_masks.shape)
                # print('images.shape: ',images.shape)

                # true_masks = true_masks.to(device=device, dtype=torch.float32)

                # print()
                # print('***********************************************')
                # print()
                # print('true_masks.shape: ',true_masks.shape)

                # true_masks_onehot = F.one_hot(true_masks.argmax(dim=1),
                #                        net.n_classes).permute(0, 3, 1,
                #                                               2).float()
                true_masks_onehot = F.one_hot(torch.squeeze(true_masks,dim=1),output_classes).permute(0, 3, 1, 2).float().to(device=device)

                # print()
                # print('***********************************************')
                # print('true_masks_onehot.shape: ',true_masks_onehot.shape)
                # print()

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)

                    # adjust true_masks size

                    # loss=crossEntropy+dice
                    # loss = criterion(masks_pred, true_masks) \
                    #        + dice_loss(F.softmax(masks_pred, dim=1).float(),
                    #                    F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                    #                    multiclass=True)

                    # sigmoid
                    BCE_loss = BCE_criterion(masks_pred, true_masks_onehot)

                    # print()
                    # print('-------------------------------------------')
                    # print('masks_pred.shape: ',masks_pred.shape)
                    # print('true_masks.shape: ',true_masks.shape)

                    true_masks_CE=torch.squeeze(true_masks,dim=1)
                    true_masks_CE=true_masks_CE.to(device=device, dtype=torch.int64)
                    # print('true_masks_CE.shape: ',true_masks_CE.shape)
                    CE_loss=CE_criterion(masks_pred,true_masks_CE)

                    # test dice
                    # print()
                    # print('true_masks.shape: ',true_masks.shape)
                    # print('true_masks.numpy(): ',Tensor.cpu(true_masks).numpy())

                    # print()
                    # print('masks_pred.shape: ',masks_pred.shape)
                    # print('masks_pred.numpy(): ',Tensor.cpu(masks_pred).detach().numpy())
                    masks_pred_softmax = F.softmax(masks_pred, dim=1).float()
                    MSE_loss = MSE_criterion(masks_pred_softmax, true_masks_onehot)
                    # print()
                    # print('masks_pred_softmax.shape: ',masks_pred_softmax.shape)
                    # print('masks_pred_softmax.numpy(): ',Tensor.cpu(masks_pred_softmax).detach().numpy())
                    masks_pred_max = masks_pred_softmax.argmax(dim=1)
                    # print()
                    # print('masks_pred_max.shape: ',masks_pred_max.shape)
                    # print('masks_pred_max.numpy(): ',Tensor.cpu(masks_pred_max).detach().numpy())
                    mask_pred_onehot = F.one_hot(masks_pred_max, output_classes).permute(0, 3, 1, 2).float()
                    # MIOU no bg
                    class_iou_train, miou_train = MIOU(mask_pred_onehot[:,1:,...], true_masks_onehot[:,1:,...])
                    num_class=0
                    for iou in class_iou_train:
                        num_class+=1
                        experiment.log({'Train NO.{}class IOU'.format(num_class):iou})


                    # print('mask_pred_onehot.shape: ',mask_pred_onehot.shape)
                    # print('mask_pred_onehot.numpy(): ',Tensor.cpu(mask_pred_onehot).detach().numpy())
                    diceloss = dice_loss(masks_pred_softmax, true_masks_onehot.float(), multiclass=True)
                    # print()

                    # print()
                    # print('diceloss.shape: ',diceloss.shape)
                    # print('diceloss.numpy(): ',Tensor.cpu(diceloss).detach().numpy())
                    # diceloss=diceloss.requires_grad_()
                    BCE_dice_loss = BCE_loss + diceloss
                    CE_dice_loss = CE_loss + diceloss
                    MSE_dice_loss = MSE_loss + diceloss

                    # change loss func here
                    # loss = CE_dice_loss
                    # loss = MSE_dice_loss
                    loss = BCE_dice_loss
                    # print(loss)

                    # mini-batch
                    # if global_step%10==1:
                    #     optimizer.zero_grad(set_to_none=True)

                    # grad_scaler.scale(loss).backward()  
                    # print(loss)

                    # loss=dice
                    # loss = dice_loss(F.softmax(masks_pred, dim=1).float(),
                    #                    F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                    #                    multiclass=True)
                    # print('---------------------------------------------------',Tensor.cpu(diceloss).numpy())

                # if global_step%10==0:
                #     grad_scaler.step(optimizer)
                #     grad_scaler.update()

                # SGD
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward() 
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                epoch_loss += loss.item() 
                experiment.log({
                    # 'BCE_loss': BCE_loss.item(),
                    # 'dice_loss': diceloss.item(),
                    'Train MIOU': miou_train,
                    'Train CE_dice_loss': CE_dice_loss.item(),
                    'Train BCE_dice_loss': BCE_dice_loss.item(),
                    'Train MSE_dice_loss': MSE_dice_loss.item(),
                    # 'step': global_step,
                    # 'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})  

                # Evaluation round    
                super_para = 10
                if global_step % (n_train // (super_para * batch_size)) == 0:
                    # print()            
                    # print('-------------------------------------------------------')
                    # print('Evaluation round ')
                    histograms = {}
                    parameters=net.module.named_parameters()  if is_gpus else  net.named_parameters()
                    for tag, value in parameters:
                        tag = tag.replace('/', '.')
                        histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                    # val_score,val_score_soft,acc = evaluate(net, val_loader, device)
                    class_iou_eva , miou_eva, dice_softmax_nobg, dice_softmax_bg, dice_onehot_nobg, dice_onehot_bg, acc = evaluate(net, val_loader, device)

                    val_score = dice_onehot_nobg
                    miou_epoch=miou_eva
                    
                    # LR adjust
                    # scheduler.step(val_score) 
                    # scheduler.step()

                    num_class=0
                    for iou in class_iou_eva:
                        num_class+=1
                        experiment.log({'Validation NO.{}class IOU'.format(num_class):iou})

                    logging.info('Validation Dice score: {}'.format(val_score))
                    logging.info('Validation MIOU score: {}'.format(miou_eva))
                    experiment.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        # 'Dice softmax nobg': dice_softmax_nobg,
                        # 'Dice softmax bg': dice_softmax_bg,
                        'Validation Dice onehot nobg': dice_onehot_nobg,
                        'Validation MIOU': miou_eva,
                        # 'Dice onehot bg': dice_onehot_bg,
                        # 'PA': acc,
                        'images':  wandb.Image(images[0].cpu()),
                        'masks': {
                            'true': wandb.Image(true_masks[0].float().cpu()),
                            'pred': wandb.Image(torch.softmax(masks_pred, dim=1)[0].float().cpu()),
                        },
                        # 'step': global_step,
                        # 'epoch': epoch,
                        **histograms
                    })
    
        # just each epoch
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            model_name=str(dir_checkpoint /
                    'checkpoint_epoch{}_jd_clean_resize_BCE_dlv3+_iou{}.pth'.format(epoch, miou_epoch))
            if is_gpus:
                torch.save(net.module.state_dict(),model_name)
            else:
                torch.save(net.state_dict(),model_name) # single GPU
            # logging.info(f'Checkpoint {epoch + 1} saved!')
            logging.info(f'Checkpoint {epoch} saved!')
    
    # just save the last one
    if save_checkpoint:
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        final_model_name=str(dir_checkpoint /
                'checkpoint_epoch{}_jd_clean_resize_BCE_dlv3+_iou{}.pth'.format(epochs, miou_epoch))
        if is_gpus:
            torch.save(net.module.state_dict(),final_model_name)
        else:
            torch.save(net.state_dict(),final_model_name)# single GPU
        # logging.info(f'Checkpoint {epoch + 1} saved!')
        logging.info(f'Checkpoint {epochs} saved!')


def get_args():
    parser = argparse.ArgumentParser(
        description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int,
                        default=100, 
                        help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int,
                        default=1,
                        help='Batch size')
    parser.add_argument('--learning-rate', '-l',  metavar='LR',  type=float,
                        default=0.00001,
                        help='Learning rate',  dest='lr')
    parser.add_argument('--load', '-f',  type=str,
                        default=False,
                        help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float,
                        default=1, # 0.5
                        help='Downscaling factor of the images')
    parser.add_argument('--validation','-v',  dest='val', type=float,
                        default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true',
                        default=False,
                        help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.getLogger().setLevel(logging.INFO)
    
    logging.info('aug on loading : {}'.format(is_tf)) 
    logging.info('aug on line : {}'.format(is_aug)) 
    logging.info('is use shift window : {}'.format(is_sw)) 
    logging.info('is use multi-GPUs : {}'.format(is_gpus)) 

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    cuda_name = 'cuda' # 'cuda:1'
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    logging.info(f'Main device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel

    # net = UNet(n_channels=3, n_classes=2, bilinear=True)
    net = DeepLab(backbone='resnet', output_stride=16,num_classes=2)
    # net = UnetResnet50(n_channels=3, n_classes=2)
    # net =hrnet48(n_channels=3, n_classes=2)
    # net =Unet_p1(n_channels=3, n_classes=2)
    # net = hrnet48_p1(n_channels=3, n_classes=2)
    # net = UNet_fp4(n_channels=3, n_classes=2)
    # net = UNet_fp16(n_channels=3, n_classes=3)

    logging.info(
        f'Network:\n'
        f'\t{net.n_channels} input channels\n'
        f'\t{net.n_classes} output channels (classes)\n')
        # f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling'

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    if is_gpus:
        net=torch.nn.DataParallel(net,device_ids=[0,1])
    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        if is_gpus:
            torch.save(net.module.state_dict(), 'INTERRUPTED.pth')
        else:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
