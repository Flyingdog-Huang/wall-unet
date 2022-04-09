# **wall-unet**



## **Introduction**


**function:** segment wall from floor plan 

**base model:** UNet 

**backbone:** basic/ResNet 

**location of pipeline:** training

**DL framework:** pytorch

**language:** python3.6

***

## Code structure


- **unet:** parts , blocks and structures of model
  - **resnet_parts.py:** parts of resnet
  - **unet_model.py:** structures of basic unet
  - **unet_parts.py:** parts of unet
  - **unet_resnet.py:** structures of resnet backbone
- **utils:** data processing , index calculation and other tools
  - **data_augmentation.py:** basic and mosaic data augmentation
  - **data_loading.py:** load data set for model
  - **dice_score.py:** dice calculation for model training and validation
  - **miou.py:** MIOU calculation for model validation
  - **resolveSVG.py:** parse svg label to mask image
  - **utils.py:** some tools like plot img and mask
- **evaluate.py:** evaluate model
- **predict.py:** predict mask
- **test_model.py:** test model
- **train.py:** train model
***


## Quick start 


```console
> python train.py --amp
```
***


## Description


This model is UNet, that is widely used in medical image segmentation, and the features of medical ima and floor plan are similar, and also according to related papers.
***


## Training


```console
> python train.py 

usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
      -h, --help                  show this help message and exit
      
      --epochs E, -e E            Number of epochs
      
      --batch-size B, -b B        Batch size
      
      --learning-rate LR, -l LR   Learning rate
      
      --load LOAD, -f LOAD        Load model from a .pth file
      
      --scale SCALE, -s SCALE     Downscaling factor of the images
      
      --validation VAL, -v VAL    Percent of the data that is used as validation (0-100)        
      --amp                       Use mixed precision
```

the paras of training model will be saved in "checkpoints", and that filedir is out of the project's filedir.
more details about paras will be found in codes.
The training progress can be visualized in real-time using [Weights & Biases](https://wandb.ai/site).
***


## Prediction


```console
 > python predict.py

 usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] 
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

 Predict masks from input images

 optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Filenames of input images
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of output images
  --viz, -v             Visualize the images as they are processed
  --no-save, -n         Do not save the output masks
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel white
  --scale SCALE, -s SCALE
                        Scale factor for the input images
```

You can specify which model file to use with --model MODEL.pth.
***


## Data


now there are only public data set, the number of all is 336, and the class is wall only, and the public data set is used for pre-train.
***


## Links


- Unet: 
  - [paper](https://arxiv.org/abs/1505.04597)
  - [baseline](https://github.com/milesial/Pytorch-UNet/tree/db72295019a2114f4c84940d9aaf1232b2a23352)

- Resnet:
  - [paper](https://arxiv.org/abs/1512.03385)
  - [baseline](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
