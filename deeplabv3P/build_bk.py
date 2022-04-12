from .backbone.resnet import ResNet101  # , xception


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return ResNet101(output_stride, BatchNorm)
        # return resnet.ResNet101(output_stride, BatchNorm)
    # elif backbone == 'xception':
    #     return xception.AlignedXception(output_stride, BatchNorm)
    else:
        raise NotImplementedError
