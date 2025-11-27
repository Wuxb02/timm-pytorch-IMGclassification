from .mobilenetv2 import mobilenetv2
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .vgg import vgg11, vgg13, vgg16, vgg11_bn, vgg13_bn, vgg16_bn
from .vision_transformer import vit_b_16
from .swin_transformer import swin_transformer_base, swin_transformer_small, swin_transformer_tiny
from .densenet import densenet121, densenet161, densenet169, densenet201
from .xception import xception
from .inceptionResnet import inceptionresnetv2
from .inception import inception_v3_google

get_model_from_name = {
    "mobilenetv2"               : mobilenetv2,
    "resnet18"                  : resnet18,
    "resnet34"                  : resnet34,
    "resnet50"                  : resnet50,
    "resnet101"                 : resnet101,
    "resnet152"                 : resnet152,
    "vgg11"                     : vgg11,
    "vgg13"                     : vgg13,
    "vgg16"                     : vgg16,
    "vgg11_bn"                  : vgg11_bn,
    "vgg13_bn"                  : vgg13_bn,
    "vgg16_bn"                  : vgg16_bn,
    "vit_b_16"                  : vit_b_16,
    "swin_transformer_tiny"     : swin_transformer_tiny,
    "swin_transformer_small"    : swin_transformer_small,
    "swin_transformer_base"     : swin_transformer_base,

    'densenet121'               : densenet121,
    'densenet161'               : densenet161,
    'densenet169'               : densenet169,
    'densenet201'               : densenet201,

    'inceptionresnetv2'         : inceptionresnetv2,
    'xception'                  : xception,
    'inceptionv3'       : inception_v3_google


}