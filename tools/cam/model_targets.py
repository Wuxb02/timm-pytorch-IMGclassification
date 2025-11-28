"""
模型目标层映射表

为不同模型架构自动选择合适的目标卷积层
"""

import torch.nn as nn


# 模型架构到目标层的映射表
TARGET_LAYER_MAP = {
    # InceptionResNet系列 (timm模型)
    'inception_resnet_v2': 'conv2d_7b',  # timm库的InceptionResNetV2最后卷积层
    'inceptionresnetv2': 'conv2d_7b',

    # ResNet系列
    'resnet18': 'layer4',
    'resnet34': 'layer4',
    'resnet50': 'layer4',
    'resnet101': 'layer4',
    'resnet152': 'layer4',

    # VGG系列
    'vgg11': 'features.30',         # 最后一个ReLU层
    'vgg13': 'features.34',
    'vgg16': 'features.40',
    'vgg11_bn': 'features.38',
    'vgg13_bn': 'features.44',
    'vgg16_bn': 'features.52',

    # DenseNet系列
    'densenet121': 'features.denseblock4',
    'densenet169': 'features.denseblock4',
    'densenet201': 'features.denseblock4',
    'densenet161': 'features.denseblock4',

    # MobileNet系列
    'mobilenetv2': 'features.18',   # 最后的InvertedResidual块
    'mobilenet_v2': 'features.18',

    # EfficientNet系列(timm模型)
    'efficientnet_b0': 'conv_head',
    'efficientnet_b1': 'conv_head',
    'efficientnet_b2': 'conv_head',
    'efficientnet_b3': 'conv_head',
    'efficientnet_b4': 'conv_head',
    'efficientnet_b5': 'conv_head',
    'efficientnet_b6': 'conv_head',
    'efficientnet_b7': 'conv_head',

    # ConvNeXt系列(timm模型)
    'convnext_tiny': 'stages.3',
    'convnext_small': 'stages.3',
    'convnext_base': 'stages.3',
    'convnext_large': 'stages.3',

    # Xception
    'xception': 'conv4',

    # Inception系列
    'inception_v3': 'Mixed_7c',
    'inceptionv3': 'Mixed_7c',
}

# 不支持的模型(无卷积层的Transformer架构)
UNSUPPORTED_MODELS = [
    'vit_b_16',                 # Vision Transformer
    'vit_b_32',
    'vit_l_16',
    'vit_l_32',
    'vit_base_patch16_224',     # timm的ViT
    'vit_base_patch32_224',
    'vit_large_patch16_224',
    'swin_t',                   # Swin Transformer
    'swin_s',
    'swin_b',
    'swin_transformer_tiny',
    'swin_transformer_small',
    'swin_transformer_base',
]


def get_target_layer(model, backbone_name):
    """
    自动获取模型的目标卷积层

    Args:
        model: 加载的PyTorch模型对象
        backbone_name: 模型架构名称(字符串)

    Returns:
        target_layer: nn.Module对象,模型的目标卷积层

    Raises:
        ValueError: 如果模型不支持或未定义映射
    """
    # 处理DataParallel包装的模型
    if hasattr(model, 'module'):
        model = model.module

    # 转换为小写以兼容不同命名风格
    backbone_lower = backbone_name.lower()

    # 检查是否为不支持的模型
    if backbone_lower in [m.lower() for m in UNSUPPORTED_MODELS]:
        raise ValueError(
            f"模型 '{backbone_name}' 不支持GRAD-CAM(Transformer架构无卷积层)。\n"
            f"建议: 使用Attention Map可视化方法。"
        )

    # 从映射表获取目标层名称
    layer_name = None
    for key, value in TARGET_LAYER_MAP.items():
        if key.lower() == backbone_lower:
            layer_name = value
            break

    if layer_name is None:
        # 尝试通用策略1: 查找最后一个卷积层
        print(f"警告: 模型 '{backbone_name}' 未在映射表中,尝试自动检测...")

        last_conv_layer = None
        last_conv_name = None

        # 遍历所有模块,找到最后一个Conv2d层
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # 跳过分类头中的卷积层
                if 'fc' not in name and 'classifier' not in name and 'head' not in name.lower():
                    last_conv_layer = module
                    last_conv_name = name

        if last_conv_layer is not None:
            print(f"  自动检测到目标层: {last_conv_name}")
            layer_name = last_conv_name
        else:
            # 尝试通用策略2: timm模型特定属性
            if hasattr(model, 'get_classifier'):
                # timm模型通常有明确的特征提取器
                if hasattr(model, 'conv_head'):
                    print(f"  使用timm模型的conv_head层")
                    return model.conv_head
                elif hasattr(model, 'stages'):
                    print(f"  使用timm模型的最后stage")
                    return model.stages[-1]
                elif hasattr(model, 'blocks'):
                    print(f"  使用timm模型的最后block")
                    return model.blocks[-1]

            # 未找到映射
            raise ValueError(
                f"模型 '{backbone_name}' 的目标层映射未定义,且无法自动检测。\n"
                f"支持的模型: {', '.join(list(TARGET_LAYER_MAP.keys())[:10])}...\n"
                f"建议:\n"
                f"  1. 运行 'python tools/print_model_structure.py --backbone {backbone_name}' 查看模型结构\n"
                f"  2. 使用 add_custom_mapping('{backbone_name}', 'layer_name') 添加映射\n"
                f"  3. 在TARGET_LAYER_MAP中添加新的映射"
            )

    # 解析嵌套层名(如 'features.denseblock4')
    target = model
    for part in layer_name.split('.'):
        if part.isdigit():
            # 处理数字索引(如features.30)
            target = target[int(part)]
        else:
            # 处理属性名
            if not hasattr(target, part):
                raise ValueError(
                    f"模型 '{backbone_name}' 中未找到层 '{layer_name}'。\n"
                    f"当前尝试访问: {part}"
                )
            target = getattr(target, part)

    # 验证返回的是nn.Module
    if not isinstance(target, nn.Module):
        raise ValueError(
            f"目标层 '{layer_name}' 不是有效的nn.Module对象。\n"
            f"类型: {type(target)}"
        )

    return target


def add_custom_mapping(backbone_name, layer_name):
    """
    添加自定义模型映射(用于扩展支持)

    Args:
        backbone_name: 模型架构名称
        layer_name: 目标层名称(如'layer4'或'features.30')

    Example:
        >>> add_custom_mapping('my_custom_model', 'final_conv')
    """
    TARGET_LAYER_MAP[backbone_name.lower()] = layer_name
    print(f"已添加映射: {backbone_name} -> {layer_name}")
