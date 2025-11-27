import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.hub import load_state_dict_from_url

# 在 model_urls 字典中加入 DenseNet 的预训练权重链接
model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Module):
    """DenseNet的基本单元 (BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3))"""

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        # 瓶颈层: 1x1 卷积
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)

        # 主卷积层: 3x3 卷积
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = float(drop_rate)

    def forward(self, features):
        # 拼接之前的特征
        concated_features = torch.cat(features, 1)
        # 通过瓶颈层
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        # 通过主卷积层
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    """由多个 _DenseLayer 组成的密集块"""

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    """位于两个密集块之间的过渡层"""

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """
    DenseNet 主网络结构
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()

        # 初始卷积层 (Stem)
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # 添加密集块 (Dense Blocks)
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            # 如果不是最后一个密集块，则添加一个过渡层
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # 最后的 BatchNorm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # 全连接分类器
        self.classifier = nn.Linear(num_features, num_classes)

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def freeze_backbone(self):
        # 冻结特征提取部分
        for param in self.features.parameters():
            param.requires_grad = False

    def Unfreeze_backbone(self):
        # 解冻特征提取部分
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def _load_state_dict(model, model_url, progress):
    # 加载预训练权重
    state_dict = load_state_dict_from_url(model_url, progress=progress, model_dir='./model_data')
    model.load_state_dict(state_dict, strict=False)


def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress, num_classes, **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, num_classes=num_classes, **kwargs)
    if pretrained:
        # 加载预训练权重
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress, model_dir='./model_data')

        # 如果期望的分类数与预训练模型(1000)不符，则不加载 fc 层的权重
        if num_classes != 1000:
            if 'classifier.weight' in state_dict:
                del state_dict['classifier.weight']
            if 'classifier.bias' in state_dict:
                del state_dict['classifier.bias']

        # 使用 strict=False 加载权重，它会忽略缺失的键
        model.load_state_dict(state_dict, strict=False)
        if num_classes != 1000:
            print(
                f"Pretrained weights for {arch} loaded, except for the final classifier layer for {num_classes} classes.")
        else:
            print(f"Pretrained weights for {arch} loaded successfully for 1000 classes.")
    return model


def densenet121(pretrained=False, progress=True, num_classes=1000, **kwargs):
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress, num_classes, **kwargs)


def densenet169(pretrained=False, progress=True, num_classes=1000, **kwargs):
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress, num_classes, **kwargs)


def densenet201(pretrained=False, progress=True, num_classes=1000, **kwargs):
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress, num_classes, **kwargs)


def densenet161(pretrained=False, progress=True, num_classes=1000, **kwargs):
    # 注意: densenet161 有不同的 growth_rate 和 num_init_features
    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress, num_classes, **kwargs)
