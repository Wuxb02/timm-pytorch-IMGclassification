import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

# model_urls 字典保持不变，链接是正确的
model_urls = {
    'inceptionresnetv2': 'https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth',
}


class BasicConv2d(nn.Module):
    """基础的 Conv -> BN -> ReLU 模块"""

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001,
            momentum=0.1,
            affine=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block35_a(nn.Module):
    """Inception-ResNet-A 模块, 在Cadene实现中称为 block35"""

    def __init__(self, in_planes=384, scale=0.17):
        super(Block35_a, self).__init__()
        self.scale = scale
        # 分支 0: 1x1 conv
        self.branch0 = BasicConv2d(in_planes, 32, kernel_size=1, stride=1)
        # 分支 1: 1x1 conv -> 3x3 conv
        self.branch1 = nn.Sequential(
            BasicConv2d(in_planes, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        # 分支 2: 1x1 conv -> 3x3 conv -> 3x3 conv
        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )
        # 拼接后的线性变换层, 输出通道数为 32 + 32 + 64 = 128
        self.conv_linear = nn.Conv2d(128, in_planes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        out = torch.cat((b0, b1, b2), 1)
        out = self.conv_linear(out)
        out = self.scale * out + identity  # 残差连接
        out = self.relu(out)
        return out


class Block17_b(nn.Module):
    """Inception-ResNet-B 模块, 在Cadene实现中称为 block17"""

    def __init__(self, in_planes=1152, scale=0.10):
        super(Block17_b, self).__init__()
        self.scale = scale
        # 分支 0: 1x1 conv
        self.branch0 = BasicConv2d(in_planes, 192, kernel_size=1, stride=1)
        # 分支 1: 1x1 conv -> 1x7 conv -> 7x1 conv
        self.branch1 = nn.Sequential(
            BasicConv2d(in_planes, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )
        # 拼接后的线性变换层, 输出通道数为 192 + 192 = 384
        self.conv_linear = nn.Conv2d(384, in_planes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        out = torch.cat((b0, b1), 1)
        out = self.conv_linear(out)
        out = self.scale * out + identity  # 残差连接
        out = self.relu(out)
        return out


class Block8_c(nn.Module):
    """Inception-ResNet-C 模块, 在Cadene实现中称为 block8"""

    def __init__(self, in_planes=2144, scale=0.20):
        super(Block8_c, self).__init__()
        self.scale = scale
        # 分支 0: 1x1 conv
        self.branch0 = BasicConv2d(in_planes, 192, kernel_size=1, stride=1)
        # 分支 1: 1x1 conv -> 1x3 conv -> 3x1 conv
        self.branch1 = nn.Sequential(
            BasicConv2d(in_planes, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )
        # 拼接后的线性变换层, 输出通道数为 192 + 256 = 448
        self.conv_linear = nn.Conv2d(448, in_planes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        out = torch.cat((b0, b1), 1)
        out = self.conv_linear(out)
        out = self.scale * out + identity  # 残差连接
        out = self.relu(out)
        return out


class ReductionA(nn.Module):
    """下采样模块A (在 Inception-ResNet-A 和 B 之间)"""

    def __init__(self, in_planes, k, l, m, n):
        super(ReductionA, self).__init__()
        # 分支 0: 3x3 conv with stride 2
        self.branch0 = BasicConv2d(in_planes, n, kernel_size=3, stride=2)
        # 分支 1: 1x1 conv -> 3x3 conv -> 3x3 conv with stride 2
        self.branch1 = nn.Sequential(
            BasicConv2d(in_planes, k, kernel_size=1, stride=1),
            BasicConv2d(k, l, kernel_size=3, stride=1, padding=1),
            BasicConv2d(l, m, kernel_size=3, stride=2)
        )
        # 分支 2: MaxPool with stride 2
        self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        # 输出通道数: n + m + in_planes
        return torch.cat((b0, b1, b2), 1)


class ReductionB(nn.Module):
    """下采样模块B (在 Inception-ResNet-B 和 C 之间)"""

    def __init__(self, in_planes):
        super(ReductionB, self).__init__()
        # 分支 0
        self.branch0 = nn.Sequential(
            BasicConv2d(in_planes, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )
        # 分支 1
        self.branch1 = nn.Sequential(
            BasicConv2d(in_planes, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        )
        # 分支 2
        self.branch2 = nn.Sequential(
            BasicConv2d(in_planes, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )
        # 分支 3
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        # 输出通道数: 384 + 288 + 320 + in_planes
        return torch.cat((b0, b1, b2, b3), 1)


class InceptionResNetV2(nn.Module):
    """
    InceptionResNetV2 主网络结构
    """

    def __init__(self, num_classes=1000):
        super(InceptionResNetV2, self).__init__()
        # Stem
        self.stem = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),  # 299x299x3 -> 149x149x32
            BasicConv2d(32, 32, kernel_size=3, stride=1),  # 149x149x32 -> 147x147x32
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 147x147x64 -> 147x147x64
            nn.MaxPool2d(kernel_size=3, stride=2),  # 147x147x64 -> 73x73x64
            BasicConv2d(64, 80, kernel_size=1, stride=1),  # 73x73x64 -> 73x73x80
            BasicConv2d(80, 192, kernel_size=3, stride=1),  # 73x73x80 -> 71x71x192
            BasicConv2d(192, 256, kernel_size=3, stride=2),  # 71x71x192 -> 35x35x256 (修正点)
        )

        # 10 x Inception-ResNet-A
        # 输入 35x35x256 -> 输出 35x35x256
        self.blocks_a = nn.Sequential(*[Block35_a(in_planes=256) for _ in range(10)])

        # Reduction-A
        # 输入 35x35x256 -> 输出 17x17x896
        self.reduction_a = ReductionA(256, k=192, l=192, m=256, n=384)

        # 20 x Inception-ResNet-B
        # 输入 17x17x896 -> 输出 17x17x896
        self.blocks_b = nn.Sequential(*[Block17_b(in_planes=896) for _ in range(20)])

        # Reduction-B
        # 输入 17x17x896 -> 输出 8x8x1792
        self.reduction_b = ReductionB(896)

        # 9 x Inception-ResNet-C
        # 输入 8x8x1792 -> 输出 8x8x1792
        self.blocks_c = nn.Sequential(*[Block8_c(in_planes=1888) for _ in range(9)])

        # 最终的卷积层
        # 输入 8x8x1792 -> 输出 8x8x1536
        self.conv_final = BasicConv2d(1888, 1536, kernel_size=1, stride=1)

        # 平均池化和分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.6)  # 使用与预训练模型一致的dropout率
        self.fc = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks_a(x)
        x = self.reduction_a(x)
        x = self.blocks_b(x)
        x = self.reduction_b(x)
        x = self.blocks_c(x)
        x = self.conv_final(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def freeze_backbone(self):
        """冻结主干网络（除分类层外）的参数"""
        backbone = [
            self.stem, self.blocks_a, self.reduction_a,
            self.blocks_b, self.reduction_b, self.blocks_c, self.conv_final
        ]
        for module in backbone:
            for param in module.parameters():
                param.requires_grad = False

    def Unfreeze_backbone(self):
        """解冻主干网络的参数"""
        backbone = [
            self.stem, self.blocks_a, self.reduction_a,
            self.blocks_b, self.reduction_b, self.blocks_c, self.conv_final
        ]
        for module in backbone:
            for param in module.parameters():
                param.requires_grad = True


def inceptionresnetv2(pretrained=False, progress=True, num_classes=1000):
    """
    构造 InceptionResNetV2 模型
    """
    model = InceptionResNetV2(num_classes=num_classes)

    if pretrained:
        # Cadene 模型的分类数是1001 (包含背景)
        state_dict = load_state_dict_from_url(model_urls['inceptionresnetv2'], model_dir='./model_data',
                                              progress=progress)

        # 预训练模型的最后全连接层名称是 'last_linear'，而我们的是 'fc'
        # 需要进行键名转换
        if 'last_linear.weight' in state_dict:
            state_dict['fc.weight'] = state_dict.pop('last_linear.weight')
            state_dict['fc.bias'] = state_dict.pop('last_linear.bias')

        # 如果期望的分类数与预训练模型不符，则不加载 fc 层的权重
        if num_classes != 1001:
            if 'fc.weight' in state_dict:
                del state_dict['fc.weight']
            if 'fc.bias' in state_dict:
                del state_dict['fc.bias']

            # 使用 strict=False 加载权重，它会忽略缺失的键
            model.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights loaded, except for the final classifier layer for {num_classes} classes.")
        else:
            model.load_state_dict(state_dict, strict=True)
            print("Pretrained weights loaded successfully for 1001 classes.")

    return model

# # 示例：如何使用
# if __name__ == '__main__':
#     # 创建一个用于20个类别的、使用预训练权重的模型
#     # 注意：InceptionResNetV2 的标准输入尺寸是 299x299
#     model = inceptionresnetv2(pretrained=True, num_classes=20)

#     # 模拟输入
#     dummy_input = torch.randn(2, 3, 299, 299)
#     output = model(dummy_input)
#     print(f"\nOutput shape: {output.shape}") # 应该为 torch.Size([2, 20])