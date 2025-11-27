import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

# 在 model_urls 字典中加入 Xception 的预训练权重链接
model_urls = {
    'xception': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth',
}


class SeparableConv2d(nn.Module):
    """
    深度可分离卷积模块
    它由一个深度卷积 (depthwise) 和一个逐点卷积 (pointwise) 组成。
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False):
        super(SeparableConv2d, self).__init__()
        # 深度卷积：groups=in_planes，每个输入通道都有自己的卷积核
        self.conv_depthwise = nn.Conv2d(in_planes, in_planes, kernel_size, stride, padding, groups=in_planes, bias=bias)
        # 逐点卷积：1x1卷积，用于组合深度卷积的输出
        self.conv_pointwise = nn.Conv2d(in_planes, out_planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv_depthwise(x)
        x = self.conv_pointwise(x)
        return x


class Block(nn.Module):
    """
    Xception 的核心残差块
    """

    def __init__(self, in_planes, out_planes, reps, stride=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        # 残差连接的下采样路径
        if out_planes != in_planes or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        else:
            self.shortcut = None

        self.relu = nn.ReLU(inplace=True)

        layers = []
        # 第一个可分离卷积，可能需要调整通道数
        if grow_first:
            layers.append(self.relu)
            layers.append(SeparableConv2d(in_planes, out_planes, 3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_planes))
            in_planes = out_planes  # 更新通道数给下一个循环

        # 中间的可分离卷积
        for i in range(reps - 1):
            layers.append(self.relu)
            layers.append(SeparableConv2d(in_planes, out_planes, 3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_planes))

        # 最后一个可分离卷积，可能需要执行下采样
        if not grow_first:
            layers.append(self.relu)
            layers.append(SeparableConv2d(in_planes, out_planes, 3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_planes))

        # 如果 stride 不为 1，在最后一个可分离卷积后进行下采样
        if stride != 1:
            layers.append(self.relu)
            layers.append(SeparableConv2d(out_planes, out_planes, 3, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_planes))

        # 是否在块的开始就使用ReLU
        if not start_with_relu:
            layers = layers[1:]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        return out


class Xception(nn.Module):
    """
    Xception 主网络结构
    """

    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()

        # --- Entry flow ---
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # block1: in=64, out=128, stride=2
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        # block2: in=128, out=256, stride=2
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        # block3: in=256, out=728, stride=2
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # --- Middle flow ---
        # 8个重复的 block
        middle_flow_blocks = []
        for i in range(8):
            middle_flow_blocks.append(Block(728, 728, 3, 1, start_with_relu=True, grow_first=True))
        self.middle_flow = nn.Sequential(*middle_flow_blocks)

        # --- Exit flow ---
        self.block4 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=True)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        # --- Classifier ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_backbone(self):
        return [
            self.conv1, self.bn1, self.relu, self.conv2, self.bn2,
            self.block1, self.block2, self.block3, self.middle_flow,
            self.block4, self.conv3, self.bn3, self.relu, self.conv4, self.bn4, self.relu
        ]

    def freeze_backbone(self):
        """冻结主干网络（除分类层外）的参数"""
        for module in self.get_backbone():
            for param in module.parameters():
                param.requires_grad = False

    def Unfreeze_backbone(self):
        """解冻主干网络的参数"""
        for module in self.get_backbone():
            for param in module.parameters():
                param.requires_grad = True

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.middle_flow(x)

        # Exit flow
        x = self.block4(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def xception(pretrained=False, progress=True, num_classes=1000):
    """
    构造 Xception 模型

    Args:
        pretrained (bool): 如果为 True, 返回在 ImageNet 上预训练的模型
        progress (bool): 如果为 True, 显示权重下载进度条
        num_classes (int): 输出分类数
    """
    model = Xception(num_classes=num_classes)

    if pretrained:
        # Cadene Xception 模型的分类数是1000
        state_dict = load_state_dict_from_url(model_urls['xception'], model_dir='./model_data',
                                              progress=progress)

        # 如果期望的分类数与预训练模型(1000)不符，则不加载 fc 层的权重
        if num_classes != 1000:
            if 'fc.weight' in state_dict:
                del state_dict['fc.weight']
            if 'fc.bias' in state_dict:
                del state_dict['fc.bias']

            # 使用 strict=False 加载权重，它会忽略缺失的键
            model.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights loaded, except for the final classifier layer for {num_classes} classes.")
        else:
            model.load_state_dict(state_dict, strict=True)
            print("Pretrained weights loaded successfully for 1000 classes.")

    return model

# # 示例：如何使用
# if __name__ == '__main__':
#     # 创建一个用于20个类别的、使用预训练权重的模型
#     # 注意：Xception 的标准输入尺寸是 299x299
#     model = xception(pretrained=True, num_classes=20)

#     # 打印模型结构
#     # print(model)

#     # 模拟输入
#     dummy_input = torch.randn(2, 3, 299, 299)
#     output = model(dummy_input)
#     print(f"\nOutput shape: {output.shape}") # 应该为 torch.Size([2, 20])