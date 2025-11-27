import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

# 在 model_urls 字典中加入 Inception V3 的预训练权重链接
model_urls = {
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class InceptionA(nn.Module):
    # 对应InceptionV3论文中的 Figure 5
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_features, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_2(self.branch5x5_1(x))
        branch3x3dbl = self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x)))
        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    # 对应InceptionV3论文中的 Figure 6, Grid Reduction
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x)))
        branch_pool = self.branch_pool(x)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    # 对应InceptionV3论文中的 Figure 7, 使用 1x7 和 7x1 分解卷积
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_3(self.branch7x7_2(self.branch7x7_1(x)))

        branch7x7dbl = self.branch7x7dbl_5(
            self.branch7x7dbl_4(self.branch7x7dbl_3(self.branch7x7dbl_2(self.branch7x7dbl_1(x)))))

        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    # 对应InceptionV3论文中的 Figure 6, Grid Reduction
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_2(self.branch3x3_1(x))
        branch7x7x3 = self.branch7x7x3_4(self.branch7x7x3_3(self.branch7x7x3_2(self.branch7x7x3_1(x))))
        branch_pool = self.branch_pool(x)

        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
    # 对应InceptionV3论文中的 Figure 7
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3_1 = self.branch3x3_1(x)
        branch3x3_2a = self.branch3x3_2a(branch3x3_1)
        branch3x3_2b = self.branch3x3_2b(branch3x3_1)
        branch3x3 = torch.cat([branch3x3_2a, branch3x3_2b], 1)

        branch3x3dbl_1 = self.branch3x3dbl_1(x)
        branch3x3dbl_2 = self.branch3x3dbl_2(branch3x3dbl_1)
        branch3x3dbl_3a = self.branch3x3dbl_3a(branch3x3dbl_2)
        branch3x3dbl_3b = self.branch3x3dbl_3b(branch3x3dbl_2)
        branch3x3dbl = torch.cat([branch3x3dbl_3a, branch3x3dbl_3b], 1)

        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    # 辅助分类器
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        # N x 768 x 17 x 17 -> N x 768 x 5 x 5
        x = self.pool(x)
        # N x 128 x 5 x 5
        x = self.conv0(x)
        # N x 768 x 1 x 1
        x = self.conv1(x)
        # N x 768
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True):
        super(InceptionV3, self).__init__()
        self.aux_logits = aux_logits

        # Stem
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.MaxPool_3a = nn.MaxPool2d(kernel_size=3, stride=2)

        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.MaxPool_5a = nn.MaxPool2d(kernel_size=3, stride=2)

        # Inception blocks
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        self.Mixed_6a = InceptionB(288)

        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)

        self.Mixed_7a = InceptionD(768)

        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, num_classes)

        # Parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_backbone(self):
        # Helper to define all layers except the final classifier
        layers = [
            self.Conv2d_1a_3x3, self.Conv2d_2a_3x3, self.Conv2d_2b_3x3, self.MaxPool_3a,
            self.Conv2d_3b_1x1, self.Conv2d_4a_3x3, self.MaxPool_5a,
            self.Mixed_5b, self.Mixed_5c, self.Mixed_5d,
            self.Mixed_6a, self.Mixed_6b, self.Mixed_6c, self.Mixed_6d, self.Mixed_6e,
            self.Mixed_7a, self.Mixed_7b, self.Mixed_7c
        ]
        if self.aux_logits:
            layers.append(self.AuxLogits)
        return layers

    def freeze_backbone(self):
        for module in self._get_backbone():
            for param in module.parameters():
                param.requires_grad = False

    def Unfreeze_backbone(self):
        for module in self._get_backbone():
            for param in module.parameters():
                param.requires_grad = True

    def forward(self, x):
        # Stem
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.MaxPool_3a(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.MaxPool_5a(x)

        # Inception blocks
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        # Auxiliary head
        aux = None
        if self.aux_logits and self.training:
            aux = self.AuxLogits(x)

        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)

        # Classifier
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.aux_logits and self.training:
            return x, aux
        return x


def inception_v3_google(pretrained=False, progress=True, num_classes=1000, aux_logits=True):
    """
    构造 InceptionV3 模型

    Args:
        pretrained (bool): 如果为 True, 返回在 ImageNet 上预训练的模型
        progress (bool): 如果为 True, 显示权重下载进度条
        num_classes (int): 输出分类数
        aux_logits (bool): 如果为 True, 创建辅助分类器
    """
    model = InceptionV3(num_classes=num_classes, aux_logits=aux_logits)

    if pretrained:
        # 预训练模型的分类数是1000
        state_dict = load_state_dict_from_url(model_urls['inception_v3_google'], model_dir='./model_data',
                                              progress=progress)

        # 如果期望的分类数与预训练模型(1000)不符，则不加载 fc 层的权重
        if num_classes != 1000:
            # 删除主分类器和辅助分类器的权重
            if 'fc.weight' in state_dict:
                del state_dict['fc.weight']
            if 'fc.bias' in state_dict:
                del state_dict['fc.bias']
            if aux_logits:
                if 'AuxLogits.fc.weight' in state_dict:
                    del state_dict['AuxLogits.fc.weight']
                if 'AuxLogits.fc.bias' in state_dict:
                    del state_dict['AuxLogits.fc.bias']

            # 使用 strict=False 加载，忽略缺失的键
            model.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights loaded, except for the classifier layers for {num_classes} classes.")
        else:
            model.load_state_dict(state_dict)
            print("Pretrained weights loaded successfully for 1000 classes.")

    return model
