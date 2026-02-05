import math
from datetime import datetime

import numpy as np
import torch
from PIL import Image
import timm

from .utils_aug import resize, center_crop


# ---------------------------------------------------------#
#   统一的模型保存函数
# ---------------------------------------------------------#
def save_checkpoint(model, optimizer, epoch, save_path,
                    val_loss=None, val_acc=None,
                    minority_score=None, extra_info=None):
    """
    统一的模型保存函数，保存完整的训练状态

    Args:
        model: 模型实例
        optimizer: 优化器实例
        epoch: 当前 epoch
        save_path: 保存路径
        val_loss: 验证损失
        val_acc: 验证准确率
        minority_score: 少数类别复合分数
        extra_info: 额外信息字典
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'minority_score': minority_score,
        'save_time': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
    }

    if extra_info:
        checkpoint.update(extra_info)

    torch.save(checkpoint, save_path)
    return save_path


#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def letterbox_image(image, size, letterbox_image):
    w, h = size
    iw, ih = image.size
    if letterbox_image:
        '''resize image with unchanged aspect ratio using padding'''
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        if h == w:
            new_image = resize(image, h)
        else:
            new_image = resize(image, [h ,w])
        new_image = center_crop(new_image, [h ,w])
    return new_image

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path, old=False):
    """
    读取类别文件,支持两种格式:
    1. 新格式: "0, normal" (类别ID, 英文名称)
    2. 旧格式: "0" (仅类别ID)

    返回:
        class_names: 用于显示的类别名称列表
        num_classes: 类别数量
    """
    with open(classes_path, encoding='utf-8') as f:
        lines = f.readlines()

    class_names = []
    for line in lines:
        line = line.strip()
        if not line:  # 跳过空行
            continue

        if old:
            class_names.append(line)
        elif ',' in line:
            # 新格式: "0, normal"
            parts = line.split(',', 1)
            if len(parts) == 2:
                class_name = parts[1].strip()
                class_names.append(class_name)
            else:
                class_names.append(parts[0].strip())
        else:
            # 旧格式: "0"
            class_names.append(line)

    return class_names, len(class_names)

#----------------------------------------#
#   预处理训练图片
#----------------------------------------#
def preprocess_input(x, backbone=None):
    """
    图像归一化预处理

    Args:
        x: 输入图像数组 (H, W, C)
        backbone: timm模型名称（可选）
                 - 如果提供，使用模型特定的归一化参数
                 - 如果为None，使用默认ImageNet参数（向后兼容）

    Returns:
        归一化后的图像数组
    """
    x = x / 255.0

    # 动态获取归一化参数
    if backbone is not None:
        try:
            data_config = timm.data.resolve_data_config({}, model=backbone)
            mean = np.array(data_config.get('mean', [0.485, 0.456, 0.406]))
            std = np.array(data_config.get('std', [0.229, 0.224, 0.225]))
        except Exception as e:
            # 回退到默认参数（兼容性保障）
            print(f"[Warning] 无法获取模型 {backbone} 的归一化配置，使用默认参数: {e}")
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
    else:
        # 默认ImageNet参数（向后兼容旧代码）
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    x -= mean
    x /= std
    return x

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    """
    获取优化器的学习率
    - 冻结阶段（单参数组）：返回该学习率
    - 解冻阶段（多参数组）：返回所有学习率，格式如 "5e-8/5e-7/2.5e-6/5e-6"
    """
    param_groups = optimizer.param_groups

    if len(param_groups) == 1:
        # 冻结阶段：只有一个参数组（分类头）
        return param_groups[0]['lr']
    else:
        # 解冻阶段：多个参数组（分层学习率）
        lrs = [f"{pg['lr']:.4e}" for pg in param_groups]
        return "/".join(lrs)

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(optimizer, lr_decay_type, total_epochs,
                     warmup_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_ratio=0.05, min_lr_ratio=0.01):
    """
    创建PyTorch官方LambdaLR学习率调度器(三阶段自动调度,保留平方加速warmup)

    Args:
        optimizer: 优化器实例
        lr_decay_type: 'cos' 或 'step'
        total_epochs: 总训练轮次
        warmup_ratio: warmup阶段占比(默认0.05)
        warmup_lr_ratio: warmup起始学习率比例(默认0.1)
        no_aug_ratio: 预先退火阶段占比(默认0.05)
        min_lr_ratio: 最小学习率比例(默认0.01)

    Returns:
        scheduler: LambdaLR实例(三阶段调度) 或 StepLR实例
    """
    import torch

    # 计算各阶段轮次(与原实现一致)
    warmup_epochs = int(max(warmup_ratio * total_epochs, 1))
    warmup_epochs = min(warmup_epochs, 10)  # 上限10轮
    no_aug_epochs = int(max(no_aug_ratio * total_epochs, 1))
    no_aug_epochs = min(no_aug_epochs, 15)  # 上限15轮
    cosine_epochs = total_epochs - warmup_epochs - no_aug_epochs

    if lr_decay_type == "cos":
        def lr_lambda(epoch):
            """
            三阶段学习率倍率函数(返回相对于base_lr的倍率)

            阶段1 (Warmup): 平方加速从 base_lr*warmup_lr_ratio 到 base_lr
            阶段2 (Cosine): 余弦衰减从 base_lr 到 base_lr*min_lr_ratio
            阶段3 (No-Aug): 固定在 base_lr*min_lr_ratio
            """
            if epoch < warmup_epochs:
                # Warmup阶段: 平方加速(与原实现一致)
                ratio = epoch / float(max(1, warmup_epochs))
                return warmup_lr_ratio + (1.0 - warmup_lr_ratio) * (ratio ** 2)

            elif epoch < total_epochs - no_aug_epochs:
                # 余弦衰减阶段
                progress = (epoch - warmup_epochs) / float(max(1, cosine_epochs))
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

            else:
                # 预先退火阶段: 固定最小学习率
                return min_lr_ratio

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    else:
        # Step衰减模式
        decay_rate = min_lr_ratio ** (1 / 9)  # 10步衰减
        step_size = max(int(total_epochs / 10), 1)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=decay_rate
        )

    return scheduler

#---------------------------------------------------------#
#   动态分类框架支持函数
#---------------------------------------------------------#
def count_samples_per_class(annotation_path):
    """
    从标注文件统计各类别样本数量

    Args:
        annotation_path: 标注文件路径（如 cls_train.txt）

    Returns:
        samples_per_class: list，各类别样本数量 [count_0, count_1, ...]
        class_counts: dict，{类别ID: 数量}
    """
    class_counts = {}
    with open(annotation_path, encoding='utf-8-sig') as f:
        for line in f:
            if ';' in line:
                class_id = int(line.split(';')[0])
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

    # 转换为有序列表
    num_classes = max(class_counts.keys()) + 1
    samples_per_class = [class_counts.get(i, 0) for i in range(num_classes)]

    return samples_per_class, class_counts


def identify_minority_class(class_counts):
    """
    识别少数类别（样本数最少的类别）

    Args:
        class_counts: dict，{类别ID: 数量}

    Returns:
        minority_idx: int，少数类别索引
        minority_ratio: float，少数类别占比
    """
    total = sum(class_counts.values())
    minority_idx = min(class_counts, key=class_counts.get)
    minority_ratio = class_counts[minority_idx] / total

    return minority_idx, minority_ratio


def get_class_display_names(classes_path):
    """
    获取类别显示名称

    优先级：
    1. 尝试加载 cls_classes_names.txt（格式：0|良性）
    2. 回退到类别ID（"0", "1", "2"）

    Returns:
        display_names: list，类别显示名称
        num_classes: int，类别数量
    """
    import os

    # 首先获取类别ID
    class_ids, num_classes = get_classes(classes_path)

    # 尝试加载映射文件
    mapping_path = classes_path.replace('.txt', '_names.txt')
    if os.path.exists(mapping_path):
        try:
            mapping = {}
            with open(mapping_path, encoding='utf-8-sig') as f:
                for line in f:
                    if '|' in line:
                        cid, name = line.strip().split('|', 1)
                        mapping[cid.strip()] = name.strip()
            display_names = [mapping.get(cid, cid) for cid in class_ids]
            return display_names, num_classes
        except Exception:
            pass

    # 回退到类别ID
    return class_ids, num_classes