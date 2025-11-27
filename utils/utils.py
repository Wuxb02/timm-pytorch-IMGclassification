import math
from functools import partial

import numpy as np
import torch
from PIL import Image

from .utils_aug import resize, center_crop


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
def get_classes(classes_path):
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

        if ',' in line:
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
def preprocess_input(x):
    # x /= 127.5
    # x -= 1.
    x /= 255
    x -= np.array([0.485, 0.456, 0.406])
    x /= np.array([0.229, 0.224, 0.225])
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
    for param_group in optimizer.param_groups:
        return param_group['lr']

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

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'mobilenetv2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
        'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
        'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
        'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
        'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
        'vit_b_16': 'https://github.com/bubbliiiing/classification-pytorch/releases/download/v1.0/vit-patch_16.pth',
        'swin_transformer_tiny': 'https://github.com/bubbliiiing/classification-pytorch/releases/download/v1.0/swin_tiny_patch4_window7_224_imagenet1k.pth',
        'swin_transformer_small': 'https://github.com/bubbliiiing/classification-pytorch/releases/download/v1.0/swin_small_patch4_window7_224_imagenet1k.pth',
        'swin_transformer_base': 'https://github.com/bubbliiiing/classification-pytorch/releases/download/v1.0/swin_base_patch4_window7_224_imagenet1k.pth',

        'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
        'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
        'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
        'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',

        'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
        'Xception': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth',
        'inceptionresnetv2': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth'
    }
    try:
        url = download_urls[backbone]
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        load_state_dict_from_url(url, model_dir)
    except:
        print("There is no pretrained model for " + backbone)


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