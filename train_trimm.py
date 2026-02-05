import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ----------------- 新增导入 -----------------
import timm
# -------------------------------------------

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.callbacks import LossHistory
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import (get_classes, get_lr_scheduler,
                         show_config, weights_init,
                         count_samples_per_class, identify_minority_class)
from utils.utils_fit import fit_one_epoch
from utils.early_stopping import EarlyStopping, ModelCheckpoint


# ----------------- 新增辅助函数用于冻结/解冻 -----------------
def freeze_timm_backbone(model):
    """
    冻结 timm 模型的主干部分，仅保留分类头可训练

    支持多种 timm 模型架构的分类头命名方式
    """
    print("Freezing model backbone...")

    # 第一步：冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 第二步：解冻分类头（按优先级尝试多种方式）
    classifier_unfrozen = False

    # 方式1：使用 timm 的标准接口 get_classifier()
    if hasattr(model, 'get_classifier'):
        try:
            classifier = model.get_classifier()
            if classifier is not None:
                for param in classifier.parameters():
                    param.requires_grad = True
                classifier_unfrozen = True
                print("  -> 通过 get_classifier() 解冻分类头")
        except Exception as e:
            print(f"  -> get_classifier() 失败: {e}")

    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  -> 可训练参数: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    if not classifier_unfrozen:
        print("\033[1;33m[Warning] 未能解冻任何分类头参数，请检查模型结构\033[0m")


def unfreeze_timm_backbone(model):
    """解冻 timm 模型的所有参数"""
    print("Unfreezing all model parameters...")
    for param in model.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  -> 全部参数已解冻: {trainable_params:,}")


def get_layer_wise_param_groups(model, base_lr, lr_mult_early=0.01,
                                 lr_mult_middle=0.1, lr_mult_late=0.5):
    """
    获取分层学习率的参数组，用于解冻阶段的微调。
    早期层使用更小的学习率，避免破坏预训练特征。

    Args:
        model: timm模型实例
        base_lr: 基础学习率（分类头使用此学习率）
        lr_mult_early: 早期层学习率倍率（默认0.01）
        lr_mult_middle: 中间层学习率倍率（默认0.1）
        lr_mult_late: 后期层学习率倍率（默认0.5）

    Returns:
        list: 参数组列表，可直接传入优化器
    """
    classifier_names = ['head', 'fc', 'classifier', 'last_linear', 'output']
    named_params = list(model.named_parameters())
    total_trainable = len([p for _, p in named_params if p.requires_grad])

    param_groups = {'early': [], 'middle': [], 'late': [], 'classifier': []}
    trainable_idx = 0

    for name, param in named_params:
        if not param.requires_grad:
            continue

        is_classifier = any(cn in name for cn in classifier_names)
        if is_classifier:
            param_groups['classifier'].append(param)
        else:
            relative_pos = trainable_idx / total_trainable if total_trainable > 0 else 0
            if relative_pos < 0.33:
                param_groups['early'].append(param)
            elif relative_pos < 0.66:
                param_groups['middle'].append(param)
            else:
                param_groups['late'].append(param)
        trainable_idx += 1

    optimizer_groups = []
    group_info = []
    if param_groups['early']:
        optimizer_groups.append({
            'params': param_groups['early'],
            'lr': base_lr * lr_mult_early
        })
        group_info.append(f"early({len(param_groups['early'])}): lr={base_lr * lr_mult_early:.2e}")
    if param_groups['middle']:
        optimizer_groups.append({
            'params': param_groups['middle'],
            'lr': base_lr * lr_mult_middle
        })
        group_info.append(f"middle({len(param_groups['middle'])}): lr={base_lr * lr_mult_middle:.2e}")
    if param_groups['late']:
        optimizer_groups.append({
            'params': param_groups['late'],
            'lr': base_lr * lr_mult_late
        })
        group_info.append(f"late({len(param_groups['late'])}): lr={base_lr * lr_mult_late:.2e}")
    if param_groups['classifier']:
        optimizer_groups.append({
            'params': param_groups['classifier'],
            'lr': base_lr
        })
        group_info.append(f"classifier({len(param_groups['classifier'])}): lr={base_lr:.2e}")

    print(f"  -> 分层学习率参数组: {', '.join(group_info)}")
    return optimizer_groups


# -------------------------------------------------------------

def create_weighted_sampler(annotation_lines, samples_per_class):
    """
    创建加权随机采样器来处理类别不平衡

    Args:
        annotation_lines: 标注文件行列表
        samples_per_class: 各类别样本数量列表

    Returns:
        WeightedRandomSampler: 加权采样器
    """
    from torch.utils.data import WeightedRandomSampler

    total_samples = sum(samples_per_class)
    class_weights = [total_samples / count for count in samples_per_class]

    sample_weights = []
    for line in annotation_lines:
        class_id = int(line.split(';')[0])
        sample_weights.append(class_weights[class_id])

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # 动态计算少数类别信息
    minority_idx = class_weights.index(max(class_weights))  # 权重最大的是少数类
    majority_idx = class_weights.index(min(class_weights))  # 权重最小的是多数类

    print(f"创建加权采样器：类别权重 = {[f'{w:.3f}' for w in class_weights]}")
    print(f"少数类别({minority_idx})被选中的概率提升了 {class_weights[minority_idx]/class_weights[majority_idx]:.1f} 倍")

    return sampler

if __name__ == "__main__":
    # ----------------------------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # ----------------------------------------------------#
    Cuda = True
    # ---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    # ---------------------------------------------------------------------#
    distributed = False
    sync_bn = False
    fp16 = True
    # ----------------------------------------------------#
    #   classes_path
    # ----------------------------------------------------#
    classes_path = './model_data/cls_classes.txt'
    #   所用模型种类 (请使用timm支持的模型名称)
    #   例如: 'resnet50', 'efficientnet_b0', 'vit_base_patch16_224',
    #        'swin_tiny_patch4_window7_224', 'inception_resnet_v2', 'inception_v3'
    #   可以通过 timm.list_models('*inception*') 查看支持的名称
    # ------------------------------------------------------#
    # 推荐模型优先级：对小数据集分类效果更好
    backbone = "inception_resnet_v2"  # 更适合小数据集的模型
    # 其他备选模型:
    # backbone = "convnext_tiny"     # 现代CNN架构
    # backbone = "swin_small_patch4_window7_224"  # 适中的Transformer
    # backbone = "vit_base_patch16_224"  # 原始选择
    # -----------------------------------------6-----------------------------------------------------------------------------------#
    #   是否使用主干网络的预训练权重
    #   timm会自动处理从网络下载预训练权重
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained = True
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   模型配置参数
    # ----------------------------------------------------------------------------------------------------------------------------#
    drop_rate = 0.5             # Dropout 比率 - 抗过拟合优化：提升至0.5
    aux_loss_weight = 0.4        # Inception 辅助损失权重
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   模型断点续练的权值路径
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path = r""

    data_config = timm.data.resolve_data_config({}, model=backbone) #模型配置
    # ----------------------------------------------------#
    #   输入的图片大小
    # ----------------------------------------------------#
    # input_shape = data_config['input_size'][1:]
    input_shape = [299,299]
    # ------------------------------------------------------#
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   优化后的训练阶段参数 - 针对类别不平衡问题调整
    # ----------------------------------------------------------------------------------------------------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 20          # 方案一优化：缩短冻结轮数，避免分类头过拟合
    Freeze_batch_size = 64      # 冻结阶段：大batch size，充分利用显存，加快训练并稳定梯度
    UnFreeze_Epoch = 200       # 适中训练轮数，配合早停机制
    Unfreeze_batch_size = 48    # 解冻阶段：小batch size，增加梯度噪声，防止过拟合
    Freeze_Train = True

    # ------------------------------------------------------------------#
    #   冻结阶段学习率配置 - 方案一优化
    # ------------------------------------------------------------------#
    Freeze_Init_lr = 1e-4          # 冻结阶段初始学习率（降低以减少验证集波动）
    Freeze_Min_lr = 1e-5           # 冻结阶段最小学习率
    Freeze_warmup_ratio = 0.15     # 冻结阶段warmup比例（20轮×0.15=3轮，更平滑过渡）

    # ------------------------------------------------------------------#
    #   解冻阶段学习率配置 - 抗过拟合优化
    # ------------------------------------------------------------------#
    Unfreeze_Init_lr = 5e-6        # 解冻阶段初始学习率（大幅降低，防止过拟合）
    Unfreeze_Min_lr = 5e-7         # 解冻阶段最小学习率
    Unfreeze_warmup_ratio = 0.03   # 解冻阶段warmup比例（占总轮次的3%）

    # ------------------------------------------------------------------#
    #   预先退火配置
    # ------------------------------------------------------------------#
    no_aug_iter_ratio = 0.05       # 预先退火比例（最后5%轮次固定在最小学习率）

    # ------------------------------------------------------------------#
    #   其他训练参数
    # ------------------------------------------------------------------#
    optimizer_type = "adamw"       # AdamW 比 Adam 有更好的正则化效果
    momentum = 0.9
    weight_decay = 5e-2            # 抗过拟合优化：大幅增加权重衰减
    lr_decay_type = "cos"          # 余弦衰减学习率
    save_period = 50               # 更频繁地保存模型
    save_dir = f'models/{backbone}'
    num_workers = 2                # 减少并行线程，避免数据加载冲突
    
    # ------------------------------------------------------#
    #   数据不平衡处理配置
    # ------------------------------------------------------#
    use_weighted_sampler = False  # 启用加权采样，有助于提升少数类别性能

    # ------------------------------------------------------#
    #   损失函数配置 (Loss Function Selection)
    # ------------------------------------------------------#
    # 可选损失函数:
    #   - 'ce':                  标准交叉熵损失 (CrossEntropyLoss) - 适合类别平衡数据
    #   - 'focal':               Focal Loss - 专注于困难样本
    #   - 'cb_focal':            类别平衡Focal Loss - 推荐用于类别不平衡 ✅
    #   - 'label_smoothing':     标签平滑交叉熵 - 减少过拟合,提升泛化能力
    # ------------------------------------------------------#
    loss_type = "ce"  

    # Focal Loss 参数 (loss_type为'focal'或'cb_focal'时生效)
    focal_alpha = None       # 类别权重,None表示自动计算
    focal_gamma = 2.0        # 聚焦参数,越大越关注困难样本

    # 类别平衡Focal Loss 参数 (loss_type为'cb_focal'时生效)
    cb_focal_beta = 0.99   # 重采样参数,越接近1类别平衡效果越强

    # 标签平滑参数 (loss_type为'label_smoothing'时生效)
    label_smoothing = 0.1    # 平滑系数,通常取0.1

    # ------------------------------------------------------#
    #   数据集路径
    # ------------------------------------------------------#
    train_annotation_path = "cls_train.txt"
    test_annotation_path = 'cls_test.txt'

    # ------------------------------------------------------#
    #   设置用到的显卡
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    # ------------------------------------------------------#
    #   获取classes
    # ------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    # ------------------------------------------------------#
    #   !!! 核心修改：使用timm创建模型 !!!
    # ------------------------------------------------------#
    #   注意: backbone变量需要是timm支持的名称
    #   timm.create_model 会自动处理ViT等模型的输入尺寸参数，代码更简洁
    if 'ception' in backbone.lower():  # 处理Inception的辅助分类头
        model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    else:
        model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    # 备注：对于ViT等模型，如果需要指定非标准的图片大小，可以传入 img_size 参数
    # model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes, img_size=input_shape[0])

    # 你的项目中的weights_init是自定义初始化，如果不用预训练可以开启
    if not pretrained:
        weights_init(model)  # 假设weights_init能兼容timm模型

    if model_path != "":
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        try:
            # 检查文件是否存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"权重文件不存在: {model_path}")

            model_dict = model.state_dict()
            checkpoint = torch.load(model_path, map_location=device)

            # 统一从完整checkpoint格式读取
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                pretrained_dict = checkpoint['model_state_dict']
                if local_rank == 0:
                    print(f"检测到完整checkpoint格式，epoch={checkpoint.get('epoch', 'N/A')}")
            else:
                pretrained_dict = checkpoint

            load_key, no_load_key, temp_dict = [], [], {}
            for k, v in pretrained_dict.items():
                if k in model_dict.keys():
                    # 检查形状和数据类型
                    if model_dict[k].shape == v.shape:
                        if model_dict[k].dtype == v.dtype:
                            temp_dict[k] = v
                            load_key.append(k)
                        else:
                            # 尝试类型转换
                            try:
                                temp_dict[k] = v.to(model_dict[k].dtype)
                                load_key.append(k)
                                if local_rank == 0:
                                    print(f"[Warning] 参数 {k} 类型转换: {v.dtype} -> {model_dict[k].dtype}")
                            except Exception as e:
                                no_load_key.append(k)
                                if local_rank == 0:
                                    print(f"[Warning] 参数 {k} 类型转换失败: {e}")
                    else:
                        no_load_key.append(k)
                        if local_rank == 0:
                            print(f"[Warning] 参数 {k} 形状不匹配: 期望{model_dict[k].shape}, 实际{v.shape}")
                else:
                    no_load_key.append(k)

            model_dict.update(temp_dict)
            model.load_state_dict(model_dict)

            if local_rank == 0:
                print(f"\n成功加载 {len(load_key)} 个参数")
                if no_load_key:
                    print(f"跳过 {len(no_load_key)} 个参数: {str(no_load_key)[:200]}...")

                # 验证加载比例
                load_ratio = len(load_key) / len(model_dict) * 100
                if load_ratio < 50:
                    print(f"\033[1;33;44m[Warning] 仅加载了 {load_ratio:.1f}% 的参数，请检查权重文件是否匹配\033[0m")
                print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

        except FileNotFoundError as e:
            print(f"\033[1;31m[Error] {e}\033[0m")
            raise
        except RuntimeError as e:
            print(f"\033[1;31m[Error] 权重加载失败: {e}\033[0m")
            print("可能原因: 1) 文件损坏 2) PyTorch版本不兼容 3) 模型结构不匹配")
            raise
        except Exception as e:
            print(f"\033[1;31m[Error] 未知错误: {e}\033[0m")
            raise

    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if fp16:
        from torch.amp import GradScaler as GradScaler

        scaler = GradScaler('cuda')
    else:
        scaler = None

    model_train = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    with open(train_annotation_path, encoding='utf-8-sig') as f:
        train_lines = f.readlines()
    with open(test_annotation_path, encoding='utf-8-sig') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    np.random.seed(10101)
    np.random.shuffle(train_lines)
    np.random.seed(None)

    # 动态统计各类别样本数量
    samples_per_class, class_counts = count_samples_per_class(train_annotation_path)
    minority_idx, minority_ratio = identify_minority_class(class_counts)
    if local_rank == 0:
        print(f"各类别样本数量: {samples_per_class}")
        print(f"少数类别: 索引{minority_idx}, 占比{minority_ratio:.1%}")

    # ------------------------------------------------------#
    #   创建损失函数 (根据配置)
    # ------------------------------------------------------#
    from utils.focal_loss import get_loss_function

    criterion = get_loss_function(
        loss_type=loss_type,
        num_classes=num_classes,
        samples_per_class=samples_per_class,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        cb_focal_beta=cb_focal_beta,
        label_smoothing=label_smoothing
    )
    if local_rank == 0:
        print("=" * 80)

    if local_rank == 0:
        show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape, \
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
            Init_lr=Freeze_Init_lr, Min_lr=Freeze_Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )

        # 显示学习率配置信息
        print("\n" + "=" * 80)
        print("学习率配置:")
        print(f"  冻结阶段: {Freeze_Init_lr:.2e} -> {Freeze_Min_lr:.2e} (Warmup: {Freeze_warmup_ratio*100:.1f}%)")
        print(f"  解冻阶段: {Unfreeze_Init_lr:.2e} -> {Unfreeze_Min_lr:.2e} (Warmup: {Unfreeze_warmup_ratio*100:.1f}%)")
        print(f"  预先退火比例: {no_aug_iter_ratio*100:.1f}%")
        print("=" * 80)

    wanted_step = 3e4 if optimizer_type == "sgd" else 1e4
    total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (optimizer_type,
                                                                                                wanted_step))
        print(
            "\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (
                num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
        print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (total_step,
                                                                                                         wanted_step,
                                                                                                         wanted_epoch))

    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            # ------------------------------------#
            #   !!! 核心修改：使用新的冻结函数 !!!
            # ------------------------------------#
            freeze_timm_backbone(model)

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # 直接使用配置的学习率（无需缩放）
        Init_lr_fit = Freeze_Init_lr
        Min_lr_fit = Freeze_Min_lr

        optimizer = {
            'adam': optim.Adam(model_train.parameters(), Init_lr_fit, betas=(momentum, 0.999),
                               weight_decay=weight_decay),
            'adamw': optim.AdamW(model_train.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay), # 推荐
            'sgd': optim.SGD(model_train.parameters(), Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]

        # 计算最小学习率比例
        min_lr_ratio_freeze = Min_lr_fit / Init_lr_fit

        # 创建PyTorch官方scheduler
        lr_scheduler = get_lr_scheduler(
            optimizer=optimizer,
            lr_decay_type=lr_decay_type,
            total_epochs=Freeze_Epoch,
            warmup_ratio=Freeze_warmup_ratio,
            no_aug_ratio=no_aug_iter_ratio,
            min_lr_ratio=min_lr_ratio_freeze
        )

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset = DataGenerator(train_lines, input_shape, backbone=backbone, random=True, autoaugment_flag=True)
        val_dataset = DataGenerator(val_lines, input_shape, backbone=backbone, random=False, autoaugment_flag=False)

        if distributed:
            # 分布式训练暂不支持加权采样（需要额外的复杂处理）
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
            if use_weighted_sampler:
                print("警告：分布式训练模式下暂不支持加权采样，将使用标准随机采样")
        else:
            # 单机训练支持加权采样
            if use_weighted_sampler:
                train_sampler = create_weighted_sampler(train_lines, samples_per_class)
                shuffle = False  # 使用采样器时必须设置shuffle=False
                print("✓ 已启用加权随机采样，有助于提升少数类别性能")
            else:
                train_sampler = None
                shuffle = True
                print("使用标准随机采样")
            
            val_sampler = None  # 验证集始终使用标准采样

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=False, collate_fn=detection_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=False, collate_fn=detection_collate, sampler=val_sampler)

        # 初始化早停和模型检查点
        early_stopping = EarlyStopping(
            patience=50,           # 50个epoch没有改善就停止
            min_delta=0.001,       # 最小改善阈值
            restore_best_weights=True,
            save_dir=save_dir,
            metric='minority_score',  # 监控交界性类别复合分数
            mode='max'             # 分数越高越好
        )
        
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(save_dir, 'best_minority_model.pth'),
            monitor='minority_score',
            mode='max',
            save_best_only=True,
            verbose=1
        )

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                # ----------------------------------------------------------#
                #   解冻阶段学习率策略改进：
                #   使用固定的较低学习率，避免batch_size变化导致学习率骤降
                #   同时防止解冻后学习率过高破坏预训练特征
                # ----------------------------------------------------------#
                Init_lr_fit = Unfreeze_Init_lr
                Min_lr_fit = Unfreeze_Min_lr

                remaining_epochs = UnFreeze_Epoch - epoch
                # 计算最小学习率比例
                min_lr_ratio_unfreeze = Min_lr_fit / Init_lr_fit

                # 重新创建PyTorch官方scheduler
                lr_scheduler = get_lr_scheduler(
                    optimizer=optimizer,
                    lr_decay_type=lr_decay_type,
                    total_epochs=remaining_epochs,
                    warmup_ratio=Unfreeze_warmup_ratio,
                    no_aug_ratio=no_aug_iter_ratio,
                    min_lr_ratio=min_lr_ratio_unfreeze
                )

                # ------------------------------------#
                #   !!! 核心修改：使用新的解冻函数 !!!
                # ------------------------------------#
                unfreeze_timm_backbone(model)

                # ----------------------------------------------------------#
                #   使用分层学习率重新创建优化器
                #   早期层使用更小的学习率，保护预训练特征
                # ----------------------------------------------------------#
                param_groups = get_layer_wise_param_groups(model, Init_lr_fit)
                optimizer = {
                    'adam': optim.Adam(param_groups, betas=(momentum, 0.999), weight_decay=weight_decay),
                    'adamw': optim.AdamW(param_groups, betas=(momentum, 0.999), weight_decay=weight_decay),
                    'sgd': optim.SGD(param_groups, momentum=momentum, nesterov=True, weight_decay=weight_decay)
                }[optimizer_type]
                print(f"[Unfreeze] 使用分层学习率，基础lr={Init_lr_fit:.2e}")

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=False, collate_fn=detection_collate, sampler=train_sampler)
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=False, collate_fn=detection_collate, sampler=val_sampler)

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            should_stop = fit_one_epoch(
                model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val,
                gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank,
                early_stopping=early_stopping, model_checkpoint=model_checkpoint,
                num_classes=num_classes, class_names=class_names,
                samples_per_class=samples_per_class, minority_idx=minority_idx,
                criterion=criterion, aux_loss_weight=aux_loss_weight
            )

            # 更新学习率(三阶段全自动,无需手动处理)
            lr_scheduler.step()

            # 检查是否应该早停
            if should_stop:
                print(f"训练在第 {epoch + 1} 轮早停！")
                break

        if local_rank == 0:
            loss_history.writer.close()

            # 训练完成提示
            print("\n" + "=" * 80)
            print("训练完成!")
            print("=" * 80)
            print("\n如需评估模型性能，请手动运行:")
            print('  Windows: "D:\\anaconda\\python.exe" eval.py')
            print('  Linux:   python eval.py')
