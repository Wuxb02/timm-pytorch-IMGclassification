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

# from nets import get_model_from_name # <- 不再需要
from utils.callbacks import LossHistory
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import (download_weights, get_classes, get_lr_scheduler,
                         set_optimizer_lr, show_config, weights_init,
                         count_samples_per_class, identify_minority_class)
from utils.utils_fit import fit_one_epoch
from utils.utils_fit_incep import fit_one_epoch_incep
from utils.early_stopping import EarlyStopping, ModelCheckpoint


# ----------------- 新增辅助函数用于冻结/解冻 -----------------
def freeze_timm_backbone(model):
    """冻结timm模型的主干部分"""
    print("Freezing model backbone...")
    # 冻结所有层
    for param in model.parameters():
        param.requires_grad = False

    # 解冻分类头
    try:
        # 大多数timm模型都有get_classifier()方法
        classifier_params = model.get_classifier().parameters()
        for param in classifier_params:
            param.requires_grad = True
        print("Successfully unfroze the classifier head.")
    except AttributeError:
        # 对于没有get_classifier()的旧模型，可能需要手动指定，例如 model.fc
        # 这里只是一个备用方案，大多数现代timm模型都支持 get_classifier()
        print("Warning: model.get_classifier() not found. Freezing might not work as expected.")


def unfreeze_timm_backbone(model):
    """解冻timm模型的所有参数"""
    print("Unfreezing all model parameters...")
    for param in model.parameters():
        param.requires_grad = True


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
    # ----------------------------------------------------#
    #   输入的图片大小
    # ----------------------------------------------------#
    input_shape = [299, 299]
    # ------------------------------------------------------#
    #   所用模型种类 (请使用timm支持的模型名称)
    #   例如: 'resnet50', 'efficientnet_b0', 'vit_base_patch16_224',
    #        'swin_tiny_patch4_window7_224', 'inception_resnet_v2', 'inception_v3'
    #   可以通过 timm.list_models('*inception*') 查看支持的名称
    # ------------------------------------------------------#
    # 推荐模型优先级：对小数据集分类效果更好
    backbone = "inception_v3"  # 更适合小数据集的模型
    # 其他备选模型:
    # backbone = "convnext_tiny"     # 现代CNN架构
    # backbone = "swin_small_patch4_window7_224"  # 适中的Transformer
    # backbone = "vit_base_patch16_224"  # 原始选择
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   是否使用主干网络的预训练权重
    #   timm会自动处理从网络下载预训练权重
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained = True
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   模型断点续练的权值路径
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path = ""

    # ----------------------------------------------------------------------------------------------------------------------------#
    #   优化后的训练阶段参数 - 针对类别不平衡问题调整
    # ----------------------------------------------------------------------------------------------------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 30          # 增加冻结轮数，稳定特征提取器训练
    Freeze_batch_size = 128      # 降低batch size，增加更新频率
    UnFreeze_Epoch = 200       # 适中训练轮数，配合早停机制
    Unfreeze_batch_size = 64    # 更小的batch size，有利于类别2的学习
    Freeze_Train = True

    # ------------------------------------------------------------------#
    #   优化后的训练参数 - 适合小样本不平衡分类任务
    # ------------------------------------------------------------------#
    Init_lr = 1e-4            # 降低学习率，避免过拟合(小样本场景建议较小学习率)
    Min_lr = Init_lr * 0.01    # 提高最小学习率，保持持续学习
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 1e-4        # 添加权重衰减，防止过拟合
    lr_decay_type = "cos"
    save_period = 50           # 更频繁地保存模型
    save_dir = f'models/{backbone}'
    num_workers = 12            # 减少并行线程，避免数据加载冲突
    
    # ------------------------------------------------------#
    #   数据不平衡处理配置
    # ------------------------------------------------------#
    use_weighted_sampler = True  # 是否使用加权采样（建议开启，有助于提升少数类别性能）

    # ------------------------------------------------------#
    #   损失函数配置 (Loss Function Selection)
    # ------------------------------------------------------#
    # 可选损失函数:
    #   - 'ce':                  标准交叉熵损失 (CrossEntropyLoss) - 适合类别平衡数据
    #   - 'focal':               Focal Loss - 专注于困难样本
    #   - 'cb_focal':            类别平衡Focal Loss - 推荐用于类别不平衡 ✅
    #   - 'label_smoothing':     标签平滑交叉熵 - 减少过拟合,提升泛化能力
    # ------------------------------------------------------#
    loss_type = "focal"  # 默认使用类别平衡Focal Loss

    # Focal Loss 参数 (loss_type为'focal'或'cb_focal'时生效)
    focal_alpha = None       # 类别权重,None表示自动计算
    focal_gamma = 2.0        # 聚焦参数,越大越关注困难样本

    # 类别平衡Focal Loss 参数 (loss_type为'cb_focal'时生效)
    cb_focal_beta = 0.9   # 重采样参数,越接近1类别平衡效果越强

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
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
    # 备注：对于ViT等模型，如果需要指定非标准的图片大小，可以传入 img_size 参数
    # model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes, img_size=input_shape[0])

    # 你的项目中的weights_init是自定义初始化，如果不用预训练可以开启
    if not pretrained:
        weights_init(model)  # 假设weights_init能兼容timm模型

    if model_path != "":
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

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
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )

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

        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        # timm模型对ViT和Swin Transformer有更标准化的处理，可以统一学习率调整策略
        if 'vit' in backbone or 'swin' in backbone:
            nbs = 256
            lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
            lr_limit_min = 1e-5 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': optim.Adam(model_train.parameters(), Init_lr_fit, betas=(momentum, 0.999),
                               weight_decay=weight_decay),
            'sgd': optim.SGD(model_train.parameters(), Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset = DataGenerator(train_lines, input_shape, random=True, autoaugment_flag=True)
        val_dataset = DataGenerator(val_lines, input_shape, random=False, autoaugment_flag=False)

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
                         drop_last=True, collate_fn=detection_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=detection_collate, sampler=val_sampler)

        # 初始化早停和模型检查点
        early_stopping = EarlyStopping(
            patience=50,           # 30个epoch没有改善就停止
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

                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
                lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                if 'vit' in backbone or 'swin' in backbone:
                    nbs = 256
                    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min = 1e-5 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                # ------------------------------------#
                #   !!! 核心修改：使用新的解冻函数 !!!
                # ------------------------------------#
                unfreeze_timm_backbone(model)

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=detection_collate, sampler=train_sampler)
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=detection_collate, sampler=val_sampler)

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            should_stop = fit_one_epoch(
                model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val,
                gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank,
                early_stopping=early_stopping, model_checkpoint=model_checkpoint,
                num_classes=num_classes, class_names=class_names,
                samples_per_class=samples_per_class, minority_idx=minority_idx,
                criterion=criterion
            )
            
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
