import os
from threading import local

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .utils import get_lr
from .focal_loss import FocalLoss, ClassBalancedFocalLoss
from .early_stopping import EarlyStopping, ModelCheckpoint, ClassBalancedMetrics


def fit_one_epoch_incep(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch,
                  cuda, fp16, scaler, save_period, save_dir, local_rank=0, early_stopping=None, model_checkpoint=None, num_classes=None, class_names=None, samples_per_class=None, minority_idx=None, criterion=None):
    """
    新增参数:
        num_classes: 类别数量(从train_trimm.py传入)
        class_names: 类别名称列表
        samples_per_class: 各类别样本数量
        minority_idx: 少数类别索引
        criterion: 损失函数实例(从train_trimm.py传入,推荐方式)
    """
    # 动态参数验证
    if samples_per_class is None or num_classes is None:
        raise ValueError("必须传入 num_classes 和 samples_per_class 参数!")
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(num_classes)]
    if minority_idx is None:
        minority_idx = num_classes - 1  # 默认最后一个类别
    total_loss = 0
    total_accuracy = 0

    val_loss = 0
    val_accuracy = 0

    # 损失函数处理：优先使用传入的criterion，否则使用默认的ClassBalancedFocalLoss
    if criterion is None:
        print("[Warning] 未传入损失函数，使用默认的ClassBalancedFocalLoss")
        criterion = ClassBalancedFocalLoss(
            beta=0.9999,
            gamma=2.0,
            samples_per_class=samples_per_class
        )

    # 初始化类别平衡指标跟踪(动态参数)
    train_metrics = ClassBalancedMetrics(num_classes=num_classes,
                                         class_names=class_names,
                                         minority_class=minority_idx)
    val_metrics = ClassBalancedMetrics(num_classes=num_classes,
                                       class_names=class_names,
                                       minority_class=minority_idx)

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(images)

            # --------------------------------------------------------------------------------#
            #   MODIFIED BLOCK FOR INCEPTIONV3                                                 #
            #   如果模型输出是元组 (主输出, 辅助输出)，则分别计算损失并加权求和               #
            # --------------------------------------------------------------------------------#
            if isinstance(outputs, tuple):
                main_output, aux_output = outputs
                loss_main = criterion(main_output, targets)
                loss_aux = criterion(aux_output, targets)
                loss_value = loss_main + 0.4 * loss_aux
                # 准确率计算只使用主输出
                outputs_for_acc = main_output
            else:
                loss_value = criterion(outputs, targets)
                outputs_for_acc = outputs
            # --------------------------------------------------------------------------------#

            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs = model_train(images)

                # --------------------------------------------------------------------------------#
                #   MODIFIED BLOCK FOR INCEPTIONV3 (FP16)                                        #
                # --------------------------------------------------------------------------------#
                if isinstance(outputs, tuple):
                    main_output, aux_output = outputs
                    loss_main = criterion(main_output, targets)
                    loss_aux = criterion(aux_output, targets)
                    loss_value = loss_main + 0.4 * loss_aux
                    outputs_for_acc = main_output
                else:
                    loss_value = criterion(outputs, targets)
                    outputs_for_acc = outputs
                # --------------------------------------------------------------------------------#

            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss_value.item()
        with torch.no_grad():
            # 使用 outputs_for_acc 进行准确率计算
            predictions = torch.argmax(F.softmax(outputs_for_acc, dim=-1), dim=-1)
            accuracy = torch.mean((predictions == targets).type(torch.FloatTensor))
            total_accuracy += accuracy.item()
            
            # 更新类别平衡指标
            train_metrics.update(predictions.cpu(), targets.cpu())

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'accuracy': total_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

            optimizer.zero_grad()

            outputs = model_train(images)
            # 在eval模式下，InceptionV3只返回一个输出，但为保持代码健壮性，仍使用相同的逻辑
            if isinstance(outputs, tuple):
                # 理论上在eval模式下不会进入此分支，但为保险起见
                main_output, _ = outputs
                loss_value = criterion(main_output, targets)
                outputs_for_acc = main_output
            else:
                loss_value = criterion(outputs, targets)
                outputs_for_acc = outputs

            val_loss += loss_value.item()
            # 使用 outputs_for_acc 进行准确率计算
            predictions = torch.argmax(F.softmax(outputs_for_acc, dim=-1), dim=-1)
            accuracy = torch.mean((predictions == targets).type(torch.FloatTensor))
            val_accuracy += accuracy.item()
            
            # 更新验证集类别平衡指标
            val_metrics.update(predictions.cpu(), targets.cpu())

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'accuracy': val_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        loss_history.append_acc(epoch + 1, total_accuracy / epoch_step, val_accuracy / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        # 计算类别平衡指标
        train_scores = train_metrics.compute()
        val_scores = val_metrics.compute()
        
        print(f'训练集 - 各类别F1: {train_scores["class_f1"]}, 少数类得分: {train_scores["minority_score"]:.3f}')
        print(f'验证集 - 各类别F1: {val_scores["class_f1"]}, 少数类得分: {val_scores["minority_score"]:.3f}')
        
        # 重置指标
        train_metrics.reset()
        val_metrics.reset()
        
        # 早停检查
        should_stop = False
        if early_stopping is not None:
            should_stop = early_stopping(val_scores["minority_score"], model, epoch)
            
        if model_checkpoint is not None:
            model_checkpoint(val_scores["minority_score"], model, epoch)

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1,
                                                                                                        total_loss / epoch_step,
                                                                                                        val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
        
    return should_stop  # 返回True表示应该早停，False表示继续训练


