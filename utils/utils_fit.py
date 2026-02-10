import os
from threading import local

import torch
from torch import nn
from tqdm import tqdm

from .utils import get_lr, save_checkpoint
from .focal_loss import FocalLoss, ClassBalancedFocalLoss, get_loss_function
from .early_stopping import EarlyStopping, ModelCheckpoint, ClassBalancedMetrics


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0, early_stopping=None, model_checkpoint=None, num_classes=None, class_names=None, samples_per_class=None, minority_idx=None, criterion=None, aux_loss_weight=0.4, freeze_bn=True):
    """
    新增参数:
        num_classes: 类别数量(从train_trimm.py传入)
        class_names: 类别名称列表
        samples_per_class: 各类别样本数量
        minority_idx: 少数类别索引
        criterion: 损失函数实例(从train_trimm.py传入,推荐方式)
        aux_loss_weight: Inception 等模型的辅助损失权重，默认 0.4
    """
    # 动态参数验证
    if samples_per_class is None or num_classes is None:
        raise ValueError("必须传入 num_classes 和 samples_per_class 参数!")
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(num_classes)]
    if minority_idx is None:
        minority_idx = num_classes - 1  # 默认最后一个类别
    total_loss      = 0
    total_accuracy  = 0

    val_loss        = 0
    val_accuracy    = 0

    # 损失函数处理：优先使用传入的criterion，否则使用默认的ClassBalancedFocalLoss
    if criterion is None:
        print("[Warning] 未传入损失函数，使用默认的ClassBalancedFocalLoss")
        criterion = ClassBalancedFocalLoss(
            beta=0.9999,
            gamma=3.0,
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
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()

    if freeze_bn:
        for module in model_train.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        images, targets = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
                
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(images)

            # ----------------------#
            #   通用损失计算逻辑
            #   自动兼容 Inception 等返回 Tuple 的模型
            # ----------------------#
            if isinstance(outputs, (tuple, list)):
                # 如果输出是元组 (主输出, 辅助输出1, ...)
                main_output = outputs[0]
                aux_outputs = outputs[1:]
                
                # 计算主损失
                loss_value = criterion(main_output, targets)
                
                # 累加辅助损失 (使用配置的权重)
                for aux in aux_outputs:
                    loss_value += aux_loss_weight * criterion(aux, targets)
                    
                # 用于计算准确率的只是主输出
                outputs_for_acc = main_output
            else:
                # 标准单输出模型
                loss_value = criterion(outputs, targets)
                outputs_for_acc = outputs
            loss_value.backward()
            optimizer.step()
        else:
            from torch.amp import autocast
            with autocast('cuda'):
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs = model_train(images)

                # ----------------------#
                #   通用损失计算逻辑
                #   自动兼容 Inception 等返回 Tuple 的模型
                # ----------------------#
                if isinstance(outputs, (tuple, list)):
                    # 如果输出是元组 (主输出, 辅助输出1, ...)
                    main_output = outputs[0]
                    aux_outputs = outputs[1:]
                    
                    # 计算主损失
                    loss_value = criterion(main_output, targets)
                    
                    # 累加辅助损失 (使用配置的权重)
                    for aux in aux_outputs:
                        loss_value += aux_loss_weight * criterion(aux, targets)
                        
                    # 用于计算准确率的只是主输出
                    outputs_for_acc = main_output
                else:
                    # 标准单输出模型
                    loss_value = criterion(outputs, targets)
                    outputs_for_acc = outputs
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss_value.item()
        with torch.no_grad():
            # argmax 直接作用于 logits，无需 softmax
            predictions = torch.argmax(outputs_for_acc, dim=-1)
            # 使用 .float() 保持设备一致性
            accuracy = (predictions == targets).float().mean()
            total_accuracy += accuracy.item()
            
            # 更新训练指标
            train_metrics.update(predictions, targets)

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'accuracy'  : total_accuracy / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)

            outputs = model_train(images)

            # 兼容返回 (main, aux, ...) 的模型，验证阶段与训练阶段保持一致
            if isinstance(outputs, (tuple, list)):
                main_output = outputs[0]
                aux_outputs = outputs[1:]
                loss_value = criterion(main_output, targets)
                for aux in aux_outputs:
                    loss_value += aux_loss_weight * criterion(aux, targets)
                outputs_for_acc = main_output
            else:
                loss_value = criterion(outputs, targets)
                outputs_for_acc = outputs
            
            val_loss    += loss_value.item()
            # argmax 直接作用于 logits，无需 softmax
            predictions = torch.argmax(outputs_for_acc, dim=-1)
            # 使用 .float() 保持设备一致性
            accuracy    = (predictions == targets).float().mean()
            val_accuracy += accuracy.item()
            
            # 更新验证指标
            val_metrics.update(predictions, targets)
            
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'accuracy'  : val_accuracy / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
                
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        loss_history.append_acc(epoch + 1, total_accuracy / epoch_step, val_accuracy / epoch_step_val)
        # 计算详细指标
        train_detailed = train_metrics.compute()
        val_detailed = val_metrics.compute()
        
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        # 动态构建类别准确率输出
        acc_str = "类别准确率 - " + ", ".join([f"{name}: {val_detailed.get(f'{name}_acc', 0):.3f}" for name in class_names])
        print(acc_str)

        # 动态输出少数类别复合分数
        print(f"平衡准确率: {val_detailed.get('balanced_accuracy', 0):.3f}, "
              f"少数类别({class_names[minority_idx]})复合分数: {val_metrics.get_minority_score():.3f}")

        #-----------------------------------------------#
        #   早停和模型保存策略
        #-----------------------------------------------#
        current_val_loss = val_loss / epoch_step_val
        minority_score = val_metrics.get_minority_score()
        balanced_acc = val_detailed.get('balanced_accuracy', 0)

        # 检查早停（使用balanced_accuracy，多类场景更稳定）
        if early_stopping is not None:
            if early_stopping(balanced_acc, model, epoch):
                print("早停触发！")
                return True  # 返回True表示应该停止训练
        
        # 模型检查点(使用动态少数类别索引)
        if model_checkpoint is not None:
            minority_acc_key = f'{class_names[minority_idx]}_acc'
            model_checkpoint(balanced_acc, model, optimizer, epoch,
                           val_loss=current_val_loss,
                           minority_acc=val_detailed.get(minority_acc_key, 0),
                           balanced_acc=val_detailed.get('balanced_accuracy', 0))
        
        # 原有的保存逻辑 - 使用统一的 checkpoint 格式
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                save_path=os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)),
                val_loss=val_loss / epoch_step_val,
                val_acc=val_accuracy / epoch_step_val,
                minority_score=minority_score
            )

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                save_path=os.path.join(save_dir, "best_epoch_weights.pth"),
                val_loss=val_loss / epoch_step_val,
                val_acc=val_accuracy / epoch_step_val,
                minority_score=minority_score
            )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            save_path=os.path.join(save_dir, "last_epoch_weights.pth"),
            val_loss=val_loss / epoch_step_val,
            val_acc=val_accuracy / epoch_step_val,
            minority_score=minority_score
        )
        
    return False  # 返回False表示继续训练
