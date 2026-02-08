"""
多模型ROC/PR曲线对比绘图脚本

功能：
1. 读取多个模型文件夹的detailed_predictions.csv
2. 提取完整概率矩阵和真实标签
3. 计算每个模型的整体AUC/AP（使用宏平均）
4. 绘制多模型对比的ROC和PR曲线

使用方法：
python plot_multi_model_curves.py --folders resnet50_cls_val densenet121_cls_val inception_resnet_v2_cls_val

作者：Claude Code
日期：2026-02-08
"""

import os
import sys
from turtle import st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from itertools import cycle

def get_model_name(name: str) -> str:
    if 'resnet50' in name:
        return 'ResNet50'
    elif 'inception_resnet_v2' in name:
        return 'InceptionResNetV2'
    elif 'xception' in name:
        return 'Xception'
    elif 'vgg19' in name:
        return 'VGG19'
    elif 'densennet121' in name:
        return 'DenseNet121'
    elif 'inception_v3' in name:
        return 'InceptionV3'
    

def load_model_data(folder_path, n_classes):
    """
    从模型文件夹加载预测数据

    参数：
        folder_path: 模型文件夹路径
        n_classes: 类别数量

    返回：
        labels: 真实标签数组 (n_samples,)
        probs: 预测概率矩阵 (n_samples, n_classes)
        model_name: 模型名称
    """
    csv_path = os.path.join(folder_path, "detailed_predictions.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    # 读取CSV
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # 提取标签
    labels = df['true'].values

    # 提取概率矩阵
    # 查找概率列（可能是class_0_prob格式或类别名_probability格式）
    prob_cols = []
    for i in range(n_classes):
        # 尝试两种列名格式
        col_name_1 = f'class_{i}_prob'
        col_name_2 = f'{i}_probability'

        if col_name_1 in df.columns:
            prob_cols.append(col_name_1)
        elif col_name_2 in df.columns:
            prob_cols.append(col_name_2)
        else:
            # 如果找不到，尝试查找包含probability的列
            prob_candidates = [col for col in df.columns if 'prob' in col.lower()]
            if len(prob_candidates) >= n_classes:
                prob_cols = prob_candidates[:n_classes]
                break
            else:
                raise ValueError(
                    f"找不到类别{i}的概率列。\n"
                    f"可用的列: {df.columns.tolist()}\n"
                    f"提示：请确保CSV文件包含 class_0_prob, class_1_prob 等列。\n"
                    f"如果没有，请重新运行 eval.py 生成包含完整概率的CSV文件。"
                )

    probs = df[prob_cols].values

    # 数据验证
    assert probs.shape[1] == n_classes, \
        f"概率矩阵列数不匹配: {probs.shape[1]} != {n_classes}"
    assert len(labels) == len(probs), \
        f"标签和概率数量不匹配: {len(labels)} != {len(probs)}"

    # 验证概率和是否接近1.0（允许一定误差）
    prob_sums = probs.sum(axis=1)
    if not np.allclose(prob_sums, 1.0, atol=0.01):
        print(f"  ⚠ 警告：部分样本的概率和不为1.0 (范围: {prob_sums.min():.3f} - {prob_sums.max():.3f})")

    # 提取模型名称（从文件夹名中提取）
    model_name = os.path.basename(folder_path)
    # 去除常见的后缀
    model_name = model_name.replace('_cls_val', '').replace('_cls_test', '')

    return labels, probs, get_model_name(model_name)


def compute_macro_metrics(labels, probs, n_classes):
    """
    计算宏平均的ROC-AUC和PR-AP

    参数：
        labels: 真实标签数组 (n_samples,)
        probs: 预测概率矩阵 (n_samples, n_classes)
        n_classes: 类别数量

    返回：
        macro_auc: 宏平均AUC
        macro_ap: 宏平均AP
        fpr_macro: 宏平均FPR数组
        tpr_macro: 宏平均TPR数组
        precision_macro: 宏平均Precision数组
        recall_macro: 宏平均Recall数组
    """
    # 标签二值化
    labels_onehot = label_binarize(labels, classes=range(n_classes))
    if labels_onehot.shape[1] == 1:
        # 二分类情况，需要补充第二列
        labels_onehot = np.hstack([1 - labels_onehot, labels_onehot])

    # 计算每类的ROC和PR
    fpr_list = []
    tpr_list = []
    precision_list = []
    recall_list = []
    auc_list = []
    ap_list = []

    for i in range(n_classes):
        # ROC
        fpr, tpr, _ = roc_curve(labels_onehot[:, i], probs[:, i])
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc(fpr, tpr))

        # PR
        precision, recall, _ = precision_recall_curve(
            labels_onehot[:, i],
            probs[:, i]
        )
        precision_list.append(precision)
        recall_list.append(recall)
        ap_list.append(average_precision_score(labels_onehot[:, i], probs[:, i]))

    # 宏平均AUC和AP
    macro_auc = np.mean(auc_list)
    macro_ap = np.mean(ap_list)

    # 计算宏平均曲线（插值到统一的FPR/Recall网格）
    # ROC宏平均
    all_fpr = np.unique(np.concatenate([fpr_list[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_list[i], tpr_list[i])
    mean_tpr /= n_classes
    fpr_macro = all_fpr
    tpr_macro = mean_tpr

    # PR宏平均
    all_recall = np.linspace(0, 1, 100)
    mean_precision = np.zeros_like(all_recall)
    for i in range(n_classes):
        # PR曲线的recall是递减的，需要反转
        mean_precision += np.interp(
            all_recall,
            recall_list[i][::-1],
            precision_list[i][::-1]
        )
    mean_precision /= n_classes
    recall_macro = all_recall
    precision_macro = mean_precision

    return (
        macro_auc,
        macro_ap,
        fpr_macro,
        tpr_macro,
        precision_macro,
        recall_macro
    )


def draw_multi_model_roc(model_data_list, output_path):
    """
    绘制多模型ROC曲线对比图

    参数：
        model_data_list: [(model_name, labels, probs, n_classes), ...]
        output_path: 输出图片路径
    """
    plt.figure(figsize=(8, 6), dpi=300)
    plt.rcParams.update({
        'font.size': 16,
        'font.family': 'Arial',
        'axes.grid': False
    })

    # 科研配色
    colors = cycle([
        '#1f77b4',
        '#d62728',
        '#2ca02c',
        '#bcbd22',
        '#17becf',
        '#9467bd'
    ])

    # 计算模型名称最大长度用于对齐
    max_name_len = max([len(name) for name, _, _, _ in model_data_list])

    # 绘制每个模型的曲线
    for (model_name, labels, probs, n_classes), color in zip(model_data_list, colors):
        macro_auc, _, fpr_macro, tpr_macro, _, _ = compute_macro_metrics(
            labels,
            probs,
            n_classes
        )

        # 构造对齐的图例标签
        label_str = f"{model_name:<{max_name_len}}   AUC = {macro_auc:.3f}"

        plt.plot(
            fpr_macro,
            tpr_macro,
            color=color,
            lw=1.5,
            marker='.',
            markersize=4,
            markevery=0.2,
            label=label_str
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right", prop={'family': 'monospace'}, framealpha=1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 多模型ROC曲线已保存: {output_path}")


def draw_multi_model_pr(model_data_list, output_path):
    """
    绘制多模型PR曲线对比图

    参数：
        model_data_list: [(model_name, labels, probs, n_classes), ...]
        output_path: 输出图片路径
    """
    plt.figure(figsize=(8, 6), dpi=300)
    plt.rcParams.update({
        'font.size': 16,
        'font.family': 'Arial',
        'axes.grid': False
    })

    # 科研配色
    colors = cycle([
        '#1f77b4',
        '#d62728',
        '#2ca02c',
        '#bcbd22',
        '#17becf',
        '#9467bd'
    ])

    # 计算模型名称最大长度用于对齐
    max_name_len = max([len(name) for name, _, _, _ in model_data_list])

    # 绘制每个模型的曲线
    for (model_name, labels, probs, n_classes), color in zip(model_data_list, colors):
        _, macro_ap, _, _, precision_macro, recall_macro = compute_macro_metrics(
            labels,
            probs,
            n_classes
        )

        # 构造对齐的图例标签
        label_str = f"{model_name:<{max_name_len}}   AP = {macro_ap:.3f}"

        plt.plot(
            recall_macro,
            precision_macro,
            color=color,
            lw=1.5,
            marker='.',
            markersize=4,
            markevery=0.2,
            label=label_str
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left", prop={'family': 'monospace'}, framealpha=1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 多模型PR曲线已保存: {output_path}")


def main():
    """主函数"""
    # ==================== 配置参数 ====================
    # 在这里直接配置需要对比的模型文件夹名称
    folders = [
        'resnet50_cls_test',
        'inception_resnet_v2_cls_test'
    ]

    # 其他配置参数
    n_classes = 3   # 类别数量
    output_prefix = 'multi_model'  # 输出文件名前缀
    # =================================================

    # 构建完整路径
    base_metrics_path = '.'

    print("=" * 80)
    print("多模型ROC/PR曲线对比绘图")
    print("=" * 80)
    print(f"\n基础目录: {base_metrics_path}")
    print(f"类别数量: {n_classes}")
    print(f"模型列表: {folders}")

    # 加载所有模型数据
    model_data_list = []
    for folder_name in folders:
        folder_path = os.path.join(base_metrics_path, folder_name)
        print(f"\n正在加载: {folder_name}")

        try:
            labels, probs, model_name = load_model_data(
                folder_path,
                n_classes
            )
            model_data_list.append((model_name, labels, probs, n_classes))
            print(f"  ✓ 成功加载 {len(labels)} 个样本")
            print(f"  模型名称: {model_name}")
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            continue

    if len(model_data_list) == 0:
        print("\n错误：没有成功加载任何模型数据")
        sys.exit(1)

    print(f"\n成功加载 {len(model_data_list)} 个模型")

    # 绘制ROC曲线
    print("\n正在绘制多模型ROC曲线...")
    roc_output = os.path.join(
        base_metrics_path,
        f"{output_prefix}_roc_curves.png"
    )
    draw_multi_model_roc(model_data_list, roc_output)

    # 绘制PR曲线
    print("正在绘制多模型PR曲线...")
    pr_output = os.path.join(
        base_metrics_path,
        f"{output_prefix}_pr_curves.png"
    )
    draw_multi_model_pr(model_data_list, pr_output)

    print("\n" + "=" * 80)
    print("绘图完成！")
    print("=" * 80)



if __name__ == "__main__":
    main()
