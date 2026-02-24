"""
患者级别多模型ROC/PR曲线对比绘图脚本

功能：
1. 读取多个模型文件夹的detailed_predictions.csv
2. 将图像级别预测聚合为患者级别（使用平均值策略）
3. 计算每个模型的患者级别整体AUC/AP（使用宏平均）
4. 绘制多模型对比的患者级别ROC和PR曲线

使用方法：
    在脚本的 main() 函数中配置要处理的文件夹列表，然后直接运行

示例:
    python tools/plot_patient_roc_pr.py

注意：
    本脚本使用mean聚合策略（平均概率），与eval_per_patient.py保持一致，
    因此生成的AUC值将与classification_patient_report.txt中的值完全相同。

作者：Claude Code
日期：2026-02-24
"""

import os
import sys
import re
from collections import Counter
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

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def get_model_name(name: str) -> str:
    """
    将文件夹名称转换为友好的模型名称

    参数:
        name: 文件夹名称，例如 "resnet50_cls_val"

    返回:
        友好的模型名称，例如 "ResNet50"
    """
    if 'resnet50' in name:
        return 'ResNet50'
    elif 'inception_resnet_v2' in name:
        return 'InceptionResNetV2'
    elif 'xception' in name:
        return 'Xception'
    elif 'vgg19' in name:
        return 'VGG19'
    elif 'densenet121' in name:
        return 'DenseNet121'
    elif 'tf_inception_v3' in name or 'inception_v3' in name:
        return 'InceptionV3'
    else:
        # 默认返回原始名称（去除后缀）
        return name.replace('_cls_val', '').replace('_cls_test', '')


def extract_patient_name(path: str) -> str:
    """
    从图片路径中提取患者标识

    参数:
        path: 图片路径
            - 中文文件名示例: "datasets\\test\\0\\102【】123林华娣20250608_111306_GYN_35.jpg"
            - 数字文件名示例: "datasets\\test\\0\\4106861_m9_4106861_7.3_1.jpg"

    返回:
        患者标识
            - 有中文字符时: 返回中文姓名，例如 "林华娣"
            - 无中文字符时: 返回第一个下划线前的数字，例如 "4106861"
    """
    filename = os.path.basename(path)
    chinese_chars = re.findall(r'[\u4e00-\u9fff]+', filename)

    if chinese_chars:
        # 有中文字符，拼接所有中文作为患者标识
        return ''.join(chinese_chars)
    else:
        # 没有中文字符，提取第一个下划线前的内容作为患者标识
        if '_' in filename:
            patient_id = filename.split('_')[0]
            return patient_id
        else:
            # 如果没有下划线，使用整个文件名（去掉扩展名）
            patient_id = os.path.splitext(filename)[0]
            print(f"警告: 文件名格式异常，使用完整文件名作为患者标识: {filename}")
            return patient_id


def aggregate_patient_predictions(df: pd.DataFrame, num_classes: int) -> tuple:
    """
    将图像级别预测聚合为患者级别

    参数:
        df: detailed_predictions.csv 的 DataFrame
        num_classes: 类别数量

    返回:
        patient_df: 患者级别的 DataFrame
        total_images: 图像总数
        image_counts: 每个患者的图片数量字典

    聚合方式：
        - 对每个患者的所有图片概率取平均值（mean aggregation）
        - 基于平均概率确定预测标签
        - 与eval_per_patient.py保持一致
    """
    # 提取患者标识
    df['patient_name'] = df['path'].apply(extract_patient_name)

    # 检查同一患者的标签一致性
    patient_labels = df.groupby('patient_name')['true'].apply(list)
    for patient, labels in patient_labels.items():
        unique_labels = set(labels)
        if len(unique_labels) > 1:
            # 使用众数
            most_common = Counter(labels).most_common(1)[0][0]
            print(f"警告: 患者 '{patient}' 的标签不一致: {labels}, 使用众数: {most_common}")

    # 聚合逻辑
    agg_dict = {'true': lambda x: Counter(x).most_common(1)[0][0]}  # 使用众数

    # 动态添加概率列的聚合方式（取平均值，与eval_per_patient.py保持一致）
    prob_cols = [col for col in df.columns if col.startswith('class_') and col.endswith('_prob')]
    for col in prob_cols:
        agg_dict[col] = 'mean'  # 使用mean与eval_per_patient.py保持一致

    # 按患者分组并聚合
    patient_df = df.groupby('patient_name').agg(agg_dict).reset_index()

    # 计算每个患者的图片数量
    image_counts = df.groupby('patient_name').size().to_dict()
    patient_df['image_count'] = patient_df['patient_name'].map(image_counts)

    # 基于最大概率确定预测标签
    prob_matrix = patient_df[prob_cols].values
    patient_df['predict'] = np.argmax(prob_matrix, axis=1)

    total_images = len(df)

    return patient_df, total_images, image_counts


def load_patient_level_data(folder_path: str, n_classes: int):
    """
    从模型文件夹加载预测数据并聚合为患者级别

    参数：
        folder_path: 模型文件夹路径
        n_classes: 类别数量

    返回：
        labels: 患者级别真实标签数组 (n_patients,)
        probs: 患者级别预测概率矩阵 (n_patients, n_classes)
        model_name: 模型名称
        num_patients: 患者数量
        num_images: 图像数量
    """
    csv_path = os.path.join(folder_path, "detailed_predictions.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    # 读取CSV
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # 聚合为患者级别
    patient_df, total_images, _ = aggregate_patient_predictions(df, n_classes)

    # 提取标签
    labels = patient_df['true'].values

    # 提取概率矩阵
    prob_cols = [col for col in patient_df.columns
                 if col.startswith('class_') and col.endswith('_prob')]
    probs = patient_df[prob_cols].values

    # 获取模型名称
    folder_name = os.path.basename(folder_path)
    model_name = get_model_name(folder_name)

    num_patients = len(patient_df)

    return labels, probs, model_name, num_patients, total_images


def draw_multi_model_roc(model_data_list: list, output_path: str):
    """
    绘制多模型患者级别ROC曲线对比图（严格参考plot_multi_model_curves.py）

    参数：
        model_data_list: 模型数据列表，每个元素为 (labels, probs, model_name, macro_auc)
        output_path: 输出文件路径
    """
    from sklearn.metrics import roc_auc_score

    plt.figure(figsize=(8, 6), dpi=300)
    plt.rcParams.update({
        'font.size': 16,
        'font.family': 'Arial',
        'axes.grid': False
    })

    # 定义颜色循环
    colors = cycle(['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b'])

    # 计算模型名称最大长度，用于图例对齐
    max_name_len = max([len(data[2]) for data in model_data_list])

    # 绘制每个模型的ROC曲线
    for (labels, probs, model_name, _), color in zip(model_data_list, colors):
        n_classes = probs.shape[1]

        # One-hot编码标签
        labels_onehot = label_binarize(labels, classes=range(n_classes))
        if labels_onehot.shape[1] == 1:
            labels_onehot = np.hstack([1 - labels_onehot, labels_onehot])

        # 计算Macro-average ROC曲线（用于绘图）
        all_fpr = np.unique(np.concatenate([
            roc_curve(labels_onehot[:, i], probs[:, i])[0]
            for i in range(n_classes)
        ]))

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(labels_onehot[:, i], probs[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)

        mean_tpr /= n_classes

        # 使用sklearn的标准方法计算Macro AUC（与eval_per_patient.py一致）
        macro_auc_value = roc_auc_score(labels_onehot, probs,
                                        average='macro', multi_class='ovr')

        # 构造对齐的图例标签
        label_str = f"{model_name:<{max_name_len}}   AUC = {macro_auc_value:.3f}"

        # 绘制曲线
        plt.plot(all_fpr, mean_tpr, color=color, lw=2,
                 marker='.', markersize=4, markevery=0.2,
                 label=label_str)

    # 装饰图像
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # 图例设置
    plt.legend(loc="lower right",
               prop={'family': 'monospace'},
               framealpha=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 患者级别ROC曲线已保存: {output_path}")


def draw_multi_model_pr(model_data_list: list, output_path: str):
    """
    绘制多模型患者级别PR曲线对比图（严格参考plot_multi_model_curves.py）

    参数：
        model_data_list: 模型数据列表，每个元素为 (labels, probs, model_name, macro_ap)
        output_path: 输出文件路径
    """
    plt.figure(figsize=(8, 6), dpi=300)
    plt.rcParams.update({
        'font.size': 16,
        'font.family': 'Arial',
        'axes.grid': False
    })

    # 定义颜色循环
    colors = cycle(['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b'])

    # 计算模型名称最大长度，用于图例对齐
    max_name_len = max([len(data[2]) for data in model_data_list])

    # 绘制每个模型的PR曲线
    for (labels, probs, model_name, macro_ap), color in zip(model_data_list, colors):
        n_classes = probs.shape[1]

        # One-hot编码标签
        labels_onehot = label_binarize(labels, classes=range(n_classes))
        if labels_onehot.shape[1] == 1:
            labels_onehot = np.hstack([1 - labels_onehot, labels_onehot])

        # 计算Macro-average PR
        all_recall = np.linspace(0, 1, 100)
        mean_precision = np.zeros_like(all_recall)

        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(
                labels_onehot[:, i], probs[:, i])
            mean_precision += np.interp(all_recall, recall[::-1], precision[::-1])

        mean_precision /= n_classes

        # 计算Macro AP
        macro_ap_value = np.mean([
            average_precision_score(labels_onehot[:, i], probs[:, i])
            for i in range(n_classes)
        ])

        # 构造对齐的图例标签
        label_str = f"{model_name:<{max_name_len}}   AP = {macro_ap_value:.3f}"

        # 绘制曲线
        plt.plot(all_recall, mean_precision, color=color, lw=2,
                 marker='.', markersize=4, markevery=0.2,
                 label=label_str)

    # 装饰图像
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # 图例设置
    plt.legend(loc="lower left",
               prop={'family': 'monospace'},
               framealpha=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 患者级别PR曲线已保存: {output_path}")


def main():
    """
    主函数：批量处理多个模型文件夹，绘制患者级别多模型对比曲线

    使用方法：
        1. 在下面的 metrics_folders 列表中添加要处理的文件夹名称
        2. 直接运行此脚本
    """
    # ========== 配置区域 ==========
    # 在此列表中添加要处理的 metrics_out 子文件夹名称
    metrics_folders = [
        'densenet121_cls_val',
        'resnet50_cls_val',
        'vgg19_bn_cls_val',
        'tf_inception_v3_cls_val',
        'xception_cls_val',
        'inception_resnet_v2_cls_val',
    ]

    # 输出文件前缀
    output_prefix = "patient_multi_model"
    # ==============================

    print("=" * 80)
    print("患者级别多模型ROC/PR曲线对比绘制")
    print("=" * 80)
    print(f"\n将处理 {len(metrics_folders)} 个模型文件夹:")
    for folder in metrics_folders:
        print(f"  - {folder}")
    print()

    # 获取项目根目录和metrics_out路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_metrics_path = os.path.join(project_root, "metrics_out")

    # 读取类别数量
    classes_path = os.path.join(project_root, "model_data", "cls_classes.txt")
    with open(classes_path, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    n_classes = len(class_names)
    print(f"类别数量: {n_classes}")
    print(f"类别名称: {class_names}\n")

    # 加载所有模型数据
    model_data_list = []
    print("正在加载模型数据...")
    print("-" * 80)

    for folder in metrics_folders:
        folder_path = os.path.join(base_metrics_path, folder)
        print(f"\n处理: {folder}")

        try:
            labels, probs, model_name, num_patients, num_images = load_patient_level_data(
                folder_path, n_classes
            )

            # 计算Macro AUC/AP（用于显示，使用sklearn标准方法）
            from sklearn.metrics import roc_auc_score

            labels_onehot = label_binarize(labels, classes=range(n_classes))
            if labels_onehot.shape[1] == 1:
                labels_onehot = np.hstack([1 - labels_onehot, labels_onehot])

            # 使用sklearn标准方法计算Macro AUC（与eval_per_patient.py一致）
            macro_auc = roc_auc_score(labels_onehot, probs,
                                      average='macro', multi_class='ovr')

            # 计算Macro AP
            macro_ap = np.mean([
                average_precision_score(labels_onehot[:, i], probs[:, i])
                for i in range(n_classes)
            ])

            model_data_list.append((labels, probs, model_name, macro_auc))

            print(f"  ✓ 加载成功")
            print(f"  模型名称: {model_name}")
            print(f"  患者数量: {num_patients}")
            print(f"  图像数量: {num_images}")
            print(f"  Macro AUC: {macro_auc:.3f}")
            print(f"  Macro AP: {macro_ap:.3f}")

        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(model_data_list) == 0:
        print("\n错误：没有成功加载任何模型数据")
        sys.exit(1)

    print(f"\n成功加载 {len(model_data_list)} 个模型")

    # 绘制ROC曲线
    print("\n正在绘制患者级别多模型ROC曲线...")
    roc_output = os.path.join(
        base_metrics_path,
        f"{output_prefix}_roc_curves.png"
    )
    draw_multi_model_roc(model_data_list, roc_output)

    # 绘制PR曲线
    print("正在绘制患者级别多模型PR曲线...")
    pr_output = os.path.join(
        base_metrics_path,
        f"{output_prefix}_pr_curves.png"
    )
    draw_multi_model_pr(model_data_list, pr_output)

    print("\n" + "=" * 80)
    print("绘图完成！")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  - {roc_output}")
    print(f"  - {pr_output}")


if __name__ == "__main__":
    main()
