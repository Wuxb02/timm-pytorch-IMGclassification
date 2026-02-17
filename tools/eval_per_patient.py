"""
患者级别性能评估脚本

该脚本用于计算患者级别的分类性能指标，通过聚合同一患者的多张图片预测结果。

使用方法:
    在脚本的 main() 函数中配置要处理的文件夹列表，然后直接运行

示例:
    python tools/eval_per_patient.py
"""

import os
import sys
import re
from collections import Counter
import numpy as np
import pandas as pd

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入现有的指标计算函数
from utils.utils_metrics import (
    fast_hist,
    per_class_Recall,
    per_class_Precision,
    compute_auc_metrics,
    compute_specificity,
    compute_bootstrap_ci
)


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
        - 对每个患者的所有图片概率取最大值（max aggregation）
        - 基于最大概率确定预测标签
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

    # 动态添加概率列的聚合方式（取最大值）
    prob_cols = [col for col in df.columns if col.startswith('class_') and col.endswith('_prob')]
    for col in prob_cols:
        agg_dict[col] = 'mean'  # 改为取最大值

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


def create_patient_classification_report(hist, recall, precision, auc_metrics,
                                         specificity, accuracy, ci_results,
                                         class_names, output_path,
                                         samples_per_class=None, minority_idx=None,
                                         top1_acc=None, top5_acc=None,
                                         total_images=None, avg_images_per_patient=None):
    """
    生成患者级别的分类性能报告（仅文本，不生成可视化）

    参数:
        hist: 混淆矩阵
        recall: 各类召回率
        precision: 各类精确度
        auc_metrics: AUC指标字典(per_class_auc, macro_auc, micro_auc)
        specificity: 各类特异度
        accuracy: 各类准确率
        ci_results: 置信区间结果
        class_names: 类别名称列表
        output_path: 输出目录
        samples_per_class: 各类别患者数量(可选)
        minority_idx: 少数类别索引,如果为None则自动识别
        top1_acc: Top-1准确率(可选)
        top5_acc: Top-5准确率(可选)
        total_images: 图像总数(可选)
        avg_images_per_patient: 每患者平均图片数(可选)

    返回:
        f1_scores: F1分数列表
    """
    # 如果未指定,自动识别少数类别
    if minority_idx is None:
        minority_idx = np.argmin(recall)

    # 计算F1分数
    f1_scores = []
    for i in range(len(class_names)):
        if precision[i] + recall[i] > 0:
            f1 = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1 = 0.0
        f1_scores.append(f1)

    # 生成文本报告
    report_path = os.path.join(output_path, "classification_patient_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Patient-Level Classification Performance Report\n")
        f.write("=" * 80 + "\n\n")

        # Part 0: Dataset Statistics and Basic Performance Summary
        if samples_per_class is not None:
            f.write("0. Dataset Statistics\n")
            f.write("-" * 80 + "\n\n")

            # 处理列表或字典类型的samples_per_class
            if isinstance(samples_per_class, list):
                total_samples = sum(samples_per_class)
                f.write(f"Total Patients: {total_samples}\n")
                if total_images is not None:
                    f.write(f"Total Images: {total_images}\n")
                if avg_images_per_patient is not None:
                    f.write(f"Average Images per Patient: {avg_images_per_patient:.2f}\n")
                f.write("\n")
                f.write("Patient Class Distribution:\n")
                for i, name in enumerate(class_names):
                    count = samples_per_class[i] if i < len(samples_per_class) else 0
                    ratio = count / total_samples * 100 if total_samples > 0 else 0
                    minority_marker = " ← Minority Class" if i == minority_idx else ""
                    f.write(f"  - {name} ({i}): {count} patients ({ratio:.1f}%){minority_marker}\n")
            else:  # 字典类型
                total_samples = sum(samples_per_class.values())
                f.write(f"Total Patients: {total_samples}\n")
                if total_images is not None:
                    f.write(f"Total Images: {total_images}\n")
                if avg_images_per_patient is not None:
                    f.write(f"Average Images per Patient: {avg_images_per_patient:.2f}\n")
                f.write("\n")
                f.write("Patient Class Distribution:\n")
                for i, name in enumerate(class_names):
                    count = samples_per_class.get(i, 0)
                    ratio = count / total_samples * 100 if total_samples > 0 else 0
                    minority_marker = " ← Minority Class" if i == minority_idx else ""
                    f.write(f"  - {name} ({i}): {count} patients ({ratio:.1f}%){minority_marker}\n")
            f.write("\n")

        # Basic Performance Summary
        if top1_acc is not None or top5_acc is not None:
            f.write("Basic Performance Summary\n")
            f.write("-" * 80 + "\n\n")

            if top1_acc is not None:
                f.write(f"Top-1 Accuracy:       {top1_acc*100:.2f}%\n")
            if top5_acc is not None:
                f.write(f"Top-5 Accuracy:       {top5_acc*100:.2f}%\n")
            f.write(f"Mean Recall:          {np.mean(recall)*100:.2f}%\n")
            f.write(f"Mean Precision:       {np.mean(precision)*100:.2f}%\n")
            f.write(f"Macro F1 Score:       {np.mean(f1_scores)*100:.2f}%\n")
            f.write(f"Balanced Accuracy:    {np.mean(recall)*100:.2f}%\n")
            f.write("\n")

        # Part I: Detailed Per-Class Metrics (with 95% CI)
        f.write("I. Detailed Per-Class Metrics (with 95% CI)\n")
        f.write("-" * 80 + "\n\n")

        for i, name in enumerate(class_names):
            f.write(f"{i+1}. {name}:\n")
            f.write(f"   Accuracy:         {accuracy[i]:.4f} "
                   f"(95% CI: [{ci_results['accuracy'][i]['lower']:.4f}, "
                   f"{ci_results['accuracy'][i]['upper']:.4f}])\n")
            f.write(f"   Precision:        {precision[i]:.4f} "
                   f"(95% CI: [{ci_results['precision'][i]['lower']:.4f}, "
                   f"{ci_results['precision'][i]['upper']:.4f}])\n")
            f.write(f"   Recall:           {recall[i]:.4f} "
                   f"(95% CI: [{ci_results['recall'][i]['lower']:.4f}, "
                   f"{ci_results['recall'][i]['upper']:.4f}])\n")
            f.write(f"   Specificity:      {specificity[i]:.4f} "
                   f"(95% CI: [{ci_results['specificity'][i]['lower']:.4f}, "
                   f"{ci_results['specificity'][i]['upper']:.4f}])\n")
            f.write(f"   F1 Score:         {f1_scores[i]:.4f} "
                   f"(95% CI: [{ci_results['f1'][i]['lower']:.4f}, "
                   f"{ci_results['f1'][i]['upper']:.4f}])\n")
            f.write(f"   AUC:              {auc_metrics['per_class_auc'][i]:.4f} "
                   f"(95% CI: [{ci_results['auc'][i]['lower']:.4f}, "
                   f"{ci_results['auc'][i]['upper']:.4f}])\n")
            f.write("\n")

        # Part II: Overall performance metrics
        f.write("II. Overall Performance\n")
        f.write("-" * 80 + "\n\n")

        # Overall Accuracy (all samples)
        f.write(f"Overall Accuracy:     {ci_results['overall_accuracy']['mean']:.4f} "
               f"(95% CI: [{ci_results['overall_accuracy']['lower']:.4f}, "
               f"{ci_results['overall_accuracy']['upper']:.4f}])\n\n")

        # Macro-average (5 core metrics)
        f.write(f"Macro-average:\n")
        f.write(f"   Accuracy:     {np.mean(accuracy):.4f} "
               f"(95% CI: [{ci_results['macro_accuracy']['lower']:.4f}, "
               f"{ci_results['macro_accuracy']['upper']:.4f}])\n")
        f.write(f"   Precision:    {np.mean(precision):.4f} "
               f"(95% CI: [{ci_results['macro_precision']['lower']:.4f}, "
               f"{ci_results['macro_precision']['upper']:.4f}])\n")
        f.write(f"   Recall:       {np.mean(recall):.4f} "
               f"(95% CI: [{ci_results['macro_recall']['lower']:.4f}, "
               f"{ci_results['macro_recall']['upper']:.4f}])\n")
        f.write(f"   Specificity:  {np.mean(specificity):.4f} "
               f"(95% CI: [{ci_results['macro_specificity']['lower']:.4f}, "
               f"{ci_results['macro_specificity']['upper']:.4f}])\n")
        f.write(f"   F1 Score:     {np.mean(f1_scores):.4f} "
               f"(95% CI: [{ci_results['macro_f1']['lower']:.4f}, "
               f"{ci_results['macro_f1']['upper']:.4f}])\n")
        f.write(f"   AUC:          {auc_metrics['macro_auc']:.4f} "
               f"(95% CI: [{ci_results['macro_auc']['lower']:.4f}, "
               f"{ci_results['macro_auc']['upper']:.4f}])\n\n")

        # Micro-average AUC
        f.write(f"Micro-average:\n")

        # 在多分类中，Micro-average的Precision、Recall、F1都等于Overall Accuracy
        micro_precision = ci_results['overall_accuracy']['mean']
        micro_recall = ci_results['overall_accuracy']['mean']
        micro_f1 = ci_results['overall_accuracy']['mean']

        f.write(f"   Precision:    {micro_precision:.4f} "
               f"(95% CI: [{ci_results['overall_accuracy']['lower']:.4f}, "
               f"{ci_results['overall_accuracy']['upper']:.4f}])\n")
        f.write(f"   Recall:       {micro_recall:.4f} "
               f"(95% CI: [{ci_results['overall_accuracy']['lower']:.4f}, "
               f"{ci_results['overall_accuracy']['upper']:.4f}])\n")
        f.write(f"   F1 Score:     {micro_f1:.4f} "
               f"(95% CI: [{ci_results['overall_accuracy']['lower']:.4f}, "
               f"{ci_results['overall_accuracy']['upper']:.4f}])\n")
        f.write(f"   AUC:          {auc_metrics['micro_auc']:.4f} "
               f"(95% CI: [{ci_results['micro_auc']['lower']:.4f}, "
               f"{ci_results['micro_auc']['upper']:.4f}])\n\n")

        f.write("Note: In multi-class classification, Micro-average Precision, Recall, and F1 "
               "are mathematically equivalent to Overall Accuracy.\n\n")

        # Balanced Accuracy and Performance Gap
        f.write(f"Balanced Accuracy:    {np.mean(recall):.4f}\n")
        f.write(f"Performance Gap (F1): {max(f1_scores) - min(f1_scores):.4f}\n\n")

        # Part III: Key findings and performance analysis
        f.write("III. Key Findings and Analysis\n")
        f.write("-" * 80 + "\n\n")

        minority_class_idx = minority_idx
        minority_f1 = f1_scores[minority_class_idx]
        minority_recall = recall[minority_class_idx]
        minority_precision = precision[minority_class_idx]
        minority_auc = auc_metrics['per_class_auc'][minority_class_idx]
        minority_specificity = specificity[minority_class_idx]

        f.write(f"Minority Class ({class_names[minority_class_idx]}) Performance:\n\n")

        # 详细指标及置信区间
        f.write(f"   Precision:    {minority_precision:.4f} "
               f"(95% CI: [{ci_results['precision'][minority_class_idx]['lower']:.4f}, "
               f"{ci_results['precision'][minority_class_idx]['upper']:.4f}])\n")
        f.write(f"   Recall:       {minority_recall:.4f} "
               f"(95% CI: [{ci_results['recall'][minority_class_idx]['lower']:.4f}, "
               f"{ci_results['recall'][minority_class_idx]['upper']:.4f}])\n")
        f.write(f"   F1 Score:     {minority_f1:.4f} "
               f"(95% CI: [{ci_results['f1'][minority_class_idx]['lower']:.4f}, "
               f"{ci_results['f1'][minority_class_idx]['upper']:.4f}])\n")
        f.write(f"   AUC:          {minority_auc:.4f} "
               f"(95% CI: [{ci_results['auc'][minority_class_idx]['lower']:.4f}, "
               f"{ci_results['auc'][minority_class_idx]['upper']:.4f}])\n")
        f.write(f"   Specificity:  {minority_specificity:.4f} "
               f"(95% CI: [{ci_results['specificity'][minority_class_idx]['lower']:.4f}, "
               f"{ci_results['specificity'][minority_class_idx]['upper']:.4f}])\n\n")

        # Performance Assessment
        f.write("Performance Assessment:\n")

        # F1 Score评估
        if minority_f1 >= 0.8:
            f1_assessment = "Excellent"
        elif minority_f1 >= 0.6:
            f1_assessment = "Good"
        elif minority_f1 >= 0.4:
            f1_assessment = "Moderate"
        else:
            f1_assessment = "Poor"
        f.write(f"   F1 Score:  {minority_f1:.4f} - {f1_assessment}\n")

        # Recall评估
        if minority_recall >= 0.9:
            recall_assessment = "Very low false negative risk"
        elif minority_recall >= 0.7:
            recall_assessment = "Low false negative risk"
        elif minority_recall >= 0.5:
            recall_assessment = "Moderate false negative risk"
        else:
            recall_assessment = "High false negative risk"
        f.write(f"   Recall:    {minority_recall:.4f} - {recall_assessment}\n")

        # AUC评估
        if minority_auc >= 0.9:
            auc_assessment = "Excellent discrimination"
        elif minority_auc >= 0.8:
            auc_assessment = "Good discrimination"
        elif minority_auc >= 0.7:
            auc_assessment = "Acceptable discrimination"
        else:
            auc_assessment = "Poor discrimination"
        f.write(f"   AUC:       {minority_auc:.4f} - {auc_assessment}\n\n")

        # Deployment Recommendations
        f.write("Deployment Recommendations:\n")
        if minority_f1 >= 0.7 and minority_recall >= 0.7 and minority_auc >= 0.8:
            f.write("   ✓ Excellent - Recommended for clinical use\n")
        elif minority_f1 >= 0.5 and minority_recall >= 0.5:
            f.write("   ⚠ Good - Recommended for supervised use\n")
        elif minority_f1 >= 0.3:
            f.write("   ⚠ Moderate - Requires careful supervision\n")
        else:
            f.write("   ✗ Poor - Not recommended for deployment\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"✓ 患者级别性能报告已保存: {report_path}")
    return f1_scores


def evaluate_patient_level(metrics_folder: str):
    """
    对指定的 metrics_out 子文件夹进行患者级别评估

    参数:
        metrics_folder: 例如 "inception_resnet_v2_cls_test"
    """
    # 获取项目根目录（tools 的父目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 构建路径
    base_path = os.path.join(project_root, "metrics_out")
    folder_path = os.path.join(base_path, metrics_folder)
    csv_path = os.path.join(folder_path, "detailed_predictions.csv")

    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: 文件不存在: {csv_path}")
        print(f"\n可用的文件夹:")
        if os.path.exists(base_path):
            folders = [f for f in os.listdir(base_path)
                      if os.path.isdir(os.path.join(base_path, f))]
            for folder in folders:
                print(f"  - {folder}")
        return

    print("=" * 80)
    print(f"患者级别评估: {metrics_folder}")
    print("=" * 80)

    # 1. 读取 detailed_predictions.csv
    print(f"\n正在读取: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"图像总数: {len(df)}")

    # 2. 读取类别名称
    classes_path = os.path.join(project_root, "model_data", "cls_classes.txt")
    with open(classes_path, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    num_classes = len(class_names)
    print(f"类别数量: {num_classes}")
    print(f"类别名称: {class_names}")

    # 3. 聚合患者级别数据
    print("\n正在聚合患者级别数据...")
    patient_df, total_images, image_counts = aggregate_patient_predictions(df, num_classes)
    num_patients = len(patient_df)
    avg_images_per_patient = total_images / num_patients

    print(f"患者总数: {num_patients}")
    print(f"每患者平均图片数: {avg_images_per_patient:.2f}")

    # 4. 准备数据
    labels = patient_df['true'].values
    preds = patient_df['predict'].values
    prob_cols = [col for col in patient_df.columns
                 if col.startswith('class_') and col.endswith('_prob')]
    probs = patient_df[prob_cols].values

    # 5. 计算指标
    print("\n正在计算性能指标...")

    # 混淆矩阵
    hist = fast_hist(labels, preds, num_classes)

    # Recall 和 Precision
    recall = per_class_Recall(hist)
    precision = per_class_Precision(hist)

    # AUC
    print("  计算AUC指标...")
    auc_metrics = compute_auc_metrics(labels, probs, num_classes)

    # Specificity
    print("  计算特异度...")
    specificity = compute_specificity(hist)

    # Accuracy (per-class)
    accuracy = np.zeros(num_classes)
    for i in range(num_classes):
        TP = hist[i, i]
        TN = hist.sum() - hist[i, :].sum() - hist[:, i].sum() + hist[i, i]
        accuracy[i] = (TP + TN) / hist.sum()

    # Bootstrap 置信区间
    print("  计算Bootstrap 95%置信区间（1000次重采样）...")
    ci_results = compute_bootstrap_ci(labels, preds, probs,
                                     n_bootstrap=1000, ci=95,
                                     random_state=42)

    # 6. 统计每类患者数量
    samples_per_class = [np.sum(labels == i) for i in range(num_classes)]
    minority_idx = np.argmin(samples_per_class)

    # 7. 计算 Top-1 和 Top-5 准确率
    top1_acc = np.sum(preds == labels) / len(labels)

    # Top-5 (如果类别数>=5)
    if num_classes >= 5:
        top5_preds = np.argsort(probs, axis=1)[:, ::-1][:, :5]
        top5_acc = sum(labels[i] in top5_preds[i] for i in range(len(labels))) / len(labels)
    else:
        top5_acc = 1.0  # 如果类别数<5，Top-5准确率为100%

    # 8. 生成报告
    print("\n正在生成患者级别性能报告...")
    f1_scores = create_patient_classification_report(
        hist, recall, precision, auc_metrics, specificity,
        accuracy, ci_results, class_names, folder_path,
        samples_per_class=samples_per_class, minority_idx=minority_idx,
        top1_acc=top1_acc, top5_acc=top5_acc,
        total_images=total_images, avg_images_per_patient=avg_images_per_patient
    )

    # 9. 打印总结
    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)
    print(f"\n患者级别统计:")
    print(f"  患者总数: {num_patients}")
    print(f"  图像总数: {total_images}")
    print(f"  每患者平均图片数: {avg_images_per_patient:.2f}")
    print(f"\n性能指标:")
    print(f"  Top-1 准确率: {top1_acc*100:.2f}%")
    print(f"  平均召回率: {np.mean(recall)*100:.2f}%")
    print(f"  平均精确度: {np.mean(precision)*100:.2f}%")
    print(f"  Macro F1: {np.mean(f1_scores)*100:.2f}%")
    print(f"  Macro AUC: {auc_metrics['macro_auc']:.4f}")
    print(f"\n少数类别 ({class_names[minority_idx]}) 性能:")
    print(f"  F1分数: {f1_scores[minority_idx]:.4f}")
    print(f"  召回率: {recall[minority_idx]:.4f}")
    print(f"  AUC: {auc_metrics['per_class_auc'][minority_idx]:.4f}")


def main():
    """
    主函数：批量处理多个模型文件夹

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
    # ==============================

    print("=" * 80)
    print("患者级别性能评估")
    print("=" * 80)
    print(f"\n将处理 {len(metrics_folders)} 个模型文件夹:")
    for folder in metrics_folders:
        print(f"  - {folder}")
    print()

    # 处理每个指定的文件夹
    for i, folder in enumerate(metrics_folders):
        if i > 0:
            print("\n\n")
        try:
            evaluate_patient_level(folder)
        except Exception as e:
            print(f"\n错误: 处理 {folder} 时发生异常: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("所有评估完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
