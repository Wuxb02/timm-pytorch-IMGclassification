# -*- coding: utf-8 -*-
"""
双模型多指标统计比较工具

使用配对Bootstrap方法比较两个深度学习模型的性能差异
支持多种整体评估指标(Macro/Micro AUC, Accuracy, Precision, Recall, F1)
生成统计显著性检验、置信区间和专业可视化报告

！！！两个CSV必须包含完全相同的样本(相同顺序、相同真实标签)

日期: 2025-11-28
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)
from sklearn.utils import resample
from tqdm import tqdm


# ==================== 数据加载与验证 ====================

def load_and_validate_predictions(csv_path1, csv_path2, model_name1="Model_A", model_name2="Model_B"):
    """
    加载并验证两个模型的预测CSV文件

    参数:
        csv_path1: 模型1的detailed_predictions.csv路径
        csv_path2: 模型2的detailed_predictions.csv路径
        model_name1: 模型1显示名称
        model_name2: 模型2显示名称

    返回:
        (df1, df2, class_names): 两个DataFrame和类别名称列表

    异常:
        FileNotFoundError: CSV文件不存在
        ValueError: 数据格式不符合要求
    """
    print("\n" + "=" * 60)
    print("双模型多指标比较工具")
    print("=" * 60)
    print(f"\n[1/6] 加载数据...")

    # 1. 文件存在性检查
    if not os.path.exists(csv_path1):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path1}")
    if not os.path.exists(csv_path2):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path2}")

    # 2. 加载CSV
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)
    print(f"  ✓ 模型1: {model_name1} ({len(df1)}样本)")
    print(f"  ✓ 模型2: {model_name2} ({len(df2)}样本)")

    # 3. 列完整性检查
    required_cols = {'path', 'true', 'predict'}
    missing1 = required_cols - set(df1.columns)
    missing2 = required_cols - set(df2.columns)

    if missing1:
        raise ValueError(f"模型1 CSV缺少必需列: {missing1}")
    if missing2:
        raise ValueError(f"模型2 CSV缺少必需列: {missing2}")

    # 4. 样本数量一致性检查
    if len(df1) != len(df2):
        raise ValueError(
            f"样本数量不一致: 模型1={len(df1)}, 模型2={len(df2)}\n"
            f"配对比较要求两个模型在完全相同的测试集上评估"
        )

    # 5. 真实标签一致性检查(逐行比对)
    labels1 = df1['true'].values
    labels2 = df2['true'].values

    if not np.array_equal(labels1, labels2):
        mismatch_count = (labels1 != labels2).sum()
        raise ValueError(
            f"真实标签不一致,发现{mismatch_count}个不匹配样本\n"
            f"配对比较要求两个CSV的'true'列必须完全相同"
        )

    # 6. 识别类别数量(从概率列推断)
    prob_cols = [col for col in df1.columns if
                 col.endswith('_probability') or '_prob' in col]

    if not prob_cols:
        raise ValueError(
            f"未找到概率列。期望列名格式:\n"
            f"  - {{class_name}}_probability (如 normal_probability)\n"
            f"  - class_{{i}}_prob (如 class_0_prob)\n"
            f"可用列: {df1.columns.tolist()}"
        )

    # 提取类别名称
    class_names = []
    for col in sorted(prob_cols):
        if col.endswith('_probability'):
            class_name = col.replace('_probability', '')
            class_names.append(class_name)
        elif '_prob' in col and col.startswith('class_'):
            # class_0_prob -> 0
            class_idx = col.split('_')[1]
            class_names.append(class_idx)

    # 去重并排序(确保顺序一致)
    class_names = sorted(set(class_names), key=lambda x: int(x) if x.isdigit() else x)

    if len(class_names) < 2:
        raise ValueError(f"类别数量必须≥2,当前识别到{len(class_names)}个类别")

    print(f"  ✓ 数据验证通过")

    return df1, df2, class_names


# ==================== 概率列智能提取 ====================

def extract_probability_columns(df, n_classes):
    """
    智能识别并提取预测概率列

    参数:
        df: 预测结果DataFrame
        n_classes: 类别数量

    返回:
        probs: (n_samples, n_classes)概率矩阵

    异常:
        ValueError: 无法识别概率列
    """
    # 策略1: 匹配 {class_name}_probability
    prob_cols_strategy1 = [col for col in df.columns if col.endswith('_probability')]

    if len(prob_cols_strategy1) == n_classes:
        prob_cols = sorted(prob_cols_strategy1)
        probs = df[prob_cols].values

        # 验证概率和
        prob_sums = probs.sum(axis=1)
        if np.allclose(prob_sums, 1.0, atol=1e-3):
            return probs
        else:
            warnings.warn(
                f"概率和不为1(范围:{prob_sums.min():.4f}-{prob_sums.max():.4f}),已自动归一化"
            )
            return probs / prob_sums[:, np.newaxis]

    # 策略2: 匹配 class_{i}_prob
    prob_cols_strategy2 = [f'class_{i}_prob' for i in range(n_classes)]

    if all(col in df.columns for col in prob_cols_strategy2):
        probs = df[prob_cols_strategy2].values

        prob_sums = probs.sum(axis=1)
        if np.allclose(prob_sums, 1.0, atol=1e-3):
            return probs
        else:
            warnings.warn(
                f"概率和不为1(范围:{prob_sums.min():.4f}-{prob_sums.max():.4f}),已自动归一化"
            )
            return probs / prob_sums[:, np.newaxis]

    # 失败: 抛出详细错误
    raise ValueError(
        f"无法识别概率列。期望{n_classes}列,但未匹配成功\n"
        f"可用列: {df.columns.tolist()}\n"
        f"支持格式:\n"
        f"  1. {{class_name}}_probability (如 normal_probability, abnormal_probability)\n"
        f"  2. class_{{i}}_prob (如 class_0_prob, class_1_prob)"
    )


# ==================== 单指标计算 ====================

def compute_single_metric(metric_name, labels, preds, probs):
    """
    计算单个评估指标

    参数:
        metric_name: 指标名称
        labels: 真实标签
        preds: 预测标签
        probs: 预测概率矩阵

    返回:
        metric_value: 指标值
    """
    if metric_name == 'macro_auc':
        # Macro AUC: 每类AUC的算术平均(OvR策略)
        n_classes = probs.shape[1]

        if n_classes == 2:
            # 二分类:手动构造One-Hot编码矩阵
            # 避免sklearn的label_binarize在二分类时只返回(n_samples, 1)的问题
            labels_bin = np.zeros((len(labels), n_classes), dtype=int)
            labels_bin[np.arange(len(labels)), labels] = 1

            auc_scores = []
            for i in range(n_classes):
                try:
                    # 计算每个类别的OvR AUC
                    auc = roc_auc_score(labels_bin[:, i], probs[:, i])
                    auc_scores.append(auc)
                except ValueError:
                    # 如果某个类别没有样本,使用NaN
                    auc_scores.append(np.nan)

            return np.nanmean(auc_scores)

        else:
            # 多分类(N>=3):直接使用sklearn的OvR
            return roc_auc_score(labels, probs, average='macro', multi_class='ovr')

    elif metric_name == 'micro_auc':
        # Micro AUC: 全局AUC(OvR策略)
        n_classes = probs.shape[1]
        if n_classes == 2:
            # 二分类:使用正类概率计算
            return roc_auc_score(labels, probs[:, 1])
        else:
            # 多分类(N>=3)
            return roc_auc_score(labels, probs, average='micro', multi_class='ovr')

    elif metric_name == 'accuracy':
        # 整体准确率
        return accuracy_score(labels, preds)

    elif metric_name == 'macro_precision':
        # Macro Precision: 每类Precision的算术平均
        return precision_score(labels, preds, average='macro', zero_division=0)

    elif metric_name == 'macro_recall':
        # Macro Recall (Sensitivity): 每类Recall的算术平均
        return recall_score(labels, preds, average='macro', zero_division=0)

    elif metric_name == 'macro_f1':
        # Macro F1: 每类F1的算术平均
        return f1_score(labels, preds, average='macro', zero_division=0)

    else:
        raise ValueError(f"不支持的指标: {metric_name}")


# ==================== 配对Bootstrap比较 ====================

def paired_bootstrap_comparison(labels, preds1, preds2, probs1, probs2,
                                n_bootstrap=1000, ci_level=95.0,
                                metrics=['macro_auc'], random_state=42):
    """
    使用配对Bootstrap方法比较两个模型的多个指标

    参数:
        labels: 真实标签 (n_samples,)
        preds1: 模型1预测标签 (n_samples,)
        preds2: 模型2预测标签 (n_samples,)
        probs1: 模型1预测概率 (n_samples, n_classes)
        probs2: 模型2预测概率 (n_samples, n_classes)
        n_bootstrap: Bootstrap重采样次数
        ci_level: 置信水平(百分比)
        metrics: 指标列表
        random_state: 随机种子

    返回:
        results: 字典,每个指标一个子字典,包含差异、置信区间、p值等
    """
    print(f"\n[4/6] 执行Bootstrap重采样 ({n_bootstrap}次 × {len(metrics)}个指标)...")

    n_samples = len(labels)
    alpha = (100 - ci_level) / 2
    results = {}

    # 对每个指标执行Bootstrap
    for metric_name in metrics:
        # 计算原始指标值
        metric1_original = compute_single_metric(metric_name, labels, preds1, probs1)
        metric2_original = compute_single_metric(metric_name, labels, preds2, probs2)
        diff_original = metric1_original - metric2_original

        # Bootstrap重采样
        bootstrap_diffs = np.zeros(n_bootstrap)

        # 使用tqdm显示进度
        metric_display = metric_name.replace('_', ' ').title()

        for i in tqdm(range(n_bootstrap), desc=f"  {metric_display:15s}",
                     ncols=80, leave=False):
            # 分层重采样(配对设计:两模型使用相同索引)
            try:
                indices = resample(
                    range(n_samples),
                    stratify=labels,
                    n_samples=n_samples,
                    replace=True,
                    random_state=random_state + i
                )
            except ValueError:
                # 样本太少或分布极端,回退到普通抽样
                indices = resample(
                    range(n_samples),
                    n_samples=n_samples,
                    replace=True,
                    random_state=random_state + i
                )

            # 提取重采样数据
            labels_boot = labels[indices]
            preds1_boot = preds1[indices]
            preds2_boot = preds2[indices]
            probs1_boot = probs1[indices]
            probs2_boot = probs2[indices]

            # 计算Bootstrap指标
            metric1_boot = compute_single_metric(metric_name, labels_boot, preds1_boot, probs1_boot)
            metric2_boot = compute_single_metric(metric_name, labels_boot, preds2_boot, probs2_boot)

            bootstrap_diffs[i] = metric1_boot - metric2_boot

        # 计算置信区间(百分位法)
        ci_lower = np.percentile(bootstrap_diffs, alpha)
        ci_upper = np.percentile(bootstrap_diffs, 100 - alpha)

        # 计算p值(双侧检验)
        p_value = 2 * min(
            np.mean(bootstrap_diffs >= 0),
            np.mean(bootstrap_diffs <= 0)
        )

        # 显著性判断
        significant = not (ci_lower <= 0 <= ci_upper)

        # 效应量分类
        abs_diff = abs(diff_original)
        if abs_diff < 0.02:
            effect_size = "Negligible"
        elif abs_diff < 0.05:
            effect_size = "Small"
        elif abs_diff < 0.10:
            effect_size = "Medium"
        else:
            effect_size = "Large"

        # 保存结果
        results[metric_name] = {
            'metric1_original': metric1_original,
            'metric2_original': metric2_original,
            'diff_original': diff_original,
            'diff_bootstrap': bootstrap_diffs,
            'diff_mean': bootstrap_diffs.mean(),
            'diff_std': bootstrap_diffs.std(),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'significant': significant,
            'effect_size': effect_size
        }

    print(f"  ✓ Bootstrap完成")

    return results


# ==================== 可视化生成 ====================

def generate_comparison_visualizations(comparison_results, model_name1, model_name2,
                                       output_dir, ci_level=95):
    """
    生成比较结果可视化

    参数:
        comparison_results: Bootstrap比较结果字典
        model_name1: 模型1名称
        model_name2: 模型2名称
        output_dir: 输出目录
        ci_level: 置信水平
    """
    print(f"\n[5/6] 生成可视化...")

    os.makedirs(output_dir, exist_ok=True)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    metrics = list(comparison_results.keys())
    n_metrics = len(metrics)

    # 图1: 多指标差异分布直方图
    if n_metrics == 1:
        fig1, axes1 = plt.subplots(1, 1, figsize=(8, 6))
        axes1 = [axes1]
    elif n_metrics == 2:
        fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
    elif n_metrics <= 4:
        fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
        axes1 = axes1.flatten()
    else:
        fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
        axes1 = axes1.flatten()

    for idx, metric_name in enumerate(metrics):
        ax = axes1[idx]
        result = comparison_results[metric_name]

        diffs = result['diff_bootstrap']
        ci_lower = result['ci_lower']
        ci_upper = result['ci_upper']
        p_value = result['p_value']
        significant = result['significant']

        # 绘制直方图
        ax.hist(diffs, bins=30, alpha=0.7, color='steelblue',
               edgecolor='black', density=False)

        # 标注置信区间
        ax.axvline(ci_lower, color='red', linestyle='--', linewidth=2,
                  label=f'{ci_level}% CI Lower')
        ax.axvline(ci_upper, color='red', linestyle='--', linewidth=2,
                  label=f'{ci_level}% CI Upper')

        # 标注0线(无差异线)
        ax.axvline(0, color='green', linestyle='-', linewidth=2,
                  label='No Difference', alpha=0.6)

        # 标题显示指标名和p值
        metric_display = metric_name.replace('_', ' ').title()
        sig_marker = '***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else ''
        ax.set_title(f'{metric_display}\n(p={p_value:.4f} {sig_marker})', fontsize=12, fontweight='bold')

        ax.set_xlabel('差异 (Model1 - Model2)', fontsize=10)
        ax.set_ylabel('频数', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)

    # 隐藏多余子图
    for idx in range(n_metrics, len(axes1)):
        axes1[idx].axis('off')

    fig1.suptitle(f'指标差异Bootstrap分布\n{model_name1} vs {model_name2}',
                 fontsize=14, fontweight='bold')
    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    output_path1 = os.path.join(output_dir, 'metrics_difference_distribution.png')
    fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"  ✓ metrics_difference_distribution.png")

    # 图2: 双模型指标对比柱状图
    fig2, ax2 = plt.subplots(figsize=(max(10, n_metrics * 2), 6))

    x_pos = np.arange(n_metrics)
    width = 0.35

    metric1_values = [comparison_results[m]['metric1_original'] for m in metrics]
    metric2_values = [comparison_results[m]['metric2_original'] for m in metrics]

    # 计算误差线(Bootstrap CI的半宽)
    metric1_ci_lower = [comparison_results[m]['metric1_original'] -
                        (comparison_results[m]['metric1_original'] -
                         (comparison_results[m]['diff_original'] - comparison_results[m]['ci_upper']))
                        for m in metrics]
    metric1_ci_upper = [((comparison_results[m]['diff_original'] - comparison_results[m]['ci_lower']) +
                         comparison_results[m]['metric1_original']) -
                        comparison_results[m]['metric1_original']
                        for m in metrics]

    metric2_ci_lower = [comparison_results[m]['metric2_original'] -
                        (comparison_results[m]['metric2_original'] -
                         (comparison_results[m]['metric2_original'] - comparison_results[m]['diff_original'] + comparison_results[m]['ci_lower']))
                        for m in metrics]
    metric2_ci_upper = [((comparison_results[m]['metric2_original'] - comparison_results[m]['diff_original'] + comparison_results[m]['ci_upper']) -
                         comparison_results[m]['metric2_original'])
                        for m in metrics]

    # 简化误差线计算:使用Bootstrap标准差估计
    metric1_errors = [[result['diff_std'] / 2] * 2 for result in comparison_results.values()]
    metric2_errors = [[result['diff_std'] / 2] * 2 for result in comparison_results.values()]

    bars1 = ax2.bar(x_pos - width/2, metric1_values, width,
                   label=model_name1, color='skyblue', edgecolor='black')
    bars2 = ax2.bar(x_pos + width/2, metric2_values, width,
                   label=model_name2, color='lightcoral', edgecolor='black')

    # 添加数值标签
    for bar, val in zip(bars1, metric1_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    for bar, val in zip(bars2, metric2_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # 添加显著性星号
    for idx, metric_name in enumerate(metrics):
        if comparison_results[metric_name]['significant']:
            max_height = max(metric1_values[idx], metric2_values[idx])
            ax2.text(idx, max_height * 1.1, '*', ha='center', va='bottom',
                    fontsize=20, color='red', fontweight='bold')

    metric_labels = [m.replace('_', ' ').title() for m in metrics]
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(metric_labels, rotation=15, ha='right')
    ax2.set_ylabel('指标值', fontsize=12)
    ax2.set_title(f'双模型指标对比\n{model_name1} vs {model_name2}',
                 fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    fig2.tight_layout()
    output_path2 = os.path.join(output_dir, 'metrics_comparison_barplot.png')
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"  ✓ metrics_comparison_barplot.png")


# ==================== 文本报告生成 ====================

def generate_comparison_report(comparison_results, model_name1, model_name2,
                               output_path, n_samples, n_bootstrap, ci_level):
    """
    生成详细的统计分析文本报告

    参数:
        comparison_results: Bootstrap比较结果字典
        model_name1: 模型1名称
        model_name2: 模型2名称
        output_path: 输出文件路径
        n_samples: 样本数量
        n_bootstrap: Bootstrap次数
        ci_level: 置信水平
    """
    print(f"\n[6/6] 生成统计报告...")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("双模型多指标统计比较报告\n")
        f.write("=" * 80 + "\n\n")

        # 实验配置
        f.write("【实验配置】\n")
        f.write(f"- 模型1: {model_name1}\n")
        f.write(f"- 模型2: {model_name2}\n")
        f.write(f"- 样本数量: {n_samples}\n")
        f.write(f"- Bootstrap重采样次数: {n_bootstrap}\n")
        f.write(f"- 置信水平: {ci_level}%\n")
        f.write(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 每个指标的详细结果
        for metric_name, result in comparison_results.items():
            metric_display = metric_name.replace('_', ' ').title()
            f.write("-" * 80 + "\n")
            f.write(f"【{metric_display}】\n\n")

            f.write("原始指标值:\n")
            f.write(f"  - {model_name1}: {result['metric1_original']:.6f}\n")
            f.write(f"  - {model_name2}: {result['metric2_original']:.6f}\n")
            f.write(f"  - 差异 (Model1 - Model2): {result['diff_original']:+.6f}\n\n")

            f.write(f"Bootstrap {ci_level}% 置信区间:\n")
            f.write(f"  - CI: [{result['ci_lower']:.6f}, {result['ci_upper']:.6f}]\n")
            f.write(f"  - 差异均值: {result['diff_mean']:.6f}\n")
            f.write(f"  - 差异标准差: {result['diff_std']:.6f}\n\n")

            f.write("统计显著性检验:\n")
            f.write(f"  - p值 (双侧): {result['p_value']:.6f}\n")

            if result['significant']:
                if result['diff_original'] > 0:
                    conclusion = f"{model_name1}性能显著优于{model_name2}"
                else:
                    conclusion = f"{model_name2}性能显著优于{model_name1}"
            else:
                conclusion = "两模型性能无显著差异"

            f.write(f"  - 结论: {conclusion}\n")
            f.write(f"  - 效应量: {result['effect_size']}\n\n")

        # 总结
        f.write("=" * 80 + "\n")
        f.write("【总体结论】\n\n")

        significant_metrics = [m for m, r in comparison_results.items() if r['significant']]

        if significant_metrics:
            f.write(f"在以下指标上检测到统计显著差异:\n")
            for metric in significant_metrics:
                result = comparison_results[metric]
                metric_display = metric.replace('_', ' ').title()
                direction = "优于" if result['diff_original'] > 0 else "劣于"
                f.write(f"  - {metric_display}: {model_name1} {direction} {model_name2} "
                       f"(p={result['p_value']:.4f})\n")
        else:
            f.write(f"所有指标均未检测到统计显著差异(α={1 - ci_level/100})\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("\n解读说明:\n")
        f.write("- 置信区间不包含0 → 差异具有统计显著性\n")
        f.write("- 置信区间包含0 → 差异无统计显著性\n")
        f.write("- p值 < 0.05 → 通常认为差异显著\n")
        f.write("- 效应量: Negligible(<0.02) < Small(<0.05) < Medium(<0.10) < Large(≥0.10)\n")
        f.write("=" * 80 + "\n")

    print(f"  ✓ comparison_report.txt")


# ==================== 主函数接口 ====================

def compare_two_models(csv_path1, csv_path2,
                      model_name1="Model_A", model_name2="Model_B",
                      output_dir="metrics_out/model_comparison",
                      n_bootstrap=1000, ci_level=95.0,
                      metrics=['macro_auc'],
                      random_state=42,
                      verbose=True):
    """
    比较两个模型的性能差异(主函数接口)

    参数:
        csv_path1 (str): 模型1的detailed_predictions.csv路径
        csv_path2 (str): 模型2的detailed_predictions.csv路径
        model_name1 (str): 模型1显示名称,默认'Model_A'
        model_name2 (str): 模型2显示名称,默认'Model_B'
        output_dir (str): 输出目录,默认'metrics_out/model_comparison'
        n_bootstrap (int): Bootstrap重采样次数,默认1000
        ci_level (float): 置信水平(百分比),默认95.0
        metrics (list): 指标列表,默认['macro_auc']
                       可选: 'macro_auc', 'micro_auc', 'accuracy',
                            'macro_precision', 'macro_recall', 'macro_f1'
        random_state (int): 随机种子,默认42
        verbose (bool): 是否打印详细信息,默认True

    返回:
        comparison_results (dict): 比较结果字典,每个指标包含:
            - metric1_original: 模型1原始值
            - metric2_original: 模型2原始值
            - diff_original: 差异值
            - ci_lower: 置信区间下界
            - ci_upper: 置信区间上界
            - p_value: p值
            - significant: 是否显著
            - effect_size: 效应量

    使用示例:
        # 示例1: 基础用法
        results = compare_two_models(
            'metrics_out/model_A/detailed_predictions.csv',
            'metrics_out/model_B/detailed_predictions.csv'
        )

        # 示例2: 多指标比较
        results = compare_two_models(
            'metrics_out/model_A/detailed_predictions.csv',
            'metrics_out/model_B/detailed_predictions.csv',
            model_name1='InceptionResNetV2',
            model_name2='ResNet50',
            metrics=['macro_auc', 'accuracy', 'macro_f1'],
            n_bootstrap=2000
        )

        # 示例3: 访问结果
        print(f"Macro AUC差异: {results['macro_auc']['diff_original']:.4f}")
        print(f"95% CI: [{results['macro_auc']['ci_lower']:.4f}, {results['macro_auc']['ci_upper']:.4f}]")
        print(f"p值: {results['macro_auc']['p_value']:.4f}")
        print(f"是否显著: {results['macro_auc']['significant']}")
    """
    # 验证指标名称
    supported_metrics = {
        'macro_auc', 'micro_auc', 'accuracy',
        'macro_precision', 'macro_recall', 'macro_f1'
    }

    if isinstance(metrics, str):
        metrics = [m.strip() for m in metrics.split(',')]

    invalid_metrics = set(metrics) - supported_metrics
    if invalid_metrics:
        raise ValueError(
            f"不支持的指标: {invalid_metrics}\n"
            f"支持的指标: {supported_metrics}"
        )

    try:
        # 1. 加载并验证数据
        if verbose:
            df1, df2, class_names = load_and_validate_predictions(
                csv_path1, csv_path2, model_name1, model_name2
            )
        else:
            # 静默模式
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                df1, df2, class_names = load_and_validate_predictions(
                    csv_path1, csv_path2, model_name1, model_name2
                )
            finally:
                sys.stdout = old_stdout

        # 2. 提取预测数据
        if verbose:
            print(f"\n[2/6] 提取预测数据...")

        labels = df1['true'].values
        preds1 = df1['predict'].values
        preds2 = df2['predict'].values

        n_classes = len(class_names)
        probs1 = extract_probability_columns(df1, n_classes)
        probs2 = extract_probability_columns(df2, n_classes)

        if verbose:
            print(f"  ✓ 识别到{n_classes}个类别: {class_names}")
            prob_cols = [col for col in df1.columns if
                        col.endswith('_probability') or '_prob' in col]
            print(f"  ✓ 概率列: {', '.join(sorted(prob_cols)[:3])}...")
            print(f"  ✓ 预测标签列: predict")

        # 3. 计算原始指标
        if verbose:
            print(f"\n[3/6] 计算原始指标...")
            for metric_name in metrics:
                metric1 = compute_single_metric(metric_name, labels, preds1, probs1)
                metric2 = compute_single_metric(metric_name, labels, preds2, probs2)
                diff = metric1 - metric2
                metric_display = metric_name.replace('_', ' ').title()
                print(f"  ✓ {metric_display:20s}: {model_name1}={metric1:.4f}, "
                     f"{model_name2}={metric2:.4f}, 差异={diff:+.4f}")

        # 4. Bootstrap比较
        comparison_results = paired_bootstrap_comparison(
            labels, preds1, preds2, probs1, probs2,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            metrics=metrics,
            random_state=random_state
        )

        # 5. 生成可视化
        generate_comparison_visualizations(
            comparison_results, model_name1, model_name2,
            output_dir, ci_level
        )

        # 6. 生成文本报告
        report_path = os.path.join(output_dir, 'comparison_report.txt')
        generate_comparison_report(
            comparison_results, model_name1, model_name2,
            report_path, len(labels), n_bootstrap, ci_level
        )

        # 7. 打印摘要
        if verbose:
            print("\n" + "=" * 60)
            print("比较结果摘要")
            print("=" * 60)

            for metric_name in metrics:
                result = comparison_results[metric_name]
                metric_display = metric_name.replace('_', ' ').title()

                # 显著性标记
                if result['p_value'] < 0.01:
                    sig_marker = '***'
                elif result['p_value'] < 0.05:
                    sig_marker = '**'
                elif result['p_value'] < 0.1:
                    sig_marker = '*'
                else:
                    sig_marker = ''

                print(f"{metric_display:15s}: {result['diff_original']:+.4f} "
                     f"[{ci_level}% CI: {result['ci_lower']:+.4f}, {result['ci_upper']:+.4f}] "
                     f"{sig_marker:3s} (p={result['p_value']:.4f})")

            print(f"\n显著性标记: *** p<0.01, ** p<0.05, * p<0.1\n")

            # 总体结论
            significant_metrics = [m for m, r in comparison_results.items() if r['significant']]
            if significant_metrics:
                model_better = model_name1 if comparison_results[significant_metrics[0]]['diff_original'] > 0 else model_name2
                metric_names_display = ', '.join([m.replace('_', ' ').title() for m in significant_metrics])
                print(f"结论: {model_better} 在 {metric_names_display} 上显著优于对比模型\n")
            else:
                print(f"结论: 两模型在所有指标上均无显著差异\n")

            print(f"所有结果已保存至: {output_dir}/")
            print("=" * 60 + "\n")

        return comparison_results

    except Exception as e:
        if verbose:
            print(f"\n错误: {str(e)}\n")
            import traceback
            traceback.print_exc()
        raise

if __name__ == '__main__':

    compare_two_models(
        '../metrics_out/inception_resnet_v2_cls_test/detailed_predictions.csv',
        '../metrics_out/inception_resnet_v2_cls_test/detailed_predictions.csv',
        model_name1='InceptionResNetV2',
        model_name2='ResNet50',
        metrics=['macro_auc', 'accuracy', 'macro_f1'],
        n_bootstrap=1000,
        output_dir='../metrics_out/model_comparison_example'
    )