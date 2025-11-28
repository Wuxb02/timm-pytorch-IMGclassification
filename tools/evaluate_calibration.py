# -*- coding: utf-8 -*-
"""
模型校准性能评估工具

功能:
- Calibration Plot (可靠性曲线): 可视化预测概率与实际正确率的对应关系
- Brier Score: 量化概率预测的准确性
- 支持整体校准和各类别校准评估

使用示例:
    # 从CSV加载 (推荐)
    from tools.evaluate_calibration import evaluate_model_calibration

    results = evaluate_model_calibration(
        csv_path='metrics_out/inception_resnet_v2_cls_test/detailed_predictions.csv',
        output_dir='metrics_out/calibration_analysis',
        n_bins=10
    )

    # 实时推理
    from eval import Eval_Classification
    model = Eval_Classification()
    results = evaluate_model_calibration(
        model_instance=model,
        annotation_path='cls_test.txt',
        output_dir='metrics_out/calibration_analysis'
    )

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
from PIL import Image
from sklearn.calibration import calibration_curve


# ==================== Brier Score计算 ====================

def compute_brier_score_multiclass(labels, probs):
    """
    计算多分类Brier Score (整体评分)

    公式: BS = (1/N) * Σ_{i=1}^{N} Σ_{j=1}^{C} (p_{ij} - y_{ij})^2

    参数:
        labels: 真实标签, numpy数组, shape (n_samples,)
        probs: 预测概率矩阵, numpy数组, shape (n_samples, n_classes)

    返回:
        brier_score: float, Brier Score值 (越小越好, 0为完美)

    示例:
        >>> labels = np.array([0, 1, 0])
        >>> probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        >>> bs = compute_brier_score_multiclass(labels, probs)
        >>> print(f"Brier Score: {bs:.4f}")
    """
    n_samples = len(labels)
    n_classes = probs.shape[1]

    # One-hot编码真实标签
    labels_onehot = np.zeros((n_samples, n_classes))
    labels_onehot[np.arange(n_samples), labels] = 1

    # 计算Brier Score
    brier_score = np.mean(np.sum((probs - labels_onehot)**2, axis=1))

    return brier_score


def compute_per_class_brier_score(labels, probs, n_classes):
    """
    计算各类别Brier Score (One-vs-Rest策略)

    参数:
        labels: 真实标签, numpy数组, shape (n_samples,)
        probs: 预测概率矩阵, numpy数组, shape (n_samples, n_classes)
        n_classes: 类别数量

    返回:
        per_class_brier: list, 长度为n_classes, 每个元素为对应类别的Brier Score

    示例:
        >>> labels = np.array([0, 1, 0, 1])
        >>> probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]])
        >>> per_class_bs = compute_per_class_brier_score(labels, probs, 2)
        >>> print(f"类别0 Brier Score: {per_class_bs[0]:.4f}")
        >>> print(f"类别1 Brier Score: {per_class_bs[1]:.4f}")
    """
    per_class_brier = []

    for class_idx in range(n_classes):
        # 二值化: 该类别为1, 其他为0
        binary_labels = (labels == class_idx).astype(int)
        # 该类别的预测概率
        class_probs = probs[:, class_idx]
        # 计算该类别的Brier Score
        brier_class = np.mean((class_probs - binary_labels)**2)
        per_class_brier.append(brier_class)

    return per_class_brier


def interpret_brier_score(brier_score, n_classes):
    """
    解释Brier Score的质量等级

    参数:
        brier_score: Brier Score值
        n_classes: 类别数量

    返回:
        (grade, description): 元组, (等级, 描述)

    等级划分 (2分类):
        - 优秀: BS < 0.10
        - 良好: 0.10 ≤ BS < 0.15
        - 一般: 0.15 ≤ BS < 0.20
        - 较差: BS ≥ 0.20

    等级划分 (多分类): 阈值根据类别数动态调整
    """
    if n_classes == 2:
        if brier_score < 0.10:
            return ("优秀", "概率预测高度可靠")
        elif brier_score < 0.15:
            return ("良好", "模型预测概率具有中等可靠性")
        elif brier_score < 0.20:
            return ("一般", "存在一定程度的校准偏差,建议重新校准")
        else:
            return ("较差", "显著的校准偏差,需要重新校准")
    else:
        # 多分类调整阈值: 随机分类器基准 BS = 1 - (1/C)
        baseline = 1.0 - (1.0 / n_classes)
        if brier_score < baseline * 0.4:
            return ("优秀", "概率预测高度可靠")
        elif brier_score < baseline * 0.6:
            return ("良好", "模型预测概率具有中等可靠性")
        elif brier_score < baseline * 0.8:
            return ("一般", "存在一定程度的校准偏差,建议重新校准")
        else:
            return ("较差", "显著的校准偏差,需要重新校准")


# ==================== 校准曲线计算 ====================

def compute_calibration_curve(labels, probs, n_bins=10, strategy='uniform'):
    """
    计算整体校准曲线 (使用sklearn.calibration.calibration_curve)

    原理:
        1. 提取每个样本的最大预测概率
        2. 使用sklearn的calibration_curve函数计算校准曲线
        3. 返回平均预测概率和真实正确率

    参数:
        labels: 真实标签, numpy数组, shape (n_samples,)
        probs: 预测概率矩阵, numpy数组, shape (n_samples, n_classes)
        n_bins: 分桶数量, 默认10
        strategy: 分桶策略, 'uniform' (等宽) 或 'quantile' (等频)

    返回:
        dict: {
            'mean_predicted_probs': 每个桶的平均预测概率
            'true_frequencies': 每个桶的真实正确率
            'brier_score': 整体Brier Score
        }
    """
    # 步骤1: 提取最大概率和预测标签
    max_probs = np.max(probs, axis=1)  # (n_samples,)
    predicted_labels = np.argmax(probs, axis=1)  # (n_samples,)
    correct = (predicted_labels == labels).astype(int)  # 0/1正确性标记

    # 步骤2: 使用sklearn的calibration_curve计算校准曲线
    # calibration_curve返回: (true_frequencies, mean_predicted_probs)
    true_frequencies, mean_predicted_probs = calibration_curve(
        y_true=correct,
        y_prob=max_probs,
        n_bins=n_bins,
        strategy=strategy
    )

    # 计算整体Brier Score
    brier_score = compute_brier_score_multiclass(labels, probs)

    return {
        'mean_predicted_probs': mean_predicted_probs,
        'true_frequencies': true_frequencies,
        'brier_score': brier_score
    }


def compute_per_class_calibration(labels, probs, class_idx, n_bins=10, strategy='uniform'):
    """
    计算单个类别的校准曲线 (One-vs-Rest策略, 使用sklearn.calibration.calibration_curve)

    参数:
        labels: 真实标签, numpy数组, shape (n_samples,)
        probs: 预测概率矩阵, numpy数组, shape (n_samples, n_classes)
        class_idx: 类别索引
        n_bins: 分桶数量, 默认10
        strategy: 分桶策略, 'uniform' (等宽) 或 'quantile' (等频)

    返回:
        dict: {
            'mean_predicted_probs': 每个桶的平均预测概率
            'true_frequencies': 每个桶的真实为该类别的比例
            'brier_score': 该类别的Brier Score
        }
    """
    # 步骤1: 二值化
    binary_labels = (labels == class_idx).astype(int)  # 该类别为1, 其他为0
    class_probs = probs[:, class_idx]  # 该类别的预测概率

    # 步骤2: 使用sklearn的calibration_curve计算校准曲线
    # calibration_curve返回: (true_frequencies, mean_predicted_probs)
    true_frequencies, mean_predicted_probs = calibration_curve(
        y_true=binary_labels,
        y_prob=class_probs,
        n_bins=n_bins,
        strategy=strategy
    )

    # 计算该类别的Brier Score
    brier_class = np.mean((class_probs - binary_labels)**2)

    return {
        'mean_predicted_probs': mean_predicted_probs,
        'true_frequencies': true_frequencies,
        'brier_score': brier_class
    }


# ==================== 数据加载与验证 ====================

def load_predictions_for_calibration(csv_path=None, annotation_path='cls_test.txt',
                                     model_instance=None, class_names=None):
    """
    加载预测数据用于校准评估

    支持两种模式:
    - 模式1 (推荐): 从CSV加载 (速度快, 来自eval.py的detailed_predictions.csv)
    - 模式2: 实时推理 (灵活但慢, 需要提供model_instance)

    参数:
        csv_path: CSV文件路径, 优先使用此参数
        annotation_path: 测试数据标注文件路径, 默认'cls_test.txt'
        model_instance: Eval_Classification实例, 仅模式2需要
        class_names: 类别名称列表, 如果为None则自动推断

    返回:
        (labels, probs, class_names): 元组
            - labels: numpy数组, shape (n_samples,)
            - probs: numpy数组, shape (n_samples, n_classes)
            - class_names: list, 类别名称列表

    异常:
        ValueError: 参数组合无效或数据格式错误
    """
    # 模式1: 从CSV加载
    if csv_path is not None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

        df = pd.read_csv(csv_path, encoding='utf-8-sig')

        # 验证必需列
        if 'true' not in df.columns or 'predict' not in df.columns:
            raise ValueError(f"CSV文件缺少必需列: 'true' 或 'predict'")

        labels = df['true'].values

        # 识别概率列
        prob_cols = [col for col in df.columns if
                     col.endswith('_probability') or '_prob' in col]

        if not prob_cols:
            raise ValueError(
                f"CSV文件未找到概率列。期望列名格式:\n"
                f"  - {{class_name}}_probability (如 normal_probability)\n"
                f"  - class_{{i}}_prob (如 class_0_prob)"
            )

        # 排序概率列
        prob_cols_sorted = sorted(prob_cols)
        probs = df[prob_cols_sorted].values

        # 提取类别名称
        if class_names is None:
            class_names = []
            for col in prob_cols_sorted:
                if col.endswith('_probability'):
                    class_name = col.replace('_probability', '')
                    class_names.append(class_name)
                elif '_prob' in col and col.startswith('class_'):
                    # class_0_prob -> 0
                    class_idx = col.split('_')[1]
                    class_names.append(class_idx)

        return labels, probs, class_names

    # 模式2: 实时推理
    elif model_instance is not None:
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"标注文件不存在: {annotation_path}")

        # 读取测试数据
        with open(annotation_path, "r", encoding='UTF-8-sig') as f:
            lines = f.readlines()

        labels_list = []
        probs_list = []

        for line in lines:
            parts = line.strip().split(';')
            if len(parts) != 2:
                continue

            label = int(parts[0])
            image_path = parts[1]

            # 加载图像
            image = Image.open(image_path)

            # 推理获取概率
            pred_probs = model_instance.detect_image(image)

            labels_list.append(label)
            probs_list.append(pred_probs)

        labels = np.array(labels_list)
        probs = np.array(probs_list)

        # 获取类别名称
        if class_names is None:
            if hasattr(model_instance, 'class_names'):
                class_names = model_instance.class_names
            else:
                # 默认使用数字类别名
                n_classes = probs.shape[1]
                class_names = [str(i) for i in range(n_classes)]

        return labels, probs, class_names

    else:
        raise ValueError(
            "必须提供以下参数之一:\n"
            "  1. csv_path: 从CSV加载预测结果 (推荐)\n"
            "  2. model_instance: 实时推理模式"
        )


# ==================== 可视化生成 ====================

def get_subplot_layout(n_classes):
    """
    根据类别数量返回最优子图布局

    参数:
        n_classes: 类别数量

    返回:
        (nrows, ncols): 元组, (行数, 列数)
    """
    if n_classes == 2:
        return (1, 2)  # 1行2列
    elif n_classes == 3:
        return (1, 3)  # 1行3列
    elif n_classes == 4:
        return (2, 2)  # 2行2列
    elif n_classes <= 6:
        return (2, 3)  # 2行3列
    elif n_classes <= 9:
        return (3, 3)  # 3行3列
    else:
        ncols = 3
        nrows = int(np.ceil(n_classes / ncols))
        return (nrows, ncols)


def plot_overall_calibration(calibration_result, output_path, title="Overall Calibration Curve"):
    """
    绘制整体校准图 (使用sklearn.calibration.calibration_curve结果)

    参数:
        calibration_result: compute_calibration_curve()的返回值
        output_path: 输出文件路径
        title: 图表标题
    """
    mean_predicted_probs = calibration_result['mean_predicted_probs']
    true_frequencies = calibration_result['true_frequencies']
    brier_score = calibration_result['brier_score']

    # 设置字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 11

    fig, ax = plt.subplots(figsize=(10, 8))

    # 1. 绘制校准曲线 (使用线图连接各点)
    ax.plot(mean_predicted_probs, true_frequencies,
            marker='o', markersize=8, linewidth=2,
            color='steelblue', label='Calibration curve',
            markeredgecolor='black', markeredgewidth=1.5)

    # 2. 完美校准参考线 (对角线)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2,
            label='Perfect calibration', alpha=0.6)

    # 3. 标题显示Brier Score
    ax.set_title(f'{title}\nBrier Score = {brier_score:.4f}',
                 fontsize=16, fontweight='bold', family='Times New Roman')

    # 4. 轴标签和网格
    ax.set_xlabel('Mean Predicted Probability', fontsize=14, family='Times New Roman')
    ax.set_ylabel('True Frequency (Accuracy)', fontsize=14, family='Times New Roman')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(fontsize=12, loc='upper left', prop={'family': 'Times New Roman'})

    # 保存
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_per_class_calibration(labels, probs, class_names, n_bins, output_path,
                                binning_strategy='uniform'):
    """
    绘制各类别校准图 (子图布局)

    参数:
        labels: 真实标签, numpy数组
        probs: 预测概率矩阵, numpy数组
        class_names: 类别名称列表
        n_bins: 分桶数量
        output_path: 输出文件路径
        binning_strategy: 分桶策略
    """
    n_classes = len(class_names)
    nrows, ncols = get_subplot_layout(n_classes)

    # 设置字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 11

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5))

    # 确保axes是数组形式
    if n_classes == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for class_idx in range(n_classes):
        ax = axes[class_idx]

        # 计算该类别校准曲线
        calib_result = compute_per_class_calibration(labels, probs, class_idx,
                                                      n_bins, binning_strategy)

        mean_predicted_probs = calib_result['mean_predicted_probs']
        true_frequencies = calib_result['true_frequencies']
        brier_score = calib_result['brier_score']

        # 绘制校准曲线 (使用线图连接各点)
        ax.plot(mean_predicted_probs, true_frequencies,
                marker='o', markersize=8, linewidth=2,
                color='coral', label='Calibration curve',
                markeredgecolor='black', markeredgewidth=1.5)

        # 完美校准线
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2,
                label='Perfect calibration', alpha=0.6)

        # 标题显示类别名和Brier Score
        ax.set_title(f'Class: {class_names[class_idx]}\nBrier Score = {brier_score:.4f}',
                     fontsize=14, fontweight='bold', family='Times New Roman')

        ax.set_xlabel('Mean Predicted Probability', fontsize=12, family='Times New Roman')
        ax.set_ylabel('True Frequency (Fraction of Positives)', fontsize=12, family='Times New Roman')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(fontsize=10, loc='upper left', prop={'family': 'Times New Roman'})

    # 隐藏多余子图
    for idx in range(n_classes, len(axes)):
        axes[idx].axis('off')

    fig.suptitle('Per-Class Calibration Curves', fontsize=18, fontweight='bold',
                 family='Times New Roman')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ==================== 文本报告生成 ====================

def generate_calibration_report(overall_calibration, per_class_calibrations,
                                overall_brier, per_class_brier, class_names,
                                output_path, n_bins, binning_strategy):
    """
    生成校准性能文本报告

    参数:
        overall_calibration: 整体校准结果字典
        per_class_calibrations: 各类别校准结果列表
        overall_brier: 整体Brier Score
        per_class_brier: 各类别Brier Score列表
        class_names: 类别名称列表
        output_path: 输出文件路径
        n_bins: 分桶数量
        binning_strategy: 分桶策略
    """
    n_classes = len(class_names)

    # 计算非空桶数量
    non_empty_bins = np.sum(~np.isnan(overall_calibration['true_frequencies']))

    # 解释整体Brier Score
    overall_grade, overall_desc = interpret_brier_score(overall_brier, n_classes)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("模型校准性能评估报告\n")
        f.write("=" * 80 + "\n\n")

        # I. 整体校准指标
        f.write("I. 整体校准指标\n")
        f.write("-" * 80 + "\n")
        f.write(f"整体Brier Score:      {overall_brier:.4f}\n")
        f.write(f"  - 解释:             {overall_grade} ({overall_brier:.2f})\n")
        f.write(f"  - 校准质量:         {overall_desc}\n\n")
        f.write(f"分桶数量:             {n_bins}\n")
        f.write(f"分桶策略:             {'等宽分桶' if binning_strategy == 'uniform' else '等频分桶'}\n")
        f.write(f"非空桶数量:           {non_empty_bins} / {n_bins}\n\n")
        f.write("-" * 80 + "\n\n")

        # II. 各类别校准指标
        f.write("II. 各类别校准指标\n")
        f.write("-" * 80 + "\n\n")

        for class_idx, class_name in enumerate(class_names):
            brier_class = per_class_brier[class_idx]
            grade, desc = interpret_brier_score(brier_class, n_classes)

            f.write(f"类别{class_idx} ({class_name}):\n")
            f.write(f"  Brier Score:        {brier_class:.4f}\n")
            f.write(f"  - 解释:             {grade} ({brier_class:.2f})\n")
            f.write(f"  - 校准质量:         {desc}\n\n")

        # 平均Brier Score
        mean_brier = np.mean(per_class_brier)
        f.write(f"平均Brier Score (宏平均): {mean_brier:.4f}\n\n")
        f.write("-" * 80 + "\n\n")

        # III. 校准质量综合评估
        f.write("III. 校准质量综合评估\n")
        f.write("-" * 80 + "\n\n")

        f.write("整体评估:\n")
        if overall_grade == "优秀":
            f.write("  ✓ 模型展现出优秀的校准性能\n")
            f.write("  ✓ 预测概率高度可靠,可直接用于决策\n\n")
        elif overall_grade == "良好":
            f.write("  ✓ 模型展现出良好的校准性能\n")
            f.write("  ✓ 预测概率可用于决策,具有中等置信度\n\n")
        elif overall_grade == "一般":
            f.write("  ⚠ 模型校准性能一般\n")
            f.write("  ⚠ 预测概率存在偏差,建议谨慎使用\n\n")
        else:
            f.write("  ✗ 模型校准性能较差\n")
            f.write("  ✗ 预测概率不可靠,不建议直接用于决策\n\n")

        f.write("各类别分析:\n")
        for class_idx, class_name in enumerate(class_names):
            brier_class = per_class_brier[class_idx]
            grade, _ = interpret_brier_score(brier_class, n_classes)

            if grade == "优秀":
                f.write(f"  • {class_name}类别: 校准优秀,概率高度可信\n")
            elif grade == "良好":
                f.write(f"  • {class_name}类别: 校准良好,概率较为可靠\n")
            elif grade == "一般":
                f.write(f"  ⚠ {class_name}类别: 检测到中等程度的校准偏差\n")
            else:
                f.write(f"  ✗ {class_name}类别: 校准较差,需要改进\n")

        f.write("\n建议:\n")

        # 根据结果给出建议
        poor_classes = [class_names[i] for i, grade in
                        enumerate([interpret_brier_score(bs, n_classes)[0]
                                   for bs in per_class_brier])
                        if grade in ["一般", "较差"]]

        if poor_classes:
            f.write(f"  1. 考虑应用校准方法(如Platt Scaling、Isotonic Regression)")
            f.write(f"改进以下类别的概率估计: {', '.join(poor_classes)}\n")
        else:
            f.write("  1. 模型校准性能良好,可考虑直接部署\n")

        if overall_grade in ["优秀", "良好"]:
            f.write("  2. 模型适合部署,可使用概率阈值进行决策\n")
        else:
            f.write("  2. 建议先进行校准优化后再部署\n")

        f.write("  3. 持续监控新数据上的校准漂移\n\n")

        f.write("-" * 80 + "\n\n")

        # IV. Brier Score解释指南
        f.write("IV. Brier Score解释指南\n")
        f.write("-" * 80 + "\n\n")

        if n_classes == 2:
            f.write("Brier Score范围 (2分类场景):\n")
            f.write("  • 优秀:  BS < 0.10\n")
            f.write("  • 良好:  0.10 ≤ BS < 0.15\n")
            f.write("  • 一般:  0.15 ≤ BS < 0.20\n")
            f.write("  • 较差:  BS ≥ 0.20\n\n")
            f.write("Brier Score越低越好 (0为完美校准)\n")
            f.write("随机分类器基准: BS = 0.25\n\n")
        else:
            baseline = 1.0 - (1.0 / n_classes)
            f.write(f"Brier Score范围 ({n_classes}分类场景):\n")
            f.write(f"  • 优秀:  BS < {baseline * 0.4:.2f}\n")
            f.write(f"  • 良好:  {baseline * 0.4:.2f} ≤ BS < {baseline * 0.6:.2f}\n")
            f.write(f"  • 一般:  {baseline * 0.6:.2f} ≤ BS < {baseline * 0.8:.2f}\n")
            f.write(f"  • 较差:  BS ≥ {baseline * 0.8:.2f}\n\n")
            f.write("Brier Score越低越好 (0为完美校准)\n")
            f.write(f"随机分类器基准: BS = {baseline:.2f}\n\n")

        f.write("=" * 80 + "\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")


# ==================== 主函数接口 ====================

def evaluate_model_calibration(
    csv_path=None,
    annotation_path='cls_test.txt',
    model_instance=None,
    class_names=None,
    output_dir='metrics_out/calibration_analysis',
    n_bins=10,
    binning_strategy='uniform',
    verbose=True
):
    """
    评估模型的校准性能 (主函数接口)

    参数:
        csv_path: CSV文件路径 (优先使用, 来自eval.py的detailed_predictions.csv)
        annotation_path: 测试数据标注文件路径, 默认'cls_test.txt'
        model_instance: Eval_Classification实例 (实时推理模式)
        class_names: 类别名称列表 (如果为None则自动推断)
        output_dir: 输出目录, 默认'metrics_out/calibration_analysis'
        n_bins: 分桶数量, 默认10
        binning_strategy: 分桶策略, 'uniform' (等宽) 或 'quantile' (等频)
        verbose: 是否打印详细信息, 默认True

    返回:
        dict: {
            'overall_brier_score': float,
            'per_class_brier_scores': list,
            'overall_calibration': dict,
            'per_class_calibrations': list
        }

    输出文件:
        - {output_dir}/calibration_overall.png
        - {output_dir}/calibration_per_class.png
        - {output_dir}/calibration_report.txt

    使用示例:
        # 从CSV加载
        results = evaluate_model_calibration(
            csv_path='metrics_out/model_A/detailed_predictions.csv',
            output_dir='metrics_out/calibration_model_A'
        )

        # 实时推理
        from eval import Eval_Classification
        model = Eval_Classification()
        results = evaluate_model_calibration(
            model_instance=model,
            annotation_path='cls_test.txt'
        )
    """
    if verbose:
        print("\n" + "=" * 80)
        print("模型校准性能评估")
        print("=" * 80)
        print("\n[1/5] 加载预测数据...")

    # 1. 加载数据
    labels, probs, class_names = load_predictions_for_calibration(
        csv_path=csv_path,
        annotation_path=annotation_path,
        model_instance=model_instance,
        class_names=class_names
    )

    n_samples = len(labels)
    n_classes = probs.shape[1]

    if verbose:
        print(f"  ✓ 加载完成: {n_samples}个样本, {n_classes}个类别")
        print(f"  ✓ 类别名称: {class_names}")

    # 2. 计算Brier Score
    if verbose:
        print("\n[2/5] 计算Brier Score...")

    overall_brier = compute_brier_score_multiclass(labels, probs)
    per_class_brier = compute_per_class_brier_score(labels, probs, n_classes)

    if verbose:
        print(f"  ✓ 整体Brier Score: {overall_brier:.4f}")
        for i, (class_name, brier) in enumerate(zip(class_names, per_class_brier)):
            print(f"  ✓ 类别{i} ({class_name}) Brier Score: {brier:.4f}")

    # 3. 计算校准曲线
    if verbose:
        print("\n[3/5] 计算校准曲线...")

    overall_calibration = compute_calibration_curve(labels, probs, n_bins, binning_strategy)

    per_class_calibrations = []
    for class_idx in range(n_classes):
        calib = compute_per_class_calibration(labels, probs, class_idx, n_bins, binning_strategy)
        per_class_calibrations.append(calib)

    if verbose:
        print(f"  ✓ 校准曲线计算完成 (分桶数: {n_bins}, 策略: {binning_strategy})")

    # 4. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 5. 生成可视化
    if verbose:
        print("\n[4/5] 生成可视化...")

    overall_plot_path = os.path.join(output_dir, "calibration_overall.png")
    plot_overall_calibration(overall_calibration, overall_plot_path)
    if verbose:
        print(f"  ✓ 整体校准图已保存: {overall_plot_path}")

    per_class_plot_path = os.path.join(output_dir, "calibration_per_class.png")
    plot_per_class_calibration(labels, probs, class_names, n_bins,
                                per_class_plot_path, binning_strategy)
    if verbose:
        print(f"  ✓ 各类别校准图已保存: {per_class_plot_path}")

    # 6. 生成文本报告
    if verbose:
        print("\n[5/5] 生成文本报告...")

    report_path = os.path.join(output_dir, "calibration_report.txt")
    generate_calibration_report(
        overall_calibration, per_class_calibrations,
        overall_brier, per_class_brier, class_names,
        report_path, n_bins, binning_strategy
    )

    if verbose:
        print(f"  ✓ 文本报告已保存: {report_path}")

    # 7. 打印摘要
    if verbose:
        print("\n" + "=" * 80)
        print("评估完成!")
        print("=" * 80)
        print(f"\n整体Brier Score: {overall_brier:.4f}")
        overall_grade, overall_desc = interpret_brier_score(overall_brier, n_classes)
        print(f"整体评价: {overall_grade} - {overall_desc}")

        print(f"\n各类别Brier Score:")
        for i, (class_name, brier) in enumerate(zip(class_names, per_class_brier)):
            grade, _ = interpret_brier_score(brier, n_classes)
            print(f"  • 类别{i} ({class_name}): {brier:.4f} ({grade})")

        print(f"\n输出文件:")
        print(f"  1. {overall_plot_path}")
        print(f"  2. {per_class_plot_path}")
        print(f"  3. {report_path}")
        print("\n" + "=" * 80 + "\n")

    # 返回结果
    return {
        'overall_brier_score': overall_brier,
        'per_class_brier_scores': per_class_brier,
        'overall_calibration': overall_calibration,
        'per_class_calibrations': per_class_calibrations
    }


# ==================== 示例用法 ====================

if __name__ == '__main__':
    # 示例: 从CSV加载
    results = evaluate_model_calibration(
        csv_path='../metrics_out/inception_resnet_v2_cls_test/detailed_predictions.csv',
        output_dir='../metrics_out/inception_resnet_v2_cls_test/calibration',
        n_bins=10,
        binning_strategy='uniform'
    )

    print(f"\n整体Brier Score: {results['overall_brier_score']:.4f}")
    print(f"各类别Brier Score: {results['per_class_brier_scores']}")
