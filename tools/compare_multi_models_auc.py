# -*- coding: utf-8 -*-
"""
多模型AUC两两比较工具

使用配对Bootstrap方法比较多个深度学习模型的Macro AUC性能差异
生成N×N比较矩阵和专业热力图，包含p值、显著性标记和AUC差值

功能特性：
- 批量加载多个模型的预测结果
- 两两配对Bootstrap统计检验（1000次重采样）
- 生成热力图：上三角显示p值+显著性，下三角显示AUC差值
- 输出完整比较矩阵CSV和文本报告

使用方法：
在脚本底部手动指定model_folders列表，然后运行：
python compare_multi_models_auc.py

日期：2026-02-17
"""

import os
import sys
import warnings
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# 设置全局字体和样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False

# 导入现有工具的核心函数
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from compare_models_auc import (
    extract_probability_columns,
    compute_single_metric,
    paired_bootstrap_comparison
)


# ==================== 数据加载函数 ====================

def load_multi_model_data(
    model_folders: List[str],
    base_path: str = '../metrics_out'
) -> Tuple[List[Dict[str, np.ndarray]], List[str], int]:
    """
    批量加载多个模型的预测数据

    参数:
        model_folders: 模型文件夹名称列表（如 ['resnet50_cls_val', 'densenet121_cls_val']）
        base_path: metrics_out基础路径

    返回:
        data_list: 数据字典列表，每个字典包含 {'labels', 'preds', 'probs'}
        class_names: 类别名称列表
        n_samples: 样本数量

    异常:
        FileNotFoundError: CSV文件不存在
        ValueError: 数据格式不符合要求或标签不一致
    """
    print("\n" + "=" * 70)
    print("多模型AUC两两比较工具")
    print("=" * 70)
    print(f"\n[1/5] 加载数据...")

    data_list = []
    reference_labels = None
    n_samples = None
    class_names = None

    for i, folder in enumerate(model_folders):
        csv_path = os.path.join(base_path, folder, 'detailed_predictions.csv')

        # 文件存在性检查
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"CSV文件不存在: {csv_path}\n"
                f"请确认模型文件夹名称正确，且已运行eval.py生成预测结果"
            )

        # 加载CSV
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"  [{i+1}/{len(model_folders)}] {folder}: {len(df)}样本")

        # 列完整性检查
        required_cols = {'path', 'true', 'predict'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"模型 {folder} 的CSV缺少必需列: {missing_cols}\n"
                f"可用列: {df.columns.tolist()}"
            )

        # 提取标签和预测
        labels = df['true'].values
        preds = df['predict'].values

        # 样本数量一致性检查
        if n_samples is None:
            n_samples = len(labels)
        elif len(labels) != n_samples:
            raise ValueError(
                f"模型 {folder} 的样本数量({len(labels)})与第一个模型({n_samples})不一致!\n"
                f"配对Bootstrap要求所有模型使用相同的测试集"
            )

        # 真实标签一致性检查（配对前提）
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(labels, reference_labels):
            raise ValueError(
                f"模型 {folder} 的真实标签与第一个模型不一致!\n"
                f"这违反了配对Bootstrap的前提条件\n"
                f"请确保所有模型使用完全相同的测试集（相同顺序）"
            )

        # 提取类别数量
        n_classes = len(np.unique(labels))

        # 提取概率矩阵
        try:
            probs = extract_probability_columns(df, n_classes)
        except ValueError as e:
            raise ValueError(
                f"模型 {folder} 的概率列提取失败: {str(e)}\n"
                f"请重新运行 eval.py 生成包含完整概率的CSV文件"
            )

        # 提取类别名称（仅第一次）
        if class_names is None:
            prob_cols = [col for col in df.columns if 'prob' in col.lower()]
            if len(prob_cols) == n_classes:
                class_names = [col.replace('_probability', '').replace('_prob', '')
                              for col in sorted(prob_cols)]
            else:
                class_names = [f'Class_{i}' for i in range(n_classes)]

        # 存储数据
        data_list.append({
            'labels': labels,
            'preds': preds,
            'probs': probs
        })

    print(f"\n  ✓ 成功加载 {len(model_folders)} 个模型")
    print(f"  ✓ 样本数量: {n_samples}")
    print(f"  ✓ 类别数量: {len(class_names)}")
    print(f"  ✓ 类别名称: {class_names}")

    return data_list, class_names, n_samples


# ==================== 两两比较函数 ====================

def pairwise_bootstrap_comparison(
    data_list: List[Dict[str, np.ndarray]],
    model_names: List[str],
    n_bootstrap: int = 1000,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    执行N×N两两Bootstrap比较

    参数:
        data_list: 数据字典列表，每个字典包含 {'labels', 'preds', 'probs'}
        model_names: 模型显示名称列表
        n_bootstrap: Bootstrap重采样次数
        random_state: 随机种子

    返回:
        pvalue_matrix: N×N p值矩阵
        auc_diff_matrix: N×N AUC差值矩阵（行模型 - 列模型）
        significance_matrix: N×N 显著性矩阵（True/False）
        model_aucs: 每个模型的Macro AUC字典
    """
    print(f"\n[2/5] 执行两两Bootstrap比较...")

    n_models = len(data_list)
    pvalue_matrix = np.zeros((n_models, n_models))
    auc_diff_matrix = np.zeros((n_models, n_models))
    significance_matrix = np.zeros((n_models, n_models), dtype=bool)
    model_aucs = {}

    # 计算每个模型的Macro AUC
    print(f"\n  计算各模型Macro AUC:")
    for i, (data, name) in enumerate(zip(data_list, model_names)):
        auc = compute_single_metric(
            'macro_auc',
            data['labels'],
            data['preds'],
            data['probs']
        )
        model_aucs[name] = auc
        print(f"    {name}: {auc:.4f}")

    # 两两比较
    print(f"\n  执行 {n_models}×{n_models} 两两比较 (共{n_models*(n_models-1)//2}对):")
    comparison_count = 0
    total_comparisons = n_models * (n_models - 1) // 2

    for i in range(n_models):
        for j in range(i + 1, n_models):
            comparison_count += 1
            print(f"    [{comparison_count}/{total_comparisons}] {model_names[i]} vs {model_names[j]}")

            # 调用现有的配对Bootstrap函数
            try:
                results = paired_bootstrap_comparison(
                    labels=data_list[i]['labels'],
                    preds1=data_list[i]['preds'],
                    preds2=data_list[j]['preds'],
                    probs1=data_list[i]['probs'],
                    probs2=data_list[j]['probs'],
                    n_bootstrap=n_bootstrap,
                    metrics=['macro_auc'],
                    random_state=random_state
                )

                # 提取结果
                macro_auc_result = results['macro_auc']
                p_value = macro_auc_result['p_value']
                auc_diff = macro_auc_result['diff_original']
                significant = macro_auc_result['significant']

                # 填充矩阵（对称）
                pvalue_matrix[i, j] = p_value
                pvalue_matrix[j, i] = p_value

                auc_diff_matrix[i, j] = auc_diff
                auc_diff_matrix[j, i] = -auc_diff

                significance_matrix[i, j] = significant
                significance_matrix[j, i] = significant

                print(f"        p={p_value:.4f}, ΔAUC={auc_diff:+.4f}, 显著={'是' if significant else '否'}")

            except Exception as e:
                warnings.warn(f"比较失败 ({model_names[i]} vs {model_names[j]}): {str(e)}")
                pvalue_matrix[i, j] = np.nan
                pvalue_matrix[j, i] = np.nan
                auc_diff_matrix[i, j] = np.nan
                auc_diff_matrix[j, i] = np.nan

    print(f"\n  ✓ 完成所有两两比较")

    return pvalue_matrix, auc_diff_matrix, significance_matrix, model_aucs


# ==================== 热力图绘制函数 ====================

def plot_pvalue_heatmap(
    pvalue_matrix: np.ndarray,
    auc_diff_matrix: np.ndarray,
    model_aucs: Dict[str, float],
    model_names: List[str],
    output_path: str
):
    """
    绘制p值热力图（完全参考test.py的实现）

    布局：
    - 下三角：p值 + 显著性标记（***/**/*）
    - 对角线：模型自身Macro AUC

    参数:
        pvalue_matrix: N×N p值矩阵
        auc_diff_matrix: N×N AUC差值矩阵
        model_aucs: 每个模型的Macro AUC字典
        model_names: 模型显示名称列表
        output_path: 输出PNG路径
    """
    print(f"\n[3/5] 生成热力图...")

    n_models = len(model_names)

    # 准备数据矩阵：下三角显示p值，对角线显示AUC
    data_values = []
    for i in range(n_models):
        row = []
        for j in range(n_models):
            if i == j:
                # 对角线：模型AUC（使用特殊值1.0标记）
                row.append(1.0)
            elif i > j:
                # 下三角：p值
                row.append(pvalue_matrix[i, j])
            else:
                # 上三角：设为NaN（将被mask遮挡）
                row.append(np.nan)
        data_values.append(row)

    df = pd.DataFrame(data_values, index=model_names, columns=model_names)

    # 准备标注矩阵：显著性标记或AUC值
    annot_labels = []
    for i in range(n_models):
        row = []
        for j in range(n_models):
            if i == j:
                # 对角线
                auc_value = model_aucs[model_names[i]]
                row.append(f'')
            elif i > j:
                # 下三角：显示显著性标记
                p_value = pvalue_matrix[i, j]
                if np.isnan(p_value):
                    row.append('')
                else:
                    if p_value < 0.001:
                        row.append('***')
                    elif p_value < 0.01:
                        row.append('**')
                    elif p_value < 0.05:
                        row.append('*')
                    else:
                        row.append('')
            else:
                # 上三角：空白
                row.append('')
        annot_labels.append(row)

    annot_df = pd.DataFrame(annot_labels, index=model_names, columns=model_names)

    # 创建掩码：遮挡上三角（不包含对角线）
    mask = np.triu(np.ones_like(df, dtype=bool), k=1)

    # 设置绘图（参考test.py）
    plt.figure(figsize=(10,8), dpi=150)
    ax = plt.gca()

    # 关键步骤：创建与主图高度一致的色带轴（参考test.py）
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)

    # 绘制热力图（参考test.py的配置）
    sns.heatmap(
        df,
        mask=mask,
        annot=annot_df,
        fmt='',
        cmap="RdYlBu_r",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=1.5,
        linecolor='white',
        ax=ax,
        cbar_ax=cax,
        cbar_kws={"label": "P value"},
        annot_kws={"color": "white"}
    )

    # 调整样式（参考test.py）
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha='right',
        color='black'
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        color='black'
    )

    # 调整色带标签（参考test.py）
    cbar = ax.collections[0].colorbar
    cbar.set_label("P value")

    ax.tick_params(left=False, bottom=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 热力图已保存: {output_path}")


# ==================== 主函数 ====================

def compare_multi_models_auc(
    model_folders: List[str],
    model_names: Optional[List[str]] = None,
    base_path: str = '../metrics_out',
    n_bootstrap: int = 1000,
    output_dir: str = '../metrics_out/multi_model_comparison',
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    多模型AUC两两比较主函数

    参数:
        model_folders: 模型文件夹名称列表（如 ['resnet50_cls_val', 'densenet121_cls_val']）
        model_names: 显示名称列表（可选，默认从文件夹名提取）
        base_path: metrics_out基础路径
        n_bootstrap: Bootstrap重采样次数
        output_dir: 结果输出目录
        random_state: 随机种子
        verbose: 是否打印详细日志

    返回:
        results: 结果字典，包含比较矩阵、p值矩阵、AUC差值矩阵等
    """
    try:
        # 自动生成模型名称
        if model_names is None:
            model_names = []
            for folder in model_folders:
                # 移除后缀并格式化
                name = folder.replace('_cls_val', '').replace('_cls_test', '')
                name = name.replace('_', ' ').title().replace(' ', '')
                model_names.append(name)

        # 验证参数
        if len(model_folders) < 2:
            raise ValueError("至少需要2个模型进行比较")

        if len(model_names) != len(model_folders):
            raise ValueError(
                f"model_names长度({len(model_names)})与model_folders长度({len(model_folders)})不一致"
            )

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: 加载数据
        data_list, class_names, n_samples = load_multi_model_data(
            model_folders, base_path
        )

        # Step 2: 两两比较
        pvalue_matrix, auc_diff_matrix, significance_matrix, model_aucs = \
            pairwise_bootstrap_comparison(
                data_list, model_names, n_bootstrap, random_state
            )

        # Step 3: 生成热力图
        heatmap_path = os.path.join(output_dir, 'pvalue_heatmap.png')
        plot_pvalue_heatmap(
            pvalue_matrix, auc_diff_matrix, model_aucs, model_names, heatmap_path
        )

        # Step 4: 保存比较矩阵CSV
        print(f"\n[4/5] 保存比较矩阵...")
        matrix_csv_path = os.path.join(output_dir, 'comparison_matrix.csv')

        # 创建DataFrame
        matrix_data = []
        for i, name_i in enumerate(model_names):
            row = [name_i]
            for j, name_j in enumerate(model_names):
                if i == j:
                    # 对角线：模型AUC
                    row.append(f"{model_aucs[name_i]:.4f}")
                elif i < j:
                    # 上三角：p值 + 显著性
                    p_value = pvalue_matrix[i, j]
                    if np.isnan(p_value):
                        row.append('N/A')
                    else:
                        if p_value < 0.01:
                            marker = '***'
                        elif p_value < 0.05:
                            marker = '**'
                        elif p_value < 0.1:
                            marker = '*'
                        else:
                            marker = 'ns'
                        row.append(f"p={p_value:.4f}{marker}")
                else:
                    # 下三角：AUC差值
                    auc_diff = auc_diff_matrix[i, j]
                    if np.isnan(auc_diff):
                        row.append('N/A')
                    else:
                        row.append(f"{auc_diff:+.4f}")
            matrix_data.append(row)

        df_matrix = pd.DataFrame(matrix_data, columns=['Model'] + model_names)
        df_matrix.to_csv(matrix_csv_path, index=False, encoding='utf-8-sig')
        print(f"  ✓ 比较矩阵已保存: {matrix_csv_path}")

        # Step 5: 生成文本报告
        print(f"\n[5/5] 生成文本报告...")
        report_path = os.path.join(output_dir, 'auc_comparison_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("多模型AUC两两比较报告\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Bootstrap重采样次数: {n_bootstrap}\n")
            f.write(f"样本数量: {n_samples}\n")
            f.write(f"类别数量: {len(class_names)}\n")
            f.write(f"类别名称: {class_names}\n\n")

            # 模型性能排名
            f.write("-" * 70 + "\n")
            f.write("模型性能排名（按Macro AUC降序）:\n")
            f.write("-" * 70 + "\n")
            sorted_models = sorted(model_aucs.items(), key=lambda x: x[1], reverse=True)
            for rank, (name, auc) in enumerate(sorted_models, 1):
                f.write(f"{rank}. {name}: {auc:.4f}\n")

            # 显著性差异总结
            f.write("\n" + "-" * 70 + "\n")
            f.write("显著性差异总结:\n")
            f.write("-" * 70 + "\n")

            significant_pairs = []
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    if significance_matrix[i, j]:
                        p_value = pvalue_matrix[i, j]
                        auc_diff = auc_diff_matrix[i, j]

                        if p_value < 0.01:
                            marker = '***'
                        elif p_value < 0.05:
                            marker = '**'
                        else:
                            marker = '*'

                        if auc_diff > 0:
                            better_model = model_names[i]
                            worse_model = model_names[j]
                        else:
                            better_model = model_names[j]
                            worse_model = model_names[i]
                            auc_diff = -auc_diff

                        significant_pairs.append({
                            'better': better_model,
                            'worse': worse_model,
                            'p_value': p_value,
                            'auc_diff': auc_diff,
                            'marker': marker
                        })

            # 按AUC差值降序排序
            significant_pairs.sort(key=lambda x: x['auc_diff'], reverse=True)

            if significant_pairs:
                for pair in significant_pairs:
                    f.write(
                        f"- {pair['better']} vs {pair['worse']}: "
                        f"p={pair['p_value']:.4f} {pair['marker']} "
                        f"(ΔAUC={pair['auc_diff']:+.4f}, {pair['better']}显著优于{pair['worse']})\n"
                    )
            else:
                f.write("所有模型间均无显著差异\n")

            # 完整比较矩阵
            f.write("\n" + "-" * 70 + "\n")
            f.write("完整比较矩阵:\n")
            f.write("-" * 70 + "\n")
            f.write("说明: 对角线=模型AUC | 上三角=p值+显著性 | 下三角=AUC差值\n\n")

            # 表头
            f.write(f"{'Model':<20}")
            for name in model_names:
                f.write(f"{name:<20}")
            f.write("\n" + "-" * (20 + 20 * len(model_names)) + "\n")

            # 表格内容
            for i, name_i in enumerate(model_names):
                f.write(f"{name_i:<20}")
                for j, name_j in enumerate(model_names):
                    if i == j:
                        f.write(f"{model_aucs[name_i]:<20.4f}")
                    elif i < j:
                        p_value = pvalue_matrix[i, j]
                        if np.isnan(p_value):
                            f.write(f"{'N/A':<20}")
                        else:
                            if p_value < 0.01:
                                marker = '***'
                            elif p_value < 0.05:
                                marker = '**'
                            elif p_value < 0.1:
                                marker = '*'
                            else:
                                marker = 'ns'
                            f.write(f"p={p_value:.4f}{marker:<13}")
                    else:
                        auc_diff = auc_diff_matrix[i, j]
                        if np.isnan(auc_diff):
                            f.write(f"{'N/A':<20}")
                        else:
                            f.write(f"{auc_diff:+.4f}{'':<15}")
                f.write("\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write(f"所有结果已保存至: {output_dir}/\n")
            f.write("=" * 70 + "\n")

        print(f"  ✓ 文本报告已保存: {report_path}")

        # 打印完成信息
        print("\n" + "=" * 70)
        print("多模型比较完成!")
        print("=" * 70)
        print(f"\n输出文件:")
        print(f"  1. 热力图: {heatmap_path}")
        print(f"  2. 比较矩阵CSV: {matrix_csv_path}")
        print(f"  3. 文本报告: {report_path}")
        print(f"\n所有结果已保存至: {output_dir}/\n")

        # 返回结果
        return {
            'pvalue_matrix': pvalue_matrix,
            'auc_diff_matrix': auc_diff_matrix,
            'significance_matrix': significance_matrix,
            'model_aucs': model_aucs,
            'model_names': model_names,
            'n_samples': n_samples,
            'n_models': len(model_names)
        }

    except Exception as e:
        if verbose:
            print(f"\n错误: {str(e)}\n")
            import traceback
            traceback.print_exc()
        raise


# ==================== 使用示例 ====================

if __name__ == '__main__':
    # 手动指定要比较的模型文件夹
    model_folders = [
        'densenet121_cls_val',
        'resnet50_cls_val',
        'vgg19_bn_cls_val',
        'tf_inception_v3_cls_val',
        'xception_cls_val',
        'inception_resnet_v2_cls_val',
    ]

    # 可选：自定义显示名称（如果不指定，将自动从文件夹名提取）
    model_names = [
        'DenseNet121',
        'ResNet50',
        'VGG19',
        'InceptionV3',
        'Xception',
        'InceptionResNetV2'
    ]

    # 执行比较
    results = compare_multi_models_auc(
        model_folders=model_folders,
        model_names=model_names,
        base_path='../metrics_out',
        n_bootstrap=1000,
        output_dir='../metrics_out/multi_model_comparison',
        random_state=42,
        verbose=True
    )

    # 可选：访问结果
    print("\n模型AUC排名:")
    sorted_models = sorted(results['model_aucs'].items(), key=lambda x: x[1], reverse=True)
    for rank, (name, auc) in enumerate(sorted_models, 1):
        print(f"  {rank}. {name}: {auc:.4f}")

