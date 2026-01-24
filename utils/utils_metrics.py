import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def evaluteTop1_5(classfication, lines, metrics_out_path):
    correct_1 = 0
    correct_5 = 0
    preds   = []
    labels  = []
    total = len(lines)
    for index, line in enumerate(lines):
        annotation_path = line.split(';')[1].split('\n')[0]
        x = Image.open(annotation_path)
        y = int(line.split(';')[0])

        pred        = classfication.detect_image(x)
        pred_1      = np.argmax(pred)
        correct_1   += pred_1 == y

        pred_5      = np.argsort(pred)[::-1]
        pred_5      = pred_5[:5]
        correct_5   += y in pred_5

        preds.append(pred_1)
        labels.append(y)
        if index % 100 == 0:
            print("[%d/%d]"%(index, total))

    hist        = fast_hist(np.array(labels), np.array(preds), len(classfication.class_names))
    Recall      = per_class_Recall(hist)
    Precision   = per_class_Precision(hist)

    show_results(metrics_out_path, hist, Recall, Precision, classfication.class_names)
    return correct_1 / total, correct_5 / total, Recall, Precision

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)

def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    plt.rcParams['font.family'] = 'Times New Roman'
    fig     = plt.gcf()
    axes    = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val)
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()

def show_results(miou_out_path, hist, Recall, Precision, name_classes, tick_font_size = 12):
    draw_plot_func(Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))

def evaluteRecall(classfication, lines, metrics_out_path):
    correct = 0
    total = len(lines)

    preds   = []
    labels  = []
    for index, line in enumerate(lines):
        annotation_path = line.split(';')[1].split()[0]
        x = Image.open(annotation_path)
        y = int(line.split(';')[0])

        pred = classfication.detect_image(x)
        pred = np.argmax(pred)

        preds.append(pred)
        labels.append(y)

    hist        = fast_hist(labels, preds, len(classfication.class_names))
    Recall      = per_class_Recall(hist)
    Precision   = per_class_Precision(hist)

    show_results(metrics_out_path, hist, Recall, Precision, classfication.class_names)
    return correct / total

def draw_confusion_matrix_detailed(hist, class_names, output_path, title="Confusion Matrix"):
    """
    绘制详细混淆矩阵，包含数值和百分比
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 输入验证
    hist = np.array(hist)
    expected_shape = (len(class_names), len(class_names))
    
    if hist.shape != expected_shape:
        print(f"警告: 混淆矩阵形状不匹配！期望{expected_shape}，实际{hist.shape}")
        # 如果形状不对，尝试修正
        if hist.size == expected_shape[0] * expected_shape[1]:
            hist = hist.reshape(expected_shape)
        else:
            raise ValueError(f"无法修正混淆矩阵形状: {hist.shape} -> {expected_shape}")
    
    print(f"混淆矩阵验证: 形状={hist.shape}, 类别数={len(class_names)}")

    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False

    # 计算百分比，防止除零错误
    row_sums = hist.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    hist_percent = hist.astype('float') / row_sums * 100
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热力图
    sns.heatmap(hist_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'}, ax=ax)
    
    # 在每个格子中添加实际数量
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j+0.5, i+0.7, f'({int(hist[i, j])})', 
                          ha="center", va="center", fontsize=10, color='red')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def draw_metrics_comparison_chart(recall, precision, f1_scores, class_names, output_path):
    """
    绘制分类指标对比图
    """
    import matplotlib.pyplot as plt
    
    # 输入验证和数据转换
    recall = np.array(recall).flatten() if not isinstance(recall, np.ndarray) else recall.flatten()
    precision = np.array(precision).flatten() if not isinstance(precision, np.ndarray) else precision.flatten()
    f1_scores = np.array(f1_scores).flatten() if not isinstance(f1_scores, np.ndarray) else f1_scores.flatten()
    
    # 数据形状验证
    expected_len = len(class_names)
    if len(recall) != expected_len or len(precision) != expected_len or len(f1_scores) != expected_len:
        print(f"警告: 数据形状不匹配！")
        print(f"  类别数: {expected_len}")
        print(f"  recall长度: {len(recall)}")  
        print(f"  precision长度: {len(precision)}")
        print(f"  f1_scores长度: {len(f1_scores)}")
        
        # 截取到正确长度
        recall = recall[:expected_len] if len(recall) > expected_len else recall
        precision = precision[:expected_len] if len(precision) > expected_len else precision
        f1_scores = f1_scores[:expected_len] if len(f1_scores) > expected_len else f1_scores
        
    print(f"绘图数据验证: recall={recall.shape}, precision={precision.shape}, f1={f1_scores.shape}")

    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False

    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制三组柱状图
    bars1 = ax.bar(x - width, recall * 100, width, label='Recall', color='skyblue', alpha=0.8)
    bars2 = ax.bar(x, precision * 100, width, label='Precision', color='lightcoral', alpha=0.8)
    bars3 = ax.bar(x + width, f1_scores * 100, width, label='F1 Score', color='lightgreen', alpha=0.8)
    
    # 添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    ax.set_xlabel('Classification Category', fontsize=14)
    ax.set_ylabel('Performance Metrics (%)', fontsize=14)
    ax.set_title('Per-Class Classification Performance', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend(fontsize=12)
    ax.set_ylim(0, max(max(recall), max(precision), max(f1_scores)) * 110)
    
    # 添加参考线
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Baseline')
    ax.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='70% Good')
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% Excellent')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_classification_report(hist, recall, precision, auc_metrics, specificity,
                                 accuracy, ci_results, class_names, output_path,
                                 samples_per_class=None, minority_idx=None):
    """
    生成完整的分类性能报告(合并基础版和高级版)

    参数:
        hist: 混淆矩阵
        recall: 各类召回率
        precision: 各类精确度
        auc_metrics: AUC指标字典(per_class_auc, macro_auc, micro_auc)
        specificity: 各类特异度
        ci_results: 置信区间结果
        class_names: 类别名称列表
        output_path: 输出目录
        samples_per_class: 各类别样本数量(可选)
        minority_idx: 少数类别索引,如果为None则自动识别

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

    # 生成可视化
    confusion_matrix_path = os.path.join(output_path, "confusion_matrix_detailed.png")
    metrics_chart_path = os.path.join(output_path, "metrics_comparison_chart.png")

    draw_confusion_matrix_detailed(hist, class_names, confusion_matrix_path)
    draw_metrics_comparison_chart(recall, precision, f1_scores, class_names, metrics_chart_path)

    print(f"[OK] 生成详细混淆矩阵: {confusion_matrix_path}")
    print(f"[OK] 生成指标对比图表: {metrics_chart_path}")

    # 生成文本报告
    report_path = os.path.join(output_path, "classification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Image Classification Performance Report\n")
        f.write("=" * 80 + "\n\n")

        # Part I: Detailed per-class metrics with confidence intervals
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
               f"{ci_results['macro_f1']['upper']:.4f}])\n\n")

        # Balanced Accuracy and Performance Gap
        f.write(f"Balanced Accuracy:    {np.mean(recall):.4f}\n")
        f.write(f"Performance Gap (F1): {max(f1_scores) - min(f1_scores):.4f}\n\n")

        # Part III: Key findings and performance analysis
        f.write("III. Key Findings and Analysis\n")
        f.write("-" * 80 + "\n\n")

        minority_class_idx = minority_idx
        minority_f1 = f1_scores[minority_class_idx]
        minority_recall = recall[minority_class_idx]
        minority_auc = auc_metrics['per_class_auc'][minority_class_idx]

        f.write(f"Minority Class ({class_names[minority_class_idx]}) Performance:\n")
        f.write(f"   F1 Score:  {minority_f1:.4f} - ")
        if minority_f1 > 0.7:
            f.write("Excellent\n")
        elif minority_f1 > 0.5:
            f.write("Good\n")
        elif minority_f1 > 0.3:
            f.write("Fair\n")
        else:
            f.write("Needs Improvement\n")

        f.write(f"   Recall:    {minority_recall:.4f} - ")
        if minority_recall > 0.8:
            f.write("Low false negative risk\n")
        elif minority_recall > 0.6:
            f.write("Moderate false negative risk\n")
        else:
            f.write("High false negative risk\n")

        f.write(f"   AUC:       {minority_auc:.4f} - ")
        if minority_auc > 0.8:
            f.write("Excellent discrimination\n")
        elif minority_auc > 0.7:
            f.write("Good discrimination\n")
        else:
            f.write("Needs improvement\n")

        f.write("\n")
        f.write("Deployment Recommendations:\n")
        if minority_f1 > 0.7 and minority_recall > 0.8 and minority_auc > 0.8:
            f.write("   ✓ Excellent - Ready for production deployment\n")
        elif minority_f1 > 0.5:
            f.write("   ⚠ Good - Recommended for supervised use\n")
        else:
            f.write("   ✗ Needs Improvement - Not recommended for direct deployment\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"[OK] 生成完整分类报告: {report_path}")

    return f1_scores


#---------------------------------------------------------#
#   评估指标计算和可视化函数
#---------------------------------------------------------#

def compute_specificity(hist):
    """
    计算每类的特异度（Specificity）- One-vs-Rest方式

    特异度（Specificity）= TN / (TN + FP)
    表示在所有实际为阴性的样本中，被正确预测为阴性的比例

    参数：
        hist: 混淆矩阵 (n_classes, n_classes)

    返回：
        specificity: 各类特异度数组 (n_classes,)
    """
    n_classes = hist.shape[0]
    specificity = np.zeros(n_classes)

    for i in range(n_classes):
        # TN = 所有非i类样本中被正确分类的数量
        TN = hist.sum() - hist[i, :].sum() - hist[:, i].sum() + hist[i, i]
        # FP = 被错误分类为i类的数量
        FP = hist[:, i].sum() - hist[i, i]

        # 防止除零
        specificity[i] = TN / np.maximum(TN + FP, 1)

    return specificity


def compute_auc_metrics(labels, probs, n_classes):
    """
    计算AUC指标（每类AUC、Macro-AUC、Micro-AUC）

    参数：
        labels: 真实标签 (n_samples,)
        probs: 预测概率 (n_samples, n_classes)
        n_classes: 类别数量

    返回：
        dict: {
            'per_class_auc': [auc_0, auc_1, auc_2],
            'macro_auc': float,
            'micro_auc': float
        }
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    # 将标签转为one-hot编码
    labels_onehot = label_binarize(labels, classes=range(n_classes))
    if labels_onehot.shape[1] == 1:
        # 如果只有2类，label_binarize返回单列，需要手动扩展
        labels_onehot = np.hstack([1 - labels_onehot, labels_onehot])

    # 计算每类AUC（One-vs-Rest）
    per_class_auc = []
    for i in range(n_classes):
        try:
            auc_i = roc_auc_score(labels_onehot[:, i], probs[:, i])
            per_class_auc.append(auc_i)
        except ValueError:
            # 如果某类样本数为0，设为NaN
            per_class_auc.append(np.nan)

    # 计算Macro-AUC
    try:
        macro_auc = roc_auc_score(labels_onehot, probs,
                                  average='macro', multi_class='ovr')
    except ValueError:
        macro_auc = np.nan

    # 计算Micro-AUC
    try:
        micro_auc = roc_auc_score(labels_onehot, probs,
                                  average='micro', multi_class='ovr')
    except ValueError:
        micro_auc = np.nan

    return {
        'per_class_auc': np.array(per_class_auc),
        'macro_auc': macro_auc,
        'micro_auc': micro_auc
    }


def compute_bootstrap_ci(labels, preds, probs, n_bootstrap=1000, ci=95,
                         random_state=42):
    """
    使用分层Bootstrap计算各指标的置信区间

    Bootstrap方法通过重采样估计统计量的分布，从而计算置信区间
    使用分层抽样保持类别分布，适合不平衡数据集

    参数：
        labels: 真实标签数组 (n_samples,)
        preds: 预测标签数组 (n_samples,)
        probs: 预测概率数组 (n_samples, n_classes)
        n_bootstrap: Bootstrap重采样次数
        ci: 置信水平（百分比）
        random_state: 随机种子

    返回：
        dict: 包含所有指标的置信区间
    """
    from sklearn.utils import resample
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score

    n_samples = len(labels)
    n_classes = probs.shape[1]
    alpha = (100 - ci) / 2  # 双侧检验

    # 存储Bootstrap结果
    metrics_bootstrap = {
        'accuracy': [[] for _ in range(n_classes)],
        'precision': [[] for _ in range(n_classes)],
        'recall': [[] for _ in range(n_classes)],
        'f1': [[] for _ in range(n_classes)],
        'auc': [[] for _ in range(n_classes)],
        'specificity': [[] for _ in range(n_classes)],
        'macro_accuracy': [],
        'macro_precision': [],
        'macro_recall': [],
        'macro_specificity': [],
        'macro_f1': [],
        'macro_auc': [],
        'micro_auc': [],
        'overall_accuracy': []
    }

    np.random.seed(random_state)

    print(f"正在进行Bootstrap重采样（{n_bootstrap}次）...")
    for i in range(n_bootstrap):
        if (i + 1) % 200 == 0:
            print(f"  进度: {i+1}/{n_bootstrap}")

        # 分层抽样
        try:
            indices = resample(range(n_samples),
                              stratify=labels,
                              n_samples=n_samples,
                              random_state=random_state + i,
                              replace=True)
        except ValueError:
            # 如果样本太少无法分层，使用普通抽样
            indices = resample(range(n_samples),
                              n_samples=n_samples,
                              random_state=random_state + i,
                              replace=True)

        labels_boot = labels[indices]
        preds_boot = preds[indices]
        probs_boot = probs[indices]

        # 计算混淆矩阵
        hist_boot = fast_hist(labels_boot, preds_boot, n_classes)

        # 计算各类指标
        recall_boot = per_class_Recall(hist_boot)
        precision_boot = per_class_Precision(hist_boot)
        specificity_boot = compute_specificity(hist_boot)

        # F1分数
        f1_boot = np.zeros(n_classes)
        for j in range(n_classes):
            if precision_boot[j] + recall_boot[j] > 0:
                f1_boot[j] = 2 * (precision_boot[j] * recall_boot[j]) / \
                             (precision_boot[j] + recall_boot[j])

        # AUC
        try:
            labels_onehot = label_binarize(labels_boot, classes=range(n_classes))
            if labels_onehot.shape[1] == 1:
                labels_onehot = np.hstack([1 - labels_onehot, labels_onehot])

            # 每类AUC
            for j in range(n_classes):
                try:
                    auc_j = roc_auc_score(labels_onehot[:, j], probs_boot[:, j])
                    metrics_bootstrap['auc'][j].append(auc_j)
                except (ValueError, IndexError):
                    pass  # 该类样本数不足，跳过

            # Macro AUC
            try:
                macro_auc = roc_auc_score(labels_onehot, probs_boot,
                                         average='macro', multi_class='ovr')
                metrics_bootstrap['macro_auc'].append(macro_auc)
            except ValueError:
                pass

            # Micro AUC
            try:
                micro_auc = roc_auc_score(labels_onehot, probs_boot,
                                         average='micro', multi_class='ovr')
                metrics_bootstrap['micro_auc'].append(micro_auc)
            except ValueError:
                pass
        except (ValueError, IndexError):
            pass  # 标签异常，跳过该Bootstrap样本

        # 计算每类Accuracy (TP + TN) / Total
        accuracy_boot = np.zeros(n_classes)
        for j in range(n_classes):
            TP = hist_boot[j, j]
            TN = hist_boot.sum() - hist_boot[j, :].sum() - hist_boot[:, j].sum() + hist_boot[j, j]
            accuracy_boot[j] = (TP + TN) / hist_boot.sum()

        # 计算整体Accuracy
        overall_acc = np.trace(hist_boot) / hist_boot.sum()

        # 存储每类指标
        for j in range(n_classes):
            metrics_bootstrap['accuracy'][j].append(accuracy_boot[j])
            metrics_bootstrap['precision'][j].append(precision_boot[j])
            metrics_bootstrap['recall'][j].append(recall_boot[j])
            metrics_bootstrap['f1'][j].append(f1_boot[j])
            metrics_bootstrap['specificity'][j].append(specificity_boot[j])

        # 存储整体指标
        metrics_bootstrap['overall_accuracy'].append(overall_acc)
        metrics_bootstrap['macro_accuracy'].append(np.mean(accuracy_boot))
        metrics_bootstrap['macro_precision'].append(np.mean(precision_boot))
        metrics_bootstrap['macro_recall'].append(np.mean(recall_boot))
        metrics_bootstrap['macro_specificity'].append(np.mean(specificity_boot))
        metrics_bootstrap['macro_f1'].append(np.mean(f1_boot))

    print(f"  Bootstrap完成！")

    # 计算置信区间（百分位数法）
    ci_results = {}

    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc',
                   'specificity']:
        ci_results[metric] = []
        for j in range(n_classes):
            if len(metrics_bootstrap[metric][j]) > 0:
                values = np.array(metrics_bootstrap[metric][j])
                lower = np.percentile(values, alpha)
                upper = np.percentile(values, 100 - alpha)
                mean = np.mean(values)
                ci_results[metric].append({
                    'mean': mean,
                    'lower': lower,
                    'upper': upper
                })
            else:
                ci_results[metric].append({
                    'mean': np.nan,
                    'lower': np.nan,
                    'upper': np.nan
                })

    # 整体指标的置信区间
    for metric in ['macro_accuracy', 'macro_precision', 'macro_recall',
                   'macro_specificity', 'macro_f1', 'macro_auc', 'micro_auc',
                   'overall_accuracy']:
        if len(metrics_bootstrap[metric]) > 0:
            values = np.array(metrics_bootstrap[metric])
            ci_results[metric] = {
                'mean': np.mean(values),
                'lower': np.percentile(values, alpha),
                'upper': np.percentile(values, 100 - alpha)
            }
        else:
            ci_results[metric] = {
                'mean': np.nan,
                'lower': np.nan,
                'upper': np.nan
            }

    return ci_results


def draw_roc_curves(labels, probs, class_names, output_path):
    """
    绘制多分类ROC曲线（修改版：无平均值、图例对齐、带数据点）
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt
    from itertools import cycle
    import numpy as np

    n_classes = len(class_names)
    labels_onehot = label_binarize(labels, classes=range(n_classes))
    if labels_onehot.shape[1] == 1:
        labels_onehot = np.hstack([1 - labels_onehot, labels_onehot])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # 计算每类的ROC
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_onehot[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # --- 1. 删除 Micro 和 Macro 平均值的计算与绘图 ---

    # 绘图设置
    plt.figure(figsize=(8, 6), dpi=300) # 调整尺寸使其更紧凑
    plt.rcParams.update({
        'font.size': 16,              # 全局基础字体大小
        'font.family': 'Arial',  # 默认字体使用非衬线体（更美观）
        'axes.grid': False            # 去掉网格（根据原图风格）
    })

    # 定义颜色循环 (参考了常见的科研配色)
    colors = cycle(['#1f77b4', '#d62728', '#2ca02c', '#bcbd22', '#17becf', '#9467bd'])
    
    # 计算类名最大长度，用于图例对齐
    max_name_len = max([len(name) for name in class_names])

    # 绘制各类别曲线
    for i, color in zip(range(n_classes), colors):
        # --- 2. 构造对齐的图例标签 ---
        # {name:<{width}} 表示左对齐并在右侧填充空格
        # AUC保留3位小数，与提供的参考图一致
        label_str = f"{class_names[i]:<{max_name_len}}   AUC = {roc_auc[i]:.3f}"
        
        # --- 3. 添加小圆点 (marker='.') ---
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 marker='.',          # 设置标记样式为小圆点
                 markersize=4,        # 设置点的大小
                 markevery=0.2,      # (可选) 如果数据点太密，可以设置每隔多少个点画一个，例如 0.05 或 20
                 label=label_str)

    # 装饰图像
    # plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5) # 对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    # --- 图例设置重点 ---
    # 使用等宽字体 (monospace) 确保空格宽度一致，从而实现对齐
    # loc='lower right' 对应参考图位置
    plt.legend(loc="lower right", 
               prop={'family': 'monospace'}, 
               framealpha=1) # 增加背景不透明度，使文字更清晰
               
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return roc_auc

def draw_pr_curves(labels, probs, class_names, output_path):
    """
    绘制多分类PR曲线（修改版：无平均值、图例对齐、带数据点）
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt
    from itertools import cycle
    import numpy as np

    n_classes = len(class_names)
    labels_onehot = label_binarize(labels, classes=range(n_classes))
    if labels_onehot.shape[1] == 1:
        labels_onehot = np.hstack([1 - labels_onehot, labels_onehot])

    precision = dict()
    recall = dict()
    ap = dict()  # Average Precision

    # 计算每类的PR曲线和AP值
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            labels_onehot[:, i], probs[:, i])
        ap[i] = average_precision_score(labels_onehot[:, i], probs[:, i])

    # --- 1. 删除 Micro 和 Macro 平均值的计算与绘图 ---

    # 绘图设置
    plt.figure(figsize=(8, 6), dpi=300)
    plt.rcParams.update({
        'font.size': 16,              # 全局基础字体大小
        'font.family': 'Arial',  # 默认字体使用非衬线体（更美观）
        'axes.grid': False            # 去掉网格（根据原图风格）
    })

    # 定义颜色循环
    colors = cycle(['#1f77b4', '#d62728', '#2ca02c', '#bcbd22', '#17becf', '#9467bd'])

    # 计算类名最大长度，用于图例对齐
    max_name_len = max([len(name) for name in class_names])

    # 绘制各类别曲线
    for i, color in zip(range(n_classes), colors):
        # --- 2. 构造对齐的图例标签 ---
        # 格式：Name (填充空格) AP = 0.xxx
        # 为了对齐，移除了原代码中的 Ratio 显示
        label_str = f"{class_names[i]:<{max_name_len}}   AP = {ap[i]:.3f}"

        # --- 3. 添加小圆点 ---
        # markevery=0.05 意味着每隔总点数的 5% 画一个点，防止点太密看不清线条
        plt.plot(recall[i], precision[i], color=color, lw=1.5,
                 marker='.', 
                 markersize=4,
                 markevery=0.2, 
                 label=label_str)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    # --- 图例设置 ---
    # 使用等宽字体 (monospace) 配合 label_str 中的空格填充实现对齐
    plt.legend(loc="lower left", # PR曲线通常左下角较空
               prop={'family': 'monospace'}, 
               framealpha=1)
               
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return ap

def draw_confidence_intervals(ci_results, class_names, output_path):
    """
    绘制置信区间可视化（2x3子图布局）

    参数：
        ci_results: 置信区间结果字典
        class_names: 类别名称列表
        output_path: 输出文件路径
    """
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('95% Confidence Intervals for Metrics', fontsize=20, fontweight='bold')

    metrics_to_plot = [
        ('accuracy', 'Accuracy'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('specificity', 'Specificity'),
        ('f1', 'F1 Score'),
        ('auc', 'AUC')
    ]

    colors = ['skyblue', 'lightcoral', 'lightgreen']
    x = np.arange(len(class_names))

    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]

        means = [ci_results[metric][i]['mean'] for i in range(len(class_names))]
        lowers = [ci_results[metric][i]['lower'] for i in range(len(class_names))]
        uppers = [ci_results[metric][i]['upper'] for i in range(len(class_names))]

        # 计算误差
        yerr_lower = [means[i] - lowers[i] for i in range(len(class_names))]
        yerr_upper = [uppers[i] - means[i] for i in range(len(class_names))]

        # 绘制柱状图和误差线
        bars = ax.bar(x, means, color=colors, alpha=0.7, edgecolor='black')
        ax.errorbar(x, means, yerr=[yerr_lower, yerr_upper],
                   fmt='none', ecolor='black', capsize=5, capthick=2)

        # 添加数值标签
        for i, (mean, lower, upper) in enumerate(zip(means, lowers, uppers)):
            if not np.isnan(mean):
                ax.text(i, mean + yerr_upper[i] + 0.02,
                       f'{mean:.3f}\n[{lower:.3f}, {upper:.3f}]',
                       ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


