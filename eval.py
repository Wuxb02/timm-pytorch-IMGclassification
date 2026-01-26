import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from PIL import Image
from classification import (Classification, cvtColor, letterbox_image,
                            preprocess_input)
from utils.utils import (letterbox_image, count_samples_per_class,
                         identify_minority_class, get_class_display_names)
from utils.utils_metrics import (
    evaluteTop1_5, create_classification_report, fast_hist,
    per_class_Recall, per_class_Precision,
    compute_specificity, compute_auc_metrics, compute_bootstrap_ci,
    draw_roc_curves, draw_pr_curves, draw_confidence_intervals
)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#------------------------------------------------------#
#   test_annotation_path    测试图片路径和标签
#------------------------------------------------------#
test_annotation_path    = 'cls_test.txt'
#------------------------------------------------------#
#   metrics_out_path        指标保存的文件夹
#------------------------------------------------------#
metrics_out_path        = "metrics_out"

class Eval_Classification(Classification):
    def detect_image(self, image):        
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------#
        #   对图片进行不失真的resize
        #---------------------------------------------------#
        image_data  = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        #---------------------------------------------------------#
        #   归一化+添加上batch_size维度+转置
        #---------------------------------------------------------#
        image_data = np.transpose(
            np.expand_dims(
                preprocess_input(np.array(image_data, np.float32), self._backbone), 0
            ),
            (0, 3, 1, 2)
        )

        with torch.no_grad():
            photo   = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            preds   = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()

        return preds

if __name__ == "__main__":
    classfication = Eval_Classification()

    # 提取模型名称和数据集名称
    model_name = classfication._backbone
    dataset_name = os.path.splitext(os.path.basename(test_annotation_path))[0]

    # 构建动态输出文件夹名称
    output_folder_name = f"{model_name}_{dataset_name}"
    base_metrics_path = metrics_out_path
    metrics_out_path = os.path.join(base_metrics_path, output_folder_name)

    if not os.path.exists(metrics_out_path):
        os.makedirs(metrics_out_path)

    print(f"\n评估结果将保存至: {metrics_out_path}")

    with open(f"./{test_annotation_path}","r",encoding='UTF-8-sig') as f:
        lines = f.readlines()
    top1, top5, Recall, Precision = evaluteTop1_5(classfication, lines, metrics_out_path)

    # 动态识别少数类别
    samples_per_class, class_counts = count_samples_per_class(test_annotation_path)
    minority_idx, minority_ratio = identify_minority_class(class_counts)

    # 动态获取类别名称
    class_names = classfication.class_names

    print(f"\n数据集统计:")
    print(f"  类别分布: {class_counts}")
    print(f"  少数类别: {class_names[minority_idx]} (索引{minority_idx}, 占比{minority_ratio:.1%})")

    # 基础指标
    print("=" * 60)
    print("基础性能指标")
    print("=" * 60)
    print("top-1 accuracy = %.2f%%" % (top1*100))
    print("top-5 accuracy = %.2f%%" % (top5*100))
    print("mean Recall = %.2f%%" % (np.mean(Recall)*100))
    print("mean Precision = %.2f%%" % (np.mean(Precision)*100))

    # 详细的类别指标
    print("\n" + "=" * 60)
    print("各类别详细指标")
    print("=" * 60)
    
    # 计算F1分数
    f1_scores = []
    for i in range(len(class_names)):
        if Precision[i] + Recall[i] > 0:
            f1 = 2 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i])
        else:
            f1 = 0.0
        f1_scores.append(f1)
        
        print(f"{class_names[i]}:")
        print(f"  精确度 (Precision): {Precision[i]:.3f} ({Precision[i]*100:.1f}%)")
        print(f"  召回率 (Recall):    {Recall[i]:.3f} ({Recall[i]*100:.1f}%)")
        print(f"  F1分数:            {f1:.3f} ({f1*100:.1f}%)")
        print()
    
    # 少数类别专项指标
    print("=" * 60)
    print("少数类别关键指标")
    print("=" * 60)

    # 对于少数类别的特殊关注（动态识别）
    minority_recall = Recall[minority_idx]
    minority_precision = Precision[minority_idx]
    minority_f1 = f1_scores[minority_idx]

    print(f"少数类别({class_names[minority_idx]})性能:")
    print(f"  召回率: {minority_recall:.3f} ({minority_recall*100:.1f}%) - 关键指标，避免假阴性")
    print(f"  精确度: {minority_precision:.3f} ({minority_precision*100:.1f}%) - 避免假阳性")
    print(f"  F1分数: {minority_f1:.3f} ({minority_f1*100:.1f}%) - 综合性能")
    
    # 计算宏平均和微平均
    macro_f1 = np.mean(f1_scores)
    balanced_accuracy = np.mean(Recall)  # 平衡准确率等于各类别召回率的平均
    
    print(f"\n综合评估指标:")
    print(f"  宏平均 F1分数:     {macro_f1:.3f} ({macro_f1*100:.1f}%)")
    print(f"  平衡准确率:        {balanced_accuracy:.3f} ({balanced_accuracy*100:.1f}%)")
    
    # 数据不平衡改善效果评估
    print(f"\n数据不平衡改善效果:")
    performance_gap = max(f1_scores) - min(f1_scores)
    print(f"  类别间F1差距:      {performance_gap:.3f} (越小越好，理想值<0.1)")
    
    if minority_f1 > 0.5:
        print(f"  ✓ 少数类别性能良好 (F1 > 0.5)")
    elif minority_f1 > 0.3:
        print(f"  ⚠ 少数类别性能中等 (F1 > 0.3)")
    else:
        print(f"  ✗ 少数类别性能较差 (F1 < 0.3)，建议进一步优化")
    
    print("\n" + "=" * 60)
    
    # ========== 收集完整数据用于高级指标计算 ==========
    print("\n" + "=" * 80)
    print("收集预测数据用于高级指标计算...")
    print("=" * 80)

    preds_list = []
    labels_list = []
    probs_list = []  # 新增：收集完整概率向量
    detailed_results = []  # 新增：存储每个样本的详细信息

    for index, line in enumerate(lines):
        annotation_path = line.split(';')[1].split('\n')[0]
        x = Image.open(annotation_path)
        y = int(line.split(';')[0])

        pred = classfication.detect_image(x)  # 返回完整概率向量
        pred_1 = np.argmax(pred)

        preds_list.append(pred_1)
        labels_list.append(y)
        probs_list.append(pred)  # 保存完整概率

        # 新增：记录详细结果
        result_dict = {
            'path': annotation_path,
            'true': y,
            'predict': pred_1,
        }

        # 动态生成概率列名
        for class_idx, class_name in enumerate(class_names):
            # 判断是否为纯数字（旧格式cls_classes.txt）
            if class_name.isdigit():
                col_name = f'class_{class_idx}_prob'
            else:
                col_name = f'{class_name}_probability'
            result_dict[col_name] = pred[class_idx]

        detailed_results.append(result_dict)

        if index % 100 == 0:
            print(f"  进度: [{index+1}/{len(lines)}]")

    # 转换为NumPy数组
    labels = np.array(labels_list)
    preds = np.array(preds_list)
    probs = np.array(probs_list)

    # 保存详细预测结果为CSV
    detailed_df = pd.DataFrame(detailed_results)
    csv_path = os.path.join(metrics_out_path, "detailed_predictions.csv")
    detailed_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 详细预测结果已保存: {csv_path}")

    # 计算混淆矩阵
    hist = fast_hist(labels, preds, len(class_names))

    # 数据一致性验证
    verify_recall = per_class_Recall(hist)
    verify_precision = per_class_Precision(hist)
    print(f"\n数据一致性验证:")
    print(f"  Recall差异: {np.abs(Recall - verify_recall).max():.6f}")
    print(f"  Precision差异: {np.abs(Precision - verify_precision).max():.6f}")

    # ========== 计算高级指标 ==========
    print("\n" + "=" * 80)
    print("计算高级指标（AUC、敏感度、特异度）")
    print("=" * 80)

    # 1. 计算AUC指标
    print("\n正在计算AUC指标...")
    auc_metrics = compute_auc_metrics(labels, probs, len(class_names))

    print(f"\nAUC指标:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {auc_metrics['per_class_auc'][i]:.4f}")
    print(f"  Macro-AUC: {auc_metrics['macro_auc']:.4f}")
    print(f"  Micro-AUC: {auc_metrics['micro_auc']:.4f}")

    # 2. 计算特异度
    print("\n正在计算特异度...")
    specificity = compute_specificity(hist)

    print(f"\n敏感度和特异度:")
    for i, name in enumerate(class_names):
        print(f"  {name}:")
        print(f"    敏感度 (Sensitivity): {Recall[i]:.4f}")
        print(f"    特异度 (Specificity): {specificity[i]:.4f}")

    # 3. 计算Bootstrap置信区间
    print("\n" + "=" * 80)
    print("计算Bootstrap 95%置信区间（1000次重采样）")
    print("=" * 80)

    ci_results = compute_bootstrap_ci(labels, preds, probs,
                                     n_bootstrap=1000, ci=95,
                                     random_state=42)

    print(f"\n各类别指标95%置信区间:")
    print("-" * 80)
    for i, name in enumerate(class_names):
        print(f"\n{name}:")
        print(f"  精确度: {Precision[i]:.4f} "
              f"[{ci_results['precision'][i]['lower']:.4f}, "
              f"{ci_results['precision'][i]['upper']:.4f}]")
        print(f"  召回率: {Recall[i]:.4f} "
              f"[{ci_results['recall'][i]['lower']:.4f}, "
              f"{ci_results['recall'][i]['upper']:.4f}]")
        print(f"  F1分数: {f1_scores[i]:.4f} "
              f"[{ci_results['f1'][i]['lower']:.4f}, "
              f"{ci_results['f1'][i]['upper']:.4f}]")
        print(f"  AUC:    {auc_metrics['per_class_auc'][i]:.4f} "
              f"[{ci_results['auc'][i]['lower']:.4f}, "
              f"{ci_results['auc'][i]['upper']:.4f}]")

    print(f"\n整体指标95%置信区间:")
    print("-" * 80)
    print(f"Macro-AUC: {auc_metrics['macro_auc']:.4f} "
          f"[{ci_results['macro_auc']['lower']:.4f}, "
          f"{ci_results['macro_auc']['upper']:.4f}]")
    print(f"Micro-AUC: {auc_metrics['micro_auc']:.4f} "
          f"[{ci_results['micro_auc']['lower']:.4f}, "
          f"{ci_results['micro_auc']['upper']:.4f}]")

    # 4. 生成可视化
    print("\n" + "=" * 80)
    print("生成可视化图表")
    print("=" * 80)

    print("\n正在绘制ROC曲线...")
    roc_path = os.path.join(metrics_out_path, "roc_curves.png")
    roc_auc_values = draw_roc_curves(labels, probs, class_names, roc_path)
    print(f"✓ ROC曲线已保存: {roc_path}")

    print("正在绘制PR曲线...")
    pr_path = os.path.join(metrics_out_path, "pr_curves.png")
    ap_values = draw_pr_curves(labels, probs, class_names, pr_path)
    print(f"✓ PR曲线已保存: {pr_path}")

    print("正在绘制置信区间图...")
    ci_path = os.path.join(metrics_out_path, "confidence_intervals.png")
    draw_confidence_intervals(ci_results, class_names, ci_path)
    print(f"✓ 置信区间图已保存: {ci_path}")

    # 5. 计算每类Accuracy
    accuracy = np.zeros(len(class_names))
    for i in range(len(class_names)):
        TP = hist[i, i]
        TN = hist.sum() - hist[i, :].sum() - hist[:, i].sum() + hist[i, i]
        accuracy[i] = (TP + TN) / hist.sum()

    # 6. 生成完整分类报告(合并版)
    print("\n正在生成完整分类性能报告...")
    f1_scores = create_classification_report(
        hist, Recall, Precision, auc_metrics, specificity,
        accuracy, ci_results, class_names, metrics_out_path,
        samples_per_class=samples_per_class, minority_idx=minority_idx,
        top1_acc=top1, top5_acc=top5
    )

    # 最终总结
    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)

    print(f"\n生成的文件列表:")
    print(f"  1. 详细预测结果:")
    print(f"     - {csv_path}")
    print(f"  2. 基础指标图表:")
    print(f"     - {os.path.join(metrics_out_path, 'Recall.png')}")
    print(f"     - {os.path.join(metrics_out_path, 'Precision.png')}")
    print(f"     - {os.path.join(metrics_out_path, 'confusion_matrix.csv')}")
    print(f"  3. 详细可视化:")
    print(f"     - {os.path.join(metrics_out_path, 'confusion_matrix_detailed.png')}")
    print(f"     - {os.path.join(metrics_out_path, 'metrics_comparison_chart.png')}")
    print(f"  4. 高级可视化:")
    print(f"     - {roc_path}")
    print(f"     - {pr_path}")
    print(f"     - {ci_path}")
    print(f"  5. 性能报告:")
    print(f"     - {os.path.join(metrics_out_path, 'classification_report.txt')}")

    print(f"\n关键发现:")
    print(f"  • 少数类别({class_names[minority_idx]}) F1分数: {f1_scores[minority_idx]:.3f}")
    print(f"  • 少数类别({class_names[minority_idx]}) AUC: {auc_metrics['per_class_auc'][minority_idx]:.3f}")
    print(f"  • Macro-AUC: {auc_metrics['macro_auc']:.3f}")
    print(f"  • Micro-AUC: {auc_metrics['micro_auc']:.3f}")

    if f1_scores[minority_idx] > 0.5 and auc_metrics['per_class_auc'][minority_idx] > 0.7:
        print(f"\n✓ 模型在少数类别上表现良好，可考虑实际部署应用")
    else:
        print(f"\n⚠ 模型在少数类别上仍有提升空间，建议继续优化")

    print("\n" + "=" * 80)

