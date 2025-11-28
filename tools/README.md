# 工具集 (Tools)

本目录包含三个高级评估和可视化工具,用于深入分析训练好的分类模型的性能。

---

## 📋 工具列表

1. **visualize_gradcam.py** - GRAD-CAM++可解释性热图可视化工具 ⭐ **新增**
2. **compare_models_auc.py** - 双模型多指标统计比较工具
3. **evaluate_calibration.py** - 模型校准性能评估工具

---

# 0️⃣ GRAD-CAM++可解释性热图可视化工具

## 简介

`visualize_gradcam.py` 是一个基于GRAD-CAM++算法的深度学习模型可解释性分析工具,用于生成和可视化模型决策的热图,帮助理解CNN模型关注的图像区域。

### 核心特性

- **GRAD-CAM++算法**: 相比标准GRAD-CAM更精确的权重计算,特别适合医学影像分析
- **智能层检测**: 自动检测最后一个卷积层,支持10+种主流CNN架构
- **Python API**: 非命令行形式,直接在代码中调用
- **单张+批量**: 支持单张图片处理和整个文件夹批量处理
- **JET颜色映射**: 经典的蓝-青-黄-红热图配色
- **GPU加速**: 自动GPU/CPU适配

### 支持的模型架构

✅ **支持的CNN**:
- InceptionResNetV2, ResNet系列 (18/34/50/101/152)
- VGG系列 (11/13/16及BN版本), DenseNet系列 (121/161/169/201)
- MobileNetV2, EfficientNet系列 (B0-B7), ConvNeXt系列

❌ **不支持**: Vision Transformer (ViT), Swin Transformer (需要Attention Map方法)

---

## 快速开始

### 方法1: 直接运行脚本

```bash
cd tools
python visualize_gradcam.py
```

脚本会自动处理预设的测试图片并生成热图。

### 方法2: Python API调用 (推荐)

```python
from tools.visualize_gradcam import generate_gradcam

# 单张图片处理
result = generate_gradcam(
    image_path='datasets/test/1/sample.jpg',
    output_path='cam_output/sample_gradcam.jpg',
    alpha=0.5  # 热图透明度
)

print(f"预测: {result['pred_name']}, 置信度: {result['confidence']:.3f}")
```

### 方法3: 批量处理

```python
from tools.visualize_gradcam import generate_gradcam_batch

# 批量处理整个文件夹
results = generate_gradcam_batch(
    image_dir='datasets/test/1/',
    output_dir='cam_output/batch_analysis',
    save_report=True  # 生成CSV报告
)

print(f"完成! 共处理{len(results)}张图片")
```

### 方法4: 快速模式

```python
from tools.visualize_gradcam import quick_gradcam

# 使用默认配置快速生成
result = quick_gradcam('test.jpg', 'test_gradcam.jpg')
```

---

## 函数参数说明

### `generate_gradcam()` 参数列表

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `image_path` | str | *必需* | 输入图片路径 |
| `model_path` | str | `'models/inception_resnet_v2/...'` | 模型权重路径 |
| `backbone` | str | `'inception_resnet_v2'` | 模型架构名称 |
| `classes_path` | str | `'model_data/cls_classes.txt'` | 类别定义文件 |
| `input_shape` | tuple | `(299, 299)` | 输入尺寸 (H, W) |
| `target_class` | int | `None` | 目标类别索引 (None=预测类别) |
| `alpha` | float | `0.5` | 热图透明度 [0, 1] |
| `output_path` | str | `None` | 输出路径 (None=不保存) |
| `cuda` | bool | `True` | 是否使用GPU |
| `return_image` | bool | `False` | 是否返回图片数组 |

### 返回值说明

返回一个字典,包含以下字段:

| 字段 | 类型 | 说明 |
|------|------|------|
| `pred_class` | int | 预测类别索引 |
| `pred_name` | str | 预测类别名称 |
| `confidence` | float | 预测置信度 |
| `cam` | np.ndarray | 热图数组 [H, W] |
| `overlay` | np.ndarray | 叠加图 [H, W, 3] (如果return_image=True) |
| `output_path` | str | 保存路径 (如果指定了output_path) |

---

## 输出文件

### 单张处理输出

```
cam_output/
└── sample_gradcam.jpg  # 热图叠加原图 (JET颜色映射)
```

### 批量处理输出

```
cam_output/
└── batch_analysis/
    ├── img001_gradcam.jpg
    ├── img002_gradcam.jpg
    ├── ...
    └── gradcam_report.csv  # CSV报告 (可选)
```

**CSV报告格式**:
```csv
图片路径,预测类别,置信度,热图路径
datasets/test/1/img001.jpg,abnormal,0.9234,cam_output/batch_xxx/img001_gradcam.jpg
```

---

## 使用案例

### 案例1: 分析单张图片的模型决策

```python
from tools.visualize_gradcam import generate_gradcam

# 生成热图以理解模型关注区域
result = generate_gradcam(
    image_path='datasets/test/1/suspicious_case.jpg',
    output_path='analysis/case_gradcam.jpg',
    alpha=0.5
)

print(f"模型预测: {result['pred_name']}")
print(f"置信度: {result['confidence']:.3f}")
# 手动查看 analysis/case_gradcam.jpg 确认模型关注的区域是否合理
```

### 案例2: 批量分析错误分类样本

```python
from tools.visualize_gradcam import generate_gradcam_batch

# 对误分类样本生成热图,分析错误原因
results = generate_gradcam_batch(
    image_dir='datasets/misclassified/',
    output_dir='analysis/error_cases',
    save_report=True
)

print(f"已生成{len(results)}个错误案例的热图")
# 查看热图判断: 模型关注区域是否正确? 是特征提取问题还是数据标注问题?
```

### 案例3: 对比不同模型的关注区域

```python
from tools.visualize_gradcam import generate_gradcam

image_path = 'datasets/test/1/sample.jpg'

# 模型A (InceptionResNetV2)
result_A = generate_gradcam(
    image_path=image_path,
    model_path='models/inception_resnet_v2/best_epoch_weights.pth',
    backbone='inception_resnet_v2',
    output_path='comparison/model_A_gradcam.jpg'
)

# 模型B (ResNet50)
result_B = generate_gradcam(
    image_path=image_path,
    model_path='models/resnet50/best_epoch_weights.pth',
    backbone='resnet50',
    input_shape=(224, 224),
    output_path='comparison/model_B_gradcam.jpg'
)

print(f"模型A关注区域 vs 模型B关注区域")
print(f"模型A置信度: {result_A['confidence']:.3f}")
print(f"模型B置信度: {result_B['confidence']:.3f}")
# 对比两张热图,判断哪个模型的关注区域更合理
```

---

## 常见问题 (FAQ)

### Q1: 提示"模型不支持GRAD-CAM"

**A**: 您使用的是Transformer架构(ViT、Swin),不支持GRAD-CAM。
- **解决**: 使用CNN架构(ResNet、InceptionResNetV2等)

---

### Q2: 提示"模型中未找到层"

**A**: 模型目标层映射不正确。
- **解决方案1**: 运行 `python tools/print_model_structure.py --backbone 你的模型名` 查看层结构
- **解决方案2**: 系统会自动尝试检测最后一个卷积层

---

### Q3: 热图质量不佳或定位不准

**A**: 可能原因和优化:
1. **调整透明度**: `alpha=0.3` (降低) 或 `alpha=0.7` (提高)
2. **确保模型性能**: 准确率低的模型热图也不可靠
3. **检查输入图像**: 确保图像质量良好

---

### Q4: GPU内存不足

**A**:
```python
# 方案1: 使用CPU
result = generate_gradcam(..., cuda=False)

# 方案2: 单张处理而非批量
```

---

### Q5: 如何查看模型结构?

**A**: 使用辅助工具:
```bash
python tools/print_model_structure.py --backbone inception_resnet_v2
```

输出会显示所有层名称和推荐的GRAD-CAM目标层。

---

## 技术细节

### GRAD-CAM++ vs GRAD-CAM

| 特性 | GRAD-CAM | GRAD-CAM++ |
|-----|----------|-----------|
| 权重计算 | 全局平均池化梯度 | 加权梯度(二阶导数) |
| 多目标场景 | 可能定位不准 | 更精确的定位 |
| 医学影像适用性 | 一般 | 优秀 |
| 计算复杂度 | 低 | 稍高(但可接受) |

**本工具选择**: GRAD-CAM++(同时也提供标准GRAD-CAM实现作为对比)

### 核心算法

GRAD-CAM++改进的权重计算公式:
```
alpha = grad^2 / (2 * grad^2 + sum(A) * grad^3 + epsilon)
weights = sum(alpha * ReLU(grad))
cam = sum(weights * activations)
```

### 适用场景

✅ **适用**:
- 理解CNN模型的决策依据
- 分析错误分类案例
- 论文中展示模型关注区域
- 医学影像分析(定位病灶)

❌ **不适用**:
- Transformer模型(无卷积层)
- 仅需要分类结果不需要解释

---

## 辅助工具

### 模型结构查看工具

```bash
# 查看任意模型的层结构
python tools/print_model_structure.py --backbone resnet50
python tools/print_model_structure.py --backbone efficientnet_b0
```

**输出内容**:
- 顶层模块列表
- 所有层名称
- 所有卷积层
- 推荐的GRAD-CAM目标层

---

## 更新日志

### v1.0.0 (2025-11-28)
- ✅ 初始版本发布
- ✅ GRAD-CAM++核心算法实现
- ✅ 支持10+种CNN架构
- ✅ 智能目标层自动检测
- ✅ Python API接口(非命令行)
- ✅ 单张+批量处理
- ✅ JET颜色映射
- ✅ GPU/CPU自适配
- ✅ 完整文档和示例

---

## 相关文档

- **完整API文档**: `tools/cam/README.md`
- **快速使用指南**: `GRADCAM_USAGE.md`
- **使用示例脚本**: `example_gradcam.py`

---

**Happy Visualizing! 🔥**

---

---



# 1️⃣ 双模型多指标统计比较工具

## 简介

`compare_models_auc.py` 是一个基于配对Bootstrap方法的模型性能统计比较工具,用于科学严谨地比较两个深度学习分类模型的整体性能差异。

### 核心特性

- **统计严谨**: 使用配对Bootstrap方法计算置信区间和p值,避免简单比较的误导性
- **多指标支持**: 支持6种整体评估指标(Macro/Micro AUC, Accuracy, Precision, Recall, F1)
- **整体性能**: 所有指标均为模型整体评估,适用于2分类和N分类场景
- **专业可视化**: 生成高质量的差异分布图和对比柱状图
- **详细报告**: 输出完整的统计分析文本报告

### 支持的指标

| 指标名称 | 说明 | 适用场景 |
|---------|------|---------|
| `macro_auc` | Macro-averaged AUC (OvR) | 整体模型性能(类别平衡) |
| `micro_auc` | Micro-averaged AUC (OvR) | 整体模型性能(样本级) |
| `accuracy` | 整体准确率 | 所有类别综合准确性 |
| `macro_precision` | Macro-averaged Precision | 整体精确度(类别平衡) |
| `macro_recall` | Macro-averaged Recall (Sensitivity) | 整体召回率(类别平衡) |
| `macro_f1` | Macro-averaged F1-score | 精确度和召回率的调和平均 |

---

## 快速开始

### 前提条件

1. 已运行 `eval.py` 生成两个模型的 `detailed_predictions.csv` 文件
2. 两个CSV文件必须在**完全相同的测试集**上评估(相同样本顺序和真实标签)

### 使用方法: Python函数调用

```python
from tools.compare_models_auc import compare_two_models

# 示例1: 基础用法 - 比较Macro AUC
results = compare_two_models(
    'metrics_out/inception_resnet_v2_cls_test/detailed_predictions.csv',
    'metrics_out/resnet50_cls_test/detailed_predictions.csv',
    model_name1='InceptionResNetV2',
    model_name2='ResNet50'
)

# 示例2: 多指标比较
results = compare_two_models(
    'metrics_out/model_A/detailed_predictions.csv',
    'metrics_out/model_B/detailed_predictions.csv',
    model_name1='Model A',
    model_name2='Model B',
    metrics=['macro_auc', 'micro_auc', 'accuracy', 'macro_f1'],
    n_bootstrap=2000,
    ci_level=99
)

# 示例3: 静默模式(不打印详细信息)
results = compare_two_models(
    'metrics_out/model_A/detailed_predictions.csv',
    'metrics_out/model_B/detailed_predictions.csv',
    verbose=False  # 静默运行,适合批量处理
)

# 示例4: 访问结果
print(f"Macro AUC差异: {results['macro_auc']['diff_original']:.4f}")
print(f"95% CI: [{results['macro_auc']['ci_lower']:.4f}, {results['macro_auc']['ci_upper']:.4f}]")
print(f"p值: {results['macro_auc']['p_value']:.4f}")
print(f"是否显著: {results['macro_auc']['significant']}")
print(f"效应量: {results['macro_auc']['effect_size']}")
```

---

## 函数参数说明

### `compare_two_models()` 参数列表

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `csv_path1` | str | *必需* | 模型1的CSV文件路径 |
| `csv_path2` | str | *必需* | 模型2的CSV文件路径 |
| `model_name1` | str | `'Model_A'` | 模型1的显示名称 |
| `model_name2` | str | `'Model_B'` | 模型2的显示名称 |
| `output_dir` | str | `'metrics_out/model_comparison'` | 输出目录路径 |
| `n_bootstrap` | int | `1000` | Bootstrap重采样次数(建议1000-5000) |
| `ci_level` | float | `95.0` | 置信水平,百分比(常用: 90, 95, 99) |
| `metrics` | list | `['macro_auc']` | 指标列表 |
| `random_state` | int | `42` | 随机种子(用于结果可复现) |
| `verbose` | bool | `True` | 是否打印详细信息 |

### 返回值说明

返回一个字典,键为指标名称,值为包含以下字段的字典:

| 字段 | 类型 | 说明 |
|------|------|------|
| `metric1_original` | float | 模型1的原始指标值 |
| `metric2_original` | float | 模型2的原始指标值 |
| `diff_original` | float | 差异值(模型1 - 模型2) |
| `diff_bootstrap` | np.ndarray | Bootstrap差异分布数组 |
| `diff_mean` | float | Bootstrap差异均值 |
| `diff_std` | float | Bootstrap差异标准差 |
| `ci_lower` | float | 置信区间下界 |
| `ci_upper` | float | 置信区间上界 |
| `p_value` | float | 双侧p值 |
| `significant` | bool | 是否具有统计显著性 |
| `effect_size` | str | 效应量("Negligible", "Small", "Medium", "Large") |

---

## 输出文件

运行完成后,在 `metrics_out/model_comparison/` 目录下生成以下文件:

### 1. 可视化文件

#### `metrics_difference_distribution.png`
- **内容**: 多指标差异的Bootstrap分布直方图
- **布局**: 根据指标数量自动调整(1个: 单图, 2个: 1x2, 3-4个: 2x2, 5-6个: 2x3)
- **标注**:
  - 红色虚线: 95% CI上下界
  - 绿色实线: 0线(无差异基准)
  - 标题显示指标名和p值
  - `*`标记表示统计显著性

#### `metrics_comparison_barplot.png`
- **内容**: 双模型指标值对比柱状图
- **特点**:
  - 分组柱状图(每组两根柱子)
  - 柱顶标注具体数值
  - 显著差异指标顶部标注 `*`

### 2. 文本报告

#### `comparison_report.txt`
- **内容**: 完整的统计分析报告
- **包含**:
  - 实验配置信息
  - 每个指标的原始值、差异、置信区间
  - p值和统计显著性结论
  - 效应量评估
  - 总体结论和解读说明

---

## 输出解读

### 控制台输出示例

```
========================================
比较结果摘要
========================================
Macro Auc      : +0.0289 [95% CI: +0.0051, +0.0527] **  (p=0.017)
Accuracy       : +0.0200 [95% CI: -0.0023, +0.0423]     (p=0.078)
Macro F1       : +0.0200 [95% CI: +0.0001, +0.0399] *   (p=0.049)

显著性标记: *** p<0.01, ** p<0.05, * p<0.1

结论: Model A 在 Macro Auc, Macro F1 上显著优于对比模型
========================================
```

### 关键指标解读

1. **差异值** (如 `+0.0289`)
   - 正值: 模型1优于模型2
   - 负值: 模型2优于模型1
   - 数值大小: 性能差距

2. **95% 置信区间** (如 `[+0.0051, +0.0527]`)
   - **不包含0**: 差异具有统计显著性
   - **包含0**: 差异无统计显著性
   - 区间宽度: 估计不确定性(窄=更可靠)

3. **p值** (如 `p=0.017`)
   - `p < 0.01`: 高度显著 (`***`)
   - `p < 0.05`: 显著 (`**`)
   - `p < 0.1`: 边缘显著 (`*`)
   - `p ≥ 0.1`: 无显著差异

4. **效应量** (报告中)
   - Negligible: |差异| < 0.02 (可忽略)
   - Small: 0.02 ≤ |差异| < 0.05 (小)
   - Medium: 0.05 ≤ |差异| < 0.10 (中等)
   - Large: |差异| ≥ 0.10 (大)

---

## 使用案例

### 案例1: 比较两个预训练模型

```python
# 场景: 比较InceptionResNetV2和ResNet50在图像分类任务上的性能
from tools.compare_models_auc import compare_two_models

results = compare_two_models(
    'metrics_out/inception_resnet_v2_cls_test/detailed_predictions.csv',
    'metrics_out/resnet50_cls_test/detailed_predictions.csv',
    model_name1='InceptionResNetV2',
    model_name2='ResNet50',
    metrics=['macro_auc', 'accuracy', 'macro_recall'],
    n_bootstrap=2000
)
```

**预期输出**: 判断哪个模型更适合该任务

### 案例2: 评估数据增强的效果

```python
# 场景: 比较使用/不使用数据增强训练的相同模型
from tools.compare_models_auc import compare_two_models

results = compare_two_models(
    'metrics_out/model_with_augmentation/detailed_predictions.csv',
    'metrics_out/model_without_augmentation/detailed_predictions.csv',
    model_name1='With Augmentation',
    model_name2='Without Augmentation',
    metrics=['macro_auc', 'macro_precision', 'macro_recall']
)
```

**预期输出**: 量化数据增强的性能提升

### 案例3: 对比不同损失函数

```python
# 场景: 比较使用Focal Loss vs Cross Entropy训练的模型
from tools.compare_models_auc import compare_two_models

results = compare_two_models(
    'metrics_out/model_focal_loss/detailed_predictions.csv',
    'metrics_out/model_cross_entropy/detailed_predictions.csv',
    model_name1='Focal Loss',
    model_name2='Cross Entropy',
    metrics=['macro_auc', 'micro_auc', 'accuracy', 'macro_f1'],
    ci_level=99
)
```

**预期输出**: 验证Focal Loss在类别不平衡数据上的优势

---

## 常见问题 (FAQ)

### Q1: 为什么两个CSV的真实标签必须完全相同?

**A**: 配对Bootstrap方法要求对相同样本进行重采样,确保比较的公平性。如果测试集不同,结论可能受数据集差异影响而非模型性能差异。

---

### Q2: Bootstrap次数选择多少合适?

**A**:
- **默认1000次**: 平衡速度和精度,适合大多数场景
- **2000-5000次**: 追求更高精度,样本量较小时推荐
- **500次**: 快速测试(不推荐用于最终结论)

---

### Q3: 如何判断差异是否显著?

**A**: 三种方法(推荐使用前两种):
1. **置信区间法** (主要): CI不包含0 → 显著
2. **p值法** (辅助): p < 0.05 → 显著
3. **效应量** (补充): 即使显著,效应量小可能实际意义有限

---

### Q4: 报错 "真实标签不一致" 怎么办?

**A**: 可能原因:
1. 两个模型在不同测试集上评估 → **确保使用相同的 `cls_test.txt`**
2. CSV文件的样本顺序不同 → **重新运行 `eval.py` 生成CSV**
3. 一个CSV是训练集,一个是测试集 → **检查文件路径**

---

### Q5: 报错 "无法识别概率列" 怎么办?

**A**: 检查CSV文件是否包含概率列:
- **正确格式1**: `normal_probability`, `abnormal_probability`
- **正确格式2**: `class_0_prob`, `class_1_prob`
- **错误格式**: `prob_0`, `probability_normal` (不支持)

如果格式错误,需要修改 `eval.py` 重新生成CSV。

---

### Q6: 能否比较3个以上的模型?

**A**: 当前版本仅支持两两比较。如需比较多个模型:
1. 两两运行本工具(如A vs B, A vs C, B vs C)
2. 汇总结果进行综合分析

未来版本可能支持多模型批量比较。

---

### Q7: Macro AUC和Micro AUC有什么区别?

**A**:
- **Macro AUC**: 每个类别AUC的算术平均,**类别平等权重**,适合类别平衡场景
- **Micro AUC**: 汇总所有样本计算全局AUC,**样本级权重**,大类别影响更大

**推荐**: 类别平衡时用Macro, 类别不平衡时同时看Macro和Micro

---

### Q8: 运行速度慢怎么办?

**A**: 优化建议:
1. 减少Bootstrap次数(如从2000降到1000)
2. 减少比较指标数量(先比较核心指标)
3. 未来版本可能支持多核并行加速

**参考速度**: 1000次Bootstrap × 3指标 × 500样本 ≈ 5-10秒

---

### Q9: 可视化中文显示为方块怎么办?

**A**: 安装中文字体:
```bash
# Windows: 系统自带SimHei,一般无问题
# Linux: 安装字体
sudo apt-get install fonts-wqy-microhei
```

或修改脚本第356行,使用系统字体:
```python
plt.rcParams['font.sans-serif'] = ['Arial']  # 使用英文字体
```

---

### Q10: 能否自定义输出目录?

**A**: 可以,使用 `output_dir` 参数:
```python
from tools.compare_models_auc import compare_two_models

results = compare_two_models(
    'metrics_out/model_A/detailed_predictions.csv',
    'metrics_out/model_B/detailed_predictions.csv',
    output_dir='my_results/comparison_2025_11_28'
)
```

---

## 技术细节

### Bootstrap方法说明

本工具使用**配对分层Bootstrap**:

1. **配对设计**: 两模型在每次重采样中使用相同的样本索引,消除数据集随机性的影响
2. **分层抽样**: 保持类别分布与原始数据一致,适合不平衡数据集
3. **百分位法**: 使用Bootstrap差异分布的百分位数计算置信区间
4. **双侧检验**: p值计算考虑双向差异,适合"是否存在差异"的假设检验

### 统计假设

- **零假设(H0)**: 两模型性能无差异(差异=0)
- **备择假设(H1)**: 两模型性能有差异(差异≠0)
- **显著性水平**: α = 1 - ci_level / 100 (默认0.05)

### 二分类AUC计算细节

本工具对**二分类**和**多分类**场景采用不同的AUC计算策略:

#### 二分类 (n_classes = 2)

- **Macro AUC**: 手动构造One-Hot编码矩阵,分别计算每个类别的AUC后取平均
  ```python
  labels_bin = np.zeros((len(labels), 2), dtype=int)
  labels_bin[np.arange(len(labels)), labels] = 1
  auc_0 = roc_auc_score(labels_bin[:, 0], probs[:, 0])
  auc_1 = roc_auc_score(labels_bin[:, 1], probs[:, 1])
  macro_auc = (auc_0 + auc_1) / 2
  ```
- **Micro AUC**: 直接使用正类(类别1)的概率计算
  ```python
  micro_auc = roc_auc_score(labels, probs[:, 1])
  ```

**设计原因**: sklearn的`label_binarize`在二分类时只返回形状为`(n_samples, 1)`的数组,无法满足OvR策略需要的完整二值化矩阵。

#### 多分类 (n_classes ≥ 3)

- **Macro AUC**: 直接使用sklearn的One-vs-Rest策略
  ```python
  macro_auc = roc_auc_score(labels, probs, average='macro', multi_class='ovr')
  ```
- **Micro AUC**: 同样使用sklearn的OvR策略
  ```python
  micro_auc = roc_auc_score(labels, probs, average='micro', multi_class='ovr')
  ```

### 适用场景

✅ **适用**:
- 比较不同模型架构的性能
- 评估训练策略(数据增强、损失函数等)的效果
- 验证模型改进是否有统计学意义

❌ **不适用**:
- 样本量过小(<30样本,Bootstrap不稳定)
- 测试集不同(违反配对设计前提)
- 训练集/验证集比较(应使用独立测试集)

---

## 引用

如果本工具对您的研究有帮助,请在论文中引用Bootstrap方法:

```
Efron, B., & Tibshirani, R. J. (1994).
An introduction to the bootstrap.
CRC press.
```

---

## 更新日志

### v1.0.1 (2025-11-28)
- 🐛 修复二分类场景下Macro AUC计算的IndexError
- 🔧 使用手动One-Hot编码替代sklearn的label_binarize
- ✅ 确保二分类和多分类场景的兼容性

### v1.0.0 (2025-11-28)
- ✅ 初始版本发布
- ✅ 支持6种整体评估指标(macro_auc, micro_auc, accuracy, macro_precision, macro_recall, macro_f1)
- ✅ 配对Bootstrap统计检验
- ✅ 专业可视化和报告生成
- ✅ 完整的错误处理和用户提示
- ✅ 纯Python函数接口,无命令行依赖

---

## 联系与反馈

如有问题或建议,请通过以下方式联系:
- 项目仓库: [GitHub Issues]
- 邮箱: [联系邮箱]

---

**Happy Comparing! 🚀**

---
---

# 2️⃣ 模型校准性能评估工具

## 简介

`evaluate_calibration.py` 是一个用于评估分类模型概率预测可靠性的工具。通过计算Calibration Plot(校准曲线)和Brier Score,帮助用户判断模型输出的概率值是否真实反映预测的置信度。

### 核心特性

- **Calibration Plot (可靠性曲线)**: 可视化预测概率与实际正确率的对应关系
- **Brier Score**: 量化概率预测的准确性(越小越好,0为完美)
- **整体+各类别校准**: 同时提供模型整体校准和各类别独立校准分析
- **sklearn API**: 基于`sklearn.calibration.calibration_curve`实现,结果可靠
- **专业可视化**: 使用Times New Roman字体,英文输出,适合学术论文

### 为什么需要校准评估?

**场景示例**:
- 模型A预测某样本为类别1的概率是80%,但实际准确率只有60% → **校准差**
- 模型B预测概率80%,实际准确率也是80% → **校准好**

**应用价值**:
- 医学诊断: 置信度决定是否需要进一步检查
- 风险评估: 概率值直接用于决策阈值
- 模型选择: AUC相似时,校准好的模型更可靠

---

## 快速开始

### 前提条件

1. 已运行 `eval.py` 生成 `detailed_predictions.csv` 文件
2. CSV文件包含完整的预测概率(每个类别的概率列)

### 使用方法: Python函数调用

```python
from tools.evaluate_calibration import evaluate_model_calibration

# 示例1: 从CSV加载 (推荐,速度快)
results = evaluate_model_calibration(
    csv_path='metrics_out/inception_resnet_v2_cls_test/detailed_predictions.csv',
    output_dir='metrics_out/calibration_analysis',
    n_bins=10
)

# 示例2: 实时推理 (灵活但慢)
from eval import Eval_Classification
model = Eval_Classification()
results = evaluate_model_calibration(
    model_instance=model,
    annotation_path='cls_test.txt',
    output_dir='metrics_out/calibration_analysis'
)

# 示例3: 使用等频分桶 (适合数据分布不均)
results = evaluate_model_calibration(
    csv_path='metrics_out/model/detailed_predictions.csv',
    output_dir='metrics_out/calibration',
    n_bins=10,
    binning_strategy='quantile'  # 'uniform'(默认) 或 'quantile'
)
```

---

## 函数参数说明

### `evaluate_model_calibration()` 参数列表

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `csv_path` | str | `None` | CSV文件路径(优先使用,来自eval.py) |
| `annotation_path` | str | `'cls_test.txt'` | 测试数据标注文件(实时推理时使用) |
| `model_instance` | object | `None` | Eval_Classification实例(实时推理时使用) |
| `class_names` | list | `None` | 类别名称列表(自动从CSV或模型推断) |
| `output_dir` | str | `'metrics_out/calibration_analysis'` | 输出目录路径 |
| `n_bins` | int | `10` | 分桶数量(建议5-20) |
| `binning_strategy` | str | `'uniform'` | 分桶策略: 'uniform'(等宽) 或 'quantile'(等频) |
| `verbose` | bool | `True` | 是否打印详细信息 |

### 返回值说明

返回一个字典,包含以下字段:

| 字段 | 类型 | 说明 |
|------|------|------|
| `overall_brier_score` | float | 整体Brier Score |
| `per_class_brier_scores` | list | 各类别Brier Score列表 |
| `overall_calibration` | dict | 整体校准曲线数据 |
| `per_class_calibrations` | list | 各类别校准曲线数据列表 |

---

## 输出文件

运行完成后,在指定的`output_dir`目录下生成以下文件:

### 1. 可视化文件

#### `calibration_overall.png`
- **内容**: 整体校准曲线(Overall Calibration Curve)
- **特点**:
  - 使用线图连接各校准点(marker='o')
  - 对角虚线表示完美校准(Perfect calibration)
  - 标题显示整体Brier Score
  - 英文标签,Times New Roman字体
- **解读**:
  - 曲线越接近对角线,校准越好
  - 曲线在对角线下方: 模型过于自信(概率偏高)
  - 曲线在对角线上方: 模型过于保守(概率偏低)

#### `calibration_per_class.png`
- **内容**: 各类别校准曲线(Per-Class Calibration Curves)
- **布局**: 根据类别数量自动调整
  - 2分类: 1行2列
  - 3分类: 1行3列
  - 4分类: 2行2列
  - 5-6分类: 2行3列
  - 7-9分类: 3行3列
- **特点**:
  - 每个子图显示一个类别的校准曲线(One-vs-Rest策略)
  - 标题显示类别名和该类别的Brier Score
  - 英文标签,Times New Roman字体

### 2. 文本报告

#### `calibration_report.txt`
- **内容**: 完整的校准性能评估报告(中文)
- **包含**:
  - I. 整体校准指标(整体Brier Score、分桶统计)
  - II. 各类别校准指标(各类别Brier Score、宏平均)
  - III. 校准质量综合评估(优秀/良好/一般/较差)
  - IV. Brier Score解释指南(阈值说明)
  - 报告生成时间

---

## 输出解读

### 控制台输出示例

```
================================================================================
模型校准性能评估
================================================================================

[1/5] 加载预测数据...
  ✓ 加载完成: 500个样本, 2个类别
  ✓ 类别名称: ['normal', 'abnormal']

[2/5] 计算Brier Score...
  ✓ 整体Brier Score: 0.0987
  ✓ 类别0 (normal) Brier Score: 0.0823
  ✓ 类别1 (abnormal) Brier Score: 0.1151

[3/5] 计算校准曲线...
  ✓ 校准曲线计算完成 (分桶数: 10, 策略: uniform)

[4/5] 生成可视化...
  ✓ 整体校准图已保存: metrics_out/calibration/calibration_overall.png
  ✓ 各类别校准图已保存: metrics_out/calibration/calibration_per_class.png

[5/5] 生成文本报告...
  ✓ 文本报告已保存: metrics_out/calibration/calibration_report.txt

================================================================================
评估完成!
================================================================================

整体Brier Score: 0.0987
整体评价: 优秀 - 概率预测高度可靠

各类别Brier Score:
  • 类别0 (normal): 0.0823 (优秀)
  • 类别1 (abnormal): 0.1151 (良好)

输出文件:
  1. metrics_out/calibration/calibration_overall.png
  2. metrics_out/calibration/calibration_per_class.png
  3. metrics_out/calibration/calibration_report.txt
```

### Brier Score解读

#### 2分类场景
| Brier Score | 等级 | 说明 |
|-------------|------|------|
| **BS < 0.10** | 优秀 | 概率预测高度可靠,可直接用于决策 |
| **0.10 ≤ BS < 0.15** | 良好 | 模型预测概率具有中等可靠性 |
| **0.15 ≤ BS < 0.20** | 一般 | 存在一定程度的校准偏差,建议重新校准 |
| **BS ≥ 0.20** | 较差 | 显著的校准偏差,需要重新校准 |

**参考基准**: 随机分类器的Brier Score ≈ 0.25

#### 多分类场景 (N≥3)
- **阈值动态调整**: 随机分类器基准 BS = 1 - (1/C)
- **优秀**: BS < baseline × 0.4
- **良好**: baseline × 0.4 ≤ BS < baseline × 0.6
- **一般**: baseline × 0.6 ≤ BS < baseline × 0.8
- **较差**: BS ≥ baseline × 0.8

---

## 使用案例

### 案例1: 评估单个模型的校准性能

```python
# 场景: 训练完成后,评估模型的概率预测是否可靠
from tools.evaluate_calibration import evaluate_model_calibration

results = evaluate_model_calibration(
    csv_path='metrics_out/inception_resnet_v2_cls_test/detailed_predictions.csv',
    output_dir='metrics_out/inception_resnet_v2_calibration'
)

# 判断是否需要重新校准
if results['overall_brier_score'] < 0.10:
    print("✓ 模型校准良好,可直接部署")
else:
    print("⚠ 建议应用校准方法(如Platt Scaling)")
```

**预期输出**: 判断模型是否适合用于概率阈值决策

### 案例2: 对比两个模型的校准性能

```python
# 场景: 比较模型A和模型B哪个校准更好
from tools.evaluate_calibration import evaluate_model_calibration

# 评估模型A
results_A = evaluate_model_calibration(
    csv_path='metrics_out/model_A/detailed_predictions.csv',
    output_dir='metrics_out/calibration_A',
    verbose=False
)

# 评估模型B
results_B = evaluate_model_calibration(
    csv_path='metrics_out/model_B/detailed_predictions.csv',
    output_dir='metrics_out/calibration_B',
    verbose=False
)

# 对比Brier Score
bs_A = results_A['overall_brier_score']
bs_B = results_B['overall_brier_score']

print(f"模型A Brier Score: {bs_A:.4f}")
print(f"模型B Brier Score: {bs_B:.4f}")
print(f"差异: {abs(bs_A - bs_B):.4f}")

if bs_A < bs_B:
    print("✓ 模型A校准更好")
else:
    print("✓ 模型B校准更好")
```

**预期输出**: 识别校准性能更好的模型

### 案例3: 实时推理评估(不使用CSV)

```python
# 场景: 快速评估新训练的模型,无需先运行eval.py
from eval import Eval_Classification
from tools.evaluate_calibration import evaluate_model_calibration

# 加载模型
model = Eval_Classification()

# 直接评估
results = evaluate_model_calibration(
    model_instance=model,
    annotation_path='cls_test.txt',
    output_dir='metrics_out/quick_calibration'
)
```

**预期输出**: 快速获取校准分析结果

---

## 常见问题 (FAQ)

### Q1: Calibration Plot和ROC曲线有什么区别?

**A**:
- **ROC曲线**: 评估模型区分类别的能力(排序能力)
- **Calibration Plot**: 评估模型输出概率的可靠性(概率准确性)

**示例**:
- 模型A: AUC=0.95, BS=0.25 → 排序能力强,但概率不可靠
- 模型B: AUC=0.90, BS=0.08 → 排序稍弱,但概率高度可靠

**推荐**: 同时关注AUC和Brier Score

---

### Q2: 什么时候应该进行模型重新校准?

**A**: 以下情况建议重新校准:
1. Brier Score ≥ 0.15 (2分类) 或 ≥ baseline × 0.6 (多分类)
2. Calibration Plot曲线明显偏离对角线
3. 需要使用概率阈值进行决策(如医学诊断、风险控制)
4. 模型在新数据分布上部署

**常用校准方法**:
- Platt Scaling (适合SVM、神经网络)
- Isotonic Regression (无参数假设,更灵活)
- Temperature Scaling (深度学习模型常用)

---

### Q3: 等宽分桶(uniform)和等频分桶(quantile)如何选择?

**A**:
- **等宽分桶** (`strategy='uniform'`):
  - 将概率空间[0, 1]均匀划分
  - 适合: 概率分布较均匀的场景
  - **默认推荐**

- **等频分桶** (`strategy='quantile'`):
  - 每个桶包含相同数量的样本
  - 适合: 概率高度集中在某些区间(如大量高置信度预测)
  - 可能出现桶边界重叠

**经验**: 优先使用等宽分桶,除非大量样本集中在某些概率区间

---

### Q4: 分桶数量(n_bins)如何选择?

**A**:
| 样本量 | 推荐n_bins | 说明 |
|--------|-----------|------|
| < 100 | 5 | 避免桶内样本过少 |
| 100-500 | 10 | **默认值,适合大多数场景** |
| 500-1000 | 15 | 更精细的校准分析 |
| > 1000 | 20 | 高精度校准曲线 |

**原则**: n_bins × 2 ≤ 样本量(确保每个桶有足够样本)

---

### Q5: 报错 "CSV文件未找到概率列" 怎么办?

**A**: 检查CSV文件格式:

**正确格式** (eval.py生成的格式):
```csv
path,true,predict,normal_probability,abnormal_probability
datasets/test/0/img1.jpg,0,0,0.92,0.08
```

或

```csv
path,true,predict,class_0_prob,class_1_prob
datasets/test/0/img1.jpg,0,0,0.92,0.08
```

**错误格式** (不支持):
- 列名: `prob_0`, `probability_normal` (不符合命名规范)
- 缺少概率列(只有path, true, predict)

**解决方法**: 重新运行 `eval.py` 生成正确格式的CSV

---

### Q6: 如何解读各类别校准图?

**A**: 以2分类为例:

**类别0 (normal) 校准图**:
- X轴: 模型预测为类别0的概率
- Y轴: 真实为类别0的样本比例
- **完美校准**: 概率0.8 → 真实比例0.8

**类别1 (abnormal) 校准图**:
- X轴: 模型预测为类别1的概率
- Y轴: 真实为类别1的样本比例

**常见问题**:
- 某类别校准差,其他类别好 → 可能数据不平衡导致
- 所有类别都校准差 → 建议重新训练或应用全局校准方法

---

### Q7: Brier Score和ECE(Expected Calibration Error)有什么区别?

**A**:
- **Brier Score**:
  - 概率预测的均方误差
  - 同时考虑校准和分辨率
  - sklearn标准API支持

- **ECE** (本工具未实现):
  - 仅衡量校准偏差
  - 更直观,但需要手动实现

**本工具选择**: Brier Score因其标准化和广泛认可而被采用

---

### Q8: 能否在训练过程中监控校准性能?

**A**: 可以,修改`train_trimm.py`添加验证集校准监控:

```python
# 在验证阶段添加
from tools.evaluate_calibration import compute_brier_score_multiclass

def validate_epoch(model, val_loader):
    # 收集预测概率和真实标签
    all_probs = []
    all_labels = []

    for images, labels in val_loader:
        probs = model(images)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    # 计算Brier Score
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    bs = compute_brier_score_multiclass(labels, probs)

    print(f"Validation Brier Score: {bs:.4f}")
    return bs
```

---

### Q9: 模型校准好但AUC低,该如何改进?

**A**:
- **分析**: 模型的概率输出可靠,但区分能力不足
- **改进方向**:
  1. 增强特征提取(更换模型架构)
  2. 数据增强或收集更多数据
  3. 调整损失函数(如Focal Loss)
  4. 超参数调优

**注意**: 校准好是前提,先提升AUC,再调整校准

---

### Q10: 如何将校准结果用于论文?

**A**: 推荐内容:

**1. 表格**: 在论文中添加模型性能对比表
```
| 模型 | AUC | Brier Score | 校准质量 |
|------|-----|-------------|---------|
| 模型A | 0.95 | 0.089 | 优秀 |
| 模型B | 0.93 | 0.145 | 良好 |
```

**2. 图表**: 使用生成的`calibration_overall.png`
- 标题: "Model Calibration Performance"
- 说明: "The calibration curve shows the relationship between predicted probabilities and actual frequencies. Closer alignment with the diagonal indicates better calibration."

**3. 文本描述**:
"模型的整体Brier Score为0.089,表明概率预测高度可靠。校准曲线(图X)显示预测概率与实际正确率高度吻合,适合用于临床决策支持系统。"

---

## 技术细节

### 校准曲线计算方法

本工具使用`sklearn.calibration.calibration_curve` API:

#### 整体校准曲线
```python
# 步骤1: 提取最大概率和正确性
max_probs = np.max(probs, axis=1)
predicted_labels = np.argmax(probs, axis=1)
correct = (predicted_labels == labels).astype(int)

# 步骤2: 使用sklearn API
true_frequencies, mean_predicted_probs = calibration_curve(
    y_true=correct,
    y_prob=max_probs,
    n_bins=10,
    strategy='uniform'
)
```

#### 各类别校准曲线 (One-vs-Rest)
```python
# 步骤1: 二值化
binary_labels = (labels == class_idx).astype(int)
class_probs = probs[:, class_idx]

# 步骤2: 使用sklearn API
true_frequencies, mean_predicted_probs = calibration_curve(
    y_true=binary_labels,
    y_prob=class_probs,
    n_bins=10,
    strategy='uniform'
)
```

### Brier Score计算公式

#### 多分类Brier Score (整体)
```
BS = (1/N) * Σ_{i=1}^{N} Σ_{j=1}^{C} (p_{ij} - y_{ij})^2

其中:
  N = 样本数量
  C = 类别数量
  p_{ij} = 样本i预测为类别j的概率
  y_{ij} = 样本i真实标签的one-hot编码
```

#### 单类别Brier Score (One-vs-Rest)
```
BS_k = (1/N) * Σ_{i=1}^{N} (p_{ik} - y_{ik})^2

其中:
  p_{ik} = 样本i预测为类别k的概率
  y_{ik} = 样本i是否为类别k (0或1)
```

### 适用场景

✅ **适用**:
- 需要使用概率阈值进行决策的任务
- 医学诊断、风险评估等高风险场景
- 模型部署前的最终验证
- 多模型选择时的辅助指标

❌ **不适用**:
- 仅关心分类准确率,不使用概率值
- 样本量过小(<50样本,分桶不稳定)
- 只需要排序能力(如推荐系统,使用AUC即可)


## 更新日志

### v1.0.0 (2025-11-28)
- ✅ 初始版本发布
- ✅ 支持整体和各类别校准分析
- ✅ 基于sklearn.calibration.calibration_curve实现
- ✅ Brier Score计算(整体+各类别)
- ✅ 专业可视化(英文+Times New Roman字体)
- ✅ 完整文本报告生成(中文)
- ✅ 支持CSV加载和实时推理两种模式
- ✅ 等宽/等频分桶策略
- ✅ 纯Python函数接口,无命令行依赖

---


**Happy Calibrating! 📊**
