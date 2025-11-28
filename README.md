# 基于timm的图像分类框架 - PyTorch实现

[![Python](https://img.shields.io/badge/Python-3.6%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.2%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个基于PyTorch的通用图像分类深度学习框架,支持**2至N类别**的图像分类任务。该项目提供了完整的训练、评估和推理流水线,具备类别不平衡处理和全面的性能评估体系。

## 📋 目录

- [主要特性](#主要特性)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [模型支持](#模型支持)
- [性能评估](#性能评估)
- [高级配置](#高级配置)
- [常见问题](#常见问题)
- [更新日志](#更新日志)

## ✨ 主要特性

### 核心功能
- ✅ **多模型支持**: 集成15+主流CNN和Transformer架构
- ✅ **类别不平衡处理**: 加权采样、Focal Loss、类别权重调整
- ✅ **完整评估体系**: Top-1/Top-5准确率、AUC、ROC/PR曲线、Bootstrap置信区间
- ✅ **分布式训练**: 支持DP/DDP多GPU训练(Linux/Ubuntu)
- ✅ **混合精度训练**: FP16支持,减少50%显存占用
- ✅ **迁移学习**: 基于timm库自动加载1000+预训练模型

### 技术亮点
- 🔥 **两阶段训练策略**: 冻结主干→解冻微调,快速收敛
- 🔥 **自适应学习率**: 余弦衰减+warmup,基于批次大小自动调整
- 🔥 **数据增强**: 支持多种增强策略,提升模型泛化能力
- 🔥 **早停与检查点**: 自动保存最佳模型,防止过拟合

## 📁 项目结构

```
classification-pytorch-cls2/
├── datasets/                      # 数据集目录
│   ├── train/                     # 训练集(按类别文件夹组织)
│   │   ├── 0/                     # 类别0图像
│   │   ├── 1/                     # 类别1图像
│   │   └── ...                    # 其他类别
│   └── test/                      # 测试集(同上结构)
├── model_data/                    # 模型配置
│   └── cls_classes.txt            # 类别定义文件
├── models/                        # 预训练权重存储
│   ├── inception_resnet_v2/       # InceptionResNetV2权重
│   ├── convnext_tiny/             # ConvNeXt权重
│   └── vit_base_patch16_224/      # ViT权重
├── nets/                          # 模型架构定义
│   ├── resnet.py                  # ResNet系列
│   ├── vgg.py                     # VGG系列
│   ├── mobilenetv2.py             # MobileNet系列
│   ├── densenet.py                # DenseNet系列
│   ├── inception.py               # Inception系列
│   ├── inceptionResnet.py         # InceptionResNetV2
│   ├── xception.py                # Xception
│   ├── vision_transformer.py      # Vision Transformer
│   └── swin_transformer.py        # Swin Transformer
├── utils/                         # 工具函数
│   ├── dataloader.py              # 数据加载器
│   ├── utils_fit.py               # 训练循环
│   ├── utils_metrics.py           # 评估指标
│   ├── callbacks.py               # 训练回调
│   ├── early_stopping.py          # 早停机制
│   └── focal_loss.py              # Focal Loss实现
├── metrics_out/                   # 评估结果输出
│   ├── confusion_matrix.csv       # 混淆矩阵
│   ├── roc_curves.png             # ROC曲线
│   ├── pr_curves.png              # PR曲线
│   ├── confidence_intervals.png   # 置信区间可视化
│   └── classification_report.txt  # 完整分类报告
├── train_trimm.py                 # 训练脚本(timm版本)
├── classification.py              # 分类推理引擎
├── predict.py                     # 单张图片预测
├── eval.py                        # 模型评估
├── txt_annotation.py              # 数据集标注生成
├── Predict_All_Precision_Calculation.py  # 批量精度计算
├── tools/                         # 高级评估工具
│   ├── compare_models_auc.py      # 双模型统计比较工具
│   ├── evaluate_calibration.py    # 模型校准评估工具
│   └── README.md                  # 工具使用文档
└── requirements.txt               # 依赖清单

```

## 🛠️ 环境配置

### 系统要求
- **操作系统**: Windows/Linux/macOS
- **Python**: 3.6+ (推荐3.8+)
- **CUDA**: 10.0+ (GPU训练需要)
- **显存**: 至少6GB (推荐8GB+)

### 依赖安装

#### 方法1: 一键安装(推荐)
```bash
pip install -r requirements.txt
```

#### 方法2: 手动安装核心依赖
```bash
# 核心深度学习框架
pip install torch>=1.2.0 torchvision>=0.4.0

# 数据处理与可视化
pip install numpy>=1.17.0 matplotlib>=3.1.2 opencv-python>=4.1.2
pip install Pillow>=8.2.0 tqdm>=4.60.0

# 高级评估依赖
pip install scikit-learn>=1.0.0  # AUC、ROC、PR曲线
pip install pandas>=2.0.0        # 数据导出
pip install seaborn>=0.13.0      # 高级可视化

# 模型库(可选,用于timm版本训练)
pip install timm
```

### 版本兼容性说明
| 环境 | PyTorch | scikit-learn | pandas | Python |
|------|---------|--------------|--------|--------|
| 原始环境 | 1.2.0 | 0.21.3 | 0.25.3 | 3.6-3.7 |
| 推荐环境 | 2.0+ | 1.7+ | 2.3+ | 3.8-3.11 |

## 🚀 快速开始

### 1. 准备数据集

**当前示例: 2分类场景** (可扩展至N分类)

将数据集按以下结构组织:
```
datasets/
├── train/
│   ├── 0/  # 类别0(示例: normal - 正常样本)
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── 1/  # 类别1(示例: abnormal - 异常样本)
│       ├── img3.jpg
│       ├── img4.jpg
│       └── ...
└── test/   # 测试集(完全相同的结构)
    ├── 0/
    └── 1/
```

**类别定义文件** (`model_data/cls_classes.txt`):
```
0, normal
1, abnormal
```

**扩展为多分类** (如3分类、4分类):
```
# 修改 model_data/cls_classes.txt
0, class_name_0
1, class_name_1
2, class_name_2
3, class_name_3

# 添加对应的数据集文件夹
datasets/train/2/
datasets/train/3/
datasets/test/2/
datasets/test/3/
```

**类别命名规范**:
- 类别ID必须从0开始连续递增(0, 1, 2, 3...)
- 类别名称可自定义(建议使用英文,避免特殊字符)
- 格式为: `类别ID, 类别名称` (逗号+空格分隔)

### 2. 生成标注文件

```bash
# Windows环境
"D:\anaconda\python.exe" txt_annotation.py

# Linux/macOS环境
python txt_annotation.py
```

运行后会生成:
- `cls_train.txt`: 训练集标注 (格式: `类别ID;图片路径`)
- `cls_test.txt`: 测试集标注

**标注文件示例**:
```
0;datasets/train/0/image1.jpg
1;datasets/train/1/image2.jpg
```

### 3. 训练模型

```bash
# Windows环境(单GPU)
"D:\anaconda\python.exe" train_trimm.py

# Linux多GPU训练 - DP模式
CUDA_VISIBLE_DEVICES=0,1 python train_trimm.py

# Linux多GPU训练 - DDP模式(推荐)
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_trimm.py
```

**关键训练参数** (在`train_trimm.py`中配置):
```python
backbone = "inception_resnet_v2"  # 模型选择
input_shape = [299, 299]          # 输入尺寸
Freeze_Epoch = 30                 # 冻结训练轮数
UnFreeze_Epoch = 200              # 解冻训练轮数
Freeze_batch_size = 16            # 冻结阶段批次大小
Unfreeze_batch_size = 8           # 解冻阶段批次大小
fp16 = True                       # 混合精度训练
```

### 4. 模型评估

```bash
# 重要说明：训练完成后需要手动运行评估（不再自动调用）

# Windows环境
"D:\anaconda\python.exe" eval.py

# Linux/macOS环境
python eval.py
```

**评估结果保存位置**:
- 输出文件夹: `metrics_out/{模型名称}_{数据集名称}/`
- 示例: `metrics_out/inception_resnet_v2_cls_test/`

**评估输出**:
- **详细预测结果**: `detailed_predictions.csv` - 每个样本的预测标签和所有类别概率
- **终端输出**: Top-1/Top-5准确率、Precision、Recall、F1、AUC、Specificity、Sensitivity
- **可视化文件**:
  - `roc_curves.png` - ROC曲线(含Macro/Micro平均)
  - `pr_curves.png` - PR曲线
  - `confidence_intervals.png` - Bootstrap 95%置信区间
  - `confusion_matrix_detailed.png` - 详细混淆矩阵
  - `classification_report.txt` - 完整分类报告

**CSV详细预测结果示例**:
```csv
图片路径,真实标签,预测标签,normal_probability,abnormal_probability
datasets/test/0/img001.jpg,0,0,0.9234,0.0766
datasets/test/1/img002.jpg,1,1,0.1234,0.8766
datasets/test/0/img003.jpg,0,1,0.4521,0.5479
```
- 列名根据`model_data/cls_classes.txt`格式自动适配
- 使用UTF-8-BOM编码，中文路径兼容Excel

### 5. 模型预测

#### 单张图片预测(交互式)
```bash
# Windows环境
"D:\anaconda\python.exe" predict.py

# 然后输入图片路径
Input image filename: path/to/your/image.jpg
```

#### 批量预测(脚本式)
```bash
# Windows环境
"D:\anaconda\python.exe" Predict_All_Precision_Calculation.py
```

## 🏆 模型支持

### 支持的模型架构

基于[timm库](https://github.com/huggingface/pytorch-image-models)(1000+预训练模型):

| 模型系列 | 模型名称 | 输入尺寸 | 参数量 | 推荐场景 |
|---------|---------|---------|--------|---------|
| **CNN - 高准确率** |
| InceptionResNetV2 | `inception_resnet_v2` | 299×299 | 55M | 小数据集,医学影像 ✅ |
| DenseNet | `densenet121/169/201` | 224×224 | 8M/14M/20M | 特征复用,准确率优先 |
| EfficientNet | `efficientnet_b0/b1/b2` | 224×224 | 5M/7M/9M | 效率与准确率平衡 |
| **CNN - 高效率** |
| MobileNetV2 | `mobilenetv2_100` | 224×224 | 3.5M | 移动端部署,实时推理 |
| **CNN - 现代架构** |
| ConvNeXt | `convnext_tiny/small` | 224×224 | 28M/50M | 现代CNN,性能强劲 |
| **Transformer** |
| Vision Transformer | `vit_base_patch16_224` | 224×224 | 86M | 大数据集,全局特征 |
| Swin Transformer | `swin_tiny/small/base` | 224×224 | 28M/50M/88M | 分层Transformer |
| **经典CNN** |
| ResNet | `resnet18/50/101/152` | 224×224 | 11M/25M/44M/60M | 通用场景,基线模型 |
| VGG | `vgg16/vgg16_bn` | 224×224 | 138M | 简单任务,可解释性强 |
| Xception | `xception` | 299×299 | 23M | 深度可分离卷积 |

### 模型选择指南

| 优先级 | 推荐模型 | 理由 |
|--------|---------|------|
| 🥇 小数据集(< 5000张) | `inception_resnet_v2` | 当前默认,泛化能力强 |
| 🥈 中等数据集(5K-50K) | `efficientnet_b0`, `convnext_tiny` | 效率与准确率平衡 |
| 🥉 大数据集(> 50K) | `vit_base_patch16_224`, `swin_transformer_base` | Transformer优势明显 |
| ⚡ 速度优先 | `mobilenetv2_100` | 推理速度快,显存占用低 |

### 模型切换方法

1. 修改 `train_trimm.py`:
```python
backbone = "efficientnet_b0"  # 更换模型
input_shape = [224, 224]      # 调整输入尺寸(根据模型要求)
```

2. 修改 `classification.py` (用于预测和评估):
```python
"backbone": 'efficientnet_b0',
"input_shape": [224, 224],
"model_path": 'models/efficientnet_b0/best_epoch_weights.pth',
```

**⚠️ 重要配置同步警告**:

训练、评估、预测中的以下参数**必须完全一致**,否则会报错:

| 参数 | train_trimm.py | classification.py | 说明 |
|------|----------------|-------------------|------|
| `backbone` | 第123行 | 第39行 | 模型架构名称 |
| `input_shape` | 第115行 | 第28行 | 输入图像尺寸 |
| `classes_path` | 第111行 | 第24行 | 类别定义文件 |

**当前配置状态检查**:
- `train_trimm.py`: backbone=`inception_resnet_v2`, input_shape=`[224, 224]`
- `classification.py`: backbone=`inception_resnet_v2`, input_shape=`[299, 299]` ⚠️ **不一致!**

**修复方法**: 将 [classification.py:28](classification.py#L28) 的 `input_shape` 改为 `[224, 224]`,或将 [train_trimm.py:115](train_trimm.py#L115) 改为 `[299, 299]` (推荐,InceptionResNetV2原始尺寸)

## 📊 性能评估

### 基础评估 (eval.py)

运行`eval.py`可获得完整的模型性能评估,包括基础指标和高级统计分析。

### 高级评估工具 (tools/)

框架提供三个专业评估工具,位于`tools/`目录:

#### 1. GRAD-CAM++可解释性热图 (`visualize_gradcam.py`) ⭐ **新增**

**功能说明**: 生成GRAD-CAM++热图,可视化模型决策依据,帮助理解模型关注的图像区域。

**Python API使用**:

```python
from tools.visualize_gradcam import generate_gradcam, generate_gradcam_batch

# 单张图片处理
result = generate_gradcam(
    image_path='datasets/test/1/sample.jpg',
    model_path='models/inception_resnet_v2/best_epoch_weights.pth',
    backbone='inception_resnet_v2',
    output_path='cam_output/sample_gradcam.jpg',
    alpha=0.5,  # 热图透明度
    cuda=True
)

print(f"预测类别: {result['pred_name']}")
print(f"置信度: {result['confidence']:.3f}")
print(f"热图已保存: {result['output_path']}")

# 批量处理
results = generate_gradcam_batch(
    image_dir='datasets/test/1/',
    output_dir='cam_output/batch_analysis',
    save_report=True  # 生成CSV报告
)

print(f"批量处理完成,共{len(results)}张图片")
```

**快速开始**:

```python
from tools.visualize_gradcam import quick_gradcam

# 使用默认配置快速生成热图
result = quick_gradcam('test.jpg', 'test_gradcam.jpg')
```

**支持的模型架构**:
- ✅ CNN架构: InceptionResNetV2, ResNet系列, VGG系列, DenseNet系列, MobileNetV2, EfficientNet系列
- ❌ Transformer架构: ViT, Swin Transformer (需要使用Attention Map方法)

**输出示例**:
- 单张处理: `cam_output/sample_gradcam.jpg` (热图叠加原图,JET颜色映射)
- 批量处理: `cam_output/batch_xxx/` 文件夹 + `gradcam_report.csv` 报告

**技术特点**:
- 使用GRAD-CAM++算法(相比GRAD-CAM更精确的权重计算)
- 自动目标层检测,无需手动指定
- GPU/CPU自动适配
- 批量处理带进度条

#### 2. 双模型统计比较 (`compare_models_auc.py`)
- **功能**: 使用配对Bootstrap方法比较两个模型的性能差异
- **支持指标**: Macro/Micro AUC、Accuracy、Precision、Recall、F1
- **输出**: 置信区间、p值、效应量、专业可视化
- **详细文档**: 见 [tools/README.md](tools/README.md#1️⃣-双模型多指标统计比较工具)

**使用示例**:
```python
from tools.compare_models_auc import compare_two_models

results = compare_two_models(
    'metrics_out/model_A/detailed_predictions.csv',
    'metrics_out/model_B/detailed_predictions.csv',
    model_name1='InceptionResNetV2',
    model_name2='ResNet50',
    metrics=['macro_auc', 'accuracy', 'macro_f1']
)
```

#### 2. 模型校准评估 (`evaluate_calibration.py`)
- **功能**: 评估模型输出概率的可靠性(Calibration Plot + Brier Score)
- **支持**: 整体校准和各类别独立校准分析
- **输出**: 校准曲线、Brier Score、质量评级、中英文报告
- **详细文档**: 见 [tools/README.md](tools/README.md#2️⃣-模型校准性能评估工具)

**使用示例**:
```python
from tools.evaluate_calibration import evaluate_model_calibration

results = evaluate_model_calibration(
    csv_path='metrics_out/inception_resnet_v2_cls_test/detailed_predictions.csv',
    output_dir='metrics_out/calibration_analysis',
    n_bins=10
)

# 判断是否需要重新校准
if results['overall_brier_score'] < 0.10:
    print("✓ 模型校准良好,可直接部署")
```

**何时使用校准评估**:
- ✅ 需要使用概率阈值进行决策(如医学诊断、风险评估)
- ✅ 模型部署前的最终验证
- ✅ AUC相似的模型,选择校准更好的版本

### 评估指标说明

#### 基础指标
- **Top-1 Accuracy**: 最高概率预测是否正确
- **Top-5 Accuracy**: 前5个最高概率中是否包含真实类别
- **Precision**: 精确率(预测为正的样本中真正为正的比例)
- **Recall**: 召回率(真实为正的样本中被正确预测的比例)
- **F1-Score**: Precision和Recall的调和平均

#### 高级指标
- **AUC (Area Under Curve)**: ROC曲线下面积
  - Per-class AUC: 每个类别的独立AUC
  - Macro AUC: 各类别AUC的算术平均
  - Micro AUC: 基于全局样本计算的AUC
- **Specificity**: 特异性(真实为负的样本中被正确预测的比例)
- **Sensitivity**: 灵敏度(等同于Recall)
- **Bootstrap 95% CI**: 基于1000次重采样的置信区间

### 评估报告示例

运行`eval.py`后的终端输出:
```
数据集统计:
  类别分布: {0: 850, 1: 150}
  少数类别: abnormal (索引1, 占比15.0%)

============================================================
基础性能指标
============================================================
top-1 accuracy = 94.32%
top-5 accuracy = 100.00%

每个类别的Recall (召回率):
  normal   : 96.50%
  abnormal : 88.67%

每个类别的Precision (精确率):
  normal   : 95.80%
  abnormal : 90.12%

============================================================
高级性能指标
============================================================
Per-class AUC:
  normal   : 0.9823
  abnormal : 0.9756

Macro AUC  : 0.9789
Micro AUC  : 0.9801

Specificity (特异性):
  normal   : 0.8867
  abnormal : 0.9650

Sensitivity (灵敏度) [等同于Recall]:
  normal   : 0.9650
  abnormal : 0.8867

============================================================
Bootstrap 95% 置信区间 (1000次重采样)
============================================================
Metric          Mean     Lower    Upper    Range
------------------------------------------------------
Accuracy       0.9432   0.9201   0.9612   ±1.05%
Precision      0.9296   0.9089   0.9478   ±0.97%
Recall         0.9259   0.9034   0.9456   ±1.06%
F1             0.9277   0.9067   0.9465   ±1.00%
Macro AUC      0.9789   0.9621   0.9912   ±0.73%
Micro AUC      0.9801   0.9645   0.9923   ±0.70%

评估完成! 所有结果已保存至 metrics_out/
```

### 输出文件说明

| 文件名 | 说明 | 用途 |
|--------|------|------|
| `detailed_predictions.csv` | **【新增】每个样本的详细预测** | 错误分析/数据审查 |
| `confusion_matrix.csv` | 混淆矩阵(CSV格式) | 数据分析 |
| `confusion_matrix_detailed.png` | 混淆矩阵可视化 | 论文/报告 |
| `roc_curves.png` | ROC曲线(含Macro/Micro) | 模型性能对比 |
| `pr_curves.png` | Precision-Recall曲线 | 不平衡数据集评估 |
| `confidence_intervals.png` | 6个指标的95%置信区间 | 统计显著性分析 |
| `Recall.png` | 各类别召回率柱状图 | 快速查看 |
| `Precision.png` | 各类别精确率柱状图 | 快速查看 |
| `metrics_comparison_chart.png` | 指标对比雷达图 | 综合评估 |
| `classification_report.txt` | 完整文本报告 | 归档记录 |

**detailed_predictions.csv详细说明**:
- **列结构**: 图片路径、真实标签、预测标签、每个类别的概率值
- **列名格式**: 根据`cls_classes.txt`自动适配（有类别名用类别名，无类别名用索引）
- **应用场景**:
  - 分析预测错误的样本
  - 找出置信度低的预测
  - 审查边界样本（概率接近0.5）
  - 导出到Excel进行进一步分析

## ⚙️ 高级配置

### 类别不平衡处理

项目针对类别不平衡问题提供了多种解决方案:

#### 1. 加权随机采样 (train_trimm.py:58-92)
```python
# 自动根据类别分布计算采样权重
# 少数类别样本被选中的概率会提升
use_weighted_sampler = True  # 默认启用
```

#### 2. Focal Loss
```python
# 在utils/focal_loss.py中实现
# 自动降低易分样本的权重,关注难分样本
loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
```

#### 3. 类别权重
```python
# 在损失函数中为少数类别赋予更高权重
class_weights = torch.tensor([1.0, 5.0])  # 类别1权重提升5倍
```

### 学习率调度策略

```python
# 余弦衰减学习率(train_trimm.py中配置)
lr_decay_type = "cos"  # 'cos'(推荐) 或 'step'

Init_lr = 1e-2             # 初始学习率(冻结阶段)
Min_lr = Init_lr * 0.01    # 最小学习率

# 解冻阶段学习率(自动降低10倍)
Unfreeze_lr = Init_lr / 10
```

### 数据增强配置

在`utils/utils_aug.py`中支持:
- RandomHorizontalFlip (随机水平翻转)
- RandomVerticalFlip (随机垂直翻转)
- RandomRotation (随机旋转)
- ColorJitter (颜色抖动)
- RandomCrop (随机裁剪)
- Mixup / CutMix (混合增强)

### 早停与检查点

```python
# 在train_trimm.py中配置
early_stopping = EarlyStopping(
    patience=20,              # 连续20轮无改善则停止
    verbose=True,
    delta=0.001               # 最小改善阈值
)

checkpoint = ModelCheckpoint(
    save_dir='models/inception_resnet_v2/',
    monitor='val_loss',       # 监控验证集损失
    save_best_only=True       # 仅保存最佳模型
)
```

## ❓ 常见问题

### Q1: 训练时显存不足怎么办?

**A**: 尝试以下方法:
1. 减小批次大小: `Freeze_batch_size = 8`, `Unfreeze_batch_size = 4`
2. 启用混合精度训练: `fp16 = True` (减少50%显存)
3. 减小输入尺寸: `input_shape = [224, 224]`
4. 更换轻量级模型: `backbone = "mobilenetv2_100"`

### Q2: 如何恢复训练?

**A**: 修改`train_trimm.py`:
```python
model_path = "models/inception_resnet_v2/ep050-loss0.234.pth"  # 检查点路径
Init_Epoch = 50  # 从第50轮继续
```

### Q3: 模型预测时报"shape不匹配"错误?

**A**: 这通常是配置不一致导致的。按以下步骤检查:

**步骤1: 检查配置一致性**
```python
# 打开train_trimm.py,查看:
backbone = "inception_resnet_v2"  # 第123行
input_shape = [224, 224]          # 第115行
classes_path = 'model_data/cls_classes.txt'  # 第111行

# 打开classification.py,确保完全相同:
"backbone": 'inception_resnet_v2',  # 第39行
"input_shape": [224, 224],          # 第28行 (注意当前为299,需修改!)
"classes_path": 'model_data/cls_classes.txt',  # 第24行
```

**步骤2: 验证权重文件匹配**
- `model_path`必须指向当前`backbone`训练的权重
- 例如: `inception_resnet_v2`的权重不能用于`efficientnet_b0`

**步骤3: 验证类别数量**
```bash
# 检查类别定义文件
cat model_data/cls_classes.txt
# 应该显示: 0, normal\n1, abnormal (2个类别)

# 权重文件必须是针对2分类训练的
```

**步骤4: 常见错误示例**
```
❌ 错误: RuntimeError: size mismatch, m1: [1 x 1536], m2: [2048 x 2]
   原因: input_shape不匹配（224 vs 299）

❌ 错误: RuntimeError: size mismatch for classifier.weight
   原因: 权重文件的类别数与cls_classes.txt不符
```

### Q4: Windows系统如何使用多GPU训练?

**A**: Windows默认使用DP模式,会自动调用所有可见GPU:
```python
# 在train_trimm.py中设置
Cuda = True
distributed = False  # Windows不支持DDP
```

### Q5: 如何查看timm支持的所有模型?

**A**: 运行以下Python代码:
```python
import timm
# 查看所有模型
print(timm.list_models())

# 搜索特定模型(如Inception系列)
print(timm.list_models('*inception*'))

# 查看模型详细信息
model = timm.create_model('inception_resnet_v2', pretrained=True)
print(model)
```

### Q6: 评估指标的置信区间如何解读?

**A**: Bootstrap置信区间表示如果重复采样1000次,真实指标值有95%的概率落在[Lower, Upper]范围内:
- **Range越小**: 模型性能越稳定
- **Range越大**: 模型对数据分布敏感,可能存在过拟合

示例: `Accuracy: 0.9432 [0.9201, 0.9612] ±1.05%`
- 真实准确率有95%概率在92.01%到96.12%之间
- 波动范围为±1.05%,表示模型较稳定

### Q7: 如何为新任务准备数据集?

**A**: 按以下步骤操作:
```bash
# 1. 组织数据集目录
datasets/
├── train/
│   ├── class_0/  # 替换为你的类别名称
│   └── class_1/
└── test/
    ├── class_0/
    └── class_1/

# 2. 修改类别定义文件 model_data/cls_classes.txt
0, class_0
1, class_1

# 3. 生成标注文件
python txt_annotation.py

# 4. 同步修改配置
# train_trimm.py 和 classification.py 中的:
# - classes_path
# - input_shape
# - backbone
```

### Q8: 训练时如何监控性能指标?

**A**: 训练过程中会实时输出:
```
Epoch 1/200
Train Loss: 0.6234 | Val Loss: 0.5123 | Val Acc: 78.45%
Epoch 2/200
Train Loss: 0.4567 | Val Loss: 0.3891 | Val Acc: 85.32%
...
```

使用TensorBoard可视化(可选):
```bash
# 在train_trimm.py中添加
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_1')

# 启动TensorBoard
tensorboard --logdir=runs
```

## 📝 更新日志

### 2025-11-28: 校准评估工具与文档完善
- ✅ **新增校准评估工具**: `tools/evaluate_calibration.py` (v1.0.0)
  - Calibration Plot (校准曲线): 可视化预测概率与实际正确率
  - Brier Score计算: 量化概率预测准确性(整体+各类别)
  - 整体+各类别校准分析(One-vs-Rest策略)
  - 基于sklearn.calibration.calibration_curve API实现
  - 专业可视化: 英文输出,Times New Roman字体,适合学术论文
  - 中文文本报告: 包含校准质量评级和改进建议
  - 支持CSV加载和实时推理两种模式
- ✅ **完善工具文档**: 更新`tools/README.md`,新增校准工具完整使用文档
  - 10个常见问题FAQ(含Calibration vs ROC曲线区别、重新校准时机等)
  - 3个实际使用案例(单模型评估、双模型对比、实时推理)
  - 技术细节与Brier Score计算公式说明
- ✅ **更新框架README**: 添加高级评估工具说明和使用示例
- ✅ **评估流程优化**: 移除训练后自动评估，改为手动运行
- ✅ **动态输出文件夹**: 评估结果按`{模型名称}_{数据集名称}`组织
- ✅ **详细预测结果**: 新增`detailed_predictions.csv`，包含每个样本的预测标签和概率
- ✅ **列名智能适配**: 根据`cls_classes.txt`格式自动选择列名风格
- ✅ **文档系统化**: 创建完整README.md，修正CLAUDE.md命名不一致问题
- ✅ **模型支持列表**: 添加15+模型的详细说明和选择指南
- ✅ **常见问题解答**: 补充8个高频问题的解决方案

### 2025-11-27: 完整评估系统与框架泛化
- ✅ 添加AUC指标(Per-class + Macro + Micro)
- ✅ 添加Specificity和Sensitivity计算(One-vs-Rest)
- ✅ 添加Bootstrap 95%置信区间(1000次重采样)
- ✅ 添加ROC/PR曲线可视化(含Macro/Micro平均)
- ✅ 添加置信区间可视化(6个指标,2x3布局)
- ✅ 合并报告生成函数,统一输出格式
- ✅ 框架泛化: 移除医学特定术语,支持通用分类场景
- ✅ 升级依赖: scikit-learn 1.7.2, pandas 2.3.3, seaborn 0.13.2

### 历史版本
- **v1.0**: 基础分类框架,支持ResNet/VGG/MobileNet
- **v1.5**: 添加Transformer支持(ViT/Swin)
- **v2.0**: 集成timm库,支持1000+模型
- **v2.5**: 添加类别不平衡处理和早停机制

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [timm](https://github.com/huggingface/pytorch-image-models) - 预训练模型库
- [scikit-learn](https://scikit-learn.org/) - 机器学习评估工具


**⭐ 如果这个项目对您有帮助,欢迎给个Star支持一下!**
