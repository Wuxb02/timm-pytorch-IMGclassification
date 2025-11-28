# GRAD-CAM++热图可视化工具包

## 概述

本工具包提供了基于GRAD-CAM++算法的深度学习模型可解释性分析功能,帮助理解卷积神经网络(CNN)的决策过程。

## 主要特性

- ✅ **GRAD-CAM++算法**: 相比标准GRAD-CAM更精确的权重计算
- ✅ **自动目标层检测**: 支持10+种主流CNN架构的自动层选择
- ✅ **Python API**: 简单易用的函数接口,无需命令行
- ✅ **批量处理**: 支持单张图片和整个文件夹的批量处理
- ✅ **GPU加速**: 自动GPU/CPU适配
- ✅ **可视化定制**: JET颜色映射,可调透明度

## 快速开始

### 安装依赖

确保已安装以下依赖包:

```bash
pip install opencv-python>=4.5.0
pip install torch torchvision
pip install tqdm
```

或使用项目的requirements.txt:

```bash
pip install -r requirements.txt
```

### 基础使用

#### 1. 单张图片处理

```python
from tools.visualize_gradcam import generate_gradcam

result = generate_gradcam(
    image_path='datasets/test/1/sample.jpg',
    output_path='cam_output/sample_gradcam.jpg',
    alpha=0.5  # 热图透明度
)

print(f"预测: {result['pred_name']}, 置信度: {result['confidence']:.3f}")
```

#### 2. 批量处理

```python
from tools.visualize_gradcam import generate_gradcam_batch

results = generate_gradcam_batch(
    image_dir='datasets/test/1/',
    output_dir='cam_output/batch_analysis',
    save_report=True  # 生成CSV报告
)

print(f"处理完成,共{len(results)}张图片")
```

#### 3. 快速模式(默认配置)

```python
from tools.visualize_gradcam import quick_gradcam

result = quick_gradcam('test.jpg', 'test_gradcam.jpg')
```

### 高级用法

#### 指定模型和参数

```python
result = generate_gradcam(
    image_path='image.jpg',
    model_path='models/resnet50/best_epoch_weights.pth',
    backbone='resnet50',
    classes_path='model_data/cls_classes.txt',
    input_shape=(224, 224),
    target_class=1,  # 指定目标类别(None则使用预测类别)
    alpha=0.6,
    output_path='output.jpg',
    cuda=True
)
```

#### 获取热图数组

```python
result = generate_gradcam(
    image_path='image.jpg',
    return_image=True  # 返回图片数组
)

# 访问热图数据
cam_array = result['cam']  # [H, W] 数组,值范围[0, 1]
overlay_image = result['overlay']  # [H, W, 3] RGB图像
```

## 支持的模型

### ✅ 支持的CNN架构

| 模型系列 | 具体模型 | 目标层 |
|---------|---------|--------|
| InceptionResNetV2 | inception_resnet_v2 | conv_final |
| ResNet | resnet18/34/50/101/152 | layer4 |
| VGG | vgg11/13/16 (含BN版本) | features.N |
| DenseNet | densenet121/161/169/201 | features.denseblock4 |
| MobileNet | mobilenetv2 | features.18 |
| EfficientNet | efficientnet_b0~b7 | conv_head |
| ConvNeXt | convnext_tiny/small/base | stages.3 |
| Xception | xception | conv4 |

### ❌ 不支持的架构

- **Vision Transformer (ViT)**: 无卷积层,需要使用Attention Map方法
- **Swin Transformer**: 需要窗口注意力可视化

## API文档

### `generate_gradcam()`

生成单张图片的GRAD-CAM++热图。

**参数**:

- `image_path` (str): 输入图片路径
- `model_path` (str): 模型权重路径,默认'models/inception_resnet_v2/best_epoch_weights.pth'
- `backbone` (str): 模型架构名称,默认'inception_resnet_v2'
- `classes_path` (str): 类别定义文件,默认'model_data/cls_classes.txt'
- `input_shape` (tuple): 输入尺寸(H, W),默认(299, 299)
- `target_class` (int, optional): 目标类别索引,None表示使用预测类别
- `alpha` (float): 热图透明度[0, 1],默认0.5
- `output_path` (str, optional): 输出路径,None表示不保存
- `cuda` (bool): 是否使用GPU,默认True
- `return_image` (bool): 是否返回图片数组,默认False

**返回**:

字典,包含以下键:
- `pred_class`: 预测类别索引
- `pred_name`: 预测类别名称
- `confidence`: 预测置信度
- `cam`: 热图数组[H, W]
- `overlay`: 叠加后的图片数组[H, W, 3] (如果return_image=True)
- `output_path`: 保存路径 (如果指定了output_path)

### `generate_gradcam_batch()`

批量处理文件夹内的所有图片。

**参数**:

- `image_dir` (str): 输入图片文件夹
- `output_dir` (str): 输出文件夹,默认'cam_output'
- `save_report` (bool): 是否保存CSV报告,默认False
- 其他参数同`generate_gradcam()`

**返回**:

结果列表,每个元素为包含预测信息的字典。

## 工作原理

### GRAD-CAM++算法

GRAD-CAM++(Gradient-weighted Class Activation Mapping Plus Plus)是对GRAD-CAM的改进版本:

1. **前向传播**: 输入图像通过网络,获取预测结果和目标层激活值
2. **反向传播**: 对目标类别分数进行反向传播,获取梯度
3. **权重计算** (GRAD-CAM++改进):
   ```
   alpha = grad^2 / (2 * grad^2 + sum(A) * grad^3 + epsilon)
   weights = sum(alpha * ReLU(grad))
   ```
4. **热图生成**: 加权求和激活值,ReLU激活,归一化
5. **可视化**: 叠加到原图,应用JET颜色映射

### 相比GRAD-CAM的优势

- **更精确的定位**: 特别适合多目标场景
- **更少的噪声**: 改进的权重归一化
- **更好的泛化**: 考虑激活值和梯度的高阶关系

## 输出说明

### 单张处理输出

```
cam_output/
└── sample_gradcam.jpg  # 热图叠加原图(JET颜色映射)
```

### 批量处理输出

```
cam_output/
└── batch_analysis/
    ├── img001_gradcam.jpg
    ├── img002_gradcam.jpg
    ├── ...
    └── gradcam_report.csv  # CSV报告(可选)
```

**CSV报告格式**:

```csv
图片路径,预测类别,置信度,热图路径
datasets/test/1/img001.jpg,abnormal,0.9234,cam_output/batch_xxx/img001_gradcam.jpg
```

## 常见问题

### Q1: 提示"模型不支持GRAD-CAM"

**原因**: 使用了Transformer架构(如ViT、Swin)。

**解决**: GRAD-CAM仅支持CNN架构,Transformer需要使用Attention Map方法。

### Q2: 提示"目标层映射未定义"

**原因**: 使用了自定义模型或不在支持列表中的模型。

**解决**:
1. 检查模型名称是否正确
2. 查看`model_targets.py`中的`TARGET_LAYER_MAP`
3. 使用`add_custom_mapping()`添加自定义映射:

```python
from tools.cam.model_targets import add_custom_mapping

add_custom_mapping('my_model', 'final_conv_layer')
```

### Q3: GPU内存不足

**解决**:
1. 设置`cuda=False`使用CPU
2. 减少输入图像尺寸
3. 单张处理而非批量处理

### Q4: 热图质量不佳

**优化建议**:
1. 调整透明度`alpha`(0.3~0.7)
2. 确保模型在该任务上性能良好
3. 尝试不同的目标层(手动指定)

## 示例脚本

项目根目录下的`example_gradcam.py`提供了完整的使用示例:

```bash
python example_gradcam.py
```

包含:
1. 单张图片处理
2. 批量处理文件夹
3. 快速使用模式
4. 多模型对比

## 技术参考

### 论文

- **GRAD-CAM++**: [Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks](https://arxiv.org/abs/1710.11063)
- **GRAD-CAM**: [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)

### 实现细节

- **Hook机制**: 使用PyTorch的`register_forward_hook`和`register_full_backward_hook`捕获中间层输出
- **梯度计算**: 使用`backward(retain_graph=True)`保留计算图以支持多次调用
- **热图叠加**: 使用OpenCV的`applyColorMap`和透明度混合

## 许可证

本工具包遵循MIT许可证。

## 更新日志

### v1.0.0 (2025-11-28)
- ✅ 初始版本发布
- ✅ GRAD-CAM++核心算法实现
- ✅ 支持10+种CNN架构
- ✅ Python API接口
- ✅ 批量处理功能
- ✅ 自动目标层检测
