"""
GRAD-CAM++可视化工具 - Python API

提供简单易用的Python函数接口,用于生成和保存GRAD-CAM++热图
不使用命令行,直接在代码中调用
"""

import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification import Classification
from tools.cam import GradCAMPlusPlus, get_target_layer
from utils.utils import cvtColor, letterbox_image, preprocess_input


def generate_gradcam(
    image_path,
    model_path='models/inception_resnet_v2/best_epoch_weights.pth',
    backbone='inception_resnet_v2',
    classes_path='model_data/cls_classes.txt',
    input_shape=(299, 299),
    target_class=None,
    alpha=0.5,
    output_path=None,
    cuda=True,
    return_image=False
):
    """
    为单张图片生成GRAD-CAM++热图

    Args:
        image_path: 输入图片路径
        model_path: 模型权重路径
        backbone: 模型架构名称
        classes_path: 类别定义文件路径
        input_shape: 输入尺寸(H, W),默认(299, 299)
        target_class: 目标类别索引(None表示使用预测类别)
        alpha: 热图透明度,默认0.5
        output_path: 输出图片路径(None表示不保存)
        cuda: 是否使用GPU,默认True
        return_image: 是否返回图片数组,默认False

    Returns:
        result: 字典,包含以下键值:
            - pred_class: 预测类别索引
            - pred_name: 预测类别名称
            - confidence: 预测置信度
            - cam: 热图数组 [H, W]
            - overlay: 叠加后的图片数组 [H, W, 3] (如果return_image=True)
            - output_path: 保存路径 (如果指定了output_path)

    Example:
        >>> result = generate_gradcam(
        ...     'datasets/test/1/sample.jpg',
        ...     output_path='cam_output/sample_gradcam.jpg'
        ... )
        >>> print(f"预测: {result['pred_name']}, 置信度: {result['confidence']:.3f}")
    """
    # 1. 加载模型
    classifier = Classification(
        model_path=model_path,
        backbone=backbone,
        classes_path=classes_path,
        input_shape=input_shape,
        cuda=cuda
    )

    # 2. 获取目标层
    target_layer = get_target_layer(classifier.model, backbone)

    # 3. 初始化GRAD-CAM++
    grad_cam = GradCAMPlusPlus(classifier.model, target_layer)

    # 4. 加载并预处理图片
    image = Image.open(image_path)
    image_rgb = cvtColor(image)
    original_image = np.array(image_rgb)

    # 调整大小
    image_data = letterbox_image(image_rgb, [input_shape[1], input_shape[0]], False)
    image_data = np.expand_dims(np.transpose(preprocess_input(
        np.array(image_data, np.float32)), (2, 0, 1)), 0)

    # 转换为Tensor
    input_tensor = torch.from_numpy(image_data).type(torch.FloatTensor)
    if cuda and torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    # 5. 生成热图
    cam, pred_class, confidence = grad_cam.generate_cam(input_tensor, target_class)

    # 6. 叠加热图
    overlay = grad_cam.overlay_heatmap(original_image, cam, alpha=alpha)

    # 7. 保存结果
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # 8. 准备返回结果
    result = {
        'pred_class': pred_class,
        'pred_name': classifier.class_names[pred_class],
        'confidence': confidence,
        'cam': cam
    }

    if return_image:
        result['overlay'] = overlay

    if output_path is not None:
        result['output_path'] = output_path

    return result


def generate_gradcam_batch(
    image_dir,
    output_dir='cam_output',
    model_path='models/inception_resnet_v2/best_epoch_weights.pth',
    backbone='inception_resnet_v2',
    classes_path='model_data/cls_classes.txt',
    input_shape=(299, 299),
    target_class=None,
    alpha=0.5,
    cuda=True,
    save_report=False
):
    """
    批量处理文件夹内的所有图片

    Args:
        image_dir: 输入图片文件夹路径
        output_dir: 输出文件夹路径
        model_path: 模型权重路径
        backbone: 模型架构名称
        classes_path: 类别定义文件路径
        input_shape: 输入尺寸(H, W)
        target_class: 目标类别索引(None表示使用预测类别)
        alpha: 热图透明度
        cuda: 是否使用GPU
        save_report: 是否保存CSV报告

    Returns:
        results: 结果列表,每个元素为字典(包含pred_class、pred_name、confidence等)

    Example:
        >>> results = generate_gradcam_batch(
        ...     'datasets/test/1/',
        ...     output_dir='cam_output/batch_test'
        ... )
        >>> print(f"处理完成,共{len(results)}张图片")
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载模型(只加载一次)
    print("加载模型...")
    classifier = Classification(
        model_path=model_path,
        backbone=backbone,
        classes_path=classes_path,
        input_shape=input_shape,
        cuda=cuda
    )

    # 2. 获取目标层
    target_layer = get_target_layer(classifier.model, backbone)

    # 3. 初始化GRAD-CAM++
    grad_cam = GradCAMPlusPlus(classifier.model, target_layer)

    # 4. 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for file in os.listdir(image_dir):
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
            image_files.append(file)

    if len(image_files) == 0:
        print(f"警告: 在 {image_dir} 中未找到图片文件")
        return []

    print(f"找到 {len(image_files)} 张图片")

    # 5. 批量处理
    results = []
    for filename in tqdm(image_files, desc="生成热图"):
        image_path = os.path.join(image_dir, filename)

        try:
            # 加载并预处理图片
            image = Image.open(image_path)
            image_rgb = cvtColor(image)
            original_image = np.array(image_rgb)

            # 调整大小
            image_data = letterbox_image(image_rgb, [input_shape[1], input_shape[0]], False)
            image_data = np.expand_dims(np.transpose(preprocess_input(
                np.array(image_data, np.float32)), (2, 0, 1)), 0)

            # 转换为Tensor
            input_tensor = torch.from_numpy(image_data).type(torch.FloatTensor)
            if cuda and torch.cuda.is_available():
                input_tensor = input_tensor.cuda()

            # 生成热图
            cam, pred_class, confidence = grad_cam.generate_cam(input_tensor, target_class)

            # 叠加热图
            overlay = grad_cam.overlay_heatmap(original_image, cam, alpha=alpha)

            # 保存结果
            basename = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{basename}_gradcam.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            # 记录结果
            results.append({
                'image_path': image_path,
                'pred_class': pred_class,
                'pred_name': classifier.class_names[pred_class],
                'confidence': confidence,
                'output_path': output_path
            })

        except Exception as e:
            print(f"\n处理 {filename} 时出错: {e}")
            continue

    print(f"\n完成! 共处理 {len(results)} 张图片")

    # 6. 保存CSV报告(可选)
    if save_report:
        report_path = os.path.join(output_dir, 'gradcam_report.csv')
        with open(report_path, 'w', encoding='utf-8-sig') as f:
            f.write('图片路径,预测类别,置信度,热图路径\n')
            for result in results:
                f.write(f"{result['image_path']},{result['pred_name']},"
                       f"{result['confidence']:.4f},{result['output_path']}\n")
        print(f"报告已保存: {report_path}")

    return results


if __name__ == '__main__':
    # 使用示例
    print("=" * 80)
    print("GRAD-CAM++可视化工具 - Python API示例")
    print("=" * 80)

    # 示例1: 处理单张图片
    print("\n示例1: 处理单张图片")
    print("-" * 80)

    # 请根据实际情况修改路径
    test_image = r"../datasets/test/1/275358_8.4_1.jpg"

    backbone = 'inception_resnet_v2'

    if os.path.exists(test_image):
        result = generate_gradcam(
            test_image,
            model_path='../models/inception_resnet_v2/best_epoch_weights.pth',
            backbone = backbone,
            classes_path='../model_data/cls_classes.txt',
            output_path = f'../metrics_out/{backbone}/sample_gradcam.jpg',
            alpha=0.5
        )
        print(f"预测类别: {result['pred_name']}")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"热图已保存: {result['output_path']}")
    else:
        print(f"测试图片不存在: {test_image}")

    # # 示例2: 批量处理
    # print("\n示例2: 批量处理")
    # print("-" * 80)

    # test_dir = "datasets/test/1"

    # if os.path.exists(test_dir):
    #     results = generate_gradcam_batch(
    #         test_dir,
    #         output_dir='../metrics_out/cam_output/batch',
    #         save_report=True
    #     )
    #     print(f"批量处理完成,共{len(results)}张图片")
    # else:
    #     print(f"测试目录不存在: {test_dir}")

    # print("\n" + "=" * 80)
    # print("使用说明:")
    # print("  在代码中导入: from tools.visualize_gradcam import generate_gradcam")
    # print("  单张处理: result = generate_gradcam('image.jpg', output_path='output.jpg')")
    # print("  批量处理: results = generate_gradcam_batch('image_folder/', 'output_folder/')")
    # print("=" * 80)
