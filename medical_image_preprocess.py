import cv2
import numpy as np
import os
import time
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count

def clean_and_crop_ultrasound(image_path, output_path, padding=10):
    """
    读取超声图像，依次执行：
    1. 识别并修复（Inpaint）图像中的黄色/青色测量文字。
    2. 自动识别最大的超声扇形区域（ROI）。
    3. 裁剪掉多余的黑边，保存处理后的图像。
    
    Args:
        image_path (str): 原图路径
        output_path (str): 保存路径
        padding (int): 裁剪时保留的边缘像素宽度，防止切到病灶
    """
    
    # 1. 读取图像（支持中文路径）
    # 使用 numpy 和 cv2.imdecode 来支持中文路径
    try:
        img_data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[Error] 读取图像失败: {image_path}, 错误: {e}")
        return False

    if img is None:
        print(f"[Error] 无法解码图像: {image_path}")
        return False

    # =========================================================
    # 第一阶段：去除干扰文字 (Inpainting)
    # =========================================================
    
    # 转换到 HSV 空间以便提取颜色
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 定义黄色范围 (针对 "2 D 4.57cm" 等黄色高亮字)
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    
    # 定义青色/绿色范围 (针对标尺或机器参数)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])

    # 创建掩膜
    mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_g = cv2.inRange(hsv, lower_green, upper_green)
    
    # 合并掩膜并进行膨胀处理（确保覆盖文字边缘）
    text_mask = cv2.bitwise_or(mask_y, mask_g)
    kernel = np.ones((3, 3), np.uint8)
    text_mask = cv2.dilate(text_mask, kernel, iterations=1)
    
    # 使用修复算法去除文字 (Telea算法速度快效果好)
    # 这一步生成的 clean_img 是没有文字的全尺寸图
    clean_img = cv2.inpaint(img, text_mask, 3, cv2.INPAINT_TELEA)

    # =========================================================
    # 第二阶段：自动 ROI 裁剪 (基于去字后的图像)
    # =========================================================
    
    # 转灰度
    gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
    
    # 二值化：提取较亮的区域 (阈值设为15，过滤掉纯黑背景)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    
    # 形态学开运算：去除噪点，连接大的区域
    morph_kernel = np.ones((5, 5), np.uint8)
    mask_roi = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, morph_kernel, iterations=2)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"[Warning] 未在图像中检测到有效轮廓: {image_path}")
        # 如果找不到轮廓，就保存去字后的原图，不再裁剪，避免报错
        # 使用 cv2.imencode 支持中文路径
        try:
            _, img_encode = cv2.imencode('.png', clean_img)
            img_encode.tofile(output_path)
        except Exception as e:
            print(f"[Error] 保存图像失败: {output_path}, 错误: {e}")
            return False
        return True

    # 找到面积最大的轮廓（假设最大的连通域就是超声扇面）
    c = max(contours, key=cv2.contourArea)
    
    # 获取外接矩形
    x, y, w, h = cv2.boundingRect(c)
    
    # 应用 Padding (确保不切坏边缘)
    h_img, w_img = clean_img.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)
    
    # 执行裁剪
    final_img = clean_img[y1:y2, x1:x2]
    
    # =========================================================
    # 保存结果（支持中文路径）
    # =========================================================
    # 使用 cv2.imencode 支持中文路径
    try:
        # 根据输出文件扩展名选择编码格式
        _, ext = os.path.splitext(output_path)
        if not ext:
            ext = '.png'  # 默认使用 PNG 格式
        _, img_encode = cv2.imencode(ext, final_img)
        img_encode.tofile(output_path)
    except Exception as e:
        print(f"[Error] 保存图像失败: {output_path}, 错误: {e}")
        return False

    # print(f"[Success] 处理完成: {os.path.basename(output_path)}")
    return True


def get_valid_image_files(directory: str) -> List[str]:
    """
    获取目录下所有有效的图片文件

    Args:
        directory (str): 目录路径

    Returns:
        List[str]: 图片文件名列表（不含路径）
    """
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}

    try:
        files = os.listdir(directory)
    except Exception as e:
        print(f"[Error] 无法读取目录 {directory}: {e}")
        return []

    # 过滤出有效的图片文件
    image_files = []
    for f in files:
        # 跳过特殊文件
        if f.startswith('.') or f == '.gitkeep':
            continue

        # 检查扩展名
        _, ext = os.path.splitext(f)
        if ext.lower() in valid_extensions:
            image_files.append(f)

    # 排序后返回
    image_files.sort()
    return image_files


def process_single_image(args: Tuple) -> Tuple[bool, str, str]:
    """
    单个图片处理的包装函数（用于多进程）

    Args:
        args (Tuple): (input_path, output_path, padding)

    Returns:
        Tuple[bool, str, str]: (是否成功, 输入路径, 错误信息)
    """
    input_path, output_path, padding = args
    try:
        success = clean_and_crop_ultrasound(input_path, output_path, padding)
        if success:
            return (True, input_path, "")
        else:
            return (False, input_path, "处理失败（函数返回False）")
    except Exception as e:
        return (False, input_path, str(e))


def print_processing_summary(stats: Dict) -> None:
    """
    打印处理结果摘要

    Args:
        stats (Dict): 处理统计信息字典
    """
    print("\n处理完成！")
    print("=" * 60)
    print(f"总文件数: {stats['total_files']}")
    print(f"成功处理: {stats['success_count']}")
    print(f"失败: {stats['failed_count']}")

    # 计算成功率
    if stats['total_files'] > 0:
        success_rate = (stats['success_count'] / stats['total_files']) * 100
        print(f"成功率: {success_rate:.2f}%")

    # 显示处理时间
    processing_time = stats['processing_time']
    minutes = int(processing_time // 60)
    seconds = int(processing_time % 60)
    print(f"总用时: {minutes}分{seconds}秒")

    # 如果有失败文件，列出详情
    if stats['failed_files']:
        print("\n失败文件列表:")
        for i, (filepath, error) in enumerate(stats['failed_files'], 1):
            print(f"{i}. {filepath}")
            print(f"   错误: {error}")

    print("=" * 60)


def process_class_directory(
    input_class_path: str,
    output_class_path: str,
    padding: int = 10,
    num_workers: Optional[int] = None
) -> Dict:
    """
    使用多进程处理单个类别目录下的所有图片

    Args:
        input_class_path (str): 输入类别目录路径
        output_class_path (str): 输出类别目录路径
        padding (int): 裁剪边缘保留像素（默认10）
        num_workers (Optional[int]): 进程数（默认为CPU核心数-1）

    Returns:
        Dict: 处理统计信息
    """
    # 创建输出目录
    os.makedirs(output_class_path, exist_ok=True)

    # 获取所有图片文件
    image_files = get_valid_image_files(input_class_path)

    if not image_files:
        print(f"  类别 {os.path.basename(input_class_path)}: 空目录，跳过")
        return {
            'total': 0,
            'success': 0,
            'failed': 0,
            'failed_files': []
        }

    # 确定进程数
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    # 准备任务列表
    tasks = []
    for filename in image_files:
        input_path = os.path.join(input_class_path, filename)
        output_path = os.path.join(output_class_path, filename)
        tasks.append((input_path, output_path, padding))

    # 显示开始信息
    class_name = os.path.basename(input_class_path)
    print(f"  处理类别 {class_name}...")
    print(f"  总计: {len(tasks)} 张图片")

    # 多进程处理
    with Pool(num_workers) as pool:
        results = pool.map(process_single_image, tasks)

    # 统计结果
    success_count = sum(1 for success, _, _ in results if success)
    failed_files = [(path, err) for success, path, err in results
                    if not success]

    # 显示完成信息
    print(f"  完成: {success_count}/{len(tasks)} | 失败: {len(failed_files)}")

    return {
        'total': len(tasks),
        'success': success_count,
        'failed': len(failed_files),
        'failed_files': failed_files
    }


def process_split_directory(
    input_split_path: str,
    output_split_path: str,
    padding: int = 10,
    num_workers: Optional[int] = None
) -> Dict:
    """
    处理单个数据集划分（train/val/test）

    Args:
        input_split_path (str): 输入划分目录路径
        output_split_path (str): 输出划分目录路径
        padding (int): 裁剪边缘保留像素
        num_workers (Optional[int]): 进程数

    Returns:
        Dict: 处理统计信息
    """
    # 创建输出目录
    os.makedirs(output_split_path, exist_ok=True)

    # 自动识别所有类别目录
    class_dirs = []
    try:
        for item in os.listdir(input_split_path):
            item_path = os.path.join(input_split_path, item)
            if os.path.isdir(item_path):
                class_dirs.append(item)
    except Exception as e:
        print(f"[Error] 无法读取目录 {input_split_path}: {e}")
        return {
            'total': 0,
            'success': 0,
            'failed': 0,
            'failed_files': []
        }

    if not class_dirs:
        print(f"  未找到类别目录，跳过")
        return {
            'total': 0,
            'success': 0,
            'failed': 0,
            'failed_files': []
        }

    # 排序类别目录
    class_dirs.sort()

    # 显示检测到的类别
    split_name = os.path.basename(input_split_path)
    print(f"\n处理 {split_name} 目录...")
    print(f"  检测到类别: {class_dirs}")

    # 汇总统计信息
    total_files = 0
    success_count = 0
    failed_count = 0
    all_failed_files = []

    # 处理每个类别
    for class_dir in class_dirs:
        input_class_path = os.path.join(input_split_path, class_dir)
        output_class_path = os.path.join(output_split_path, class_dir)

        stats = process_class_directory(
            input_class_path,
            output_class_path,
            padding,
            num_workers
        )

        total_files += stats['total']
        success_count += stats['success']
        failed_count += stats['failed']
        all_failed_files.extend(stats['failed_files'])

    return {
        'total': total_files,
        'success': success_count,
        'failed': failed_count,
        'failed_files': all_failed_files
    }


def batch_process_dataset(
    input_root: str,
    output_root: Optional[str] = None,
    padding: int = 10,
    num_workers: Optional[int] = None,
    verbose: bool = True
) -> Dict:
    """
    批量处理整个数据集（主函数）

    Args:
        input_root (str): 数据集根目录
        output_root (Optional[str]): 输出根目录（默认为input_root_processed）
        padding (int): 裁剪边缘保留像素（默认10）
        num_workers (Optional[int]): 进程数（默认为CPU核心数-1）
        verbose (bool): 是否显示详细信息（默认True）

    Returns:
        Dict: 处理统计信息
    """
    # 验证输入目录
    if not os.path.exists(input_root):
        print(f"[Error] 输入目录不存在: {input_root}")
        return None

    if not os.path.isdir(input_root):
        print(f"[Error] 输入路径不是目录: {input_root}")
        return None

    # 确定输出目录
    if output_root is None:
        parent_dir = os.path.dirname(input_root.rstrip('/\\'))
        base_name = os.path.basename(input_root.rstrip('/\\'))
        output_root = os.path.join(parent_dir, base_name + '_processed')

    # 创建输出根目录
    os.makedirs(output_root, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("批量处理超声图像数据集")
        print("=" * 60)
        print(f"输入目录: {input_root}")
        print(f"输出目录: {output_root}")
        print(f"边缘保留: {padding} 像素")
        print("=" * 60)

    # 记录开始时间
    start_time = time.time()

    # 检测数据集划分（train/val/test）
    split_dirs = []
    for item in ['train', 'val', 'test']:
        item_path = os.path.join(input_root, item)
        if os.path.exists(item_path) and os.path.isdir(item_path):
            split_dirs.append(item)

    if not split_dirs:
        print("[Warning] 未找到 train/val/test 子目录")
        print("尝试直接处理当前目录...")
        # 如果没有标准划分，直接处理当前目录
        stats = process_split_directory(
            input_root,
            output_root,
            padding,
            num_workers
        )
        total_files = stats['total']
        success_count = stats['success']
        failed_count = stats['failed']
        all_failed_files = stats['failed_files']
    else:
        if verbose:
            print(f"\n检测到数据集划分: {split_dirs}")

        # 汇总统计信息
        total_files = 0
        success_count = 0
        failed_count = 0
        all_failed_files = []

        # 处理每个划分
        for split_dir in split_dirs:
            input_split_path = os.path.join(input_root, split_dir)
            output_split_path = os.path.join(output_root, split_dir)

            stats = process_split_directory(
                input_split_path,
                output_split_path,
                padding,
                num_workers
            )

            total_files += stats['total']
            success_count += stats['success']
            failed_count += stats['failed']
            all_failed_files.extend(stats['failed_files'])

    # 计算总处理时间
    processing_time = time.time() - start_time

    return {
        'total_files': total_files,
        'success_count': success_count,
        'failed_count': failed_count,
        'failed_files': all_failed_files,
        'processing_time': processing_time
    }

if __name__ == "__main__":
    # 批量处理数据集
    input_root = r'datasets'

    print("=" * 60)
    print("批量处理超声图像数据集")
    print("=" * 60)
    print(f"输入目录: {input_root}")
    print(f"输出目录: {input_root}_processed")
    print("=" * 60)

    stats = batch_process_dataset(
        input_root=input_root,
        padding=20,
        num_workers=None,  # 自动使用 CPU核心数-1
        verbose=True
    )

    if stats:
        print("\n" + "=" * 60)
        print_processing_summary(stats)
        print("=" * 60)
    else:
        print("\n处理失败！")
