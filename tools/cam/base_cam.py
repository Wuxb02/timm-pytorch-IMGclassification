"""
CAM方法的抽象基类

提供统一的接口和通用工具函数
"""

from abc import ABC, abstractmethod
import cv2
import numpy as np
import torch
import torch.nn as nn


class BaseCAM(ABC):
    """CAM方法的抽象基类"""

    def __init__(self, model, target_layer):
        """
        初始化CAM对象

        Args:
            model: PyTorch模型
            target_layer: 目标卷积层(nn.Module对象)
        """
        self.model = self.unwrap_model(model)
        self.target_layer = target_layer

    @abstractmethod
    def generate_cam(self, input_tensor, target_class=None):
        """
        生成CAM热图(抽象方法,子类必须实现)

        Args:
            input_tensor: 输入图像张量 [1, 3, H, W]
            target_class: 目标类别索引(None表示使用预测类别)

        Returns:
            cam: 热图数组 [H, W],值范围 [0, 1]
            pred_class: 预测类别索引
            confidence: 预测置信度
        """
        pass

    @staticmethod
    def unwrap_model(model):
        """
        解包DataParallel/DistributedDataParallel包装的模型

        Args:
            model: PyTorch模型

        Returns:
            unwrapped_model: 解包后的模型
        """
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return model.module
        return model

    @staticmethod
    def overlay_heatmap(image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        将热图叠加到原图

        Args:
            image: 原始RGB图像,numpy数组 (H, W, 3),值范围 [0, 255]
            cam: 热图,numpy数组 (H, W),值范围 [0, 1]
            alpha: 热图透明度 [0, 1],默认0.5
            colormap: OpenCV颜色映射,默认JET

        Returns:
            overlay: 叠加后的RGB图像 (H, W, 3),值范围 [0, 255]
        """
        # 确保image是uint8类型
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # 调整热图大小以匹配原图
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        # 转换为彩色热图
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 叠加
        overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)

        return overlay

    def __del__(self):
        """析构函数:清理资源(子类应该重写以移除Hook)"""
        pass
