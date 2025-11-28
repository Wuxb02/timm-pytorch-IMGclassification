"""
GRAD-CAM++实现

论文: Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks
链接: https://arxiv.org/abs/1710.11063

GRAD-CAM++通过改进的权重计算提供更精确的热图定位
"""

import torch
import torch.nn.functional as F
import numpy as np
from .base_cam import BaseCAM


class GradCAMPlusPlus(BaseCAM):
    """
    GRAD-CAM++: 改进的梯度加权类激活映射

    相比GRAD-CAM的改进:
    - 使用加权梯度而非简单的全局平均池化
    - 考虑激活值和梯度的二阶、三阶关系
    - 对多目标场景定位更准确
    """

    def __init__(self, model, target_layer):
        """
        初始化GRAD-CAM++

        Args:
            model: PyTorch模型
            target_layer: 目标卷积层(nn.Module对象)
        """
        super().__init__(model, target_layer)
        self.gradients = None
        self.activations = None

        # 注册前向钩子(捕获激活值)
        self.forward_hook = target_layer.register_forward_hook(self._forward_hook_fn)
        # 注册反向钩子(捕获梯度)
        self.backward_hook = target_layer.register_full_backward_hook(self._backward_hook_fn)

    def _forward_hook_fn(self, module, input, output):
        """前向传播钩子:保存激活值"""
        self.activations = output.detach()

    def _backward_hook_fn(self, module, grad_input, grad_output):
        """反向传播钩子:保存梯度"""
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        """
        生成GRAD-CAM++热图

        Args:
            input_tensor: 输入图像张量 [1, 3, H, W]
            target_class: 目标类别索引(None表示使用预测类别)

        Returns:
            cam: 热图数组 [H, W],值范围 [0, 1]
            pred_class: 预测类别索引
            confidence: 预测置信度
        """
        # 确保模型处于评估模式
        self.model.eval()

        # 1. 前向传播获取预测
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()

        # 确定目标类别
        if target_class is None:
            target_class = pred_class

        # 2. 反向传播获取梯度
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward(retain_graph=True)

        # 3. GRAD-CAM++改进的权重计算
        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]

        # 计算梯度的二阶和三阶
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)

        # 计算激活值在空间维度的总和
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # GRAD-CAM++的alpha权重公式
        # alpha = grad^2 / (2 * grad^2 + sum(A) * grad^3 + epsilon)
        alpha = grad_2 / (2 * grad_2 + sum_activations * grad_3 + 1e-8)

        # 计算每个通道的权重
        # weights = sum(alpha * ReLU(grad))
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # 4. 加权求和激活值
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, H, W]

        # 5. ReLU激活(只保留正贡献)
        cam = F.relu(cam)

        # 6. 归一化到 [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            # 处理全零情况
            cam = np.zeros_like(cam)

        return cam, pred_class, confidence

    def __del__(self):
        """析构函数:移除钩子,避免内存泄漏"""
        if hasattr(self, 'forward_hook'):
            self.forward_hook.remove()
        if hasattr(self, 'backward_hook'):
            self.backward_hook.remove()


class GradCAM(BaseCAM):
    """
    标准GRAD-CAM实现(作为对比参考)

    论文: Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
    链接: https://arxiv.org/abs/1610.02391
    """

    def __init__(self, model, target_layer):
        """
        初始化GRAD-CAM

        Args:
            model: PyTorch模型
            target_layer: 目标卷积层(nn.Module对象)
        """
        super().__init__(model, target_layer)
        self.gradients = None
        self.activations = None

        # 注册钩子
        self.forward_hook = target_layer.register_forward_hook(self._forward_hook_fn)
        self.backward_hook = target_layer.register_full_backward_hook(self._backward_hook_fn)

    def _forward_hook_fn(self, module, input, output):
        """前向传播钩子:保存激活值"""
        self.activations = output.detach()

    def _backward_hook_fn(self, module, grad_input, grad_output):
        """反向传播钩子:保存梯度"""
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        """
        生成GRAD-CAM热图

        Args:
            input_tensor: 输入图像张量 [1, 3, H, W]
            target_class: 目标类别索引(None表示使用预测类别)

        Returns:
            cam: 热图数组 [H, W],值范围 [0, 1]
            pred_class: 预测类别索引
            confidence: 预测置信度
        """
        self.model.eval()

        # 前向传播
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()

        if target_class is None:
            target_class = pred_class

        # 反向传播
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        # GRAD-CAM权重计算:全局平均池化梯度
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # 加权求和激活值
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H, W]

        # ReLU + 归一化
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, pred_class, confidence

    def __del__(self):
        """析构函数:移除钩子"""
        if hasattr(self, 'forward_hook'):
            self.forward_hook.remove()
        if hasattr(self, 'backward_hook'):
            self.backward_hook.remove()
