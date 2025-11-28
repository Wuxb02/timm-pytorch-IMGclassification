"""
CAM (Class Activation Mapping) 可视化工具包

提供多种CAM方法用于深度学习模型的可解释性分析
"""

from .base_cam import BaseCAM
from .gradcam_plus_plus import GradCAMPlusPlus
from .model_targets import get_target_layer, TARGET_LAYER_MAP, UNSUPPORTED_MODELS

__all__ = [
    'BaseCAM',
    'GradCAMPlusPlus',
    'get_target_layer',
    'TARGET_LAYER_MAP',
    'UNSUPPORTED_MODELS'
]

__version__ = '1.0.0'
