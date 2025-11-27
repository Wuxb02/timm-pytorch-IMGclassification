import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    """
    Focal Loss 实现，专门处理类别不平衡问题
    
    Args:
        alpha (float or tensor): 各类别的权重，可以是标量或张量
        gamma (float): 聚焦参数，控制困难样本的权重
        num_classes (int): 类别数量
        size_average (bool): 是否计算平均值
    """
    def __init__(self, alpha=None, gamma=2, num_classes=None, size_average=True):
        super(FocalLoss, self).__init__()
        if num_classes is None:
            raise ValueError("必须指定 num_classes 参数!")
        self.size_average = size_average
        self.gamma = gamma
        self.num_classes = num_classes

        if alpha is None:
            self.alpha = Variable(torch.ones(num_classes, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)

    def forward(self, inputs, targets):
        """
        前向传播
        
        Args:
            inputs: 模型输出的logits [N, C]
            targets: 真实标签 [N]
        
        Returns:
            focal_loss: 计算得到的focal loss
        """
        N = inputs.size(0)
        C = inputs.size(1)
        
        # 计算类别概率
        P = F.softmax(inputs, dim=1)
        
        # 创建one-hot编码
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        
        # 如果alpha在cuda上，将alpha移到和inputs相同的设备
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        
        # 获取对应类别的alpha值
        alpha = self.alpha[ids.data.view(-1)]
        
        # 获取正确类别的概率
        probs = (P * class_mask).sum(1).view(-1, 1)
        
        # 计算log概率
        log_p = probs.log()
        
        # 计算focal loss
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
            
        return loss


class ClassBalancedFocalLoss(nn.Module):
    """
    类别平衡的Focal Loss，基于有效样本数调整权重
    
    Args:
        beta (float): 重采样参数
        gamma (float): 聚焦参数
        samples_per_class (list): 每个类别的样本数
    """
    def __init__(self, beta=0.9999, gamma=2, samples_per_class=None):
        super(ClassBalancedFocalLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        
        if samples_per_class is not None:
            # 计算有效样本数
            effective_num = 1.0 - torch.pow(beta, torch.FloatTensor(samples_per_class))
            # 计算类别权重
            weights = (1.0 - beta) / effective_num
            self.weights = weights / weights.sum() * len(weights)
        else:
            self.weights = None
            
    def forward(self, inputs, targets):
        """前向传播"""
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算概率
        pt = torch.exp(-ce_loss)
        
        # 应用类别权重
        if self.weights is not None:
            if inputs.is_cuda and not self.weights.is_cuda:
                self.weights = self.weights.cuda()
            weight_t = self.weights[targets]
            ce_loss = weight_t * ce_loss
        
        # 计算focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失
    
    Args:
        smoothing (float): 平滑参数，通常取0.1
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        """前向传播"""
        log_prob = F.log_softmax(inputs, dim=-1)
        weight = inputs.new_ones(inputs.size()) * self.smoothing / (inputs.size(-1) - 1.)
        weight.scatter_(-1, targets.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss