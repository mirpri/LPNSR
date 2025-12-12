"""
基础损失函数

包含标准的L2损失（MSE Loss）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class L2Loss(nn.Module):
    """
    L2损失（均方误差损失）
    
    这是超分辨率任务中最常用的基础损失函数，用于衡量预测图像和真实图像之间的像素级差异。
    
    优点：
    - 计算简单高效
    - 优化稳定
    - 能够保证基本的重建质量
    
    缺点：
    - 倾向于产生过度平滑的结果
    - 对高频细节的恢复能力有限
    
    Args:
        reduction: 损失的归约方式，可选 'mean', 'sum', 'none'
        loss_weight: 损失权重，用于多损失加权
    
    Examples:
        >>> criterion = L2Loss(reduction='mean', loss_weight=1.0)
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss = criterion(pred, target)
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        loss_weight: float = 1.0
    ):
        super().__init__()
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction}")
        
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算L2损失
        
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
            weight: 可选的像素权重 [B, 1, H, W] 或 [B, C, H, W]
        
        Returns:
            loss: L2损失值
        """
        # 计算平方误差
        loss = F.mse_loss(pred, target, reduction='none')
        
        # 应用权重（如果提供）
        if weight is not None:
            loss = loss * weight
        
        # 归约
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        # 'none'时不进行归约
        
        return self.loss_weight * loss
    
    def __repr__(self):
        return f"{self.__class__.__name__}(reduction={self.reduction}, loss_weight={self.loss_weight})"


class CharbonnierLoss(nn.Module):
    """
    Charbonnier损失（L1损失的平滑版本）
    
    Charbonnier损失是L1损失的可微分近似，在超分辨率任务中也很常用。
    相比MSE损失，它对异常值更加鲁棒。
    
    公式: loss = sqrt((pred - target)^2 + eps^2)
    
    Args:
        eps: 平滑参数，避免梯度为0
        reduction: 损失的归约方式
        loss_weight: 损失权重
    
    Examples:
        >>> criterion = CharbonnierLoss(eps=1e-3)
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss = criterion(pred, target)
    """
    
    def __init__(
        self,
        eps: float = 1e-3,
        reduction: str = 'mean',
        loss_weight: float = 1.0
    ):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算Charbonnier损失
        
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
            weight: 可选的像素权重
        
        Returns:
            loss: Charbonnier损失值
        """
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        
        if weight is not None:
            loss = loss * weight
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return self.loss_weight * loss
    
    def __repr__(self):
        return f"{self.__class__.__name__}(eps={self.eps}, reduction={self.reduction}, loss_weight={self.loss_weight})"
