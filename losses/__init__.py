"""
超分辨率训练损失函数模块

包含以下损失函数：
1. L2Loss: 标准的L2损失（MSE）
2. FocalFrequencyLoss: 频域感知损失（ICCV 2021）
3. StatisticalFeatureLoss: 局部统计特征感知损失
"""

from .basic_loss import L2Loss
from .frequency_loss import FocalFrequencyLoss
from .statistical_loss import StatisticalFeatureLoss

__all__ = [
    'L2Loss',
    'FocalFrequencyLoss',
    'StatisticalFeatureLoss',
]
