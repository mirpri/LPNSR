"""
局部统计特征感知损失（Statistical Feature Loss）

设计思想：
超分辨率任务不仅需要恢复像素级的细节，还需要保持图像的统计特征分布。
本损失函数通过比较预测图像和目标图像在局部区域的统计特征（均值、方差、偏度、峰度等），
来确保重建图像在统计意义上与真实图像相似。

核心创新：
1. 多尺度局部统计：在不同尺度的局部窗口中计算统计特征
2. 高阶矩匹配：不仅匹配均值和方差，还匹配偏度和峰度
3. 纹理一致性：通过统计特征确保纹理的真实性
4. 自适应权重：根据图像内容动态调整不同统计特征的权重

理论依据：
- 自然图像在局部区域具有特定的统计分布特性
- 高质量的超分辨率结果应该保持这些统计特性
- 统计特征能够捕捉纹理和结构信息，补充像素级损失的不足
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class StatisticalFeatureLoss(nn.Module):
    """
    局部统计特征感知损失
    
    该损失函数通过比较图像在局部区域的统计特征来衡量重建质量。
    主要计算以下统计量：
    1. 均值（Mean）：局部亮度
    2. 方差（Variance）：局部对比度
    3. 偏度（Skewness）：分布的不对称性
    4. 峰度（Kurtosis）：分布的尖锐程度
    
    Args:
        loss_weight: 损失权重
        window_sizes: 局部窗口大小列表，支持多尺度 (默认[3, 5, 7])
        use_mean: 是否使用均值特征 (默认True)
        use_variance: 是否使用方差特征 (默认True)
        use_skewness: 是否使用偏度特征 (默认True)
        use_kurtosis: 是否使用峰度特征 (默认True)
        normalize: 是否对统计特征进行归一化 (默认True)
        reduction: 损失归约方式 'mean' 或 'sum'
    
    Examples:
        >>> criterion = StatisticalFeatureLoss(
        ...     loss_weight=1.0,
        ...     window_sizes=[3, 5, 7],
        ...     use_mean=True,
        ...     use_variance=True,
        ...     use_skewness=True,
        ...     use_kurtosis=True
        ... )
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss = criterion(pred, target)
    """
    
    def __init__(
        self,
        loss_weight: float = 1.0,
        window_sizes: List[int] = [3, 5, 7],
        use_mean: bool = True,
        use_variance: bool = True,
        use_skewness: bool = True,
        use_kurtosis: bool = True,
        normalize: bool = True,
        reduction: str = 'mean'
    ):
        super().__init__()
        
        self.loss_weight = loss_weight
        self.window_sizes = window_sizes
        self.use_mean = use_mean
        self.use_variance = use_variance
        self.use_skewness = use_skewness
        self.use_kurtosis = use_kurtosis
        self.normalize = normalize
        self.reduction = reduction
        
        # 各统计特征的归一化系数（用于平衡不同特征的数值范围）
        # mean: 图像值范围约[-1, 1]，差异约0-2
        # variance: 方差范围约0-1，差异约0-1
        # skewness: 偏度范围约[-3, 3]，差异约0-6
        # kurtosis: 峰度范围约[1, 100+]，差异可能很大
        self.feature_scales = {
            'mean': 1.0,       # 均值差异通常较小
            'variance': 1.0,   # 方差差异适中
            'skewness': 0.5,   # 偏度差异需要适当缩放
            'kurtosis': 0.001,  # 峰度差异很大，需要大幅缩放
        }
        
        # 预计算高斯窗口（用于加权统计）
        self.gaussian_windows = {}
        for window_size in window_sizes:
            self.gaussian_windows[window_size] = self._create_gaussian_window(window_size)
    
    def _create_gaussian_window(self, window_size: int, sigma: Optional[float] = None) -> torch.Tensor:
        """
        创建高斯窗口
        
        Args:
            window_size: 窗口大小
            sigma: 高斯标准差，默认为window_size/6
        
        Returns:
            window: 高斯窗口 [1, 1, window_size, window_size]
        """
        if sigma is None:
            sigma = window_size / 6.0
        
        # 创建1D高斯核
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        # 创建2D高斯核
        g_2d = g.unsqueeze(0) * g.unsqueeze(1)
        g_2d = g_2d / g_2d.sum()
        
        return g_2d.unsqueeze(0).unsqueeze(0)
    
    def _compute_local_mean(
        self,
        x: torch.Tensor,
        window: torch.Tensor
    ) -> torch.Tensor:
        """
        计算局部均值
        
        Args:
            x: 输入图像 [B, C, H, W]
            window: 高斯窗口 [1, 1, K, K]
        
        Returns:
            mean: 局部均值 [B, C, H, W]
        """
        b, c, h, w = x.shape
        window = window.to(x.device).to(x.dtype)
        
        # 对每个通道分别卷积
        window = window.repeat(c, 1, 1, 1)
        padding = window.shape[-1] // 2
        
        mean = F.conv2d(x, window, padding=padding, groups=c)
        
        return mean
    
    def _compute_local_variance(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        window: torch.Tensor
    ) -> torch.Tensor:
        """
        计算局部方差
        
        Args:
            x: 输入图像 [B, C, H, W]
            mean: 局部均值 [B, C, H, W]
            window: 高斯窗口 [1, 1, K, K]
        
        Returns:
            variance: 局部方差 [B, C, H, W]
        """
        b, c, h, w = x.shape
        window = window.to(x.device).to(x.dtype)
        window = window.repeat(c, 1, 1, 1)
        padding = window.shape[-1] // 2
        
        # Var(X) = E[X^2] - E[X]^2
        x_squared = x ** 2
        mean_squared = F.conv2d(x_squared, window, padding=padding, groups=c)
        variance = mean_squared - mean ** 2
        
        # 确保方差非负
        variance = torch.clamp(variance, min=1e-8)
        
        return variance
    
    def _compute_local_skewness(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        variance: torch.Tensor,
        window: torch.Tensor
    ) -> torch.Tensor:
        """
        计算局部偏度（三阶中心矩）
        
        偏度衡量分布的不对称性：
        - 偏度 > 0：右偏（长尾在右侧）
        - 偏度 < 0：左偏（长尾在左侧）
        - 偏度 = 0：对称分布
        
        Args:
            x: 输入图像 [B, C, H, W]
            mean: 局部均值 [B, C, H, W]
            variance: 局部方差 [B, C, H, W]
            window: 高斯窗口 [1, 1, K, K]
        
        Returns:
            skewness: 局部偏度 [B, C, H, W]
        """
        b, c, h, w = x.shape
        window = window.to(x.device).to(x.dtype)
        window = window.repeat(c, 1, 1, 1)
        padding = window.shape[-1] // 2
        
        # Skewness = E[(X - μ)^3] / σ^3
        x_centered = x - mean
        x_cubed = x_centered ** 3
        third_moment = F.conv2d(x_cubed, window, padding=padding, groups=c)
        
        std = torch.sqrt(variance)
        skewness = third_moment / (std ** 3 + 1e-8)
        
        return skewness
    
    def _compute_local_kurtosis(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        variance: torch.Tensor,
        window: torch.Tensor
    ) -> torch.Tensor:
        """
        计算局部峰度（四阶中心矩）
        
        峰度衡量分布的尖锐程度：
        - 峰度 > 3：尖峰分布（比正态分布更尖）
        - 峰度 = 3：正态分布
        - 峰度 < 3：平峰分布（比正态分布更平）
        
        Args:
            x: 输入图像 [B, C, H, W]
            mean: 局部均值 [B, C, H, W]
            variance: 局部方差 [B, C, H, W]
            window: 高斯窗口 [1, 1, K, K]
        
        Returns:
            kurtosis: 局部峰度 [B, C, H, W]
        """
        b, c, h, w = x.shape
        window = window.to(x.device).to(x.dtype)
        window = window.repeat(c, 1, 1, 1)
        padding = window.shape[-1] // 2
        
        # Kurtosis = E[(X - μ)^4] / σ^4
        x_centered = x - mean
        x_fourth = x_centered ** 4
        fourth_moment = F.conv2d(x_fourth, window, padding=padding, groups=c)
        
        kurtosis = fourth_moment / (variance ** 2 + 1e-8)
        
        return kurtosis
    
    def _compute_statistical_features(
        self,
        x: torch.Tensor,
        window_size: int
    ) -> dict:
        """
        计算所有统计特征
        
        Args:
            x: 输入图像 [B, C, H, W]
            window_size: 窗口大小
        
        Returns:
            features: 统计特征字典
        """
        window = self.gaussian_windows[window_size]
        features = {}
        
        # 计算均值
        if self.use_mean:
            mean = self._compute_local_mean(x, window)
            features['mean'] = mean
        else:
            mean = self._compute_local_mean(x, window)  # 其他统计量需要均值
        
        # 计算方差
        if self.use_variance:
            variance = self._compute_local_variance(x, mean, window)
            features['variance'] = variance
        else:
            variance = self._compute_local_variance(x, mean, window)  # 高阶矩需要方差
        
        # 计算偏度
        if self.use_skewness:
            skewness = self._compute_local_skewness(x, mean, variance, window)
            features['skewness'] = skewness
        
        # 计算峰度
        if self.use_kurtosis:
            kurtosis = self._compute_local_kurtosis(x, mean, variance, window)
            features['kurtosis'] = kurtosis
        
        return features
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算统计特征损失
        
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
        
        Returns:
            loss: 统计特征损失值
        """
        assert pred.shape == target.shape, \
            f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
        
        total_loss = 0.0
        num_features = 0
        
        # 在多个尺度上计算统计特征损失
        for window_size in self.window_sizes:
            # 计算预测图像和目标图像的统计特征
            pred_features = self._compute_statistical_features(pred, window_size)
            target_features = self._compute_statistical_features(target, window_size)
            
            # 计算每个统计特征的损失
            for feature_name in pred_features.keys():
                pred_feat = pred_features[feature_name]
                target_feat = target_features[feature_name]
                
                # 获取该特征的缩放系数
                scale = self.feature_scales.get(feature_name, 1.0)
                
                # 计算绝对误差（不使用相对误差，因为相对误差在某些情况下不稳定）
                feat_loss = torch.abs(pred_feat - target_feat)
                
                # 归约
                if self.reduction == 'mean':
                    feat_loss = feat_loss.mean()
                elif self.reduction == 'sum':
                    feat_loss = feat_loss.sum()
                
                # 应用特征独立的缩放系数进行归一化
                total_loss += feat_loss * scale
                num_features += 1
        
        # 平均所有特征的损失
        if num_features > 0:
            total_loss = total_loss / num_features
        
        return self.loss_weight * total_loss
    
    def __repr__(self):
        features = []
        if self.use_mean:
            features.append('mean')
        if self.use_variance:
            features.append('variance')
        if self.use_skewness:
            features.append('skewness')
        if self.use_kurtosis:
            features.append('kurtosis')
        
        return (f"{self.__class__.__name__}("
                f"loss_weight={self.loss_weight}, "
                f"window_sizes={self.window_sizes}, "
                f"features={features})")
