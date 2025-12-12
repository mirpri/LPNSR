"""
频域感知损失（Focal Frequency Loss）

论文: Focal Frequency Loss for Image Reconstruction and Synthesis (ICCV 2021)
论文链接: https://arxiv.org/abs/2012.12821

核心思想：
- 在频域中计算损失，关注不同频率成分的重建质量
- 使用自适应权重机制，动态调整不同频率的重要性
- 特别关注难以重建的频率成分（类似Focal Loss的思想）

优势：
- 能够更好地恢复高频细节和纹理
- 自适应地关注难以重建的频率成分
- 与空域损失互补，提升整体重建质量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class FocalFrequencyLoss(nn.Module):
    """
    频域感知损失（Focal Frequency Loss）
    
    该损失函数在频域中计算预测图像和目标图像的差异，并使用自适应权重
    机制来关注难以重建的频率成分。
    
    主要特点：
    1. 频域计算：使用FFT将图像转换到频域
    2. 自适应权重：根据频率成分的重建难度动态调整权重
    3. 多尺度支持：可以在多个尺度上计算损失
    
    Args:
        loss_weight: 损失权重
        alpha: focal权重的指数，控制对难重建频率的关注程度（默认1.0）
        patch_factor: 将图像分块计算的因子（默认1，不分块）
        ave_spectrum: 是否对频谱取平均（默认False）
        log_matrix: 是否对频谱取对数（默认False）
        batch_matrix: 是否在batch维度上计算矩阵（默认False）
    
    Examples:
        >>> criterion = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0)
        >>> pred = torch.randn(4, 3, 256, 256)
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss = criterion(pred, target)
    """
    
    def __init__(
        self,
        loss_weight: float = 1.0,
        alpha: float = 1.0,
        patch_factor: int = 1,
        ave_spectrum: bool = False,
        log_matrix: bool = False,
        batch_matrix: bool = False
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix
    
    def tensor2freq(self, x: torch.Tensor) -> torch.Tensor:
        """
        将图像转换到频域
        
        Args:
            x: 输入图像 [B, C, H, W]
        
        Returns:
            freq: 频域表示 [B, C, H, W, 2] (实部和虚部)
        """
        # 使用2D FFT转换到频域
        freq = torch.fft.fft2(x, norm='ortho')
        # 将复数转换为实数表示 [real, imag]
        freq = torch.stack([freq.real, freq.imag], dim=-1)
        return freq
    
    def loss_formulation(
        self,
        recon_freq: torch.Tensor,
        real_freq: torch.Tensor,
        matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算频域损失
        
        Args:
            recon_freq: 重建图像的频域表示 [B, C, H, W, 2]
            real_freq: 真实图像的频域表示 [B, C, H, W, 2]
            matrix: 频率权重矩阵 [B, C, H, W] 或 None
        
        Returns:
            loss: 频域损失值
        """
        # 计算频域差异（欧氏距离）
        # recon_freq和real_freq的最后一维是[real, imag]
        diff = recon_freq - real_freq
        # 计算复数的模：sqrt(real^2 + imag^2)
        freq_distance = torch.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2 + 1e-8)
        
        # 应用focal权重
        if matrix is not None:
            # matrix是自适应权重，用于关注难以重建的频率
            weight_matrix = matrix ** self.alpha
            freq_distance = freq_distance * weight_matrix
        
        return torch.mean(freq_distance)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算Focal Frequency Loss
        
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
            matrix: 可选的频率权重矩阵
        
        Returns:
            loss: 频域感知损失值
        """
        # 确保输入形状一致
        assert pred.shape == target.shape, \
            f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
        
        # 如果需要分块处理
        if self.patch_factor > 1:
            return self._forward_with_patches(pred, target, matrix)
        
        # 转换到频域
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)
        
        # 如果没有提供权重矩阵，则计算自适应权重
        if matrix is None:
            # 计算频谱的幅度
            pred_amp = torch.sqrt(pred_freq[..., 0] ** 2 + pred_freq[..., 1] ** 2 + 1e-8)
            target_amp = torch.sqrt(target_freq[..., 0] ** 2 + target_freq[..., 1] ** 2 + 1e-8)
            
            # 计算自适应权重：频率成分差异越大，权重越高
            matrix = torch.abs(pred_amp - target_amp) / (target_amp + 1e-8)
            
            if self.ave_spectrum:
                # 在通道维度上取平均
                matrix = torch.mean(matrix, dim=1, keepdim=True)
            
            if self.log_matrix:
                # 对权重取对数，压缩动态范围
                matrix = torch.log(matrix + 1.0)
            
            if self.batch_matrix:
                # 在batch维度上取平均
                matrix = torch.mean(matrix, dim=0, keepdim=True)
        
        # 计算损失
        loss = self.loss_formulation(pred_freq, target_freq, matrix)
        
        return self.loss_weight * loss
    
    def _forward_with_patches(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        分块计算频域损失
        
        将图像分成多个patch，分别计算频域损失后取平均。
        这样可以更好地捕捉局部频率特征。
        
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
            matrix: 可选的频率权重矩阵
        
        Returns:
            loss: 平均频域损失
        """
        b, c, h, w = pred.shape
        patch_size_h = h // self.patch_factor
        patch_size_w = w // self.patch_factor
        
        total_loss = 0.0
        num_patches = 0
        
        for i in range(self.patch_factor):
            for j in range(self.patch_factor):
                # 提取patch
                h_start = i * patch_size_h
                h_end = (i + 1) * patch_size_h if i < self.patch_factor - 1 else h
                w_start = j * patch_size_w
                w_end = (j + 1) * patch_size_w if j < self.patch_factor - 1 else w
                
                pred_patch = pred[:, :, h_start:h_end, w_start:w_end]
                target_patch = target[:, :, h_start:h_end, w_start:w_end]
                
                # 转换到频域
                pred_freq = self.tensor2freq(pred_patch)
                target_freq = self.tensor2freq(target_patch)
                
                # 计算patch的权重矩阵
                if matrix is None:
                    pred_amp = torch.sqrt(pred_freq[..., 0] ** 2 + pred_freq[..., 1] ** 2 + 1e-8)
                    target_amp = torch.sqrt(target_freq[..., 0] ** 2 + target_freq[..., 1] ** 2 + 1e-8)
                    patch_matrix = torch.abs(pred_amp - target_amp) / (target_amp + 1e-8)
                    
                    if self.ave_spectrum:
                        patch_matrix = torch.mean(patch_matrix, dim=1, keepdim=True)
                    if self.log_matrix:
                        patch_matrix = torch.log(patch_matrix + 1.0)
                else:
                    patch_matrix = matrix[:, :, h_start:h_end, w_start:w_end]
                
                # 计算patch损失
                patch_loss = self.loss_formulation(pred_freq, target_freq, patch_matrix)
                total_loss += patch_loss
                num_patches += 1
        
        # 返回平均损失
        return self.loss_weight * (total_loss / num_patches)
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"loss_weight={self.loss_weight}, "
                f"alpha={self.alpha}, "
                f"patch_factor={self.patch_factor})")
