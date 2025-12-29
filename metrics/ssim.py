"""
SSIM (Structural Similarity Index) 结构相似性指数

全参考图像质量评估指标，综合考虑图像的亮度、对比度和结构信息。
SSIM值范围为[-1, 1]，值越大表示图像越相似。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple
from scipy.ndimage import uniform_filter
from .metric_utils import reorder_image, to_y_channel


def _ssim_single_channel(
    img1: np.ndarray,
    img2: np.ndarray,
    k1: float = 0.01,
    k2: float = 0.03,
    win_size: int = 11,
    data_range: float = 255.0
) -> float:
    """计算单通道图像的SSIM值"""
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    
    mu1 = uniform_filter(img1, size=win_size, mode='reflect')
    mu2 = uniform_filter(img2, size=win_size, mode='reflect')
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = uniform_filter(img1 ** 2, size=win_size, mode='reflect') - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, size=win_size, mode='reflect') - mu2_sq
    sigma12 = uniform_filter(img1 * img2, size=win_size, mode='reflect') - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return float(ssim_map.mean())


def calculate_ssim(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor],
    crop_border: int = 0,
    input_order: str = 'HWC',
    test_y_channel: bool = False,
    data_range: float = 255.0,
    win_size: int = 11
) -> float:
    """
    计算两张图像之间的SSIM值
    
    Args:
        img1: 第一张图像
        img2: 第二张图像
        crop_border: 计算前裁剪的边界像素数
        input_order: 输入图像的维度顺序
        test_y_channel: 是否只在Y通道上测试
        data_range: 图像数据范围
        win_size: 滑动窗口大小
        
    Returns:
        SSIM值
    """
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    
    if img1.ndim == 2:
        ssim_val = _ssim_single_channel(img1, img2, win_size=win_size, data_range=data_range)
    elif img1.ndim == 3:
        ssim_vals = []
        for i in range(img1.shape[2]):
            ssim_val = _ssim_single_channel(
                img1[..., i], img2[..., i], win_size=win_size, data_range=data_range
            )
            ssim_vals.append(ssim_val)
        ssim_val = np.mean(ssim_vals)
    else:
        raise ValueError(f'Wrong image dimension: {img1.ndim}')
    
    return float(ssim_val)


class SSIM(nn.Module):
    """SSIM计算类"""
    
    def __init__(
        self,
        crop_border: int = 0,
        input_order: str = 'HWC',
        test_y_channel: bool = False,
        data_range: float = 255.0,
        window_size: int = 11
    ):
        super().__init__()
        self.crop_border = crop_border
        self.input_order = input_order
        self.test_y_channel = test_y_channel
        self.data_range = data_range
        self.window_size = window_size
    
    def forward(
        self,
        img1: Union[np.ndarray, torch.Tensor],
        img2: Union[np.ndarray, torch.Tensor]
    ) -> float:
        return calculate_ssim(
            img1, img2,
            crop_border=self.crop_border,
            input_order=self.input_order,
            test_y_channel=self.test_y_channel,
            data_range=self.data_range,
            win_size=self.window_size
        )
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'crop_border={self.crop_border}, '
                f'test_y_channel={self.test_y_channel})')
