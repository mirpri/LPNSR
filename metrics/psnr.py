"""
PSNR (Peak Signal-to-Noise Ratio) 峰值信噪比

全参考图像质量评估指标，用于衡量图像重建质量。
PSNR值越高表示图像质量越好，通常用dB表示。
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Union
from .metric_utils import reorder_image, to_y_channel


def calculate_psnr(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor],
    crop_border: int = 0,
    input_order: str = 'HWC',
    test_y_channel: bool = False,
    data_range: float = 255.0
) -> float:
    """
    计算两张图像之间的PSNR值
    
    Args:
        img1: 第一张图像 (通常是重建图像/SR图像)
        img2: 第二张图像 (通常是参考图像/GT图像)
        crop_border: 计算前裁剪的边界像素数
        input_order: 输入图像的维度顺序，'HWC'或'CHW'
        test_y_channel: 是否只在Y通道上测试 (用于彩色图像)
        data_range: 图像数据范围，默认255
        
    Returns:
        PSNR值 (dB)
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
    
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 10.0 * np.log10((data_range ** 2) / mse)
    
    return float(psnr)


class PSNR(nn.Module):
    """PSNR计算类"""
    
    def __init__(
        self,
        crop_border: int = 0,
        input_order: str = 'HWC',
        test_y_channel: bool = False,
        data_range: float = 255.0
    ):
        super().__init__()
        self.crop_border = crop_border
        self.input_order = input_order
        self.test_y_channel = test_y_channel
        self.data_range = data_range
    
    def forward(
        self,
        img1: Union[np.ndarray, torch.Tensor],
        img2: Union[np.ndarray, torch.Tensor]
    ) -> float:
        return calculate_psnr(
            img1, img2,
            crop_border=self.crop_border,
            input_order=self.input_order,
            test_y_channel=self.test_y_channel,
            data_range=self.data_range
        )
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'crop_border={self.crop_border}, '
                f'test_y_channel={self.test_y_channel})')
