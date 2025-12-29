"""
图像质量评估的通用工具函数
"""

import numpy as np
import torch
import cv2
from typing import Union, Tuple, Optional


def img2tensor(
    imgs: Union[np.ndarray, list],
    bgr2rgb: bool = True,
    float32: bool = True
) -> Union[torch.Tensor, list]:
    """
    将numpy图像转换为PyTorch张量
    
    Args:
        imgs: 输入图像(numpy数组)或图像列表
        bgr2rgb: 是否将BGR转换为RGB
        float32: 是否转换为float32类型
        
    Returns:
        PyTorch张量或张量列表
    """
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(
    tensor: torch.Tensor,
    rgb2bgr: bool = True,
    out_type: type = np.uint8,
    min_max: Tuple[float, float] = (0, 1)
) -> Union[np.ndarray, list]:
    """
    将PyTorch张量转换为numpy图像
    
    Args:
        tensor: 输入张量，形状为(B, C, H, W)或(C, H, W)
        rgb2bgr: 是否将RGB转换为BGR
        out_type: 输出类型
        min_max: 输入张量的最小最大值范围
        
    Returns:
        numpy图像或图像列表
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
        
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(0, 2, 3, 1)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
            
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
        
    if len(result) == 1:
        result = result[0]
    return result


def rgb2ycbcr(img: np.ndarray, y_only: bool = False) -> np.ndarray:
    """
    将RGB图像转换为YCbCr颜色空间
    
    Args:
        img: 输入RGB图像，值范围[0, 255]
        y_only: 是否只返回Y通道
        
    Returns:
        YCbCr图像或Y通道
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    
    if y_only:
        out_img = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        out_img = np.zeros(img.shape, dtype=np.float32)
        out_img[:, :, 0] = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
        out_img[:, :, 1] = np.dot(img, [-37.797, -74.203, 112.0]) / 255.0 + 128.0
        out_img[:, :, 2] = np.dot(img, [112.0, -93.786, -18.214]) / 255.0 + 128.0
        
    if img_type != np.float32:
        out_img = out_img.astype(img_type)
    return out_img


def bgr2ycbcr(img: np.ndarray, y_only: bool = False) -> np.ndarray:
    """
    将BGR图像转换为YCbCr颜色空间
    
    Args:
        img: 输入BGR图像，值范围[0, 255]
        y_only: 是否只返回Y通道
        
    Returns:
        YCbCr图像或Y通道
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        out_img = np.zeros(img.shape, dtype=np.float32)
        out_img[:, :, 0] = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
        out_img[:, :, 1] = np.dot(img, [112.0, -74.203, -37.797]) / 255.0 + 128.0
        out_img[:, :, 2] = np.dot(img, [-18.214, -93.786, 112.0]) / 255.0 + 128.0
        
    if img_type != np.float32:
        out_img = out_img.astype(img_type)
    return out_img


def to_y_channel(img: np.ndarray) -> np.ndarray:
    """
    将图像转换为Y通道 (亮度通道)
    
    Args:
        img: 输入图像，BGR格式，值范围[0, 255]
        
    Returns:
        Y通道图像
    """
    img = img.astype(np.float32) / 255.0
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img * 255.0, y_only=True)
        img = img[..., None]
    return img


def reorder_image(
    img: np.ndarray,
    input_order: str = 'HWC'
) -> np.ndarray:
    """
    重新排列图像维度到HWC格式
    
    Args:
        img: 输入图像
        input_order: 输入图像的维度顺序，'HWC'或'CHW'
        
    Returns:
        HWC格式的图像
    """
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported orders are "HWC" and "CHW"')
        
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img
