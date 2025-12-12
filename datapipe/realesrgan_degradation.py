"""
RealESRGAN 图像退化管道实现
根据配置文件参数实现图像退化过程，包括模糊、噪声、调整大小和JPEG压缩等操作
"""

import cv2
import math
import numpy as np
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image



class RealESRGANDegradation:
    """
    RealESRGAN退化管道类
    实现图像的两阶段退化过程，用于生成低质量图像
    """
    
    def __init__(self, config_path: str):
        """
        初始化退化管道
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.opts = self.config['opts']
        self.degradation = self.config['degradation']
        
        # 初始化模糊核范围
        self.kernel_range1 = [x for x in range(3, self.opts['blur_kernel_size'], 2)]
        self.kernel_range2 = [x for x in range(3, self.opts['blur_kernel_size2'], 2)]
        
        # 创建脉冲张量（用于无模糊效果）
        self.pulse_tensor = torch.zeros(
            self.opts['blur_kernel_size2'], 
            self.opts['blur_kernel_size2']
        ).float()
        self.pulse_tensor[
            self.opts['blur_kernel_size2']//2, 
            self.opts['blur_kernel_size2']//2
        ] = 1
        
        # JPEG压缩器（需要时初始化）
        self.jpeger = None
    
    def _load_config(self, config_path: str) -> Dict:
        """加载YAML配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _sigma_matrix2(self, sig_x: float, sig_y: float, theta: float) -> np.ndarray:
        """
        计算旋转的sigma矩阵（二维矩阵）
        
        Args:
            sig_x: x方向的标准差
            sig_y: y方向的标准差
            theta: 旋转角度（弧度）
        
        Returns:
            旋转后的sigma矩阵
        """
        d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
        u_matrix = np.array([
            [np.cos(theta), -np.sin(theta)], 
            [np.sin(theta), np.cos(theta)]
        ])
        return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))
    
    def _mesh_grid(self, kernel_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """生成网格坐标"""
        ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        return xx, yy
    
    def _pdf2(self, sigma_matrix: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        """计算二维概率密度函数"""
        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.exp(-0.5 * (inverse_sigma[0, 0] * (grid_x**2) + 
                                 inverse_sigma[1, 1] * (grid_y**2) + 
                                 2 * inverse_sigma[0, 1] * grid_x * grid_y))
        return kernel
    
    def _bivariate_gaussian(
        self, 
        kernel_size: int, 
        sig_x: float, 
        sig_y: float, 
        theta: float, 
        grid: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        isotropic: bool = True
    ) -> np.ndarray:
        """
        生成二元高斯核
        
        Args:
            kernel_size: 核大小
            sig_x: x方向标准差
            sig_y: y方向标准差
            theta: 旋转角度
            grid: 网格坐标
            isotropic: 是否为各向同性
        
        Returns:
            归一化的高斯核
        """
        if grid is None:
            grid = self._mesh_grid(kernel_size)
        
        if isotropic:
            sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
        else:
            sigma_matrix = self._sigma_matrix2(sig_x, sig_y, theta)
        
        kernel = self._pdf2(sigma_matrix, grid[0], grid[1])
        kernel = kernel / np.sum(kernel)
        return kernel
    
    def _bivariate_generalized_gaussian(
        self,
        kernel_size: int,
        sig_x: float,
        sig_y: float,
        theta: float,
        beta: float,
        grid: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        isotropic: bool = True
    ) -> np.ndarray:
        """生成广义高斯核"""
        if grid is None:
            grid = self._mesh_grid(kernel_size)
        
        if isotropic:
            sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
        else:
            sigma_matrix = self._sigma_matrix2(sig_x, sig_y, theta)
        
        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.exp(-0.5 * np.power(
            inverse_sigma[0, 0] * (grid[0]**2) + 
            inverse_sigma[1, 1] * (grid[1]**2) + 
            2 * inverse_sigma[0, 1] * grid[0] * grid[1], 
            beta
        ))
        kernel = kernel / np.sum(kernel)
        return kernel
    
    def _bivariate_plateau(
        self,
        kernel_size: int,
        sig_x: float,
        sig_y: float,
        theta: float,
        beta: float,
        grid: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        isotropic: bool = True
    ) -> np.ndarray:
        """生成平台核"""
        if grid is None:
            grid = self._mesh_grid(kernel_size)
        
        if isotropic:
            sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
        else:
            sigma_matrix = self._sigma_matrix2(sig_x, sig_y, theta)
        
        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.reciprocal(
            np.power(
                inverse_sigma[0, 0] * (grid[0]**2) + 
                inverse_sigma[1, 1] * (grid[1]**2) + 
                2 * inverse_sigma[0, 1] * grid[0] * grid[1], 
                beta
            ) + 1
        )
        kernel = kernel / np.sum(kernel)
        return kernel
    
    def _random_mixed_kernels(
        self,
        kernel_list: List[str],
        kernel_prob: List[float],
        kernel_size: int,
        sigma_x_range: List[float],
        sigma_y_range: List[float],
        rotation_range: List[float],
        betag_range: List[float],
        betap_range: List[float],
        noise_range: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        随机生成混合模糊核
        
        Args:
            kernel_list: 核类型列表
            kernel_prob: 各核类型的概率
            kernel_size: 核大小
            sigma_x_range: x方向sigma范围
            sigma_y_range: y方向sigma范围
            rotation_range: 旋转角度范围
            betag_range: 广义高斯beta范围
            betap_range: 平台核beta范围
            noise_range: 噪声范围
        
        Returns:
            生成的模糊核
        """
        kernel_type = random.choices(kernel_list, kernel_prob)[0]
        
        # 生成随机参数
        sigma_x = random.uniform(sigma_x_range[0], sigma_x_range[1])
        sigma_y = random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = random.uniform(rotation_range[0], rotation_range[1])
        
        # 根据核类型生成
        if kernel_type == 'iso':
            kernel = self._bivariate_gaussian(
                kernel_size, sigma_x, sigma_x, 0, isotropic=True
            )
        elif kernel_type == 'aniso':
            kernel = self._bivariate_gaussian(
                kernel_size, sigma_x, sigma_y, rotation, isotropic=False
            )
        elif kernel_type == 'generalized_iso':
            beta = random.uniform(betag_range[0], betag_range[1])
            kernel = self._bivariate_generalized_gaussian(
                kernel_size, sigma_x, sigma_x, 0, beta, isotropic=True
            )
        elif kernel_type == 'generalized_aniso':
            beta = random.uniform(betag_range[0], betag_range[1])
            kernel = self._bivariate_generalized_gaussian(
                kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=False
            )
        elif kernel_type == 'plateau_iso':
            beta = random.uniform(betap_range[0], betap_range[1])
            kernel = self._bivariate_plateau(
                kernel_size, sigma_x, sigma_x, 0, beta, isotropic=True
            )
        elif kernel_type == 'plateau_aniso':
            beta = random.uniform(betap_range[0], betap_range[1])
            kernel = self._bivariate_plateau(
                kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=False
            )
        else:
            raise ValueError(f'Unknown kernel type: {kernel_type}')
        
        # 添加噪声（如果指定）
        if noise_range is not None:
            noise_level = random.uniform(noise_range[0], noise_range[1])
            noise = np.random.randn(*kernel.shape) * noise_level
            kernel = kernel + noise
            kernel = np.clip(kernel, 0, None)
        
        # 归一化
        kernel = kernel / np.sum(kernel)
        return kernel
    
    def _circular_lowpass_kernel(
        self, 
        cutoff: float, 
        kernel_size: int, 
        pad_to: int = 0
    ) -> np.ndarray:
        """
        生成圆形低通滤波器（sinc核）
        
        Args:
            cutoff: 截止频率
            kernel_size: 核大小
            pad_to: 填充到的大小
        
        Returns:
            sinc滤波核
        """
        assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
        
        kernel = np.fromfunction(
            lambda x, y: cutoff * np.sinc(
                cutoff * np.sqrt(
                    (x - (kernel_size - 1) / 2)**2 + 
                    (y - (kernel_size - 1) / 2)**2
                ) / np.pi
            ),
            [kernel_size, kernel_size]
        )
        kernel = kernel / np.sum(kernel)
        
        if pad_to > kernel_size:
            pad_size = (pad_to - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        
        return kernel
    
    def generate_kernels(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成退化所需的三个核：第一次模糊核、第二次模糊核、最终sinc核
        
        Returns:
            (kernel1, kernel2, sinc_kernel)
        """
        # 生成第一次模糊核
        kernel_size = random.choice(self.kernel_range1)
        if np.random.uniform() < self.opts['sinc_prob']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel1 = self._circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel1 = self._random_mixed_kernels(
                self.opts['kernel_list'],
                self.opts['kernel_prob'],
                kernel_size,
                self.opts['blur_sigma'],
                self.opts['blur_sigma'],
                [-math.pi, math.pi],
                self.opts['betag_range'],
                self.opts['betap_range'],
                noise_range=None
            )
        
        # 填充第一次核
        pad_size = (self.opts['blur_kernel_size'] - kernel_size) // 2
        kernel1 = np.pad(kernel1, ((pad_size, pad_size), (pad_size, pad_size)))
        
        # 生成第二次模糊核
        kernel_size = random.choice(self.kernel_range2)
        if np.random.uniform() < self.opts['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = self._circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = self._random_mixed_kernels(
                self.opts['kernel_list2'],
                self.opts['kernel_prob2'],
                kernel_size,
                self.opts['blur_sigma2'],
                self.opts['blur_sigma2'],
                [-math.pi, math.pi],
                self.opts['betag_range2'],
                self.opts['betap_range2'],
                noise_range=None
            )
        
        # 填充第二次核
        pad_size = (self.opts['blur_kernel_size2'] - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
        
        # 生成最终sinc核
        if np.random.uniform() < self.opts['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range2)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = self._circular_lowpass_kernel(
                omega_c, kernel_size, pad_to=self.opts['blur_kernel_size2']
            )
        else:
            sinc_kernel = self.pulse_tensor.numpy()
        
        return kernel1, kernel2, sinc_kernel
    
    def _filter2D(self, img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        使用给定核对图像进行2D卷积滤波
        
        Args:
            img: 输入图像张量 [B, C, H, W]
            kernel: 卷积核 [H, W]
        
        Returns:
            滤波后的图像
        """
        k = kernel.size(-1)
        b, c, h, w = img.size()
        
        # 扩展核以匹配通道数
        kernel = kernel.view(1, 1, k, k).repeat(c, 1, 1, 1)
        
        # 计算填充
        pad = (k - 1) // 2
        
        # 应用卷积
        return F.conv2d(img, kernel, padding=pad, groups=c)
    
    def _random_add_gaussian_noise_pt(
        self,
        img: torch.Tensor,
        sigma_range: List[float],
        gray_prob: float = 0,
        clip: bool = True,
        rounds: bool = False
    ) -> torch.Tensor:
        """
        添加随机高斯噪声
        
        Args:
            img: 输入图像张量
            sigma_range: 噪声强度范围
            gray_prob: 灰度噪声概率
            clip: 是否裁剪到[0,1]
            rounds: 是否四舍五入
        
        Returns:
            添加噪声后的图像
        """
        sigma = random.uniform(sigma_range[0], sigma_range[1]) / 255.0
        
        # 生成噪声
        if random.random() < gray_prob:
            # 灰度噪声
            noise = torch.randn(img.size(0), 1, img.size(2), img.size(3), 
                              dtype=img.dtype, device=img.device) * sigma
            noise = noise.repeat(1, img.size(1), 1, 1)
        else:
            # 彩色噪声
            noise = torch.randn_like(img) * sigma
        
        out = img + noise
        
        if clip and rounds:
            out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        elif clip:
            out = torch.clamp(out, 0, 1)
        elif rounds:
            out = (out * 255.0).round() / 255.
        
        return out
    
    def _random_add_poisson_noise_pt(
        self,
        img: torch.Tensor,
        scale_range: List[float],
        gray_prob: float = 0,
        clip: bool = True,
        rounds: bool = False
    ) -> torch.Tensor:
        """
        添加随机泊松噪声
        
        Args:
            img: 输入图像张量
            scale_range: 缩放范围
            gray_prob: 灰度噪声概率
            clip: 是否裁剪
            rounds: 是否四舍五入
        
        Returns:
            添加噪声后的图像
        """
        scale = random.uniform(scale_range[0], scale_range[1])
        
        # 确保输入图像值非负（泊松分布要求rate >= 0）
        img = torch.clamp(img, min=0)
        
        if random.random() < gray_prob:
            # 灰度噪声
            img_gray = img.mean(dim=1, keepdim=True)
            img_gray = torch.poisson(img_gray * 255.0 * scale) / scale / 255.0
            noise = img_gray.repeat(1, img.size(1), 1, 1) - img
            out = img + noise
        else:
            # 彩色噪声
            out = torch.poisson(img * 255.0 * scale) / scale / 255.0
        
        if clip and rounds:
            out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        elif clip:
            out = torch.clamp(out, 0, 1)
        elif rounds:
            out = (out * 255.0).round() / 255.
        
        return out
    
    def _jpeg_compress(self, img: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        """
        模拟JPEG压缩
        
        Args:
            img: 输入图像张量
            quality: JPEG质量参数
        
        Returns:
            压缩后的图像
        """
        # 这里使用简化的JPEG压缩模拟
        # 实际应用中可以使用DiffJPEG库
        if self.jpeger is None:
            try:
                from basicsr.utils import DiffJPEG
                self.jpeger = DiffJPEG(differentiable=False)
            except ImportError:
                # 如果没有DiffJPEG，使用简单的量化模拟
                return img
        
        return self.jpeger(img, quality=quality)
    
    def degrade(
        self, 
        img_gt: torch.Tensor,
        kernel1: Optional[torch.Tensor] = None,
        kernel2: Optional[torch.Tensor] = None,
        sinc_kernel: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        对高质量图像应用退化管道
        
        Args:
            img_gt: 高质量图像张量 [B, C, H, W]
            kernel1: 第一次模糊核（可选，不提供则自动生成）
            kernel2: 第二次模糊核（可选）
            sinc_kernel: 最终sinc核（可选）
        
        Returns:
            包含'lq'（低质量图像）和'gt'（高质量图像）的字典
        """
        # 如果没有提供核，则生成
        if kernel1 is None or kernel2 is None or sinc_kernel is None:
            k1, k2, sk = self.generate_kernels()
            kernel1 = torch.FloatTensor(k1) if kernel1 is None else kernel1
            kernel2 = torch.FloatTensor(k2) if kernel2 is None else kernel2
            sinc_kernel = torch.FloatTensor(sk) if sinc_kernel is None else sinc_kernel
        
        # 确保在同一设备上
        device = img_gt.device
        kernel1 = kernel1.to(device)
        kernel2 = kernel2.to(device)
        sinc_kernel = sinc_kernel.to(device)
        
        ori_h, ori_w = img_gt.size()[2:4]
        sf = self.degradation['sf']
        
        # ----------------------- 第一次退化过程 ----------------------- #
        # 模糊
        out = self._filter2D(img_gt, kernel1)
        
        # 随机调整大小
        updown_type = random.choices(
            ['up', 'down', 'keep'],
            self.degradation['resize_prob']
        )[0]
        
        if updown_type == 'up':
            scale = random.uniform(1, self.degradation['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.degradation['resize_range'][0], 1)
        else:
            scale = 1
        
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        
        # 添加噪声
        gray_noise_prob = self.degradation['gray_noise_prob']
        if random.random() < self.degradation['gaussian_noise_prob']:
            out = self._random_add_gaussian_noise_pt(
                out,
                sigma_range=self.degradation['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob
            )
        else:
            out = self._random_add_poisson_noise_pt(
                out,
                scale_range=self.degradation['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False
            )
        
        # JPEG压缩
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degradation['jpeg_range'])
        out = torch.clamp(out, 0, 1)
        out = self._jpeg_compress(out, quality=jpeg_p)
        
        # ----------------------- 第二次退化过程 ----------------------- #
        if random.random() < self.degradation['second_order_prob']:
            # 模糊
            if random.random() < self.degradation['second_blur_prob']:
                out = self._filter2D(out, kernel2)
            
            # 随机调整大小
            updown_type = random.choices(
                ['up', 'down', 'keep'],
                self.degradation['resize_prob2']
            )[0]
            
            if updown_type == 'up':
                scale = random.uniform(1, self.degradation['resize_range2'][1])
            elif updown_type == 'down':
                scale = random.uniform(self.degradation['resize_range2'][0], 1)
            else:
                scale = 1
            
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out,
                size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
                mode=mode
            )
            
            # 添加噪声
            gray_noise_prob = self.degradation['gray_noise_prob2']
            if random.random() < self.degradation['gaussian_noise_prob2']:
                out = self._random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.degradation['noise_range2'],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob
                )
            else:
                out = self._random_add_poisson_noise_pt(
                    out,
                    scale_range=self.degradation['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False
                )
        
        # ----------------------- JPEG压缩 + 最终sinc滤波 ----------------------- #
        # 随机选择顺序
        if random.random() < 0.5:
            # 先调整大小和sinc滤波，再JPEG压缩
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out,
                size=(ori_h // sf, ori_w // sf),
                mode=mode
            )
            out = self._filter2D(out, sinc_kernel)
            
            # JPEG压缩
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self._jpeg_compress(out, quality=jpeg_p)
        else:
            # 先JPEG压缩，再调整大小和sinc滤波
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self._jpeg_compress(out, quality=jpeg_p)
            
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out,
                size=(ori_h // sf, ori_w // sf),
                mode=mode
            )
            out = self._filter2D(out, sinc_kernel)
        
        # 裁剪和四舍五入
        im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        
        return {
            'lq': im_lq.contiguous(),
            'gt': img_gt
        }


def test_degradation():
    """测试退化管道"""
    import os
    import glob
    import matplotlib.pyplot as plt
    
    # 获取配置文件路径
    config_path = Path(__file__).parent.parent / 'configs' / 'realesrgan_degradation.yaml'
    
    # 创建退化管道
    degrader = RealESRGANDegradation(str(config_path))
    
    # 读取traindata目录下的图像
    traindata_dir = Path(__file__).parent.parent / 'results'
    image_files = glob.glob(str(traindata_dir / '*.png')) + glob.glob(str(traindata_dir / '*.jpg')) + glob.glob(str(traindata_dir / '*.jpeg'))
    
    if not image_files:
        print(f"错误：在 {traindata_dir} 目录下没有找到图像文件")
        return None
    
    # 读取第一张图像作为测试
    img_path = image_files[1]
    print(f"读取图像: {img_path}")
    
    # 使用PIL读取图像
    img_pil = Image.open(img_path).convert('RGB')
    
    # 转换为tensor [1, 3, H, W]，值范围[0, 1]
    to_tensor = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
    img_gt = to_tensor(img_pil).unsqueeze(0)
    
    print(f"输入图像形状: {img_gt.shape}")
    print(f"输入图像值范围: [{img_gt.min():.3f}, {img_gt.max():.3f}]")
    
    # 应用退化
    result = degrader.degrade(img_gt)
    
    print(f"输出低质量图像形状: {result['lq'].shape}")
    print(f"输出低质量图像值范围: [{result['lq'].min():.3f}, {result['lq'].max():.3f}]")
    print(f"退化管道测试成功！")
    
    # 显示退化前后的图像对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 转换为numpy数组用于显示 [C, H, W] -> [H, W, C]
    img_gt_np = img_gt.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_lq_np = result['lq'].squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # 显示原始图像
    axes[0].imshow(img_gt_np)
    axes[0].set_title('Original (GT)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 显示退化后的图像
    axes[1].imshow(img_lq_np)
    axes[1].set_title('Degraded (LQ)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    plt.show()

if __name__ == '__main__':
    test_degradation()
