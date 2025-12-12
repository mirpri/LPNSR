"""
增强型自适应噪声预测器 (Enhanced Adaptive Noise Predictor)
用于替代ResShift反向采样过程中的随机高斯噪声

设计思想:
1. 多尺度特征融合: 同时捕获局部细节和全局结构信息
2. 时间步感知: 根据扩散时间步动态调整噪声预测策略
3. 条件引导: 利用原始LR图像的潜在表示提供先验信息
4. 残差学习: 预测噪声残差而非绝对噪声，提高训练稳定性
5. 注意力机制: 自适应地关注图像中需要更精细噪声的区域
6. 跨尺度特征交互: U-Net风格的编码器-解码器架构
7. 频域感知: 结合频域特征增强纹理细节预测
8. 自适应归一化: 根据时间步和条件动态调整归一化参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from SR.ldm.modules.diffusionmodules.openaimodel import (
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)

# 尝试导入xformers以启用内存高效的注意力机制
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILABLE = True
except ImportError:
    XFORMERS_IS_AVAILABLE = False
    print("xformers not available, using standard attention implementation")


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    用于自适应地关注图像中不同区域
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 8, 1)
        self.conv2 = nn.Conv2d(channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            attention_map: [B, 1, H, W]
        """
        att = self.conv1(x)
        att = F.relu(att, inplace=True)
        att = self.conv2(att)
        att = self.sigmoid(att)
        return att


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    用于自适应地调整不同特征通道的重要性
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            attention_weights: [B, C, 1, 1]
        """
        b, c, _, _ = x.size()
        
        # 平均池化和最大池化
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # 融合并归一化
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        return out


class DualAttentionBlock(nn.Module):
    """
    双重注意力块
    结合空间注意力和通道注意力
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            refined_features: [B, C, H, W]
        """
        # 通道注意力
        x = x * self.channel_att(x)
        # 空间注意力
        x = x * self.spatial_att(x)
        return x


class CrossAttentionBlock(nn.Module):
    """
    交叉注意力块（标准实现）
    用于融合含噪特征和LR特征
    """
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm1 = nn.GroupNorm(32, channels)
        self.norm2 = nn.GroupNorm(32, channels)
        
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_k = nn.Conv2d(channels, channels, 1)
        self.to_v = nn.Conv2d(channels, channels, 1)
        self.to_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 查询特征 [B, C, H, W]
            context: 上下文特征 [B, C, H, W]
        Returns:
            attended_features: [B, C, H, W]
        """
        b, c, h, w = x.shape
        
        # 归一化
        x_norm = self.norm1(x)
        context_norm = self.norm2(context)
        
        # 计算Q, K, V
        q = self.to_q(x_norm)
        k = self.to_k(context_norm)
        v = self.to_v(context_norm)
        
        # 重塑为多头格式
        q = q.reshape(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        k = k.reshape(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        v = v.reshape(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        
        # 计算注意力
        scale = self.head_dim ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        
        # 应用注意力
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        out = self.to_out(out)
        
        return x + out


class MemoryEfficientCrossAttentionBlock(nn.Module):
    """
    内存高效的交叉注意力块（使用xFormers加速）
    用于融合含噪特征和LR特征，显著降低显存占用和计算时间
    
    优势:
    - 显存占用降低 30-50%
    - 推理速度提升 20-40%
    - 支持更大的batch size和分辨率
    """
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm1 = nn.GroupNorm(32, channels)
        self.norm2 = nn.GroupNorm(32, channels)
        
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_k = nn.Conv2d(channels, channels, 1)
        self.to_v = nn.Conv2d(channels, channels, 1)
        self.to_out = nn.Conv2d(channels, channels, 1)
        
        self.attention_op: Optional[any] = None
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 查询特征 [B, C, H, W]
            context: 上下文特征 [B, C, H, W]
        Returns:
            attended_features: [B, C, H, W]
        """
        b, c, h, w = x.shape
        
        # 归一化
        x_norm = self.norm1(x)
        context_norm = self.norm2(context)
        
        # 计算Q, K, V
        q = self.to_q(x_norm)  # [B, C, H, W]
        k = self.to_k(context_norm)  # [B, C, H, W]
        v = self.to_v(context_norm)  # [B, C, H, W]
        
        # 重塑为xformers所需的格式: [B*num_heads, H*W, head_dim]
        q = q.reshape(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        k = k.reshape(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        v = v.reshape(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        
        # 合并batch和heads维度
        q = q.reshape(b * self.num_heads, h * w, self.head_dim).contiguous()
        k = k.reshape(b * self.num_heads, h * w, self.head_dim).contiguous()
        v = v.reshape(b * self.num_heads, h * w, self.head_dim).contiguous()
        
        # 使用xformers的内存高效注意力
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
        
        # 重塑回原始格式: [B, C, H, W]
        out = out.reshape(b, self.num_heads, h * w, self.head_dim)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        out = self.to_out(out)
        
        return x + out


class FrequencyAwareBlock(nn.Module):
    """
    频域感知块
    结合频域特征增强纹理细节预测
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # 频域处理分支
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1)
        )
        
        # 空域处理分支
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
        # 融合层
        self.fusion = nn.Conv2d(channels * 2, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            enhanced_features: [B, C, H, W]
        """
        # FFT到频域
        x_freq = torch.fft.rfft2(x, norm='ortho')
        x_freq_real = x_freq.real
        x_freq_imag = x_freq.imag
        
        # 频域处理
        freq_feat = self.freq_conv(x_freq_real)
        
        # IFFT回空域
        freq_feat_complex = torch.complex(freq_feat, x_freq_imag)
        freq_feat_spatial = torch.fft.irfft2(freq_feat_complex, s=x.shape[-2:], norm='ortho')
        
        # 空域处理
        spatial_feat = self.spatial_conv(x)
        
        # 融合频域和空域特征
        fused = torch.cat([freq_feat_spatial, spatial_feat], dim=1)
        out = self.fusion(fused)
        
        return x + out


class DownsampleBlock(nn.Module):
    """
    下采样块
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleBlock(nn.Module):
    """
    上采样块
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MultiScaleFeatureExtractor(nn.Module):
    """
    多尺度特征提取器
    使用不同感受野的卷积核捕获多尺度信息
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_channels = out_channels // 4
        
        # 1x1卷积 - 捕获点特征
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, padding=0),
            nn.GroupNorm(8, mid_channels),
            nn.SiLU()
        )
        
        # 3x3卷积 - 捕获局部特征
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.SiLU()
        )
        
        # 5x5卷积 - 捕获中等范围特征
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 5, padding=2),
            nn.GroupNorm(8, mid_channels),
            nn.SiLU()
        )
        
        # 7x7卷积 - 捕获大范围特征
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 7, padding=3),
            nn.GroupNorm(8, mid_channels),
            nn.SiLU()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W]
        Returns:
            multi_scale_features: [B, C_out, H, W]
        """
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        f4 = self.branch4(x)
        
        # 拼接多尺度特征
        out = torch.cat([f1, f2, f3, f4], dim=1)
        out = self.fusion(out)
        return out


class ResidualBlock(nn.Module):
    """
    残差块，带有时间步嵌入，支持输入输出通道数不同
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm
        
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * out_channels if use_scale_shift_norm else out_channels)
        )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        )
        
        # 如果输入输出通道数不同，需要投影
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H, W]
            emb: [B, emb_channels]
        Returns:
            output: [B, out_channels, H, W]
        """
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = self.out_layers[0](h) * (1 + scale) + shift
            h = self.out_layers[1](h)
            h = self.out_layers[2](h)
            h = self.out_layers[3](h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        
        return self.skip_connection(x) + h


class AdaptiveNoisePredictor(nn.Module):
    """
    增强型自适应噪声预测器
    
    核心创新点:
    1. 多尺度特征提取: 捕获不同尺度的图像特征
    2. 双重注意力机制: 自适应地关注重要区域和通道
    3. 交叉注意力: 融合含噪特征和LR特征
    4. 时间步条件化: 根据扩散时间步调整预测策略
    5. LR图像引导: 利用原始低分辨率图像的先验信息
    6. 残差学习: 预测噪声残差，提高训练稳定性
    7. U-Net架构: 编码器-解码器结构，跨尺度特征融合
    8. 频域感知: 结合频域特征增强纹理预测
    
    输入:
        - noisy_latent: 当前步的含噪潜在特征 [B, C, H, W]
        - timesteps: 扩散时间步 [B]
        - lr_latent: 原始LR图像的潜在表示 [B, C, H, W]
    
    输出:
        - predicted_noise: 预测的噪声 [B, C, H, W]
        - noise_scale: 噪声尺度 [B, 1]
    """
    
    def __init__(
        self,
        latent_channels: int = 3,
        model_channels: int = 192,
        channel_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_levels: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True,
        use_cross_attention: bool = False,
        use_frequency_aware: bool = False,
        use_xformers: bool = True,
        use_checkpoint: bool = False
    ):
        """
        Args:
            latent_channels: 潜在空间的通道数
            model_channels: 模型的基础通道数
            channel_mult: 各层的通道倍数
            num_res_blocks: 每层的残差块数量
            attention_levels: 使用注意力机制的层级数
            num_heads: 多头注意力的头数
            dropout: Dropout概率
            use_scale_shift_norm: 是否使用scale-shift归一化
            use_cross_attention: 是否使用交叉注意力
            use_frequency_aware: 是否使用频域感知
            use_xformers: 是否使用xFormers加速（需要安装xformers）
        """
        super().__init__()
        
        self.latent_channels = latent_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.num_levels = len(channel_mult)
        self.use_cross_attention = use_cross_attention
        self.use_frequency_aware = use_frequency_aware
        self.use_xformers = use_xformers and XFORMERS_IS_AVAILABLE
        self.use_checkpoint = use_checkpoint
        
        # 选择交叉注意力实现
        if self.use_xformers:
            print(f"[NoisePredictor] 使用xFormers加速的交叉注意力")
            CrossAttnBlock = MemoryEfficientCrossAttentionBlock
        else:
            print(f"[NoisePredictor] 使用标准交叉注意力实现")
            CrossAttnBlock = CrossAttentionBlock
        
        # 时间步嵌入
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim)
        )
        
        # 输入投影
        self.input_proj = nn.Conv2d(latent_channels, model_channels, 3, padding=1)
        
        # 为每个层级创建LR特征投影层
        self.lr_projections = nn.ModuleList()
        
        # 编码器路径
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attentions = nn.ModuleList()
        self.encoder_cross_attentions = nn.ModuleList()
        self.encoder_freq_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        ch = model_channels
        for level in range(self.num_levels):
            out_ch = model_channels * channel_mult[level]
            
            # 残差块
            blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                in_ch = ch if i == 0 else out_ch
                blocks.append(
                    ResidualBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                )
            self.encoder_blocks.append(blocks)
            
            # 注意力
            if level < attention_levels:
                self.encoder_attentions.append(DualAttentionBlock(out_ch))
                if use_cross_attention:
                    self.encoder_cross_attentions.append(CrossAttnBlock(out_ch, num_heads))
                    # 为当前层级创建LR特征投影
                    self.lr_projections.append(nn.Conv2d(latent_channels, out_ch, 3, padding=1))
                else:
                    self.encoder_cross_attentions.append(nn.Identity())
                    self.lr_projections.append(None)
                if use_frequency_aware:
                    self.encoder_freq_blocks.append(FrequencyAwareBlock(out_ch))
                else:
                    self.encoder_freq_blocks.append(nn.Identity())
            else:
                self.encoder_attentions.append(nn.Identity())
                self.encoder_cross_attentions.append(nn.Identity())
                self.encoder_freq_blocks.append(nn.Identity())
                self.lr_projections.append(None)
            
            # 下采样
            if level < self.num_levels - 1:
                self.downsamplers.append(DownsampleBlock(out_ch, out_ch))
            
            ch = out_ch
        
        # 中间块
        mid_ch = model_channels * channel_mult[-1]
        self.middle_blocks = nn.ModuleList([
            ResidualBlock(mid_ch, mid_ch, time_embed_dim, dropout, use_scale_shift_norm),
            DualAttentionBlock(mid_ch),
            CrossAttnBlock(mid_ch, num_heads) if use_cross_attention else nn.Identity(),
            FrequencyAwareBlock(mid_ch) if use_frequency_aware else nn.Identity(),
            ResidualBlock(mid_ch, mid_ch, time_embed_dim, dropout, use_scale_shift_norm)
        ])
        
        # 中间块的LR投影
        self.middle_lr_proj = nn.Conv2d(latent_channels, mid_ch, 3, padding=1) if use_cross_attention else None
        
        # 解码器路径
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attentions = nn.ModuleList()
        self.decoder_cross_attentions = nn.ModuleList()
        self.decoder_freq_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        for level in reversed(range(self.num_levels)):
            out_ch = model_channels * channel_mult[level]
            
            # 残差块（解码器使用较少的残差块）
            blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                if i == 0:
                    # 第一个块处理拼接后的特征
                    in_ch = ch + out_ch
                else:
                    in_ch = out_ch
                blocks.append(
                    ResidualBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm
                    )
                )
            self.decoder_blocks.append(blocks)
            
            # 注意力
            if level < attention_levels:
                self.decoder_attentions.append(DualAttentionBlock(out_ch))
                if use_cross_attention:
                    self.decoder_cross_attentions.append(CrossAttnBlock(out_ch, num_heads))
                else:
                    self.decoder_cross_attentions.append(nn.Identity())
                if use_frequency_aware:
                    self.decoder_freq_blocks.append(FrequencyAwareBlock(out_ch))
                else:
                    self.decoder_freq_blocks.append(nn.Identity())
            else:
                self.decoder_attentions.append(nn.Identity())
                self.decoder_cross_attentions.append(nn.Identity())
                self.decoder_freq_blocks.append(nn.Identity())
            
            # 上采样
            if level > 0:
                self.upsamplers.append(UpsampleBlock(out_ch, out_ch))
            
            ch = out_ch
        
        # 输出层
        self.output = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, latent_channels, 3, padding=1))
        )
        
        # 噪声尺度预测器
        self.noise_scale_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            linear(model_channels, model_channels),
            nn.SiLU(),
            linear(model_channels, model_channels // 2),
            nn.SiLU(),
            linear(model_channels // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        lr_latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            noisy_latent: 当前步的含噪潜在特征 [B, C, H, W]
            timesteps: 扩散时间步 [B]
            lr_latent: 原始LR图像的潜在表示 [B, C, H, W]
        
        Returns:
            predicted_noise: 预测的噪声 [B, C, H, W]
            noise_scale: 噪声尺度 [B, 1]
        """
        # 时间步嵌入
        t_emb = timestep_embedding(timesteps, self.model_channels)
        t_emb = self.time_embed(t_emb)
        
        # 输入投影
        h = self.input_proj(noisy_latent)
        
        # 编码器路径
        encoder_features = []
        for level in range(self.num_levels):
            # 残差块
            for block in self.encoder_blocks[level]:
                if self.use_checkpoint:
                    h = torch.utils.checkpoint.checkpoint(block, h, t_emb, use_reentrant=False)
                else:
                    h = block(h, t_emb)
            
            # 注意力
            h = self.encoder_attentions[level](h)
            
            # 交叉注意力（融合LR特征）
            if level < len(self.encoder_cross_attentions):
                if isinstance(self.encoder_cross_attentions[level], CrossAttentionBlock):
                    # 使用对应层级的投影层处理LR特征
                    lr_resized = F.adaptive_avg_pool2d(lr_latent, h.shape[2:])
                    lr_feat = self.lr_projections[level](lr_resized)
                    h = self.encoder_cross_attentions[level](h, lr_feat)
            
            # 频域感知
            if level < len(self.encoder_freq_blocks):
                if isinstance(self.encoder_freq_blocks[level], FrequencyAwareBlock):
                    h = self.encoder_freq_blocks[level](h)
            
            # 保存特征用于跳跃连接
            encoder_features.append(h)
            
            # 下采样
            if level < self.num_levels - 1:
                h = self.downsamplers[level](h)
        
        # 中间块
        for i, block in enumerate(self.middle_blocks):
            if isinstance(block, ResidualBlock):
                if self.use_checkpoint:
                    h = torch.utils.checkpoint.checkpoint(block, h, t_emb, use_reentrant=False)
                else:
                    h = block(h, t_emb)
            elif isinstance(block, (CrossAttentionBlock, MemoryEfficientCrossAttentionBlock)):
                lr_resized = F.adaptive_avg_pool2d(lr_latent, h.shape[2:])
                lr_feat = self.middle_lr_proj(lr_resized)
                if self.use_checkpoint:
                    h = torch.utils.checkpoint.checkpoint(block, h, lr_feat, use_reentrant=False)
                else:
                    h = block(h, lr_feat)
            else:
                if self.use_checkpoint:
                    h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
                else:
                    h = block(h)
        
        # 解码器路径
        for level in range(self.num_levels):
            # 跳跃连接
            skip_feat = encoder_features[-(level + 1)]
            h = torch.cat([h, skip_feat], dim=1)
            
            # 残差块
            for i, block in enumerate(self.decoder_blocks[level]):
                if i == 0:
                    # 第一个块处理拼接后的特征
                    if self.use_checkpoint:
                        h = torch.utils.checkpoint.checkpoint(block, h, t_emb, use_reentrant=False)
                    else:
                        h = block(h, t_emb)
                else:
                    if self.use_checkpoint:
                        h = torch.utils.checkpoint.checkpoint(block, h, t_emb, use_reentrant=False)
                    else:
                        h = block(h, t_emb)
            
            # 注意力
            h = self.decoder_attentions[level](h)
            
            # 交叉注意力
            decoder_level = self.num_levels - 1 - level
            if decoder_level < len(self.lr_projections) and self.lr_projections[decoder_level] is not None:
                if isinstance(self.decoder_cross_attentions[level], CrossAttentionBlock):
                    lr_resized = F.adaptive_avg_pool2d(lr_latent, h.shape[2:])
                    lr_feat = self.lr_projections[decoder_level](lr_resized)
                    h = self.decoder_cross_attentions[level](h, lr_feat)
            
            # 频域感知
            if level < len(self.decoder_freq_blocks):
                if isinstance(self.decoder_freq_blocks[level], FrequencyAwareBlock):
                    h = self.decoder_freq_blocks[level](h)
            
            # 上采样
            if level < self.num_levels - 1:
                h = self.upsamplers[level](h)
        
        # 预测噪声
        noise = self.output(h)
        
        # 预测噪声尺度
        noise_scale = self.noise_scale_predictor(h)
        
        # 应用噪声尺度
        scaled_noise = noise * noise_scale.view(-1, 1, 1, 1)
        
        return scaled_noise, noise_scale


def create_noise_predictor(
    latent_channels: int = 3,
    **kwargs
) -> nn.Module:
    """
    工厂函数：创建噪声预测器
    
    Args:
        latent_channels: 潜在空间的通道数
        **kwargs: 其他模型参数
    
    Returns:
        noise_predictor: 增强型自适应噪声预测器
    """
    return AdaptiveNoisePredictor(latent_channels=latent_channels, **kwargs)


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = create_noise_predictor(
        latent_channels=3,
        model_channels=160,
        channel_mult=(1, 2, 3),
        num_res_blocks=2,
        attention_levels=2,
        num_heads=8,
        use_cross_attention=True,
        use_frequency_aware=True
    ).to(device)
    
    # 测试输入
    batch_size = 2
    noisy_latent = torch.randn(batch_size, 3, 64, 64).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    lr_latent = torch.randn(batch_size, 3, 64, 64).to(device)
    
    # 前向传播
    predicted_noise, noise_scale = model(noisy_latent, timesteps, lr_latent)
    
    print(f"输入形状: {noisy_latent.shape}")
    print(f"预测噪声形状: {predicted_noise.shape}")
    print(f"噪声尺度形状: {noise_scale.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
