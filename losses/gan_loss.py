#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GAN损失函数

包含：
1. PatchGAN判别器（用于图像空间的对抗训练）
2. GAN损失计算（支持多种GAN类型）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class NLayerDiscriminator(nn.Module):
    """
    PatchGAN判别器

    将图像分成多个patch，每个patch独立判断真假
    输出是一个特征图，每个位置代表对应patch的真假判断

    参考: pix2pix, pix2pixHD
    """

    def __init__(
            self,
            input_nc=3,
            ndf=64,
            n_layers=3,
            norm_type='spectral',
            use_sigmoid=False
    ):
        """
        Args:
            input_nc: 输入通道数
            ndf: 第一层卷积的通道数
            n_layers: 判别器层数
            norm_type: 归一化类型 ('batch', 'instance', 'spectral', 'none')
            use_sigmoid: 是否在输出使用sigmoid（vanilla GAN需要，LSGAN/WGAN不需要）
        """
        super().__init__()

        self.use_sigmoid = use_sigmoid

        # 选择归一化层
        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
            use_spectral = False
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d
            use_spectral = False
        elif norm_type == 'spectral':
            norm_layer = None
            use_spectral = True
        else:
            norm_layer = None
            use_spectral = False

        kw = 4  # 卷积核大小
        padw = 1  # padding大小

        # 第一层：不使用归一化
        sequence = []
        if use_spectral:
            sequence.append(spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)))
        else:
            sequence.append(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw))
        sequence.append(nn.LeakyReLU(0.2, True))

        # 中间层
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)

            if use_spectral:
                sequence.append(spectral_norm(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw)
                ))
            else:
                sequence.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw))

            if norm_layer is not None:
                sequence.append(norm_layer(ndf * nf_mult))

            sequence.append(nn.LeakyReLU(0.2, True))

        # 倒数第二层
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        if use_spectral:
            sequence.append(spectral_norm(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw)
            ))
        else:
            sequence.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw))

        if norm_layer is not None:
            sequence.append(norm_layer(ndf * nf_mult))

        sequence.append(nn.LeakyReLU(0.2, True))

        # 最后一层：输出1通道
        if use_spectral:
            sequence.append(spectral_norm(
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
            ))
        else:
            sequence.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """
        Args:
            x: 输入图像 [B, C, H, W]

        Returns:
            判别结果 [B, 1, H', W']
        """
        out = self.model(x)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out


class UNetDiscriminator(nn.Module):
    """
    UNet结构的判别器

    结合了全局和局部判断能力，适合高分辨率图像
    参考: Real-ESRGAN
    """

    def __init__(
            self,
            input_nc=3,
            ndf=64,
            skip_connection=True
    ):
        """
        Args:
            input_nc: 输入通道数
            ndf: 基础通道数
            skip_connection: 是否使用跳跃连接
        """
        super().__init__()

        self.skip_connection = skip_connection

        # 编码器
        self.conv0 = nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1)

        self.conv1 = spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv3 = spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False))

        # 解码器
        self.conv4 = spectral_norm(nn.Conv2d(ndf * 8, ndf * 4, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv5 = spectral_norm(nn.Conv2d(ndf * 4, ndf * 2, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv6 = spectral_norm(nn.Conv2d(ndf * 2, ndf, kernel_size=3, stride=1, padding=1, bias=False))

        # 最终输出
        self.conv7 = spectral_norm(nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv8 = spectral_norm(nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv9 = nn.Conv2d(ndf, 1, kernel_size=3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        """
        Args:
            x: 输入图像 [B, C, H, W]

        Returns:
            判别结果 [B, 1, H, W]
        """
        # 编码
        feat0 = self.lrelu(self.conv0(x))
        feat1 = self.lrelu(self.conv1(feat0))
        feat2 = self.lrelu(self.conv2(feat1))
        feat3 = self.lrelu(self.conv3(feat2))

        # 解码 + 上采样
        feat3 = F.interpolate(feat3, scale_factor=2, mode='bilinear', align_corners=False)
        feat4 = self.lrelu(self.conv4(feat3))
        if self.skip_connection:
            feat4 = feat4 + feat2

        feat4 = F.interpolate(feat4, scale_factor=2, mode='bilinear', align_corners=False)
        feat5 = self.lrelu(self.conv5(feat4))
        if self.skip_connection:
            feat5 = feat5 + feat1

        feat5 = F.interpolate(feat5, scale_factor=2, mode='bilinear', align_corners=False)
        feat6 = self.lrelu(self.conv6(feat5))
        if self.skip_connection:
            feat6 = feat6 + feat0

        # 最终输出
        out = self.lrelu(self.conv7(feat6))
        out = self.lrelu(self.conv8(out))
        out = self.conv9(out)

        return out


class GANLoss(nn.Module):
    """
    GAN损失计算

    支持多种GAN类型：
    - vanilla: 原始GAN (BCE loss)
    - lsgan: 最小二乘GAN (MSE loss)
    - wgan: Wasserstein GAN
    - wgan-gp: WGAN with gradient penalty
    - hinge: Hinge loss (常用于BigGAN, StyleGAN)
    """

    def __init__(
            self,
            gan_type='lsgan',
            real_label=1.0,
            fake_label=0.0,
            loss_weight=1.0
    ):
        """
        Args:
            gan_type: GAN类型 ('vanilla', 'lsgan', 'wgan', 'wgan-gp', 'hinge')
            real_label: 真实标签值
            fake_label: 假标签值
            loss_weight: 损失权重
        """
        super().__init__()

        self.gan_type = gan_type
        self.real_label = real_label
        self.fake_label = fake_label
        self.loss_weight = loss_weight

        if gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_type in ['wgan', 'wgan-gp', 'hinge']:
            self.loss = None
        else:
            raise ValueError(f"不支持的GAN类型: {gan_type}")

    def _get_target_label(self, pred, target_is_real):
        """获取目标标签"""
        if target_is_real:
            target_label = self.real_label
        else:
            target_label = self.fake_label
        return torch.full_like(pred, target_label)

    def forward(self, pred, target_is_real, is_disc=False):
        """
        计算GAN损失

        Args:
            pred: 判别器输出
            target_is_real: 目标是否为真实图像
            is_disc: 是否是判别器的损失（用于hinge loss）

        Returns:
            GAN损失值
        """
        if self.gan_type == 'vanilla' or self.gan_type == 'lsgan':
            target_label = self._get_target_label(pred, target_is_real)
            loss = self.loss(pred, target_label)

        elif self.gan_type == 'wgan':
            if target_is_real:
                loss = -pred.mean()
            else:
                loss = pred.mean()

        elif self.gan_type == 'wgan-gp':
            if target_is_real:
                loss = -pred.mean()
            else:
                loss = pred.mean()

        elif self.gan_type == 'hinge':
            if is_disc:
                if target_is_real:
                    loss = F.relu(1.0 - pred).mean()
                else:
                    loss = F.relu(1.0 + pred).mean()
            else:
                # 生成器损失
                loss = -pred.mean()
        else:
            raise ValueError(f"不支持的GAN类型: {self.gan_type}")

        return self.loss_weight * loss

    def compute_gradient_penalty(self, discriminator, real_samples, fake_samples):
        """
        计算WGAN-GP的梯度惩罚

        Args:
            discriminator: 判别器
            real_samples: 真实样本
            fake_samples: 生成样本

        Returns:
            梯度惩罚值
        """
        batch_size = real_samples.size(0)
        device = real_samples.device

        # 随机插值系数
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)

        # 插值样本
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)

        # 判别器输出
        d_interpolates = discriminator(interpolates)

        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # 计算梯度惩罚
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty


def create_discriminator(
        disc_type='patch',
        input_nc=3,
        ndf=64,
        n_layers=3,
        norm_type='spectral'
):
    """
    创建判别器

    Args:
        disc_type: 判别器类型 ('patch', 'unet')
        input_nc: 输入通道数
        ndf: 基础通道数
        n_layers: 层数（仅PatchGAN）
        norm_type: 归一化类型

    Returns:
        判别器实例
    """
    if disc_type == 'patch':
        return NLayerDiscriminator(
            input_nc=input_nc,
            ndf=ndf,
            n_layers=n_layers,
            norm_type=norm_type,
            use_sigmoid=False
        )
    elif disc_type == 'unet':
        return UNetDiscriminator(
            input_nc=input_nc,
            ndf=ndf,
            skip_connection=True
        )
    else:
        raise ValueError(f"不支持的判别器类型: {disc_type}")


if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试PatchGAN判别器
    print("测试PatchGAN判别器...")
    disc_patch = create_discriminator('patch', input_nc=3, ndf=64, n_layers=3).to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    out = disc_patch(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in disc_patch.parameters()) / 1e6:.2f}M")

    # 测试UNet判别器
    print("\n测试UNet判别器...")
    disc_unet = create_discriminator('unet', input_nc=3, ndf=64).to(device)
    out = disc_unet(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in disc_unet.parameters()) / 1e6:.2f}M")

    # 测试GAN损失
    print("\n测试GAN损失...")
    gan_loss = GANLoss(gan_type='lsgan', loss_weight=1.0)

    fake_pred = torch.randn(2, 1, 30, 30).to(device)
    real_pred = torch.randn(2, 1, 30, 30).to(device)

    # 判别器损失
    d_loss_real = gan_loss(real_pred, target_is_real=True, is_disc=True)
    d_loss_fake = gan_loss(fake_pred, target_is_real=False, is_disc=True)
    d_loss = d_loss_real + d_loss_fake
    print(f"  判别器损失: {d_loss.item():.4f}")

    # 生成器损失
    g_loss = gan_loss(fake_pred, target_is_real=True, is_disc=False)
    print(f"  生成器损失: {g_loss.item():.4f}")

    print("\n测试完成！")
