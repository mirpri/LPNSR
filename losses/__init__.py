"""
超分辨率训练损失函数模块

包含以下损失函数：
1. L2Loss: 标准的L2损失（MSE）
2. GANLoss: GAN对抗损失
3. LPIPSLoss: LPIPS感知损失（CVPR 2018）
"""

from .basic_loss import L2Loss
from .gan_loss import GANLoss, NLayerDiscriminator, UNetDiscriminator, create_discriminator
from .lpips_loss import LPIPSLoss

__all__ = [
    'L2Loss',
    'GANLoss',
    'NLayerDiscriminator',
    'UNetDiscriminator',
    'create_discriminator',
    'LPIPSLoss',
]