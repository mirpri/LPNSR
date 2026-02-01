#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练SwinIR超分辨率模型

训练思路：
1. 输入LR图像
2. 使用SwinIR直接输出HR图像
3. 计算预测HR图像与真实HR图像之间的损失
4. 反向传播更新SwinIR参数

关键：
- 使用LPNSR的数据加载器（RealESRGAN退化管道）
- 支持多种损失函数（L2、L1、Charbonnier、LPIPS、GAN）
- 支持AMP混合精度训练
- 支持checkpoint保存和恢复
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*A matching Triton is not available.*")
warnings.filterwarnings("ignore", message=".*No module named 'triton'.*")
warnings.filterwarnings("ignore", message=".*Importing from timm.models.layers is deprecated.*")
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import numpy as np

# LPNSR模块导入
from LPNSR.models.network_swinir import SwinIR
from LPNSR.losses.basic_loss import L2Loss, CharbonnierLoss
from LPNSR.losses.lpips_loss import LPIPSLoss
from LPNSR.losses.gan_loss import GANLoss, create_discriminator
from LPNSR.datapipe.train_dataloader import create_train_dataloader


class SwinIRTrainer:
    """SwinIR训练器"""

    def __init__(self, config_path):
        """
        初始化训练器

        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建实验目录
        self.exp_dir = Path(self.config['experiment']['save_dir'])
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.exp_dir / 'samples').mkdir(exist_ok=True)

        # 初始化模型
        self._init_models()

        # 初始化损失函数
        self._init_losses()

        # 初始化优化器
        self._init_optimizer()

        # 初始化数据加载器
        self._init_dataloaders()

        # 初始化AMP
        if self.config['training']['use_amp']:
            self.scaler_g = GradScaler()
            if self.discriminator is not None:
                self.scaler_d = GradScaler()
        else:
            self.scaler_g = None
            self.scaler_d = None

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

        print(f"训练器初始化完成！")
        print(f"实验目录: {self.exp_dir}")
        print(f"设备: {self.device}")

    def _init_models(self):
        """初始化模型"""
        print("\n" + "=" * 70)
        print("初始化模型")
        print("=" * 70)

        # 创建SwinIR模型
        print("\n创建SwinIR模型...")
        model_config = self.config['swinir']

        # 计算patch_size（根据图像大小和缩放因子）
        img_size = self.config['data']['train']['crop_size']
        scale = self.config['data']['train']['scale']
        lr_img_size = img_size // scale

        self.model = SwinIR(
            img_size=lr_img_size,
            patch_size=model_config.get('patch_size', 1),
            in_chans=3,
            embed_dim=model_config.get('embed_dim', 60),
            depths=model_config.get('depths', [6, 6, 6, 6, 6, 6]),
            num_heads=model_config.get('num_heads', [6, 6, 6, 6, 6, 6]),
            window_size=model_config.get('window_size', 8),
            mlp_ratio=model_config.get('mlp_ratio', 2.0),
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=model_config.get('drop_path_rate', 0.1),
            norm_layer=nn.LayerNorm,
            ape=model_config.get('ape', False),
            patch_norm=True,
            use_checkpoint=self.config['training'].get('use_gradient_checkpointing', False),
            upscale=scale,
            img_range=1.0,
            upsampler=model_config.get('upsampler', 'pixelshuffle'),
            resi_connection=model_config.get('resi_connection', '1conv'),
        ).to(self.device)

        # 加载预训练权重（如果指定）
        if 'pretrained_path' in model_config and model_config['pretrained_path']:
            print(f"\n加载预训练权重: {model_config['pretrained_path']}")
            checkpoint = torch.load(model_config['pretrained_path'], map_location=self.device)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # 去除前缀
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                if key.startswith('module._orig_mod.'):
                    new_key = key.replace('module._orig_mod.', '')
                elif key.startswith('module.'):
                    new_key = key.replace('module.', '')
                new_state_dict[new_key] = value

            self.model.load_state_dict(new_state_dict, strict=False)
            print(f"✓ 预训练权重加载完成")

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ SwinIR模型创建完成")
        print(f"  - 参数量: {num_params / 1e6:.2f}M")
        print(f"  - 输入尺寸: {lr_img_size}x{lr_img_size}")
        print(f"  - 输出尺寸: {img_size}x{img_size}")

        # 统计可训练参数
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n总可训练参数: {total_params / 1e6:.2f}M")

    def _init_losses(self):
        """初始化损失函数"""
        print("\n" + "=" * 70)
        print("初始化损失函数")
        print("=" * 70)

        loss_config = self.config['loss']

        # 基础损失（L2/L1/Charbonnier）
        loss_type = loss_config.get('loss_type', 'l2')
        if loss_type == 'l2':
            self.pixel_loss = L2Loss()
            print(f"✓ L2损失 (权重: {loss_config['pixel_weight']})")
        elif loss_type == 'l1':
            self.pixel_loss = nn.L1Loss()
            print(f"✓ L1损失 (权重: {loss_config['pixel_weight']})")
        elif loss_type == 'charbonnier':
            self.pixel_loss = CharbonnierLoss()
            print(f"✓ Charbonnier损失 (权重: {loss_config['pixel_weight']})")
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")

        # LPIPS感知损失
        if loss_config.get('lpips_weight', 0) > 0:
            self.lpips_loss = LPIPSLoss(
                loss_weight=1.0,
                net_type=loss_config.get('lpips_net_type', 'alex')
            )
            print(f"✓ LPIPS感知损失 (权重: {loss_config['lpips_weight']})")
        else:
            self.lpips_loss = None

        # GAN损失
        if loss_config.get('gan_weight', 0) > 0:
            # 创建判别器
            self.discriminator = create_discriminator(
                disc_type=loss_config.get('disc_type', 'patch'),
                input_nc=3,
                ndf=loss_config.get('disc_ndf', 64),
                n_layers=loss_config.get('disc_n_layers', 3),
                norm_type=loss_config.get('disc_norm_type', 'spectral')
            ).to(self.device)

            # 创建GAN损失
            self.gan_loss = GANLoss(
                gan_type=loss_config.get('gan_type', 'lsgan'),
                loss_weight=1.0
            )

            # 统计判别器参数量
            disc_params = sum(p.numel() for p in self.discriminator.parameters())
            print(f"✓ GAN损失 (权重: {loss_config['gan_weight']})")
            print(f"  - 判别器类型: {loss_config.get('disc_type', 'patch')}")
            print(f"  - GAN类型: {loss_config.get('gan_type', 'lsgan')}")
            print(f"  - 判别器参数量: {disc_params / 1e6:.2f}M")
        else:
            self.discriminator = None
            self.gan_loss = None

    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        print("\n" + "=" * 70)
        print("初始化优化器")
        print("=" * 70)

        opt_config = self.config['optimizer']

        # 优化器
        if opt_config['type'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                betas=(opt_config['beta1'], opt_config['beta2']),
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['type'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                betas=(opt_config['beta1'], opt_config['beta2']),
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"不支持的优化器类型: {opt_config['type']}")

        print(f"✓ 优化器: {opt_config['type']}")
        print(f"  - 学习率: {opt_config['lr']}")
        print(f"  - Weight decay: {opt_config['weight_decay']}")

        # 学习率调度器
        scheduler_config = self.config['scheduler']
        if scheduler_config['type'] == 'CosineAnnealing':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=scheduler_config['min_lr']
            )
        elif scheduler_config['type'] == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        else:
            self.scheduler = None

        if self.scheduler:
            print(f"✓ 学习率调度器: {scheduler_config['type']}")

        # 判别器优化器（如果启用GAN损失）
        loss_config = self.config['loss']
        if loss_config.get('gan_weight', 0) > 0 and self.discriminator is not None:
            disc_lr = loss_config.get('disc_lr', 1.0e-4)
            self.optimizer_d = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=disc_lr,
                betas=(opt_config['beta1'], opt_config['beta2']),
                weight_decay=opt_config['weight_decay']
            )
            print(f"✓ 判别器优化器: AdamW")
            print(f"  - 学习率: {disc_lr}")
        else:
            self.optimizer_d = None

    def _init_dataloaders(self):
        """初始化数据加载器"""
        print("\n" + "=" * 70)
        print("初始化数据加载器")
        print("=" * 70 + "\n")

        data_config = self.config['data']
        train_config = self.config['training']

        # 训练数据加载器
        if self.config['degradation']['use_degradation']:
            # 使用退化管道生成LR图像
            print("使用退化管道生成LR图像...")
            self.train_loader = create_train_dataloader(
                data_dir=data_config['train']['hr_dir'],
                config_path=self.config['degradation']['config_path'],
                batch_size=train_config['batch_size'],
                num_workers=train_config['num_workers'],
                gt_size=data_config['train']['crop_size'],
                use_hflip=data_config['train']['use_flip'],
                use_rot=data_config['train']['use_rot'],
                shuffle=True,
                pin_memory=True
            )
            print(f"✓ 训练数据加载器创建成功：{len(self.train_loader)} batches")
        else:
            raise NotImplementedError("暂不支持直接加载LR-HR图像对")

    def freeze_model(self, model):
        """冻结模型参数"""
        for param in model.parameters():
            param.requires_grad = False

    def unfreeze_model(self, model):
        """解冻模型参数"""
        for param in model.parameters():
            param.requires_grad = True

    def train_step(self, lr_images, hr_images):
        """
        单步训练

        Args:
            lr_images: LR图像 [B, 3, H, W], 范围[0, 1]
            hr_images: HR图像 [B, 3, H, W], 范围[0, 1]

        Returns:
            loss_dict: 损失字典
        """
        self.model.train()
        batch_size = lr_images.shape[0]
        loss_config = self.config['loss']

        # 获取判别器更新频率
        disc_update_freq = loss_config.get('disc_update_freq', 1)
        should_update_disc = (self.global_step % disc_update_freq == 0)

        # 1. 先计算生成器损失
        if self.config['training']['use_amp']:
            with autocast(device_type='cuda'):
                pred_hr = self.model(lr_images)
                loss, loss_dict = self._compute_generator_loss(pred_hr, hr_images)
        else:
            pred_hr = self.model(lr_images)
            loss, loss_dict = self._compute_generator_loss(pred_hr, hr_images)

        # 2. 交替训练逻辑
        if should_update_disc and self.discriminator is not None and loss_config.get('gan_weight', 0) > 0:
            # === 这个step只更新判别器 ===
            disc_start_epoch = loss_config.get('disc_start_epoch', 0)
            if self.current_epoch >= disc_start_epoch:
                # 获取生成器产生的图像（已在_compute_generator_loss中保存）
                if hasattr(self, '_pred_hr_for_disc') and hasattr(self, '_gt_hr_for_disc'):
                    fake_image = self._pred_hr_for_disc
                    real_image = self._gt_hr_for_disc

                    self.discriminator.train()

                    # 冻结生成器，解冻判别器
                    self.freeze_model(self.model)
                    self.unfreeze_model(self.discriminator)

                    # 计算判别器损失
                    if self.config['training']['use_amp']:
                        with autocast(device_type='cuda'):
                            # 判别真实图像
                            real_pred = self.discriminator(real_image)
                            d_loss_real = self.gan_loss(real_pred, target_is_real=True, is_disc=True)

                            # 判别生成图像
                            fake_pred = self.discriminator(fake_image)
                            d_loss_fake = self.gan_loss(fake_pred, target_is_real=False, is_disc=True)

                            d_loss = (d_loss_real + d_loss_fake) / 2

                        self.scaler_d.scale(d_loss).backward()

                        # 梯度裁剪
                        if self.config['training']['gradient_clip'] > 0:
                            self.scaler_d.unscale_(self.optimizer_d)
                            torch.nn.utils.clip_grad_norm_(
                                self.discriminator.parameters(),
                                self.config['training']['gradient_clip']
                            )
                        self.scaler_d.step(self.optimizer_d)
                        self.scaler_d.update()
                        self.optimizer_d.zero_grad()
                    else:
                        # 判别真实图像
                        real_pred = self.discriminator(real_image)
                        d_loss_real = self.gan_loss(real_pred, target_is_real=True, is_disc=True)

                        # 判别生成图像
                        fake_pred = self.discriminator(fake_image)
                        d_loss_fake = self.gan_loss(fake_pred, target_is_real=False, is_disc=True)

                        d_loss = (d_loss_real + d_loss_fake) / 2

                        d_loss.backward()

                        # 梯度裁剪
                        if self.config['training']['gradient_clip'] > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.discriminator.parameters(),
                                self.config['training']['gradient_clip']
                            )
                        self.optimizer_d.step()
                        self.optimizer_d.zero_grad()

                    # 记录损失
                    loss_dict['d_loss'] = d_loss.item()
                    loss_dict['d_loss_real'] = d_loss_real.item()
                    loss_dict['d_loss_fake'] = d_loss_fake.item()

                # 恢复生成器参数状态
                self.unfreeze_model(self.model)
        else:
            # === 这个step只更新生成器 ===

            # 冻结判别器，解冻生成器
            if self.discriminator is not None:
                self.freeze_model(self.discriminator)
            self.unfreeze_model(self.model)

            # 反向传播生成器损失
            if self.config['training']['use_amp']:
                self.scaler_g.scale(loss).backward()

                # 梯度裁剪
                if self.config['training']['gradient_clip'] > 0:
                    self.scaler_g.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )

                self.scaler_g.step(self.optimizer)
                self.scaler_g.update()
                self.optimizer.zero_grad()
            else:
                loss.backward()

                # 梯度裁剪
                if self.config['training']['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()

            # 恢复判别器参数状态
            if self.discriminator is not None:
                self.unfreeze_model(self.discriminator)

        return loss_dict

    def _compute_generator_loss(self, pred_hr, gt_hr):
        """
        计算生成器损失

        Args:
            pred_hr: 预测的HR图像 [B, 3, H, W]
            gt_hr: 真实的HR图像 [B, 3, H, W]

        Returns:
            total_loss: 总损失
            loss_dict: 损失字典
        """
        loss_dict = {}
        loss_config = self.config['loss']
        total_loss = 0.0

        # 像素级损失
        pixel_loss = self.pixel_loss(pred_hr, gt_hr)
        loss_dict['pixel'] = pixel_loss.item()
        total_loss += loss_config['pixel_weight'] * pixel_loss

        # LPIPS感知损失
        if self.lpips_loss is not None and loss_config.get('lpips_weight', 0) > 0:
            lpips_loss = self.lpips_loss(pred_hr, gt_hr)
            loss_dict['lpips'] = lpips_loss.item()
            total_loss += loss_config['lpips_weight'] * lpips_loss

        # GAN生成器损失
        if self.gan_loss is not None and loss_config.get('gan_weight', 0) > 0:
            disc_start_epoch = loss_config.get('disc_start_epoch', 0)
            if self.current_epoch >= disc_start_epoch:
                # 计算生成器损失：让判别器认为生成图像是真的
                fake_pred = self.discriminator(pred_hr)
                g_loss = self.gan_loss(fake_pred, target_is_real=True, is_disc=False)
                loss_dict['g_loss'] = g_loss.item()
                total_loss += loss_config['gan_weight'] * g_loss

            # 保存图像供判别器训练使用
            self._pred_hr_for_disc = pred_hr.detach()
            self._gt_hr_for_disc = gt_hr

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict

    def train_epoch(self, epoch):
        """
        训练一个epoch

        Args:
            epoch: 当前epoch数

        Returns:
            avg_loss_dict: 平均损失字典
        """
        self.model.train()
        self.current_epoch = epoch

        total_loss_dict = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['training']['num_epochs']}")

        for step, batch in enumerate(pbar):
            # 获取数据
            hr_images = batch['gt'].to(self.device)
            lr_images = batch['lq'].to(self.device)

            # 训练一步
            loss_dict = self.train_step(lr_images, hr_images)

            # 累积损失
            for key, value in loss_dict.items():
                if key not in total_loss_dict:
                    total_loss_dict[key] = 0.0
                total_loss_dict[key] += value

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # 打印日志
            if (step + 1) % self.config['experiment']['log_interval'] == 0:
                print(f"\nEpoch [{epoch}/{self.config['training']['num_epochs']}] "
                      f"Step [{step + 1}/{len(self.train_loader)}]")
                for key, value in loss_dict.items():
                    print(f"  {key}: {value:.4f}")
                print(f"  lr: {self.optimizer.param_groups[0]['lr']:.2e}")

            self.global_step += 1

        # 计算平均损失
        num_batches = len(self.train_loader)
        avg_loss_dict = {key: value / num_batches for key, value in total_loss_dict.items()}

        return avg_loss_dict

    def save_checkpoint(self, epoch, is_best=False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'config': self.config,
        }

        # 保存判别器状态（如果启用GAN损失）
        if self.discriminator is not None:
            checkpoint['discriminator'] = self.discriminator.state_dict()
        if self.optimizer_d is not None:
            checkpoint['optimizer_d'] = self.optimizer_d.state_dict()

        # 保存最新的checkpoint
        ckpt_path = self.exp_dir / 'checkpoints' / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, ckpt_path)
        print(f"✓ Checkpoint已保存: {ckpt_path}")

        # 保存最佳模型（只保存模型权重）
        if is_best:
            best_path = self.exp_dir / 'checkpoints' / 'swinir_best.pth'
            torch.save(self.model.state_dict(), best_path)
            print(f"✓ 最佳模型已保存: {best_path}")

        # 删除旧的checkpoint（保留最近N个）
        keep_recent = self.config['experiment'].get('keep_recent_checkpoints', 5)
        checkpoints = sorted(
            (self.exp_dir / 'checkpoints').glob('checkpoint_epoch_*.pth'),
            key=lambda x: int(x.stem.split('_')[-1])
        )
        if len(checkpoints) > keep_recent:
            for old_ckpt in checkpoints[:-keep_recent]:
                old_ckpt.unlink()
                print(f"✓ 删除旧checkpoint: {old_ckpt.name}")

    def load_checkpoint(self, checkpoint_path):
        """加载checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler and checkpoint['scheduler']:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']

        # 加载判别器状态（如果存在）
        if self.discriminator is not None and 'discriminator' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            print(f"  - 已加载判别器权重")
        if self.optimizer_d is not None and 'optimizer_d' in checkpoint:
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
            print(f"  - 已加载判别器优化器状态")

        print(f"✓ Checkpoint已加载: {checkpoint_path}")
        print(f"  - Epoch: {self.current_epoch}")
        print(f"  - Global step: {self.global_step}")
        print(f"  - Best loss: {self.best_loss:.6f}")

        return checkpoint


def main():
    parser = argparse.ArgumentParser(description='训练SwinIR超分辨率模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的checkpoint路径')
    args = parser.parse_args()

    # 创建训练器
    trainer = SwinIRTrainer(args.config)

    # 恢复训练
    start_epoch = 1
    if args.resume:
        checkpoint = trainer.load_checkpoint(args.resume)
        if checkpoint is not None:
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"\n从epoch {start_epoch}恢复训练")
        else:
            print("\n警告: 无法加载checkpoint，从头开始训练")

    print("\n" + "=" * 70)
    print("开始训练！")
    print("=" * 70 + "\n")

    # 训练循环
    num_epochs = trainer.config['training']['num_epochs']

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'=' * 70}\n")

            # 训练一个epoch
            avg_loss_dict = trainer.train_epoch(epoch)

            # 打印平均损失
            print(f"\nEpoch {epoch} 平均损失:")
            for key, value in avg_loss_dict.items():
                print(f"  {key}: {value:.4f}")

            # 更新学习率
            if trainer.scheduler is not None:
                trainer.scheduler.step()
                print(f"\n当前学习率: {trainer.optimizer.param_groups[0]['lr']:.2e}")

            # 保存checkpoint
            if epoch % trainer.config['experiment']['save_interval'] == 0:
                is_best = avg_loss_dict['total'] < trainer.best_loss
                if is_best:
                    trainer.best_loss = avg_loss_dict['total']
                trainer.save_checkpoint(epoch, is_best=is_best)

        print("\n" + "=" * 70)
        print("训练完成！")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n训练被中断！")
        print("保存当前checkpoint...")
        trainer.save_checkpoint(epoch, is_best=False)
        print("Checkpoint已保存，可以使用--resume恢复训练")


if __name__ == '__main__':
    main()
