"""
示例：如何使用配置文件加载UNet和VQVAE模型

此脚本演示如何：
1. 从YAML配置文件加载可配置参数
2. 使用预训练权重初始化模型
3. 应用配置参数到模型
"""

import yaml
import torch
from pathlib import Path
from SR.models.unet import UNetModelSwin
from SR.ldm.models.autoencoder import VQModelTorch


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_unet_with_config(config_path, pretrained_path=None):
    """
    使用配置文件创建UNet模型
    
    Args:
        config_path: 配置文件路径
        pretrained_path: 预训练权重路径（可选）
    
    Returns:
        model: UNet模型
        config: 配置字典
    """
    # 加载配置
    config = load_config(config_path)
    
    # 模型结构参数（必须与预训练权重一致）
    # 这些参数应该从预训练权重的元数据中读取，或者硬编码
    model_structure = {
        'image_size': 64,
        'in_channels': 3,
        'model_channels': 160,
        'out_channels': 3,
        'attention_resolutions': [64, 32, 16, 8],
        'channel_mult': [1, 2, 2, 4],
        'num_res_blocks': [2, 2, 2, 2],
        'num_head_channels': 32,
        'use_scale_shift_norm': True,
        'resblock_updown': False,
        'swin_depth': 2,
        'swin_embed_dim': 192,
        'window_size': 8,
        'mlp_ratio': 4,
        'cond_lq': True,
        'lq_size': 64,
    }
    
    # 从配置文件中获取可配置参数
    model_config = {
        **model_structure,
        'dropout': config.get('dropout', 0.0),
        'use_fp16': config.get('use_fp16', False),
        'conv_resample': config.get('conv_resample', True),
        'dims': config.get('dims', 2),
        'patch_norm': config.get('patch_norm', False),
    }
    
    print("创建UNet模型...")
    print(f"  - dropout: {model_config['dropout']}")
    print(f"  - use_fp16: {model_config['use_fp16']}")
    print(f"  - conv_resample: {model_config['conv_resample']}")
    
    # 创建模型
    model = UNetModelSwin(**model_config)
    
    # 加载预训练权重
    if pretrained_path and Path(pretrained_path).exists():
        print(f"\n从 {pretrained_path} 加载预训练权重...")
        ckpt = torch.load(pretrained_path, map_location='cpu')
        
        # 处理state_dict
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        
        # 去除前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith('module._orig_mod.'):
                new_key = key.replace('module._orig_mod.', '')
            elif key.startswith('module.'):
                new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        
        # 加载权重
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        print(f"  - 成功加载: {len(new_state_dict) - len(missing_keys)} 个参数")
        print(f"  - 缺失: {len(missing_keys)} 个参数")
        print(f"  - 多余: {len(unexpected_keys)} 个参数")
    
    return model, config


def create_vqvae_with_config(config_path, pretrained_path=None):
    """
    使用配置文件创建VQVAE模型
    
    Args:
        config_path: 配置文件路径
        pretrained_path: 预训练权重路径（可选）
    
    Returns:
        model: VQVAE模型
        config: 配置字典
    """
    # 加载配置
    config = load_config(config_path)
    
    # 模型结构参数（必须与预训练权重一致）
    ddconfig = {
        'double_z': False,
        'z_channels': 3,
        'resolution': 256,
        'in_channels': 3,
        'out_ch': 3,
        'ch': 128,
        'ch_mult': [1, 2, 4],
        'num_res_blocks': 2,
        'attn_resolutions': [],
        'dropout': 0.0,
        'padding_mode': 'zeros',
    }
    
    # 从配置文件中获取可配置参数
    lora_config = config.get('lora', {})
    quant_config = config.get('quantization', {})
    
    model_config = {
        'ddconfig': ddconfig,
        'n_embed': 8192,
        'embed_dim': 3,
        'rank': lora_config.get('rank', 8),
        'lora_alpha': lora_config.get('alpha', 1.0),
        'lora_tune_decoder': lora_config.get('tune_decoder', False),
        'remap': quant_config.get('remap', None),
        'sane_index_shape': quant_config.get('sane_index_shape', False),
    }
    
    print("创建VQVAE模型...")
    print(f"  - LoRA rank: {model_config['rank']}")
    print(f"  - LoRA alpha: {model_config['lora_alpha']}")
    print(f"  - LoRA tune decoder: {model_config['lora_tune_decoder']}")
    
    # 创建模型
    model = VQModelTorch(**model_config)
    
    # 加载预训练权重
    if pretrained_path and Path(pretrained_path).exists():
        print(f"\n从 {pretrained_path} 加载预训练权重...")
        ckpt = torch.load(pretrained_path, map_location='cpu')
        
        # 处理state_dict
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        
        # 去除前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith('module._orig_mod.'):
                new_key = key.replace('module._orig_mod.', '')
            elif key.startswith('module.'):
                new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        
        # 加载权重
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        print(f"  - 成功加载: {len(new_state_dict) - len(missing_keys)} 个参数")
        print(f"  - 缺失: {len(missing_keys)} 个参数")
        print(f"  - 多余: {len(unexpected_keys)} 个参数")
    
    return model, config


def example_unet():
    """UNet使用示例"""
    print("="*70)
    print("UNet模型加载示例")
    print("="*70)
    
    # 配置文件路径
    config_path = Path(__file__).parent / "configs" / "unet_config.yaml"
    pretrained_path = Path(__file__).parent / "pretrained" / "resshift_realsrx4_s4_v3.pth"
    
    # 创建模型
    model, config = create_unet_with_config(config_path, pretrained_path)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    model = model.to(device)
    model.eval()
    
    # 测试推理
    print(f"\n测试推理...")
    batch_size = 1
    x = torch.randn(batch_size, 3, 64, 64).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    lq = torch.randn(batch_size, 3, 64, 64).to(device)
    
    with torch.no_grad():
        output = model(x, timesteps, lq=lq)
    
    print(f"  - 输入形状: {x.shape}")
    print(f"  - 输出形状: {output.shape}")
    print(f"  - 输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print("\n✓ UNet模型加载和推理成功！")


def example_vqvae():
    """VQVAE使用示例"""
    print("\n" + "="*70)
    print("VQVAE模型加载示例")
    print("="*70)
    
    # 配置文件路径
    config_path = Path(__file__).parent / "configs" / "vqvae_config.yaml"
    pretrained_path = Path(__file__).parent / "pretrained" / "autoencoder_vq_f4.pth"
    
    # 创建模型
    model, config = create_vqvae_with_config(config_path, pretrained_path)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    model = model.to(device)
    model.eval()
    
    # 测试推理
    print(f"\n测试推理...")
    batch_size = 1
    x = torch.randn(batch_size, 3, 256, 256).to(device)
    
    with torch.no_grad():
        # 编码
        latent = model.encode(x)
        print(f"  - 输入形状: {x.shape}")
        print(f"  - 潜在表示形状: {latent.shape}")
        
        # 解码
        reconstructed = model.decode(latent, force_not_quantize=False)
        print(f"  - 重建形状: {reconstructed.shape}")
        
        # 计算重建误差
        mse = torch.mean((x - reconstructed) ** 2).item()
        print(f"  - 重建MSE: {mse:.6f}")
    
    print("\n✓ VQVAE模型加载和推理成功！")


if __name__ == '__main__':
    # 运行示例
    example_unet()
    example_vqvae()
    
    print("\n" + "="*70)
    print("所有示例运行完成！")
    print("="*70)
