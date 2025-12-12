"""
UNet-SwinTransformer模型用于ResShift反向过程中预测X0
基于ResShift项目的UNetModelSwin实现
"""

from abc import abstractmethod
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .basic_ops import (
    linear,
    conv_nd,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from .swin_transformer import BasicLayer


class TimestepBlock(nn.Module):
    """
    任何forward()接受时间步嵌入作为第二个参数的模块
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        将模块应用于给定时间步嵌入的x
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    一个顺序模块，将时间步嵌入传递给支持它的子模块作为额外输入
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    带有可选卷积的上采样层
    :param channels: 输入和输出的通道数
    :param use_conv: 是否应用卷积
    :param dims: 信号是1D、2D还是3D
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    带有可选卷积的下采样层
    :param channels: 输入和输出的通道数
    :param use_conv: 是否应用卷积
    :param dims: 信号是1D、2D还是3D
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    可以选择性地改变通道数的残差块
    :param channels: 输入通道数
    :param emb_channels: 时间步嵌入通道数
    :param dropout: dropout率
    :param out_channels: 如果指定，输出通道数
    :param use_conv: 如果为True且指定了out_channels，使用空间卷积而不是1x1卷积
    :param dims: 信号是1D、2D还是3D
    :param up: 如果为True，使用此块进行上采样
    :param down: 如果为True，使用此块进行下采样
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class UNetModelSwin(nn.Module):
    """
    完整的UNet-SwinTransformer模型，用于ResShift反向过程中预测X0
    
    :param image_size: 输入图像大小
    :param in_channels: 输入张量的通道数
    :param model_channels: 模型的基础通道数
    :param out_channels: 输出张量的通道数
    :param num_res_blocks: 每个下采样的残差块数量
    :param attention_resolutions: 应用注意力的下采样率集合
    :param dropout: dropout概率
    :param channel_mult: 每个UNet级别的通道乘数
    :param conv_resample: 如果为True，使用学习的卷积进行上采样和下采样
    :param dims: 信号是1D、2D还是3D
    :param use_fp16: 是否使用float16精度
    :param num_heads: 每个注意力层的注意力头数
    :param num_head_channels: 如果指定，忽略num_heads，使用固定的每个注意力头的通道宽度
    :param use_scale_shift_norm: 使用类似FiLM的条件机制
    :param resblock_updown: 使用残差块进行上/下采样
    :param swin_depth: Swin Transformer的深度
    :param swin_embed_dim: Swin Transformer的嵌入维度
    :param window_size: Swin Transformer的窗口大小
    :param mlp_ratio: MLP隐藏维度与嵌入维度的比率
    :param patch_norm: Swin Transformer中的patch归一化
    :param cond_lq: 是否条件化低质量图像
    :param cond_mask: 是否条件化mask
    :param lq_size: 低质量图像大小
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        swin_depth=2,
        swin_embed_dim=96,
        window_size=8,
        mlp_ratio=2.0,
        patch_norm=False,
        cond_lq=True,
        cond_mask=False,
        lq_size=256,
    ):
        super().__init__()

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks,] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        if num_heads == -1:
            assert swin_embed_dim % num_head_channels == 0 and num_head_channels > 0
        self.num_res_blocks = num_res_blocks

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.cond_lq = cond_lq
        self.cond_mask = cond_mask

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # 特征提取器用于处理低质量图像
        if cond_lq and lq_size == image_size:
            self.feature_extractor = nn.Identity()
            base_chn = 4 if cond_mask else 3
        else:
            feature_extractor = []
            feature_chn = 4 if cond_mask else 3
            base_chn = 16
            for ii in range(int(math.log(lq_size / image_size) / math.log(2))):
                feature_extractor.append(nn.Conv2d(feature_chn, base_chn, 3, 1, 1))
                feature_extractor.append(nn.SiLU())
                feature_extractor.append(Downsample(base_chn, True, out_channels=base_chn*2))
                base_chn *= 2
                feature_chn = base_chn
            self.feature_extractor = nn.Sequential(*feature_extractor)

        ch = input_ch = int(channel_mult[0] * model_channels)
        in_channels += base_chn
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        ds = image_size
        
        # 构建输入块（编码器）
        for level, mult in enumerate(channel_mult):
            for jj in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                # 在每个级别的第一个块添加Swin Transformer
                if ds in attention_resolutions and jj==0:
                    layers.append(
                        BasicLayer(
                                in_chans=ch,
                                embed_dim=swin_embed_dim,
                                num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                                window_size=window_size,
                                depth=swin_depth,
                                img_size=ds,
                                patch_size=1,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=True,
                                qk_scale=None,
                                drop=dropout,
                                attn_drop=0.,
                                drop_path=0.,
                                use_checkpoint=False,
                                norm_layer=normalization,
                                patch_norm=patch_norm,
                                 )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds //= 2

        # 中间块
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            BasicLayer(
                    in_chans=ch,
                    embed_dim=swin_embed_dim,
                    num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                    window_size=window_size,
                    depth=swin_depth,
                    img_size=ds,
                    patch_size=1,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=dropout,
                    attn_drop=0.,
                    drop_path=0.,
                    use_checkpoint=False,
                    norm_layer=normalization,
                    patch_norm=patch_norm,
                     ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # 构建输出块（解码器）
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                # 在每个级别的第一个块添加Swin Transformer
                if ds in attention_resolutions and i==0:
                    layers.append(
                        BasicLayer(
                                in_chans=ch,
                                embed_dim=swin_embed_dim,
                                num_heads=num_heads if num_head_channels == -1 else swin_embed_dim // num_head_channels,
                                window_size=window_size,
                                depth=swin_depth,
                                img_size=ds,
                                patch_size=1,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=True,
                                qk_scale=None,
                                drop=dropout,
                                attn_drop=0.,
                                drop_path=0.,
                                use_checkpoint=False,
                                norm_layer=normalization,
                                patch_norm=patch_norm,
                                 )
                    )
                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds *= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # 输出层
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, input_ch, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, lq=None, mask=None):
        """
        将模型应用于输入批次
        :param x: [N x C x ...] 输入张量
        :param timesteps: 1-D时间步批次
        :param lq: [N x C x ...] 低质量图像张量
        :param mask: [N x 1 x ...] mask张量
        :return: [N x C x ...] 输出张量
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).type(self.dtype)

        # 处理低质量图像条件
        if lq is not None:
            assert self.cond_lq
            if mask is not None:
                assert self.cond_mask
                lq = th.cat([lq, mask], dim=1)
            lq = self.feature_extractor(lq.type(self.dtype))
            x = th.cat([x, lq], dim=1)

        # 前向传播
        h = x.type(self.dtype)
        for ii, module in enumerate(self.input_blocks):
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        out = self.out(h)
        return out

    def convert_to_fp16(self):
        """
        将模型的主体转换为float16
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.feature_extractor.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        将模型的主体转换为float32
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
