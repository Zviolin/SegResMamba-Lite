"""
SegResMamba-Lite V3 - 优化版（充分利用Mamba）
设计理念：
1. 基础架构同 V2（稳定）
2. 归一化：InstanceNorm（同 V2，不改动，保证速度）
3. 补充优化：
   - SE 通道注意力（轻量有效）
   - 解码器加一层 Mamba
   - Bottleneck 双重 Mamba
   - init_filters 从 22 降到 20（省参数量给上面的改进）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.models.mamba import get_mamba_layer


def _get_norm_layer(num_channels, use_group_norm=False):
    """获取归一化层（默认 InstanceNorm，可选 GroupNorm）"""
    if use_group_norm:
        if num_channels % 8 == 0:
            return nn.GroupNorm(8, num_channels)
        elif num_channels % 4 == 0:
            return nn.GroupNorm(4, num_channels)
        elif num_channels % 2 == 0:
            return nn.GroupNorm(2, num_channels)
        else:
            return nn.InstanceNorm3d(num_channels)
    else:
        return nn.InstanceNorm3d(num_channels)


class SELayer(nn.Module):
    """轻量级 SE 通道注意力模块（可选）"""
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class LightResBlock(nn.Module):
    """轻量级残差块（默认同 V2，无 SE）"""
    def __init__(self, in_ch, out_ch, stride=1, use_se=False, use_group_norm=False):
        super().__init__()
        mid_ch = min(in_ch, out_ch)
        self.conv1 = nn.Conv3d(in_ch, mid_ch, 3, padding=1, stride=stride)
        self.norm1 = _get_norm_layer(mid_ch, use_group_norm=use_group_norm)
        self.conv2 = nn.Conv3d(mid_ch, out_ch, 3, padding=1)
        self.norm2 = _get_norm_layer(out_ch, use_group_norm=use_group_norm)
        self.shortcut = nn.Conv3d(in_ch, out_ch, 1, stride=stride) if stride != 1 or in_ch != out_ch else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.use_se = use_se
        if use_se:
            self.se = SELayer(out_ch, reduction=4)
        else:
            self.se = None
    
    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        out = h + self.shortcut(x)
        if self.se is not None:
            out = self.se(out)
        return self.act(out)


class BiMambaLayer(nn.Module):
    """
    双向 Mamba 层 - V3 核心
    正向处理 + 反向处理
    """
    def __init__(self, dim, d_state=8, d_conv=2, expand=2, use_mamba2=True):
        super().__init__()
        self.dim = dim
        self.mamba = get_mamba_layer(
            dim=dim, d_state=d_state, d_conv=d_conv, expand=expand, use_mamba2=use_mamba2
        )
    
    def forward(self, x):
        # 正向 Mamba
        x_forward = self.mamba(x)
        
        # 反向 Mamba
        x_rev = torch.flip(x, dims=[2, 3, 4])
        x_backward_rev = self.mamba(x_rev)
        x_backward = torch.flip(x_backward_rev, dims=[2, 3, 4])
        
        # 双向融合
        out = x_forward + x_backward
        
        return out


class SegResMambaLiteV3(nn.Module):
    """
    SegResMamba-Lite V3 - 优化版（充分利用Mamba）
    
    Architecture：
    - Stem: 4 → init_filters
    - Encoder: LightResBlock + SE + BiMamba
    - Bottleneck: BiMamba × 2（双重，补充优化）
    - Decoder: LightResBlock + SE + BiMamba（额外加的）
    - 归一化：InstanceNorm（同 V2，不改动，保证速度）
    """
    def __init__(self, in_channels=4, out_channels=4, init_filters=20, 
                 d_state=8, d_conv=2, expand=2, use_mamba2=True,
                 use_deep_supervision=False, use_attention=False, 
                 use_decoder_mamba=True, use_se=True, use_group_norm=False, **kwargs):
        super().__init__()
        self.use_deep_supervision = use_deep_supervision
        self.use_decoder_mamba = use_decoder_mamba
        
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, init_filters, 3, padding=1),
            _get_norm_layer(init_filters, use_group_norm=use_group_norm),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        # 64³ → 32³：LightResBlock
        self.enc1 = LightResBlock(init_filters, init_filters, use_se=use_se, use_group_norm=use_group_norm)
        self.down1 = nn.Conv3d(init_filters, init_filters*2, 2, stride=2)
        
        # 32³ → 16³：LightResBlock + BiMamba
        self.enc2 = LightResBlock(init_filters*2, init_filters*2, use_se=use_se, use_group_norm=use_group_norm)
        self.mamba_32 = BiMambaLayer(init_filters*2, d_state, d_conv, expand, use_mamba2)
        self.down2 = nn.Conv3d(init_filters*2, init_filters*4, 2, stride=2)
        
        # 16³ → 8³：LightResBlock + BiMamba
        self.enc3 = LightResBlock(init_filters*4, init_filters*4, use_se=use_se, use_group_norm=use_group_norm)
        self.mamba_16 = BiMambaLayer(init_filters*4, d_state, d_conv, expand, use_mamba2)
        self.down3 = nn.Conv3d(init_filters*4, init_filters*4, 2, stride=2)
        
        # 8³：双重 BiMamba（补充优化）
        self.mamba_8a = BiMambaLayer(init_filters*4, d_state, d_conv, expand, use_mamba2)
        self.mamba_8b = BiMambaLayer(init_filters*4, d_state, d_conv, expand, use_mamba2)
        
        # Decoder
        # 8³ → 16³（跳跃连接）
        self.up2 = nn.ConvTranspose3d(init_filters*4, init_filters*4, 2, stride=2)
        self.dec2 = LightResBlock(init_filters*8, init_filters*4, use_se=use_se, use_group_norm=use_group_norm)
        if use_decoder_mamba:
            self.mamba_dec2 = BiMambaLayer(init_filters*4, d_state, d_conv, expand, use_mamba2)
        else:
            self.mamba_dec2 = None
        
        # 16³ → 32³（跳跃连接）
        self.up1 = nn.ConvTranspose3d(init_filters*4, init_filters*2, 2, stride=2)
        self.dec1 = LightResBlock(init_filters*4, init_filters*2, use_se=use_se, use_group_norm=use_group_norm)
        
        # 32³ → 64³（跳跃连接）
        self.up0 = nn.ConvTranspose3d(init_filters*2, init_filters, 2, stride=2)
        self.dec0 = LightResBlock(init_filters*2, init_filters, use_se=use_se, use_group_norm=use_group_norm)
        
        # Output heads
        self.out = nn.Conv3d(init_filters, out_channels, 1)
        
        # Deep supervision heads
        if use_deep_supervision:
            self.out2 = nn.Conv3d(init_filters*4, out_channels, 1)
            self.out1 = nn.Conv3d(init_filters*2, out_channels, 1)
        else:
            self.out2 = None
            self.out1 = None
    
    def forward(self, x):
        x0 = self.stem(x)
        
        # Encoder
        x1 = self.enc1(x0)
        x1_down = self.down1(x1)
        
        x2 = self.enc2(x1_down)
        x2_mamba = self.mamba_32(x2) + x2
        x2_down = self.down2(x2_mamba)
        
        x3 = self.enc3(x2_down)
        x3_mamba = self.mamba_16(x3) + x3
        x3_down = self.down3(x3_mamba)
        
        # Bottleneck（双重）
        x_bot = self.mamba_8a(x3_down)
        x_bot = self.mamba_8b(x_bot)
        
        # Decoder (同 V2 的正确跳跃连接)
        x2_up = self.up2(x_bot)
        x2_skip = torch.cat([x2_up, x3_mamba], dim=1)
        x2_dec = self.dec2(x2_skip)
        if self.use_decoder_mamba and self.mamba_dec2 is not None:
            x2_dec = self.mamba_dec2(x2_dec) + x2_dec
        
        x1_up = self.up1(x2_dec)
        x1_skip = torch.cat([x1_up, x2_mamba], dim=1)
        x1_dec = self.dec1(x1_skip)
        
        x0_up = self.up0(x1_dec)
        x0_skip = torch.cat([x0_up, x0], dim=1)
        x0_dec = self.dec0(x0_skip)
        
        out = self.out(x0_dec)
        
        if self.use_deep_supervision and self.training:
            out2 = self.out2(x2_dec)
            out1 = self.out1(x1_dec)
            return out, out1, out2
        else:
            return out


def get_model(in_channels=4, out_channels=4, init_filters=None, d_state=8, expand=2, **kwargs):
    """
    获取 V3 模型
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param init_filters: 初始滤波器数（默认 None=自动选 20）
    :param d_state: Mamba 状态维度
    :param expand: Mamba 扩展因子
    :return: 模型实例
    """
    if init_filters is None:
        # V3 默认 init_filters=20（省参数量给双重Mamba）
        init_filters = 20
    
    model = SegResMambaLiteV3(
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,
        d_state=d_state,
        expand=expand,
        **kwargs
    )
    return model
