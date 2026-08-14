"""
SegResMamba-Lite V5 - V2基础 + V3优势深度融合

参数量目标：<1.5M（基于V2的1.42M基础优化）

设计思路：
1. init_filters=20（保持V2的参数量基础）
2. Encoder：3个尺度都用BiMamba（继承V2的多尺度优势）
3. Bottleneck：双重BiMamba + SE（继承V3的双重设计，但精简）
4. Decoder：加入BiMamba（继承V3的边界优化）
5. SE策略：仅Bottleneck和Decoder关键层使用（控制参数量）
6. 归一化：InstanceNorm（保持V2/V3的速度优势）

目标：保持V2的Dice表现，提升HD95到V3水平
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.models.mamba import get_mamba_layer


def _get_norm_layer(num_channels, use_group_norm=False):
    """获取归一化层（默认InstanceNorm，可选GroupNorm）"""
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
    """轻量级SE通道注意力模块"""
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
    """轻量级残差块（V5版本，可选SE）"""
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
    双向Mamba层 - V5核心
    
    设计：正向+反向处理，充分利用长距离依赖
    """
    def __init__(self, dim, d_state=8, d_conv=2, expand=2, use_mamba2=True):
        super().__init__()
        self.dim = dim
        self.mamba = get_mamba_layer(
            dim=dim, d_state=d_state, d_conv=d_conv, expand=expand, use_mamba2=use_mamba2
        )
    
    def forward(self, x):
        x_forward = self.mamba(x)
        x_rev = torch.flip(x, dims=[2, 3, 4])
        x_backward_rev = self.mamba(x_rev)
        x_backward = torch.flip(x_backward_rev, dims=[2, 3, 4])
        out = x_forward + x_backward
        return out


class SegResMambaLiteV5(nn.Module):
    """
    SegResMamba-Lite V5 - V2基础 + V3优势融合
    
    Architecture：
    - Stem: 4 -> 20 (保持V2的轻量)
    - Encoder:
      - 64³->32³: LightResBlock (无SE，无Mamba)
      - 32³->16³: LightResBlock + BiMamba (无SE)
      - 16³->8³: LightResBlock + BiMamba (无SE)
    - Bottleneck: 8³ -> 双重BiMamba + SE (V3的核心设计)
    - Decoder:
      - 8³->16³: LightResBlock + BiMamba + SE
      - 16³->32³: LightResBlock + BiMamba + SE
      - 32³->64³: LightResBlock (无SE，无Mamba)
    """
    def __init__(self, in_channels=4, out_channels=4, init_f