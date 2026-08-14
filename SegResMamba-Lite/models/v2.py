"""
SegResMamba-Lite V2 - 优化版
设计理念：
1. 基础架构同 V1（稳定）
2. 在编码器和解码器中都加入 Mamba（深度融合）
3. 保持轻量 - init_filters 可调，目标 <1.5M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.models.mamba import get_mamba_layer


class LightResBlock(nn.Module):
    """轻量级残差块"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        mid_ch = min(in_ch, out_ch)
        self.conv1 = nn.Conv3d(in_ch, mid_ch, 3, padding=1, stride=stride)
        self.norm1 = nn.InstanceNorm3d(mid_ch)
        self.conv2 = nn.Conv3d(mid_ch, out_ch, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_ch)
        self.shortcut = nn.Conv3d(in_ch, out_ch, 1, stride=stride) if stride != 1 or in_ch != out_ch else nn.Identity()
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.shortcut(x))


class BiMambaLayer(nn.Module):
    """
    双向 Mamba 层 - V2 核心
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


class SegResMambaLiteV2(nn.Module):
    """
    SegResMamba-Lite V2 - 优化版（保持轻量）
    
    Architecture:
    - Stem: 4 → init_filters
    - Encoder: LightResBlock + BiMamba（深度融合）
    - Decoder: LightResBlock + 跳跃连接
    - Mamba 在多个尺度充分利用
    - 总参数量：init_filters=22 时 ~1.4M
    """
    def __init__(self, in_channels=4, out_channels=4, init_filters=22, 
                 d_state=8, d_conv=2, expand=2, use_mamba2=True,
                 use_deep_supervision=False):
        super().__init__()
        self.use_deep_supervision = use_deep_supervision
        
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, init_filters, 3, padding=1),
            nn.InstanceNorm3d(init_filters),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        # 64³ → 32³：LightResBlock
        self.enc1 = LightResBlock(init_filters, init_filters)
        self.down1 = nn.Conv3d(init_filters, init_filters*2, 2, stride=2)
        
        # 32³ → 16³：LightResBlock + BiMamba（深度融合）
        self.enc2 = LightResBlock(init_filters*2, init_filters*2)
        self.mamba_32 = BiMambaLayer(init_filters*2, d_state, d_conv, expand, use_mamba2)
        self.down2 = nn.Conv3d(init_filters*2, init_filters*4, 2, stride=2)
        
        # 16³ → 8³：LightResBlock + BiMamba（深度融合）
        self.enc3 = LightResBlock(init_filters*4, init_filters*4)
        self.mamba_16 = BiMambaLayer(init_filters*4, d_state, d_conv, expand, use_mamba2)
        self.down3 = nn.Conv3d(init_filters*4, init_filters*4, 2, stride=2)
        
        # 8³：BiMamba（最深层）
        self.mamba_8 = BiMambaLayer(init_filters*4, d_state, d_conv, expand, use_mamba2)
        
        # Decoder
        # 8³ → 16³（跳跃连接）
        self.up2 = nn.ConvTranspose3d(init_filters*4, init_filters*4, 2, stride=2)
        self.dec2 = LightResBlock(init_filters*8, init_filters*4)
        
        # 16³ → 32³（跳跃连接）
        self.up1 = nn.ConvTranspose3d(init_filters*4, init_filters*2, 2, stride=2)
        self.dec1 = LightResBlock(init_filters*4, init_filters*2)
        
        # 32³ → 64³（跳跃连接）
        self.up0 = nn.ConvTranspose3d(init_filters*2, init_filters, 2, stride=2)
        self.dec0 = LightResBlock(init_filters*2, init_filters)
        
        # Output heads
        self.out = nn.Conv3d(init_filters, out_channels, 1)
        
        # Deep supervision heads
        if use_deep_supervision:
            self.ds1 = nn.Conv3d(init_filters*4, out_channels, 1)
            self.ds2 = nn.Conv3d(init_filters*2, out_channels, 1)
    
    def forward(self, x):
        # Stem
        x0 = self.stem(x)
        
        # Encoder
        # 64³ → 32³
        x1 = self.enc1(x0)
        x1_down = self.down1(x1)
        
        # 32³ → 16³（深度融合：LightResBlock + BiMamba）
        x2 = self.enc2(x1_down)
        x2_mamba = self.mamba_32(x2)
        x2_down = self.down2(x2_mamba)
        
        # 16³ → 8³（深度融合：LightResBlock + BiMamba）
        x3 = self.enc3(x2_down)
        x3_mamba = self.mamba_16(x3)
        x3_down = self.down3(x3_mamba)
        
        # 8³：BiMamba（最深层）
        x3_bottleneck = self.mamba_8(x3_down)
        
        # Decoder
        # 8³ → 16³ + 跳跃连接
        x2_up = self.up2(x3_bottleneck)
        x2_skip_cat = torch.cat([x2_up, x3_mamba], dim=1)
        x2_dec = self.dec2(x2_skip_cat)
        
        # 16³ → 32³ + 跳跃连接
        x1_up = self.up1(x2_dec)
        x1_skip_cat = torch.cat([x1_up, x1_down], dim=1)
        x1_dec = self.dec1(x1_skip_cat)
        
        # 32³ → 64³ + 跳跃连接
        x0_up = self.up0(x1_dec)
        x0_skip_cat = torch.cat([x0_up, x0], dim=1)
        x0_dec = self.dec0(x0_skip_cat)
        
        # Final output
        out = self.out(x0_dec)
        
        if self.use_deep_supervision and self.training:
            ds1 = self.ds1(x2_dec)
            ds2 = self.ds2(x1_dec)
            return (out, ds1, ds2)
        
        return out


def get_model(in_channels=4, out_channels=4, init_filters=22, 
              d_state=8, d_conv=2, expand=2, use_mamba2=True,
              use_attention=True, use_deep_supervision=False, device="cuda"):
    """
    获取 V2 模型（优化版）
    
    关键改进（相比 V1）：
    - 在 32³ 和 16³ 尺度都加入 BiMamba（深度融合）
    - 保持轻量架构，参数量可控
    
    Args:
        init_filters: 推荐 18-24，22 时 ~1.4M（<1.5M）
        d_state: Mamba 状态维度，推荐 4-8
        expand: Mamba 扩展因子，推荐 1-2
    """
    model = SegResMambaLiteV2(
        in_channels, out_channels, init_filters,
        d_state, d_conv, expand, use_mamba2,
        use_deep_supervision
    ).to(device)
    return model