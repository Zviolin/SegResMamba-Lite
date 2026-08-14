"""
SegResMamba-Lite V1 - 主版本
双向 Mamba + 跳跃连接 + Deep Supervision
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
    """双向 Mamba 层"""
    def __init__(self, dim, d_state=8, d_conv=2, expand=2, use_mamba2=True):
        super().__init__()
        self.dim = dim
        self.mamba = get_mamba_layer(
            dim=dim, d_state=d_state, d_conv=d_conv, expand=expand, use_mamba2=use_mamba2
        )
    
    def forward(self, x):
        # 前向 Mamba
        x_forward = self.mamba(x)
        
        # 反向 Mamba：反转空间维度
        x_rev = torch.flip(x, dims=[2, 3, 4])
        x_backward_rev = self.mamba(x_rev)
        x_backward = torch.flip(x_backward_rev, dims=[2, 3, 4])
        
        # 双向特征融合
        out = x_forward + x_backward
        return out


class SegResMambaLiteV1(nn.Module):
    """
    SegResMamba-Lite V1 - 主版本
    
    Architecture:
    - Stem: 4→26
    - Encoder: 26→52→104（带跳跃连接）
    - BiMamba @16³ (4096 tokens)
    - BiMamba @8³ (512 tokens)
    - Decoder: 104→52→26（带跳跃连接融合）
    - 总参数量：~1.4M (<1.5M)
    """
    def __init__(self, in_channels=4, out_channels=4, init_filters=26, use_deep_supervision=False):
        super().__init__()
        self.use_deep_supervision = use_deep_supervision
        
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, init_filters, 3, padding=1),
            nn.InstanceNorm3d(init_filters),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        # 64³ → 32³
        self.enc1 = LightResBlock(init_filters, init_filters)
        self.down1 = nn.Conv3d(init_filters, init_filters*2, 2, stride=2)
        
        # 32³ → 16³
        self.enc2 = LightResBlock(init_filters*2, init_filters*2)
        self.down2 = nn.Conv3d(init_filters*2, init_filters*4, 2, stride=2)
        
        # 16³: 双向 Mamba
        self.mamba_16 = BiMambaLayer(
            dim=init_filters*4,
            d_state=8,
            d_conv=2,
            expand=2,
            use_mamba2=True
        )
        
        # 16³ → 8³
        self.down3 = nn.Conv3d(init_filters*4, init_filters*4, 2, stride=2)
        
        # 8³: 双向 Mamba
        self.mamba_8 = BiMambaLayer(
            dim=init_filters*4,
            d_state=8,
            d_conv=2,
            expand=2,
            use_mamba2=True
        )
        
        # Decoder
        # 8³ → 16³（与 encoder 16³ 跳跃连接）
        self.up2 = nn.ConvTranspose3d(init_filters*4, init_filters*4, 2, stride=2)
        self.dec2 = LightResBlock(init_filters*8, init_filters*4)
        
        # 16³ → 32³（与 encoder 32³ 跳跃连接）
        self.up1 = nn.ConvTranspose3d(init_filters*4, init_filters*2, 2, stride=2)
        self.dec1 = LightResBlock(init_filters*4, init_filters*2)
        
        # 32³ → 64³（与 stem 跳跃连接）
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
        x1 = self.enc1(x0)
        x1_down = self.down1(x1)
        
        x2 = self.enc2(x1_down)
        x2_down = self.down2(x2)
        
        # 16³: 双向 Mamba
        x2_mamba = self.mamba_16(x2_down)
        
        # 8³
        x3 = self.down3(x2_mamba)
        
        # 8³: 双向 Mamba
        x3_mamba = self.mamba_8(x3)
        
        # Decoder
        # 8³ → 16³ + 跳跃连接
        x2_up = self.up2(x3_mamba)
        x2_skip = torch.cat([x2_up, x2_mamba], dim=1)
        x2_dec = self.dec2(x2_skip)
        
        # 16³ → 32³ + 跳跃连接
        x1_up = self.up1(x2_dec)
        x1_skip = torch.cat([x1_up, x1_down], dim=1)
        x1_dec = self.dec1(x1_skip)
        
        # 32³ → 64³ + 跳跃连接
        x0_up = self.up0(x1_dec)
        x0_skip = torch.cat([x0_up, x0], dim=1)
        x0_dec = self.dec0(x0_skip)
        
        # Final output
        out = self.out(x0_dec)
        
        if self.use_deep_supervision and self.training:
            ds1 = self.ds1(x2_dec)
            ds2 = self.ds2(x1_dec)
            return (out, ds1, ds2)
        
        return out


def get_model(in_channels=4, out_channels=4, init_filters=26, use_attention=True, use_deep_supervision=False, device="cuda"):
    """获取 V1 模型（主版本）"""
    model = SegResMambaLiteV1(in_channels, out_channels, init_filters, use_deep_supervision=use_deep_supervision).to(device)
    return model
