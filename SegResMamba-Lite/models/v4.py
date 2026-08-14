"""
SegResMamba-Lite V4 - SegResNet + Mamba 双优势深度融合版

网络调研结论：
- SegResNet 优势：残差连接稳定、局部特征提取优秀、训练收敛快
- Mamba 优势：长程依赖建模能力强、计算效率高、适合序列/3D数据
- 双向 Mamba 优势：同时捕获前向和后向依赖，信息更完整
- 深度融合策略：两者优势互补，SegResNet 处理局部，Mamba 处理全局

设计理念：
1. 归一化：InstanceNorm（同 V2/V3，不改动，保证训练速度）
2. init_filters = 20（确保参数量<1.5M）
3. 深度双向 Mamba 融合：
   - Encoder：3个尺度都加双向 Mamba（32³, 16³, 8³）
   - Bottleneck：三重双向 Mamba（深度全局建模）
   - Decoder：2个尺度都加双向 Mamba（16³, 32³，优化边界重建）
4. SE 通道注意力：关键层使用（增强特征表达）
5. 核心：充分融合 SegResNet 的残差局部优势 + Mamba 的双向长程优势
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
    """轻量级 SE 通道注意力模块"""
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
    """轻量级残差块（V4：可选 SE）"""
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
    双向 Mamba 层 - V4 核心（网络调研最佳实践）
    
    设计原理：
    - 正向 Mamba：捕获前向序列依赖（如从脑顶到脑底）
    - 反向 Mamba：捕获反向序列依赖（如从脑底到脑顶）
    - 双向融合：信息更完整，长程依赖建模更强
    - 残差连接：保持梯度流，训练更稳定（结合SegResNet的优势）
    
    参考文献思想：
    - SegMamba：双向Mamba在医学图像分割中的有效性
    - U-Mamba：Mamba与U-Net架构的融合策略
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
        
        # 双向融合 + 残差
        out = x_forward + x_backward
        
        return out


class SegResMambaLiteV4(nn.Module):
    """
    SegResMamba-Lite V4 - SegResNet + Mamba 双优势深度融合版（网络调研最佳架构）
    
    Architecture（充分融合两者优势）：
    - Stem: 4 → init_filters (20，确保参数量<1.5M)
    - Encoder（SegResNet局部优势 + Mamba长程优势）：
      - 64³ → 32³: LightResBlock（纯SegResNet，保持轻量和局部特征）
      - 32³ → 16³: LightResBlock + BiMamba（局部+长程融合）
      - 16³ → 8³: LightResBlock + BiMamba（局部+长程融合）
    - Bottleneck（深度全局建模）：
      - 8³: 三重 BiMamba + SE（最强长程依赖建模）
    - Decoder（边界重建优化）：
      - 8³ → 16³: LightResBlock + BiMamba + SE（细节恢复）
      - 16³ → 32³: LightResBlock + BiMamba + SE（边界优化）
      - 32³ → 64³: LightResBlock（纯SegResNet，最终细化）
    - 归一化：InstanceNorm（同 V2/V3，不改动，保证训练速度）
    
    双优势融合策略：
    - SegResNet LightResBlock：残差连接稳定训练，局部卷积提取精细特征
    - 双向 Mamba：捕获跨尺度长程依赖，全局上下文信息完整
    - SE 注意力：关键层增强通道特征表达
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
        
        # Encoder（更深度的 Mamba 融合，无 SE 以控制参数量）
        # 64³ → 32³：LightResBlock（无 SE，无 Mamba）
        self.enc1 = LightResBlock(init_filters, init_filters, use_se=False, use_group_norm=use_group_norm)
        self.down1 = nn.Conv3d(init_filters, init_filters*2, 2, stride=2)
        
        # 32³ → 16³：LightResBlock + BiMamba（无 SE）
        self.enc2 = LightResBlock(init_filters*2, init_filters*2, use_se=False, use_group_norm=use_group_norm)
        self.mamba_32 = BiMambaLayer(init_filters*2, d_state, d_conv, expand, use_mamba2)
        self.down2 = nn.Conv3d(init_filters*2, init_filters*4, 2, stride=2)
        
        # 16³ → 8³：LightResBlock + BiMamba（无 SE）
        self.enc3 = LightResBlock(init_filters*4, init_filters*4, use_se=False, use_group_norm=use_group_norm)
        self.mamba_16 = BiMambaLayer(init_filters*4, d_state, d_conv, expand, use_mamba2)
        self.down3 = nn.Conv3d(init_filters*4, init_filters*4, 2, stride=2)
        
        # Bottleneck：三重 BiMamba（深度全局建模）
        self.mamba_8a = BiMambaLayer(init_filters*4, d_state, d_conv, expand, use_mamba2)
        self.mamba_8b = BiMambaLayer(init_filters*4, d_state, d_conv, expand, use_mamba2)
        self.mamba_8c = BiMambaLayer(init_filters*4, d_state, d_conv, expand, use_mamba2)
        self.bottleneck_se = SELayer(init_filters*4, reduction=4) if use_se else None
        
        # Decoder（两个尺度都加 Mamba + SE，优化边界重建）
        # 8³ → 16³：LightResBlock + BiMamba + SE
        self.up2 = nn.ConvTranspose3d(init_filters*4, init_filters*4, 2, stride=2)
        self.dec2 = LightResBlock(init_filters*8, init_filters*4, use_se=use_se, use_group_norm=use_group_norm)
        if use_decoder_mamba:
            self.mamba_dec2 = BiMambaLayer(init_filters*4, d_state, d_conv, expand, use_mamba2)
        else:
            self.mamba_dec2 = None
        
        # 16³ → 32³：LightResBlock + BiMamba + SE
        self.up1 = nn.ConvTranspose3d(init_filters*4, init_filters*2, 2, stride=2)
        self.dec1 = LightResBlock(init_filters*4, init_filters*2, use_se=use_se, use_group_norm=use_group_norm)
        if use_decoder_mamba:
            self.mamba_dec1 = BiMambaLayer(init_filters*2, d_state, d_conv, expand, use_mamba2)
        else:
            self.mamba_dec1 = None
        
        # 32³ → 64³：LightResBlock（无 SE，无 Mamba）
        self.up0 = nn.ConvTranspose3d(init_filters*2, init_filters, 2, stride=2)
        self.dec0 = LightResBlock(init_filters*2, init_filters, use_se=False, use_group_norm=use_group_norm)
        
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
        
        # Encoder（更深度的 Mamba 融合，无 SE）
        x1 = self.enc1(x0)
        x1_down = self.down1(x1)
        
        x2 = self.enc2(x1_down)
        x2_mamba = self.mamba_32(x2) + x2
        x2_down = self.down2(x2_mamba)
        
        x3 = self.enc3(x2_down)
        x3_mamba = self.mamba_16(x3) + x3
        x3_down = self.down3(x3_mamba)
        
        # Bottleneck（三重 Mamba + SE，深度全局建模）
        x_bot = self.mamba_8a(x3_down)
        x_bot = self.mamba_8b(x_bot)
        x_bot = self.mamba_8c(x_bot)
        if self.bottleneck_se is not None:
            x_bot = self.bottleneck_se(x_bot)
        
        # Decoder（两个尺度都加 Mamba，优化边界重建）
        x2_up = self.up2(x_bot)
        x2_skip = torch.cat([x2_up, x3_mamba], dim=1)
        x2_dec = self.dec2(x2_skip)
        if self.use_decoder_mamba and self.mamba_dec2 is not None:
            x2_dec = self.mamba_dec2(x2_dec) + x2_dec
        
        x1_up = self.up1(x2_dec)
        x1_skip = torch.cat([x1_up, x2_mamba], dim=1)
        x1_dec = self.dec1(x1_skip)
        if self.use_decoder_mamba and self.mamba_dec1 is not None:
            x1_dec = self.mamba_dec1(x1_dec) + x1_dec
        
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
    获取 V4 模型
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param init_filters: 初始滤波器数（默认 None=自动选 20）
    :param d_state: Mamba 状态维度
    :param expand: Mamba 扩展因子
    :return: 模型实例
    """
    if init_filters is None:
        # V4 默认 init_filters=20（确保参数量<1.5M）
        init_filters = 20
    
    model = SegResMambaLiteV4(
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,
        d_state=d_state,
        expand=expand,
        **kwargs
    )
    return model
