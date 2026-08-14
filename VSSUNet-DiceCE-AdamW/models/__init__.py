"""
【模型模块】
VSS-UNet: 基于 Vision State Space 的轻量级 3D 医学图像分割网络
参考 Mamba_UNet_Version/model.py 实现

核心设计:
- VSS (Vision State Space) 模块: 门控机制 + 深度可分离卷积
- 轻量级设计，参数量约 0.2M
- 注意：此模型使用 VSSBlock（卷积近似），而非真正的 Mamba
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VSSBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.channels = channels

        self.dwconv = nn.Conv3d(
            channels, channels,
            kernel_size=3, padding=1, groups=channels
        )

        self.norm = nn.BatchNorm3d(channels)
        self.act = nn.SiLU()

        self.gate_conv = nn.Conv3d(channels, channels * 2, kernel_size=1)

        self.output_conv = nn.Conv3d(channels, channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = x

        x = self.dwconv(x)
        x = self.norm(x)
        x = self.act(x)

        gate = self.gate_conv(x)
        g1, g2 = gate.chunk(2, dim=1)
        g1 = torch.sigmoid(g1)
        g2 = torch.sigmoid(g2)

        out = self.output_conv(x)
        out = out * g1 + x * (1 - g2)

        out = out + residual
        out = self.dropout(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.vss = VSSBlock(out_channels, dropout=dropout)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.vss(x)
        skip = x
        x = self.pool(x)
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)
        self.vss = VSSBlock(out_channels, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.vss(x)
        return x


class VSSUNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, base_filters=8, dropout=0.1):
        super().__init__()

        f = base_filters

        self.input_proj = ConvBlock(in_channels, f)

        self.enc1 = EncoderBlock(f, f * 2, dropout)
        self.enc2 = EncoderBlock(f * 2, f * 4, dropout)

        self.bottleneck = nn.Sequential(
            ConvBlock(f * 4, f * 8),
            VSSBlock(f * 8, dropout=dropout)
        )

        self.dec2 = DecoderBlock(f * 8, f * 4, f * 4, dropout)
        self.dec1 = DecoderBlock(f * 4, f * 2, f * 2, dropout)

        self.output_proj = nn.Conv3d(f * 2, out_channels, kernel_size=1)

    def forward(self, x):
        x0 = self.input_proj(x)

        x1, s1 = self.enc1(x0)
        x2, s2 = self.enc2(x1)

        b = self.bottleneck(x2)

        d2 = self.dec2(b, s2)
        d1 = self.dec1(d2, s1)

        return self.output_proj(d1)


def get_model(model_name="vss_unet", in_channels=4, out_channels=4, device="cuda", use_deep_supervision=False):
    if model_name == "vss_unet":
        model = VSSUNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=8,
            dropout=0.1
        ).to(device)
    else:
        raise ValueError(f"未知模型名称: {model_name}")

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "get_model",
    "VSSUNet3D",
    "VSSBlock",
    "ConvBlock",
    "EncoderBlock",
    "DecoderBlock",
    "count_parameters",
]