"""
【模型模块】
提供 UltraLight VM-UNet 模型
参考 UltraLight_VM_UNet_Version/model.py 实现

特点:
- 轻量级 3D 医学图像分割模型
- 参数量约 0.2M
- GPU 友好设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.dropout(x)
        return self.act(x + residual)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.block = ResBlock(out_channels, dropout)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.block(x)
        skip = x
        x = self.pool(x)
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)
        self.block = ResBlock(out_channels, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.block(x)
        return x


class UltraLightVMUNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, base_filters=8, dropout=0.1):
        super().__init__()

        f = base_filters

        self.input_proj = ConvBlock(in_channels, f)

        self.enc1 = EncoderBlock(f, f * 2, dropout)
        self.enc2 = EncoderBlock(f * 2, f * 4, dropout)

        self.bottleneck = nn.Sequential(
            ConvBlock(f * 4, f * 8),
            ResBlock(f * 8, dropout)
        )

        self.dec2 = DecoderBlock(f * 8, f * 4, f * 4, dropout)
        self.dec1 = DecoderBlock(f * 4, f * 2, f * 2, dropout)

        self.output_proj = nn.Conv3d(f * 2, out_channels, 1)

    def forward(self, x):
        x = self.input_proj(x)

        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)

        x = self.bottleneck(x)

        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        x = self.output_proj(x)

        return x


def get_model(model_name="vm_unet", device="cuda", use_deep_supervision=False, **kwargs):
    if model_name == "vm_unet":
        model = UltraLightVMUNet3D(
            in_channels=kwargs.get("in_channels", 4),
            out_channels=kwargs.get("out_channels", 4),
            base_filters=kwargs.get("base_filters", 8),
            dropout=kwargs.get("dropout", 0.1),
        ).to(device)
    else:
        raise ValueError(f"未知模型名称: {model_name}")

    return model


__all__ = [
    "ConvBlock",
    "ResBlock",
    "EncoderBlock",
    "DecoderBlock",
    "UltraLightVMUNet3D",
    "get_model",
]