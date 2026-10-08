"""
【LightSegMamba 模型模块 V3】

核心模块：
- Stem: 7×7×7 卷积（步长2）提取初始特征
- DASPPMamba: 深度空洞空间金字塔池化 + Tri-Oriented Mamba 组合块
- LightSegMamba3D: 编码器（Stem + 2×DASPPMamba）→ 解码器（U-Net 转置卷积 + 跳跃连接）

参考论文：
- Kim et al., Mathematics 2025, "Lightweight Mamba Model for 3D Tumor
  Segmentation in Automated Breast Ultrasounds"

V3 默认 base_channels=32（约 1.4M 参数，实测 1,369,604），通过 CLI 可调。
"""

from .lightsegmamba import (
    get_model,
    LightSegMamba3D,
    Stem,
    DASPP,
    TriOrientedMamba,
    DASPPMamba,
    DecoderBlock,
    DWSConv3d,
    count_parameters,
)

__all__ = [
    "get_model",
    "LightSegMamba3D",
    "Stem",
    "DASPP",
    "TriOrientedMamba",
    "DASPPMamba",
    "DecoderBlock",
    "DWSConv3d",
    "count_parameters",
]
