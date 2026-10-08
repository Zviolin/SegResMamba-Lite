"""
【LightSegMamba 模型实现 V3】

严格依据论文：
- Kim et al., Mathematics 2025, "Lightweight Mamba Model for 3D Tumor
  Segmentation in Automated Breast Ultrasounds"

适配到 BraTS 2023 GLI 脑肿瘤场景：
- 4 模态输入（T1n, T1c, T2w, T2f），out_channels=4（BG/TC/WT/ET）
- 网络结构：Stem + 2× DASPPMamba + U-Net 风格解码器

V3 相对 V2 改动：
- 默认 base_channels=32（Mamba2 版实测约 1.27M 参数，V2 24/807K 容量翻倍）
- 修复 V2 TriOrientedMamba 中 _flip_last_dim 的注释歧义
- 显式导出 DWSConv3d 便于调试

依赖：
- mamba_ssm（Mamba2/SSD，triton JIT 运行时编译，无需预编译 CUDA 扩展）
- 在已安装上述库的环境（如 mamba_sm120_v3）中运行

与论文的结构偏差注记：
- 论文 Tri-Oriented Mamba 使用 Mamba1（S6 selective scan）
- 本实现改用 Mamba2（SSD）：接口一致（(B, L, C) → (B, L, C)），
  训练更快（块分解映射 Tensor Core），且无需编译 selective_scan_cuda
"""

from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba2
    _HAS_MAMBA_SSM = True
    _IMPORT_ERR = None
except ImportError as _e:
    Mamba2 = None
    _HAS_MAMBA_SSM = False
    _IMPORT_ERR = _e


# ════════════════════════════════════════════════════════════════════════════
# 基础卷积块（深度可分离 3D 卷积）
# ════════════════════════════════════════════════════════════════════════════
class DWSConv3d(nn.Module):
    """3D 深度可分离卷积：depthwise(3×3×3) + pointwise(1×1×1)。

    用 depthwise 卷积提取每通道空间特征，再用 pointwise 卷积跨通道融合，
    相对普通 Conv3d 参数量减少至约 1/C（C 为通道数）。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, dilation=1, bias=False):
        super().__init__()
        if padding is None:
            # 自动计算 same-padding（考虑 dilation）
            padding = ((kernel_size - 1) // 2) * dilation
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=bias,
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# ════════════════════════════════════════════════════════════════════════════
# Stem 层：7×7×7 卷积（步长2）+ BN + SiLU
# 论文 Section 2：Stem 使用 7×7×7 stride=2 提取初始特征
# ════════════════════════════════════════════════════════════════════════════
class Stem(nn.Module):
    """Stem: 7×7×7 conv stride=2 + BN + SiLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=7,
                              stride=2, padding=3, bias=False)
        self.norm = nn.BatchNorm3d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# ════════════════════════════════════════════════════════════════════════════
# DASPP（Deep Atrous Spatial Pyramid Pooling）
# 6 条并行路径，全部使用深度可分离卷积：
#  1) 全局平均池化 → 1×1×1 DWS Conv   （全局上下文）
#  2) 1×1×1 DWS Conv                   （细粒度局部）
#  3) 3×3×3 DWS Conv, dilation=3
#  4) 3×3×3 DWS Conv, dilation=6
#  5) 3×3×3 DWS Conv, dilation=9
#  6) 跳跃连接                          （保留输入特征）
# 融合后 1×1×1 DWS Conv 调通道 + 残差连接
# ════════════════════════════════════════════════════════════════════════════
class DASPP(nn.Module):
    def __init__(self, channels, dilations=(3, 6, 9)):
        super().__init__()
        c = channels

        # 路径 1：全局平均池化 + 1×1×1 DWS Conv
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.global_conv = DWSConv3d(c, c, kernel_size=1)

        # 路径 2：1×1×1 DWS Conv
        self.conv1x1 = DWSConv3d(c, c, kernel_size=1)

        # 路径 3~5：多扩张率 3×3×3 DWS Conv
        self.atrous = nn.ModuleList([
            DWSConv3d(c, c, kernel_size=3, padding=d, dilation=d)
            for d in dilations
        ])

        # 融合 5 条路径（不含跳跃）后的 1×1×1 DWS Conv
        # 通道数：c × (2 + len(dilations)) = c × 5
        self.fuse = DWSConv3d(c * (2 + len(dilations)), c, kernel_size=1)

        self.norm = nn.BatchNorm3d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        size = x.shape[2:]

        # 路径 1：全局池化 → 1×1×1 DWS Conv → 上采样回原尺寸
        gp = self.global_pool(x)
        gp = self.global_conv(gp)
        gp = F.interpolate(gp, size=size, mode="trilinear", align_corners=False)

        # 路径 2：1×1×1 DWS Conv
        c1 = self.conv1x1(x)

        # 路径 3~5：多扩张率 3×3×3 DWS Conv
        atrous_feats = [a(x) for a in self.atrous]

        # 拼接 5 条路径（路径 6 = 跳跃连接，靠残差实现）
        out = torch.cat([gp, c1, *atrous_feats], dim=1)

        # 融合 + 残差
        out = self.fuse(out)
        out = self.act(self.norm(out))
        return out + x


# ════════════════════════════════════════════════════════════════════════════
# Tri-Oriented Mamba（ToM）
# 沿 x, y, z 三个方向对 3D 特征进行扫描，
# 每个方向做双向 Mamba 扫描（forward + backward 等权融合），
# 三个方向输出取平均。
#
# 使用 mamba_ssm.Mamba2（SSD）替代论文原始的 Mamba1（selective_scan）：
# - 每个方向用 2 个 Mamba2 分别做正向与反向扫描
# - 反向扫描：先把序列翻转 → 过 Mamba2 → 翻转回原顺序
# - 正反输出按 0.5 + 0.5 加权融合，等价于双向扫描
# - 三个方向再取平均
# ════════════════════════════════════════════════════════════════════════════
class TriOrientedMamba(nn.Module):
    """沿 x, y, z 三个解剖方向做双向 Mamba 扫描，三个方向输出平均。"""

    def __init__(self, channels, d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        if not _HAS_MAMBA_SSM:
            raise ImportError(
                "mamba_ssm 未安装！LightSegMamba-V3 需在已安装 mamba_ssm 的环境（如 mamba_sm120）中运行。"
                f"\n原始错误：{_IMPORT_ERR}"
            )

        self.norm = nn.LayerNorm(channels)
        self.channels = channels
        # 保留 num_slices 参数占位（兼容旧调用），当前实现不使用
        del num_slices

        # Mamba2 约束 d_inner % headdim == 0：从 64 逐级减半找到可整除的头维度
        # （回退策略与主项目 shared/models/mamba.py 的 Mamba2SSMWrapper 一致）
        d_inner = expand * channels
        headdim = 64
        while d_inner % headdim != 0:
            headdim //= 2
            if headdim < 1:
                headdim = 1
                break

        # 每个方向 = 一个 forward + 一个 backward（双向扫描）
        # 用 ModuleList 收集 6 个 Mamba2
        self.directional_mambas = nn.ModuleList([
            Mamba2(d_model=channels, d_state=d_state, d_conv=d_conv,
                   expand=expand, headdim=headdim)
            for _ in range(6)
        ])
        # 顺序：[x_fwd, x_bwd, y_fwd, y_bwd, z_fwd, z_bwd]

        # 三方向融合投影
        self.fuse = DWSConv3d(channels, channels, kernel_size=1)

    @staticmethod
    def _flip_seq_dim(x):
        """沿序列维（dim=1）翻转，用于反向扫描。

        x shape: (B, L, C)，dim=1 即序列长度维 L。
        """
        return x.flip(dims=[1])

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        assert C == self.channels, f"通道不匹配：got {C}, expect {self.channels}"

        # 先做 LayerNorm（在 channels-last 形式上）
        x_norm = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
        x_norm = self.norm(x_norm)

        # ── 沿 x 方向扫描（即 W 方向）：(B*D*H, W, C) ───────────────────
        x_x_seq = x_norm.reshape(B * D * H, W, C)
        x_x_fwd = self.directional_mambas[0](x_x_seq)
        x_x_bwd = self._flip_seq_dim(
            self.directional_mambas[1](self._flip_seq_dim(x_x_seq))
        )
        x_x = (x_x_fwd + x_x_bwd) / 2.0
        x_x = x_x.reshape(B, D, H, W, C)

        # ── 沿 y 方向扫描（即 H 方向）：(B*D*W, H, C) ───────────────────
        x_y_seq = x_norm.permute(0, 1, 3, 2, 4).contiguous().reshape(B * D * W, H, C)
        x_y_fwd = self.directional_mambas[2](x_y_seq)
        x_y_bwd = self._flip_seq_dim(
            self.directional_mambas[3](self._flip_seq_dim(x_y_seq))
        )
        x_y = (x_y_fwd + x_y_bwd) / 2.0
        x_y = x_y.reshape(B, D, W, H, C).permute(0, 1, 3, 2, 4).contiguous()

        # ── 沿 z 方向扫描（即 D 方向）：(B*H*W, D, C) ───────────────────
        x_z_seq = x_norm.permute(0, 2, 3, 1, 4).contiguous().reshape(B * H * W, D, C)
        x_z_fwd = self.directional_mambas[4](x_z_seq)
        x_z_bwd = self._flip_seq_dim(
            self.directional_mambas[5](self._flip_seq_dim(x_z_seq))
        )
        x_z = (x_z_fwd + x_z_bwd) / 2.0
        x_z = x_z.reshape(B, H, W, D, C).permute(0, 3, 1, 2, 4).contiguous()

        # 三方向平均 → 回到 channels-first
        out = (x_x + x_y + x_z) / 3.0
        out = out.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D, H, W)

        # 1×1×1 DWS Conv 融合 + 残差
        out = self.fuse(out)
        return out + x


# ════════════════════════════════════════════════════════════════════════════
# DASPPMamba：DASPP + ToM 组合模块
# 输入 → DASPP → ToM → 输出（DASPP/ToM 内部均自带残差）
# ════════════════════════════════════════════════════════════════════════════
class DASPPMamba(nn.Module):
    def __init__(self, channels, dilations=(3, 6, 9),
                 d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        self.daspp = DASPP(channels, dilations=dilations)
        self.tom = TriOrientedMamba(
            channels, d_state=d_state, d_conv=d_conv,
            expand=expand, num_slices=num_slices,
        )

    def forward(self, x):
        x = self.daspp(x)
        x = self.tom(x)
        return x


# ════════════════════════════════════════════════════════════════════════════
# 解码器块：转置卷积上采样 + 与编码器跳跃连接拼接 + 双卷积（深度可分离）
# ════════════════════════════════════════════════════════════════════════════
class DecoderBlock(nn.Module):
    """上采样 + 跳跃连接拼接 + 双卷积（深度可分离）"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels,
                                     kernel_size=2, stride=2)
        # 拼接后通道数 = out_channels + skip_channels
        self.conv1 = DWSConv3d(out_channels + skip_channels, out_channels,
                               kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.act1 = nn.SiLU()

        self.conv2 = DWSConv3d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm3d(out_channels)
        self.act2 = nn.SiLU()

    def forward(self, x, skip):
        x = self.up(x)
        # 如果 skip 与 x 空间尺寸不一致，做中心裁剪或填充
        if x.shape[2:] != skip.shape[2:]:
            x = self._match_size(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x

    @staticmethod
    def _match_size(x, ref):
        """中心裁剪或零填充，使 x 空间尺寸与 ref 一致。"""
        diff = [r - s for r, s in zip(ref.shape[2:], x.shape[2:])]
        # 先 pad 再裁剪（pad 顺序：W, H, D 反向）
        pads = []
        for d in reversed(diff):
            if d > 0:
                pads.extend([d // 2, d - d // 2])
            else:
                pads.extend([0, 0])
        if any(p > 0 for p in pads):
            x = F.pad(x, pads)
        # 中心裁剪（如果 pad 后还大）
        if x.shape[2:] != ref.shape[2:]:
            slices = [slice(None), slice(None)]
            for r, s in zip(ref.shape[2:], x.shape[2:]):
                if r < s:
                    start = (s - r) // 2
                    end = start + r
                    slices.append(slice(start, end))
                else:
                    slices.append(slice(None))
            x = x[tuple(slices)]
        return x


# ════════════════════════════════════════════════════════════════════════════
# LightSegMamba 3D 主网络
# ════════════════════════════════════════════════════════════════════════════
class LightSegMamba3D(nn.Module):
    """
    LightSegMamba 网络（论文 2 阶段编码器版本）：

    Encoder:
        Stem (7×7×7 conv stride=2)                       → skip0 (c1)
        Down1 + DASPPMamba-1                             → skip1 (c2)
        Down2 + DASPPMamba-2                             → bottleneck (c3)

    Decoder:
        DecoderBlock(bottleneck, skip1) → DecoderBlock(d1, skip0) → output conv
        最终 trilinear 上采样回原图尺寸

    总下采样倍数：8×（Stem ×2 + Down1 ×2 + Down2 ×2）
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 32,   # V3 默认 32（V2 是 24）
        dilations: tuple = (3, 6, 9),
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_slices: int = None,
    ):
        super().__init__()
        if not _HAS_MAMBA_SSM:
            raise ImportError(
                "mamba_ssm 未安装！LightSegMamba-V3 需在 mamba_sm120 环境运行。"
                f"\n原始错误：{_IMPORT_ERR}"
            )

        c1 = base_channels              # 默认 32
        c2 = base_channels * 2          # 默认 64
        c3 = base_channels * 4          # 默认 128

        # ── Encoder ────────────────────────────────────────────────────────
        # Stem: stride=2，输出尺寸 = 输入 / 2
        self.stem = Stem(in_channels, c1)

        # Encoder Stage 1: stride=2 下采样 + DASPPMamba → 输出尺寸 = 输入 / 4
        self.down1 = nn.Conv3d(c1, c2, kernel_size=2, stride=2)
        self.enc1 = DASPPMamba(c2, dilations=dilations,
                               d_state=d_state, d_conv=d_conv,
                               expand=expand, num_slices=num_slices)

        # Encoder Stage 2: stride=2 下采样 + DASPPMamba → 输出尺寸 = 输入 / 8
        self.down2 = nn.Conv3d(c2, c3, kernel_size=2, stride=2)
        self.enc2 = DASPPMamba(c3, dilations=dilations,
                               d_state=d_state, d_conv=d_conv,
                               expand=expand, num_slices=num_slices)

        # ── Decoder ────────────────────────────────────────────────────────
        # Decoder 1: bottleneck (c3) → c2，skip = enc1 (c2)
        self.dec1 = DecoderBlock(in_channels=c3, skip_channels=c2, out_channels=c2)

        # Decoder 2: c2 → c1，skip = stem (c1)
        self.dec2 = DecoderBlock(in_channels=c2, skip_channels=c1, out_channels=c1)

        # 输出投影
        self.out_conv = nn.Conv3d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        # ── Encoder ────────────────────────────────────────────────────────
        skip0 = self.stem(x)               # (B, c1, D/2, H/2, W/2)
        x1 = self.down1(skip0)             # (B, c2, D/4, H/4, W/4)
        skip1 = self.enc1(x1)              # (B, c2, D/4, H/4, W/4)

        x2 = self.down2(skip1)             # (B, c3, D/8, H/8, W/8)
        bottleneck = self.enc2(x2)         # (B, c3, D/8, H/8, W/8)

        # ── Decoder ────────────────────────────────────────────────────────
        d1 = self.dec1(bottleneck, skip1)  # (B, c2, D/4, H/4, W/4)
        d2 = self.dec2(d1, skip0)          # (B, c1, D/2, H/2, W/2)

        # 输出：1×1×1 卷积 + 上采样回原图尺寸
        out = self.out_conv(d2)
        out = F.interpolate(out, size=x.shape[2:], mode="trilinear",
                            align_corners=False)
        return out


# ════════════════════════════════════════════════════════════════════════════
# 工厂函数 + 参数量统计
# ════════════════════════════════════════════════════════════════════════════
def get_model(model_name: str = "lightsegmamba",
              in_channels: int = 4,
              out_channels: int = 4,
              device: str = "cuda",
              base_channels: int = 32,
              **kwargs) -> nn.Module:
    """
    构造模型并移到 device。

    Args:
        model_name: 仅支持 "lightsegmamba"
        in_channels: 输入模态数（BraTS = 4）
        out_channels: 输出类别数（BG/TC/WT/ET = 4）
        device: cuda / cpu
        base_channels: Stem 输出通道数。默认 32（V3，Mamba2 版实测 ~1.27M）。
                       24 → ~0.74M（V2 量级）
                       32 → ~1.27M（V3 默认，4GB 显存安全）
                       38 → ~1.8M
                       44 → ~2.5M（接近论文 3.08M）
        **kwargs: 传给 LightSegMamba3D 的额外参数（dilations, d_state, d_conv, expand）
    """
    if model_name != "lightsegmamba":
        raise ValueError(f"未知模型: {model_name}（仅支持 lightsegmamba）")

    model = LightSegMamba3D(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        **kwargs,
    )
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
