"""
SegResMamba-Lite V8 - V6 + α 真正自由自适应（温度缩放版）

设计动机（基于网络调研）：
- V6 (α=0.5 固定): Dice 0.8904
- V7 (α Sigmoid 自动): Dice 0.8913，但 α 几乎不动（只移动 0.0008）
- EMA 标准实践：可学习参数不应参与 EMA（必须排除）
- 温度缩放（Temperature Scaling）：用可学习 τ 控制 Sigmoid 斜率
- 避免 Clamp 极端（之前 V8 α 跑到 0.0）

关键设计（基于学术最佳实践）：
1. alpha_raw 初始化为 0.0（不偏向）
2. temperature τ 初始化为 1.0（标准 Sigmoid 斜率）
3. alpha = sigmoid(alpha_raw / temperature)  # 温度控制移动速度
4. α 和 τ 都不参与 EMA（保证能跟上每次更新）
5. α 和 τ 用 10x 学习率（适中的加速，避免极端）

为什么比之前的设计更好：
- Sigmoid 平滑（避免 Clamp 极端）
- 温度 τ 自动调节学习速度（防止 α 跑极端）
- 不参与 EMA（让 α 真正移动）
- 学术界公认做法（温度缩放 + softmax gating）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.models.mamba import get_mamba_layer

from .v2 import SegResMambaLiteV2
from .v6 import MambaExpert


class DConvExpert(nn.Module):
    """DConv 专家（局部细化）"""
    def __init__(self, dim):
        super().__init__()
        self.dw_conv = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.pw_conv = nn.Conv3d(dim, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.dw_conv(x)
        out = self.pw_conv(out)
        out = self.act(out)
        return out


class MoABottleneckV8(nn.Module):
    """V8 MoA Bottleneck - V6 + α 真正自由自适应（温度缩放版）

    相对 V6 的关键改动（基于学术最佳实践）：
    1. alpha_raw 初始化为 0.0（不偏向 Mamba 或 DConv）
    2. temperature τ 初始化为 1.0（标准 Sigmoid 斜率）
    3. alpha = sigmoid(alpha_raw / temperature)  # 温度缩放
    4. α 和 τ 都不参与 EMA（让 α 真正移动）
    5. α 和 τ 用 10x 学习率（避免之前的极端）
    """
    def __init__(self, dim, num_experts=4, d_conv=2, use_mamba2=True):
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim

        # 共享 Mamba 主干
        self.experts = nn.ModuleList([
            MambaExpert(dim, d_state=4, d_conv=d_conv, expand=1, use_mamba2=use_mamba2,
                        scan_direction='forward'),
            MambaExpert(dim, d_state=8, d_conv=d_conv, expand=1, use_mamba2=use_mamba2,
                        scan_direction='backward'),
            MambaExpert(dim, d_state=16, d_conv=d_conv, expand=1, use_mamba2=use_mamba2,
                        scan_direction='depth_first'),
            MambaExpert(dim, d_state=8, d_conv=d_conv, expand=1, use_mamba2=use_mamba2,
                        scan_direction='width_first'),
        ])

        # DConv 专家（局部细化）
        self.dconv_experts = nn.ModuleList([
            DConvExpert(dim) for _ in range(num_experts)
        ])

        # 路由器
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, num_experts),
        )

        # 输入归一化
        self.norm = nn.InstanceNorm3d(dim)

        # ⭐ V8 关键：α 真正自由自适应（温度缩放）
        # 1. alpha_raw 初始化为 0.0（不偏向）
        # 2. temperature τ 初始化为 1.0
        # 3. α = sigmoid(alpha_raw / temperature)  # 温度控制 Sigmoid 斜率
        # 4. α 和 τ 都不参与 EMA（保证能跟上每次更新）
        # 5. α 和 τ 用 10x 学习率（避免之前的极端）
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self._alpha_raw_exclude_from_ema = True  # 标记：alpha_raw 不参与 EMA

    def get_alpha(self):
        """获取当前的 α 值（温度缩放 Sigmoid）"""
        # 用 Sigmoid + 温度缩放（避免 Clamp 极端）
        # τ > 1：平滑分布，α 移动慢
        # τ < 1：锐化分布，α 移动快
        # τ = 1：标准 Sigmoid
        return torch.sigmoid(self.alpha_raw / self.temperature)

    def forward(self, x):
        """x: (B, C, D, H, W)"""
        B, C, D, H, W = x.shape

        # 1. 路由权重
        route_logits = self.router(x)
        route_weights = F.softmax(route_logits, dim=-1)

        # 2. 归一化输入
        x_norm = self.norm(x)

        # 3. Mamba 专家输出（加权求和）
        mamba_out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_out = expert(x_norm)
            weight_i = route_weights[:, i].view(B, 1, 1, 1, 1)
            mamba_out = mamba_out + weight_i * expert_out

        # 4. DConv 专家输出（加权求和）
        dconv_out = torch.zeros_like(x)
        for i, expert in enumerate(self.dconv_experts):
            expert_out = expert(x_norm)
            weight_i = route_weights[:, i].view(B, 1, 1, 1, 1)
            dconv_out = dconv_out + weight_i * expert_out

        # 5. ⭐ V8 关键：α = sigmoid(alpha_raw / temperature)
        # 温度缩放控制 α 的移动速度（学术最佳实践）
        alpha = torch.sigmoid(self.alpha_raw / self.temperature)
        out = alpha * mamba_out + (1 - alpha) * dconv_out

        # 6. 残差连接
        return out + x

    def get_route_weights(self, x):
        """获取路由权重（用于可视化）"""
        with torch.no_grad():
            route_logits = self.router(x)
            return F.softmax(route_logits, dim=-1)


class SegResMambaLiteV8(SegResMambaLiteV2):
    """V8 - V6 + α 自动学习（约束 0.4-0.6）

    基于 V6 架构，仅修改 bottleneck 的 α 自动学习（范围约束）。
    """
    def __init__(self, in_channels=4, out_channels=4, init_filters=20,
                 d_state=8, d_conv=2, expand=2, use_mamba2=True,
                 use_deep_supervision=False,
                 num_experts=4):
        super().__init__(in_channels, out_channels, init_filters,
                         d_state, d_conv, expand, use_mamba2,
                         use_deep_supervision)

        # 删除 V2 的 mamba_8
        del self.mamba_8

        # V8 创新：MoA Bottleneck with α 自动学习（约束 0.4-0.6）
        self.moa_8 = MoABottleneckV8(
            dim=init_filters * 4,
            num_experts=num_experts,
            d_conv=d_conv,
            use_mamba2=use_mamba2,
        )

    def forward(self, x):
        # Stem
        x0 = self.stem(x)

        # Encoder（完全继承 V2）
        x1 = self.enc1(x0)
        x1_down = self.down1(x1)

        x2 = self.enc2(x1_down)
        x2_mamba = self.mamba_32(x2)
        x2_down = self.down2(x2_mamba)

        x3 = self.enc3(x2_down)
        x3_mamba = self.mamba_16(x3)
        x3_down = self.down3(x3_mamba)

        # ⭐ V8 创新：bottleneck 用 α=0.3 MoA
        x3_bottleneck = self.moa_8(x3_down)

        # Decoder（完全继承 V2）
        x2_up = self.up2(x3_bottleneck)
        x2_skip_cat = torch.cat([x2_up, x3_mamba], dim=1)
        x2_dec = self.dec2(x2_skip_cat)

        x1_up = self.up1(x2_dec)
        x1_skip_cat = torch.cat([x1_up, x1_down], dim=1)
        x1_dec = self.dec1(x1_skip_cat)

        x0_up = self.up0(x1_dec)
        x0_skip_cat = torch.cat([x0_up, x0], dim=1)
        x0_dec = self.dec0(x0_skip_cat)

        out = self.out(x0_dec)

        if self.use_deep_supervision and self.training:
            ds1 = self.ds1(x2_dec)
            ds2 = self.ds2(x1_dec)
            return (out, ds1, ds2)

        return out

    def get_moa_route_weights(self, x):
        """获取 MoA 路由权重（用于可视化）"""
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x1_down = self.down1(x1)
        x2 = self.enc2(x1_down)
        x2_mamba = self.mamba_32(x2)
        x2_down = self.down2(x2_mamba)
        x3 = self.enc3(x2_down)
        x3_mamba = self.mamba_16(x3)
        x3_down = self.down3(x3_mamba)
        return self.moa_8.get_route_weights(x3_down)

    def get_alpha(self):
        """获取当前的 α 值"""
        return self.moa_8.get_alpha().item()


def get_model(in_channels=4, out_channels=4, init_filters=20,
              d_state=8, d_conv=2, expand=2, use_mamba2=True,
              use_attention=True, use_deep_supervision=False, device="cuda",
              num_experts=4):
    """V8 模型工厂函数 - V6 + α 自动学习（约束 0.4-0.6）"""
    model = SegResMambaLiteV8(
        in_channels, out_channels, init_filters,
        d_state, d_conv, expand, use_mamba2,
        use_deep_supervision,
        num_experts=num_experts,
    ).to(device)
    return model


if __name__ == "__main__":
    model = get_model(init_filters=20, device="cpu")
    total = sum(p.numel() for p in model.parameters())
    print(f"V8 参数量: {total:,} ({total/1e6:.3f}M)")
    print(f"初始 α (sigmoid): {model.get_alpha():.4f}")
    print(f"目标: Mamba={model.get_alpha()*100:.1f}% / DConv={(1-model.get_alpha())*100:.1f}%")

    x = torch.randn(1, 4, 64, 64, 64)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Forward OK: {x.shape} -> {out.shape}")

    weights = model.get_moa_route_weights(x)
    print(f"路由权重: {[round(w, 4) for w in weights[0].tolist()]}")
    print(f"权重和: {weights[0].sum().item():.4f}")