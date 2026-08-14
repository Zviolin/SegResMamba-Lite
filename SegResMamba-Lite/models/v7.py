"""
SegResMamba-Lite V7 - V6 + Sigmoid 约束的 α 自动学习版

基于 V6 设计的两个关键改进：
1. α 用 Sigmoid 约束（不再固定 0.5 或 0.8）
   - self.alpha_raw = nn.Parameter(torch.tensor(1.4))
   - 在 forward 中：alpha = sigmoid(alpha_raw)
   - 训练时 α 自动学习最优值（始终在 0~1）
2. 初始 alpha_raw = 1.4 → sigmoid 后 ≈ 0.8（偏向 Mamba）

为什么用 Sigmoid？
- 输出范围天然 (0, 1)
- 处处可微（梯度不消失）
- 中点对称：sigmoid(0) = 0.5
- GAN、MoE、StyleGAN 都用它约束混合权重
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.models.mamba import get_mamba_layer

# 复用 V2 的基础组件
from .v2 import SegResMambaLiteV2


class MambaExpert(nn.Module):
    """单个 Mamba 专家（继承 V6）"""
    def __init__(self, dim, d_state, d_conv, expand, use_mamba2, scan_direction):
        super().__init__()
        self.scan_direction = scan_direction
        self.mamba = get_mamba_layer(
            dim=dim, d_state=d_state, d_conv=d_conv, expand=expand,
            use_mamba2=use_mamba2
        )

    def forward(self, x):
        """x: (B, C, D, H, W)；scan_direction 决定扫描方向"""
        if self.scan_direction == 'forward':
            return self.mamba(x)
        elif self.scan_direction == 'backward':
            x_rev = torch.flip(x, dims=[2, 3, 4])
            out_rev = self.mamba(x_rev)
            return torch.flip(out_rev, dims=[2, 3, 4])
        elif self.scan_direction == 'depth_first':
            x_rev = torch.flip(x, dims=[2])
            out_rev = self.mamba(x_rev)
            return torch.flip(out_rev, dims=[2])
        elif self.scan_direction == 'width_first':
            x_rev = torch.flip(x, dims=[4])
            out_rev = self.mamba(x_rev)
            return torch.flip(out_rev, dims=[4])
        else:
            return self.mamba(x)


class DConvExpert(nn.Module):
    """单个 DConv 专家（局部细化）

    设计：Depthwise 3×3×3 Conv + Pointwise 1×1 Conv + GELU
    极轻量（~1K 参数），学习局部空间细化
    """
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


class MoABottleneckV7(nn.Module):
    """V7 MoA Bottleneck - 实用 MoA + Sigmoid 约束的 α

    相对 V6 的改动：
    1. alpha 改为 alpha_raw + sigmoid 约束
    2. alpha_raw 初始为 1.4（sigmoid 后 ≈ 0.8）
    3. 训练时 α 会自动调整到最优值
    """
    def __init__(self, dim, num_experts=4, d_conv=2, use_mamba2=True):
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim

        # 共享 Mamba 主干（继承 V6）
        from .v6 import MambaExpert
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

        # ⭐ V7 创新点：4 个 DConv 专家（局部细化）
        self.dconv_experts = nn.ModuleList([
            DConvExpert(dim) for _ in range(num_experts)
        ])

        # 路由器：GAP → MLP → Softmax
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, num_experts),
        )

        # 输入归一化
        self.norm = nn.InstanceNorm3d(dim)

        # ⭐ V7 关键改进：Sigmoid 约束的 α（自动学习）
        # sigmoid(1.4) ≈ 0.802（偏向 Mamba）
        self.alpha_raw = nn.Parameter(torch.tensor(1.4))

    def get_alpha(self):
        """获取当前的 α 值（用于日志/可视化）"""
        return torch.sigmoid(self.alpha_raw)

    def forward(self, x):
        """x: (B, C, D, H, W)

        V7 MoA Bottleneck:
        1. Mamba 专家路由加权输出
        2. DConv 专家路由加权输出
        3. α 自动学习（Mamba vs DConv 混合权重）
        4. 残差连接
        """
        B, C, D, H, W = x.shape

        # 1. 路由权重（soft weights）
        route_logits = self.router(x)  # (B, num_experts)
        route_weights = F.softmax(route_logits, dim=-1)  # (B, num_experts)

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

        # 5. ⭐ Sigmoid 约束的 α 自动学习混合
        alpha = torch.sigmoid(self.alpha_raw)  # 自动在 (0, 1)
        out = alpha * mamba_out + (1 - alpha) * dconv_out

        # 6. 残差连接
        return out + x

    def get_route_weights(self, x):
        """获取路由权重（用于可视化）"""
        with torch.no_grad():
            route_logits = self.router(x)
            return F.softmax(route_logits, dim=-1)


class SegResMambaLiteV7(SegResMambaLiteV2):
    """V7 - V6 + Sigmoid 约束的 α 自动学习版

    基于 V6 架构，仅替换 bottleneck：
    - V6: MoABottleneck (固定 α=0.5)
    - V7: MoABottleneckV7 (Sigmoid 约束 α 自动学习)

    其他完全继承 V2。
    """
    def __init__(self, in_channels=4, out_channels=4, init_filters=20,
                 d_state=8, d_conv=2, expand=2, use_mamba2=True,
                 use_deep_supervision=False,
                 use_mamba2_in_moa=True,
                 num_experts=4):
        super().__init__(in_channels, out_channels, init_filters,
                         d_state, d_conv, expand, use_mamba2,
                         use_deep_supervision)

        # 删除 V2 的 mamba_8，替换为 V7 MoA
        del self.mamba_8

        # V7 创新：MoA Bottleneck (Sigmoid α)
        self.moa_8 = MoABottleneckV7(
            dim=init_filters * 4,
            num_experts=num_experts,
            d_conv=d_conv,
            use_mamba2=use_mamba2_in_moa,
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

        # ⭐ V7 创新：bottleneck 用 Sigmoid α MoA
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
        """获取当前的 α 值（Mamba vs DConv 混合权重）"""
        return self.moa_8.get_alpha().item()


def get_model(in_channels=4, out_channels=4, init_filters=20,
              d_state=8, d_conv=2, expand=2, use_mamba2=True,
              use_attention=True, use_deep_supervision=False, device="cuda",
              num_experts=4):
    """V7 模型工厂函数

    Args:
        num_experts: MoA 专家数量（默认 4）
        其他参数与 V6 一致
    """
    model = SegResMambaLiteV7(
        in_channels, out_channels, init_filters,
        d_state, d_conv, expand, use_mamba2,
        use_deep_supervision,
        num_experts=num_experts,
    ).to(device)
    return model


if __name__ == "__main__":
    # 快速测试
    model = get_model(init_filters=20, device="cpu")
    total = sum(p.numel() for p in model.parameters())
    print(f"V7 参数量: {total:,} ({total/1e6:.3f}M)")
    print(f"初始 α (sigmoid): {model.get_alpha():.4f}")

    x = torch.randn(1, 4, 64, 64, 64)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Forward OK: input={x.shape}, output={out.shape}")

    weights = model.get_moa_route_weights(x)
    print(f"路由权重: {[round(w, 4) for w in weights[0].tolist()]}")
    print(f"权重和: {weights[0].sum().item():.4f}")