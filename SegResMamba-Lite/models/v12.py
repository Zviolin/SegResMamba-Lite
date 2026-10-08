"""
SegResMamba-Lite V12 - V10 + 边界门控（探索版）

对应 实验记录总集.md Part F·§四-I（I1）：
- 动机：V11 相对 V10 同时改了专家池（2×Attention 换 2×DConv）与边界模块，
  LW Dice 从 0.8304 回退到 0.8130 的归因不明。V12 隔离边界模块的净贡献。
- 配置：保持 V10 的 4×Attention 专家 + 共享 K/V + token 级 Top-2 不变，
  仅加 V11 的边界门控（BoundaryAttention3D）。
- 判据：I1 LW Dice > 0.8304 → V12 升级为最终模型；≤ → 论文结论
  「边界门控在 Attention 专家体系下无增益」，V11 回退归因于专家池替换。

边界模块（与 V11 完全同构，从 v11.py 引入）：
- 固定 26 邻域中心差分核（3×3×3 中心差分，不参与训练，冻结 2,160 参数）
- 可学习门控（Conv1×1 → ReLU → Conv1×1 → Sigmoid，3,300 参数）
- 前向：out = x * (1 + gate(x) * edge_mag)

参数量：V10 锚点 + 5,460（2,160 冻结 + 3,300 可学习）
日志口径预期 = 1,422,734 + 5,460 = 1,428,194（<1.5M）
"""
import torch

from .v10 import MoABottleneckV10, SegResMambaLiteV10
from .v11 import BoundaryAttention3D


class MoABottleneckV12(MoABottleneckV10):
    """V10 论文级 MoA + 边界注意力瓶颈模块

    完整继承 V10 的 MoA（专家/路由/KV/噪声/负载均衡全部一致），
    仅覆盖路由前增强钩子 _pre_route_enhance：在归一化后、路由前做边界增强。
    """
    def __init__(self, dim, **kwargs):
        super().__init__(dim, **kwargs)
        # 边界注意力：显式建模肿瘤边界（项目硬约束）
        self.boundary_attn = BoundaryAttention3D(dim)

    def _pre_route_enhance(self, x_norm):
        """边界增强：out = x * (1 + gate(x) * edge_mag)（与 V11 同构）"""
        return self.boundary_attn(x_norm)


class SegResMambaLiteV12(SegResMambaLiteV10):
    """V12 = V10（4×Attention 专家 + token 级 Top-2）+ 边界门控

    forward / 路由可视化接口 / 负载均衡接口全部继承 V10；
    唯一差异是 bottleneck 换成 MoABottleneckV12（插入边界注意力）。
    """
    def __init__(self, in_channels=4, out_channels=4, init_filters=20,
                 d_state=8, d_conv=2, expand=2, use_mamba2=True,
                 use_deep_supervision=False,
                 num_experts=4, top_k=2,
                 expert_type='attention', route_granularity='token',
                 share_kv=True, route_noise=0.0, decoder_moa=False):
        super().__init__(in_channels, out_channels, init_filters,
                         d_state, d_conv, expand, use_mamba2,
                         use_deep_supervision,
                         num_experts=num_experts, top_k=top_k,
                         expert_type=expert_type,
                         route_granularity=route_granularity,
                         share_kv=share_kv, route_noise=route_noise,
                         decoder_moa=decoder_moa)

        # 将 bottleneck 换成带边界注意力的 V12 版（配置与 V10 完全一致）
        self.moa_10 = MoABottleneckV12(
            dim=init_filters * 4,
            num_experts=num_experts,
            top_k=top_k,
            expert_type=expert_type,
            route_granularity=route_granularity,
            share_kv=share_kv,
            route_noise=route_noise,
            use_mamba2=use_mamba2,
            d_state=d_state,
            d_conv=d_conv,
        )

        # G1 联动：若启用解码器侧 MoA，同步换成 V12 版
        if decoder_moa:
            self.moa_dec16 = MoABottleneckV12(
                dim=init_filters * 4,
                num_experts=num_experts,
                top_k=top_k,
                expert_type=expert_type,
                route_granularity=route_granularity,
                share_kv=share_kv,
                route_noise=route_noise,
                use_mamba2=use_mamba2,
                d_state=d_state,
                d_conv=d_conv,
            )


def get_model(in_channels=4, out_channels=4, init_filters=20,
              d_state=8, d_conv=2, expand=2, use_mamba2=True,
              use_attention=True, use_deep_supervision=False, device="cuda",
              num_experts=4, top_k=2,
              expert_type='attention', route_granularity='token',
              share_kv=True, route_noise=0.0, decoder_moa=False):
    """V12 模型工厂函数（参数与 V10 完全一致）"""
    model = SegResMambaLiteV12(
        in_channels, out_channels, init_filters,
        d_state, d_conv, expand, use_mamba2,
        use_deep_supervision,
        num_experts=num_experts,
        top_k=top_k,
        expert_type=expert_type,
        route_granularity=route_granularity,
        share_kv=share_kv,
        route_noise=route_noise,
        decoder_moa=decoder_moa,
    ).to(device)
    return model


if __name__ == "__main__":
    # 快速测试
    model = get_model(init_filters=20, device="cpu")
    total = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"V12 参数量: {total:,} ({total/1e6:.3f}M)，冻结: {frozen:,}")

    x = torch.randn(1, 4, 64, 64, 64)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"输入: {x.shape}, 输出: {out.shape}")
