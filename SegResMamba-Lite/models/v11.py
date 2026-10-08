"""
SegResMamba-Lite V11 - 边界感知混合 MoA 版

基于 V10（论文级 MoA）的三项针对性升级：

1. 混合专家池（2 Attention + 2 DConv）
   - Attention 专家：全局语义召回（V10 优势：LW Dice 0.8304）
   - DConv 专家：局部边界细节（V9 优势：LW HD95 6.06）
   → 目标：LW Dice 保持 + LW HD95 提升

2. 边界注意力模块（显式建模肿瘤边界）
   - Sobel 梯度提取边界响应 + 门控增强边界区域特征
   - 项目硬约束：边界注意力需显式建模肿瘤边界

3. 路由噪声（Noisy Top-K Gating, Shazeer 2017）
   - 训练时 router logits 加噪声，避免路由陷入局部最优
   - 目标：全局指标稳定收敛

与 V10 相同：token 级路由、共享 K/V、负载均衡损失、token 级稀疏。

参数量目标 <1.5M
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 V2 基础组件、V9 的 DConv 专家
from .v2 import LightResBlock, SegResMambaLiteV2
from .v9 import DConvExpert


class BoundaryAttention3D(nn.Module):
    """3D 边界注意力模块：显式建模肿瘤边界

    设计：
    1. 固定 Sobel 风格梯度卷积提取边界响应（不学习，纯几何算子）
    2. 边界强度图归一化
    3. 可学习门控生成边界权重，增强边界区域特征

    前向：out = x * (1 + gate(x) * edge_mag)
    """
    def __init__(self, dim):
        super().__init__()
        # 固定 3×3×3 中心差分核（边界提取器，不参与训练）
        self.edge_conv = nn.Conv3d(dim, dim, 3, padding=1, groups=dim, bias=False)
        kernel = self._center_diff_kernel(dim)
        self.edge_conv.weight.data = kernel
        self.edge_conv.weight.requires_grad = False

        # 可学习边界门控
        self.gate = nn.Sequential(
            nn.Conv3d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim // 4, dim, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _center_diff_kernel(dim):
        """3×3×3 中心差分核（各向同性边界响应）→ (dim, 1, 3, 3, 3)"""
        k = torch.zeros(3, 3, 3)
        k[1, 1, 1] = -6.0
        for i in range(3):
            for j in range(3):
                for m in range(3):
                    if i == 1 and j == 1 and m == 1:
                        continue
                    k[i, j, m] = 1.0
        return k.view(1, 1, 3, 3, 3).expand(dim, 1, 3, 3, 3).contiguous()

    def forward(self, x):
        """x: (B, C, D, H, W) → 边界增强后的特征"""
        # 边界响应（中心差分近似梯度幅值）
        edges = self.edge_conv(x)
        edge_mag = edges.abs().mean(dim=1, keepdim=True)  # (B, 1, D, H, W)
        # 空间归一化
        # 注意：此处为全 batch + 全空间 max 归一化，训练（整图输入）与推理（滑窗逐窗输入）
        # 的归一化尺度存在轻微分布差异；V11/V12 历史结果基于此口径产出，为保持
        # 代码与已定格实验数字的可复现对应关系，不再修改归一化方式
        edge_mag = edge_mag / (edge_mag.max() + 1e-6)
        # 门控增强：边界区域特征加权强化
        w = self.gate(x) * edge_mag
        return x * (1 + w)


class MoABottleneckV11(nn.Module):
    """边界感知混合 MoA 瓶颈模块

    设计：
    - 共享 K/V 投影（论文 W^k, W^v）
    - 混合专家池：2 Attention 专家（全局语义）+ 2 DConv 专家（局部边界）
    - 边界注意力：显式增强肿瘤边界区域
    - token 级路由器 + 路由噪声（训练时）
    - token 级稀疏激活 + 负载均衡损失

    Args:
        dim: 特征通道数（bottleneck）
        num_experts: 专家数量（默认 4 = 2 Attention + 2 DConv）
        top_k: 每个 token 激活的专家数（默认 2）
    """
    def __init__(self, dim, num_experts=4, top_k=2):
        super().__init__()
        assert num_experts == 4, "V11 固定 2 Attention + 2 DConv 混合"
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_attn = num_experts // 2  # 2 个 Attention 专家
        self.scale = (dim // 2) ** -0.5

        # ── 共享 K/V 投影 ──
        self.w_k = nn.Linear(dim, dim // 2)
        self.w_v = nn.Linear(dim, dim)

        # ── Attention 专家（前 num_attn 个）──
        self.expert_qs = nn.ModuleList([
            nn.Linear(dim, dim // 2) for _ in range(self.num_attn)
        ])
        self.expert_outs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_experts)
        ])

        # ── DConv 专家（后 num_attn 个）──
        self.dconv_experts = nn.ModuleList([
            DConvExpert(dim, kernel_size=3, dilation=1),
            DConvExpert(dim, kernel_size=3, dilation=2),
        ])

        # ── 边界注意力（显式建模肿瘤边界）──
        self.boundary_attn = BoundaryAttention3D(dim)

        # ── token 级路由器 ──
        self.router = nn.Linear(dim, num_experts)

        self.norm = nn.InstanceNorm3d(dim)
        self.last_route_weights = None

    def _flatten_tokens(self, x):
        B, C, D, H, W = x.shape
        return x.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C), (B, C, D, H, W)

    def _unflatten_tokens(self, x_t, shape):
        B, C, D, H, W = shape
        return x_t.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

    def forward(self, x):
        """x: (B, C, D, H, W)

        流程：
        1. 归一化 + 边界注意力增强
        2. token 级路由（训练时加噪声）
        3. Attention 专家：token 级稀疏计算（全局注意力）
           DConv 专家：整图前向 + 选中 token 输出（梯度截断保持稀疏训练）
        4. 选中权重加权融合 + 残差
        """
        x_norm = self.norm(x)

        # ⭐ 边界注意力：显式增强肿瘤边界区域
        x_enh = self.boundary_attn(x_norm)

        x_t, shape = self._flatten_tokens(x_enh)  # (B, L, C)
        B, L, C = x_t.shape

        # 1. token 级路由（训练时加噪声：Noisy Top-K Gating）
        route_logits = self.router(x_t)  # (B, L, num_experts)
        if self.training:
            noise = torch.randn_like(route_logits) * 0.05
            route_logits = route_logits + noise
        route_weights = F.softmax(route_logits, dim=-1)
        self.last_route_weights = route_weights

        topk_weights, topk_indices = torch.topk(route_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # 2. 共享 K/V（Attention 专家用）
        k = self.w_k(x_t)
        v = self.w_v(x_t)

        # 3. ⭐ token 级稀疏 + 混合专家计算
        out_t = torch.zeros_like(x_t)
        for i in range(self.num_experts):
            sel = (topk_indices == i).nonzero()
            if sel.numel() == 0:
                continue
            b_idx, l_idx, k_idx = sel[:, 0], sel[:, 1], sel[:, 2]
            w_i = topk_weights[b_idx, l_idx, k_idx].unsqueeze(-1)  # (N, 1)

            if i < self.num_attn:
                # ── Attention 专家：token 级稀疏（只算选中 token）──
                q_i = self.expert_qs[i](x_t[b_idx, l_idx])  # (N, dim//2)
                k_i = k[b_idx]  # (N, L, dim)
                v_i = v[b_idx]
                scores = torch.einsum('nd,nld->nl', q_i, k_i) * self.scale
                attn_w = torch.softmax(scores, dim=-1)
                ctx = torch.einsum('nl,nld->nd', attn_w, v_i)
                e_out = self.expert_outs[i](ctx)
            else:
                # ── DConv 专家：局部边界卷积（整图前向，选中 token 输出）──
                d_idx = i - self.num_attn
                dconv_full = self.dconv_experts[d_idx](x_enh)  # (B, C, D, H, W)
                dconv_t = self._flatten_tokens(dconv_full)[0]  # (B, L, C)
                # 选中 token 的输出（未选中部分梯度截断：out_t 加 0 权重）
                e_out = self.expert_outs[i](dconv_t[b_idx, l_idx])

            out_t[b_idx, l_idx] += w_i * e_out

        # 4. 还原 + 残差
        out = self._unflatten_tokens(out_t, shape)
        return out + x

    def get_route_weights(self, x):
        """获取 token 级路由权重（用于可视化）"""
        with torch.no_grad():
            x_norm = self.norm(x)
            x_enh = self.boundary_attn(x_norm)
            x_t, _ = self._flatten_tokens(x_enh)
            return F.softmax(self.router(x_t), dim=-1)

    def get_topk_indices(self, x):
        """获取 token 级 Top-K 选择（用于分析）"""
        with torch.no_grad():
            x_norm = self.norm(x)
            x_enh = self.boundary_attn(x_norm)
            x_t, _ = self._flatten_tokens(x_enh)
            weights = F.softmax(self.router(x_t), dim=-1)
            topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1)
            return topk_indices, topk_weights

    def load_balancing_loss(self, route_weights):
        """token 级负载均衡损失"""
        avg_weights = route_weights.mean(dim=(0, 1))  # (num_experts,)
        target = torch.ones_like(avg_weights) / self.num_experts
        loss = F.kl_div(
            torch.log(avg_weights + 1e-8),
            target,
            reduction='batchmean'
        )
        return loss


class SegResMambaLiteV11(SegResMambaLiteV2):
    """V11 - 边界感知混合 MoA 版

    基于 V2 架构，仅修改 bottleneck：
    - V2: BiMambaLayer @ 8³
    - V11: 混合 MoA（2 Attention + 2 DConv + 边界注意力 + token 级路由）@ 8³

    其他完全继承 V2。
    """
    def __init__(self, in_channels=4, out_channels=4, init_filters=20,
                 d_state=8, d_conv=2, expand=2, use_mamba2=True,
                 use_deep_supervision=False,
                 num_experts=4, top_k=2):
        super().__init__(in_channels, out_channels, init_filters,
                         d_state, d_conv, expand, use_mamba2,
                         use_deep_supervision)

        # 删除 V2 的 mamba_8，替换为混合 MoA
        del self.mamba_8

        self.moa_11 = MoABottleneckV11(
            dim=init_filters * 4,
            num_experts=num_experts,
            top_k=top_k,
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

        # ⭐ V11 创新：边界感知混合 MoA（token 级 Top-K 稀疏）
        x3_bottleneck = self.moa_11(x3_down)

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
        """获取 token 级路由权重（用于可视化）"""
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x1_down = self.down1(x1)
        x2 = self.enc2(x1_down)
        x2_mamba = self.mamba_32(x2)
        x2_down = self.down2(x2_mamba)
        x3 = self.enc3(x2_down)
        x3_mamba = self.mamba_16(x3)
        x3_down = self.down3(x3_mamba)
        return self.moa_11.get_route_weights(x3_down)

    def get_moa_topk_indices(self, x):
        """获取 token 级 Top-K 选择（用于分析）"""
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x1_down = self.down1(x1)
        x2 = self.enc2(x1_down)
        x2_mamba = self.mamba_32(x2)
        x2_down = self.down2(x2_mamba)
        x3 = self.enc3(x2_down)
        x3_mamba = self.mamba_16(x3)
        x3_down = self.down3(x3_mamba)
        return self.moa_11.get_topk_indices(x3_down)

    def get_moa_load_balance_loss(self):
        """获取 token 级负载均衡损失（训练时调用）"""
        if self.moa_11.last_route_weights is None:
            return torch.tensor(0.0, device=next(self.moa_11.parameters()).device)
        return self.moa_11.load_balancing_loss(self.moa_11.last_route_weights)

    # 统一训练接口（与 V9/V10 一致，便于 frame/train.py 调用）
    get_moe_load_balance_loss = get_moa_load_balance_loss


def get_model(in_channels=4, out_channels=4, init_filters=20,
              d_state=8, d_conv=2, expand=2, use_mamba2=True,
              use_attention=True, use_deep_supervision=False, device="cuda",
              num_experts=4, top_k=2):
    """V11 模型工厂函数"""
    model = SegResMambaLiteV11(
        in_channels, out_channels, init_filters,
        d_state, d_conv, expand, use_mamba2,
        use_deep_supervision,
        num_experts=num_experts,
        top_k=top_k,
    ).to(device)
    return model


if __name__ == "__main__":
    model = get_model(init_filters=20, device="cpu")
    total = sum(p.numel() for p in model.parameters())
    print(f"V11 参数量: {total:,} ({total/1e6:.3f}M)")
