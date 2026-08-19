"""
SegResMamba-Lite V10 - 论文级 MoA (Mixture of Attention) 版

严格按 SHMoAReg 论文 (arXiv:2509.20073) 实现：

  论文公式：
    E_i(q_t, K, V) = Attention_i(q_t, K, V) · W_i^o     # 每个专家的注意力头 + 独有输出投影
    y_t = Σ_{i ∈ G(q_t)} w_{i,t} · E_i(q_t, K, V)       # 只对选中专家加权融合
    G(q_t) = TopK(Router(q_t), k)                        # token 级 Top-K 稀疏路由

  V10 完整实现论文三大要素：
  1. 多个 Attention 头专家（每个专家有独有 query 变换 + 独有输出投影 W_i^o）
  2. 共享 K/V 投影（W^k, W^v 对所有专家共享）
  3. token 级稀疏路由（每个 token 独立选 Top-K 专家，只计算选中专家）

演进历史（重要）：
- V10 第一版：Mamba 专家 + 样本级稀疏（效果差：LW Dice 0.7961，HD95 短板）
  原因：Mamba 专家在 8³ 短序列上分工趋同，稀疏丢弃信息得不偿失
- V10 第二版（本版）：严格论文级 MoA，替换失败的第一版

参数量目标 <1.5M
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 V2 的基础组件
from .v2 import LightResBlock, SegResMambaLiteV2


class MoABottleneckV10(nn.Module):
    """论文级 MoA 瓶颈模块（共享 K/V + Attention 专家 + token 级 Top-K 稀疏）

    设计：
    - 共享 K/V 投影：W_k, W_v（论文 W^k, W^v 共享）
    - 4 个 Attention 专家：每个有独有 query 变换（决定注意力模式）+ 独有输出投影 W_i^o
    - token 级路由器：每个 token 独立选择 Top-K 专家
    - token 级稀疏激活：只对选中专家的 token 计算注意力（未选中不计算、无梯度）
    - 负载均衡损失（防止路由坍缩）

    Args:
        dim: 特征通道数（bottleneck）
        num_experts: 专家数量（默认 4）
        top_k: 每个 token 激活的专家数（默认 2）
    """
    def __init__(self, dim, num_experts=4, top_k=2):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.scale = (dim // 2) ** -0.5  # 注意力缩放

        # ── 共享 K/V 投影（论文：W^k, W^v 共享）──
        # K 与专家 query 同维度（dim//2），保证注意力打分维度匹配
        self.w_k = nn.Linear(dim, dim // 2)
        self.w_v = nn.Linear(dim, dim)

        # ── 4 个 Attention 专家 ──
        # 每个专家独有 query 变换（不同 → 不同注意力模式）+ 独有输出投影 W_i^o
        self.expert_qs = nn.ModuleList([
            nn.Linear(dim, dim // 2) for _ in range(num_experts)
        ])
        self.expert_outs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_experts)
        ])

        # ── token 级路由器（每个 token 独立打分）──
        self.router = nn.Linear(dim, num_experts)

        self.norm = nn.InstanceNorm3d(dim)

        # 最近一次 forward 的路由权重缓存（供训练时计算负载均衡损失，避免重复前向）
        self.last_route_weights = None

    def _flatten_tokens(self, x):
        """(B, C, D, H, W) → (B, L, C)"""
        B, C, D, H, W = x.shape
        return x.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C), (B, C, D, H, W)

    def _unflatten_tokens(self, x_t, shape):
        """(B, L, C) → (B, C, D, H, W)"""
        B, C, D, H, W = shape
        return x_t.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

    def forward(self, x):
        """x: (B, C, D, H, W)

        论文级 token 稀疏 MoA：
        1. token 级路由：每个 token 独立选 Top-K 专家
        2. 共享 K/V 投影
        3. ⭐ token 级稀疏：只对选中专家的 token 计算注意力（未选中不计算）
        4. 选中权重加权融合 + 残差
        """
        x_norm = self.norm(x)
        x_t, shape = self._flatten_tokens(x_norm)  # (B, L, C)
        B, L, C = x_t.shape

        # 1. token 级路由（论文 G(q_t) = TopK(Router(q_t), k)）
        route_logits = self.router(x_t)  # (B, L, num_experts)
        route_weights = F.softmax(route_logits, dim=-1)  # (B, L, num_experts)
        self.last_route_weights = route_weights

        topk_weights, topk_indices = torch.topk(route_weights, self.top_k, dim=-1)
        # 选中权重重新归一化
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # 2. 共享 K/V 投影（论文 W^k, W^v）
        k = self.w_k(x_t)  # (B, L, dim)
        v = self.w_v(x_t)  # (B, L, dim)

        # 3. ⭐ token 级稀疏：只对每个 token 选中的专家计算注意力
        out_t = torch.zeros_like(x_t)  # (B, L, C)
        for i in range(self.num_experts):
            # 收集选中专家 i 的 token 位置 (b, l, k_idx)
            sel = (topk_indices == i).nonzero()
            if sel.numel() == 0:
                continue
            b_idx, l_idx, k_idx = sel[:, 0], sel[:, 1], sel[:, 2]
            w_i = topk_weights[b_idx, l_idx, k_idx].unsqueeze(-1)  # (N, 1)

            # 专家专属 query（不同专家 → 不同注意力分布）
            q_i = self.expert_qs[i](x_t[b_idx, l_idx])  # (N, dim//2)
            # 与所属图像的全局上下文做注意力（K/V 共享）
            k_i = k[b_idx]  # (N, L, dim)
            v_i = v[b_idx]  # (N, L, dim)
            scores = torch.einsum('nd,nld->nl', q_i, k_i) * self.scale  # (N, L)
            attn_w = torch.softmax(scores, dim=-1)
            ctx = torch.einsum('nl,nld->nd', attn_w, v_i)  # (N, dim)

            # 专家输出投影 W_i^o
            e_out = self.expert_outs[i](ctx)  # (N, dim)
            out_t[b_idx, l_idx] += w_i * e_out

        # 4. 还原空间维度 + 残差
        out = self._unflatten_tokens(out_t, shape)
        return out + x

    def get_route_weights(self, x):
        """获取 token 级路由权重（用于可视化）"""
        with torch.no_grad():
            x_norm = self.norm(x)
            x_t, _ = self._flatten_tokens(x_norm)
            logits = self.router(x_t)
            return F.softmax(logits, dim=-1)

    def get_topk_indices(self, x):
        """获取 token 级 Top-K 选择（用于分析）"""
        with torch.no_grad():
            x_norm = self.norm(x)
            x_t, _ = self._flatten_tokens(x_norm)
            logits = self.router(x_t)
            weights = F.softmax(logits, dim=-1)
            topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1)
            return topk_indices, topk_weights

    def load_balancing_loss(self, route_weights):
        """token 级负载均衡损失（Switch Transformer 风格）

        Args:
            route_weights: (B, L, num_experts) token 级路由权重

        Returns:
            标量损失
        """
        # 跨 token/batch 的平均路由概率 → 每个专家的使用占比
        avg_weights = route_weights.mean(dim=(0, 1))  # (num_experts,)
        target = torch.ones_like(avg_weights) / self.num_experts
        loss = F.kl_div(
            torch.log(avg_weights + 1e-8),
            target,
            reduction='batchmean'
        )
        return loss


class SegResMambaLiteV10(SegResMambaLiteV2):
    """V10 - 论文级 MoA 版

    基于 V2 架构，仅修改 bottleneck：
    - V2: BiMambaLayer @ 8³
    - V10: 论文级 MoA（共享 K/V + Attention 专家 + token 级 Top-K）@ 8³

    其他完全继承 V2。
    """
    def __init__(self, in_channels=4, out_channels=4, init_filters=20,
                 d_state=8, d_conv=2, expand=2, use_mamba2=True,
                 use_deep_supervision=False,
                 num_experts=4, top_k=2):
        super().__init__(in_channels, out_channels, init_filters,
                         d_state, d_conv, expand, use_mamba2,
                         use_deep_supervision)

        # 删除 V2 的 mamba_8，替换为论文级 MoA
        del self.mamba_8

        # V10 创新：论文级 MoA bottleneck（token 级稀疏）
        self.moa_10 = MoABottleneckV10(
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

        # ⭐ V10 创新：bottleneck 用论文级 MoA（token 级 Top-K 稀疏）
        x3_bottleneck = self.moa_10(x3_down)

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
        """获取 token 级 MoA 路由权重（用于可视化）"""
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x1_down = self.down1(x1)
        x2 = self.enc2(x1_down)
        x2_mamba = self.mamba_32(x2)
        x2_down = self.down2(x2_mamba)
        x3 = self.enc3(x2_down)
        x3_mamba = self.mamba_16(x3)
        x3_down = self.down3(x3_mamba)
        return self.moa_10.get_route_weights(x3_down)

    def get_moa_topk_indices(self, x):
        """获取 token 级 MoA Top-K 选择（用于分析）"""
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x1_down = self.down1(x1)
        x2 = self.enc2(x1_down)
        x2_mamba = self.mamba_32(x2)
        x2_down = self.down2(x2_mamba)
        x3 = self.enc3(x2_down)
        x3_mamba = self.mamba_16(x3)
        x3_down = self.down3(x3_mamba)
        return self.moa_10.get_topk_indices(x3_down)

    def get_moa_load_balance_loss(self):
        """获取 token 级负载均衡损失（训练时调用）

        复用最近一次 forward 缓存的路由权重，避免重复前向。
        必须在模型 forward 之后调用（train_step 中），否则返回 0。
        """
        if self.moa_10.last_route_weights is None:
            return torch.tensor(0.0, device=next(self.moa_10.parameters()).device)
        return self.moa_10.load_balancing_loss(self.moa_10.last_route_weights)

    # 统一训练接口（与 V9 一致，便于 frame/train.py 调用）
    get_moe_load_balance_loss = get_moa_load_balance_loss


def get_model(in_channels=4, out_channels=4, init_filters=20,
              d_state=8, d_conv=2, expand=2, use_mamba2=True,
              use_attention=True, use_deep_supervision=False, device="cuda",
              num_experts=4, top_k=2):
    """V10 模型工厂函数

    Args:
        num_experts: MoA 专家数量（默认 4）
        top_k: 每个 token 激活的专家数（默认 2，token 级稀疏）
        其他参数与 V2 一致
    """
    model = SegResMambaLiteV10(
        in_channels, out_channels, init_filters,
        d_state, d_conv, expand, use_mamba2,
        use_deep_supervision,
        num_experts=num_experts,
        top_k=top_k,
    ).to(device)
    return model


if __name__ == "__main__":
    # 快速测试
    model = get_model(init_filters=20, device="cpu")
    total = sum(p.numel() for p in model.parameters())
    print(f"V10 参数量: {total:,} ({total/1e6:.3f}M)")
