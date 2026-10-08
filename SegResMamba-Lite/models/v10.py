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
- V10 第二版（本版默认）：严格论文级 MoA，替换失败的第一版

消融开关（对应 实验记录总集.md Part F·§四，全部默认值 = V10-v2 锚点行为）：
- expert_type: 'attention'（锚点）/ 'mamba'（A3，同构 V10-v1 专家，expand=1 血统）
  / 'dconv'（H1，同构 V9 DConv 专家）
- route_granularity: 'token'（锚点，token 级 Top-K）/ 'sample'（A2，样本级路由，
  参照 v9 样本循环 + v6 GAP 路由）
- share_kv: True（锚点，论文共享 K/V）/ False（D1，每专家独立 K/V）
- route_noise: 0.0（锚点）/ 0.05（F1，训练时 Noisy Top-K Gating，Shazeer 2017）
- decoder_moa（模型级）: False（锚点，仅 8³ Bottleneck）/ True（G1，16³ 解码器侧再加一层 MoA）

⚠ 锚点纪律：默认参数下代码路径与参数量逐位复现 V10-v2
  （1,422,734 @ Mamba2SSM 训练日志口径；本机 MiniMamba 回退为 1,426,548）。

参数量目标 <1.5M
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 V2 的基础组件
from .v2 import LightResBlock, SegResMambaLiteV2
# A3 mamba 专家：与 V6/V2 同一 Mamba 层实现（血统一致）
from shared.models.mamba import get_mamba_layer
# H1 dconv 专家：直接复用 V9 的 DConvExpert
from .v9 import DConvExpert


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
        expert_type: 专家类型（'attention'/'mamba'/'dconv'，默认 'attention'）
        route_granularity: 路由粒度（'token'/'sample'，默认 'token'）
        share_kv: 是否共享 K/V 投影（默认 True；False = 每专家独立，D1）
        route_noise: 训练时路由 logits 高斯噪声 σ（默认 0.0；F1 用 0.05）
        use_mamba2: mamba 专家的后端开关（透传 get_mamba_layer）
        d_state: mamba 专家的状态维度（默认 8）
        d_conv: mamba 专家的卷积核（默认 2）
    """
    def __init__(self, dim, num_experts=4, top_k=2,
                 expert_type='attention', route_granularity='token',
                 share_kv=True, route_noise=0.0,
                 use_mamba2=True, d_state=8, d_conv=2):
        super().__init__()
        assert expert_type in ('attention', 'mamba', 'dconv'), f"未知专家类型: {expert_type}"
        assert route_granularity in ('token', 'sample'), f"未知路由粒度: {route_granularity}"
        assert 1 <= top_k <= num_experts, f"top_k={top_k} 必须在 [1, {num_experts}] 内"
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_type = expert_type
        self.route_granularity = route_granularity
        self.share_kv = share_kv
        self.route_noise = route_noise
        self.scale = (dim // 2) ** -0.5  # 注意力缩放

        # ── 专家池（按 expert_type 三选一）──
        if expert_type == 'attention':
            # 共享 K/V 投影（论文：W^k, W^v 共享；K 与专家 query 同维度 dim//2）
            if share_kv:
                self.w_k = nn.Linear(dim, dim // 2)
                self.w_v = nn.Linear(dim, dim)
            else:
                # D1 消融：每专家独立 K/V（+num_experts × (w_k + w_v) 参数）
                self.expert_ks = nn.ModuleList([
                    nn.Linear(dim, dim // 2) for _ in range(num_experts)
                ])
                self.expert_vs = nn.ModuleList([
                    nn.Linear(dim, dim) for _ in range(num_experts)
                ])
            # 每个专家独有 query 变换（不同 → 不同注意力模式）+ 独有输出投影 W_i^o
            self.expert_qs = nn.ModuleList([
                nn.Linear(dim, dim // 2) for _ in range(num_experts)
            ])
            self.expert_outs = nn.ModuleList([
                nn.Linear(dim, dim) for _ in range(num_experts)
            ])
        elif expert_type == 'mamba':
            # A3 消融：Mamba 专家（同构 V10-v1：d_state=8, d_conv=2, expand=1，
            # 对齐 V6 专家 expand=1 血统并控制参数量 ≈1.46M）
            self.expert_mambas = nn.ModuleList([
                get_mamba_layer(dim=dim, d_state=d_state, d_conv=d_conv,
                                expand=1, use_mamba2=use_mamba2)
                for _ in range(num_experts)
            ])
        else:  # dconv
            # H1 消融：DConv 专家（同构 V9 专家池，kernel/dilation 循环差异化）
            dconv_cfgs = [(3, 1), (3, 2), (5, 1), (3, 1)]
            self.expert_dconvs = nn.ModuleList([
                DConvExpert(dim, kernel_size=dconv_cfgs[i % 4][0],
                            dilation=dconv_cfgs[i % 4][1])
                for i in range(num_experts)
            ])

        # ── 路由器（token 级直接打分；sample 级在 GAP 后打分，同一线性层）──
        self.router = nn.Linear(dim, num_experts)

        self.norm = nn.InstanceNorm3d(dim)

        # 最近一次 forward 的路由权重缓存（供训练时计算负载均衡损失，避免重复前向）
        self.last_route_weights = None

    def _pre_route_enhance(self, x_norm):
        """路由前的特征增强钩子（V12 在此插入边界注意力；V10 默认恒等）"""
        return x_norm

    def _flatten_tokens(self, x):
        """(B, C, D, H, W) → (B, L, C)"""
        B, C, D, H, W = x.shape
        return x.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C), (B, C, D, H, W)

    def _unflatten_tokens(self, x_t, shape):
        """(B, L, C) → (B, C, D, H, W)"""
        B, C, D, H, W = shape
        return x_t.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

    def _run_expert_full(self, i, x_norm):
        """mamba/dconv 专家的整图前向（(B, C, D, H, W) → 同形状）"""
        if self.expert_type == 'mamba':
            return self.expert_mambas[i](x_norm)
        return self.expert_dconvs[i](x_norm)

    def _route_logits_noisy(self, route_logits):
        """训练时按 route_noise 加高斯噪声（Noisy Top-K Gating，推理恒不加）"""
        if self.training and self.route_noise > 0:
            route_logits = route_logits + torch.randn_like(route_logits) * self.route_noise
        return route_logits

    def forward(self, x):
        """x: (B, C, D, H, W)

        按路由粒度分发：
        - token 级（锚点）：每个 token 独立选 Top-K 专家，只计算选中专家
        - sample 级（A2）：GAP 路由，每样本选 Top-K 专家
        """
        x_norm = self.norm(x)
        x_norm = self._pre_route_enhance(x_norm)
        if self.route_granularity == 'token':
            return self._forward_token(x, x_norm)
        return self._forward_sample(x, x_norm)

    def _forward_token(self, x, x_norm):
        """token 级稀疏 MoA（锚点路径，默认参数下与 V10-v2 逐位一致）

        论文级 token 稀疏 MoA：
        1. token 级路由：每个 token 独立选 Top-K 专家
        2. 共享 K/V 投影
        3. ⭐ token 级稀疏：只对选中专家的 token 计算注意力（未选中不计算）
        4. 选中权重加权融合 + 残差
        """
        x_t, shape = self._flatten_tokens(x_norm)  # (B, L, C)
        B, L, C = x_t.shape

        # 1. token 级路由（论文 G(q_t) = TopK(Router(q_t), k)）
        route_logits = self.router(x_t)  # (B, L, num_experts)
        route_logits = self._route_logits_noisy(route_logits)
        route_weights = F.softmax(route_logits, dim=-1)  # (B, L, num_experts)
        self.last_route_weights = route_weights

        topk_weights, topk_indices = torch.topk(route_weights, self.top_k, dim=-1)
        # 选中权重重新归一化
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # 2. Attention 专家的共享 K/V 投影（论文 W^k, W^v）
        if self.expert_type == 'attention' and self.share_kv:
            k = self.w_k(x_t)  # (B, L, dim//2)
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

            if self.expert_type == 'attention':
                # 专家专属 query（不同专家 → 不同注意力分布）
                q_i = self.expert_qs[i](x_t[b_idx, l_idx])  # (N, dim//2)
                # 与所属图像的全局上下文做注意力（K/V 共享或每专家独立）
                if self.share_kv:
                    # ⭐ 分 batch 块 + SDPA：与 k[b_idx] gather 后计算数学等价
                    # （每个 token 只对自己所属样本的全部 L 个 K/V 做注意力）。
                    # SDPA(flash/mem-efficient 内核) backward 不实体化 (N, L) 注意力
                    # 矩阵——16³ 解码器侧（L=4096）可省 ~0.8–1.6 GiB，避免 WDDM
                    # 显存溢出到内存条导致的 paging 病态慢
                    ctx = x_t.new_zeros((q_i.shape[0], v.shape[-1]))  # (N, dim)
                    for b in b_idx.unique():
                        m_b = b_idx == b                              # 该样本选中的 token 掩码
                        q_b = q_i[m_b].unsqueeze(1)                   # (N_b, 1, dim//2)
                        ctx_b = F.scaled_dot_product_attention(
                            q_b, k[b].unsqueeze(0), v[b].unsqueeze(0),
                            scale=self.scale)                         # (N_b, 1, dim)
                        ctx[m_b] = ctx_b.squeeze(1)
                else:
                    k_i = self.expert_ks[i](x_t[b_idx])  # (N, L, dim//2)
                    v_i = self.expert_vs[i](x_t[b_idx])  # (N, L, dim)
                    scores = torch.einsum('nd,nld->nl', q_i, k_i) * self.scale  # (N, L)
                    attn_w = torch.softmax(scores, dim=-1)
                    ctx = torch.einsum('nl,nld->nd', attn_w, v_i)  # (N, dim)

                # 专家输出投影 W_i^o
                e_out = self.expert_outs[i](ctx)  # (N, dim)
            else:
                # mamba/dconv 专家：整图前向后收集选中 token 输出
                # （参照 V11 对 DConv 专家的处理，未选中 token 梯度自然截断）
                e_full = self._run_expert_full(i, x_norm)  # (B, C, D, H, W)
                e_t = self._flatten_tokens(e_full)[0]  # (B, L, C)
                e_out = e_t[b_idx, l_idx]  # (N, dim)

            out_t[b_idx, l_idx] += w_i * e_out

        # 4. 还原空间维度 + 残差
        out = self._unflatten_tokens(out_t, shape)
        return out + x

    def _forward_sample(self, x, x_norm):
        """样本级稀疏 MoA（A2：Attention 专家 + 样本级路由）

        参照 v9 样本循环与 v6 GAP 路由：
        1. GAP → 路由器 → (B, num_experts) softmax → Top-K → renorm
        2. Attention 专家：对样本全部 token 用选中专家做注意力
           mamba/dconv 专家：整图前向后加权（与 V9 逐样本循环一致）
        3. 残差
        """
        # 1. 样本级路由（GAP → Linear → Softmax）
        pooled = x_norm.mean(dim=(2, 3, 4))  # (B, C)
        route_logits = self.router(pooled)  # (B, num_experts)
        route_logits = self._route_logits_noisy(route_logits)
        route_weights = F.softmax(route_logits, dim=-1)  # (B, num_experts)
        self.last_route_weights = route_weights

        topk_weights, topk_indices = torch.topk(route_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)

        B = x_norm.shape[0]

        if self.expert_type == 'attention':
            x_t, shape = self._flatten_tokens(x_norm)  # (B, L, C)
            out_t = torch.zeros_like(x_t)  # (B, L, C)
            # 共享 K/V 只需计算一次（独立 K/V 在专家循环内计算）
            if self.share_kv:
                k_all = self.w_k(x_t)  # (B, L, dim//2)
                v_all = self.w_v(x_t)  # (B, L, dim)
            # 2. ⭐ 样本级稀疏：每样本只对选中的 Top-K 专家计算
            for b in range(B):
                for k_i in range(self.top_k):
                    e = int(topk_indices[b, k_i].item())
                    w = topk_weights[b, k_i]
                    # 专家专属 query（对样本全部 token 计算注意力）
                    q = self.expert_qs[e](x_t[b])  # (L, dim//2)
                    if self.share_kv:
                        kb, vb = k_all[b], v_all[b]  # (L, dim//2), (L, dim)
                    else:
                        kb = self.expert_ks[e](x_t[b])  # (L, dim//2)
                        vb = self.expert_vs[e](x_t[b])  # (L, dim)
                    scores = torch.einsum('ld,md->lm', q, kb) * self.scale  # (L, L)
                    attn_w = torch.softmax(scores, dim=-1)
                    ctx = torch.einsum('lm,md->ld', attn_w, vb)  # (L, dim)
                    out_t[b] += w * self.expert_outs[e](ctx)  # (L, dim)
            out = self._unflatten_tokens(out_t, shape)
            return out + x

        # mamba/dconv 专家：逐样本稀疏前向（与 V9 循环一致）
        out = torch.zeros_like(x_norm)
        for b in range(B):
            for k_i in range(self.top_k):
                e = int(topk_indices[b, k_i].item())
                expert_out = self._run_expert_full(e, x_norm[b:b + 1])
                out[b] += topk_weights[b, k_i] * expert_out[0]
        return out + x

    def get_route_weights(self, x):
        """获取路由权重（用于可视化）

        Returns:
            token 粒度: (B, L, num_experts)；sample 粒度: (B, num_experts)
        """
        with torch.no_grad():
            x_norm = self._pre_route_enhance(self.norm(x))
            if self.route_granularity == 'sample':
                pooled = x_norm.mean(dim=(2, 3, 4))
                return F.softmax(self.router(pooled), dim=-1)
            x_t, _ = self._flatten_tokens(x_norm)
            logits = self.router(x_t)
            return F.softmax(logits, dim=-1)

    def get_topk_indices(self, x):
        """获取 Top-K 选择（用于分析）

        Returns:
            (topk_indices, topk_weights)；
            token 粒度: (B, L, k)；sample 粒度: (B, k)
        """
        with torch.no_grad():
            x_norm = self._pre_route_enhance(self.norm(x))
            if self.route_granularity == 'sample':
                pooled = x_norm.mean(dim=(2, 3, 4))
                weights = F.softmax(self.router(pooled), dim=-1)
            else:
                x_t, _ = self._flatten_tokens(x_norm)
                weights = F.softmax(self.router(x_t), dim=-1)
            topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1)
            return topk_indices, topk_weights

    def load_balancing_loss(self, route_weights):
        """负载均衡损失（Switch Transformer 风格，token/sample 粒度通用）

        Args:
            route_weights: (B, L, num_experts) 或 (B, num_experts) 路由权重

        Returns:
            标量损失
        """
        # 跨 batch/token 的平均路由概率 → 每个专家的使用占比
        # token 粒度 (B, L, N)：对 batch+token 维求均值；sample 粒度 (B, N)：仅对 batch 维求均值
        if route_weights.dim() == 3:
            avg_weights = route_weights.mean(dim=(0, 1))  # (num_experts,)
        else:
            avg_weights = route_weights.mean(dim=0)  # (num_experts,)
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
    - decoder_moa=True（G1 消融）：16³ 解码器侧 dec2 之后再加一层同配置 MoA

    其他完全继承 V2。
    """
    def __init__(self, in_channels=4, out_channels=4, init_filters=20,
                 d_state=8, d_conv=2, expand=2, use_mamba2=True,
                 use_deep_supervision=False,
                 num_experts=4, top_k=2,
                 expert_type='attention', route_granularity='token',
                 share_kv=True, route_noise=0.0, decoder_moa=False):
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
            expert_type=expert_type,
            route_granularity=route_granularity,
            share_kv=share_kv,
            route_noise=route_noise,
            use_mamba2=use_mamba2,
            d_state=d_state,
            d_conv=d_conv,
        )

        # G1 消融：16³ 解码器侧（dec2 之后）再加一层同配置 MoA
        self.decoder_moa = decoder_moa
        if decoder_moa:
            self.moa_dec16 = MoABottleneckV10(
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

        # G1 消融：16³ 解码器侧 MoA（在深层监督头之前）
        if self.decoder_moa:
            x2_dec = self.moa_dec16(x2_dec)

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
        """获取负载均衡损失（训练时调用；G1 解码器侧 MoA 存在时两项求和）

        复用最近一次 forward 缓存的路由权重，避免重复前向。
        必须在模型 forward 之后调用（train_step 中），否则对应项为 0。
        """
        device = next(self.moa_10.parameters()).device
        loss = torch.tensor(0.0, device=device)
        if self.moa_10.last_route_weights is not None:
            loss = loss + self.moa_10.load_balancing_loss(self.moa_10.last_route_weights)
        if self.decoder_moa and self.moa_dec16.last_route_weights is not None:
            loss = loss + self.moa_dec16.load_balancing_loss(self.moa_dec16.last_route_weights)
        return loss

    # 统一训练接口（与 V9 一致，便于 frame/train.py 调用）
    get_moe_load_balance_loss = get_moa_load_balance_loss


def get_model(in_channels=4, out_channels=4, init_filters=20,
              d_state=8, d_conv=2, expand=2, use_mamba2=True,
              use_attention=True, use_deep_supervision=False, device="cuda",
              num_experts=4, top_k=2,
              expert_type='attention', route_granularity='token',
              share_kv=True, route_noise=0.0, decoder_moa=False):
    """V10 模型工厂函数

    Args:
        num_experts: MoA 专家数量（默认 4）
        top_k: 每个 token 激活的专家数（默认 2，token 级稀疏）
        expert_type: 专家类型（'attention'/'mamba'/'dconv'，默认 'attention'）
        route_granularity: 路由粒度（'token'/'sample'，默认 'token'）
        share_kv: 是否共享 K/V（默认 True）
        route_noise: 训练时路由噪声 σ（默认 0.0）
        decoder_moa: 是否启用 16³ 解码器侧 MoA（G1，默认 False）
        其他参数与 V2 一致
    """
    model = SegResMambaLiteV10(
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
    # 快速测试（锚点配置）
    model = get_model(init_filters=20, device="cpu")
    total = sum(p.numel() for p in model.parameters())
    print(f"V10 参数量: {total:,} ({total/1e6:.3f}M)")
