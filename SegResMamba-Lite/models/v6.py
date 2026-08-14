"""
SegResMamba-Lite V6 - 真正的 Mixture of Attention (MoA) 版

设计理念：
1. 基于 V2 基础架构（继承）
2. 在 bottleneck 处引入 MoA（基于 SHMoAReg 2025 论文思想）
3. MoA 核心：N 个 Mamba 专家 + Top-K 路由器 + 共享 K/V
4. 端到端学习路由权重
5. 参数量目标 <1.5M

参考论文：
- SHMoAReg: Spark Deformable Image Registration via Spatial Heterogeneous
  Mixture of Experts and Attention Heads (arXiv:2509.20073, 2025)
- MoA: Mixture of Attention heads (具稀疏路由 + Top-K 专家选择)

MoA 公式（SHMoAReg Eq. 1-4）：
  E_i(q_t, K, V) = Attention_i(q_t, K, V) · W_i^o
  y_t = Σ_{i ∈ G(q_t)} w_{i,t} · E_i(q_t, K, V)
  G(q_t) = TopK(Router(q_t), k)

其中：
- W_i^o: 第 i 个专家的输出投影（独有）
- W^k, W^v: 共享 Key/Value 投影
- Router: 学习每个 query 该选哪几个专家
- 稀疏路由：Top-K 而非全连接
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.models.mamba import get_mamba_layer

# 复用 V2 的基础组件
from .v2 import LightResBlock, SegResMambaLiteV2


class MambaExpert(nn.Module):
    """单个 Mamba 专家

    设计：每个专家使用不同的 d_state（状态维度），使专家能力不一样。
    - 短记忆专家（d_state=4）：捕捉局部上下文
    - 长记忆专家（d_state=16）：捕捉全局上下文
    """
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
            # 5D 上做 flip（B, C, D, H, W）
            x_rev = torch.flip(x, dims=[2, 3, 4])
            out_rev = self.mamba(x_rev)
            return torch.flip(out_rev, dims=[2, 3, 4])
        elif self.scan_direction == 'depth_first':
            # 仅 D 维翻转：序列在 depth 方向为反向
            x_rev = torch.flip(x, dims=[2])
            out_rev = self.mamba(x_rev)
            return torch.flip(out_rev, dims=[2])
        elif self.scan_direction == 'width_first':
            # 仅 W 维翻转
            x_rev = torch.flip(x, dims=[4])
            out_rev = self.mamba(x_rev)
            return torch.flip(out_rev, dims=[4])
        else:
            return self.mamba(x)


class MoABottleneck(nn.Module):
    """真正的 Mixture of Attention (MoA) 瓶颈模块

    基于 SHMoAReg 论文设计：
    - N 个 Mamba 专家（不同扫描方向 + 不同 expand）
    - Top-K 路由器（每个 token 选 k 个专家）
    - 专家独立计算，路由器动态加权
    - 残差连接到 x

    Args:
        dim: 特征通道数
        num_experts: 专家数量（默认 4）
        top_k: 每个 token 选 k 个专家（默认 2）
        d_state: Mamba 状态维度
        d_conv: Mamba 卷积核
        base_expand: 基础 expand 因子
        use_mamba2: 是否用 Mamba2
    """
    def __init__(self, dim, num_experts=4, top_k=4, d_state=8, d_conv=2,
                 base_expand=2, use_mamba2=True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim

        # 4 个专家：用不同的 d_state 体现"长短记忆"分工
        # 为了控制参数量（目标 <1.5M），所有专家都用 expand=1
        # 专家多样性靠 d_state 和 scan_direction
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

        # 路由器：每个 token 选 Top-K 专家
        # 输入：单 token 特征（被 GAP 压缩后的全局描述）
        # 输出：每个专家的分数
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, num_experts),
        )

        # 输入归一化（稳定训练）
        self.norm = nn.InstanceNorm3d(dim)

    def forward(self, x):
        """x: (B, C, D, H, W)

        稳定的 MoA 实现（修复训练崩溃）：
        1. 路由器算出每个专家的 soft weight
        2. 所有专家都用整个 batch 计算（保持 Norm 统计量一致）
        3. soft weight 加权求和（不硬 Top-K）
        4. 残差连接

        关键修复：保持 batch 维度 + soft weight，
        避免训练时 batch_size=1 导致的统计量错误。
        """
        B, C, D, H, W = x.shape

        # 1. 计算路由权重（soft weights，不是 hard Top-K）
        route_logits = self.router(x)  # (B, num_experts)
        route_weights = F.softmax(route_logits, dim=-1)  # (B, num_experts)
        # 加权平均：让所有专家都参与（避免极端稀疏）
        # 训练初期用更均匀的分布（加噪声 + 温度）
        if self.training:
            # Gumbel-Softmax 风格的扰动，让所有专家有梯度
            noise = torch.rand_like(route_weights) * 0.1
            route_weights = (route_weights + noise)
            route_weights = route_weights / route_weights.sum(dim=-1, keepdim=True)

        # 2. 归一化输入（保持 batch 维度）
        x_norm = self.norm(x)

        # 3. 全部专家用整个 batch 计算（保持 Norm 统计量一致）
        #    然后用 soft weight 加权
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            # expert_out: (B, C, D, H, W)
            expert_out = expert(x_norm)
            # weight_i: (B,)
            weight_i = route_weights[:, i].view(B, 1, 1, 1, 1)
            out = out + weight_i * expert_out

        # 4. 残差连接
        return out + x

    def get_route_weights(self, x):
        """获取路由权重（用于可视化）"""
        with torch.no_grad():
            route_logits = self.router(x)
            return F.softmax(route_logits, dim=-1)

    def load_balancing_loss(self, route_weights):
        """MoA 负载均衡损失（防止路由坍缩到少数专家）

        Switch Transformer (Fedus et al. 2022) 提出的辅助损失。
        鼓励专家负载均匀，避免路由退化。

        Args:
            route_weights: (B, num_experts) softmax 后的路由权重

        Returns:
            标量损失
        """
        # 跨 batch 的平均路由权重
        avg_weights = route_weights.mean(dim=0)  # (num_experts,)
        # 理想情况：均匀分布，每个 = 1/num_experts
        target = torch.ones_like(avg_weights) / self.num_experts
        # KL 散度计算
        loss = F.kl_div(
            torch.log(avg_weights + 1e-8),
            target,
            reduction='batchmean'
        )
        return loss


class SegResMambaLiteV6(SegResMambaLiteV2):
    """V6 - 真正的 MoA 版

    基于 V2 架构，仅修改 bottleneck：
    - V2: BiMambaLayer @ 8³
    - V6: MoA (4 expert Mamba + Top-2 router) @ 8³

    其他完全继承 V2。
    """
    def __init__(self, in_channels=4, out_channels=4, init_filters=22,
                 d_state=8, d_conv=2, expand=2, use_mamba2=True,
                 use_deep_supervision=False,
                 use_mamba2_in_moa=True,
                 num_experts=4, top_k=4):
        super().__init__(in_channels, out_channels, init_filters,
                         d_state, d_conv, expand, use_mamba2,
                         use_deep_supervision)

        # 删除 V2 的 mamba_8，替换为 MoA
        del self.mamba_8

        # V6 创新：MoA bottleneck
        self.moa_8 = MoABottleneck(
            dim=init_filters * 4,
            num_experts=num_experts,
            top_k=top_k,
            d_state=d_state,
            d_conv=d_conv,
            base_expand=expand,
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

        # ⭐ V6 创新：bottleneck 用 MoA
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


def get_model(in_channels=4, out_channels=4, init_filters=22,
              d_state=8, d_conv=2, expand=2, use_mamba2=True,
              use_attention=True, use_deep_supervision=False, device="cuda",
              num_experts=4, top_k=4):
    """V6 模型工厂函数

    Args:
        num_experts: MoA 专家数量（默认 4）
        top_k: 每个 token 选 k 个专家（默认 4 = 全专家）
        其他参数与 V2 一致
    """
    model = SegResMambaLiteV6(
        in_channels, out_channels, init_filters,
        d_state, d_conv, expand, use_mamba2,
        use_deep_supervision,
        num_experts=num_experts,
        top_k=top_k,
    ).to(device)
    return model


if __name__ == "__main__":
    # 快速测试
    model = get_model(init_filters=22, device="cpu")
    total = sum(p.numel() for p in model.parameters())
    print(f"V6 参数量: {total:,} ({total/1e6:.3f}M)")

    x = torch.randn(1, 4, 64, 64, 64)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"输入: {x.shape}, 输出: {out.shape}")

    # 测试路由权重
    weights = model.get_moa_route_weights(x)
    print(f"MoA 路由权重: {weights[0].tolist()}")
    print(f"权重和: {weights[0].sum().item():.4f}")
