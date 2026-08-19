"""
SegResMamba-Lite V9 - 真正的 MoE (Top-K Sparse Mixture of Experts) 版

设计理念：
1. 基于 V2 基础架构（继承）
2. 在 bottleneck 处引入真正的 MoE（区别于 V6 的 MoA）
3. V9 的关键创新：
 - Top-K=2 稀疏激活（节省 50% 计算）
 - 4 个差异化 DConv 专家（不同 kernel/dilation）
 - 真正的稀疏计算：每样本只对选中的专家 forward，未选中专家不更新
 - 软路由权重（带梯度，保证路由器可学习、训练稳定）
 - 负载均衡损失（防止路由坍缩）
4. 参数量目标 <1.5M

与 V6/V7/V8 的区别：
- V6/V7/V8: Soft MoA (所有专家激活 + 加权求和) = 100% 计算
- V9: Top-K MoE (只激活 Top-2 专家) = 50% 计算
- V9 优势：计算效率高，保留路由灵活性
- V9 风险：稀疏激活可能导致训练不稳定

参考论文：
- Shazeer et al. 2017 "Outrageously Large Neural Networks" (Sparsely-Gated MoE)
- Switch Transformer (Fedus et al. 2022)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 V2 的基础组件
from .v2 import LightResBlock, SegResMambaLiteV2


class DConvExpert(nn.Module):
    """单个 DConv 专家（不同 kernel/dilation 体现差异化）"""
    def __init__(self, dim, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        # Depthwise 3D Conv
        self.dw_conv = nn.Conv3d(
            dim, dim, kernel_size,
            padding=padding, dilation=dilation, groups=dim
        )
        # Pointwise 3D Conv
        self.pw_conv = nn.Conv3d(dim, dim, 1)
        self.norm = nn.InstanceNorm3d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        x_norm = self.norm(x)
        out = self.dw_conv(x_norm)
        out = self.pw_conv(out)
        # 残差 + 激活
        out = self.act(out + x_norm)
        return out + x


class MoEBottleneckV9(nn.Module):
    """真正的 MoE 瓶颈模块（Top-K 稀疏激活）

    设计：
    - 4 个 DConv 专家（不同 kernel/dilation）
    - 路由器（GAP → MLP → Softmax）
    - Top-K=2 稀疏激活：每样本只计算选中的专家（真正的稀疏 MoE）
    - 负载均衡损失（防止路由坍缩）
    - 选中专家用 renormalized soft 权重融合

    Args:
        dim: 特征通道数
        num_experts: 专家数量（默认 4）
        top_k: 每次激活的专家数（默认 2）
    """
    def __init__(self, dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim

        # 4 个差异化专家
        self.experts = nn.ModuleList([
            DConvExpert(dim, kernel_size=3, dilation=1),  # 标准
            DConvExpert(dim, kernel_size=3, dilation=2),  # 扩大感受野
            DConvExpert(dim, kernel_size=5, dilation=1),  # 更大卷积核
            DConvExpert(dim, kernel_size=3, dilation=1),  # 标准（边界细化）
        ])

        # 路由器：GAP → MLP → Softmax
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, num_experts),
        )

        self.norm = nn.InstanceNorm3d(dim)

        # 最近一次 forward 的路由权重缓存（供训练时计算负载均衡损失，避免重复前向）
        self.last_route_weights = None

    def forward(self, x):
        """x: (B, C, D, H, W)

        真正的 Top-K 稀疏 MoE（每样本只激活选中的专家）：
        1. 路由器算出每个专家的 softmax 权重（保留梯度供路由器学习）
        2. Top-K 选择每个样本激活的专家
        3. ⭐ 只对选中的专家执行 forward —— 未选中专家不计算、梯度不回流
        4. 用重新归一化的 Top-K 权重加权融合
        5. 残差连接

        与 V6 MoA 的本质区别：
        - V6: 计算全部 4 个专家 + 全部 soft 加权 = 100% 计算，所有专家都更新
        - V9: 每样本只计算 Top-2 专家 = 50% 计算，未选中专家参数不更新
        """
        B, C, D, H, W = x.shape

        # 1. 路由权重（带梯度：软路由权重保证路由器可学习、训练稳定）
        route_logits = self.router(x)  # (B, num_experts)
        route_weights = F.softmax(route_logits, dim=-1)  # (B, num_experts)
        # 缓存本次路由权重，供训练时计算负载均衡损失（避免重复前向）
        self.last_route_weights = route_weights

        # 2. Top-K 选择
        topk_weights, topk_indices = torch.topk(route_weights, self.top_k, dim=-1)
        # 选中权重重新归一化（Switch Transformer 风格）
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # 3. 归一化输入（InstanceNorm 按样本统计，逐样本送入不影响统计量）
        x_norm = self.norm(x)

        # 4. ⭐ 真正的稀疏激活：只对每个样本选中的 Top-K 专家做 forward
        #    未选中专家在本次迭代中完全不参与计算（50% 计算量，参数不更新）
        out = torch.zeros_like(x)
        for b in range(B):
            for k in range(self.top_k):
                expert_idx = int(topk_indices[b, k].item())
                expert_out = self.experts[expert_idx](x_norm[b:b + 1])
                out[b] = out[b] + topk_weights[b, k] * expert_out[0]

        # 5. 残差连接
        return out + x

    def get_route_weights(self, x):
        """获取路由权重（用于可视化）"""
        with torch.no_grad():
            route_logits = self.router(x)
            return F.softmax(route_logits, dim=-1)

    def get_topk_indices(self, x):
        """获取 Top-K 选择的索引（用于分析）"""
        with torch.no_grad():
            route_logits = self.router(x)
            weights = F.softmax(route_logits, dim=-1)
            topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1)
            return topk_indices, topk_weights

    def load_balancing_loss(self, route_weights):
        """MoE 负载均衡损失

        Switch Transformer (Fedus et al. 2022) 提出的辅助损失。
        鼓励专家负载均匀，避免路由坍缩到少数专家。

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


class SegResMambaLiteV9(SegResMambaLiteV2):
    """V9 - 真正的 MoE 版

    基于 V2 架构，仅修改 bottleneck：
    - V2: BiMambaLayer @ 8³
    - V9: MoE (4 DConv experts + Top-2 router) @ 8³

    其他完全继承 V2。
    """
    def __init__(self, in_channels=4, out_channels=4, init_filters=22,
                 d_state=8, d_conv=2, expand=2, use_mamba2=True,
                 use_deep_supervision=False,
                 num_experts=4, top_k=2):
        super().__init__(in_channels, out_channels, init_filters,
                         d_state, d_conv, expand, use_mamba2,
                         use_deep_supervision)

        # 删除 V2 的 mamba_8，替换为 MoE
        del self.mamba_8

        # V9 创新：MoE bottleneck（Top-K 稀疏激活）
        self.moe_8 = MoEBottleneckV9(
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

        # ⭐ V9 创新：bottleneck 用真正的 MoE（Top-K=2）
        x3_bottleneck = self.moe_8(x3_down)

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

    def get_moe_route_weights(self, x):
        """获取 MoE 路由权重（用于可视化）"""
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x1_down = self.down1(x1)
        x2 = self.enc2(x1_down)
        x2_mamba = self.mamba_32(x2)
        x2_down = self.down2(x2_mamba)
        x3 = self.enc3(x2_down)
        x3_mamba = self.mamba_16(x3)
        x3_down = self.down3(x3_mamba)
        return self.moe_8.get_route_weights(x3_down)

    def get_moe_topk_indices(self, x):
        """获取 MoE Top-K 选择"""
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x1_down = self.down1(x1)
        x2 = self.enc2(x1_down)
        x2_mamba = self.mamba_32(x2)
        x2_down = self.down2(x2_mamba)
        x3 = self.enc3(x2_down)
        x3_mamba = self.mamba_16(x3)
        x3_down = self.down3(x3_mamba)
        return self.moe_8.get_topk_indices(x3_down)

    def get_moe_load_balance_loss(self):
        """获取 MoE 负载均衡损失（训练时调用）

        复用最近一次 forward 缓存的路由权重，避免重复前向计算。
        必须在模型 forward 之后调用（train_step 中），否则返回 0。

        Returns:
            标量损失（带梯度，可反向传播到路由器）
        """
        if self.moe_8.last_route_weights is None:
            return torch.tensor(0.0, device=next(self.moe_8.parameters()).device)
        return self.moe_8.load_balancing_loss(self.moe_8.last_route_weights)


def get_model(in_channels=4, out_channels=4, init_filters=22,
              d_state=8, d_conv=2, expand=2, use_mamba2=True,
              use_attention=True, use_deep_supervision=False, device="cuda",
              num_experts=4, top_k=2):
    """V9 模型工厂函数

    Args:
        num_experts: MoE 专家数量（默认 4）
        top_k: Top-K 激活数量（默认 2，节省 50% 计算）
        其他参数与 V2 一致
    """
    model = SegResMambaLiteV9(
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
    print(f"V9 参数量: {total:,} ({total/1e6:.3f}M)")

    x = torch.randn(1, 4, 64, 64, 64)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"输入: {x.shape}, 输出: {out.shape}")

    # 测试路由权重
    weights = model.get_moe_route_weights(x)
    print(f"MoE 路由权重: {weights[0].tolist()}")
    print(f"权重和: {weights[0].sum().item():.4f}")

    # 测试 Top-K
    topk_indices, topk_weights = model.get_moe_topk_indices(x)
    print(f"Top-K 索引: {topk_indices[0].tolist()}")
    print(f"Top-K 权重: {topk_weights[0].tolist()}")