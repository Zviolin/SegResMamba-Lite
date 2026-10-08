"""
token 级路由可视化（实验记录总集.md Part F·可视化 V2 → 论文 Fig 3）

内容：
  ① 专家利用率直方图：Top-1 指派占比（E1 负载均衡关闭时的专家坍缩 vs E0 均衡对照）
  ② 指定切片的 Top-1 专家指派图 + 各专家路由权重热图（仅 token 级路由；
     sample 级路由模型只有样本级权重，自动跳过切片热图）

用法（在 SegResMamba-Lite 目录下执行）：
  # 随机张量演示（无 checkpoint，仅验证脚本）
  python scripts/visualize_routing.py --version v10 --output_dir routing_vis/

  # 加载训练好的 checkpoint + 真实病例
  python scripts/visualize_routing.py --version v10 \
      --checkpoint pipeline/models/2.0mm_v10_cuda/best_metric_model.pth \
      --input <预处理的 (4,64,64,64) .npy 或 .pt> \
      --slice_z 4 --output_dir routing_vis/

输入约定：
  --input 支持 .npy / .pt；形状接受 (4, 64, 64, 64)（C,D,H,W）或
  (64, 64, 64, 4)（D,H,W,C，自动转置）；未提供时用固定种子随机张量演示。
"""
import os
import sys
import argparse

# matplotlib 无头后端（服务器/无显示器环境可用）
import matplotlib
matplotlib.use("Agg")
# 中文字体（Windows 优先雅黑/黑体，缺失时回退 DejaVu）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch

# 路径配置：版本根目录（models 包）+ Code 目录（shared.models.mamba 需要）
version_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
code_root = os.path.dirname(version_root)
for _p in (version_root, code_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import get_model


def load_input(input_path, device):
    """加载输入张量并归一化为 (1, 4, 64, 64, 64)"""
    if input_path is None:
        # 演示模式：固定种子随机张量（未含 batch 维）
        x = torch.randn(4, 64, 64, 64, generator=torch.Generator().manual_seed(0))
        print("未提供 --input，使用随机张量演示")
    elif input_path.endswith(".npy"):
        arr = np.load(input_path)
        x = torch.from_numpy(arr).float()
    elif input_path.endswith(".pt"):
        x = torch.load(input_path, map_location="cpu")
        if isinstance(x, dict):
            # 常见缓存格式兼容：取第一个张量值
            x = next(v for v in x.values() if torch.is_tensor(v)).float()
        else:
            x = x.float()
    else:
        raise ValueError(f"不支持的输入格式: {input_path}（仅 .npy/.pt）")

    # 形状归一化: (D,H,W,C) → (C,D,H,W)，再补 batch 维
    if x.dim() == 4 and x.shape[-1] == 4:
        x = x.permute(3, 0, 1, 2)
    assert x.dim() == 4 and x.shape[0] == 4, f"输入应为 (4,D,H,W) 或 (D,H,W,4)，实得 {tuple(x.shape)}"
    return x.unsqueeze(0).to(device)


def load_model(version, checkpoint, init_filters, num_experts, top_k, device,
               expert_type='attention', route_granularity='token',
               share_kv=True, decoder_moa=False):
    """构建模型并可选加载 checkpoint 权重（消融参数须与训练时一致）"""
    model = get_model(version=version, init_filters=init_filters,
                      use_deep_supervision=False, device=device,
                      num_experts=num_experts, top_k=top_k,
                      expert_type=expert_type,
                      route_granularity=route_granularity,
                      share_kv=share_kv, decoder_moa=decoder_moa)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        # 兼容不同保存格式：直接 state_dict / 包裹 dict
        if isinstance(ckpt, dict):
            state = None
            for key in ("model_state_dict", "state_dict", "model", "net"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]
                    break
            if state is None:
                state = ckpt
        else:
            state = ckpt
        # 去除 DataParallel 前缀
        state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"已加载 checkpoint: {checkpoint}")
        if missing:
            print(f"  缺失键 {len(missing)} 个（如为有意省略可忽略）")
        if unexpected:
            print(f"  多余键 {len(unexpected)} 个")
    model.eval()
    return model


def plot_utilization(top1_assign, num_experts, out_path):
    """专家利用率直方图（Top-1 指派占比）"""
    counts = np.bincount(top1_assign, minlength=num_experts)
    ratio = counts / counts.sum() * 100.0
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(range(num_experts), ratio, color="#4C72B0", edgecolor="white")
    for bar, r in zip(bars, ratio):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{r:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(range(num_experts))
    ax.set_xticklabels([f"E{i}" for i in range(num_experts)])
    ax.set_ylabel("Top-1 指派占比 (%)")
    ax.set_title("MoA 专家利用率（token 级 Top-1）")
    ax.set_ylim(0, max(ratio.max() * 1.25, 10))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return ratio


def plot_slice_routing(top1_map, weights_map, num_experts, slice_z, out_path):
    """指定切片的 Top-1 专家指派图 + 各专家路由权重热图

    Args:
        top1_map: (D, H, W) Top-1 专家索引
        weights_map: (D, H, W, num_experts) 路由权重
        num_experts: 专家数
        slice_z: 切片索引（D 维）
        out_path: 输出 PNG 路径
    """
    fig, axes = plt.subplots(2, num_experts, figsize=(3 * num_experts, 6.2))
    # 上行：各专家路由权重热图
    vmax = float(weights_map[slice_z].max())
    for e in range(num_experts):
        ax = axes[0, e]
        im = ax.imshow(weights_map[slice_z, :, :, e], cmap="viridis",
                       vmin=0, vmax=max(vmax, 1e-6))
        ax.set_title(f"E{e} weight", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    # 下行：Top-1 指派离散图（同一配色贯穿）
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(num_experts)
    ax = axes[1, 0]
    ax.imshow(top1_map[slice_z], cmap=cmap, vmin=-0.5, vmax=num_experts - 0.5,
              interpolation="nearest")
    ax.set_title("Top-1 assign", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    # 其余下排子图隐藏（指派图只占一个）
    for e in range(1, num_experts):
        axes[1, e].axis("off")
    fig.suptitle(f"MoA token 级路由切片可视化（slice z={slice_z}, 8³ bottleneck）")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    """入口：解析参数 → 加载模型与输入 → 路由前向 → 出图"""
    parser = argparse.ArgumentParser(description="MoA token 级路由可视化（论文 Fig 3）")
    parser.add_argument("--version", type=str, default="v10", choices=["v10", "v12"],
                        help="模型版本（v10/v12）")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint 路径（可选）")
    parser.add_argument("--input", type=str, default=None, help="输入 .npy/.pt（缺省随机张量演示）")
    parser.add_argument("--init_filters", type=int, default=20, help="初始滤波器数量")
    parser.add_argument("--num_experts", type=int, default=4, help="MoA 专家数量")
    parser.add_argument("--top_k", type=int, default=2, help="Top-K 激活专家数")
    parser.add_argument("--expert_type", type=str, default="attention",
                        choices=["attention", "mamba", "dconv"], help="V10 专家类型")
    parser.add_argument("--route_granularity", type=str, default="token",
                        choices=["token", "sample"], help="V10 路由粒度")
    parser.add_argument("--share_kv", type=int, default=1, choices=[0, 1], help="V10 是否共享 K/V")
    parser.add_argument("--decoder_moa", action="store_true", help="V10 G1：16³ 解码器侧追加 MoA")
    parser.add_argument("--slice_z", type=int, default=None, help="可视化切片索引（缺取中间层）")
    parser.add_argument("--output_dir", type=str, default="routing_vis", help="输出目录")
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda（缺省自动）")
    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir if os.path.isabs(args.output_dir) \
        else os.path.join(version_root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    model = load_model(args.version, args.checkpoint, args.init_filters,
                       args.num_experts, args.top_k, device,
                       expert_type=args.expert_type,
                       route_granularity=args.route_granularity,
                       share_kv=bool(args.share_kv),
                       decoder_moa=args.decoder_moa)
    x = load_input(args.input, device)

    with torch.no_grad():
        indices, topk_weights = model.get_moa_topk_indices(x)
        weights = model.get_moa_route_weights(x)

    granularity = model.moa_10.route_granularity
    print(f"路由粒度: {granularity}; 路由权重形状: {tuple(weights.shape)}")

    if granularity == "token":
        # (1, L, N) → (D, H, W, N)
        w_map = weights[0].reshape(8, 8, 8, args.num_experts).cpu().numpy()
        top1_map = indices[0, :, 0].reshape(8, 8, 8).cpu().numpy()
        slice_z = args.slice_z if args.slice_z is not None else 4
        slice_z = max(0, min(7, slice_z))
        util_path = os.path.join(out_dir, "expert_utilization.png")
        ratio = plot_utilization(top1_map.reshape(-1), args.num_experts, util_path)
        print(f"专家利用率: " + ", ".join(f"E{i}={r:.1f}%" for i, r in enumerate(ratio)))
        heat_path = os.path.join(out_dir, f"routing_slice_z{slice_z}.png")
        plot_slice_routing(top1_map, w_map, args.num_experts, slice_z, heat_path)
        print(f"已输出: {util_path}")
        print(f"已输出: {heat_path}")
    else:
        # sample 级路由：无 token 级切片热图，仅样本级利用率
        top1_sample = indices[0].cpu().numpy()  # (top_k,)
        util_path = os.path.join(out_dir, "expert_utilization_sample.png")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        w = weights[0].cpu().numpy()
        ax.bar(range(len(w)), w, color="#4C72B0", edgecolor="white")
        ax.set_xticks(range(len(w)))
        ax.set_xticklabels([f"E{i}" for i in range(len(w))])
        ax.set_ylabel("样本级路由权重")
        ax.set_title(f"MoA 样本级路由权重（Top-{args.top_k}: {list(top1_sample)}）")
        fig.tight_layout()
        fig.savefig(util_path, dpi=200)
        plt.close(fig)
        print(f"sample 粒度无切片热图；已输出: {util_path}")

    print("可视化完成")


if __name__ == "__main__":
    main()
