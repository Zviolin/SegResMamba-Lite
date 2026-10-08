"""
效率与部署基准脚本（实验记录总集.md Part F·效率基准 → 论文 Table 6）

对象：
  自研: v2, v6, v7, v10, v11（+ v12 附带）——按源码实例化（实验配置 init_filters=20）
  基线: SegResNet（DiceFocal 基线）、VMUNet、VSSUNet —— 动态加载各项目 models/get_model
        （SegMamba 74.87M 不实测，论文引用官方数字）

指标（64³ 输入，batch=1，单卡）：
  ① 参数量（total/trainable；论文引用一律以训练日志 total_params 为准）
  ② FLOPs（thop，模型含自定义算子时可能无法统计，标 N/A 并注明）
  ③ 峰值显存（torch.cuda.max_memory_allocated）
  ④ 单例推理延迟（fp32 与 fp16 各：预热 10 次 + 测 100 次取均值；--quick 为预热 2 + 测 10）

用法：
  python bench_efficiency.py                 # 全部对象
  python bench_efficiency.py --quick         # 冒烟模式（迭代次数减少）
  python bench_efficiency.py --out Table6.md # 结果另存 markdown

注意：
  - 显存/延迟需要 CUDA；无 GPU 时仅报告参数量与 FLOPs
  - FLOPs 对含 Mamba/Top-K 稀疏路由的模型仅统计可静态追踪算子，
    与实际计算量存在偏差，仅作同构参考
"""
import os
import sys
import time
import copy
import argparse
import importlib.util

# 路径配置：版本根目录（models 包）+ Code 目录（shared 及各基线项目）
version_root = os.path.dirname(os.path.abspath(__file__))
code_root = os.path.dirname(version_root)
for _p in (version_root, code_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

# 自研对象: (名称, 版本, 实验配置 kwargs)
OURS = [
    ("V2",  "v2",  {}),
    ("V6",  "v6",  {"num_experts": 4}),
    ("V7",  "v7",  {}),
    ("V10", "v10", {"num_experts": 4, "top_k": 2}),
    ("V11", "v11", {}),
    ("V12", "v12", {"num_experts": 4, "top_k": 2}),
]

# 基线对象: (名称, 项目目录, get_model 参数)
BASELINES = [
    ("SegResNet (DiceFocal)", "SegResNet-DiceFocal-AdamW-SWA-DS",
     {"model_name": "segresnet", "in_channels": 4, "out_channels": 4}),
    ("VMUNet", "VMUNet-DiceCE-AdamW", {"model_name": "vm_unet"}),
    ("VSSUNet", "VSSUNet-DiceCE-AdamW",
     {"model_name": "vss_unet", "in_channels": 4, "out_channels": 4}),
]


def load_baseline_models_module(project_dir):
    """动态加载基线项目的 models 包（独立命名，避免与本项目 models 包冲突）"""
    path = os.path.join(code_root, project_dir, "models", "__init__.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(
        f"bench_{project_dir.replace('-', '_')}_models", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def count_params(model):
    """统计 total / trainable 参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_flops(model, x):
    """thop 统计 FLOPs（失败返回 None，调用方标 N/A）

    注意：必须在 no_grad 下运行——thop 默认保留 autograd 图，
    64³ 前向 + 自研模型的多循环结构会慢到不可用。
    """
    try:
        from thop import profile
        with torch.no_grad():
            flops, _ = profile(copy.deepcopy(model), inputs=(x,), verbose=False)
        return flops
    except Exception as exc:
        print(f"    [thop 不可用: {type(exc).__name__}]")
        return None


def measure_latency(model, x, warmup, iters):
    """推理延迟（毫秒）：预热 warmup 次 + 计时 iters 次取均值（fp32 前提下调用）"""
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return elapsed / iters * 1000.0


def bench_one(name, build_fn, x, warmup, iters, device):
    """对单个模型执行完整基准并返回结果行

    Args:
        name: 展示名称
        build_fn: 无参构建函数（返回 nn.Module，已迁移到 device）
        x: 输入张量（已在 device 上）
        warmup: 预热次数
        iters: 计时次数
        device: 'cuda' / 'cpu'

    Returns:
        dict: 基准结果行
    """
    model = build_fn()
    model.eval()
    total, trainable = count_params(model)
    row = {"name": name, "total": total, "trainable": trainable,
           "flops": None, "mem": None, "lat32": None, "lat16": None}

    # ② FLOPs
    row["flops"] = measure_flops(model, x)

    if device != "cuda":
        print(f"    [无 CUDA：跳过显存/延迟]")
        del model
        return row

    # ③ 峰值显存（fp32 单次前向）
    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats()
        _ = model(x)
        torch.cuda.synchronize()
        row["mem"] = torch.cuda.max_memory_allocated() / 2 ** 20  # MiB

    # ④ fp32 延迟
    row["lat32"] = measure_latency(model, x, warmup, iters)

    # ④ fp16 延迟（半精度模型 + 半精度输入）
    try:
        model_h = model.half()
        x_h = x.half()
        row["lat16"] = measure_latency(model_h, x_h, warmup, iters)
    except Exception as exc:
        print(f"    [fp16 失败: {type(exc).__name__}]")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def fmt_flops(flops):
    """FLOPs 格式化（GFLOPs）"""
    if flops is None:
        return "N/A"
    return f"{flops / 1e9:.2f}"


def main():
    """入口：解析参数，逐对象基准并输出汇总表"""
    # 行缓冲：后台运行时也能实时看到进度
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="SegResMamba-Lite 效率基准（论文 Table 6）")
    parser.add_argument("--quick", action="store_true", help="冒烟模式（预热 2 + 计时 10）")
    parser.add_argument("--out", type=str, default=None, help="结果另存 markdown 文件路径")
    args = parser.parse_args()

    warmup, iters = (2, 10) if args.quick else (10, 100)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print("SegResMamba-Lite 效率基准（64³ 输入, batch=1, "
          f"预热 {warmup} + 计时 {iters}）")
    print(f"设备: {device}" + (f" - {torch.cuda.get_device_name(0)}" if device == "cuda" else ""))
    print("=" * 78)

    x = torch.randn(1, 4, 64, 64, 64, device=device)
    rows = []

    # 自研对象
    from models import get_model as get_ours
    for name, ver, kw in OURS:
        print(f"[bench] {name}")
        build = lambda v=ver, k=kw: get_ours(
            version=v, init_filters=20, use_deep_supervision=False, device=device, **k)
        try:
            rows.append(bench_one(name, build, x, warmup, iters, device))
        except Exception as exc:
            print(f"    [失败: {exc}]")

    # 基线对象
    for name, proj, kw in BASELINES:
        print(f"[bench] {name}")
        try:
            mod = load_baseline_models_module(proj)
            if mod is None or not hasattr(mod, "get_model"):
                print(f"    [跳过: 未找到 {proj}/models/__init__.py 或 get_model]")
                continue
            build = lambda m=mod, k=kw: m.get_model(
                use_deep_supervision=False, device=device, **k)
            rows.append(bench_one(name, build, x, warmup, iters, device))
        except Exception as exc:
            print(f"    [跳过: {type(exc).__name__}: {exc}]")

    # 汇总表（markdown）
    lines = []
    lines.append("# 效率基准（实验记录总集.md Part F → Table 6）")
    lines.append("")
    lines.append(f"- 设备: {device}" +
                 (f" - {torch.cuda.get_device_name(0)}" if device == "cuda" else "") +
                 f"；输入 64³, batch=1；延迟 = 预热 {warmup} + 计时 {iters} 均值")
    lines.append(f"- 参数量注意：论文引用一律以训练日志 total_params 为准（本表为当前后端实例化值）")
    lines.append("")
    lines.append("| 模型 | 参数量 | FLOPs (G) | 峰值显存 (MiB) | 延迟 fp32 (ms) | 延迟 fp16 (ms) |")
    lines.append("|------|-------:|----------:|---------------:|---------------:|---------------:|")
    for r in rows:
        flops_str = fmt_flops(r["flops"])
        mem_str = f"{r['mem']:.1f}" if r["mem"] is not None else "-"
        lat32_str = f"{r['lat32']:.2f}" if r["lat32"] is not None else "-"
        lat16_str = f"{r['lat16']:.2f}" if r["lat16"] is not None else "-"
        lines.append(
            f"| {r['name']} | {r['total']:,} ({r['total'] / 1e6:.2f}M) "
            f"| {flops_str} | {mem_str} | {lat32_str} | {lat16_str} |")

    table = "\n".join(lines)
    print("\n" + table + "\n")

    if args.out:
        out_path = args.out if os.path.isabs(args.out) else os.path.join(version_root, args.out)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
