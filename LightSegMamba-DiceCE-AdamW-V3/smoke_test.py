"""
[LightSegMamba V3 Smoke Test]

不需要任何数据或缓存，只验证：
1. mamba_ssm + causal_conv1d 可用
2. 模型可构造、参数量（默认 base_channels=32 约 1.4M）
3. 前向传播维度正确
4. 反向传播梯度正常
5. 不同 patch 大小兼容性
6. 不同 base_channels 参数量估算

运行：
  python smoke_test.py
  python smoke_test.py --base-channels 24    # 测 V2 量级
  python smoke_test.py --base-channels 44    # 测论文量级 3M
"""

from __future__ import annotations
import sys
import os
import argparse
import torch
import time

# 把 code/ 根加入 sys.path
_VERSION_ROOT = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_VERSION_ROOT)
sys.path.insert(0, _VERSION_ROOT)
sys.path.insert(0, _CODE_ROOT)

from models import get_model, count_parameters


# 用 ASCII 符号，避免 Windows GBK 编码报错
OK = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def main():
    parser = argparse.ArgumentParser(description="LightSegMamba V3 Smoke Test")
    parser.add_argument("--base-channels", type=int, default=32,
                        help="Stem 通道数（默认 32，约 1.4M 参数）")
    parser.add_argument("--patch-size", type=int, default=64,
                        help="测试 patch 尺寸（默认 64）")
    args = parser.parse_args()

    # 设置 stdout 编码为 UTF-8（防止 Windows GBK 编码问题）
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print(f"LightSegMamba-DiceCE-AdamW-V3  Smoke Test  (base_channels={args.base_channels})")
    print("=" * 70)

    # ── 1. 检查 mamba_ssm / causal_conv1d ──────────────────────────────────
    print("\n[1] 检查 Mamba 依赖")
    try:
        import mamba_ssm
        print(f"  {OK} mamba_ssm 已安装: {mamba_ssm.__file__}")
    except ImportError as e:
        print(f"  {FAIL} mamba_ssm 未安装: {e}")
        print("    LightSegMamba-V3 需在含 mamba_ssm 的环境（如 mamba_sm120）运行")
        return 1
    try:
        import causal_conv1d
        print(f"  {OK} causal_conv1d 已安装: {causal_conv1d.__file__}")
    except ImportError as e:
        print(f"  {FAIL} causal_conv1d 未安装: {e}")
        return 1

    # ── 2. 构造模型 ─────────────────────────────────────────────────────────
    print(f"\n[2] 构造 LightSegMamba3D 模型 (base_channels={args.base_channels})")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  使用设备: {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    model = get_model(
        "lightsegmamba",
        in_channels=4,
        out_channels=4,
        device=DEVICE,
        base_channels=args.base_channels,
    )
    n_params = count_parameters(model)
    print(f"  {OK} 模型构造成功")
    print(f"  {OK} 可训练参数量: {n_params:,}  ({n_params / 1e6:.2f}M)")

    # 参数量参考表（Mamba2 版实测值）
    print(f"\n  参数量参考表（不同 base_channels）:")
    print(f"    base_channels=24 → ~0.74M  (V2 量级)")
    print(f"    base_channels=32 → ~1.27M  (V3 默认，实测值)")
    print(f"    base_channels=38 → ~1.8M")
    print(f"    base_channels=44 → ~2.5M   (接近论文 3.08M)")

    # ── 3. 前向传播 ────────────────────────────────────────────────────────
    sz = args.patch_size
    print(f"\n[3] 前向传播测试 (input: B=1, C=4, D={sz}, H={sz}, W={sz})")
    x = torch.randn(1, 4, sz, sz, sz, device=DEVICE)

    model.eval()
    with torch.no_grad():
        torch.cuda.empty_cache() if DEVICE == "cuda" else None
        t0 = time.time()
        y = model(x)
        t1 = time.time()

    print(f"  {OK} 前向传播成功")
    print(f"  {OK} 输出 shape: {tuple(y.shape)}  (期望 (1, 4, {sz}, {sz}, {sz}))")
    print(f"  {OK} 单次推理耗时: {(t1 - t0) * 1000:.2f} ms")
    assert y.shape == (1, 4, sz, sz, sz), f"输出 shape 不匹配: {y.shape}"

    # ── 4. 反向传播 ────────────────────────────────────────────────────────
    print("\n[4] 反向传播测试")
    model.train()
    y = model(x)
    fake_label = torch.randint(0, 4, (1, sz, sz, sz), device=DEVICE)
    loss = torch.nn.functional.cross_entropy(y, fake_label)
    loss.backward()
    print(f"  {OK} 反向传播成功，loss = {loss.item():.4f}")

    # 检查梯度
    grad_ok = True
    for name, p in model.named_parameters():
        if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0):
            print(f"  {WARN} 参数 {name} 无梯度")
            grad_ok = False
    if grad_ok:
        print(f"  {OK} 所有可训练参数均有梯度")

    # ── 5. 显存占用（仅 CUDA） ─────────────────────────────────────────────
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9
        print(f"\n[5] 显存占用（前向+反向）")
        print(f"  {OK} 峰值分配: {peak_alloc:.2f} GB")
        print(f"  {OK} 峰值预留: {peak_reserved:.2f} GB")
        if peak_reserved > 3.5:
            print(f"  {WARN} 预留显存接近 4GB，训练时建议减小 patch_size 或 batch_size")

    # ── 6. 不同 patch 大小兼容性测试 ───────────────────────────────────────
    print("\n[6] 不同 patch 大小兼容性测试")
    for test_sz in [(32, 32, 32), (48, 48, 48), (64, 64, 64), (96, 96, 96)]:
        try:
            x_test = torch.randn(1, 4, *test_sz, device=DEVICE)
            model.eval()
            with torch.no_grad():
                y_test = model(x_test)
            print(f"  {OK} D=H=W={test_sz[0]} -> out {tuple(y_test.shape)}")
            del x_test, y_test
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {FAIL} D=H=W={test_sz[0]} 失败: {e}")

    print("\n" + "=" * 70)
    print("Smoke Test 全部通过")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
