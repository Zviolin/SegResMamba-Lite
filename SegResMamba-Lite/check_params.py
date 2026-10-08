"""
SegResMamba-Lite 参数量自检脚本（v1-v12 全部注册版本）

对照基准: pipeline/logs/ 训练日志 total_params（Mamba2SSM CUDA 后端的实验值）

用法:
    python check_params.py            # 检查全部版本（附汇总表）
    python check_params.py v10        # 单版本，附各模块分层明细
    python check_params.py v10 v11    # 多个版本

注意:
- 本脚本一律使用「实验配置」：init_filters V1=24、其余=20（与训练日志一致，
  并非各文件签名默认值 26/22/20）
- 本机若回退 MiniMamba 纯 PyTorch 后端，实测参数量会比日志高 0.3%-1%，
  脚本会标注「预期偏差」；论文引用一律以训练日志为准
- 改动模型代码 / 新增消融开关后，先跑本脚本对照参数量，再开训
  （见 实验记录总集.md Part F·自检要求）
- v5 为未注册的废弃草稿，不在检查范围
"""
import os
import sys

# 路径配置：版本根目录（models 包）+ Code 目录（shared.models.mamba 需要）
version_root = os.path.dirname(os.path.abspath(__file__))
code_root = os.path.dirname(version_root)
for _p in (version_root, code_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from models import get_model


def detect_backend():
    """探测当前 Mamba 后端，提示与训练日志环境的可比性"""
    try:
        import shared.models.mamba as mamba_mod
        if getattr(mamba_mod, "MAMBA2_SSM_AVAILABLE", False):
            return "Mamba2SSM (CUDA) —— 与训练日志同后端"
        if getattr(mamba_mod, "MINIMAMBA_AVAILABLE", False):
            return "MiniMamba (纯 PyTorch 回退) —— 参数量将高于日志 0.3%-1%"
        return "未知回退后端"
    except Exception as exc:
        # 后端模块导入失败时给出提示，不阻断检查
        return f"探测失败: {exc}"


# 预期值: (版本, 实验 init_filters, 额外表参, 日志 total, 日志 trainable, 说明)
# trainable 说明: v1-v10 无冻结参数（日志未单列，按=total 计）；
#                 v11 冻结边界核 80×27=2,160（源码 v11.py 固定 26 邻域中心差分核）
#                 v12 已回填（训练日志 20260924_094608：total 1,428,194 / trainable 1,426,034）；
#                     = V10 锚点 + 5,460（冻结边界核 2,160 + 可学习门控 3,300）
EXPECTED = [
    ("v1",  24, {},                              1_523_892, 1_523_892, "第一代: 2 尺度 BiMamba"),
    ("v2",  20, {},                              1_415_998, 1_415_998, "第一代: 3 尺度 BiMamba"),
    ("v3",  20, {},                              1_508_774, 1_508_774, "第一代: 双重 Bottleneck + 全量 SE + Dec Mamba"),
    ("v4",  20, {},                              1_560_880, 1_560_880, "第一代: 三重 Bottleneck（超限失败）"),
    ("v6",  20, {"num_experts": 4},              1_463_530, 1_463_530, "第二代: Soft MoA，4 Mamba 专家软路由（无 α）"),
    ("v7",  20, {},                              1_498_411, 1_498_411, "第二代: +4 DConv 专家，sigmoid α"),
    ("v8",  20, {},                              1_498_411, 1_498_411, "第二代: 温度缩放 α"),
    ("v9",  20, {},                              1_418_234, 1_418_234, "第三代: Top-2 稀疏 DConv MoE"),
    ("v10", 20, {"num_experts": 4, "top_k": 2},  1_422_734, 1_422_734, "第三代: 论文级 MoA（最终模型）"),
    ("v11", 20, {},                              1_439_154, 1_436_994, "第三代: 边界感知混合 MoA（冻结核 2,160）"),
    ("v12", 20, {"num_experts": 4, "top_k": 2},  1_428_194, 1_426_034, "P2 探索: V10 + 边界门控（探索性负结果，未采用）"),
]


def check_one(version, init_filters, extra, exp_total, exp_train, note, verbose=False):
    """实例化单个版本并与日志预期对照

    Args:
        version: 版本名（v1-v11）
        init_filters: 实验 init_filters
        extra: 额外表参（如 num_experts/top_k）
        exp_total: 日志 total_params 预期值
        exp_train: 日志 trainable_params 预期值
        note: 版本说明
        verbose: 是否打印各模块分层明细

    Returns:
        dict: 含实测/预期/偏差的结果
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(version=version, init_filters=init_filters,
                      use_deep_supervision=False, device=device, **extra)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'=' * 70}")
    print(f"{version.upper()}  |  {note}")
    extra_str = f"  extra={extra}" if extra else ""
    print(f"配置: init_filters={init_filters}{extra_str}  use_deep_supervision=False  device={device}")
    print(f"实测: total={total:,}  trainable={trainable:,}  ({total / 1e6:.2f}M, <1.5M: {'是' if total < 1.5e6 else '否'})")

    # v12 等新注册版本尚无训练日志，仅确认可实例化
    if exp_total is None:
        print(f"日志: 待训练回填（新注册版本，无训练日志）")
        print(f"对照: ➖ 新版本无日志预期，仅确认可实例化")
        if verbose:
            print("-" * 70)
            print("各模块参数量:")
            for name, module in model.named_children():
                n = sum(p.numel() for p in module.parameters())
                if n > 0:
                    print(f"  {name:<26}{n:>12,}  ({n / 1e6:.3f}M)")
        del model
        return {"version": version, "total": total, "trainable": trainable,
                "exp_total": None, "diff_pct": float('nan')}

    diff_pct = (total - exp_total) / exp_total * 100.0
    print(f"日志: total={exp_total:,}  trainable={exp_train:,}")
    if diff_pct == 0:
        mark, reason = "✅", "与日志完全一致"
    elif abs(diff_pct) <= 2.0:
        mark = "⚠️ "
        reason = f"偏差 {diff_pct:+.2f}%（MiniMamba 后端所致，属预期；论文引用以日志为准）"
    else:
        mark, reason = "❌", f"偏差 {diff_pct:+.2f}%（>2%，请检查配置或近期代码改动）"
    print(f"对照: {mark} {reason}")

    if verbose:
        print("-" * 70)
        print("各模块参数量:")
        for name, module in model.named_children():
            n = sum(p.numel() for p in module.parameters())
            if n > 0:
                print(f"  {name:<26}{n:>12,}  ({n / 1e6:.3f}M)")

    del model
    return {"version": version, "total": total, "trainable": trainable,
            "exp_total": exp_total, "diff_pct": diff_pct}


def main():
    """入口：解析命令行版本参数并逐一检查"""
    targets = sys.argv[1:]
    lookup = {row[0]: row for row in EXPECTED}

    if targets:
        bad = [t for t in targets if t not in lookup]
        if bad:
            print(f"未知版本: {bad}；可用: {list(lookup)}")
            sys.exit(1)
        selected = [lookup[t] for t in targets]
    else:
        selected = list(EXPECTED)

    print("=" * 70)
    print("SegResMamba-Lite 参数量自检（对照 pipeline/logs 训练日志）")
    print(f"Mamba 后端: {detect_backend()}")
    print(f"检查版本: {', '.join(row[0] for row in selected)}")
    print("=" * 70)

    results = []
    for version, init_filters, extra, exp_total, exp_train, note in selected:
        try:
            results.append(check_one(version, init_filters, extra,
                                     exp_total, exp_train, note,
                                     verbose=bool(targets)))
        except Exception as exc:
            # 单版本失败不阻断其余版本
            print(f"\n❌ {version.upper()} 实例化失败: {exc}")

    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print("汇总（对照训练日志）")
        print("-" * 70)
        print(f"{'版本':<6}{'实测 total':>14}{'日志 total':>14}{'偏差':>10}   <1.5M")
        for r in results:
            diff_str = f"{r['diff_pct']:>+9.2f}%" if r['exp_total'] is not None else f"{'待回填':>10}"
            print(f"{r['version']:<6}{r['total']:>14,}"
                  f"{r['exp_total'] if r['exp_total'] is not None else 'None':>14}"
                  f"{diff_str}   {'是' if r['total'] < 1.5e6 else '否'}")


if __name__ == "__main__":
    main()
