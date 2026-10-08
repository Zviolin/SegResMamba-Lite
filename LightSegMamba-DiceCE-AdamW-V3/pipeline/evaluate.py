"""
【LightSegMamba 评估流程 V3】

CLI（对齐 MambaUNet evaluate 风格）：

  # 评估 200 例训练的模型
  python pipeline/evaluate.py \\
    --model lightsegmamba \\
    --resolution 2.0 \\
    --cache \\
    --max-samples 200 \\
    --base-channels 32 \\
    --checkpoint pipeline/models/2.0mm_200_cuda/best_metric_model.pth \\
    --device cuda

  # 评估全量训练的模型
  python pipeline/evaluate.py \\
    --model lightsegmamba \\
    --resolution 2.0 \\
    --cache \\
    --base-channels 32 \\
    --checkpoint pipeline/models/2.0mm_full_cuda/best_metric_model.pth \\
    --device cuda
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

_VERSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE_ROOT = os.path.dirname(_VERSION_ROOT)
# 默认缓存父目录（2026-10-07 定案：全库统一默认主项目体系缓存，与其他 6 项目一致；换机器用 SRTP_CACHE_PARENT 整机重定向）
_DEFAULT_CACHE_PARENT = r"D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache"
sys.path.insert(0, _VERSION_ROOT)
sys.path.insert(0, _CODE_ROOT)

from shared.data.dataloader import get_val_dataloader, auto_roi_size_from_cache
from shared.utils.checkpoint import load_checkpoint
from shared.utils.device import setup_cuda_optimization
from shared.utils.device_info import print_device_info
from shared.utils.logger import setup_logger
from shared.utils.paths import resolve_cache_dir
from shared.metrics import lesionwise_evaluation
from shared.inference.sliding_window import sliding_window_inference
from shared.inference.postprocess import brats_label_mapping
from monai.metrics import DiceMetric as MonaiDiceMetric, HausdorffDistanceMetric as MonaiHD95Metric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

from models import get_model
from frame.train import LightSegMambaTrainer


@torch.no_grad()
def evaluate(
    model_name: str = "lightsegmamba",
    version_name: str = "lightsegmamba_v3",
    resolution: float = 2.0,
    use_disk_cache: bool = False,
    max_samples: int | None = None,
    checkpoint_path: str = None,
    base_channels: int = 32,
    num_workers: int = 0,
    patch_size: tuple = (64, 64, 64),
    lesion_wise: bool = True,
    device: str = None,
    # 可迁移性：路径覆盖（默认空=历史行为不变）
    cache_parent: str = "",   # 数据缓存父目录覆盖（优先级：命令行 > 环境变量 SRTP_CACHE_PARENT > 各项目默认）
    data_root: str = "",      # 原始 BraTS TrainingData 目录覆盖（本项目评估仅读缓存，保留参数以对齐跨项目 CLI 口径）
):
    """评估入口。

    Args:
        model_name: 模型名称（仅 lightsegmamba）
        version_name: 版本名（用于日志与结果文件命名）
        resolution: 重采样分辨率
        use_disk_cache: 是否使用缓存（必须 True）
        max_samples: 限样本数（None=全量；200=快速验证集）
        checkpoint_path: 检查点路径
        base_channels: 重建模型时的 Stem 通道数（必须与训练时一致）
        num_workers: DataLoader worker 数
        patch_size: 推理 patch 尺寸（与滑窗 roi_size 一致）
        lesion_wise: 是否启用 lesion-wise 指标
        device: cuda / cpu
        cache_parent: 数据缓存父目录覆盖（空=默认 / 环境变量 SRTP_CACHE_PARENT）
        data_root: 原始数据目录覆盖（本项目评估仅读缓存，保留以对齐跨项目 CLI 口径）
    """
    if not use_disk_cache:
        raise ValueError("LightSegMamba-V3 评估必须启用缓存（--cache）")

    setup_cuda_optimization()
    DEVICE = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger = setup_logger(LOG_DIR, f"eval_{version_name}_{resolution}mm")
    print_device_info(logger)

    samples_tag = "full" if (max_samples is None or max_samples >= 1251) else f"{max_samples}"

    logger.print_config_grouped({
        "模型设置": {
            "model_name": model_name,
            "version_name": version_name,
            "resolution": f"{resolution} mm",
            "base_channels": base_channels,
            "device": DEVICE,
            "checkpoint": checkpoint_path if checkpoint_path else "未指定",
        },
        "评估设置": {
            "max_samples": "全量 1251" if max_samples is None else f"{max_samples} 例",
            "patch_size": patch_size,
            "num_workers": num_workers,
            "lesion_wise": lesion_wise,
            "loss_name": "DiceCELoss",
        },
    }, title="评估配置")

    # ── 缓存与划分 ────────────────────────────────────────────────────────
    # max_samples 必须以关键字传参（新版 shared 签名第 3 位是 cli_parent）：
    # cli_parent 覆盖父目录，max_samples 追加 _{N} 采样后缀，两者正交工作；
    # 两者皆空/None 时与历史目录完全一致。
    # 注意：本项目评估仅读 PersistentDataset 缓存、不直接访问原始数据，
    # 故无 DATA_DIR 可参数化（data_root 仅为跨项目 CLI 口径对齐保留）。
    current_cache_dir = resolve_cache_dir(resolution, _DEFAULT_CACHE_PARENT, max_samples=max_samples, cli_parent=cache_parent)

    split_file = os.path.join(current_cache_dir, "split_info.json")
    if not os.path.exists(split_file):
        raise FileNotFoundError(
            f"划分信息文件不存在: {split_file}\n"
            f"请先运行：python pipeline/gen_cache.py --resolution {resolution} "
            f"--max-samples {max_samples if max_samples else '1251'} --generate --cache"
        )

    with open(split_file, "r") as f:
        split_info = json.load(f)
    print(f"从缓存读取测试集划分 (seed={split_info.get('seed', 'unknown')})")
    print(f"  test: {len(split_info['test_case_ids'])} 例")

    test_cache_dir = os.path.join(current_cache_dir, "test")
    pixdim = (resolution, resolution, resolution)
    spacing = pixdim

    val_loader = get_val_dataloader(
        [],
        batch_size=1,
        num_workers=num_workers,
        pixdim=pixdim,
        patch_size=patch_size,
        cache_dir=test_cache_dir,
    )

    # ── 模型与检查点 ──────────────────────────────────────────────────────
    model = get_model(
        model_name,
        in_channels=4,
        out_channels=4,
        device=DEVICE,
        base_channels=base_channels,
    )
    if checkpoint_path:
        print(f"加载检查点: {checkpoint_path}")
        model, _ = load_checkpoint(model, checkpoint_path, device=DEVICE)
    else:
        print("⚠️  未指定 checkpoint，使用随机初始化的模型（仅用于流程测试）")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    from shared.utils.logger import format_num_params
    logger.print_config_grouped({
        "模型参数": {
            "total_params": format_num_params(total_params),
            "trainable_params": format_num_params(trainable_params),
        },
    }, title="模型信息")

    # ── 训练器（仅借用 validate_verbose） ─────────────────────────────────
    trainer = LightSegMambaTrainer(
        model=model, device=DEVICE, loss_name="DiceCELoss",
        optimizer_type="AdamW", lr=1e-4, max_epochs=1,
    )

    roi_size = auto_roi_size_from_cache(test_cache_dir, pixdim)

    # ── 结果文件 ──────────────────────────────────────────────────────────
    results_file = os.path.join(OUTPUT_DIR, f"metrics_{version_name}_{resolution}mm_{samples_tag}.csv")
    lesion_results_file = os.path.join(OUTPUT_DIR, f"metrics_{version_name}_{resolution}mm_{samples_tag}_lesion.csv")

    with open(results_file, "w") as f:
        f.write("CaseID,Dice_WT,Dice_TC,Dice_ET,HD95_WT,HD95_TC,HD95_ET\n")
    if lesion_wise:
        with open(lesion_results_file, "w") as f:
            f.write("CaseID,LW_Dice_WT,LW_Dice_TC,LW_Dice_ET,LW_HD95_WT,LW_HD95_TC,LW_HD95_ET,"
                    "LDW_GT,LDW_FP,LDT_GT,LDT_FP,LDE_GT,LDE_FP\n")

    print("-" * 60)
    logger.print_section("开始评估")

    metrics = trainer.validate_verbose(
        val_loader,
        spacing=spacing,
        roi_size=roi_size,
        results_file=results_file,
        lesion_results_file=lesion_results_file if lesion_wise else None,
    )

    print("\n" + "=" * 50)
    print(f"全局 Dice - WT: {metrics['dice_wt']:.4f}, TC: {metrics['dice_tc']:.4f}, ET: {metrics['dice_et']:.4f}")
    print(f"全局 HD95 - WT: {metrics['hd95_wt']:.4f}, TC: {metrics['hd95_tc']:.4f}, ET: {metrics['hd95_et']:.4f}")
    if lesion_wise:
        print(f"Lesion-wise Dice - WT: {metrics.get('lw_dice_wt', 0):.4f}, "
              f"TC: {metrics.get('lw_dice_tc', 0):.4f}, ET: {metrics.get('lw_dice_et', 0):.4f}")
        print(f"Lesion-wise HD95 - WT: {metrics.get('lw_hd95_wt', 0):.4f}, "
              f"TC: {metrics.get('lw_hd95_tc', 0):.4f}, ET: {metrics.get('lw_hd95_et', 0):.4f}")
    print("=" * 50)
    print(f"详细结果已保存至: {results_file}")
    if lesion_wise:
        print(f"Lesion-wise 结果已保存至: {lesion_results_file}")

    logger.close()
    sys.stdout = sys.__stdout__
    return metrics


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    _PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
    _default_ckpt = os.path.join(
        _PIPELINE_DIR, "models", "2.0mm_full_cuda", "best_metric_model.pth",
    )

    parser = argparse.ArgumentParser(description="LightSegMamba-V3 评估")
    parser.add_argument("--model", type=str, default="lightsegmamba", help="模型名称")
    parser.add_argument("--resolution", type=float, default=2.0, help="分辨率")
    parser.add_argument("--cache", action="store_true", help="启用硬盘缓存（必须）")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本数（None=全量；200=快速验证集评估）")
    parser.add_argument("--checkpoint", type=str, default=_default_ckpt, help="检查点路径")
    parser.add_argument("--base-channels", type=int, default=32,
                        help="Stem 通道数（必须与训练时一致）")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker 数")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[64, 64, 64],
                        help="推理 patch 尺寸（D H W）")
    parser.add_argument("--no-lesion-wise", action="store_true", help="禁用 lesion-wise")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    # 跨卡包合并的可选开关（默认关闭，行为与历史完全一致）
    parser.add_argument("--cache_parent", type=str, default="",
                        help="数据缓存父目录覆盖（优先级：命令行 > 环境变量 SRTP_CACHE_PARENT > 各项目默认；最终目录 = {parent}_{分辨率}）")
    parser.add_argument("--data_root", type=str, default="",
                        help="原始 BraTS TrainingData 目录覆盖（优先级：命令行 > 环境变量 BRATS_DATA_ROOT > 默认）")
    args = parser.parse_args()

    evaluate(
        model_name=args.model,
        resolution=args.resolution,
        use_disk_cache=args.cache,
        max_samples=args.max_samples,
        checkpoint_path=args.checkpoint,
        base_channels=args.base_channels,
        num_workers=args.workers,
        patch_size=tuple(args.patch_size),
        lesion_wise=not args.no_lesion_wise,
        device=args.device,
        cache_parent=args.cache_parent,
        data_root=args.data_root,
    )
