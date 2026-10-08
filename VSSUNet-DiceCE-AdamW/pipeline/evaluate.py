"""
【评估流程】
完整评估流程：数据加载 + 模型 + 框架，支持 Dice/HD95 + Lesion-wise 指标
"""

import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm

_version_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_code_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _version_root)
sys.path.insert(0, _code_root)

from shared.data.dataloader import get_val_dataloader, get_data_list, auto_roi_size_from_cache
from shared.utils.paths import resolve_cache_dir, resolve_data_root
from shared.utils.checkpoint import load_checkpoint
from shared.utils.device import setup_cuda_optimization
from shared.utils.device_info import print_device_info
from shared.utils.logger import setup_logger
from shared.metrics.lesion_metric import lesionwise_evaluation

from models import get_model
from frame.train import VSSUNetTrainer


@torch.no_grad()
def evaluate(
    # ═══════════════════════════════════════════════════════════════════════
    # 命令行参数
    # ═══════════════════════════════════════════════════════════════════════
    model_name="vss_unet",     # 模型名称 (segresnet/vss_unet/vm_unet)
    version_name="vss_unet",   # 版本名称（内部使用）
    resolution=1.0,             # 数据分辨率 (1.0/2.0/3.0/4.0 mm)
    checkpoint_path=None,         # 检查点路径
    runs=1,                      # 评估次数 (>1 时取平均)
    cache_parent="",             # 数据缓存父目录覆盖（优先级：命令行 > 环境变量 SRTP_CACHE_PARENT > 各项目默认）
    data_root="",                # 原始 BraTS TrainingData 目录覆盖（优先级：命令行 > 环境变量 BRATS_DATA_ROOT > 默认）
    # ═══════════════════════════════════════════════════════════════════════
    # 评估参数
    # ═══════════════════════════════════════════════════════════════════════
    num_workers=None,              # 数据加载线程数，None 表示自动计算
    use_disk_cache=False,        # 是否使用数据缓存
    lesion_wise=True,            # 是否启用 lesion-wise 指标
    # ═══════════════════════════════════════════════════════════════════════
    # 设备参数
    # ═══════════════════════════════════════════════════════════════════════
    device=None,                 # 设备类型 (cuda/cpu)
):
    """
    评估入口

    Args:
        model_name: 模型名称
        version_name: 版本名称（内部使用）
        resolution: 数据分辨率
        checkpoint_path: 检查点路径
        runs: 评估次数 (>1 时取平均)
        cache_parent: 数据缓存父目录覆盖（空=使用默认/环境变量）
        data_root: 原始 BraTS TrainingData 目录覆盖（空=使用默认/环境变量）
        num_workers: 数据加载线程数
        use_disk_cache: 是否使用数据缓存
        lesion_wise: 是否启用 lesion-wise 指标
        device: 设备类型 (cuda/cpu)
    """
    # ─────────────────────────────────────────────────────────────────────
    # CUDA 优化
    # ─────────────────────────────────────────────────────────────────────
    setup_cuda_optimization()

    # ─────────────────────────────────────────────────────────────────────
    # 设备配置
    # ─────────────────────────────────────────────────────────────────────
    DEVICE = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    # ─────────────────────────────────────────────────────────────────────
    # 路径配置（可迁移性：路径解析优先级 命令行 > 环境变量 > 历史默认）
    # ─────────────────────────────────────────────────────────────────────
    DATA_DIR = resolve_data_root(
        r"g:\Codes\Python\SRTP\Essay\Data\Brain\TCIA-BraTS\DATA\BraTS2023\BraTS-GLI\TrainingData\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
        data_root,
    )
    CACHE_DIR = r"D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache"

    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger = setup_logger(LOG_DIR, f"eval_{model_name}_{resolution}mm")

    # ─────────────────────────────────────────────────────────────────────
    # 设备信息
    # ─────────────────────────────────────────────────────────────────────
    print_device_info(logger)

    logger.print_config_grouped({
        "模型设置": {
            "model_name": model_name,
            "version_name": version_name,
            "resolution": f"{resolution} mm",
            "device": DEVICE,
            "checkpoint": checkpoint_path if checkpoint_path else "未指定（使用最新模型）",
        },
        "评估设置": {
            "num_workers": num_workers,
            "use_disk_cache": use_disk_cache,
            "lesion_wise": lesion_wise,
            "loss_name": "DiceCELoss",
        },
    }, title="评估配置")

    all_data = get_data_list(DATA_DIR, mode="train")

    current_cache_dir = resolve_cache_dir(resolution, CACHE_DIR, cli_parent=cache_parent) if use_disk_cache else None

    if use_disk_cache and current_cache_dir:
        split_file = os.path.join(current_cache_dir, "split_info.json")
        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                split_info = json.load(f)
            test_case_ids = set(split_info["test_case_ids"])
            print(f"从缓存读取测试集划分 (seed={split_info.get('seed', 'unknown')})")
        else:
            raise FileNotFoundError(f"划分信息文件不存在: {split_file}")
        test_files = [d for d in all_data if d["id"] in test_case_ids]
        test_cache_dir = os.path.join(current_cache_dir, "test")
    else:
        raise ValueError("必须启用缓存 (--cache) 并确保已生成缓存")

    print(f"测试样本数: {len(test_files)}")
    pixdim = (resolution, resolution, resolution)
    spacing = pixdim

    val_loader = get_val_dataloader(
        [],
        batch_size=1,
        num_workers=num_workers,
        pixdim=pixdim,
        cache_dir=test_cache_dir,
    )

    model = get_model(model_name, device=DEVICE)

    if checkpoint_path:
        print(f"加载检查点: {checkpoint_path}")
        model, _ = load_checkpoint(model, checkpoint_path, device=DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    from shared.utils.logger import format_num_params

    logger.print_config_grouped({
        "模型参数": {
            "total_params": format_num_params(total_params),
            "trainable_params": format_num_params(trainable_params),
        },
    }, title="模型信息")

    trainer = VSSUNetTrainer(
        model=model,
        device=DEVICE,
        loss_name="DiceCELoss",
        optimizer_type="AdamW",
        lr=1e-4,
        max_epochs=1,
    )

    roi_size = auto_roi_size_from_cache(test_cache_dir, pixdim)

    results_file = os.path.join(OUTPUT_DIR, f"metrics_{model_name}_{resolution}mm.csv")
    lesion_results_file = os.path.join(OUTPUT_DIR, f"metrics_{model_name}_{resolution}mm_lesion.csv")

    with open(results_file, "w") as f:
        f.write("CaseID,Dice_WT,Dice_TC,Dice_ET,HD95_WT,HD95_TC,HD95_ET\n")

    if lesion_wise:
        with open(lesion_results_file, "w") as f:
            f.write("CaseID,LW_Dice_WT,LW_Dice_TC,LW_Dice_ET,LW_HD95_WT,LW_HD95_TC,LW_HD95_ET\n")

    print("-" * 60)
    logger.print_section("开始评估")

    all_metrics = []
    for run_idx in range(runs):
        if runs > 1:
            print(f"\n运行 {run_idx + 1}/{runs}:")
            print("-" * 40)

        metrics = trainer.validate_verbose(
            val_loader,
            spacing=spacing,
            roi_size=roi_size,
            results_file=results_file if runs == 1 else None,
            lesion_results_file=lesion_results_file if (runs == 1 and lesion_wise) else None,
        )
        all_metrics.append(metrics)

        print(f"全局 Dice - WT: {metrics['dice_wt']:.4f}, TC: {metrics['dice_tc']:.4f}, ET: {metrics['dice_et']:.4f}")
        print(f"全局 HD95 - WT: {metrics['hd95_wt']:.4f}, TC: {metrics['hd95_tc']:.4f}, ET: {metrics['hd95_et']:.4f}")
        if lesion_wise:
            print(f"Lesion-wise Dice - WT: {metrics.get('lw_dice_wt', 0):.4f}, TC: {metrics.get('lw_dice_tc', 0):.4f}, ET: {metrics.get('lw_dice_et', 0):.4f}")
            print(f"Lesion-wise HD95 - WT: {metrics.get('lw_hd95_wt', 0):.4f}, TC: {metrics.get('lw_hd95_tc', 0):.4f}, ET: {metrics.get('lw_hd95_et', 0):.4f}")
        print("-" * 40)

    if runs > 1:
        import numpy as np
        dice_wt_mean = np.mean([m['dice_wt'] for m in all_metrics])
        dice_wt_std = np.std([m['dice_wt'] for m in all_metrics])
        dice_tc_mean = np.mean([m['dice_tc'] for m in all_metrics])
        dice_tc_std = np.std([m['dice_tc'] for m in all_metrics])
        dice_et_mean = np.mean([m['dice_et'] for m in all_metrics])
        dice_et_std = np.std([m['dice_et'] for m in all_metrics])

        hd95_wt_mean = np.mean([m['hd95_wt'] for m in all_metrics])
        hd95_wt_std = np.std([m['hd95_wt'] for m in all_metrics])
        hd95_tc_mean = np.mean([m['hd95_tc'] for m in all_metrics])
        hd95_tc_std = np.std([m['hd95_tc'] for m in all_metrics])
        hd95_et_mean = np.mean([m['hd95_et'] for m in all_metrics])
        hd95_et_std = np.std([m['hd95_et'] for m in all_metrics])

        print("\n" + "=" * 50)
        logger.print_section("多次评估结果")
        print(f"Dice (WT): {dice_wt_mean:.4f} ± {dice_wt_std:.4f}")
        print(f"Dice (TC): {dice_tc_mean:.4f} ± {dice_tc_std:.4f}")
        print(f"Dice (ET): {dice_et_mean:.4f} ± {dice_et_std:.4f}")
        print(f"HD95 (WT): {hd95_wt_mean:.4f} ± {hd95_wt_std:.4f}")
        print(f"HD95 (TC): {hd95_tc_mean:.4f} ± {hd95_tc_std:.4f}")
        print(f"HD95 (ET): {hd95_et_mean:.4f} ± {hd95_et_std:.4f}")

        float_wt = (dice_wt_std / dice_wt_mean * 100) if dice_wt_mean > 0 else 0
        float_tc = (dice_tc_std / dice_tc_mean * 100) if dice_tc_mean > 0 else 0
        float_et = (dice_et_std / dice_et_mean * 100) if dice_et_mean > 0 else 0
        print(f"\n浮动范围:")
        print(f"  WT Dice: {float_wt:.2f}%")
        print(f"  TC Dice: {float_tc:.2f}%")
        print(f"  ET Dice: {float_et:.2f}%")
        print("=" * 50)

        metrics = all_metrics[0]

    print(f"详细结果已保存至: {results_file}")
    if lesion_wise:
        print(f"Lesion-wise 结果已保存至: {lesion_results_file}")

    logger.close()
    sys.stdout = sys.__stdout__

    return metrics


if __name__ == "__main__":
    import argparse
    _version_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _default_ckpt = os.path.join(_version_root, "pipeline", "models", "2.0mm_cuda", "best_metric_model.pth")
    parser = argparse.ArgumentParser(description="VSSUNet 评估")
    parser.add_argument("--model", type=str, default="vss_unet", help="模型名称")
    parser.add_argument("--resolution", type=float, default=1.0, help="分辨率")
    parser.add_argument("--checkpoint", type=str, default=_default_ckpt, help="检查点路径")
    parser.add_argument("--workers", type=str, default="auto", help="数据加载线程数 (auto/数字，auto表示自动计算)")
    parser.add_argument("--cache", action="store_true", help="启用硬盘缓存")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    parser.add_argument("--no_lesion_wise", action="store_true", help="禁用 lesion-wise 指标")
    parser.add_argument("--runs", type=int, default=1, help="评估次数 (>1 时取平均)")
    # 跨卡包合并的可选开关（默认关闭，行为与历史完全一致）
    parser.add_argument("--cache_parent", type=str, default="",
                        help="数据缓存父目录覆盖（优先级：命令行 > 环境变量 SRTP_CACHE_PARENT > 各项目默认；最终目录 = {parent}_{分辨率}）")
    parser.add_argument("--data_root", type=str, default="",
                        help="原始 BraTS TrainingData 目录覆盖（优先级：命令行 > 环境变量 BRATS_DATA_ROOT > 默认）")
    args = parser.parse_args()
    
    # 解析 num_workers 参数
    if args.workers.lower() == "auto":
        workers_arg = None
    else:
        try:
            workers_arg = int(args.workers)
        except ValueError:
            print(f"警告: 无效的 --workers 值 '{args.workers}'，使用 auto")
            workers_arg = None

    evaluate(
        model_name=args.model,
        version_name="vss_unet",
        resolution=args.resolution,
        checkpoint_path=args.checkpoint,
        num_workers=workers_arg,
        use_disk_cache=args.cache,
        lesion_wise=not args.no_lesion_wise,
        runs=args.runs,
        cache_parent=args.cache_parent,
        data_root=args.data_root,
        device=args.device,
    )
