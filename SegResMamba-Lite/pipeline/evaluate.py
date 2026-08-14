"""
【评估流程】
完整评估流程：数据加载 + 模型 + 框架，支持 Dice/HD95 + Lesion-wise 指标

支持版本：v1, v2, v3, v4, v6
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
from shared.utils.checkpoint import load_checkpoint
from shared.utils.device import setup_cuda_optimization
from shared.utils.device_info import print_device_info
from shared.utils.logger import setup_logger
from shared.metrics.lesion_metric import lesionwise_evaluation

from models import get_model
from frame.train import OptimizedTrainer


@torch.no_grad()
def evaluate(
    # ════════════════════════════════════════════════════════════════════════
    # 命令行参数
    # ════════════════════════════════════════════════════════════════════════
    model_name="segresmamba_lite",    # 模型名称
    version="v2",                     # 模型版本 (v1/v2/v3/v4/v6)
    resolution=2.0,                   # 数据分辨率
    checkpoint_path=None,
    runs=1,
    # ════════════════════════════════════════════════════════════════════════
    # 模型参数
    # ════════════════════════════════════════════════════════════════════════
    init_filters=None,                # 初始滤波器数量
    use_attention=True,               # 是否使用注意力机制
    d_state=8,
    d_conv=2,
    expand=2,
    num_experts=4,                    # MoA 专家数量 (V6/V7 专用)
    use_deep_supervision=True,
    # V6 专属（已废弃，保留兼容）
    use_boundary=True,
    use_mamba_in_moa=True,
    moa_reduction=4,
    moa_dropout=0.1,
    # ═══════════════════════════════════════════════════════════════════════
    # 评估参数
    # ═══════════════════════════════════════════════════════════════════════
    num_workers=None,
    use_disk_cache=False,
    lesion_wise=True,
    visualize_attention=False,        # V6: 可视化 MoA 注意力权重
    # ═══════════════════════════════════════════════════════════════════════
    # 设备参数
    # ═══════════════════════════════════════════════════════════════════════
    device=None,
):
    """评估入口"""
    setup_cuda_optimization()

    DEVICE = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR = r"g:\Codes\Python\SRTP\Essay\Data\Brain\TCIA-BraTS\DATA\BraTS2023\BraTS-GLI\TrainingData\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    CACHE_DIR = r"D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache"

    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    is_ema_model = False
    if checkpoint_path:
        is_ema_model = "ema" in os.path.basename(checkpoint_path).lower()

    model_suffix = "_ema" if is_ema_model else ""
    logger = setup_logger(LOG_DIR, f"eval_{model_name}_{resolution}mm{model_suffix}")

    print_device_info(logger)

    if init_filters is None:
        defaults = {'v1': 26, 'v2': 22, 'v3': 18, 'v4': 18, 'v6': 20, 'v7': 18, 'v8': 20}
        init_filters = defaults.get(version, 20)

    logger.print_config_grouped({
        "模型设置": {
            "model_name": model_name,
            "version": version,
            "resolution": f"{resolution} mm",
            "init_filters": init_filters,
            "d_state": d_state,
            "expand": expand,
            "use_attention": use_attention,
            "device": DEVICE,
            "checkpoint": checkpoint_path if checkpoint_path else "未指定",
        },
        "评估设置": {
            "num_workers": num_workers,
            "use_disk_cache": use_disk_cache,
            "lesion_wise": lesion_wise,
            "loss_name": "DiceFocalLoss",
        },
    }, title="评估配置")

    all_data = get_data_list(DATA_DIR, mode="train")

    case_ids = []
    for d in all_data:
        image_path = d["image"]
        if isinstance(image_path, list):
            image_path = image_path[0]
        filename = os.path.basename(image_path)
        parts = filename.split("-")
        if len(parts) >= 4:
            case_id = "-".join(parts[:4])
        else:
            case_id = filename
        case_ids.append(case_id)

    unique_case_ids = list(set(case_ids))
    unique_case_ids.sort()

    current_cache_dir = f"{CACHE_DIR}_{resolution}" if use_disk_cache else None

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
        test_case_ids = unique_case_ids[int(len(unique_case_ids) * 0.85):]
        test_files = [d for d, cid in zip(all_data, case_ids) if cid in test_case_ids]
        test_cache_dir = current_cache_dir

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

    # ─────────────────────────────────────────────────────────────────────
    # 模型创建（按 version 分发）
    # ─────────────────────────────────────────────────────────────────────
    model = get_model(
        version=version,
        in_channels=4,
        out_channels=4,
        init_filters=init_filters,
        use_attention=use_attention,
        use_deep_supervision=use_deep_supervision,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        num_experts=num_experts,
        device=DEVICE,
    )

    if checkpoint_path:
        print(f"加载检查点: {checkpoint_path}")
        model, _ = load_checkpoint(model, checkpoint_path, device=DEVICE, strict=False)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    from shared.utils.logger import format_num_params

    logger.print_config_grouped({
        "模型参数": {
            "total_params": format_num_params(total_params),
            "trainable_params": format_num_params(trainable_params),
        },
    }, title="模型信息")

    # ─────────────────────────────────────────────────────────────────────
    # V6: MoA 注意力权重可视化
    # ─────────────────────────────────────────────────────────────────────
    if visualize_attention and version == 'v6' and hasattr(model, 'get_moa_attention_weights'):
        logger.print_info("=" * 70)
        logger.print_info("MoA 注意力权重统计")
        logger.print_info("=" * 70)
        model.eval()
        x = torch.randn(1, 4, 64, 64, 64, device=DEVICE)
        with torch.no_grad():
            weights = model.get_moa_attention_weights(x)
        weights_np = weights.cpu().numpy()[0]
        att_names = ["Channel", "Spatial"]
        if use_mamba_in_moa:
            att_names.append("Mamba")
        if use_boundary:
            att_names.append("Boundary")
        for name, w in zip(att_names, weights_np):
            logger.print_info(f"  {name:10s} 注意力权重: {w:.4f} ({w*100:.1f}%)")

    trainer = OptimizedTrainer(
        model=model,
        device=DEVICE,
        loss_name="DiceFocalLoss",
        optimizer_type="AdamW",
        lr=1e-4,
        weight_decay=1e-5,
        scheduler_type="CosineAnnealingLR",
        max_epochs=1,
        use_ema=False,
        use_deep_supervision=False,
    )

    roi_size = auto_roi_size_from_cache(test_cache_dir, pixdim)

    results_file = os.path.join(OUTPUT_DIR, f"metrics_{model_name}_{resolution}mm{model_suffix}.csv")
    lesion_results_file = os.path.join(OUTPUT_DIR, f"metrics_{model_name}_{resolution}mm{model_suffix}_lesion.csv")

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
            lesion_results_file=lesion_results_file if (runs == 1 and lesion_wise) else (lesion_results_file if lesion_wise else None),
        )
        all_metrics.append(metrics)

        print(f"全局 Dice - WT: {metrics['dice_wt']:.4f}, TC: {metrics['dice_tc']:.4f}, ET: {metrics['dice_et']:.4f}")
        print(f"全局 HD95 - WT: {metrics['hd95_wt']:.4f}, TC: {metrics['hd95_tc']:.4f}, ET: {metrics['hd95_et']:.4f}")
        if lesion_wise:
            print(f"Lesion-wise Dice - WT: {metrics.get('lw_dice_wt', 0):.4f}, TC: {metrics.get('lw_dice_tc', 0):.4f}, ET: {metrics.get('lw_dice_et', 0):.4f}")
            print(f"Lesion-wise HD95 - WT: {metrics.get('lw_hd95_wt', 0):.4f}, TC: {metrics.get('lw_hd95_tc', 0):.4f}, ET: {metrics.get('lw_hd95_et', 0):.4f}")
        print("-" * 40)

    if runs > 1:
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
    parser = argparse.ArgumentParser(description="SegResMamba 评估")
    parser.add_argument("--model", type=str, default="segresmamba_lite", help="模型名称")
    parser.add_argument("--version", type=str, default="v2",
                        choices=["v1", "v2", "v3", "v4", "v6", "v7", "v8"],
                        help="模型版本")
    parser.add_argument("--resolution", type=float, default=2.0, help="分辨率")
    parser.add_argument("--checkpoint", type=str, default=None, help="检查点路径")
    parser.add_argument("--init_filters", type=int, default=None, help="初始滤波器数量")
    parser.add_argument("--d_state", type=int, default=8, help="Mamba 状态维度")
    parser.add_argument("--expand", type=int, default=2, help="Mamba 扩展因子")
    parser.add_argument("--no_attention", action="store_true", help="禁用注意力机制")
    parser.add_argument("--deep_supervision", action="store_true", default=True, help="启用深层监督")
    parser.add_argument("--no_deep_supervision", action="store_false", dest="deep_supervision", help="禁用深层监督")
    parser.add_argument("--workers", type=str, default="auto", help="数据加载线程数")
    parser.add_argument("--cache", action="store_true", help="启用硬盘缓存")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    parser.add_argument("--no_lesion_wise", action="store_true", help="禁用 lesion-wise 指标")
    parser.add_argument("--runs", type=int, default=1, help="评估次数")
    parser.add_argument("--visualize_attention", action="store_true", help="V6: 可视化 MoA 注意力权重")
    args = parser.parse_args()

    _default_ckpt = os.path.join(_version_root, "pipeline", "models", f"2.0mm_{args.version}_cuda", "best_metric_model.pth")
    if args.checkpoint is None:
        args.checkpoint = _default_ckpt

    if not os.path.exists(args.checkpoint):
        print(f"警告: checkpoint 不存在 {args.checkpoint}")
        print(f"请先训练: python -m pipeline.train --version {args.version} ...")

    if args.workers == "auto" or args.workers is None:
        num_workers = None
    else:
        num_workers = int(args.workers)

    evaluate(
        model_name=args.model,
        version=args.version,
        resolution=args.resolution,
        checkpoint_path=args.checkpoint,
        init_filters=args.init_filters,
        use_attention=not args.no_attention,
        d_state=args.d_state,
        expand=args.expand,
        use_deep_supervision=args.deep_supervision,
        num_workers=num_workers,
        use_disk_cache=args.cache,
        lesion_wise=not args.no_lesion_wise,
        runs=args.runs,
        device=args.device,
        visualize_attention=args.visualize_attention,
    )
