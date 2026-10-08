"""
【LightSegMamba 训练流程 V3】

CLI 接口（对齐 MambaUNet-DiceCE-AdamW 风格 + LightSegMamba 特有参数）：

  # 200 例快速验证
  python pipeline/train.py \\
    --model lightsegmamba \\
    --resolution 2.0 \\
    --cache \\
    --max-samples 200 \\
    --batch 1 --workers 0 \\
    --patch-size 64 64 64 \\
    --base-channels 32 \\
    --epochs 50 --val-interval 5 \\
    --device cuda

  # 1251 例全量训练
  python pipeline/train.py \\
    --model lightsegmamba \\
    --resolution 2.0 \\
    --cache \\
    --batch 1 --workers 0 \\
    --patch-size 64 64 64 \\
    --base-channels 32 \\
    --epochs 100 --val-interval 5 \\
    --device cuda

缓存目录约定（V3 改动）：
  - 不传 --max-samples 或 --max-samples 1251：persistent_cache_{resolution}/
  - --max-samples N (N < 1251)：persistent_cache_{resolution}_{N}/
  例：--max-samples 200 → persistent_cache_2.0_200/
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import torch

_VERSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE_ROOT = os.path.dirname(_VERSION_ROOT)
# 默认缓存父目录（2026-10-07 定案：全库统一默认主项目体系缓存，与其他 6 项目一致；换机器用 SRTP_CACHE_PARENT 整机重定向）
_DEFAULT_CACHE_PARENT = r"D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache"
sys.path.insert(0, _VERSION_ROOT)
sys.path.insert(0, _CODE_ROOT)

from shared.data.dataloader import get_train_dataloader, get_val_dataloader, auto_roi_size_from_cache
from shared.utils.logger import setup_logger
from shared.utils.device import setup_cuda_optimization, setup_determinism
from shared.utils.device_info import print_device_info
from shared.utils.paths import resolve_log_prefix, resolve_model_dir, resolve_cache_dir

from models import get_model
from frame.train import LightSegMambaTrainer


# ════════════════════════════════════════════════════════════════════════════
# 训练入口
# ════════════════════════════════════════════════════════════════════════════
def train(
    # 模型
    model_name: str = "lightsegmamba",
    version_name: str = "lightsegmamba_v3",
    base_channels: int = 32,
    # 数据
    resolution: float = 2.0,
    use_disk_cache: bool = False,
    max_samples: int | None = None,
    num_workers: int = 0,
    batch_size: int = 1,
    patch_size: tuple = (64, 64, 64),
    # 训练
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    max_epochs: int = 50,
    val_interval: int = 5,
    use_swa: bool = False,
    # 设备
    device: str = None,
    # 命令行扩展
    model_dir: str = "",        # 自定义权重保存目录（空=默认 pipeline/models/{分辨率}mm_{样本数}_{设备}）
    log_name: str = "",         # 自定义日志名称前缀（空=默认 train_{版本}_{分辨率}mm）
    # 可选训练特性（默认关闭，行为与历史完全一致）
    seed: int = 42,                # 训练随机种子（--deterministic 时锁定随机性，不影响数据划分缓存）
    deterministic: bool = False,   # 可复现性：开启确定性训练（cudnn.deterministic + 确定性算法，不触碰 TF32）
    cache_parent: str = "",        # 数据缓存父目录覆盖（优先级：命令行 > 环境变量 SRTP_CACHE_PARENT > 各项目默认）
    skip_nonfinite: bool = False,  # 跳过非有限 loss（NaN/Inf）的 batch，不 backward/不更新/不进平均权重
    best_weights: str = "raw",     # best_metric_model.pth 保存来源：raw=原始模型权重（历史口径）/ avg=验证所用的平均权重（SWA）
):
    """
    训练入口。

    Args:
        model_name: 模型名称（仅 lightsegmamba）
        version_name: 版本名（用于日志与保存目录命名）
        base_channels: Stem 输出通道数（默认 32，约 1.4M 参数）
        resolution: 重采样分辨率（mm）
        use_disk_cache: 是否使用 PersistentDataset 缓存（必须 True，且需先 gen_cache）
        max_samples: 限样本数（None=全量 1251；200=快速验证）
        num_workers: DataLoader worker 数（4GB 显存机器建议 0-2）
        batch_size: 批大小（4GB 显存建议 1）
        patch_size: 训练 patch 尺寸（4GB 显存建议 64³）
        lr: 初始学习率
        weight_decay: AdamW 权重衰减
        max_epochs: 总 epoch 数
        val_interval: 每 N 个 epoch 验证一次
        use_swa: 是否启用权重滑动平均（SWA 等权）
        device: cuda / cpu
        seed: 训练随机种子（--deterministic 时锁定随机性，不影响数据划分缓存）
        deterministic: 是否开启确定性训练（cudnn.deterministic + 确定性算法，不触碰 TF32）
        cache_parent: 数据缓存父目录覆盖（空=默认 / 环境变量 SRTP_CACHE_PARENT）
        skip_nonfinite: 是否跳过非有限 loss（NaN/Inf）的 batch
        best_weights: best_metric_model.pth 保存来源（raw=原始权重 / avg=SWA 平均权重）
    """
    # 可复现性：cuBLAS 确定性工作区配置必须在任何 CUDA 上下文创建之前设置
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # ── 日志 ──────────────────────────────────────────────────────────────
    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    logger = setup_logger(LOG_DIR, resolve_log_prefix(log_name, f"train_{version_name}_{resolution}mm"))

    # ── 设备信息 ──────────────────────────────────────────────────────────
    print_device_info(logger)

    # ── CUDA 优化 ─────────────────────────────────────────────────────────
    setup_cuda_optimization()

    if deterministic:
        # 可复现性：关 benchmark + 锁定确定性算法（不触碰 TF32，保持论文口径可比）
        setup_determinism(seed)

    DEVICE = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    # ── 训练超参 ──────────────────────────────────────────────────────────
    LR = lr
    BATCH_SIZE = batch_size
    PATCH_SIZE = patch_size if isinstance(patch_size, tuple) else tuple(patch_size)

    # ── 缓存目录解析 ──────────────────────────────────────────────────────
    if not use_disk_cache:
        raise ValueError("LightSegMamba-V3 必须启用缓存（--cache），请先运行 pipeline/gen_cache.py")

    # max_samples 必须以关键字传参（新版 shared 签名第 3 位是 cli_parent）：
    # cli_parent 覆盖父目录，max_samples 追加 _{N} 采样后缀，两者正交工作；
    # 两者皆空/None 时与历史目录完全一致
    current_cache_dir = resolve_cache_dir(resolution, _DEFAULT_CACHE_PARENT, max_samples=max_samples, cli_parent=cache_parent)

    # 标签用于模型保存目录区分（200 vs 全量）
    samples_tag = "full" if (max_samples is None or max_samples >= 1251) else f"{max_samples}"

    logger.print_config_grouped({
        "模型设置": {
            "model_name": model_name,
            "version_name": version_name,
            "resolution": f"{resolution} mm",
            "base_channels": base_channels,
            "device": DEVICE,
        },
        "训练设置": {
            "batch_size": BATCH_SIZE,
            "patch_size": PATCH_SIZE,
            "num_workers": num_workers,
            "max_epochs": max_epochs,
            "use_swa": use_swa,
            "val_interval": val_interval,
            "use_disk_cache": use_disk_cache,
        },
        "数据设置": {
            "max_samples": "全量 1251" if max_samples is None else f"{max_samples} 例",
            "cache_dir": current_cache_dir,
        },
        "优化器设置": {
            "loss_name": "DiceCELoss",
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": weight_decay,
            "scheduler": "CosineAnnealingLR",
        },
    }, title="训练配置")

    # ── 数据划分 / 缓存 ───────────────────────────────────────────────────
    split_file = os.path.join(current_cache_dir, "split_info.json")
    if not os.path.exists(split_file):
        raise FileNotFoundError(
            f"划分信息文件不存在: {split_file}\n"
            f"请先运行：python pipeline/gen_cache.py --resolution {resolution} "
            f"--max-samples {max_samples if max_samples else '1251'} --generate --cache"
        )
    with open(split_file, "r") as f:
        split_info = json.load(f)
    print(f"从缓存读取数据划分 (seed={split_info.get('seed', 'unknown')})")
    print(f"  train: {len(split_info['train_case_ids'])} 例, "
          f"val: {len(split_info['val_case_ids'])} 例, "
          f"test: {len(split_info['test_case_ids'])} 例")

    train_cache_dir = os.path.join(current_cache_dir, "train")
    val_cache_dir = os.path.join(current_cache_dir, "val")

    pixdim = (resolution, resolution, resolution)

    train_loader = get_train_dataloader(
        [],
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pixdim=pixdim,
        patch_size=PATCH_SIZE,
        cache_dir=train_cache_dir,
    )
    val_loader = get_val_dataloader(
        [],
        batch_size=1,
        num_workers=num_workers,
        pixdim=pixdim,
        patch_size=PATCH_SIZE,
        cache_dir=val_cache_dir,
    )

    roi_size = auto_roi_size_from_cache(val_cache_dir, pixdim)

    print(f"训练样本数: {len(train_loader.dataset)}, 验证样本数: {len(val_loader.dataset)}")

    # ── 模型 ──────────────────────────────────────────────────────────────
    model = get_model(
        model_name,
        in_channels=4,
        out_channels=4,
        device=DEVICE,
        base_channels=base_channels,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    from shared.utils.logger import format_num_params
    logger.print_config_grouped({
        "模型参数": {
            "total_params": format_num_params(total_params),
            "trainable_params": format_num_params(trainable_params),
        },
    }, title="模型信息")

    # ── 训练器 ────────────────────────────────────────────────────────────
    trainer = LightSegMambaTrainer(
        model=model,
        device=DEVICE,
        loss_name="DiceCELoss",
        optimizer_type="AdamW",
        lr=LR,
        weight_decay=weight_decay,
        scheduler_type="CosineAnnealingLR",
        max_epochs=max_epochs,
        use_swa=use_swa,
        skip_nonfinite=skip_nonfinite,
    )

    # ── 模型保存目录（区分 max_samples） ─────────────────────────────────
    MODEL_DIR = resolve_model_dir(
        model_dir,
        os.path.dirname(os.path.abspath(__file__)),
        f"{resolution}mm_{samples_tag}_{DEVICE.replace(':', '')}",
    )
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 断点续传 ──────────────────────────────────────────────────────────
    best_metric = -1.0
    best_metric_epoch = -1
    start_epoch = 0
    last_validated_epoch = 0

    latest_checkpoint = os.path.join(MODEL_DIR, "latest_checkpoint.pth")
    if os.path.exists(latest_checkpoint):
        print(f"发现完整检查点：{latest_checkpoint}，正在恢复训练...")
        checkpoint = torch.load(latest_checkpoint, map_location=DEVICE, weights_only=False)

        if "model_state_dict" not in checkpoint:
            print("⚠️  检测到旧格式 checkpoint，尝试从 epoch 文件恢复...")
            import glob
            epoch_files = glob.glob(os.path.join(MODEL_DIR, "epoch_*.pth"))
            if epoch_files:
                latest_epoch_file = max(
                    epoch_files,
                    key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]),
                )
                model.load_state_dict(torch.load(latest_epoch_file, map_location=DEVICE))
                print(f"✓ 已从 {latest_epoch_file} 恢复模型权重（优化器状态丢失，从头训练）")
            else:
                print("⚠️  未找到 epoch 文件，将从头开始训练")
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint["epoch"]
            best_metric = checkpoint.get("best_metric", -1.0)
            best_metric_epoch = checkpoint.get("best_metric_epoch", -1)
            last_validated_epoch = checkpoint.get("last_validated_epoch", 0)
            print(f"已恢复至 Epoch {start_epoch}，当前最佳 WT Dice: {best_metric:.4f}")

    # ── 训练循环 ──────────────────────────────────────────────────────────
    logger.print_section("开始训练")

    for epoch in range(start_epoch, max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        train_loss = trainer.train_epoch(train_loader)
        print(f"Train Loss: {train_loss:.4f}, LR: {trainer.optimizer.param_groups[0]['lr']:.6f}")

        if (epoch + 1) % val_interval == 0:
            if last_validated_epoch >= epoch + 1:
                print(f"\nEpoch {epoch + 1} 已验证过，跳过")
            else:
                print(f"\n开始验证 (Epoch {epoch + 1})...")
                try:
                    metrics = trainer.validate(
                        val_loader,
                        spacing=pixdim,
                        roi_size=roi_size,
                    )
                    print(
                        f"Val Dice (WT/TC/ET): {metrics['dice_wt']:.4f}/"
                        f"{metrics['dice_tc']:.4f}/{metrics['dice_et']:.4f}"
                    )
                    print(f"Val Dice Avg: {metrics['dice_avg']:.4f}")
                    print(
                        f"Val HD95 (WT/TC/ET): {metrics['hd95_wt']:.4f}/"
                        f"{metrics['hd95_tc']:.4f}/{metrics['hd95_et']:.4f}"
                    )
                    print(f"Val HD95 Avg: {metrics['hd95_avg']:.4f}")

                    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"epoch_{epoch + 1}.pth"))

                    if metrics["dice_wt"] > best_metric:
                        best_metric = metrics["dice_wt"]
                        best_metric_epoch = epoch + 1
                        # --best_weights avg：保存验证所用的平均权重（SWA，与 validate 同源）；
                        # raw（默认，历史口径）：保存原始模型权重
                        if best_weights == "avg" and trainer.swa is not None:
                            torch.save(trainer.swa.averaged_model.state_dict(), os.path.join(MODEL_DIR, "best_metric_model.pth"))
                            print(f"新的最佳 WT Dice: {best_metric:.4f}（来源: 平均权重）")
                        else:
                            if best_weights == "avg":
                                print("⚠️  未启用权重平均（--swa），--best_weights avg 回退保存原始权重")
                            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_metric_model.pth"))
                            print(f"新的最佳 WT Dice: {best_metric:.4f}")

                    last_validated_epoch = epoch + 1
                except Exception as e:
                    print(f"\n⚠️  验证失败 (Epoch {epoch + 1}): {str(e)}")
                    print(f"⚠️  建议：减小 batch size 或 patch size 后重新训练")

        # 每个 epoch 都保存 latest_checkpoint（断点续传）
        checkpoint_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
            "scaler_state_dict": trainer.scaler.state_dict(),
            "epoch": epoch + 1,
            "best_metric": best_metric,
            "best_metric_epoch": best_metric_epoch,
            "last_validated_epoch": last_validated_epoch,
        }
        torch.save(checkpoint_dict, latest_checkpoint)
        print(f"✓ Epoch {epoch + 1} checkpoint 已保存")

    print(f"\n训练完成！最佳 WT Dice: {best_metric:.4f} (Epoch {best_metric_epoch})")
    logger.close()
    sys.stdout = sys.__stdout__


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightSegMamba-V3 训练")
    parser.add_argument("--model", type=str, default="lightsegmamba", help="模型名称")
    parser.add_argument("--resolution", type=float, default=2.0,
                        help="重采样分辨率 (mm)")
    parser.add_argument("--cache", action="store_true", help="启用硬盘缓存（必须）")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本数（None=全量1251；200=快速验证）")
    parser.add_argument("--batch", type=int, default=1, help="批次大小（4GB 显存建议 1）")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader worker 数（4GB 显存机器建议 0-2）")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[64, 64, 64],
                        help="训练 patch 尺寸（D H W），4GB 显存建议 64³")
    parser.add_argument("--base-channels", type=int, default=32,
                        help="Stem 输出通道数（默认 32，约 1.4M 参数；44≈论文 3M）")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--val-interval", type=int, default=5, help="验证间隔")
    parser.add_argument("--lr", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW 权重衰减")
    parser.add_argument("--swa", action="store_true", help="启用权重滑动平均（SWA 等权；历史 --ema/--ema-decay 已废弃，历史命名 EMA 实为 SWA 且 decay 从未生效）")
    parser.add_argument("--seed", type=int, default=42, help="训练随机种子（不影响数据划分缓存）")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    parser.add_argument("--model_dir", type=str, default="",
                        help="自定义权重保存目录（优先级高于默认规则；默认 pipeline/models/{分辨率}mm_{样本数}_{设备}）")
    parser.add_argument("--log_name", type=str, default="",
                        help="自定义日志名称前缀（默认 train_{版本}_{分辨率}mm）")
    # 跨卡包合并的可选开关（默认关闭，行为与历史完全一致）
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="开启确定性训练：cudnn.deterministic + 确定性算法 + CUBLAS_WORKSPACE_CONFIG（不触碰 TF32）")
    parser.add_argument("--cache_parent", type=str, default="",
                        help="数据缓存父目录覆盖（优先级：命令行 > 环境变量 SRTP_CACHE_PARENT > 各项目默认；最终目录 = {parent}_{分辨率}）")
    parser.add_argument("--skip_nonfinite", action="store_true", default=False,
                        help="跳过非有限 loss（NaN/Inf）的 batch，不 backward/不更新/不进平均权重")
    parser.add_argument("--best_weights", type=str, default="raw", choices=["raw", "avg"],
                        help="best_metric_model.pth 保存来源：raw=原始权重（默认，历史口径）/ avg=验证所用的平均权重（SWA，需配合 --swa）")
    args = parser.parse_args()

    train(
        model_name=args.model,
        resolution=args.resolution,
        use_disk_cache=args.cache,
        max_samples=args.max_samples,
        batch_size=args.batch,
        patch_size=tuple(args.patch_size),
        max_epochs=args.epochs,
        val_interval=args.val_interval,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.workers,
        use_swa=args.swa,
        base_channels=args.base_channels,
        device=args.device,
        model_dir=args.model_dir,
        log_name=args.log_name,
        seed=args.seed,
        deterministic=args.deterministic,
        cache_parent=args.cache_parent,
        skip_nonfinite=args.skip_nonfinite,
        best_weights=args.best_weights,
    )
