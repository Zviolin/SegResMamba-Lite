"""
【Optimized 训练流程】
完整训练流程：数据加载 + 模型 + 框架
支持深层监督和权重滑动平均（SWA 等权；历史命名"EMA"实为 SWA）
"""

import os
import sys
import json
import time
import torch

_version_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_code_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _version_root)
sys.path.insert(0, _code_root)

from shared.data.dataloader import get_train_dataloader, get_val_dataloader, auto_roi_size_from_cache
from shared.utils.logger import setup_logger
from shared.utils.device import setup_cuda_optimization
from shared.utils.device_info import print_device_info
from shared.utils.paths import resolve_log_prefix, resolve_model_dir, resolve_cache_dir

from models import get_model
from frame.train import OptimizedTrainer


def train(
    # ═══════════════════════════════════════════════════════════════════════
    # 命令行参数
    # ═══════════════════════════════════════════════════════════════════════
    model_name="segresnet",      # 模型名称 (segresnet/mamba_unet/vm_unet)
    version_name="optimized",     # 版本名称（内部使用）
    resolution=1.0,             # 数据分辨率 (1.0/2.0/3.0/4.0 mm)
    # ═══════════════════════════════════════════════════════════════════════
    # 训练参数
    # ═══════════════════════════════════════════════════════════════════════
    num_workers=None,              # 数据加载线程数，None 表示自动计算
    batch_size=None,            # 批次大小，None 表示自动计算
    use_disk_cache=False,        # 是否使用数据缓存
    use_swa=True,               # 是否启用权重滑动平均（SWA 等权；历史命名"EMA"实为 SWA）
    use_deep_supervision=False, # 是否启用深层监督
    max_epochs=150,              # 训练轮数
    val_interval=5,             # 验证间隔 (每N个epoch验证一次)
    # ═══════════════════════════════════════════════════════════════════════
    # 设备参数
    # ═══════════════════════════════════════════════════════════════════════
    device=None,                 # 设备类型 (cuda/cpu)
    model_dir="",                # 自定义权重保存目录（空=默认 pipeline/models/{分辨率}mm_{设备}）
    log_name="",                 # 自定义日志名称前缀（空=默认 train_{版本}_{分辨率}mm）
):
    """
    训练入口

    Args:
        model_name: 模型名称
        version_name: 版本名称（内部使用）
        resolution: 数据分辨率
        num_workers: 数据加载线程数
        batch_size: 批次大小
        use_disk_cache: 是否使用数据缓存
        use_swa: 是否启用权重滑动平均（SWA 等权）
        use_deep_supervision: 是否启用深层监督
        max_epochs: 训练轮数
        val_interval: 验证间隔
        device: 设备类型 (cuda/cpu)
    """
    # ─────────────────────────────────────────────────────────────────────
    # 日志配置
    # ─────────────────────────────────────────────────────────────────────
    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    logger = setup_logger(LOG_DIR, resolve_log_prefix(log_name, f"train_{version_name}_{resolution}mm"))

    # ─────────────────────────────────────────────────────────────────────
    # 设备信息
    # ─────────────────────────────────────────────────────────────────────
    print_device_info(logger)

    # ─────────────────────────────────────────────────────────────────────
    # CUDA 优化
    # ─────────────────────────────────────────────────────────────────────
    setup_cuda_optimization()

    # ─────────────────────────────────────────────────────────────────────
    # 设备配置
    # ─────────────────────────────────────────────────────────────────────
    DEVICE = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    # ─────────────────────────────────────────────────────────────────────
    # 训练超参数
    # ─────────────────────────────────────────────────────────────────────
    BATCH_SIZE = batch_size
    LR = 1e-4

    CACHE_DIR = r"D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache"

    logger.print_config_grouped({
        "模型设置": {
            "model_name": model_name,
            "version_name": version_name,
            "resolution": f"{resolution} mm",
            "device": DEVICE,
        },
        "训练设置": {
            "batch_size": BATCH_SIZE,
            "num_workers": num_workers,
            "max_epochs": max_epochs,
            "use_swa": use_swa,
            "use_deep_supervision": use_deep_supervision,
            "use_disk_cache": use_disk_cache,
            "val_interval": val_interval,
        },
        "优化器设置": {
            "loss_name": "DiceFocalLoss",
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": 1e-5,
            "scheduler": "CosineAnnealingLR",
        },
    }, title="训练配置")

    current_cache_dir = resolve_cache_dir(resolution, CACHE_DIR) if use_disk_cache else None
    pixdim = (resolution, resolution, resolution)

    if use_disk_cache and current_cache_dir:
        split_file = os.path.join(current_cache_dir, "split_info.json")
        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                split_info = json.load(f)
            train_case_ids = set(split_info["train_case_ids"])
            val_case_ids = set(split_info["val_case_ids"])
            print(f"从缓存读取数据划分 (seed={split_info.get('seed', 'unknown')})")
        else:
            raise FileNotFoundError(f"划分信息文件不存在: {split_file}，请先运行 prepare_data 生成缓存")
    else:
        raise ValueError("必须启用缓存 (--cache) 并确保已生成缓存")

    train_cache_dir = os.path.join(current_cache_dir, "train")
    val_cache_dir = os.path.join(current_cache_dir, "val")

    train_loader = get_train_dataloader(
        [],
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pixdim=pixdim,
        cache_dir=train_cache_dir,
    )
    val_loader = get_val_dataloader(
        [],
        batch_size=1,
        num_workers=num_workers,
        pixdim=pixdim,
        cache_dir=val_cache_dir,
    )

    roi_size = auto_roi_size_from_cache(val_cache_dir, pixdim)

    print(f"训练样本数: {len(train_loader.dataset)}, 验证样本数: {len(val_loader.dataset)}")

    model = get_model(model_name, device=DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    from shared.utils.logger import format_num_params

    logger.print_config_grouped({
        "模型参数": {
            "total_params": format_num_params(total_params),
            "trainable_params": format_num_params(trainable_params),
        },
    }, title="模型信息")

    trainer = OptimizedTrainer(
        model=model,
        device=DEVICE,
        loss_name="DiceFocalLoss",
        optimizer_type="AdamW",
        lr=LR,
        weight_decay=1e-5,
        scheduler_type="CosineAnnealingLR",
        max_epochs=max_epochs,
        use_swa=use_swa,
        use_deep_supervision=use_deep_supervision,
    )

    MODEL_DIR = resolve_model_dir(
        model_dir,
        os.path.dirname(os.path.abspath(__file__)),
        f"{resolution}mm_{DEVICE.replace(':', '')}",
    )
    os.makedirs(MODEL_DIR, exist_ok=True)

    best_metric = -1
    best_metric_epoch = -1
    start_epoch = 0
    last_validated_epoch = 0  # 记录最后成功验证的 epoch

    spacing = pixdim

    latest_checkpoint = os.path.join(MODEL_DIR, "latest_checkpoint.pth")
    if os.path.exists(latest_checkpoint):
        print(f"发现完整检查点：{latest_checkpoint}，正在恢复训练...")
        checkpoint = torch.load(latest_checkpoint, map_location=DEVICE, weights_only=False)
        
        # 向后兼容：检查 checkpoint 格式
        if "model_state_dict" not in checkpoint:
            print(f"⚠️  检测到旧格式 checkpoint，尝试从 epoch 文件恢复...")
            import glob
            epoch_files = glob.glob(os.path.join(MODEL_DIR, "epoch_*.pth"))
            if epoch_files:
                latest_epoch_file = max(epoch_files, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
                print(f"使用文件：{latest_epoch_file}")
                model.load_state_dict(torch.load(latest_epoch_file, map_location=DEVICE))
                if trainer.swa is not None:
                    # 平均权重恢复兼容双文件名：新名 best_metric_swa_model.pth 优先，旧名 best_metric_ema_model.pth（历史命名）回退
                    _swa_new = os.path.join(MODEL_DIR, "best_metric_swa_model.pth")
                    _swa_old = os.path.join(MODEL_DIR, "best_metric_ema_model.pth")
                    _swa_path = _swa_new if os.path.exists(_swa_new) else (_swa_old if os.path.exists(_swa_old) else None)
                    if _swa_path:
                        trainer.swa.load_state_dict(torch.load(_swa_path, map_location=DEVICE))
                        print(f"✓ 已恢复平均权重状态（{os.path.basename(_swa_path)}）")
                print(f"✓ 已从 {latest_epoch_file} 恢复模型权重")
                print(f"⚠️  但优化器状态和其他信息已丢失，将从头开始训练")
            else:
                print(f"⚠️  未找到 epoch 文件，将从头开始训练")
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            # 断点兼容：新键 swa_state_dict 优先，旧键 ema_state_dict（历史命名）同样可恢复
            _avg_sd = checkpoint.get("swa_state_dict") or checkpoint.get("ema_state_dict")
            if trainer.swa is not None and _avg_sd is not None:
                trainer.swa.load_state_dict(_avg_sd)
            start_epoch = checkpoint["epoch"]
            best_metric = checkpoint.get("best_metric", -1)
            best_metric_epoch = checkpoint.get("best_metric_epoch", -1)
            last_validated_epoch = checkpoint.get("last_validated_epoch", 0)
            print(f"已恢复至 Epoch {start_epoch}，当前最佳 Dice: {best_metric:.4f}，最后验证 epoch: {last_validated_epoch}")

    logger.print_section("开始训练")

    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.perf_counter()
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        train_loss = trainer.train_epoch(train_loader)
        epoch_time = time.perf_counter() - epoch_start
        print(f"Train Loss: {train_loss:.4f}, LR: {trainer.optimizer.param_groups[0]['lr']:.6f}, Time: {epoch_time:.1f}s")

        # 每个 epoch 都保存 latest_checkpoint（断点续传）
        # 注意：验证前保存基础状态，验证后更新最佳状态
        checkpoint_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
            "scaler_state_dict": trainer.scaler.state_dict(),
            "epoch": epoch + 1,
            "best_metric": best_metric,
            "best_metric_epoch": best_metric_epoch,
        }
        if trainer.swa is not None:
            checkpoint_dict["swa_state_dict"] = trainer.swa.get_state_dict()
        torch.save(
            checkpoint_dict,
            latest_checkpoint,
        )

        # 验证循环
        if (epoch + 1) % val_interval == 0:
            val_start = time.perf_counter()
            print(f"\n开始验证 (Epoch {epoch + 1})...")
            try:
                metrics = trainer.validate(val_loader, spacing=spacing, roi_size=roi_size)
                val_time = time.perf_counter() - val_start
                print(f"Val Dice (WT/TC/ET): {metrics['dice_wt']:.4f}/{metrics['dice_tc']:.4f}/{metrics['dice_et']:.4f}")
                print(f"Val Dice Avg: {metrics['dice_avg']:.4f}")
                print(f"Val HD95 (WT/TC/ET): {metrics['hd95_wt']:.4f}/{metrics['hd95_tc']:.4f}/{metrics['hd95_et']:.4f}")
                print(f"Val HD95 Avg: {metrics['hd95_avg']:.4f}, Time: {val_time:.1f}s")

                epoch_model_path = os.path.join(MODEL_DIR, f"epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), epoch_model_path)

                if metrics["dice_wt"] > best_metric:
                    best_metric = metrics["dice_wt"]
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_metric_model.pth"))
                    if trainer.swa is not None:
                        torch.save(
                            trainer.swa.get_state_dict(),
                            os.path.join(MODEL_DIR, "best_metric_swa_model.pth")
                        )
                    print(f"新的最佳 WT Dice: {best_metric:.4f}")

                # 验证成功后，更新 checkpoint 中的最佳状态
                checkpoint_dict = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "scheduler_state_dict": trainer.scheduler.state_dict(),
                    "scaler_state_dict": trainer.scaler.state_dict(),
                    "epoch": epoch + 1,
                    "best_metric": best_metric,
                    "best_metric_epoch": best_metric_epoch,
                }
                if trainer.swa is not None:
                    checkpoint_dict["swa_state_dict"] = trainer.swa.get_state_dict()
                torch.save(
                    checkpoint_dict,
                    latest_checkpoint,
                )
                print(f"✓ 验证完成，checkpoint 已更新")

            except Exception as e:
                print(f"\n⚠️  验证失败 (Epoch {epoch + 1}): {str(e)}")
                print(f"⚠️  已保存当前训练状态，下次运行时将跳过本次验证")
                print(f"⚠️  建议：减小 batch size 或 patch size 后重新训练")
                continue

    print(f"\n训练完成！最佳 Dice: {best_metric:.4f} (Epoch {best_metric_epoch})")

    logger.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optimized 训练")
    parser.add_argument("--model", type=str, default="segresnet", help="模型名称")
    parser.add_argument("--resolution", type=float, default=1.0, help="分辨率")
    parser.add_argument("--workers", type=str, default="auto", help="数据加载线程数 (auto/数字，auto表示自动计算)")
    parser.add_argument("--batch", type=str, default="2", help="批次大小 (auto/数字，auto表示自动计算)")
    parser.add_argument("--cache", action="store_true", help="启用硬盘缓存")
    parser.add_argument("--swa", action="store_true", help="启用权重滑动平均（SWA 等权；历史命名 --ema 实为 SWA）")
    parser.add_argument("--ds", action="store_true", help="启用深层监督")
    parser.add_argument("--epochs", type=int, default=150, help="训练轮数")
    parser.add_argument("--val_interval", type=int, default=5, help="验证间隔")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    parser.add_argument("--model_dir", type=str, default="",
                        help="自定义权重保存目录（优先级高于默认规则；默认 pipeline/models/{分辨率}mm_{设备}）")
    parser.add_argument("--log_name", type=str, default="",
                        help="自定义日志名称前缀（默认 train_{版本}_{分辨率}mm）")
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

    # 解析 batch_size 参数
    if args.batch.lower() == "auto":
        batch_arg = None
    else:
        try:
            batch_arg = int(args.batch)
        except ValueError:
            print(f"警告: 无效的 --batch 值 '{args.batch}'，使用 auto")
            batch_arg = None

    train(
        model_name=args.model,
        version_name="optimized",
        resolution=args.resolution,
        num_workers=workers_arg,
        batch_size=batch_arg,
        use_disk_cache=args.cache,
        use_swa=args.swa,
        use_deep_supervision=args.ds,
        max_epochs=args.epochs,
        val_interval=args.val_interval,
        device=args.device,
        model_dir=args.model_dir,
        log_name=args.log_name,
    )
