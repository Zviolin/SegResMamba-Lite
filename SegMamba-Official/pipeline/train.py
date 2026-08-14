"""
SegMamba-Official 训练流程 - 官方SegMamba实现

基于 https://github.com/ge-xing/SegMamba 1:1 实现
论文: SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation
arXiv: arXiv:2401.13560 (MICCAI 2024)
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

from models import get_model
from frame.train import OptimizedTrainer


def train(
    model_name="segmamba",
    version_name="official",
    resolution=2.0,
    num_workers=None,              # 数据加载线程数，None 表示自动计算
    batch_size=None,            # 批次大小，None 表示自动计算
    use_disk_cache=False,
    use_ema=True,
    use_deep_supervision=False,
    max_epochs=1000,
    val_interval=2,
    device=None,
):
    """
    SegMamba官方训练流程
    按照官方配置: SGD, lr=1e-2, batch_size=2, max_epochs=1000
    """
    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    logger = setup_logger(LOG_DIR, f"train_{version_name}_{resolution}mm")
    
    print_device_info(logger)
    setup_cuda_optimization()
    
    DEVICE = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # 官方SegMamba配置
    BATCH_SIZE = batch_size
    
    # 官方优化器: SGD
    LR = 1e-2
    WEIGHT_DECAY = 3e-5
    
    CACHE_DIR = r"D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache"
    
    logger.print_config_grouped({
        "模型设置": {
            "model_name": model_name,
            "version_name": version_name,
            "resolution": f"{resolution} mm",
            "model_info": "OFFICIAL SegMamba (feat_size=[48, 96, 192, 384])",
            "device": DEVICE,
        },
        "训练设置": {
            "batch_size": BATCH_SIZE,
            "num_workers": num_workers,
            "max_epochs": max_epochs,
            "use_ema": use_ema,
            "use_deep_supervision": use_deep_supervision,
            "use_disk_cache": use_disk_cache,
            "val_interval": val_interval,
        },
        "优化器设置": {
            "loss_name": "CrossEntropyLoss",
            "optimizer": "SGD (official)",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "momentum": 0.99,
            "nesterov": True,
            "scheduler": "Poly (official)",
        },
    }, title="训练配置 - OFFICIAL SegMamba")
    
    current_cache_dir = f"{CACHE_DIR}_{resolution}" if use_disk_cache else None
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
        batch_size=batch_size,
        num_workers=num_workers,
        pixdim=pixdim,
        cache_dir=train_cache_dir,
        model_name=model_name,
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
    
    model = get_model(model_name, in_channels=4, out_channels=4, device=DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    from shared.utils.logger import format_num_params
    
    logger.print_config_grouped({
        "模型参数": {
            "total_params": format_num_params(total_params),
            "trainable_params": format_num_params(trainable_params),
            "note": "Official SegMamba (74.87M)",
        },
    }, title="模型信息")
    
    trainer = OptimizedTrainer(
        model=model,
        device=DEVICE,
        loss_name="CrossEntropyLoss",
        optimizer_type="SGD",
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        scheduler_type="Poly",
        max_epochs=max_epochs,
        use_ema=use_ema,
        use_deep_supervision=use_deep_supervision,
    )
    
    # 配置官方优化器
    trainer.optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        momentum=0.99,
        nesterov=True
    )
    
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", f"{resolution}mm_{DEVICE.replace(':', '')}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    best_metric = -1
    best_metric_epoch = -1
    start_epoch = 0
    
    spacing = pixdim
    
    latest_checkpoint = os.path.join(MODEL_DIR, "latest_checkpoint.pth")
    if os.path.exists(latest_checkpoint):
        print(f"发现完整检查点：{latest_checkpoint}，正在恢复训练...")
        checkpoint = torch.load(latest_checkpoint, map_location=DEVICE, weights_only=False)
        
        if "model_state_dict" not in checkpoint:
            print(f"⚠️  检测到旧格式 checkpoint")
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            if trainer.ema is not None and "ema_state_dict" in checkpoint:
                trainer.ema.load_state_dict(checkpoint["ema_state_dict"])
            start_epoch = checkpoint["epoch"]
            best_metric = checkpoint.get("best_metric", -1)
            best_metric_epoch = checkpoint.get("best_metric_epoch", -1)
            print(f"已恢复至 Epoch {start_epoch}，当前最佳 Dice: {best_metric:.4f}")
    
    logger.print_section("开始训练 - OFFICIAL SegMamba")
    
    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.perf_counter()
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        train_loss = trainer.train_epoch(train_loader)
        epoch_time = time.perf_counter() - epoch_start
        print(f"Train Loss: {train_loss:.4f}, LR: {trainer.optimizer.param_groups[0]['lr']:.6f}, Time: {epoch_time:.1f}s")
        
        checkpoint_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
            "scaler_state_dict": trainer.scaler.state_dict(),
            "epoch": epoch + 1,
            "best_metric": best_metric,
            "best_metric_epoch": best_metric_epoch,
        }
        if trainer.ema is not None:
            checkpoint_dict["ema_state_dict"] = trainer.ema.get_state_dict()
        torch.save(checkpoint_dict, latest_checkpoint)
        
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
                    if trainer.ema is not None:
                        torch.save(
                            trainer.ema.get_state_dict(),
                            os.path.join(MODEL_DIR, "best_metric_ema_model.pth")
                        )
                    print(f"新的最佳 WT Dice: {best_metric:.4f}")
                
                checkpoint_dict = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "scheduler_state_dict": trainer.scheduler.state_dict(),
                    "scaler_state_dict": trainer.scaler.state_dict(),
                    "epoch": epoch + 1,
                    "best_metric": best_metric,
                    "best_metric_epoch": best_metric_epoch,
                }
                if trainer.ema is not None:
                    checkpoint_dict["ema_state_dict"] = trainer.ema.get_state_dict()
                torch.save(checkpoint_dict, latest_checkpoint)
                print(f"✓ 验证完成，checkpoint 已更新")
                
            except Exception as e:
                print(f"\n⚠️  验证失败 (Epoch {epoch + 1}): {str(e)}")
                print(f"⚠️  已保存当前训练状态，下次运行时将跳过本次验证")
                continue
    
    print(f"\n训练完成！最佳 Dice: {best_metric:.4f} (Epoch {best_metric_epoch})")
    
    logger.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SegMamba-Official 训练 - 官方SegMamba实现")
    parser.add_argument("--model", type=str, default="segmamba", help="模型名称")
    parser.add_argument("--resolution", type=float, default=2.0, help="分辨率")
    parser.add_argument("--workers", type=str, default="auto", help="数据加载线程数 (auto/数字，auto表示自动计算)")
    parser.add_argument("--batch", type=int, default=2, help="批次大小 (official: 2)")
    parser.add_argument("--cache", action="store_true", help="启用硬盘缓存")
    parser.add_argument("--ema", action="store_true", help="启用EMA")
    parser.add_argument("--epochs", type=int, default=1000, help="训练轮数 (official: 1000)")
    parser.add_argument("--val_interval", type=int, default=2, help="验证间隔 (official: 2)")
    parser.add_argument("--device", type=str, default=None, help="设备")
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
        version_name="official",
        resolution=args.resolution,
        num_workers=workers_arg,
        batch_size=batch_arg,
        use_disk_cache=args.cache,
        use_ema=args.ema,
        max_epochs=args.epochs,
        val_interval=args.val_interval,
        device=args.device,
    )

