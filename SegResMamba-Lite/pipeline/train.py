"""
【SegResMamba-Lite 训练流程 - 高性能优化版】
申报表设计版本：双向VSS + 深度融合 + 特征回流
参数量：init_filters=26 (1.4M) - 满足<1.5M要求

支持版本：v1, v2, v3, v4, v6
v6 在瓶颈处加入 BoundaryAttention（边界注意力）

优化特性：
- torch.compile 模型编译加速 (PyTorch 2.x)
- non_blocking GPU 异步数据传输
- Windows DataLoader 多进程兼容 (PersistentDataset 死锁修复)
- 步长时间分离统计 (数据加载 vs 计算)
"""

import os
import sys
import json
import time
import torch
import argparse
import multiprocessing

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
    # ═══════════════════════════════════════════════════════════════════════
    # 命令行参数
    # ═══════════════════════════════════════════════════════════════════════
    model_name="segresmamba_lite",  # 模型名称
    version_name="lite",            # 版本名称
    version="v2",                   # 模型版本 ('v1'/'v2'/'v3'/'v4'/'v6')
    resolution=2.0,                 # 数据分辨率 (1.0/2.0/3.0/4.0 mm)
    # ═══════════════════════════════════════════════════════════════════════
    # 模型参数
    # ═══════════════════════════════════════════════════════════════════════
    init_filters=None,              # 初始滤波器数量 (None 表示使用默认值)
    use_attention=True,             # 是否使用注意力机制
    d_state=8,                      # Mamba 状态维度
    d_conv=2,                       # Mamba 卷积核大小
    expand=2,                       # Mamba 扩展因子
    num_experts=4,                  # MoA 专家数量 (V6/V7 专用)
    # ═══════════════════════════════════════════════════════════════════════
    # 训练参数
    # ═══════════════════════════════════════════════════════════════════════
    num_workers=None,
    batch_size=None,
    use_disk_cache=False,
    use_ema=True,
    use_deep_supervision=False,
    max_epochs=100,
    val_interval=5,
    # ═══════════════════════════════════════════════════════════════════════
    # 损失函数
    # ═══════════════════════════════════════════════════════════════════════
    loss_name="DiceFocalLoss",
    v6_alpha=0.5,
    v6_beta=0.3,
    v6_gamma=0.2,
    v6_focal_alpha=0.25,
    v6_focal_gamma=2.0,
    # ═══════════════════════════════════════════════════════════════════════
    # 设备参数
    # ═══════════════════════════════════════════════════════════════════════
    device=None,
    use_compile=False,
):
    """
    训练入口

    Args:
        version: 模型版本
            - v1: 原始双向 Mamba
            - v2: 终极深度融合（默认）
            - v3: SE + 双重 Bottleneck
            - v4: 深度融合优化版
            - v6: 边界注意力版（V2 + BoundaryAttention）
    """
    # ─────────────────────────────────────────────────────────────────────
    # 日志配置
    # ─────────────────────────────────────────────────────────────────────
    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    logger = setup_logger(LOG_DIR, f"train_{version_name}_{resolution}mm")

    print_device_info(logger)
    setup_cuda_optimization()

    DEVICE = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = batch_size
    LR = 1e-4

    CACHE_DIR = r"D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache"

    # 设置默认的 init_filters
    if init_filters is None:
        if version == 'v1':
            init_filters = 26
        elif version == 'v2':
            init_filters = 22
        elif version == 'v3':
            init_filters = 18
        elif version == 'v4':
            init_filters = 18
        elif version == 'v6':
            init_filters = 20
        elif version == 'v7':
            init_filters = 18
        elif version == 'v8':
            init_filters = 20
        else:
            init_filters = 20

    logger.print_config_grouped({
        "模型设置": {
            "model_name": model_name,
            "version_name": version_name,
            "version": version,
            "resolution": f"{resolution} mm",
            "init_filters": init_filters,
            "d_state": d_state,
            "expand": expand,
            "use_attention": use_attention,
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
            "loss_name": loss_name,
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": 1e-5,
            "scheduler": "CosineAnnealingLR",
        },
    }, title=f"训练配置 - SegResMamba-Lite {version}")

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

    # ─────────────────────────────────────────────────────────────────────
    # 模型创建
    # ─────────────────────────────────────────────────────────────────────
    model_kwargs = dict(
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
        device=DEVICE
    )

    model = get_model(**model_kwargs)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    from shared.utils.logger import format_num_params

    logger.print_config_grouped({
        "模型参数": {
            "total_params": format_num_params(total_params),
            "trainable_params": format_num_params(trainable_params),
            "note": "init_filters={} ({:.2f}M) - {}".format(
                init_filters, total_params/1e6,
                "✅ 满足<1.5M要求" if total_params < 1.5e6 else "❌ 超过1.5M限制"
            ),
        },
    }, title="模型信息")

    # ─────────────────────────────────────────────────────────────────────
    # 损失函数参数
    # ─────────────────────────────────────────────────────────────────────
    if loss_name == "V6LossSimple":
        loss_kwargs = {
            "alpha": v6_alpha, "beta": v6_beta, "gamma": v6_gamma,
            "focal_alpha": v6_focal_alpha, "focal_gamma": v6_focal_gamma,
        }
    else:
        loss_kwargs = {}

    # ─────────────────────────────────────────────────────────────────────
    # 训练器（统一使用 OptimizedTrainer）
    # ─────────────────────────────────────────────────────────────────────
    trainer = OptimizedTrainer(
        model=model,
        device=DEVICE,
        loss_name=loss_name,
        optimizer_type="AdamW",
        lr=LR,
        weight_decay=1e-5,
        scheduler_type="CosineAnnealingLR",
        max_epochs=max_epochs,
        use_ema=use_ema,
        use_deep_supervision=use_deep_supervision,
        non_blocking=True,
        **loss_kwargs,
    )

    if use_compile and hasattr(torch, 'compile'):
        print("正在编译模型 (torch.compile)...")
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model, mode='max-autotune')
        trainer.model = model
        print("模型编译完成")

    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", f"{resolution}mm_{version}_{DEVICE.replace(':', '')}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    best_metric = -1
    best_metric_epoch = -1
    start_epoch = 0
    last_validated_epoch = 0

    spacing = pixdim

    latest_checkpoint = os.path.join(MODEL_DIR, "latest_checkpoint.pth")
    if os.path.exists(latest_checkpoint):
        print(f"发现完整检查点：{latest_checkpoint}，正在恢复训练...")
        checkpoint = torch.load(latest_checkpoint, map_location=DEVICE, weights_only=False)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            try:
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"警告：optimizer 状态加载失败 ({e})")
        if 'scheduler_state_dict' in checkpoint:
            try:
                trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception:
                pass
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"从 epoch {start_epoch} 继续训练")

    logger.print_info("开始训练...")
    print("=" * 70)
    print("开始训练 SegResMamba-Lite ({} @ {}mm)".format(version, resolution))
    print("=" * 70)

    import time
    t0 = time.time()
    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        print("-" * 50)

        train_loss = trainer.train_epoch(train_loader)
        epoch_time = time.time() - epoch_start

        print(f"  train_loss: {train_loss:.4f}, time: {epoch_time:.1f}s")

        if (epoch + 1) % val_interval == 0 or epoch == max_epochs - 1:
            last_validated_epoch = epoch
            metrics = trainer.validate(
                val_loader,
                spacing=spacing,
                roi_size=roi_size,
            )
            # OptimizedTrainer.validate 返回 dice_avg（不是 mean_dice）
            mean_dice = metrics.get('dice_avg', metrics.get('mean_dice', 0))
            mean_hd95 = metrics.get('hd95_avg', 0)
            print(f"  val_dice: {mean_dice:.4f}  |  val_hd95: {mean_hd95:.4f}")
            print(f"    per-region: WT(D={metrics.get('dice_wt', 0):.4f}/H={metrics.get('hd95_wt', 0):.2f}) "
                  f"TC(D={metrics.get('dice_tc', 0):.4f}/H={metrics.get('hd95_tc', 0):.2f}) "
                  f"ET(D={metrics.get('dice_et', 0):.4f}/H={metrics.get('hd95_et', 0):.2f})")

            if mean_dice > best_metric:
                best_metric = mean_dice
                best_metric_epoch = epoch + 1
                torch.save(
                    trainer.model.state_dict(),
                    os.path.join(MODEL_DIR, "best_metric_model.pth")
                )
                print(f"  ✅ 新最佳模型已保存 (Dice: {mean_dice:.4f})")

        torch.save({
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
        }, latest_checkpoint)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == max_epochs:
            torch.save(
                trainer.model.state_dict(),
                os.path.join(MODEL_DIR, f"epoch_{epoch+1}.pth")
            )

    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print(f"训练完成！")
    print(f"  总时间: {total_time/3600:.2f} 小时")
    print(f"  最佳 Dice: {best_metric:.4f} @ epoch {best_metric_epoch}")
    print("=" * 70)

    logger.print_info(f"训练完成，最佳 Dice={best_metric:.4f} @ epoch {best_metric_epoch}")


def main():
    parser = argparse.ArgumentParser(description="SegResMamba-Lite Training")
    parser.add_argument("--model", type=str, default="segresmamba_lite", help="模型名称")
    parser.add_argument("--version", type=str, default="v2",
                        choices=["v1", "v2", "v3", "v4", "v6", "v7", "v8"],
                        help="模型版本")
    parser.add_argument("--resolution", type=float, default=2.0, help="数据分辨率")
    parser.add_argument("--init_filters", type=int, default=None, help="初始滤波器数量")
    parser.add_argument("--use_attention", action="store_true", default=True, help="使用注意力")
    parser.add_argument("--no_attention", action="store_false", dest="use_attention", help="禁用注意力")
    parser.add_argument("--d_state", type=int, default=8, help="Mamba 状态维度")
    parser.add_argument("--expand", type=int, default=2, help="Mamba 扩展因子")
    parser.add_argument("--num_experts", type=int, default=4, help="MoA 专家数量 (V6/V7 专用)")
    parser.add_argument("--use_deep_supervision", action="store_true", default=False, help="启用深层监督")
    parser.add_argument("--no_deep_supervision", action="store_false", dest="use_deep_supervision", help="禁用深层监督")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--val_interval", type=int, default=5, help="验证间隔")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小")
    parser.add_argument("--batch", type=int, default=None, dest="batch_size",
                        help="批次大小（--batch_size 的简写）")
    parser.add_argument("--workers", type=int, default=None, help="数据加载线程数")
    parser.add_argument("--cache", action="store_true", help="启用硬盘缓存")
    parser.add_argument("--no_ema", action="store_false", dest="use_ema", help="禁用 EMA")
    parser.add_argument("--use_ema", action="store_true", default=True, help="启用 EMA")
    parser.add_argument("--ema", action="store_true", dest="use_ema",
                        help="启用 EMA（--use_ema 的简写）")
    parser.add_argument("--no_ema_flag", action="store_true", dest="no_ema",
                        help="禁用 EMA（--no_ema 的简写）")
    parser.add_argument("--compile", action="store_true", help="启用 torch.compile")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    # 损失函数
    parser.add_argument("--loss_name", type=str, default="DiceFocalLoss",
                        choices=["DiceCELoss", "DiceFocalLoss", "DiceLoss", "V6LossSimple"],
                        help="损失函数")
    parser.add_argument("--v6_alpha", type=float, default=0.5, help="V6Loss: Dice 权重")
    parser.add_argument("--v6_beta", type=float, default=0.3, help="V6Loss: Focal 权重")
    parser.add_argument("--v6_gamma", type=float, default=0.2, help="V6Loss: Boundary 权重")
    args = parser.parse_args()

    train(
        model_name=args.model,
        version_name=args.version,
        version=args.version,
        resolution=args.resolution,
        init_filters=args.init_filters,
        use_attention=args.use_attention,
        d_state=args.d_state,
        expand=args.expand,
        use_deep_supervision=args.use_deep_supervision,
        num_experts=args.num_experts,
        num_workers=args.workers,
        batch_size=args.batch_size,
        use_disk_cache=args.cache,
        use_ema=args.use_ema,
        max_epochs=args.epochs,
        val_interval=args.val_interval,
        device=args.device,
        use_compile=args.compile,
        # 损失
        loss_name=args.loss_name,
        v6_alpha=args.v6_alpha,
        v6_beta=args.v6_beta,
        v6_gamma=args.v6_gamma,
    )


if __name__ == "__main__":
    main()
