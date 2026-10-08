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
import random
import numpy as np
import torch
import argparse
import multiprocessing

_version_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_code_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _version_root)
sys.path.insert(0, _code_root)

from shared.data.dataloader import get_train_dataloader, get_val_dataloader, auto_roi_size_from_cache
from shared.utils.logger import setup_logger
from shared.utils.device import setup_cuda_optimization, setup_determinism
from shared.utils.device_info import print_device_info
from shared.utils.paths import resolve_log_prefix, resolve_model_dir, resolve_cache_dir

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
    top_k=2,                        # MoA Top-K 激活专家数 (V10/V12)
    expert_type='attention',        # V10 专家类型 (attention/mamba/dconv)
    route_granularity='token',      # V10 路由粒度 (token/sample)
    share_kv=True,                  # V10 是否共享 K/V 投影
    route_noise=0.0,                # V10 训练时路由噪声 σ (F1 用 0.05)
    decoder_moa=False,              # V10 G1：16³ 解码器侧追加 MoA
    seed=42,                        # 训练随机种子（不影响数据划分缓存）
    run_tag="",                     # 消融实验标签：隔离检查点/日志目录，防止误恢复锚点权重
    model_dir="",                   # 自定义权重保存目录（空=默认 pipeline/models/{分辨率}mm_{版本}_{设备}[_{run_tag}]）
    log_name="",                    # 自定义日志名称前缀（空=默认 train_{版本}_{分辨率}mm[_{run_tag}]）
    # ═══════════════════════════════════════════════════════════════════════
    # 训练参数
    # ═══════════════════════════════════════════════════════════════════════
    num_workers=None,
    batch_size=None,
    use_disk_cache=False,
    use_swa=True,
    weight_mode="swa",              # 权重平均模式：swa=等权平均（历史口径）/ ema=真指数移动平均（V7/V8 与 EMA1 对照消融复现）
    skip_nonfinite=False,           # 跳过非有限 loss（NaN/Inf）的 batch，防污染 optimizer/平均权重（默认关闭，保持历史行为）
    best_weights="raw",             # best_metric_model.pth 保存来源：raw=原始模型权重（历史口径）/ avg=验证所用的平均权重（SWA/EMA）
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
    lb_weight=0.01,               # V9 MoE 负载均衡损失权重（0 表示关闭）
    # ═══════════════════════════════════════════════════════════════════════
    # 设备参数
    # ═══════════════════════════════════════════════════════════════════════
    device=None,
    use_compile=False,
    deterministic=False,            # 可复现性：开启确定性训练（cudnn.deterministic + 确定性算法，不触碰 TF32）
    cache_parent="",                # 数据缓存父目录覆盖（优先级：命令行 > 环境变量 SRTP_CACHE_PARENT > 各项目默认）
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
    # 可复现性：cuBLAS 确定性工作区配置必须在任何 CUDA 上下文创建之前设置
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # ─────────────────────────────────────────────────────────────────────
    # 日志配置
    # ─────────────────────────────────────────────────────────────────────
    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    # 日志名称：命令行 --log_name 优先，否则默认 train_{版本}_{分辨率}mm[_{run_tag}]
    _log_base = resolve_log_prefix(log_name, f"train_{version_name}_{resolution}mm" + (f"_{run_tag}" if run_tag else ""))
    logger = setup_logger(LOG_DIR, _log_base)

    print_device_info(logger)
    setup_cuda_optimization()

    DEVICE = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = batch_size
    LR = 1e-4

    # 固定训练随机性（不影响数据划分：划分由缓存 split_info.json 固定 seed=42）
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"训练随机种子: seed={seed}（数据划分仍由缓存固定）")
    if deterministic:
        # 可复现性：关 benchmark + 锁定确定性算法（不触碰 TF32，保持论文口径可比）
        setup_determinism(seed)

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
            "num_experts": num_experts,
            "top_k": top_k,
            "expert_type": expert_type if version in ('v10', 'v12') else "-",
            "route_granularity": route_granularity if version in ('v10', 'v12') else "-",
            "share_kv": share_kv if version in ('v10', 'v12') else "-",
            "route_noise": route_noise if version in ('v10', 'v12') else "-",
            "decoder_moa": decoder_moa if version in ('v10', 'v12') else "-",
            "device": DEVICE,
        },
        "训练设置": {
            "batch_size": BATCH_SIZE,
            "num_workers": num_workers,
            "max_epochs": max_epochs,
            "use_swa": use_swa,
            "weight_mode": weight_mode if use_swa else "-",
            "use_deep_supervision": use_deep_supervision,
            "use_disk_cache": use_disk_cache,
            "val_interval": val_interval,
            "seed": seed,
        },
        "优化器设置": {
            "loss_name": loss_name,
            "optimizer": "AdamW",
            "lr": LR,
            "weight_decay": 1e-5,
            "scheduler": "CosineAnnealingLR",
        },
    }, title=f"训练配置 - SegResMamba-Lite {version}")

    current_cache_dir = resolve_cache_dir(resolution, CACHE_DIR, cli_parent=cache_parent) if use_disk_cache else None
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
    # V10/V12 消融参数（其余版本的 get_model 不接受这些参数，不透传）
    if version in ('v10', 'v12'):
        model_kwargs.update(
            top_k=top_k,
            expert_type=expert_type,
            route_granularity=route_granularity,
            share_kv=share_kv,
            route_noise=route_noise,
            decoder_moa=decoder_moa,
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
        use_swa=use_swa,
        weight_mode=weight_mode,
        use_deep_supervision=use_deep_supervision,
        non_blocking=True,
        version=version,
        lb_weight=lb_weight,
        skip_nonfinite=skip_nonfinite,
        **loss_kwargs,
    )

    if use_compile and hasattr(torch, 'compile'):
        print("正在编译模型 (torch.compile)...")
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model, mode='max-autotune')
        trainer.model = model
        print("模型编译完成")

    # torch.compile 包装后 state_dict 键会带 _orig_mod. 前缀，保存/恢复需取原始模型
    _raw_model = getattr(trainer.model, '_orig_mod', trainer.model)

    # 权重目录：命令行 --model_dir 优先（支持绝对/相对路径），否则默认
    # models/{分辨率}mm_{版本}_{设备}[_{run_tag}]（run_tag 用于消融变体隔离，防止误恢复锚点权重）
    _model_dir_name = f"{resolution}mm_{version}_{DEVICE.replace(':', '')}" + (f"_{run_tag}" if run_tag else "")
    MODEL_DIR = resolve_model_dir(model_dir, os.path.dirname(os.path.abspath(__file__)), _model_dir_name)
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
        _raw_model_restore = getattr(trainer.model, '_orig_mod', trainer.model)
        _raw_model_restore.load_state_dict(checkpoint['model_state_dict'])
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
        # 还原历史最佳指标，防止续训后首次验证以次优权重覆盖 best_metric_model.pth
        best_metric = checkpoint.get('best_metric', -1)
        best_metric_epoch = checkpoint.get('best_metric_epoch', -1)
        if best_metric > -1:
            print(f"已恢复历史最佳: Dice {best_metric:.4f} @ epoch {best_metric_epoch}")
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
                # --best_weights avg：保存验证所用的平均权重（SWA/EMA，跟随 --weight_mode）；
                # raw（默认，历史口径）：保存原始模型权重
                if best_weights == "avg" and trainer.weight_avg is not None:
                    torch.save(
                        trainer._get_eval_model().state_dict(),
                        os.path.join(MODEL_DIR, "best_metric_model.pth")
                    )
                    print(f"  ✅ 新最佳模型已保存 (Dice: {mean_dice:.4f}, 来源: 平均权重)")
                else:
                    torch.save(
                        _raw_model.state_dict(),
                        os.path.join(MODEL_DIR, "best_metric_model.pth")
                    )
                    print(f"  ✅ 新最佳模型已保存 (Dice: {mean_dice:.4f})")

        torch.save({
            'epoch': epoch,
            'model_state_dict': _raw_model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            # 历史最佳一并存档：断点续训时还原，防止 best_metric_model.pth 被覆盖
            'best_metric': best_metric,
            'best_metric_epoch': best_metric_epoch,
        }, latest_checkpoint)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == max_epochs:
            torch.save(
                _raw_model.state_dict(),
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
                        choices=["v1", "v2", "v3", "v4", "v6", "v7", "v8", "v9", "v10", "v11", "v12"],
                        help="模型版本")
    parser.add_argument("--resolution", type=float, default=2.0, help="数据分辨率")
    parser.add_argument("--init_filters", type=int, default=None, help="初始滤波器数量")
    parser.add_argument("--use_attention", action="store_true", default=True, help="使用注意力")
    parser.add_argument("--no_attention", action="store_false", dest="use_attention", help="禁用注意力")
    parser.add_argument("--d_state", type=int, default=8, help="Mamba 状态维度")
    parser.add_argument("--expand", type=int, default=2, help="Mamba 扩展因子")
    parser.add_argument("--num_experts", type=int, default=4, help="MoA 专家数量 (V6/V7 专用)")
    parser.add_argument("--top_k", type=int, default=2, help="MoA Top-K 激活专家数 (V10/V12)")
    parser.add_argument("--expert_type", type=str, default="attention",
                        choices=["attention", "mamba", "dconv"], help="V10 专家类型")
    parser.add_argument("--route_granularity", type=str, default="token",
                        choices=["token", "sample"], help="V10 路由粒度")
    parser.add_argument("--share_kv", type=int, default=1, choices=[0, 1],
                        help="V10 是否共享 K/V（1=共享，0=每专家独立）")
    parser.add_argument("--route_noise", type=float, default=0.0,
                        help="V10 训练时路由噪声 σ（F1 用 0.05）")
    parser.add_argument("--decoder_moa", type=int, default=0, choices=[0, 1],
                        help="V10 G1：16³ 解码器侧追加 MoA（1=启用）")
    parser.add_argument("--seed", type=int, default=42, help="训练随机种子（不影响数据划分缓存）")
    parser.add_argument("--use_deep_supervision", action="store_true", default=False, help="启用深层监督")
    parser.add_argument("--no_deep_supervision", action="store_false", dest="use_deep_supervision", help="禁用深层监督")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--val_interval", type=int, default=5, help="验证间隔")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小")
    parser.add_argument("--batch", type=int, default=None, dest="batch_size",
                        help="批次大小（--batch_size 的简写）")
    parser.add_argument("--workers", type=int, default=None, help="数据加载线程数")
    parser.add_argument("--cache", action="store_true", help="启用硬盘缓存")
    parser.add_argument("--no_swa", action="store_false", dest="use_swa", help="禁用权重滑动平均（SWA）")
    parser.add_argument("--use_swa", action="store_true", default=True, help="启用权重滑动平均（SWA 等权）")
    parser.add_argument("--swa", action="store_true", dest="use_swa",
                        help="启用权重滑动平均（--use_swa 的简写）")
    parser.add_argument("--weight_mode", type=str, default="swa", choices=["swa", "ema"],
                        help="权重平均模式：swa=等权平均（历史口径，默认）/ ema=真指数移动平均（V7/V8 与 EMA1 对照消融复现，decay=0.999 生效）。"
                             "历史 CLI --ema/--ema_mode 对应本参数（历史 'ema' 命名实为等权 SWA）")
    parser.add_argument("--compile", action="store_true", help="启用 torch.compile")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    # 损失函数
    parser.add_argument("--loss_name", type=str, default="DiceFocalLoss",
                        choices=["DiceCELoss", "DiceFocalLoss", "DiceLoss", "V6LossSimple"],
                        help="损失函数")
    parser.add_argument("--v6_alpha", type=float, default=0.5, help="V6Loss: Dice 权重")
    parser.add_argument("--v6_beta", type=float, default=0.3, help="V6Loss: Focal 权重")
    parser.add_argument("--v6_gamma", type=float, default=0.2, help="V6Loss: Boundary 权重")
    parser.add_argument("--lb_weight", type=float, default=0.01,
                        help="V9 MoE 负载均衡损失权重（默认 0.01，0 表示关闭）")
    parser.add_argument("--run_tag", type=str, default="",
                        help="消融实验标签：检查点/日志目录追加 _{tag} 后缀，隔离各变体（如 A2/B1/S1）")
    parser.add_argument("--model_dir", type=str, default="",
                        help="自定义权重保存目录（优先级高于 --run_tag；默认 pipeline/models/{分辨率}mm_{版本}_{设备}[_{run_tag}]）")
    parser.add_argument("--log_name", type=str, default="",
                        help="自定义日志名称前缀（默认 train_{版本}_{分辨率}mm[_{run_tag}]）")
    # 跨卡包合并的可选开关（默认关闭，行为与历史完全一致）
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="开启确定性训练：cudnn.deterministic + 确定性算法 + CUBLAS_WORKSPACE_CONFIG（不触碰 TF32）")
    parser.add_argument("--cache_parent", type=str, default="",
                        help="数据缓存父目录覆盖（优先级：命令行 > 环境变量 SRTP_CACHE_PARENT > 各项目默认；最终目录 = {parent}_{分辨率}）")
    parser.add_argument("--skip_nonfinite", action="store_true", default=False,
                        help="跳过非有限 loss（NaN/Inf）的 batch，不 backward/不更新/不进平均权重")
    parser.add_argument("--best_weights", type=str, default="raw", choices=["raw", "avg"],
                        help="best_metric_model.pth 保存来源：raw=原始权重（默认，历史口径）/ avg=验证所用的平均权重（SWA/EMA，跟随 --weight_mode）")
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
        top_k=args.top_k,
        expert_type=args.expert_type,
        route_granularity=args.route_granularity,
        share_kv=bool(args.share_kv),
        route_noise=args.route_noise,
        decoder_moa=bool(args.decoder_moa),
        seed=args.seed,
        run_tag=args.run_tag,
        model_dir=args.model_dir,
        log_name=args.log_name,
        deterministic=args.deterministic,
        cache_parent=args.cache_parent,
        skip_nonfinite=args.skip_nonfinite,
        best_weights=args.best_weights,
        num_workers=args.workers,
        batch_size=args.batch_size,
        use_disk_cache=args.cache,
        use_swa=args.use_swa,
        weight_mode=args.weight_mode,
        max_epochs=args.epochs,
        val_interval=args.val_interval,
        device=args.device,
        use_compile=args.compile,
        # 损失
        loss_name=args.loss_name,
        v6_alpha=args.v6_alpha,
        v6_beta=args.v6_beta,
        v6_gamma=args.v6_gamma,
        lb_weight=args.lb_weight,
    )


if __name__ == "__main__":
    main()
