"""
模型检查点工具
保存和加载模型权重

参数说明：
- save_dir: 检查点保存目录
- prefix: 文件名前缀
- mode: 保存模式 ("max" 保留最大值 / "min" 保留最小值)
- model: PyTorch模型
- optimizer: 优化器
- scheduler: 学习率调度器
- epoch: 当前训练轮数
- metric: 当前指标值
- swa_model: SWA 平均权重模型（历史命名 ema_model，实为等权 SWA，已改名）
- checkpoint_path: 检查点文件路径
- device: 设备类型
"""

import os
import torch


class ModelCheckpoint:
    """
    模型检查点保存管理器

    支持自动保存最佳模型，只保留最优的检查点

    Args:
        save_dir: 检查点保存目录
        prefix: 文件名前缀 (默认 "model")
        mode: 保存模式 ("max" 保留最大值 / "min" 保留最小值, 默认 "max")
    """

    def __init__(self, save_dir, prefix="model", mode="max"):
        self.save_dir = save_dir
        self.prefix = prefix
        self.mode = mode
        self.best_value = -1 if mode == "max" else float("inf")

    def save(self, model, optimizer=None, scheduler=None, epoch=None, metric=None, swa_model=None):
        """
        保存检查点

        Args:
            model: 模型
            optimizer: 优化器 (可选)
            scheduler: 学习率调度器 (可选)
            epoch: 当前 epoch (可选)
            metric: 当前指标值 (可选，用于判断是否保存最佳模型)
            swa_model: SWA 平均权重模型 (可选；历史参数名 ema_model 实为等权 SWA，已改名)
        """
        os.makedirs(self.save_dir, exist_ok=True)
        is_best = False

        if metric is not None:
            if self.mode == "max":
                if metric > self.best_value:
                    self.best_value = metric
                    is_best = True
            else:
                if metric < self.best_value:
                    self.best_value = metric
                    is_best = True

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        }
        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if metric is not None:
            checkpoint["metric"] = metric

        torch.save(checkpoint, os.path.join(self.save_dir, f"latest_checkpoint_{self.prefix}.pth"))

        if is_best:
            torch.save(model.state_dict(), os.path.join(self.save_dir, f"best_metric_model_{self.prefix}.pth"))
            if swa_model is not None:
                # 平均权重文件名：历史为 best_metric_ema_model_{prefix}.pth（历史命名实为 SWA），
                # 新保存统一用 best_metric_swa_model_{prefix}.pth，评估端对两种命名均兼容
                torch.save(swa_model.state_dict(), os.path.join(self.save_dir, f"best_metric_swa_model_{self.prefix}.pth"))
            print(f"已保存新的最佳模型！指标: {metric:.4f}")

    def save_latest(self, model, optimizer=None, scheduler=None, epoch=None, metric=None):
        """
        仅保存最新检查点

        Args:
            model: 模型
            optimizer: 优化器 (可选)
            scheduler: 学习率调度器 (可选)
            epoch: 当前 epoch (可选)
            metric: 当前指标值 (可选)
        """
        os.makedirs(self.save_dir, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        }
        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if metric is not None:
            checkpoint["metric"] = metric

        torch.save(checkpoint, os.path.join(self.save_dir, f"latest_checkpoint_{self.prefix}.pth"))


def _warn_on_mismatch(result, strict):
    """strict=False 加载时打印 missing/unexpected 键摘要，避免静默部分加载

    静默的部分加载是评估链最危险的失败模式（如消融参数漏传时，
    结构多出的随机初始化权重会被无声使用）。此护栏让任何不匹配显式可见。
    """
    if strict or result is None:
        return
    n_missing, n_unexpected = len(result.missing_keys), len(result.unexpected_keys)
    if n_missing or n_unexpected:
        print(f"[checkpoint 警告] state_dict 不完全匹配: missing={n_missing}, unexpected={n_unexpected}")
        if n_missing:
            print(f"  missing 示例: {list(result.missing_keys)[:5]}")
        if n_unexpected:
            print(f"  unexpected 示例: {list(result.unexpected_keys)[:5]}")


def load_checkpoint(model, checkpoint_path, device="cuda", strict=True):
    """
    加载检查点

    Args:
        model: 模型
        checkpoint_path: 检查点路径
        device: 设备 (默认 "cuda")
        strict: 是否严格匹配 state_dict (默认 True)

    Returns:
        model: 加载后的模型
        checkpoint: 检查点字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        _warn_on_mismatch(model.load_state_dict(checkpoint["model_state_dict"], strict=strict), strict)
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # EMA 模型格式
        state_dict = checkpoint["state_dict"]
        # 移除 EMA 特有的键
        if "n_averaged" in state_dict:
            state_dict.pop("n_averaged", None)
        # 移除 module. 前缀（如果是 DataParallel 保存的）
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        _warn_on_mismatch(model.load_state_dict(new_state_dict, strict=strict), strict)
    else:
        # 直接是 state_dict 的情况
        if isinstance(checkpoint, dict):
            # 移除 EMA 特有的键
            state_dict = checkpoint.copy()
            state_dict.pop("n_averaged", None)
            # 移除 module. 前缀
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v
            _warn_on_mismatch(model.load_state_dict(new_state_dict, strict=strict), strict)
        else:
            _warn_on_mismatch(model.load_state_dict(checkpoint, strict=strict), strict)

    return model, checkpoint