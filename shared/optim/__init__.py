"""
优化器工厂
统一获取优化器和学习率调度器

参数说明：
- model_params: 模型参数
- optimizer_type: 优化器类型 ("AdamW" / "Adam")
- scheduler_type: 调度器类型 ("CosineAnnealingLR")
- kwargs: 其他参数
"""

import torch
from torch.optim import AdamW, Adam

from .adamw import AdamWOptimizer
from .cosine_scheduler import CosineAnnealingLR
from .ema import EMA
from .deep_supervision import DeepSupervision


OPTIM_REGISTRY = {
    "AdamW": AdamW,
    "Adam": Adam,
}


SCHEDULER_REGISTRY = {
    "CosineAnnealingLR": CosineAnnealingLR,
}


def get_optimizer(model_params, optimizer_type="AdamW", **kwargs):
    """
    获取优化器

    Args:
        model_params: 模型参数
        optimizer_type: 优化器类型 ("AdamW" / "Adam")
        **kwargs: 其他参数传递给优化器（如 lr, weight_decay 等）

    Returns:
        Optimizer实例

    Raises:
        ValueError: 当优化器类型未知时
    """
    if optimizer_type not in OPTIM_REGISTRY:
        raise ValueError(f"未知优化器类型: {optimizer_type}")
    return OPTIM_REGISTRY[optimizer_type](params=model_params, **kwargs)


def get_scheduler(optimizer, scheduler_type="CosineAnnealingLR", **kwargs):
    """
    获取学习率调度器

    Args:
        optimizer: 优化器实例
        scheduler_type: 调度器类型 ("CosineAnnealingLR")
        **kwargs: 其他参数传递给调度器（如 T_max, eta_min 等）

    Returns:
        Scheduler实例

    Raises:
        ValueError: 当调度器类型未知时
    """
    if scheduler_type not in SCHEDULER_REGISTRY:
        raise ValueError(f"未知调度器类型: {scheduler_type}")
    return SCHEDULER_REGISTRY[scheduler_type](optimizer, **kwargs)


__all__ = [
    "get_optimizer",
    "get_scheduler",
    "AdamWOptimizer",
    "CosineAnnealingLR",
    "EMA",
    "DeepSupervision",
]