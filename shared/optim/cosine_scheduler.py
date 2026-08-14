"""
CosineAnnealingLR 学习率调度器
余弦退火学习率调度

参数说明：
- optimizer: 优化器实例
- T_max: 最大迭代次数
- eta_min: 最小学习率
"""

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR as TorchCosineAnnealingLR


class CosineAnnealingLR:
    """
    余弦退火学习率调度器

    学习率按照余弦曲线从初始值下降到最小值

    Args:
        optimizer: 优化器实例
        T_max: 最大迭代次数 (默认 100)
        eta_min: 最小学习率 (默认 1e-6)
    """

    def __init__(self, optimizer, T_max=100, eta_min=1e-6):
        self.scheduler = TorchCosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
        )

    def step(self):
        """执行一步调度"""
        self.scheduler.step()

    def state_dict(self):
        """返回调度器状态字典"""
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        """加载调度器状态字典"""
        self.scheduler.load_state_dict(state_dict)


__all__ = ["CosineAnnealingLR"]