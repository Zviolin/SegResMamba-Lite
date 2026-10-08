"""
PolyLR 多项式学习率衰减调度器
官方 SegMamba 复现专用（SegMamba-Official 的 train.py 历史口径即 Poly，
见 2026-05-13 训练日志：scheduler = Poly (official)）

学习率按 lr = base_lr * (1 - step / T_max) ** power 衰减，逐 epoch 调用 step。
"""

import torch


class PolyLR:
    """
    多项式衰减学习率调度器（包装 torch.optim.lr_scheduler.PolynomialLR）

    Args:
        optimizer: 优化器实例
        T_max: 总 epoch 数（映射到 PolynomialLR 的 total_iters）
        power: 多项式幂次（官方 SegMamba 惯例 0.9）
    """

    def __init__(self, optimizer, T_max=100, power=0.9):
        self.scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=T_max,
            power=power,
        )

    def step(self):
        """执行一步调度（每 epoch 调用一次）"""
        self.scheduler.step()

    def state_dict(self):
        """返回调度器状态字典"""
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        """加载调度器状态字典"""
        self.scheduler.load_state_dict(state_dict)


__all__ = ["PolyLR"]
