"""
AdamW 优化器
基于 PyTorch AdamW 的封装

参数说明：
- model_params: 模型参数
- lr: 学习率
- weight_decay: 权重衰减系数
- betas: Adam 的 beta 参数 (beta1, beta2)
- eps: epsilon 数值稳定性参数
"""

import torch
from torch.optim import AdamW as TorchAdamW


class AdamWOptimizer:
    """
    AdamW 优化器封装

    Args:
        model_params: 模型参数
        lr: 学习率 (默认 1e-4)
        weight_decay: 权重衰减系数 (默认 1e-5)
        betas: Adam 的 beta 参数 (默认 (0.9, 0.999))
        eps: epsilon 数值稳定性参数 (默认 1e-8)
    """

    def __init__(
        self,
        model_params,
        lr=1e-4,                # 学习率
        weight_decay=1e-5,      # 权重衰减系数
        betas=(0.9, 0.999),    # Adam 的 beta 参数
        eps=1e-8,              # epsilon 数值稳定性参数
    ):
        self.optimizer = TorchAdamW(
            params=model_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )

    def step(self):
        """执行一步优化"""
        self.optimizer.step()

    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()

    def state_dict(self):
        """返回优化器状态字典"""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """加载优化器状态字典"""
        self.optimizer.load_state_dict(state_dict)

    def get_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]["lr"]

    def param_groups(self):
        """获取参数组"""
        return self.optimizer.param_groups


__all__ = ["AdamWOptimizer"]