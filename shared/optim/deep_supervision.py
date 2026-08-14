"""
Deep Supervision 深层监督
在多个尺度上计算损失，帮助训练深层网络

参数说明：
- weights: 各尺度损失的权重列表
- outputs: 模型输出列表 [out1, out2, out3, ...]
- loss_fn: 损失函数
- target: 目标标签
"""

import torch


class DeepSupervision:
    """
    深层监督辅助器

    在多个尺度（深层特征）上计算损失，然后加权求和，
    帮助梯度更好地流向深层网络层

    Args:
        weights: 各尺度损失的权重列表 (默认 [0.5, 0.3, 0.2])
    """

    def __init__(self, weights=[0.5, 0.3, 0.2]):
        self.weights = weights

    def compute_loss(self, outputs, loss_fn, target):
        """
        计算深层监督损失

        Args:
            outputs: 模型输出列表 [out1, out2, out3, ...]
            loss_fn: 损失函数
            target: 目标标签

        Returns:
            total_loss: 加权损失和
        """
        if not isinstance(outputs, (list, tuple)):
            return loss_fn(outputs, target)

        total_loss = 0.0
        for i, out in enumerate(outputs):
            weight = self.weights[i] if i < len(self.weights) else self.weights[-1]
            total_loss = total_loss + weight * loss_fn(out, target)

        return total_loss


__all__ = ["DeepSupervision"]