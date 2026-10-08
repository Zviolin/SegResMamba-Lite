"""
CrossEntropyLoss 多类交叉熵损失
官方 SegMamba 复现专用（SegMamba-Official 的 train.py 历史口径即 CE，
见 2026-05-13 训练日志：loss_name = CrossEntropyLoss）

与 shared 内其他损失的接口对齐：接受 one-hot 标签，内部自动转类别索引。
"""

import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    """
    多类交叉熵损失（one-hot 标签自动转索引）

    Args:
        **kwargs: 透传给 torch.nn.CrossEntropyLoss（如 weight、label_smoothing）
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(**kwargs)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: logits (B, C, D, H, W)
            label: one-hot 标签 (B, C, D, H, W) 或类别索引 (B, D, H, W)

        Returns:
            标量损失
        """
        # one-hot (B, C, ...) → 类别索引 (B, ...)
        if label.ndim == pred.ndim:
            label = torch.argmax(label, dim=1)
        return self.ce(pred, label)


__all__ = ["CrossEntropyLoss"]
