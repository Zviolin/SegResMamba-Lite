"""
Dice 系数指标
"""

import torch
from monai.metrics import DiceMetric


class DiceMetricWrapper:
    """Dice 指标封装"""

    def __init__(self, include_background=False, reduction="mean"):
        self.metric = DiceMetric(
            include_background=include_background,
            reduction=reduction,
        )

    def __call__(self, y_pred, y):
        return self.metric(y_pred=y_pred, y=y)

    def aggregate(self):
        return self.metric.aggregate()

    def reset(self):
        self.metric.reset()


class BraTSDiceMetric:
    """BraTS 三个区域的 Dice 指标"""

    def __init__(self):
        self.wt_metric = DiceMetric(include_background=False, reduction="mean")
        self.tc_metric = DiceMetric(include_background=False, reduction="mean")
        self.et_metric = DiceMetric(include_background=False, reduction="mean")

    def __call__(self, outputs, labels):
        """
        计算 Dice

        Args:
            outputs: 模型输出 (batch, 4, H, W, D) - 4个类别
            labels: 标签 (batch, 1, H, W, D)
        """
        pred_wt = [(o[1] + o[2] + o[3] > 0).float().unsqueeze(0) for o in outputs]
        true_wt = [(l.sum(dim=0) > 0).float().unsqueeze(0) for l in labels]

        pred_tc = [(o[1] + o[3] > 0).float().unsqueeze(0) for o in outputs]
        true_tc = [((l == 1) | (l == 3)).float().unsqueeze(0) for l in labels]

        pred_et = [(o[3] > 0).float().unsqueeze(0) for o in outputs]
        true_et = [(l == 3).float().unsqueeze(0) for l in labels]

        self.wt_metric(y_pred=pred_wt, y=true_wt)
        self.tc_metric(y_pred=pred_tc, y=true_tc)
        self.et_metric(y_pred=pred_et, y=true_et)

    def aggregate(self):
        wt = self.wt_metric.aggregate().item()
        tc = self.tc_metric.aggregate().item()
        et = self.et_metric.aggregate().item()
        return {"wt": wt, "tc": tc, "et": et, "avg": (wt + tc + et) / 3}

    def reset(self):
        self.wt_metric.reset()
        self.tc_metric.reset()
        self.et_metric.reset()


__all__ = ["DiceMetricWrapper", "BraTSDiceMetric"]
