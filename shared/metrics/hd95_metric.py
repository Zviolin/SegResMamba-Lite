"""
HD95 距离指标
Hausdorff Distance (95th percentile) 用于评估分割边界距离

参数说明：
- include_background: 是否包含背景类
- percentile: 百分位数 (默认95)
- reduction: 聚合方式 ("mean" / "none" / "mean_batch")
- spacing: 体素间距
"""

import torch
import numpy as np
from monai.metrics import HausdorffDistanceMetric


class HD95MetricWrapper:
    """
    HD95 指标封装

    Args:
        include_background: 是否包含背景类
        percentile: 百分位数 (默认95)
        reduction: 聚合方式 ("mean" / "none" / "mean_batch")
    """

    def __init__(self, include_background=False, percentile=95, reduction="mean"):
        self.metric = HausdorffDistanceMetric(
            include_background=include_background,
            percentile=percentile,
            reduction=reduction,
        )

    def __call__(self, y_pred, y, spacing):
        """
        计算 HD95 指标

        Args:
            y_pred: 预测结果
            y: 真实标签
            spacing: 体素间距

        Returns:
            HD95 值
        """
        return self.metric(y_pred=y_pred, y=y, spacing=spacing)

    def aggregate(self):
        """
        聚合计算结果

        Returns:
            聚合后的 HD95 值（处理了 NaN/Inf）
        """
        value = self.metric.aggregate().item()
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return value

    def reset(self):
        """重置指标状态"""
        self.metric.reset()


class BraSTHD95Metric:
    """
    BraTS 三个区域的 HD95 指标

    计算 WT (Whole Tumor)、TC (Tumor Core)、ET (Enhancing Tumor) 三个区域的 HD95

    Args:
        percentile: 百分位数 (默认95)
    """

    def __init__(self, percentile=95):
        self.percentile = percentile
        self.wt_metric = HausdorffDistanceMetric(
            include_background=False, percentile=percentile, reduction="mean"
        )
        self.tc_metric = HausdorffDistanceMetric(
            include_background=False, percentile=percentile, reduction="mean"
        )
        self.et_metric = HausdorffDistanceMetric(
            include_background=False, percentile=percentile, reduction="mean"
        )

    def __call__(self, outputs, labels, spacing):
        """
        计算 BraTS 三个区域的 HD95

        Args:
            outputs: 模型输出列表 [(batch, 4, H, W, D), ...]
            labels: 标签列表 [(batch, 1, H, W, D), ...]
            spacing: 体素间距
        """
        pred_wt = [(o[1] + o[2] + o[3] > 0).float().unsqueeze(0) for o in outputs]
        true_wt = [(l.sum(dim=0) > 0).float().unsqueeze(0) for l in labels]

        pred_tc = [(o[1] + o[3] > 0).float().unsqueeze(0) for o in outputs]
        true_tc = [((l == 1) | (l == 3)).float().unsqueeze(0) for l in labels]

        pred_et = [(o[3] > 0).float().unsqueeze(0) for o in outputs]
        true_et = [(l == 3).float().unsqueeze(0) for l in labels]

        self.wt_metric(y_pred=pred_wt, y=true_wt, spacing=spacing)
        self.tc_metric(y_pred=pred_tc, y=true_tc, spacing=spacing)
        self.et_metric(y_pred=pred_et, y=true_et, spacing=spacing)

    def aggregate(self):
        """
        聚合计算结果

        Returns:
            dict: 包含 wt, tc, et, avg 四个区域的 HD95 值
        """
        def safe(metric):
            if metric is None:
                return 0.0
            val = metric.aggregate().item()
            return 0.0 if np.isnan(val) or np.isinf(val) else val

        wt = safe(self.wt_metric)
        tc = safe(self.tc_metric)
        et = safe(self.et_metric)
        return {"wt": wt, "tc": tc, "et": et, "avg": (wt + tc + et) / 3}

    def reset(self):
        """重置三个区域的指标状态"""
        self.wt_metric.reset()
        self.tc_metric.reset()
        self.et_metric.reset()


__all__ = ["HD95MetricWrapper", "BraSTHD95Metric"]