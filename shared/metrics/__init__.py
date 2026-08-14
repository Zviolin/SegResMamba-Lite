"""
评估指标工厂
统一获取评估指标

参数说明：
- include_background: 是否包含背景类
- reduction: 聚合方式 ("mean" / "none" / "mean_batch")
- spacing: 体素间距
- lesion_results_file: 病灶评估结果文件路径
"""

from .dice_metric import DiceMetricWrapper, BraTSDiceMetric
from .hd95_metric import HD95MetricWrapper, BraSTHD95Metric
from .lesion_metric import lesionwise_evaluation, label_components, compute_hd95
from .aggregate_lesion import aggregate_lesion_results


__all__ = [
    "DiceMetricWrapper",
    "BraTSDiceMetric",
    "HD95MetricWrapper",
    "BraSTHD95Metric",
    "lesionwise_evaluation",
    "label_components",
    "compute_hd95",
    "aggregate_lesion_results",
]