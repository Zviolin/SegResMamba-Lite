"""
推理工具
滑动窗口推理和后处理函数

参数说明：
- model: 分割模型
- inputs: 输入图像
- outputs: 模型输出
- num_classes: 类别数
- spacing: 体素间距
"""

from .sliding_window import sliding_window_inference
from .postprocess import (
    argmax_postprocess,
    brats_label_mapping,
    brats_post_process_with_labels,
)


__all__ = [
    "sliding_window_inference",
    "argmax_postprocess",
    "brats_label_mapping",
    "brats_post_process_with_labels",
]