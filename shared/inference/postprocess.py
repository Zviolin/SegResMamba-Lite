"""
后处理函数
包括 argmax、标签映射等

参数说明：
- outputs: 模型输出
- num_classes: 类别数
- labels: 标签张量
- label_4_to_3: 是否将标签4映射为3
- etich_label: 原始ET标签 (默认3)
- target_label: 目标标签 (默认4)
"""

import torch
from monai.transforms import AsDiscrete
from monai.data import decollate_batch


def argmax_postprocess(outputs, num_classes=4):
    """
    Argmax 后处理

    对模型输出应用 argmax 操作，转换为类别预测

    Args:
        outputs: 模型输出张量 (batch, num_classes, ...)
        num_classes: 类别数 (默认 4)

    Returns:
        离散化后的预测标签列表
    """
    return [AsDiscrete(argmax=True)(i) for i in decollate_batch(outputs)]


def brats_label_mapping(labels, label_4_to_3=True):
    """
    BraTS 标签映射

    用于训练时统一标签格式（将 ET 标签 4 映射为 3）

    Args:
        labels: 标签张量
        label_4_to_3: 是否将标签 4 映射为 3 (用于训练时统一标签)

    Returns:
        映射后的标签
    """
    if label_4_to_3:
        labels[labels == 4] = 3
    return labels


def brats_post_process_with_labels(outputs, etich_label=3, target_label=4):
    """
    BraTS 后处理：将 ET 标签映射为目标标签

    用于推理后将统一格式的 ET 标签 (3) 恢复为 BraTS 格式 (4)

    Args:
        outputs: 预测输出张量
        etich_label: 原始 ET 标签 (默认 3)
        target_label: 目标标签 (默认 4)

    Returns:
        处理后的输出
    """
    outputs[outputs == etich_label] = target_label
    return outputs


__all__ = [
    "argmax_postprocess",
    "brats_label_mapping",
    "brats_post_process_with_labels",
]