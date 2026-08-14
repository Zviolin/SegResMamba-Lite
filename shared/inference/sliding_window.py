"""
滑动窗口推理
用于大图像的分割推理

参数说明：
- model: 分割模型
- inputs: 输入图像 (B, C, H, W, D)
- roi_size: 滑动窗口大小
- sw_batch_size: 每批窗口数
- overlap: 重叠比例
- mode: 融合模式 ("gaussian" 或 "constant")
"""

import torch
from monai.inferers import sliding_window_inference as _sliding_window_inference


def sliding_window_inference(
    model,
    inputs,
    roi_size=(64, 64, 64),
    sw_batch_size=4,
    overlap=0.5,
    mode="gaussian",
):
    """
    滑动窗口推理

    用于大图像的分割推理，通过滑动窗口方式处理大体积医学图像

    Args:
        model: 分割模型
        inputs: 输入图像 (B, C, H, W, D)
        roi_size: 滑动窗口大小 (默认 (64, 64, 64))
        sw_batch_size: 每批窗口数 (默认 4)
        overlap: 重叠比例 (默认 0.5)
        mode: 融合模式 ("gaussian" 或 "constant", 默认 "gaussian")

    Returns:
        模型输出
    """
    def _forward(x):
        outputs = model(x)
        if isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs

    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        outputs = _sliding_window_inference(
            inputs,
            roi_size,
            sw_batch_size,
            _forward,
            overlap=overlap,
            mode=mode,
        )
    return outputs


__all__ = ["sliding_window_inference"]