"""
仅 Dice Loss
Dice系数损失函数，用于分割任务

参数说明：
- include_background: 是否包含背景类计算
- to_onehot_y: 是否将标签转换为 one-hot 格式
- softmax: 是否在输出应用 softmax
- squared_pred: 是否使用平方项计算 Dice
- smooth_nr: 分子平滑项
- smooth_dr: 分母平滑项
"""

from monai.losses import DiceLoss as MonaiDiceLoss


class DiceLoss:
    """
    Dice Loss

    用于医学图像分割的Dice系数损失函数

    参数说明：
    - include_background: 是否包含背景类计算（True/False）
    - to_onehot_y: 是否将标签转换为 one-hot 格式（True/False）
    - softmax: 是否在输出应用 softmax（True/False）
    - squared_pred: 是否使用平方项计算 Dice（True/False）
    - smooth_nr: 分子平滑项（防止除零）
    - smooth_dr: 分母平滑项（防止除零）
    """

    def __init__(
        self,
        include_background=True,    # 是否包含背景类计算
        to_onehot_y=True,           # 是否将标签转换为 one-hot 格式
        softmax=True,                # 是否在输出应用 softmax
        squared_pred=True,          # 是否使用平方项计算 Dice
        smooth_nr=0,                # 分子平滑项
        smooth_dr=1e-6,            # 分母平滑项
    ):
        super().__init__()
        self.loss_fn = MonaiDiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            squared_pred=squared_pred,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )

    def __call__(self, outputs, targets):
        """
        计算 Dice Loss

        Args:
            outputs: 模型输出 (batch, num_classes, ...)
            targets: 标签 (batch, 1, ...) 或 (batch, ...)

        Returns:
            Dice损失值
        """
        return self.loss_fn(outputs, targets)


__all__ = ["DiceLoss"]