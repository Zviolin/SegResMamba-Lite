"""
损失函数工厂
统一获取损失函数

参数说明：
- loss_name: 损失函数名称 ("DiceCELoss" / "DiceFocalLoss" / "DiceLoss" / "V6Loss" / "V6LossSimple")
- kwargs: 传递给损失函数的其他参数
"""

from .dice_ce_loss import DiceCELoss
from .dice_focal_loss import DiceFocalLoss
from .dice_loss import DiceLoss
from .v6_loss import V6Loss, V6LossSimple


LOSS_REGISTRY = {
    "DiceCELoss": DiceCELoss,
    "DiceFocalLoss": DiceFocalLoss,
    "DiceLoss": DiceLoss,
    "V6Loss": V6Loss,
    "V6LossSimple": V6LossSimple,
}


def get_loss(loss_name="DiceCELoss", **kwargs):
    """
    获取损失函数实例

    Args:
        loss_name: 损失函数名称
        **kwargs: 传递给损失函数的其他参数

    Returns:
        Loss实例

    Raises:
        ValueError: 当损失函数名称未知时
    """
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"未知损失函数: {loss_name}")
    return LOSS_REGISTRY[loss_name](**kwargs)


__all__ = [
    "DiceCELoss",
    "DiceFocalLoss",
    "DiceLoss",
    "V6Loss",
    "V6LossSimple",
    "get_loss",
    "LOSS_REGISTRY",
]
