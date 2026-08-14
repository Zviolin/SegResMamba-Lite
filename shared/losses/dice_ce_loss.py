"""
Dice + CrossEntropy 组合损失函数
"""

from monai.losses import DiceCELoss as MonaiDiceCELoss


class DiceCELoss:
    """
    Dice + CrossEntropy 组合损失函数

    结合 Dice Loss 和 CrossEntropy Loss，平衡前景和背景的分割效果

    参数说明：
    - include_background: 是否包含背景类计算（True/False）
    - to_onehot_y: 是否将标签转换为 one-hot 格式（True/False）
    - softmax: 是否在输出应用 softmax（True/False）
    - squared_pred: 是否使用平方项计算 Dice（True/False）
    - smooth_nr: 分子平滑项（防止除零）
    - smooth_dr: 分母平滑项（防止除零）
    - lambda_dice: Dice Loss 权重系数
    - lambda_ce: CrossEntropy Loss 权重系数
    """

    def __init__(
        self,
        include_background=True,    # 是否包含背景类计算
        to_onehot_y=True,           # 是否将标签转换为 one-hot 格式
        softmax=True,                # 是否在输出应用 softmax
        squared_pred=True,          # 是否使用平方项计算 Dice
        smooth_nr=0,                # 分子平滑项
        smooth_dr=1e-6,            # 分母平滑项
        lambda_dice=1.0,            # Dice Loss 权重系数
        lambda_ce=1.0,              # CrossEntropy Loss 权重系数
    ):
        super().__init__()
        self.loss_fn = MonaiDiceCELoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            squared_pred=squared_pred,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def __call__(self, outputs, targets):
        """
        计算组合损失

        Args:
            outputs: 模型输出 (batch, num_classes, ...)
            targets: 标签 (batch, 1, ...) 或 (batch, ...)

        Returns:
            组合损失值
        """
        return self.loss_fn(outputs, targets)


__all__ = ["DiceCELoss"]