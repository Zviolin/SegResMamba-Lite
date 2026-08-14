"""
V6 损失函数 - 形态感知 MoA 版
Dice + Focal + Boundary + 路由正则

针对 V6 模型的 Morphology-Aware MoA 架构优化的组合损失：
- Dice Loss：区域重叠度量
- Focal Loss：难例挖掘，处理类别不平衡
- Boundary Loss：形态学边界监督，提升 HD95
- Router Reg：鼓励注意力权重分散，防止路由坍缩

参数说明：
- include_background: 是否包含背景类
- to_onehot_y: 是否将标签转换为 one-hot
- softmax: 是否在输出应用 softmax
- smooth_nr: 分子平滑项
- smooth_dr: 分母平滑项
- alpha: Dice Loss 权重
- beta: Focal Loss 权重
- gamma: Boundary Loss 权重
- focal_alpha: Focal Loss 的 alpha 系数
- focal_gamma: Focal Loss 的 gamma 指数
- router_reg_weight: 路由正则权重
- boundary_kernel_size: 形态学膨胀核大小
"""

import torch
import torch.nn.functional as F


def _dice_loss(pred, target, smooth_nr=0, smooth_dr=1e-6, include_background=True):
    """Dice Loss 实现（per-class）

    自动对 pred 应用 sigmoid 转概率。
    """
    pred = torch.sigmoid(pred)  # logits → 概率
    if not include_background:
        pred = pred[:, 1:]
        target = target[:, 1:]

    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2.0 * intersection + smooth_nr) / (union + smooth_dr)
    return 1 - dice.mean()


def _focal_loss(pred, target, alpha=0.25, gamma=2.0, include_background=True):
    """Focal Loss 实现"""
    if not include_background:
        pred = pred[:, 1:]
        target = target[:, 1:]

    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


def _boundary_loss(pred, target, kernel_size=3, include_background=True):
    """Boundary Loss：基于形态学梯度的边界监督"""
    pred_soft = torch.sigmoid(pred)
    if not include_background:
        pred_soft = pred_soft[:, 1:]
        target = target[:, 1:]

    padding = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=pred.device)

    boundary_loss = 0.0
    n_classes = pred_soft.shape[1]
    for c in range(n_classes):
        target_c = target[:, c:c+1].float()
        pred_c = pred_soft[:, c:c+1]

        target_dilated = F.conv3d(target_c, kernel, padding=padding)
        target_boundary = (target_dilated > 0).float() - target_c

        pred_dilated = F.conv3d(pred_c, kernel, padding=padding)
        pred_boundary = (pred_dilated > 0.5).float() - pred_c

        boundary_loss = boundary_loss + F.mse_loss(pred_boundary, target_boundary)
    return boundary_loss / n_classes


class V6Loss:
    """
    V6 组合损失函数：Dice + Focal + Boundary + 路由正则

    组合损失，针对 V6 的 MoA 架构优化

    参数说明：
    - include_background: 是否包含背景类（True/False）
    - to_onehot_y: 是否将标签转换为 one-hot（True/False）
    - softmax: 是否在输出应用 softmax（True/False，V6Loss 内部用 sigmoid 转换）
    - smooth_nr: 分子平滑项
    - smooth_dr: 分母平滑项
    - alpha: Dice Loss 权重
    - beta: Focal Loss 权重
    - gamma: Boundary Loss 权重
    - focal_alpha: Focal Loss 的 alpha 系数
    - focal_gamma: Focal Loss 的 gamma 指数
    - router_reg_weight: 路由正则权重
    - boundary_kernel_size: 形态学膨胀核大小
    """

    def __init__(
        self,
        include_background=False,    # V6 默认排除背景
        to_onehot_y=True,             # 将标签转为 one-hot
        softmax=False,                # V6Loss 内部使用 sigmoid
        smooth_nr=1e-6,              # 分子平滑项
        smooth_dr=1e-6,              # 分母平滑项
        alpha=0.5,                    # Dice 权重
        beta=0.3,                     # Focal 权重
        gamma=0.2,                    # Boundary 权重
        focal_alpha=0.25,             # Focal 内部 alpha
        focal_gamma=2.0,              # Focal 内部 gamma
        router_reg_weight=0.01,       # 路由正则权重
        boundary_kernel_size=3,       # 形态学膨胀核大小
    ):
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.router_reg_weight = router_reg_weight
        self.boundary_kernel_size = boundary_kernel_size

    def __call__(self, outputs, targets, model=None):
        """
        计算 V6 组合损失

        Args:
            outputs: 模型输出 (B, C, D, H, W)，未归一化的 logits
            targets: 标签 (B, 1, D, H, W) 或 (B, D, H, W)
            model: 模型实例（用于路由正则，可选）

        Returns:
            V6 总损失（标量 tensor）
        """
        num_classes = outputs.shape[1]

        # 标签预处理
        if targets.shape[1] == 1 and num_classes > 1:
            # 单通道标签 → one-hot
            targets_onehot = F.one_hot(targets.squeeze(1).long(), num_classes=num_classes)
            targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()
        else:
            targets_onehot = targets.float()

        # 1. Dice Loss
        dice = _dice_loss(
            outputs, targets_onehot,
            smooth_nr=self.smooth_nr,
            smooth_dr=self.smooth_dr,
            include_background=self.include_background,
        )

        # 2. Focal Loss
        focal = _focal_loss(
            outputs, targets_onehot,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            include_background=self.include_background,
        )

        # 3. Boundary Loss（不含背景）
        boundary = _boundary_loss(
            outputs, targets_onehot,
            kernel_size=self.boundary_kernel_size,
            include_background=False,
        )

        # 4. 路由正则（鼓励注意力权重分散）
        router_reg = outputs.new_zeros(())
        if model is not None and hasattr(model, 'bottleneck_moa'):
            try:
                B = outputs.shape[0]
                device = outputs.device
                dummy = torch.randn(B, outputs.shape[1], 8, 8, 8, device=device)
                weights = model.bottleneck_moa.get_attention_weights(dummy)
                # 标准差越大 = 权重越分散 = 越不容易坍缩
                router_reg = -weights.std()
            except Exception:
                router_reg = outputs.new_zeros(())

        # 组合损失
        total = (
            self.alpha * dice
            + self.beta * focal
            + self.gamma * boundary
            + self.router_reg_weight * router_reg
        )
        return total


class V6LossSimple:
    """
    V6 简化损失：Dice + Focal + Boundary（无路由正则，速度快）

    参数说明同 V6Loss（无 router_reg_weight）
    """

    def __init__(
        self,
        include_background=False,
        to_onehot_y=True,
        softmax=False,
        smooth_nr=1e-6,
        smooth_dr=1e-6,
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        focal_alpha=0.25,
        focal_gamma=2.0,
        boundary_kernel_size=3,
    ):
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.boundary_kernel_size = boundary_kernel_size

    def __call__(self, outputs, targets, model=None):
        """计算 V6LossSimple（不返回 dict，无 router_reg）"""
        num_classes = outputs.shape[1]
        if targets.shape[1] == 1 and num_classes > 1:
            targets_onehot = F.one_hot(targets.squeeze(1).long(), num_classes=num_classes)
            targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()
        else:
            targets_onehot = targets.float()

        dice = _dice_loss(
            outputs, targets_onehot,
            smooth_nr=self.smooth_nr, smooth_dr=self.smooth_dr,
            include_background=self.include_background,
        )
        focal = _focal_loss(
            outputs, targets_onehot,
            alpha=self.focal_alpha, gamma=self.focal_gamma,
            include_background=self.include_background,
        )
        boundary = _boundary_loss(
            outputs, targets_onehot,
            kernel_size=self.boundary_kernel_size, include_background=False,
        )

        return self.alpha * dice + self.beta * focal + self.gamma * boundary


__all__ = ["V6Loss", "V6LossSimple"]
