"""
EMA 指数移动平均
基于 PyTorch 官方 torch.optim.swa_utils.AveragedModel 实现
用于稳定模型权重
"""

import torch
from torch.optim.swa_utils import AveragedModel


class EMA:
    """
    Exponential Moving Average 权重更新

    基于 PyTorch 官方 AveragedModel，兼容 PyTorch 生态
    支持 exclude_param_names 排除特定参数（如 alpha_raw）
    """

    def __init__(self, model, decay=0.999, exclude_param_names=None):
        self.ema_model = AveragedModel(model, avg_fn=None)
        self.decay = decay
        self.model = model
        self.exclude_param_names = exclude_param_names or []

    def update(self):
        """更新 EMA 权重（排除指定参数）"""
        if not self.exclude_param_names:
            self.ema_model.update_parameters(self.model)
            return

        # 手动更新：排除特定参数（直接复制，不平均）
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                # 找到对应 EMA 参数
                for ema_name, ema_p in self.ema_model.module.named_parameters():
                    if ema_name == name:
                        if any(ex in name for ex in self.exclude_param_names):
                            # 排除：直接复制（保留最新值）
                            ema_p.data.copy_(p.data)
                        else:
                            # EMA 更新
                            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
                        break

    @torch.no_grad()
    def validate(self, val_loader, spacing, roi_size, sw_batch_size=4, overlap=0.5, mode="gaussian"):
        """
        使用 EMA 模型进行验证

        Args:
            val_loader: 验证数据加载器
            spacing: 体素间距
            roi_size: ROI 大小
            sw_batch_size: 滑动窗口批大小
            overlap: 重叠比例
            mode: 融合模式

        Returns:
            验证指标字典
        """
        from monai.metrics import DiceMetric, HausdorffDistanceMetric
        from monai.transforms import AsDiscrete
        from monai.data import decollate_batch
        from tqdm import tqdm
        from shared.inference.sliding_window import sliding_window_inference
        import numpy as np

        self.ema_model.eval()

        dice_metric_wt = DiceMetric(include_background=False, reduction="mean")
        dice_metric_tc = DiceMetric(include_background=False, reduction="mean")
        dice_metric_et = DiceMetric(include_background=False, reduction="mean")
        hd95_metric_wt = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        hd95_metric_tc = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        hd95_metric_et = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

        for val_data in tqdm(val_loader, desc="Validating"):
            val_inputs = val_data["image"].to(self.model.device)
            val_labels = val_data["label"].to(self.model.device)

            with torch.amp.autocast('cuda', enabled=(self.model.device == "cuda")):
                val_outputs = sliding_window_inference(
                    self.ema_model, val_inputs, roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=overlap, mode=mode
                )

            val_outputs_list = [AsDiscrete(argmax=True, to_onehot=4)(i) for i in decollate_batch(val_outputs)]
            val_labels_list = [AsDiscrete(to_onehot=4)(i) for i in decollate_batch(val_labels)]

            pred_wt = [(i[1] + i[2] + i[3] > 0).float().unsqueeze(0) for i in val_outputs_list]
            true_wt = [(i[1] + i[2] + i[3] > 0).float().unsqueeze(0) for i in val_labels_list]

            pred_tc = [(i[1] + i[3] > 0).float().unsqueeze(0) for i in val_outputs_list]
            true_tc = [(i[1] + i[3] > 0).float().unsqueeze(0) for i in val_labels_list]

            pred_et = [(i[3] > 0).float().unsqueeze(0) for i in val_outputs_list]
            true_et = [(i[3] > 0).float().unsqueeze(0) for i in val_labels_list]

            dice_metric_wt(y_pred=pred_wt, y=true_wt)
            dice_metric_tc(y_pred=pred_tc, y=true_tc)
            dice_metric_et(y_pred=pred_et, y=true_et)
            hd95_metric_wt(y_pred=pred_wt, y=true_wt, spacing=spacing)
            hd95_metric_tc(y_pred=pred_tc, y=true_tc, spacing=spacing)
            hd95_metric_et(y_pred=pred_et, y=true_et, spacing=spacing)

        metric_wt = dice_metric_wt.aggregate().item()
        metric_tc = dice_metric_tc.aggregate().item()
        metric_et = dice_metric_et.aggregate().item()
        hd95_wt = hd95_metric_wt.aggregate().item()
        hd95_tc = hd95_metric_tc.aggregate().item()
        hd95_et = hd95_metric_et.aggregate().item()

        dice_metric_wt.reset()
        dice_metric_tc.reset()
        dice_metric_et.reset()
        hd95_metric_wt.reset()
        hd95_metric_tc.reset()
        hd95_metric_et.reset()

        return {
            "dice_wt": metric_wt,
            "dice_tc": metric_tc,
            "dice_et": metric_et,
            "dice_avg": (metric_wt + metric_tc + metric_et) / 3,
            "hd95_wt": hd95_wt,
            "hd95_tc": hd95_tc,
            "hd95_et": hd95_et,
            "hd95_avg": (hd95_wt + hd95_tc + hd95_et) / 3,
        }

    def get_state_dict(self):
        """获取 EMA 模型状态字典（包含 module. 前缀）"""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        """加载 EMA 模型状态字典"""
        self.ema_model.load_state_dict(state_dict)


__all__ = ["EMA"]
