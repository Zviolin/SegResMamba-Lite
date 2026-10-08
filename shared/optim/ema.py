"""
真指数移动平均（Exponential Moving Average, EMA）
递推公式：θ_ema ← decay × θ_ema + (1 - decay) × θ
用于 V7/V8（含 alpha_raw/temperature 排除）与 EMA1 对照消融的语义复现

与 SWA 的区别（命名定案 2026-10-05）：
- 本类为真指数移动平均：近期权重按 decay 指数加权，decay 参数生效；
- V10 及其余全部版本/消融的"平均权重"是 SWA 等权平均（decay 不生效），
  请使用 shared.optim.swa.SWA（历史命名"EMA"实为等权 SWA，已修正命名）；
- 内部仍以 AveragedModel 包装，state_dict 键格式（module.* + n_averaged）
  与历史 checkpoint（含旧键名 ema_state_dict、旧文件名
  best_metric_ema_model.pth）完全一致，旧权重可直接加载。
"""

import torch
from torch.optim.swa_utils import AveragedModel


class EMA:
    """
    真指数移动平均器

    基于 AveragedModel 包装（仅为保持 checkpoint 键格式兼容），
    更新走手动递推路径，decay 参数真实生效。
    支持 exclude_param_names 排除特定参数（如 V7/V8 的 alpha_raw / temperature，
    排除参数不参与递推，直接复制最新值）。

    历史行为对应：
    - V7/V8：exclude_param_names 非空（alpha_raw/temperature 复制、其余按 decay 递推）；
    - EMA1 对照消融：exclude_param_names 为空（全部参数按 decay=0.999 递推）。
    """

    def __init__(self, model, decay=0.999, exclude_param_names=None):
        self.ema_model = AveragedModel(model, avg_fn=None)
        self.decay = decay
        self.model = model
        self.exclude_param_names = exclude_param_names or []

    @property
    def averaged_model(self):
        """平均权重模型（统一访问口，与 SWA 类同名属性对齐）"""
        return self.ema_model

    def update(self):
        """更新指数移动平均（排除指定参数：直接复制，不递推）"""
        # 手动递推：全部参数按 decay 指数加权（排除参数直接复制保留最新值）
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                # 找到对应 EMA 参数
                for ema_name, ema_p in self.ema_model.module.named_parameters():
                    if ema_name == name:
                        if any(ex in name for ex in self.exclude_param_names):
                            # 排除：直接复制（保留最新值）
                            ema_p.data.copy_(p.data)
                        else:
                            # EMA 递推
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
        """获取 EMA 模型状态字典（AveragedModel 格式，与历史 checkpoint 键完全一致）"""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        """加载 EMA 模型状态字典（兼容历史旧 checkpoint 的键格式）"""
        self.ema_model.load_state_dict(state_dict)


__all__ = ["EMA"]
