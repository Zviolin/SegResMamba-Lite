"""
SWA 等权滑动平均（Stochastic Weight Averaging）
基于 PyTorch 官方 torch.optim.swa_utils.AveragedModel 实现
用于稳定模型权重

口径定案（重要）：
- V10 及全部消融实验、SegResMamba-Lite 全部版本（V7/V8 除外）、
  SegResNet-DiceFocal 基线的"平均权重"实际执行的是 SWA 等权平均
  （θ_avg = (θ_1 + ... + θ_n) / n，无衰减加权）；
- 历史命名说明：上述实验的代码/CLI/checkpoint 文件名中曾统一叫 "EMA"，
  实际语义即本文件的等权 SWA。2026-10-05 起命名修正：
  代码与 CLI 全部改为 SWA 命名，EMA 一词仅指真指数移动平均（见 ema.py）；
- 旧权重文件名（best_metric_ema_model.pth）与旧 checkpoint 键
  （ema_state_dict）保持原名不改，加载端（evaluate.py / 断点续训）
  对新旧两种命名均兼容；state_dict 键格式（AveragedModel 的
  module.* + n_averaged）与历史完全一致，旧权重可直接加载。
"""

import torch
from torch.optim.swa_utils import AveragedModel


class SWA:
    """
    SWA 等权权重平均器

    基于 PyTorch 官方 AveragedModel，兼容 PyTorch 生态
    支持 exclude_param_names 排除特定参数（排除参数不参与平均，直接复制最新值）

    注意：本类不含衰减率 decay——等权平均中所有历史权重等比重，
    无任何指数加权（decay 参数在历史"EMA"命名下从未生效，即为此语义）。
    真指数移动平均请使用 shared.optim.ema.EMA。
    """

    def __init__(self, model, exclude_param_names=None):
        self.swa_model = AveragedModel(model, avg_fn=None)
        self.model = model
        self.exclude_param_names = exclude_param_names or []

    @property
    def averaged_model(self):
        """平均权重模型（统一访问口，EMA 类同名属性对齐）"""
        return self.swa_model

    def update(self):
        """更新等权平均（排除指定参数）"""
        # 快路径：无排除参数 → 直接走 AveragedModel 等权平均
        # （与历史全部 SWA 实验（旧命名"EMA"）逐位一致）
        if not self.exclude_param_names:
            self.swa_model.update_parameters(self.model)
            return

        # 排除场景：先整体等权平均，再将排除参数覆盖为最新值
        # （该组合在历史实验中从未启用过，属新语义，仅保留 API 完整性）
        self.swa_model.update_parameters(self.model)
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if any(ex in name for ex in self.exclude_param_names):
                    for swa_name, swa_p in self.swa_model.module.named_parameters():
                        if swa_name == name:
                            swa_p.data.copy_(p.data)  # 排除：直接复制（保留最新值）
                            break

    @torch.no_grad()
    def validate(self, val_loader, spacing, roi_size, sw_batch_size=4, overlap=0.5, mode="gaussian"):
        """
        使用平均权重模型进行验证

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

        self.swa_model.eval()

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
                    self.swa_model, val_inputs, roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=overlap, mode=mode
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
        """获取平均权重模型状态字典（AveragedModel 格式，与历史 checkpoint 键完全一致）"""
        return self.swa_model.state_dict()

    def load_state_dict(self, state_dict):
        """加载平均权重模型状态字典（兼容历史旧 checkpoint 的键格式）"""
        self.swa_model.load_state_dict(state_dict)


__all__ = ["SWA"]
