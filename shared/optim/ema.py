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

    def get_state_dict(self):
        """获取 EMA 模型状态字典（AveragedModel 格式，与历史 checkpoint 键完全一致）"""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        """加载 EMA 模型状态字典（兼容历史旧 checkpoint 的键格式）"""
        self.ema_model.load_state_dict(state_dict)


__all__ = ["EMA"]
