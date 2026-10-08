"""
【LightSegMamba 训练框架 V3】

- 损失：DiceCELoss（共享模块）
- 优化器：AdamW（共享模块）
- 调度器：CosineAnnealingLR（共享模块）
- 可选 EMA
- 验证：滑窗推理 + Dice(WT/TC/ET) + HD95(WT/TC/ET)
- 接口与 MambaUNetTrainer 完全一致
"""

from .train import LightSegMambaTrainer

__all__ = ["LightSegMambaTrainer"]
