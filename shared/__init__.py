"""
Shared 共享模块
统一的共享组件，供所有版本使用

目录结构:
├── shared/
│   ├── data/              # 数据处理
│   │   ├── transforms.py  # 数据变换
│   │   ├── dataset.py     # 数据集
│   │   └── dataloader.py  # 数据加载器
│   ├── losses/            # 损失函数
│   │   ├── dice_ce_loss.py
│   │   ├── dice_focal_loss.py
│   │   └── dice_loss.py
│   ├── optim/             # 优化器
│   │   ├── adamw.py       # AdamW
│   │   ├── cosine_scheduler.py # CosineAnnealingLR
│   │   ├── deep_supervision.py  # 深层监督
│   │   └── ema.py         # EMA
│   ├── metrics/           # 评估指标
│   │   ├── dice_metric.py
│   │   └── hd95_metric.py
│   ├── inference/         # 推理工具
│   │   ├── sliding_window.py
│   │   └── postprocess.py
│   ├── models/           # 可复用的模型组件
│   │   └── mamba.py      # 官方 Mamba 实现
│   └── utils/             # 工具函数
│       ├── logger.py
│       ├── device.py
│       └── checkpoint.py
"""

from . import data
from . import losses
from . import optim
from . import metrics
from . import inference
from . import models
from . import utils

__all__ = [
    "data",
    "losses",
    "optim",
    "metrics",
    "inference",
    "models",
    "utils",
]
