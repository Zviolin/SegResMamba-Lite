"""
【Baseline 模块】
模型 + 框架
"""

from .models import get_model
from .frame.train import BaselineTrainer

__all__ = ["get_model", "BaselineTrainer"]
