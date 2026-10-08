"""
【Optimized 模块】
模型 + 框架
"""

from .models import get_model
from .frame.train import OptimizedTrainer

__all__ = ["get_model", "OptimizedTrainer"]
