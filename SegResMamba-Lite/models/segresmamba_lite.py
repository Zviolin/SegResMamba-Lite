"""
SegResMamba-Lite - 主入口文件
默认指向主版本 (v1)

这是一个兼容性文件，实际模型定义在 v1.py 中
"""

from .v1 import SegResMambaLiteV1 as SegResMambaLite
from .v1 import get_model


__all__ = ['SegResMambaLite', 'get_model']
