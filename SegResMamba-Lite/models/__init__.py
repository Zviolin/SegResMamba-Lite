"""
SegResMamba-Lite 模型入口
支持 v1, v2, v3, v4, v6, v7, v8, v9, v10 多个版本
"""

from .v1 import get_model as _get_model_v1
from .v1 import SegResMambaLiteV1
from .v2 import get_model as _get_model_v2
from .v2 import SegResMambaLiteV2
from .v3 import get_model as _get_model_v3
from .v3 import SegResMambaLiteV3
from .v4 import get_model as _get_model_v4
from .v4 import SegResMambaLiteV4
from .v6 import get_model as _get_model_v6
from .v6 import SegResMambaLiteV6
from .v7 import get_model as _get_model_v7
from .v7 import SegResMambaLiteV7
from .v8 import get_model as _get_model_v8
from .v8 import SegResMambaLiteV8
from .v9 import get_model as _get_model_v9
from .v9 import SegResMambaLiteV9
from .v10 import get_model as _get_model_v10
from .v10 import SegResMambaLiteV10
from .v11 import get_model as _get_model_v11
from .v11 import SegResMambaLiteV11


def get_model(version='v4', **kwargs):
    """
    获取模型

    Args:
        version: 'v1', 'v2', 'v3', 'v4', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11'
            - v1: 原始双向 Mamba + 跳跃连接
            - v2: 终极深度融合版本
            - v3: 轻量优化版：SE注意力 + 双重Bottleneck + 解码器Mamba
            - v4: 深度融合优化版：保持HD95优势 + 提升Dice
            - v6: 实用 MoA 版（共享 Mamba + 4×DConv 专家，软权重路由，α=0.5）
            - v7: V6 + Sigmoid α 自动学习
            - v8: V6 + α 真正自由自适应（温度缩放，学术最佳实践）
            - v9: 真正 MoE 版（Top-K=2 稀疏激活，4 个 DConv 专家，负载均衡）
            - v10: 论文级 MoA 版（共享 K/V + Attention 专家 + token 级 Top-K 稀疏）
            - v11: 边界感知混合 MoA（2 Attention + 2 DConv + 边界注意力 + 路由噪声）
    """
    if version == 'v1':
        return _get_model_v1(**kwargs)
    elif version == 'v2':
        return _get_model_v2(**kwargs)
    elif version == 'v3':
        return _get_model_v3(**kwargs)
    elif version == 'v4':
        return _get_model_v4(**kwargs)
    elif version == 'v6':
        return _get_model_v6(**kwargs)
    elif version == 'v7':
        return _get_model_v7(**kwargs)
    elif version == 'v8':
        return _get_model_v8(**kwargs)
    elif version == 'v9':
        return _get_model_v9(**kwargs)
    elif version == 'v10':
        return _get_model_v10(**kwargs)
    elif version == 'v11':
        return _get_model_v11(**kwargs)
    else:
        raise ValueError(f"Unknown version: {version}")


def list_versions():
    """列出可用版本"""
    return {
        'v1': 'SegResMamba-Lite V1 - 双向 Mamba + 跳跃连接',
        'v2': 'SegResMamba-Lite V2 - 终极深度融合版本',
        'v3': 'SegResMamba-Lite V3 - 轻量优化版：SE注意力 + 双重Bottleneck + 解码器Mamba',
        'v4': 'SegResMamba-Lite V4 - 深度融合优化版：保持HD95优势 + 提升Dice',
        'v6': 'SegResMamba-Lite V6 - 实用 MoA（共享 Mamba + DConv 专家，α=0.5）',
        'v7': 'SegResMamba-Lite V7 - V6 + Sigmoid α 自动学习',
        'v8': 'SegResMamba-Lite V8 - V6 + α 真正自由自适应（温度缩放，学术最佳实践）',
        'v9': 'SegResMamba-Lite V9 - 真正 MoE（Top-K=2 稀疏激活，4 个 DConv 专家）',
        'v10': 'SegResMamba-Lite V10 - 论文级 MoA（共享 K/V + Attention 专家 + token 级 Top-K）',
        'v11': 'SegResMamba-Lite V11 - 边界感知混合 MoA（2 Attention + 2 DConv + 边界注意力 + 路由噪声）'
    }


__all__ = [
    'get_model', 'list_versions',
    'SegResMambaLiteV1', 'SegResMambaLiteV2', 'SegResMambaLiteV3',
    'SegResMambaLiteV4', 'SegResMambaLiteV6', 'SegResMambaLiteV7',
    'SegResMambaLiteV8', 'SegResMambaLiteV9', 'SegResMambaLiteV10',
    'SegResMambaLiteV11'
]
