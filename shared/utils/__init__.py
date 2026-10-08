"""
工具函数
包括日志、设备管理、检查点、模型量化等功能
"""

import os

from .logger import Logger, setup_logger
from .device import get_device, check_gpu_compatibility, setup_cuda_optimization
from .paths import resolve_model_dir, resolve_log_prefix, resolve_cache_parent, resolve_cache_dir
from .checkpoint import ModelCheckpoint, load_checkpoint
from .quantize import quantize_model, benchmark_inference, get_model_from_checkpoint


def get_model_dir(base_dir, version_name, resolution):
    """
    获取模型保存目录

    Args:
        base_dir: 基础目录
        version_name: 版本名称
        resolution: 分辨率

    Returns:
        模型保存目录路径
    """
    model_dir = os.path.join(base_dir, version_name, "models", f"{resolution}mm")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def get_output_dir(base_dir, version_name, resolution):
    """
    获取输出保存目录

    Args:
        base_dir: 基础目录
        version_name: 版本名称
        resolution: 分辨率

    Returns:
        输出保存目录路径
    """
    output_dir = os.path.join(base_dir, version_name, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


__all__ = [
    "Logger",
    "setup_logger",
    "get_device",
    "check_gpu_compatibility",
    "setup_cuda_optimization",
    "ModelCheckpoint",
    "load_checkpoint",
    "get_model_dir",
    "get_output_dir",
    "quantize_model",
    "benchmark_inference",
    "get_model_from_checkpoint",
    "resolve_model_dir",
    "resolve_log_prefix",
    "resolve_cache_parent",
    "resolve_cache_dir",
]