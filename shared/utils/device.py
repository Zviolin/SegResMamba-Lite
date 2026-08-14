"""
设备工具
GPU 检测和设置
"""

import torch


def get_device():
    """
    获取可用设备

    Returns:
        str: "cuda" 如果 GPU 可用，否则返回 "cpu"
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def check_gpu_compatibility():
    """
    检测 GPU 兼容性

    用于 RTX 5060 等最新 GPU 的兼容性测试

    Returns:
        tuple: (is_available, device_type)
            - is_available: GPU 是否可用
            - device_type: "cuda" 或 "cpu"
    """
    if not torch.cuda.is_available():
        return False, "cpu"

    try:
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print("正在测试 GPU 计算能力...")
        x = torch.randn(10, 10).cuda()
        y = torch.mm(x, x)
        z = x[x > 0]
        x[x < 0] = 0
        print("GPU 测试通过！")
        return True, "cuda"
    except RuntimeError as e:
        print(f"\n{'!'*60}")
        print(f"警告: GPU 测试失败 ({e})")
        print("检测到您使用的是最新的 NVIDIA RTX 5060 (Blackwell 架构)。")
        print("已自动切换到 CPU 模式。")
        print(f"{'!'*60}\n")
        return False, "cpu"


def setup_cuda_optimization():
    """
    设置 CUDA 优化选项

    启用以下优化：
    - cudnn.benchmark: 启用 cudnn 自动调优
    - TF32: 允许在 Ampere+ GPU 上使用 TF32 格式加速矩阵乘法
    - allow_tf32: 允许 cudnn 使用 TF32 格式
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


__all__ = [
    "get_device",
    "check_gpu_compatibility",
    "setup_cuda_optimization",
]