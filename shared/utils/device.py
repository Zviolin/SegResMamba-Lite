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


def setup_determinism(seed: int = 42):
    """
    【可复现性】固定全链路随机性，使同卡同 seed 逐次可复现（--deterministic 开启时调用）

    - 固定 python/numpy/torch/cuda 种子
    - cudnn 关闭自动调优（benchmark 引入非确定性 kernel 选择）
    - torch 确定性算法（warn_only：个别算子无确定性实现时仅告警不崩溃）

    注意：不触碰 TF32 开关（allow_tf32 维持 setup_cuda_optimization 的设置），
    保证与论文历史数字口径可比；TF32 的矩阵乘结果在同代卡上逐位一致。

    调用前提：os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8" 必须在任何
    CUDA 上下文创建之前设置（由各项目 pipeline/train.py 入口负责）。

    Args:
        seed: 全局随机种子

    Returns:
        seed 原样返回，便于链式使用
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    print(f"[Determinism] seed={seed} | cudnn.deterministic=True | benchmark=False | TF32 保持 setup_cuda_optimization 设置不变")
    return seed


__all__ = [
    "get_device",
    "check_gpu_compatibility",
    "setup_cuda_optimization",
    "setup_determinism",
]