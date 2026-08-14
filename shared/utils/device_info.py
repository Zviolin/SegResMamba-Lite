import torch
import platform
import psutil
import os
import sys


def print_device_info(logger=None):
    """打印设备详细信息

    Args:
        logger: Logger实例，如果为None则使用print
    """
    if logger is None:
        _print = lambda msg: print(msg)
    else:
        _print = lambda msg: (logger.terminal.write(msg + "\n"), logger.file.write(msg + "\n"))

    lines = [
        "=" * 50,
        f"PyTorch: {torch.__version__} | Python: {platform.python_version()}",
    ]

    if torch.cuda.is_available():
        lines.append(f"使用设备: cuda")
        lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
        lines.append(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        lines.append(f"CUDA: {torch.version.cuda} | cuDNN: {torch.backends.cudnn.version()}")
    else:
        lines.append(f"使用设备: cpu")
        lines.append(f"CPU: {platform.processor()}")
        lines.append(f"CPU核心数: {os.cpu_count()}")

    mem = psutil.virtual_memory()
    lines.append(f"内存: {mem.total / 1024**3:.1f} GB | 可用: {mem.available / 1024**3:.1f} GB")
    lines.append("=" * 50)

    for line in lines:
        _print(line)


if __name__ == "__main__":
    print_device_info()
