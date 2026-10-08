"""
【数据预处理】
数据划分、缓存预计算
供所有版本统一使用
"""

import os
import sys
import json
import random
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from monai.data import PersistentDataset, DataLoader

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # 向上两级到 Code 目录
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from shared.data.dataloader import get_data_list, get_train_pre_transforms
from shared.utils.device_info import print_device_info
from shared.utils.paths import resolve_cache_dir


def prepare_data(
    # ═══════════════════════════════════════════════════════════════════════
    # 命令行参数
    # ═══════════════════════════════════════════════════════════════════════
    resolution=1.0,             # 数据分辨率 (1.0/2.0/3.0/4.0 mm)
    cache_dir=None,            # 缓存目录路径
    max_samples=None,          # 最大样本数 (None=全部)
    seed=42,                  # 随机种子
    dtype="float16",           # 数据类型: float16 或 float32
    cache_format="numpy",      # 缓存格式: numpy 或 tensor
    # ═══════════════════════════════════════════════════════════════════════
    # 数据划分参数
    # ═══════════════════════════════════════════════════════════════════════
    train_ratio=0.7,           # 训练集比例
    val_ratio=0.15,           # 验证集比例
    test_ratio=0.15,          # 测试集比例
    # ═══════════════════════════════════════════════════════════════════════
    # 生成参数
    # ═══════════════════════════════════════════════════════════════════════
    num_workers=0,             # 数据加载线程数
    generate_cache_flag=False,  # 是否生成缓存
):
    """
    数据预处理和划分 (70% 训练, 15% 验证, 15% 测试)

    Args:
        resolution: 数据分辨率 (1.0/2.0/3.0/4.0 mm)
        cache_dir: 缓存目录路径
        max_samples: 最大样本数 (None=全部)
        seed: 随机种子
        dtype: 数据类型，可选 "float16" 或 "float32"
        cache_format: 缓存格式，可选 "numpy" 或 "tensor"
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        num_workers: 数据加载线程数
        generate_cache_flag: 是否生成缓存
    """
    DATA_DIR = r"g:\Codes\Python\SRTP\Essay\Data\Brain\TCIA-BraTS\DATA\BraTS2023\BraTS-GLI\TrainingData\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    CACHE_DIR = r"D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache"

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    # ═══ 修复：固定全局随机种子（否则 --samples 随机选取不可复现）═══
    # 此前 L94 的 random.sample 用的是 OS 熵种子，--seed 只被打印未生效，
    # 导致指定 max_samples 时每次运行选出的病例不同；sklearn 的
    # train_test_split 用独立 random_state 不受影响。加这行后：
    # 同 seed 下 random.sample 选取结果逐位一致，全流程可复现。
    random.seed(seed)

    print(f"数据目录: {DATA_DIR}")

    all_data = get_data_list(DATA_DIR, mode="train")
    print(f"总样本数: {len(all_data)}")

    case_ids = []
    for d in all_data:
        image_path = d["image"]
        if isinstance(image_path, list):
            image_path = image_path[0]
        filename = os.path.basename(image_path)
        parts = filename.split("-")
        if len(parts) >= 4:
            case_id = "-".join(parts[:4])
        else:
            case_id = filename
        case_ids.append(case_id)

    unique_case_ids = list(set(case_ids))
    unique_case_ids.sort()
    print(f"唯一病例数: {len(unique_case_ids)}")

    if max_samples is not None and max_samples < len(unique_case_ids):
        print(f"随机选取 {max_samples} 样本 (seed={seed})")
        unique_case_ids = sorted(random.sample(unique_case_ids, max_samples))
        all_data = [d for d, cid in zip(all_data, case_ids) if cid in unique_case_ids]
        case_ids = [cid for cid in case_ids if cid in unique_case_ids]
        print(f"选取后样本数: {len(unique_case_ids)}")

    train_case_ids, temp_case_ids = train_test_split(
        unique_case_ids,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        shuffle=True
    )

    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    val_case_ids, test_case_ids = train_test_split(
        temp_case_ids,
        test_size=(1 - val_test_ratio),
        random_state=seed,
        shuffle=True
    )

    print(f"\n数据划分 (seed={seed}):")
    print(f"训练集: {len(train_case_ids)} 病例 ({train_ratio*100:.0f}%)")
    print(f"验证集: {len(val_case_ids)} 病例 ({val_ratio*100:.0f}%)")
    print(f"测试集: {len(test_case_ids)} 病例 ({test_ratio*100:.0f}%)")

    train_files = [d for d, cid in zip(all_data, case_ids) if cid in train_case_ids]
    val_files = [d for d, cid in zip(all_data, case_ids) if cid in val_case_ids]
    test_files = [d for d, cid in zip(all_data, case_ids) if cid in test_case_ids]

    print(f"\n训练样本数: {len(train_files)}")
    print(f"验证样本数: {len(val_files)}")
    print(f"测试样本数: {len(test_files)}")

    current_cache_dir = resolve_cache_dir(resolution, CACHE_DIR) if generate_cache_flag else None
    if current_cache_dir:
        print(f"\n缓存目录: {current_cache_dir}")

    pixdim = (resolution, resolution, resolution)
    print(f"重采样分辨率: {pixdim} mm")
    print(f"缓存数据类型: {dtype}")
    print(f"缓存格式: {cache_format}")
    
    print(f"\n【缓存配置说明】")
    print(f"- {dtype} 格式: {'节省50%存储空间，推荐用于GPU训练' if dtype == 'float16' else '更高精度，适合CPU训练或需要高精度场景'}")
    print(f"- {cache_format} 格式: {'跨框架兼容，调试方便，推荐用于大多数场景' if cache_format == 'numpy' else '加载速度稍快，仅支持PyTorch'}")
    if cache_format == "numpy" and dtype == "float16":
        print(f"✓ 推荐配置: numpy + float16 组合，兼容性好且内存效率高")
    elif cache_format == "tensor" and dtype == "float32":
        print(f"⚠ 注意: tensor + float32 组合缓存文件较大，适合纯PyTorch高精度场景")

    if generate_cache_flag and current_cache_dir:
        print("\n开始生成训练集缓存...")
        generate_cache(train_files, os.path.join(current_cache_dir, "train"), pixdim, num_workers, dtype, cache_format)
        print("\n开始生成验证集缓存...")
        generate_cache(val_files, os.path.join(current_cache_dir, "val"), pixdim, num_workers, dtype, cache_format)
        print("\n开始生成测试集缓存...")
        generate_cache(test_files, os.path.join(current_cache_dir, "test"), pixdim, num_workers, dtype, cache_format)

    if current_cache_dir:
        split_info = {
            "resolution": resolution,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "train_case_ids": train_case_ids,
            "val_case_ids": val_case_ids,
            "test_case_ids": test_case_ids,
        }
        os.makedirs(current_cache_dir, exist_ok=True)
        split_file = os.path.join(current_cache_dir, "split_info.json")
        with open(split_file, "w") as f:
            json.dump(split_info, f, indent=2)
        print(f"\n划分信息已保存: {split_file}")

    print("\n数据预处理完成！")

    return {
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
        "train_case_ids": train_case_ids,
        "val_case_ids": val_case_ids,
        "test_case_ids": test_case_ids,
        "pixdim": pixdim,
        "cache_dir": current_cache_dir,
    }


def _hash_func(data):
    """
    Custom hash function to use readable filenames for cache.
    Returns the 'id' of the data item (e.g., BraTS-GLI-00000-000).
    """
    id_str = data.get("id", str(hash(str(data))))
    return id_str.encode("utf-8")


def cleanup_corrupted_cache(cache_dir):
    """
    扫描缓存目录，尝试读取所有 .pt 文件。
    如果读取失败（说明文件损坏），则删除它。
    """
    if not os.path.exists(cache_dir):
        return

    print(f"扫描损坏的缓存文件: {cache_dir}...")
    files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
    corrupted_count = 0

    for f in tqdm(files, desc="验证缓存"):
        path = os.path.join(cache_dir, f)
        try:
            torch.load(path, map_location="cpu")
        except Exception:
            try:
                os.remove(path)
                corrupted_count += 1
            except OSError:
                pass

    if corrupted_count > 0:
        print(f"已删除 {corrupted_count} 个损坏的缓存文件")
    else:
        print("所有缓存文件验证通过")


def generate_cache(data_files, cache_dir, pixdim, num_workers=0, dtype="float16", cache_format="numpy"):
    """
    多进程生成缓存

    Args:
        data_files: 数据文件列表
        cache_dir: 缓存目录
        pixdim: 重采样分辨率
        num_workers: worker 进程数
        dtype: 数据类型，可选 "float16" 或 "float32"
        cache_format: 缓存格式，可选 "numpy" 或 "tensor"
    """
    os.makedirs(cache_dir, exist_ok=True)

    cleanup_corrupted_cache(cache_dir)

    pre_transforms = get_train_pre_transforms(pixdim=pixdim, dtype=dtype, cache_format=cache_format)

    ds = PersistentDataset(
        data=data_files,
        transform=pre_transforms,
        cache_dir=cache_dir,
        hash_func=_hash_func
    )

    print(f"开始缓存生成 (支持断点续传)...")
    
    # 处理 num_workers 为 None 的情况
    if num_workers is None:
        cpu_count = os.cpu_count() or 4
        num_workers = min(8, max(1, cpu_count // 2))
        print(f"自动设置 num_workers: {num_workers} (CPU: {cpu_count})")
    
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
    )

    for _ in tqdm(loader, desc="缓存中"):
        pass

    print("缓存生成完成！")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    import argparse
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument("--resolution", type=float, default=1.0, help="分辨率")
    parser.add_argument("--cache", action="store_true", help="启用缓存")
    parser.add_argument("--generate", action="store_true", help="生成缓存")
    parser.add_argument("--train", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val", type=float, default=0.15, help="验证集比例")
    parser.add_argument("--test", type=float, default=0.15, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--workers", type=str, default="auto", help="worker 数 (auto/数字，auto表示自动计算)")
    parser.add_argument("--samples", type=int, default=None, help="最大使用的样本数")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"], help="缓存数据类型")
    parser.add_argument("--cache-format", type=str, default="numpy", choices=["numpy", "tensor"], help="缓存格式")
    args = parser.parse_args()

    resolution = args.resolution
    enable_cache = args.cache  # True/False 开关
    train_ratio = args.train
    val_ratio = args.val
    test_ratio = args.test
    seed = args.seed
    
    # 解析 num_workers 参数
    if args.workers.lower() == "auto":
        num_workers = None
    else:
        try:
            num_workers = int(args.workers)
        except ValueError:
            print(f"警告: 无效的 --workers 值 '{args.workers}'，使用 auto")
            num_workers = None
    max_samples = args.samples
    dtype = args.dtype
    cache_format = args.cache_format

    DATA_DIR = r"g:\Codes\Python\SRTP\Essay\Data\Brain\TCIA-BraTS\DATA\BraTS2023\BraTS-GLI\TrainingData\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    CACHE_DIR = r"D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache"
    cache_dir = resolve_cache_dir(resolution, CACHE_DIR) if enable_cache else None  # 修正：正确的目录路径
    current_cache_dir = cache_dir  # 保持变量名一致性

    import logging
    import sys
    from shared.utils.logger import setup_logger

    logger = setup_logger(os.path.dirname(CACHE_DIR) if enable_cache else DATA_DIR, "preprocess")

    # ─────────────────────────────────────────────────────────────────────
    # 设备信息
    # ─────────────────────────────────────────────────────────────────────
    print_device_info(logger)

    logger.print_config_grouped({
        "数据设置": {
            "resolution": f"{resolution} mm",
            "train_ratio": f"{train_ratio * 100:.0f}%",
            "val_ratio": f"{val_ratio * 100:.0f}%",
            "test_ratio": f"{test_ratio * 100:.0f}%",
            "seed": seed,
            "max_samples": max_samples if max_samples else "全部",
        },
        "路径设置": {
            "data_dir": DATA_DIR,
            "cache_dir": cache_dir if cache_dir else "未启用缓存",
        },
        "缓存设置": {
            "dtype": dtype,
            "cache_format": cache_format,
        },
        "生成设置": {
            "generate_cache": enable_cache,  # 修正：使用 enable_cache
            "num_workers": num_workers if num_workers else "主进程",
        },
    }, title="数据预处理配置")

    prepare_data(
        resolution=resolution,
        cache_dir=cache_dir,
        generate_cache_flag=enable_cache,  # 修正：使用 enable_cache
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        num_workers=num_workers,
        max_samples=max_samples,
        dtype=dtype,
        cache_format=cache_format,
    )
