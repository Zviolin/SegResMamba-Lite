"""
数据加载器
获取训练、验证、测试数据加载器

参数说明：
- data_files: 数据文件列表
- batch_size: 批次大小
- num_workers: 数据加载线程数
- pixdim: 重采样分辨率
- patch_size: 裁剪patch大小
- cache_dir: 缓存目录
- persistent_workers: 是否保持worker进程
- prefetch_factor: 预取因子
"""

import os
import glob
import json
import torch
import numpy as np
from monai.data import DataLoader, PersistentDataset, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    SpatialPadd,
    NormalizeIntensityd,
    CastToTyped,
    EnsureTyped,
)

from .transforms import (
    get_train_pre_transforms,
    get_train_aug_transforms,
    get_val_transforms,
    get_test_transforms,
)
from .dataset import AugmentedDataset


def get_data_list(data_dir, mode="train"):
    """
    获取数据文件列表

    Args:
        data_dir: 数据目录路径
        mode: "train" 或 "test"

    Returns:
        data_dicts: 包含 image 和 label 路径的字典列表
    """
    case_dirs = sorted(glob.glob(os.path.join(data_dir, "BraTS-GLI-*")))
    data_dicts = []

    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)
        t1 = os.path.join(case_dir, f"{case_id}-t1n.nii.gz")
        t1ce = os.path.join(case_dir, f"{case_id}-t1c.nii.gz")
        t2 = os.path.join(case_dir, f"{case_id}-t2w.nii.gz")
        flair = os.path.join(case_dir, f"{case_id}-t2f.nii.gz")
        seg = os.path.join(case_dir, f"{case_id}-seg.nii.gz")

        if not os.path.exists(t1):
            t1 = glob.glob(os.path.join(case_dir, "*t1n.nii.gz"))[0]
            t1ce = glob.glob(os.path.join(case_dir, "*t1c.nii.gz"))[0]
            t2 = glob.glob(os.path.join(case_dir, "*t2w.nii.gz"))[0]
            flair = glob.glob(os.path.join(case_dir, "*t2f.nii.gz"))[0]
            if mode != "test":
                seg = glob.glob(os.path.join(case_dir, "*seg.nii.gz"))[0]

        if mode == "test":
            data_dicts.append({
                "image": [t1, t1ce, t2, flair],
                "id": case_id,
            })
        else:
            data_dicts.append({
                "image": [t1, t1ce, t2, flair],
                "label": seg,
                "id": case_id,
            })

    return data_dicts


def _get_data_files_from_cache(cache_dir):
    """
    从缓存目录自动获取数据文件列表

    Args:
        cache_dir: 缓存目录，格式: {parent}/persistent_cache_{resolution}/train|val|test

    Returns:
        data_files: 数据文件列表
    """
    parent_cache_dir = os.path.dirname(os.path.abspath(cache_dir))
    split_file = os.path.join(parent_cache_dir, "split_info.json")

    with open(split_file, "r") as f:
        split_info = json.load(f)

    subdir_name = os.path.basename(cache_dir)
    if subdir_name == "train":
        case_ids = split_info["train_case_ids"]
        mode = "train"
    elif subdir_name == "val":
        case_ids = split_info["val_case_ids"]
        mode = "val"
    else:
        case_ids = split_info["test_case_ids"]
        mode = "test"

    original_data_dir = r"g:\Codes\Python\SRTP\Essay\Data\Brain\TCIA-BraTS\DATA\BraTS2023\BraTS-GLI\TrainingData\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    all_data = get_data_list(original_data_dir, mode=mode if mode != "test" else "val")
    case_ids_set = set(case_ids)
    data_files = [d for d in all_data if d["id"] in case_ids_set]
    return data_files


def _auto_num_workers():
    """
    自动计算最优的 num_workers 数量
    
    Returns:
        int: 计算得到的 workers 数量
    """
    cpu_count = os.cpu_count() or 4
    
    # Windows 上更保守，避免 WinError 1455
    if os.name == 'nt':  # Windows
        auto_workers = min(2, max(0, cpu_count // 4))
    else:  # Linux/macOS
        auto_workers = min(8, max(1, cpu_count // 2))
    
    print(f"自动设置 num_workers: {auto_workers} (CPU: {cpu_count}, OS: {os.name})")
    return auto_workers


def _auto_batch_size(patch_size=(64, 64, 64), model_name="unknown"):
    """
    根据 GPU 显存自动计算合适的 batch_size
    
    估算方法：
    - 基础估算：每 GB 显存约 0.5 batch（fp16）
    - 分辨率调整：分辨率越高，batch 越小
    - patch_size 调整：patch 越大，batch 越小
    
    Args:
        patch_size: patch 大小
        model_name: 模型名称（用于日志）
    
    Returns:
        int: 计算得到的 batch_size
    """
    if not torch.cuda.is_available():
        print("CPU 模式，使用 batch_size=1")
        return 1
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    patch_volume = patch_size[0] * patch_size[1] * patch_size[2]
    
    # 基础 batch size 估算（基于显存）
    base_batch = max(1, int(total_memory * 0.4))
    
    # 分辨率调整因子（假设分辨率越高体积越大）
    resolution_factor = 1.0
    
    # patch 大小调整因子
    if patch_volume > 128**3:
        size_factor = 0.5
    elif patch_volume > 96**3:
        size_factor = 0.7
    elif patch_volume > 64**3:
        size_factor = 1.0
    else:
        size_factor = 1.5
    
    batch_size = max(1, int(base_batch * resolution_factor * size_factor))
    
    # 显存安全上限
    max_batch = int(total_memory * 0.5)
    batch_size = min(batch_size, max(1, max_batch))
    
    print(f"自动设置 batch_size: {batch_size} (GPU显存: {total_memory:.1f} GB, patch: {patch_size}, 模型: {model_name})")
    return batch_size


def get_train_dataloader(
    data_files,
    batch_size=None,
    num_workers=None,
    pixdim=(1.0, 1.0, 1.0),
    patch_size=(64, 64, 64),
    cache_dir=None,
    persistent_workers=True,
    prefetch_factor=4,
    dtype="float16",
    cache_format="numpy",
    model_name="unknown",
):
    """
    获取训练数据加载器

    Args:
        data_files: 数据文件列表
        batch_size: 批次大小，None 表示自动计算
        num_workers: 数据加载线程数，None 表示自动计算
        pixdim: 重采样分辨率
        patch_size: 裁剪patch大小
        cache_dir: 缓存目录
        persistent_workers: 是否保持worker进程
        prefetch_factor: 预取因子
        dtype: 数据类型，可选 "float16" 或 "float32"
        cache_format: 缓存格式，可选 "numpy" 或 "tensor"
        model_name: 模型名称（用于自动计算 batch_size）

    Returns:
        DataLoader: 训练数据加载器
    """
    if cache_dir and not data_files:
        data_files = _get_data_files_from_cache(cache_dir)

    # 自动计算 batch_size
    if batch_size is None:
        batch_size = _auto_batch_size(patch_size=patch_size, model_name=model_name)

    # 自动计算 num_workers
    if num_workers is None:
        num_workers = _auto_num_workers()

    pre_transforms = Compose(get_train_pre_transforms(pixdim=pixdim, dtype=dtype, cache_format=cache_format))
    aug_transforms = Compose(get_train_aug_transforms(patch_size=patch_size, cache_format=cache_format))

    if cache_dir:
        cached_ds = PersistentDataset(data=data_files, transform=pre_transforms, cache_dir=cache_dir)
        dataset = AugmentedDataset(cached_ds, aug_transforms)
    else:
        dataset = Dataset(data=data_files, transform=pre_transforms + aug_transforms)

    use_persistent = persistent_workers and num_workers > 0

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": use_persistent,
    }

    if use_persistent:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)


def get_val_dataloader(
    data_files,
    batch_size=1,
    num_workers=4,
    pixdim=(1.0, 1.0, 1.0),
    patch_size=(64, 64, 64),
    cache_dir=None,
):
    """
    获取验证数据加载器

    Args:
        data_files: 数据文件列表
        batch_size: 批次大小，None 表示自动计算
        num_workers: 数据加载线程数，None 表示自动计算
        pixdim: 重采样分辨率
        patch_size: 裁剪 patch 大小
        cache_dir: 缓存目录

    Returns:
        DataLoader: 验证数据加载器
    """
    if cache_dir and not data_files:
        data_files = _get_data_files_from_cache(cache_dir)
    
    # 自动计算 num_workers
    if num_workers is None:
        num_workers = _auto_num_workers()
    
    # 自动计算 batch_size（验证集通常用 1，可选自动）
    if batch_size is None:
        batch_size = 1

    # 两阶段变换：避免验证时重复加载原始 NIfTI 文件
    from .transforms import get_val_transforms
    from .dataset import AugmentedDataset
    
    # 阶段 1：预处理（缓存部分）- 不含 EnsureTyped
    val_pre_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CastToTyped(keys=["image"], dtype=np.float32),
        CastToTyped(keys=["label"], dtype=np.uint8),
    ]
    
    # 阶段 2：后处理（实时部分）- 仅转换为 Tensor
    val_post_transforms = Compose([EnsureTyped(keys=["image", "label"])])
    
    if cache_dir:
        # 使用缓存：从缓存读取预处理后的数据
        cached_ds = PersistentDataset(data=data_files, transform=Compose(val_pre_transforms), cache_dir=cache_dir)
        dataset = Dataset(data=cached_ds, transform=val_post_transforms)
    else:
        # 不使用缓存：实时应用所有变换
        val_transforms = Compose(get_val_transforms(pixdim=pixdim, patch_size=patch_size))
        dataset = Dataset(data=data_files, transform=val_transforms)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_test_dataloader(
    data_files,
    batch_size=1,
    num_workers=0,
    pixdim=(1.0, 1.0, 1.0),
    cache_dir=None,
):
    """
    获取测试数据加载器

    Args:
        data_files: 数据文件列表
        batch_size: 批次大小，None 表示自动计算
        num_workers: 数据加载线程数，None 表示自动计算
        pixdim: 重采样分辨率
        cache_dir: 缓存目录

    Returns:
        DataLoader: 测试数据加载器
    """
    # 自动计算 num_workers
    if num_workers is None:
        num_workers = _auto_num_workers()
    
    # 自动计算 batch_size（测试集通常用 1）
    if batch_size is None:
        batch_size = 1
    
    test_transforms = Compose(get_test_transforms(pixdim=pixdim))

    if cache_dir:
        dataset = PersistentDataset(data=data_files, transform=test_transforms, cache_dir=cache_dir)
    else:
        dataset = Dataset(data=data_files, transform=test_transforms)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def auto_roi_size_from_cache(cache_dir, pixdim=(1.0, 1.0, 1.0)):
    """
    从缓存目录读取一个样本，自动推断合适的 roi_size

    Args:
        cache_dir: 缓存目录
        pixdim: 重采样分辨率

    Returns:
        roi_size: tuple，符合滑动窗口要求的尺寸
    """
    data_files = _get_data_files_from_cache(cache_dir)
    if not data_files:
        raise ValueError(f"缓存目录为空: {cache_dir}")

    pre_transforms = Compose(get_val_transforms(pixdim=pixdim, patch_size=(128, 128, 128)))
    temp_ds = Dataset(data=data_files[:1], transform=pre_transforms)
    sample = temp_ds[0]
    image = sample["image"]

    if isinstance(image, torch.Tensor):
        shape = image.shape
    else:
        shape = image[0].shape if isinstance(image, (list, tuple)) else image.shape

    if len(shape) == 4:
        spatial_shape = shape[1:]
    else:
        spatial_shape = shape

    roi_size = tuple(max(64, s // 2) for s in spatial_shape)
    print(f"自动推断 roi_size: {roi_size} (基于缓存图像空间尺寸: {spatial_shape})")
    return roi_size


__all__ = [
    "get_data_list",
    "get_train_dataloader",
    "get_val_dataloader",
    "get_test_dataloader",
    "auto_roi_size_from_cache",
]