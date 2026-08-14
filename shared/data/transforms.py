"""
数据变换
训练和验证的数据预处理与增强

参数说明：
- pixdim: 重采样分辨率 (tuple)
- patch_size: 裁剪patch大小 (tuple)
- aug_transforms: 数据增强变换
"""

import numpy as np
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandZoomd,
    EnsureTyped,
    CastToTyped,
    SpatialPadd,
    Compose,
    Transform,
)


class CleanExtraFieldsd(Transform):
    """
    清理 CropForegroundd 等生成的额外字段，避免缓存兼容性问题
    
    只保留必要字段：image, label, id
    """
    def __init__(self, keys_to_keep=["image", "label", "id"]):
        super().__init__()
        self.keys_to_keep = keys_to_keep
    
    def __call__(self, data):
        keys_to_remove = [k for k in data if k not in self.keys_to_keep]
        for k in keys_to_remove:
            data.pop(k, None)
        return data


def get_train_pre_transforms(pixdim=(1.0, 1.0, 1.0), dtype="float16", cache_format="numpy", lazy=True):
    """
    训练预处理变换（缓存部分）

    Args:
        pixdim: 重采样体素间距
        dtype: 数据类型，可选 "float16" 或 "float32"
        cache_format: 缓存格式，可选 "numpy" 或 "tensor"
        lazy: 是否启用延迟重采样 (MONAI 1.2+) - 默认启用，减少插值损失

    Returns:
        Compose 变换链
    """
    dtype_map = {
        "float16": np.float16,
        "float32": np.float32,
    }
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest"), lazy=lazy),
        CropForegroundd(keys=["image", "label"], source_key="image", lazy=lazy),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CastToTyped(keys=["image"], dtype=dtype_map[dtype]),
        CastToTyped(keys=["label"], dtype=np.uint8),
        CleanExtraFieldsd(),
    ]
    
    if cache_format == "tensor":
        transforms.append(EnsureTyped(keys=["image", "label"]))
    
    return transforms


def get_train_aug_transforms(patch_size=(64, 64, 64), cache_format="numpy"):
    """
    训练数据增强变换（实时部分）

    Args:
        patch_size: 随机裁剪的patch大小
        cache_format: 缓存格式，可选 "numpy" 或 "tensor"

    Returns:
        Compose变换链
    """
    transforms = []
    
    if cache_format == "numpy":
        transforms.append(EnsureTyped(keys=["image", "label"]))  # 缓存是 NumPy，训练时转 Tensor
    
    transforms.extend([
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
        RandAdjustContrastd(keys=["image"], prob=0.1, gamma=(0.5, 4.5)),
        RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.9, max_zoom=1.1, mode=("bilinear", "nearest")),
        CastToTyped(keys=["image"], dtype=np.float32),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    return transforms


def get_val_transforms(pixdim=(1.0, 1.0, 1.0), patch_size=(64, 64, 64), lazy=True):
    """
    验证/测试变换

    Args:
        pixdim: 重采样体素间距
        patch_size: 空间填充大小
        lazy: 是否启用延迟重采样

    Returns:
        Compose变换链
    """
    return [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest"), lazy=lazy),
        CropForegroundd(keys=["image", "label"], source_key="image", lazy=lazy),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CastToTyped(keys=["image"], dtype=np.float32),
        CastToTyped(keys=["label"], dtype=np.uint8),
        EnsureTyped(keys=["image", "label"]),
    ]


def get_test_transforms(pixdim=(1.0, 1.0, 1.0), lazy=True):
    """
    仅图像的测试变换

    Args:
        pixdim: 重采样体素间距
        lazy: 是否启用延迟重采样

    Returns:
        Compose变换链
    """
    return [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
        Spacingd(keys=["image"], pixdim=pixdim, mode="bilinear", lazy=lazy),
        CropForegroundd(keys=["image"], source_key="image", lazy=lazy),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CastToTyped(keys=["image"], dtype=np.float32),
        EnsureTyped(keys=["image"]),
    ]


__all__ = [
    "CleanExtraFieldsd",
    "get_train_pre_transforms",
    "get_train_aug_transforms",
    "get_val_transforms",
    "get_test_transforms",
]