"""
数据处理模块
统一的数据加载、预处理、数据增强

参数说明：
- data_dir: 数据目录路径
- mode: 数据模式 ("train" / "val" / "test")
- resolution: 数据分辨率
- num_workers: 数据加载线程数
- use_disk_cache: 是否使用磁盘缓存
- transform: 数据增强变换（可选）
"""

from .transforms import (
    get_train_pre_transforms,
    get_train_aug_transforms,
    get_val_transforms,
    get_test_transforms,
)
from .dataset import AugmentedDataset
from .dataloader import (
    get_data_list,
    get_train_dataloader,
    get_val_dataloader,
    get_test_dataloader,
)


__all__ = [
    "get_train_pre_transforms",
    "get_train_aug_transforms",
    "get_val_transforms",
    "get_test_transforms",
    "AugmentedDataset",
    "get_data_list",
    "get_train_dataloader",
    "get_val_dataloader",
    "get_test_dataloader",
]