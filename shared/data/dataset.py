"""
数据集
分阶段数据集和普通数据集
"""

import torch
import numpy as np
from monai.data import CacheDataset, Dataset, PersistentDataset


class AugmentedDataset(torch.utils.data.Dataset):
    """
    分阶段数据集：从缓存读取预处理数据，实时应用增强

    参数说明：
    - cached_ds: 缓存数据集，从预计算缓存加载的原始数据
    - aug_transforms: 数据增强变换，实时应用于每个样本
    """

    def __init__(self, cached_ds, aug_transforms):
        self.cached_ds = cached_ds
        self.aug_transforms = aug_transforms

    def __len__(self):
        return len(self.cached_ds)

    def __getitem__(self, idx):
        data = self.cached_ds[idx]
        return self.aug_transforms(data)


__all__ = ["AugmentedDataset", "get_data_list"]