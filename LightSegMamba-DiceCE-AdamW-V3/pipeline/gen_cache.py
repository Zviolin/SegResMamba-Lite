"""
【LightSegMamba 缓存生成器（快照重建版）】

对应原 d:\\srtp 项目 pipeline/gen_cache.py 的重建实现，CLI 与 RUN_FULL_1251.md
记录的原始接口保持一致：

  # 全量 1251 例缓存（约 30-60 分钟，已存在样本自动跳过）
  python pipeline/gen_cache.py --resolution 2.0 --generate --cache

  # 200 例快速缓存
  python pipeline/gen_cache.py --resolution 2.0 --max-samples 200 --generate --cache

  # 更高精度 1.0mm 全量（约 2-4 小时）
  python pipeline/gen_cache.py --resolution 1.0 --generate --cache --workers 4

职责：
  1. 枚举原始 BraTS 2023 GLI 数据（默认指向本快照 Data 目录下的 TrainingData）
  2. --max-samples N：seed 固定采样（random.Random(seed).sample，可复现）
  3. 70:15:15 划分（seed 固定；全量 1251 → 875/188/188，200 例 → 140/30/30，
     与历史训练/评估日志中的测试集例数一致）
  4. --generate：写 split_info.json（键名 train_case_ids / val_case_ids /
     test_case_ids / seed，与 pipeline/train.py、pipeline/evaluate.py 的读取一致）
  5. --cache：用与 shared/data/dataloader.py 完全一致的预处理变换链预跑
     PersistentDataset（train/val/test 三个子目录），MONAI 对已缓存样本自动跳过
     （即"断点续传"）

缓存目录约定（与 pipeline/train.py 的 resolve_cache_dir 一致）：
  - 全量：{cache_parent}/persistent_cache_{resolution}/
  - 采样：{cache_parent}/persistent_cache_{resolution}_{N}/
  cache_parent 默认 {Code根}/../../Data/persistent_cache，
  可用 --cache-parent 或环境变量 SRTP_CACHE_PARENT 覆盖

注意：缓存 hash 由（样本 dict + 变换链）共同决定，本脚本与 dataloader 使用同一
get_data_list、同一 case_id 过滤、逐参数一致的变换链，因此本脚本写入的缓存可被
训练/评估直接命中；从零重建的划分与历史存档的 split_info.json 可能不同（划分算法
未随快照保留），严格复用历史划分请使用存档的 split_info.json。
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import argparse

_VERSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE_ROOT = os.path.dirname(_VERSION_ROOT)
sys.path.insert(0, _VERSION_ROOT)
sys.path.insert(0, _CODE_ROOT)

import numpy as np
from monai.data import PersistentDataset, DataLoader
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
)

from shared.data.dataloader import get_data_list
from shared.utils.paths import resolve_cache_parent as _shared_resolve_cache_parent

# 默认数据根：本快照内的 BraTS 2023 GLI 训练数据
# （与 shared/data/dataloader.py::_get_data_files_from_cache 中硬编码的路径一致）
_DEFAULT_DATA_ROOT = os.path.normpath(os.path.join(
    _CODE_ROOT, "..", "..", "Data", "Brain", "TCIA-BraTS", "DATA",
    "BraTS2023", "BraTS-GLI", "TrainingData",
    "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
))


# 默认缓存父目录（2026-10-07 定案：全库统一默认主项目体系缓存，与其他 6 项目一致；换机器用 SRTP_CACHE_PARENT 整机重定向）
_DEFAULT_CACHE_PARENT = r"D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache"


def resolve_cache_parent(cli_value: str | None) -> str:
    """
    解析缓存父目录

    优先级：--cache-parent > 环境变量 SRTP_CACHE_PARENT > 快照相对默认值
    （env 与默认值的统一解析委托给 shared.utils.paths.resolve_cache_parent）

    Args:
        cli_value: 命令行传入的 --cache-parent

    Returns:
        缓存父目录绝对路径
    """
    if cli_value:
        return os.path.normpath(cli_value)
    return _shared_resolve_cache_parent(_DEFAULT_CACHE_PARENT)


def resolve_cache_dir(cache_parent: str, resolution: float, max_samples: int | None) -> str:
    """
    解析完整缓存目录（与 pipeline/train.py、pipeline/evaluate.py 的命名约定一致）

    保留本地拼接实现：入参 cache_parent 已完成 CLI > env > 默认 的优先级解析，
    若改用 shared 版 resolve_cache_dir 会再次读取环境变量，破坏该优先级顺序。

    Args:
        cache_parent: 缓存父目录
        resolution: 重采样分辨率
        max_samples: 采样病例数（None 或 >= 1251 视为全量）

    Returns:
        缓存目录路径，如 {parent}/persistent_cache_2.0_200
    """
    if max_samples is None or max_samples >= 1251:
        return f"{cache_parent}_{resolution}"
    return f"{cache_parent}_{resolution}_{max_samples}"


def build_split(case_ids: list[str], max_samples: int | None, seed: int = 42):
    """
    采样 + 70:15:15 划分（seed 固定，可复现）

    全量 1251 → 875/188/188；200 例 → 140/30/30（与历史日志中的测试集例数一致）

    Args:
        case_ids: 全部病例 ID
        max_samples: 采样病例数（None 表示全量）
        seed: 随机种子（同时用于采样与划分打乱）

    Returns:
        (train_ids, val_ids, test_ids)
    """
    rng = random.Random(seed)
    ids = sorted(case_ids)
    if max_samples is not None and max_samples < len(ids):
        # 采样同样固定 seed，保证 --max-samples 选取病例可复现
        ids = sorted(rng.sample(ids, max_samples))
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * 0.7)
    n_val = int(round(n * 0.15))
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def build_val_pre_transforms(pixdim: tuple, patch_size: tuple) -> Compose:
    """
    构建验证/测试子集的缓存预处理链

    与 shared/data/dataloader.py::get_val_dataloader 中内联的 val_pre_transforms
    逐参数一致（Spacingd/CropForegroundd 隐式 lazy=False；EnsureTyped 属于阶段 2
    实时变换，不进缓存），保证缓存 hash 可被 dataloader 命中

    Args:
        pixdim: 重采样体素间距
        patch_size: SpatialPadd 填充尺寸（与训练 patch 一致）

    Returns:
        Compose 变换链
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CastToTyped(keys=["image"], dtype=np.float32),
        CastToTyped(keys=["label"], dtype=np.uint8),
    ])


def warm_cache(data_dicts: list[dict], transform: Compose, cache_dir: str, workers: int, tag: str):
    """
    预跑 PersistentDataset 触发缓存写入（已缓存样本由 MONAI 自动跳过）

    Args:
        data_dicts: 样本 dict 列表（与 dataloader 的构造完全一致）
        transform: 缓存预处理链
        cache_dir: PersistentDataset 缓存子目录（train/val/test）
        workers: 预跑线程数（0 表示单进程）
        tag: 进度显示标签
    """
    os.makedirs(cache_dir, exist_ok=True)
    ds = PersistentDataset(data=data_dicts, transform=transform, cache_dir=cache_dir)
    n = len(ds)
    print(f"  [{tag}] {n} 例 → {cache_dir}", flush=True)
    t0 = time.time()
    if workers > 0:
        loader = DataLoader(ds, batch_size=None, num_workers=workers)
        items = loader
    else:
        items = ds
    done = 0
    for _ in items:
        done += 1
        if done % 20 == 0 or done == n:
            print(f"  [{tag}] {done}/{n} ({time.time() - t0:.1f}s)", flush=True)
    print(f"  [{tag}] 完成，耗时 {time.time() - t0:.1f}s", flush=True)


def filter_by_ids(all_data: list[dict], case_ids: list[str]) -> list[dict]:
    """
    按 case_id 过滤样本 dict（与 dataloader._get_data_files_from_cache 的过滤方式一致）

    Args:
        all_data: get_data_list 产出的完整样本列表
        case_ids: 目标病例 ID

    Returns:
        过滤后的样本 dict 列表
    """
    id_set = set(case_ids)
    return [d for d in all_data if d["id"] in id_set]


def main():
    parser = argparse.ArgumentParser(description="生成 PersistentDataset 缓存与数据划分（快照重建版）")
    parser.add_argument("--resolution", type=float, default=2.0, help="重采样体素间距 mm（默认 2.0）")
    parser.add_argument("--max-samples", type=int, default=None, help="采样病例数（默认 None=全量 1251）")
    parser.add_argument("--data-root", default=_DEFAULT_DATA_ROOT,
                        help="原始 BraTS-GLI-* case 根目录（默认指向快照内 TrainingData）")
    parser.add_argument("--cache-parent", default=None,
                        help="缓存父目录（默认 {Code根}/../../Data/persistent_cache，可用环境变量 SRTP_CACHE_PARENT 覆盖）")
    parser.add_argument("--seed", type=int, default=42, help="采样与划分随机种子（默认 42）")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[64, 64, 64],
                        help="val 链 SpatialPadd 尺寸（与训练 patch 一致，默认 64 64 64）")
    parser.add_argument("--generate", action="store_true", help="枚举数据并生成划分（写 split_info.json）")
    parser.add_argument("--cache", action="store_true", help="预跑 PersistentDataset 写缓存")
    parser.add_argument("--force-split", action="store_true", help="split_info.json 已存在时强制覆盖")
    parser.add_argument("--workers", type=int, default=0, help="预跑线程数（默认 0 单进程）")
    args = parser.parse_args()

    pixdim = (args.resolution,) * 3
    patch_size = tuple(args.patch_size)

    # 1. 枚举原始数据（带 label 的完整 dict，与 dataloader 侧一致）
    if not os.path.isdir(args.data_root):
        print(f"数据根目录不存在: {args.data_root}")
        print("可通过 --data-root 指定原始 BraTS-GLI-* case 根目录")
        sys.exit(1)
    all_data = get_data_list(args.data_root, mode="train")
    if not all_data:
        print(f"未在 {args.data_root} 下找到 BraTS-GLI-* 病例")
        sys.exit(1)
    print(f"枚举到 {len(all_data)} 例（data_root: {args.data_root}）")

    cache_dir = resolve_cache_dir(resolve_cache_parent(args.cache_parent), args.resolution, args.max_samples)
    print(f"缓存目录: {cache_dir}")

    split_file = os.path.join(cache_dir, "split_info.json")

    # 2. 生成 / 读取划分
    if args.generate:
        if os.path.exists(split_file) and not args.force_split:
            with open(split_file, "r", encoding="utf-8") as f:
                split_info = json.load(f)
            print(f"split_info.json 已存在，跳过（--force-split 覆盖）；seed={split_info.get('seed')}")
        else:
            train_ids, val_ids, test_ids = build_split(
                [d["id"] for d in all_data], args.max_samples, seed=args.seed
            )
            split_info = {
                "seed": args.seed,
                "resolution": args.resolution,
                "max_samples": args.max_samples,
                "data_root": args.data_root,
                "train_case_ids": train_ids,
                "val_case_ids": val_ids,
                "test_case_ids": test_ids,
            }
            os.makedirs(cache_dir, exist_ok=True)
            with open(split_file, "w", encoding="utf-8") as f:
                json.dump(split_info, f, ensure_ascii=False, indent=2)
            print(f"划分完成并写入 {split_file}")
    else:
        if not os.path.exists(split_file):
            print(f"split_info.json 不存在: {split_file}")
            print("请先带 --generate 运行以生成划分")
            sys.exit(1)
        with open(split_file, "r", encoding="utf-8") as f:
            split_info = json.load(f)

    print(f"  train: {len(split_info['train_case_ids'])} 例, "
          f"val: {len(split_info['val_case_ids'])} 例, "
          f"test: {len(split_info['test_case_ids'])} 例 "
          f"(seed={split_info.get('seed', 'unknown')})")

    # 3. 预跑缓存（train 用训练预处理链，val/test 用验证/测试链，均与 dataloader 一致）
    if args.cache:
        from shared.data.transforms import get_train_pre_transforms, get_test_transforms

        subsets = [
            ("train", split_info["train_case_ids"], "train",
             Compose(get_train_pre_transforms(pixdim=pixdim, dtype="float16", cache_format="numpy"))),
            ("val", split_info["val_case_ids"], "val",
             build_val_pre_transforms(pixdim, patch_size)),
            ("test", split_info["test_case_ids"], "test",
             Compose(get_test_transforms(pixdim=pixdim))),
        ]
        for tag, ids, subdir, transform in subsets:
            data_dicts = filter_by_ids(all_data, ids)
            warm_cache(data_dicts, transform,
                       os.path.join(cache_dir, subdir), args.workers, tag)

    print("全部完成")


if __name__ == "__main__":
    main()
