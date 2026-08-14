"""
病灶级别评估指标
逐个计算每个病灶的 Dice 和 HD95

参数说明：
- pred_mask: 预测分割掩码 (numpy array)
- true_mask: 真实分割掩码 (numpy array)
- spacing: 体素间距 (tuple)
- mask: 二值掩码图像
- num: 连通域数量

BraTS 2023 官方标准：
- 使用 26-连通性进行连通域分析
- Dilation 因子 = 3
- 体积阈值 = 50 mm³
- HD95 惩罚值 = 374.0
"""

import torch
import numpy as np
import cc3d
from monai.utils import optional_import

ndimage, has_ndimage = optional_import("scipy.ndimage")


def label_components(mask):
    """
    标记连通域（BraTS 2023 官方标准：26-连通性）

    Args:
        mask: 二值掩码图像 (numpy array)

    Returns:
        labeled: 标记后的图像，每个连通域有唯一的整数标签
        num: 连通域数量
    """
    mask = mask.astype(np.uint8)
    # BraTS 2023 官方使用 26-连通性
    labeled = cc3d.connected_components(mask, connectivity=26)
    num = int(np.max(labeled)) if np.max(labeled) > 0 else 0
    return labeled.astype(np.int32), num


def compute_hd95(pred_mask, true_mask, spacing):
    """
    计算 HD95 (95th percentile Hausdorff Distance)

    Args:
        pred_mask: 预测分割掩码 (numpy array)
        true_mask: 真实分割掩码 (numpy array)
        spacing: 体素间距 (tuple)

    Returns:
        HD95 值，如果结果为 NaN/Inf 则返回 0.0
    """
    from monai.metrics import HausdorffDistanceMetric

    pred_t = torch.from_numpy(pred_mask.astype(np.float32)[None, None])
    true_t = torch.from_numpy(true_mask.astype(np.float32)[None, None])

    metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean")
    metric(y_pred=pred_t, y=true_t, spacing=spacing)
    value = metric.aggregate().item()
    metric.reset()

    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(value)


def lesionwise_evaluation(pred_mask, true_mask, spacing, tissue_type='WT'):
    """
    病灶级别评估（BraTS 2023 官方标准）

    关键修正：WT/TC/ET 的 GT 合并应该独立进行
    - WT: 所有 label>0 的区域
    - TC: label 1 和 3 的区域
    - ET: label 3 的区域
    
    每个组织类型独立进行 dilation 和连通域分析

    Args:
        pred_mask: 预测分割掩码 (numpy array, 已提取对应组织)
        true_mask: 真实分割掩码 (numpy array, 已提取对应组织)
        spacing: 体素间距 (tuple)
        tissue_type: 组织类型 (WT/TC/ET)

    Returns:
        tuple: (dice_sum, hd_sum, gt_lesion_count, fp_count)
    """
    # BraTS-GLI 参数
    DILATION_FACTOR = 3
    LESION_VOLUME_THRESH = 50.0
    
    if pred_mask.sum() == 0 and true_mask.sum() == 0:
        return 1.0, 0.0, 0, 0
    
    # 1. 对 GT 进行 dilation
    import scipy.ndimage
    dilation_struct = scipy.ndimage.generate_binary_structure(3, 2)
    true_mask_dilated = scipy.ndimage.binary_dilation(
        true_mask.astype(np.uint8), 
        structure=dilation_struct, 
        iterations=DILATION_FACTOR
    ).astype(np.uint8)

    # 2. 连通域分析（每个组织类型独立进行）
    pred_cc, pred_count = label_components(pred_mask)
    true_cc, true_count = label_components(true_mask)
    dilated_cc, _ = label_components(true_mask_dilated)

    # 3. 合并 GT 病灶（在 dilation 区域内）- 官方方法
    gt_combined = get_GTseg_combinedByDilation(dilated_cc, true_cc)

    # 计算体素体积
    voxel_volume = spacing[0] * spacing[1] * spacing[2]

    # 4. 获取有效 GT 病灶（体积 > 阈值）
    gt_valid_comps = []
    for gid in range(1, np.max(gt_combined) + 1):
        gmask = gt_combined == gid
        gt_vol = np.sum(gmask) * voxel_volume
        if gt_vol > LESION_VOLUME_THRESH:
            gt_valid_comps.append(gid)

    num_gt = len(gt_valid_comps)
    
    if num_gt == 0 and pred_count == 0:
        return 1.0, 0.0, 0, 0

    # 5. 对每个 GT 病灶计算 Dice 和 HD95（官方方法）
    matched_pred = set()
    dice_sum = 0.0
    hd_sum = 0.0
    HD95_PENALTY = 374.0  # 未匹配 GT 的惩罚值

    for gid in gt_valid_comps:
        # 获取 GT 病灶
        gt_tmp = (gt_combined == gid)
        
        # 对 GT 病灶进行 dilation（官方方法）
        gt_tmp_dilation = scipy.ndimage.binary_dilation(
            gt_tmp, 
            structure=dilation_struct, 
            iterations=DILATION_FACTOR
        )
        
        # 找到与该 GT 膨胀区域重叠的所有预测病灶
        pred_tmp = pred_cc.copy()
        pred_tmp = pred_tmp * gt_tmp_dilation  # 只保留重叠区域
        intersecting_cc = np.unique(pred_tmp)
        intersecting_cc = intersecting_cc[intersecting_cc != 0]  # 去除背景
        
        # 记录匹配的预测病灶
        for cc in intersecting_cc:
            matched_pred.add(cc)
        
        # 将所有重叠的预测病灶合并为一个整体
        pred_merged = np.zeros_like(pred_tmp)
        pred_merged[np.isin(pred_tmp, intersecting_cc)] = 1
        
        # 计算 Dice 和 HD95
        dice_score = compute_dice(pred_merged, gt_tmp)
        
        # 如果没有匹配的 Pred，给 HD95 惩罚值
        if len(intersecting_cc) == 0:
            hd_score = HD95_PENALTY
        else:
            hd_score = compute_hd95(pred_merged, gt_tmp, spacing)
        
        dice_sum += dice_score
        hd_sum += hd_score

    # 6. 计算 FP（未匹配任何 GT 的预测病灶）- 官方方法：无体积阈值
    # 注意：官方代码对所有未匹配的 Pred 都计数为 FP，没有体积阈值过滤
    fp_count = 0
    for pid in range(1, pred_count + 1):
        if pid in matched_pred:
            continue
        
        # 官方方法：直接计数，不进行体积阈值检查
        fp_count += 1

    return dice_sum, hd_sum, num_gt, fp_count


def compute_dice(pred, gt):
    """计算 Dice 系数"""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return (2.0 * intersection) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 0.0


def get_GTseg_combinedByDilation(gt_dilated_cc_mat, gt_label_cc):
    """
    官方函数：合并 dilation 区域内的 GT 病灶
    """
    gt_seg_combinedByDilation_mat = np.zeros_like(gt_dilated_cc_mat)
    for comp in range(np.max(gt_dilated_cc_mat)):
        comp += 1
        gt_d_tmp = np.zeros_like(gt_dilated_cc_mat)
        gt_d_tmp[gt_dilated_cc_mat == comp] = 1
        gt_d_tmp = (gt_label_cc * gt_d_tmp)
        np.place(gt_d_tmp, gt_d_tmp > 0, comp)
        gt_seg_combinedByDilation_mat += gt_d_tmp
    return gt_seg_combinedByDilation_mat


__all__ = [
    "label_components",
    "compute_hd95",
    "lesionwise_evaluation",
]