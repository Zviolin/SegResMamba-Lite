"""
病灶评估结果汇总（本项目口径，2026-10-05 定案；引用时需注明口径）
从 CSV 文件读取并汇总病灶级别的评估结果

本项目口径（与 BraTS 2023 官方公式的差异见下）：
- 对每个病例计算：lesion_dice = sum(dice) / (num_gt_lesions + num_fp)（与官方一致）
- 对每个病例计算：lesion_hd95 = sum(hd) / num_gt_lesions
  （FN 病灶的 374 mm 满罚已在 hd_sum 内；官方口径分母还应含 FP——本项目为
   FP 惩罚只作用于 Dice，HD95 分母仅为 GT 病灶数，各模型版本统一采用本口径）
- 全局汇总：对所有病例的 lesion-wise 分数取算术平均

参数说明：
- lesion_results_file: 病灶评估结果 CSV 文件路径
"""

import numpy as np


def aggregate_lesion_results(lesion_results_file):
    """
    从 CSV 文件读取 lesion-wise 结果并汇总（本项目口径，2026-10-05 定案）

    CSV 文件格式：
    case_id,dice_wt,dice_tc,dice_et,hd95_wt,hd95_tc,hd95_et,gt_wt,fp_wt,gt_tc,fp_tc,gt_et,fp_et

    Args:
        lesion_results_file: 病灶评估结果 CSV 文件路径

    Returns:
        dict: 包含以下键的汇总结果
            - lw_dice_wt: Whole Tumor lesion-wise Dice
            - lw_dice_tc: Tumor Core lesion-wise Dice
            - lw_dice_et: Enhancing Tumor lesion-wise Dice
            - lw_hd95_wt: Whole Tumor lesion-wise HD95
            - lw_hd95_tc: Tumor Core lesion-wise HD95
            - lw_hd95_et: Enhancing Tumor lesion-wise HD95
    """
    # 本项目口径：全局 lesion-wise 指标 = 对所有病例的 lesion-wise 分数取算术平均
    # 每个病例的分数已经是：sum(dice) / (num_gt + num_fp)
    # 所以全局分数 = mean(case_lesion_dice for all cases)
    
    case_count = 0
    dice_wt_sum, dice_tc_sum, dice_et_sum = 0.0, 0.0, 0.0
    hd95_wt_sum, hd95_tc_sum, hd95_et_sum = 0.0, 0.0, 0.0
    
    with open(lesion_results_file, "r") as f:
        lines = f.readlines()[1:]  # 跳过表头
        case_count = len(lines)
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) >= 7:
                # CSV 中的值已经是该病例的 lesion-wise 分数（包含 FP 惩罚）
                dice_wt_sum += float(parts[1])
                dice_tc_sum += float(parts[2])
                dice_et_sum += float(parts[3])
                hd95_wt_sum += float(parts[4])
                hd95_tc_sum += float(parts[5])
                hd95_et_sum += float(parts[6])
    
    # 全局 lesion-wise 分数 = 所有病例分数的算术平均（口径定案见文件头说明）
    return {
        "lw_dice_wt": float(dice_wt_sum / case_count) if case_count > 0 else 0.0,
        "lw_dice_tc": float(dice_tc_sum / case_count) if case_count > 0 else 0.0,
        "lw_dice_et": float(dice_et_sum / case_count) if case_count > 0 else 0.0,
        "lw_hd95_wt": float(hd95_wt_sum / case_count) if case_count > 0 else 0.0,
        "lw_hd95_tc": float(hd95_tc_sum / case_count) if case_count > 0 else 0.0,
        "lw_hd95_et": float(hd95_et_sum / case_count) if case_count > 0 else 0.0,
    }