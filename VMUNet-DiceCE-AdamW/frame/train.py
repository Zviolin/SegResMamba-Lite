"""
【UltraLight 训练框架】
框架：预测+Loss+反向+更新
轻量级设计
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.losses import get_loss
from shared.optim import get_optimizer, get_scheduler
from shared.optim.swa import SWA
from shared.inference.sliding_window import sliding_window_inference
from shared.inference.postprocess import brats_label_mapping

import os

# 空边界 HD95 满额惩罚距离（mm），与 lesion-wise 口径保持一致
HD95_EMPTY_PENALTY = 374.0


def _is_empty_volume(x):
    """判断张量或张量列表是否全为 0（空体积）"""
    if isinstance(x, (list, tuple)):
        return all(_is_empty_volume(t) for t in x)
    return not bool(torch.any(x > 0))


def _safe_hd95(metric, pred, gt, spacing):
    """单病例 HD95 计算（空边界安全版）

    - 空-空（无病灶且未误检）：记 0（完美预测）
    - 一空一非空（整区漏检或假阳性）：记 374 满额惩罚（与 lesion-wise 口径一致）
    - 其余正常情形走 MONAI 计算，inf/nan 兜底为 374
    避免旧版对 inf 记 0 美化最差预测的问题；spacing 沿用基线历史口径（mm 单位）
    """
    pred_empty = _is_empty_volume(pred)
    gt_empty = _is_empty_volume(gt)
    if pred_empty and gt_empty:
        return 0.0
    if pred_empty or gt_empty:
        return HD95_EMPTY_PENALTY
    metric.reset()
    metric(y_pred=pred, y=gt, spacing=spacing)
    val = metric.aggregate().item()
    metric.reset()
    if np.isnan(val) or np.isinf(val):
        return HD95_EMPTY_PENALTY
    return val


class UltraLightTrainer:
    """UltraLight 训练器"""

    def __init__(
        self,
        model,
        device="cuda",
        loss_name="DiceCELoss",
        optimizer_type="AdamW",
        lr=1e-4,
        weight_decay=1e-5,
        scheduler_type="CosineAnnealingLR",
        max_epochs=100,
        use_swa=False,
        skip_nonfinite=False,  # 跳过非有限 loss（NaN/Inf）的 batch：不 backward/不更新/不进平均权重（默认关闭，保持历史行为）
    ):
        self.device = device
        self.max_epochs = max_epochs
        self.use_swa = use_swa
        self.skip_nonfinite = skip_nonfinite

        self.model = model.to(device)
        self.loss_fn = get_loss(loss_name)
        self.optimizer = get_optimizer(
            self.model.parameters(),
            optimizer_type=optimizer_type,
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_type=scheduler_type,
            T_max=max_epochs,
        )
        self.scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

        if use_swa:
            # SWA 等权平均（历史 shadow 式 EMA 代码从未启用，此处为新实现）
            self.swa = SWA(model)
        else:
            self.swa = None

    def train_step(self, inputs, labels):
        """单步训练"""
        labels = brats_label_mapping(labels, label_4_to_3=True)
        self.optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)

        # 🛡️ 非有限损失防护（--skip_nonfinite 开启时）：NaN/Inf 一旦 backward
        # 会污染 optimizer 与平均权重，后续权重全部 NaN 再也回不来，必须直接跳过该 batch
        if self.skip_nonfinite and not torch.isfinite(loss):
            # 跳过 backward/optimizer/平均权重，返回 (0.0, False) 让 epoch 累加跳过
            return 0.0, False

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.swa:
            self.swa.update()

        return loss.item(), True

    def train_epoch(self, train_loader):
        """训练一个 epoch"""
        self.model.train()
        epoch_loss = 0
        valid_steps = 0
        skipped_steps = 0
        step = 0

        for batch_data in tqdm(train_loader, desc="Training"):
            step += 1
            inputs = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)
            loss, is_valid = self.train_step(inputs, labels)
            if is_valid:
                epoch_loss += loss
                valid_steps += 1
            else:
                skipped_steps += 1

        if skipped_steps > 0:
            print(f"[WARN] 本 epoch 跳过 {skipped_steps}/{step} 个 batch（非有限 loss）")

        self.scheduler.step()
        # 用有效 step 数做平均；若全部跳过，返回 0 不除零
        # （默认关闭时 skipped 恒为 0，valid_steps == step，与历史行为逐位一致）
        return (epoch_loss / valid_steps) if valid_steps > 0 else 0.0

    @torch.no_grad()
    def validate(self, val_loader, spacing=(1.0, 1.0, 1.0), roi_size=(64, 64, 64), lesion_results_file=None):
        """验证"""
        # 验证用模型：启用 SWA 时用平均权重模型，否则用原始模型
        eval_model = self.swa.averaged_model if self.swa else self.model
        eval_model.eval()

        dice_metric_wt = DiceMetric(include_background=False, reduction="mean")
        dice_metric_tc = DiceMetric(include_background=False, reduction="mean")
        dice_metric_et = DiceMetric(include_background=False, reduction="mean")
        hd95_metric_wt = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        hd95_metric_tc = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        hd95_metric_et = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

        for val_data in tqdm(val_loader, desc="Validating"):
            val_inputs = val_data["image"].to(self.device)
            val_labels = val_data["label"].to(self.device)
            val_labels = brats_label_mapping(val_labels, label_4_to_3=True)

            with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                val_outputs = sliding_window_inference(
                    eval_model, val_inputs, roi_size=roi_size, sw_batch_size=4, overlap=0.5, mode="gaussian"
                )

            val_outputs_list = [AsDiscrete(argmax=True, to_onehot=4)(i) for i in decollate_batch(val_outputs)]
            val_labels_list = [AsDiscrete(to_onehot=4)(i) for i in decollate_batch(val_labels)]

            pred_wt = [(i[1] + i[2] + i[3] > 0).float().unsqueeze(0) for i in val_outputs_list]
            true_wt = [(i[1] + i[2] + i[3] > 0).float().unsqueeze(0) for i in val_labels_list]

            pred_tc = [(i[1] + i[3] > 0).float().unsqueeze(0) for i in val_outputs_list]
            true_tc = [(i[1] + i[3] > 0).float().unsqueeze(0) for i in val_labels_list]

            pred_et = [(i[3] > 0).float().unsqueeze(0) for i in val_outputs_list]
            true_et = [(i[3] > 0).float().unsqueeze(0) for i in val_labels_list]

            dice_metric_wt(y_pred=pred_wt, y=true_wt)
            dice_metric_tc(y_pred=pred_tc, y=true_tc)
            dice_metric_et(y_pred=pred_et, y=true_et)
            hd95_metric_wt(y_pred=pred_wt, y=true_wt, spacing=spacing)
            hd95_metric_tc(y_pred=pred_tc, y=true_tc, spacing=spacing)
            hd95_metric_et(y_pred=pred_et, y=true_et, spacing=spacing)

        metric_wt = dice_metric_wt.aggregate().item()
        metric_tc = dice_metric_tc.aggregate().item()
        metric_et = dice_metric_et.aggregate().item()
        hd95_wt = hd95_metric_wt.aggregate().item()
        hd95_tc = hd95_metric_tc.aggregate().item()
        hd95_et = hd95_metric_et.aggregate().item()

        dice_metric_wt.reset()
        dice_metric_tc.reset()
        dice_metric_et.reset()
        hd95_metric_wt.reset()
        hd95_metric_tc.reset()
        hd95_metric_et.reset()

        result = {
            "dice_wt": metric_wt,
            "dice_tc": metric_tc,
            "dice_et": metric_et,
            "dice_avg": (metric_wt + metric_tc + metric_et) / 3,
            "hd95_wt": hd95_wt,
            "hd95_tc": hd95_tc,
            "hd95_et": hd95_et,
            "hd95_avg": (hd95_wt + hd95_tc + hd95_et) / 3,
        }

        if lesion_results_file is not None:
            from shared.metrics.aggregate_lesion import aggregate_lesion_results
            result.update(aggregate_lesion_results(lesion_results_file))

        return result

    @torch.no_grad()
    def validate_verbose(
        self,
        val_loader,
        spacing=(1.0, 1.0, 1.0),
        roi_size=(64, 64, 64),
        results_file=None,
        lesion_results_file=None,
    ):
        """逐病例评估 + 写入 CSV（含 lesion-wise）"""
        from shared.metrics.lesion_metric import lesionwise_evaluation

        # 验证用模型：启用 SWA 时用平均权重模型，否则用原始模型
        eval_model = self.swa.averaged_model if self.swa else self.model
        eval_model.eval()

        dice_metric_wt = DiceMetric(include_background=False, reduction="mean")
        dice_metric_tc = DiceMetric(include_background=False, reduction="mean")
        dice_metric_et = DiceMetric(include_background=False, reduction="mean")
        hd95_metric_wt = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        hd95_metric_tc = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        hd95_metric_et = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

        for val_data in tqdm(val_loader, desc="Validating"):
            val_inputs = val_data["image"].to(self.device)
            val_labels = val_data["label"].to(self.device)
            val_labels = brats_label_mapping(val_labels, label_4_to_3=True)
            case_id = val_data.get("id", ["unknown"])[0]

            with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                val_outputs = sliding_window_inference(
                    eval_model, val_inputs, roi_size=roi_size, sw_batch_size=4, overlap=0.5, mode="gaussian"
                )

            val_outputs_list = [AsDiscrete(argmax=True, to_onehot=4)(i) for i in decollate_batch(val_outputs)]
            val_labels_list = [AsDiscrete(to_onehot=4)(i) for i in decollate_batch(val_labels)]

            pred_wt = [(i[1] + i[2] + i[3] > 0).float().unsqueeze(0) for i in val_outputs_list]
            true_wt = [(i[1] + i[2] + i[3] > 0).float().unsqueeze(0) for i in val_labels_list]

            pred_tc = [(i[1] + i[3] > 0).float().unsqueeze(0) for i in val_outputs_list]
            true_tc = [(i[1] + i[3] > 0).float().unsqueeze(0) for i in val_labels_list]

            pred_et = [(i[3] > 0).float().unsqueeze(0) for i in val_outputs_list]
            true_et = [(i[3] > 0).float().unsqueeze(0) for i in val_labels_list]

            dice_metric_wt(y_pred=pred_wt, y=true_wt)
            dice_metric_tc(y_pred=pred_tc, y=true_tc)
            dice_metric_et(y_pred=pred_et, y=true_et)
            hd95_metric_wt(y_pred=pred_wt, y=true_wt, spacing=spacing)
            hd95_metric_tc(y_pred=pred_tc, y=true_tc, spacing=spacing)
            hd95_metric_et(y_pred=pred_et, y=true_et, spacing=spacing)

            if results_file is not None:
                temp_dice = DiceMetric(include_background=False, reduction="mean")
                temp_hd95 = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

                temp_dice(y_pred=pred_wt, y=true_wt)
                d_wt = temp_dice.aggregate().item()
                temp_dice.reset()
                temp_dice(y_pred=pred_tc, y=true_tc)
                d_tc = temp_dice.aggregate().item()
                temp_dice.reset()
                temp_dice(y_pred=pred_et, y=true_et)
                d_et = temp_dice.aggregate().item()
                temp_dice.reset()

                h_wt = _safe_hd95(temp_hd95, pred_wt, true_wt, spacing)
                h_tc = _safe_hd95(temp_hd95, pred_tc, true_tc, spacing)
                h_et = _safe_hd95(temp_hd95, pred_et, true_et, spacing)

                d_wt = 1.0 if np.isnan(d_wt) or np.isinf(d_wt) else d_wt
                d_tc = 1.0 if np.isnan(d_tc) or np.isinf(d_tc) else d_tc
                d_et = 1.0 if np.isnan(d_et) or np.isinf(d_et) else d_et

                with open(results_file, "a") as f:
                    f.write(f"{case_id},{d_wt:.4f},{d_tc:.4f},{d_et:.4f},{h_wt:.4f},{h_tc:.4f},{h_et:.4f}\n")

            if lesion_results_file is not None:
                pw = pred_wt[0].detach().cpu().numpy().astype(np.uint8)
                if pw.ndim == 4:
                    pw = pw.squeeze(0)
                tw = true_wt[0].detach().cpu().numpy().astype(np.uint8)
                if tw.ndim == 4:
                    tw = tw.squeeze(0)

                pt = pred_tc[0].detach().cpu().numpy().astype(np.uint8)
                if pt.ndim == 4:
                    pt = pt.squeeze(0)
                tt = true_tc[0].detach().cpu().numpy().astype(np.uint8)
                if tt.ndim == 4:
                    tt = tt.squeeze(0)

                pe = pred_et[0].detach().cpu().numpy().astype(np.uint8)
                if pe.ndim == 4:
                    pe = pe.squeeze(0)
                te = true_et[0].detach().cpu().numpy().astype(np.uint8)
                if te.ndim == 4:
                    te = te.squeeze(0)

                # BraTS 2023 官方标准：返回 (dice_sum, hd_sum, gt_lesion_count, fp_count)
                ldw_sum, lhw_sum, ldw_gt, ldw_fp = lesionwise_evaluation(pw, tw, spacing)
                ldt_sum, lht_sum, ldt_gt, ldt_fp = lesionwise_evaluation(pt, tt, spacing)
                lde_sum, lhe_sum, lde_gt, lde_fp = lesionwise_evaluation(pe, te, spacing)

                # 按照 BraTS 2023 官方公式计算 lesion-wise 指标
                # lesion_wise_dice = sum(dice) / (num_gt_lesions + num_fp)
                # lesion_wise_hd95 = (sum(hd) + num_fp * 374) / (num_gt_lesions + num_fp)
                # 注意：lesionwise_evaluation 函数中已经对未匹配的 GT 病灶给了 374 的 HD95 惩罚
                # 所以这里只需要按照官方公式汇总即可
                
                if ldw_gt + ldw_fp > 0:
                    ldw = ldw_sum / (ldw_gt + ldw_fp)
                    lhw = (lhw_sum + ldw_fp * 374.0) / (ldw_gt + ldw_fp)
                else:
                    ldw, lhw = 1.0, 0.0
                
                if ldt_gt + ldt_fp > 0:
                    ldt = ldt_sum / (ldt_gt + ldt_fp)
                    lht = (lht_sum + ldt_fp * 374.0) / (ldt_gt + ldt_fp)
                else:
                    ldt, lht = 1.0, 0.0
                
                if lde_gt + lde_fp > 0:
                    lde = lde_sum / (lde_gt + lde_fp)
                    lhe = (lhe_sum + lde_fp * 374.0) / (lde_gt + lde_fp)
                else:
                    lde, lhe = 1.0, 0.0

                with open(lesion_results_file, "a") as f:
                    f.write(f"{case_id},{ldw:.4f},{ldt:.4f},{lde:.4f},{lhw:.4f},{lht:.4f},{lhe:.4f},{ldw_gt},{ldw_fp},{ldt_gt},{ldt_fp},{lde_gt},{lde_fp}\n")

        metric_wt = dice_metric_wt.aggregate().item()
        metric_tc = dice_metric_tc.aggregate().item()
        metric_et = dice_metric_et.aggregate().item()
        hd95_wt = hd95_metric_wt.aggregate().item()
        hd95_tc = hd95_metric_tc.aggregate().item()
        hd95_et = hd95_metric_et.aggregate().item()

        dice_metric_wt.reset()
        dice_metric_tc.reset()
        dice_metric_et.reset()
        hd95_metric_wt.reset()
        hd95_metric_tc.reset()
        hd95_metric_et.reset()

        result = {
            "dice_wt": metric_wt,
            "dice_tc": metric_tc,
            "dice_et": metric_et,
            "dice_avg": (metric_wt + metric_tc + metric_et) / 3,
            "hd95_wt": hd95_wt,
            "hd95_tc": hd95_tc,
            "hd95_et": hd95_et,
            "hd95_avg": (hd95_wt + hd95_tc + hd95_et) / 3,
        }
        if lesion_results_file is not None:
            from shared.metrics.aggregate_lesion import aggregate_lesion_results
            result.update(aggregate_lesion_results(lesion_results_file))
        return result
