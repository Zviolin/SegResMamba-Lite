"""
【LightSegMamba 训练框架 V3】

- 损失：DiceCELoss（共享模块）
- 优化器：AdamW（共享模块）
- 调度器：CosineAnnealingLR（共享模块）
- 可选权重滑动平均（SWA 等权；历史命名"EMA"实为 SWA，2026-10-05 命名修正）
- 验证：滑窗推理 + Dice(WT/TC/ET) + HD95(WT/TC/ET)
- 接口与 MambaUNetTrainer 完全一致（含 validate_verbose 逐病例评估）
"""

from __future__ import annotations
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

# 把 code/ 根加入 sys.path，方便 import shared.*
_VERSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE_ROOT = os.path.dirname(_VERSION_ROOT)
sys.path.insert(0, _CODE_ROOT)

from shared.losses import get_loss
from shared.optim import get_optimizer, get_scheduler
from shared.optim.swa import SWA
from shared.inference.sliding_window import sliding_window_inference
from shared.inference.postprocess import brats_label_mapping

# 全局 HD95 空边界满额惩罚（与 SegResMamba-Lite 主项目口径一致）
HD95_EMPTY_PENALTY = 374.0


def _is_empty_volume(x):
    """判断张量或张量列表是否全为 0（空体积）"""
    if isinstance(x, (list, tuple)):
        return all(_is_empty_volume(t) for t in x)
    return not bool(torch.any(x > 0))


def _safe_hd95(metric, pred, gt, spacing):
    """单病例 HD95 计算（空边界安全版，与主项目 2026-09-28 修复口径对齐）

    - 空-空（无病灶且未误检）：记 0（完美预测）
    - 一空一非空（整区漏检或假阳性）：记 374 满额惩罚（与 lesion-wise 口径一致）
    - 其余正常情形走 MONAI 计算（带 spacing，保持与本项目历史 CSV 数值口径一致），
      inf/nan 兜底为 374
    避免旧版对 inf 记 0 美化最差预测的问题
    """
    pred_empty = _is_empty_volume(pred)
    gt_empty = _is_empty_volume(gt)
    if pred_empty and gt_empty:
        return 0.0
    if pred_empty or gt_empty:
        return HD95_EMPTY_PENALTY
    metric.reset()
    # 2026-10-07 形状修复：(1, D, H, W) 4 维走 MONAI batch-first 路径时 dim1 被当 channel、
    # 空间维被判为 2，与 3 元组 spacing 长度冲突直接 ValueError（训练期验证全失败，
    # best_metric 永不更新导致 best_metric_model.pth 无法生成）。
    # 升为 (1, 1, D, H, W) 5 维 = (B=1, C=1, 空间 3 维)，与主项目 list-of-(1,D,H,W)
    # 的 channel-first 口径（_compute_list 按 C=1 + 空间 3 维计算 3D HD95）数值等价。
    if pred.ndim == 4:
        pred = pred.unsqueeze(0)
    if gt.ndim == 4:
        gt = gt.unsqueeze(0)
    metric(y_pred=pred, y=gt, spacing=spacing)
    val = metric.aggregate().item()
    metric.reset()
    if np.isnan(val) or np.isinf(val):
        return HD95_EMPTY_PENALTY
    return val


class LightSegMambaTrainer:
    """LightSegMamba 训练器（与 MambaUNetTrainer 接口完全一致）。"""

    def __init__(
        self,
        model,
        device: str = "cuda",
        loss_name: str = "DiceCELoss",
        optimizer_type: str = "AdamW",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        scheduler_type: str = "CosineAnnealingLR",
        max_epochs: int = 100,
        use_swa: bool = False,
    ):
        self.device = device
        self.max_epochs = max_epochs
        self.use_swa = use_swa

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
        self.scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

        if use_swa:
            # SWA 等权平均（修复：历史 shadow 式 EMA API 与 shared.optim 现实现不兼容，开 --ema 必崩；现为可用实现）
            self.swa = SWA(model)
        else:
            self.swa = None

    # ────────────────────────────────────────────────────────────────────────
    # 训练
    # ────────────────────────────────────────────────────────────────────────
    def train_step(self, inputs, labels):
        """单步训练：前向 → Loss → 反向 → 更新。"""
        labels = brats_label_mapping(labels, label_4_to_3=True)
        self.optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.swa:
            self.swa.update()

        return loss.item()

    def train_epoch(self, train_loader):
        """训练一个 epoch。"""
        self.model.train()
        epoch_loss = 0.0
        step = 0

        for batch_data in tqdm(train_loader, desc="Training"):
            step += 1
            inputs = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)
            loss = self.train_step(inputs, labels)
            epoch_loss += loss

        self.scheduler.step()
        return epoch_loss / max(step, 1)

    # ────────────────────────────────────────────────────────────────────────
    # 验证（汇总指标）
    # ────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def validate(self, val_loader, spacing=(1.0, 1.0, 1.0),
                 roi_size=(64, 64, 64), lesion_results_file=None):
        """验证：返回汇总 Dice/HD95（WT/TC/ET）。"""
        # 验证用模型：启用 SWA 时用平均权重模型，否则用原始模型
        eval_model = self.swa.averaged_model if self.swa else self.model
        eval_model.eval()

        dice_wt = DiceMetric(include_background=False, reduction="mean")
        dice_tc = DiceMetric(include_background=False, reduction="mean")
        dice_et = DiceMetric(include_background=False, reduction="mean")
        hd_wt = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        hd_tc = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        hd_et = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        # 逐病例 HD95 收集列表（空边界安全版聚合）
        hd95_vals_wt, hd95_vals_tc, hd95_vals_et = [], [], []

        for val_data in tqdm(val_loader, desc="Validating"):
            val_inputs = val_data["image"].to(self.device)
            val_labels = val_data["label"].to(self.device)
            val_labels = brats_label_mapping(val_labels, label_4_to_3=True)

            with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                val_outputs = sliding_window_inference(
                    eval_model, val_inputs, roi_size=roi_size,
                    sw_batch_size=4, overlap=0.5, mode="gaussian",
                )

            val_outputs_list = [AsDiscrete(argmax=True, to_onehot=4)(i)
                                for i in decollate_batch(val_outputs)]
            val_labels_list = [AsDiscrete(to_onehot=4)(i)
                               for i in decollate_batch(val_labels)]

            pred_wt = [(i[1] + i[2] + i[3] > 0).float().unsqueeze(0) for i in val_outputs_list]
            true_wt = [(i[1] + i[2] + i[3] > 0).float().unsqueeze(0) for i in val_labels_list]
            pred_tc = [(i[1] + i[3] > 0).float().unsqueeze(0) for i in val_outputs_list]
            true_tc = [(i[1] + i[3] > 0).float().unsqueeze(0) for i in val_labels_list]
            pred_et = [(i[3] > 0).float().unsqueeze(0) for i in val_outputs_list]
            true_et = [(i[3] > 0).float().unsqueeze(0) for i in val_labels_list]

            dice_wt(y_pred=pred_wt, y=true_wt)
            dice_tc(y_pred=pred_tc, y=true_tc)
            dice_et(y_pred=pred_et, y=true_et)
            # 逐病例安全计算 HD95：空-空=0、一空一非空=374 惩罚，防 inf 污染聚合均值
            for pw_v, tw_v in zip(pred_wt, true_wt):
                hd95_vals_wt.append(_safe_hd95(hd_wt, pw_v, tw_v, spacing))
            for pc_v, tc_v in zip(pred_tc, true_tc):
                hd95_vals_tc.append(_safe_hd95(hd_tc, pc_v, tc_v, spacing))
            for pe_v, te_v in zip(pred_et, true_et):
                hd95_vals_et.append(_safe_hd95(hd_et, pe_v, te_v, spacing))

        mwt = dice_wt.aggregate().item()
        mtc = dice_tc.aggregate().item()
        met = dice_et.aggregate().item()
        hwt = float(np.mean(hd95_vals_wt)) if hd95_vals_wt else 0.0
        htc = float(np.mean(hd95_vals_tc)) if hd95_vals_tc else 0.0
        het = float(np.mean(hd95_vals_et)) if hd95_vals_et else 0.0

        for m in (dice_wt, dice_tc, dice_et):
            m.reset()

        result = {
            "dice_wt": mwt, "dice_tc": mtc, "dice_et": met,
            "dice_avg": (mwt + mtc + met) / 3,
            "hd95_wt": hwt, "hd95_tc": htc, "hd95_et": het,
            "hd95_avg": (hwt + htc + het) / 3,
        }

        if lesion_results_file is not None:
            from shared.metrics.aggregate_lesion import aggregate_lesion_results
            result.update(aggregate_lesion_results(lesion_results_file))

        return result

    # ────────────────────────────────────────────────────────────────────────
    # 验证（逐病例 + 写 CSV，含 lesion-wise）
    # ────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def validate_verbose(
        self,
        val_loader,
        spacing=(1.0, 1.0, 1.0),
        roi_size=(64, 64, 64),
        results_file=None,
        lesion_results_file=None,
    ):
        """逐病例评估 + 写入 CSV（含 lesion-wise）。"""
        from shared.metrics.lesion_metric import lesionwise_evaluation

        # 验证用模型：启用 SWA 时用平均权重模型，否则用原始模型
        eval_model = self.swa.averaged_model if self.swa else self.model
        eval_model.eval()

        dice_wt = DiceMetric(include_background=False, reduction="mean")
        dice_tc = DiceMetric(include_background=False, reduction="mean")
        dice_et = DiceMetric(include_background=False, reduction="mean")
        hd_wt = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        hd_tc = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        hd_et = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        # 逐病例 HD95 收集列表（空边界安全版聚合，与 CSV 同口径同源）
        hd95_vals_wt, hd95_vals_tc, hd95_vals_et = [], [], []

        for val_data in tqdm(val_loader, desc="Validating"):
            val_inputs = val_data["image"].to(self.device)
            val_labels = val_data["label"].to(self.device)
            val_labels = brats_label_mapping(val_labels, label_4_to_3=True)
            case_id = val_data.get("id", ["unknown"])[0]

            with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                val_outputs = sliding_window_inference(
                    eval_model, val_inputs, roi_size=roi_size,
                    sw_batch_size=4, overlap=0.5, mode="gaussian",
                )

            val_outputs_list = [AsDiscrete(argmax=True, to_onehot=4)(i)
                                for i in decollate_batch(val_outputs)]
            val_labels_list = [AsDiscrete(to_onehot=4)(i)
                               for i in decollate_batch(val_labels)]

            pred_wt = [(i[1] + i[2] + i[3] > 0).float().unsqueeze(0) for i in val_outputs_list]
            true_wt = [(i[1] + i[2] + i[3] > 0).float().unsqueeze(0) for i in val_labels_list]
            pred_tc = [(i[1] + i[3] > 0).float().unsqueeze(0) for i in val_outputs_list]
            true_tc = [(i[1] + i[3] > 0).float().unsqueeze(0) for i in val_labels_list]
            pred_et = [(i[3] > 0).float().unsqueeze(0) for i in val_outputs_list]
            true_et = [(i[3] > 0).float().unsqueeze(0) for i in val_labels_list]

            dice_wt(y_pred=pred_wt, y=true_wt)
            dice_tc(y_pred=pred_tc, y=true_tc)
            dice_et(y_pred=pred_et, y=true_et)
            # 逐病例安全计算 HD95：空-空=0、一空一非空=374 惩罚，防 inf 污染聚合均值
            for pw_v, tw_v in zip(pred_wt, true_wt):
                hd95_vals_wt.append(_safe_hd95(hd_wt, pw_v, tw_v, spacing))
            for pc_v, tc_v in zip(pred_tc, true_tc):
                hd95_vals_tc.append(_safe_hd95(hd_tc, pc_v, tc_v, spacing))
            for pe_v, te_v in zip(pred_et, true_et):
                hd95_vals_et.append(_safe_hd95(hd_et, pe_v, te_v, spacing))

            # 逐病例写 CSV
            if results_file is not None:
                temp_dice = DiceMetric(include_background=False, reduction="mean")

                temp_dice(y_pred=pred_wt, y=true_wt); d_wt = temp_dice.aggregate().item(); temp_dice.reset()
                temp_dice(y_pred=pred_tc, y=true_tc); d_tc = temp_dice.aggregate().item(); temp_dice.reset()
                temp_dice(y_pred=pred_et, y=true_et); d_et = temp_dice.aggregate().item(); temp_dice.reset()
                # 与汇总列表同源，保证日志汇总行与 CSV 逐位一致
                h_wt = hd95_vals_wt[-1]
                h_tc = hd95_vals_tc[-1]
                h_et = hd95_vals_et[-1]

                # NaN/Inf 处理（与 MambaUNet 一致）
                d_wt = 1.0 if np.isnan(d_wt) or np.isinf(d_wt) else d_wt
                d_tc = 1.0 if np.isnan(d_tc) or np.isinf(d_tc) else d_tc
                d_et = 1.0 if np.isnan(d_et) or np.isinf(d_et) else d_et

                with open(results_file, "a") as f:
                    f.write(f"{case_id},{d_wt:.4f},{d_tc:.4f},{d_et:.4f},{h_wt:.4f},{h_tc:.4f},{h_et:.4f}\n")

            # 逐病例 lesion-wise
            if lesion_results_file is not None:
                pw = pred_wt[0].detach().cpu().numpy().astype(np.uint8)
                if pw.ndim == 4: pw = pw.squeeze(0)
                tw = true_wt[0].detach().cpu().numpy().astype(np.uint8)
                if tw.ndim == 4: tw = tw.squeeze(0)
                pt = pred_tc[0].detach().cpu().numpy().astype(np.uint8)
                if pt.ndim == 4: pt = pt.squeeze(0)
                tt = true_tc[0].detach().cpu().numpy().astype(np.uint8)
                if tt.ndim == 4: tt = tt.squeeze(0)
                pe = pred_et[0].detach().cpu().numpy().astype(np.uint8)
                if pe.ndim == 4: pe = pe.squeeze(0)
                te = true_et[0].detach().cpu().numpy().astype(np.uint8)
                if te.ndim == 4: te = te.squeeze(0)

                # BraTS 2023 官方标准：返回 (dice_sum, hd_sum, gt_lesion_count, fp_count)
                ldw_sum, lhw_sum, ldw_gt, ldw_fp = lesionwise_evaluation(pw, tw, spacing)
                ldt_sum, lht_sum, ldt_gt, ldt_fp = lesionwise_evaluation(pt, tt, spacing)
                lde_sum, lhe_sum, lde_gt, lde_fp = lesionwise_evaluation(pe, te, spacing)

                # lesion_wise_dice = sum(dice) / (num_gt_lesions + num_fp)
                # lesion_wise_hd95 = sum(hd) / num_gt_lesions
                if ldw_gt + ldw_fp > 0:
                    ldw = ldw_sum / (ldw_gt + ldw_fp)
                    lhw = lhw_sum / ldw_gt if ldw_gt > 0 else 0.0
                else:
                    ldw, lhw = 1.0, 0.0
                if ldt_gt + ldt_fp > 0:
                    ldt = ldt_sum / (ldt_gt + ldt_fp)
                    lht = lht_sum / ldt_gt if ldt_gt > 0 else 0.0
                else:
                    ldt, lht = 1.0, 0.0
                if lde_gt + lde_fp > 0:
                    lde = lde_sum / (lde_gt + lde_fp)
                    lhe = lhe_sum / lde_gt if lde_gt > 0 else 0.0
                else:
                    lde, lhe = 1.0, 0.0

                with open(lesion_results_file, "a") as f:
                    f.write(
                        f"{case_id},{ldw:.4f},{ldt:.4f},{lde:.4f},"
                        f"{lhw:.4f},{lht:.4f},{lhe:.4f},"
                        f"{ldw_gt},{ldw_fp},{ldt_gt},{ldt_fp},{lde_gt},{lde_fp}\n"
                    )

        mwt = dice_wt.aggregate().item()
        mtc = dice_tc.aggregate().item()
        met = dice_et.aggregate().item()
        hwt = float(np.mean(hd95_vals_wt)) if hd95_vals_wt else 0.0
        htc = float(np.mean(hd95_vals_tc)) if hd95_vals_tc else 0.0
        het = float(np.mean(hd95_vals_et)) if hd95_vals_et else 0.0

        for m in (dice_wt, dice_tc, dice_et):
            m.reset()

        result = {
            "dice_wt": mwt, "dice_tc": mtc, "dice_et": met,
            "dice_avg": (mwt + mtc + met) / 3,
            "hd95_wt": hwt, "hd95_tc": htc, "hd95_et": het,
            "hd95_avg": (hwt + htc + het) / 3,
        }
        if lesion_results_file is not None:
            from shared.metrics.aggregate_lesion import aggregate_lesion_results
            result.update(aggregate_lesion_results(lesion_results_file))
        return result
