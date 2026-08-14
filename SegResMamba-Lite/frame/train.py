"""
【Optimized 训练框架】
框架：预测+Loss+反向+更新
支持深层监督和 EMA（基于 PyTorch AveragedModel）
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
from shared.optim.ema import EMA
from shared.inference.sliding_window import sliding_window_inference
from shared.inference.postprocess import brats_label_mapping


class OptimizedTrainer:
    """Optimized 训练器，支持深层监督 + 异步 GPU 传输"""

    def __init__(
        self,
        model,
        device="cuda",
        loss_name="DiceFocalLoss",
        optimizer_type="AdamW",
        lr=1e-4,
        weight_decay=1e-5,
        scheduler_type="CosineAnnealingLR",
        max_epochs=150,
        use_ema=True,
        ema_decay=0.999,
        use_deep_supervision=False,
        non_blocking=True,
    ):
        self.device = device
        self.max_epochs = max_epochs
        self.use_ema = use_ema
        self.use_deep_supervision = use_deep_supervision
        self.non_blocking = non_blocking if device == "cuda" else False

        self.model = model.to(device)
        self.loss_fn = get_loss(loss_name)

        # 区分 alpha_raw / temperature 与其他参数（用 10x 学习率）
        special_params = []  # alpha_raw 和 temperature
        other_params = []
        for name, p in self.model.named_parameters():
            if 'alpha_raw' in name or 'temperature' in name:
                special_params.append(p)
            else:
                other_params.append(p)

        # alpha_raw / temperature 用 10x 学习率（适中的加速，避免极端）
        # 直接用 torch.optim.AdamW 避免 scaler.step() 不兼容问题
        if special_params:
            self.optimizer = torch.optim.AdamW(
                [
                    {'params': other_params, 'lr': lr},
                    {'params': special_params, 'lr': lr * 10},  # 10x 学习率
                ],
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
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

        if use_ema:
            # 自动检测需要排除 EMA 的参数（alpha_raw、temperature）
            exclude_param_names = []
            for name, _ in model.named_parameters():
                if 'alpha_raw' in name or 'temperature' in name:
                    exclude_param_names.append(name)
            self.ema = EMA(model, decay=ema_decay, exclude_param_names=exclude_param_names)
        else:
            self.ema = None

    def compute_loss(self, outputs, labels):
        """计算损失，支持深层监督"""
        if isinstance(outputs, (list, tuple)) and self.use_deep_supervision:
            loss = 0
            ds_weights = [1.0, 0.5, 0.25, 0.125]
            for i, output in enumerate(outputs):
                if i == 0:
                    target = labels
                else:
                    scale_factor = output.shape[2] / labels.shape[2]
                    if scale_factor != 1.0:
                        target = torch.nn.functional.interpolate(
                            labels.float(), scale_factor=scale_factor, mode="nearest"
                        )
                    else:
                        target = labels
                curr_weight = ds_weights[i] if i < len(ds_weights) else 0.1
                loss += curr_weight * self.loss_fn(output, target)
            return loss
        return self.loss_fn(outputs, labels)

    def train_step(self, inputs, labels):
        """单步训练"""
        labels = brats_label_mapping(labels, label_4_to_3=True)
        self.optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
            outputs = self.model(inputs)
            loss = self.compute_loss(outputs, labels)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.ema:
            self.ema.update()

        return loss.item()

    def train_epoch(self, train_loader):
        """训练一个 epoch (异步 GPU 传输)"""
        self.model.train()
        epoch_loss = 0
        step = 0
        nb = self.non_blocking

        for batch_data in tqdm(train_loader, desc="Training"):
            step += 1
            inputs = batch_data["image"].to(self.device, non_blocking=nb)
            labels = batch_data["label"].to(self.device, non_blocking=nb)
            loss = self.train_step(inputs, labels)
            epoch_loss += loss

        self.scheduler.step()
        return epoch_loss / step

    def _get_eval_model(self):
        """获取验证用的模型（原始模型或 EMA 模型）"""
        if self.ema is not None:
            return self.ema.ema_model
        return self.model

    @torch.no_grad()
    def validate(self, val_loader, spacing=(1.0, 1.0, 1.0), roi_size=(64, 64, 64), lesion_results_file=None):
        """验证 (异步 GPU 传输)"""
        eval_model = self._get_eval_model()
        eval_model.eval()
        nb = self.non_blocking

        dice_metric_wt = DiceMetric(include_background=False, reduction="mean")
        dice_metric_tc = DiceMetric(include_background=False, reduction="mean")
        dice_metric_et = DiceMetric(include_background=False, reduction="mean")
        hd95_metric_wt = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        hd95_metric_tc = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        hd95_metric_et = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

        for val_data in tqdm(val_loader, desc="Validating"):
            val_inputs = val_data["image"].to(self.device, non_blocking=nb)
            val_labels = val_data["label"].to(self.device, non_blocking=nb)
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
        """逐病例评估 + 写入 CSV（异步 GPU 传输）"""
        from shared.metrics.lesion_metric import lesionwise_evaluation

        eval_model = self._get_eval_model()
        eval_model.eval()
        nb = self.non_blocking

        dice_metric_wt = DiceMetric(include_background=False, reduction="mean")
        dice_metric_tc = DiceMetric(include_background=False, reduction="mean")
        dice_metric_et = DiceMetric(include_background=False, reduction="mean")
        hd95_metric_wt = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        hd95_metric_tc = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        hd95_metric_et = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

        for val_data in tqdm(val_loader, desc="Validating"):
            val_inputs = val_data["image"].to(self.device, non_blocking=nb)
            val_labels = val_data["label"].to(self.device, non_blocking=nb)
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

                temp_hd95(y_pred=pred_wt, y=true_wt, spacing=spacing)
                h_wt = temp_hd95.aggregate().item()
                temp_hd95.reset()
                temp_hd95(y_pred=pred_tc, y=true_tc, spacing=spacing)
                h_tc = temp_hd95.aggregate().item()
                temp_hd95.reset()
                temp_hd95(y_pred=pred_et, y=true_et, spacing=spacing)
                h_et = temp_hd95.aggregate().item()
                temp_hd95.reset()

                d_wt = 1.0 if np.isnan(d_wt) or np.isinf(d_wt) else d_wt
                d_tc = 1.0 if np.isnan(d_tc) or np.isinf(d_tc) else d_tc
                d_et = 1.0 if np.isnan(d_et) or np.isinf(d_et) else d_et
                h_wt = 0.0 if np.isnan(h_wt) or np.isinf(h_wt) else h_wt
                h_tc = 0.0 if np.isnan(h_tc) or np.isinf(h_tc) else h_tc
                h_et = 0.0 if np.isnan(h_et) or np.isinf(h_et) else h_et

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

                # 按照官方公式计算 lesion-wise 指标
                # lesion_wise_dice = sum(dice) / (num_gt_lesions + num_fp)
                # lesion_wise_hd95 = sum(hd) / num_gt_lesions
                # 注意：FP 惩罚只应用于 Dice，HD95 不需要额外惩罚（表面距离已自然惩罚 FP）
                
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
