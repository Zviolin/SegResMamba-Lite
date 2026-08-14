# SegResNet-DiceFocal-AdamW-EMA-DS - 优化版本

> SegResNet + DiceFocalLoss + EMA + 深层监督 (Deep Supervision)

---

## 一、模型概述

在 Baseline (SegResNet-DiceCE-AdamW) 基础上，引入三项优化技术：

| 优化项 | 技术 | 作用 |
|--------|------|------|
| **损失函数** | DiceFocalLoss | 关注难分类样本，处理类别不平衡 |
| **EMA** | 指数移动平均 | 平滑权重波动，提高稳定性 |
| **深层监督** | Deep Supervision | 多尺度损失，加速收敛 |

---

## 二、模型架构

使用 MONAI 的 **SegResNetDS** (带深层监督的 SegResNet)：

```
输入 [B, 4, 64, 64, 64]
    │
    ▼
┌─────────────────────────────────────────────────┐
│  编码器 (与 Baseline 相同)                        │
│  convInit → down1 → down2 → down3 → down4        │
│  [B,256,4,4,4]                                   │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  解码器 (转置卷积上采样)                           │
│                                                  │
│  up4: Deconv + skip3 + ResBlock → Head1 → DS_out │
│  up3: Deconv + skip2 + ResBlock → Head2 → DS_out │
│  up2: Deconv + skip1 + ResBlock → Head3 → DS_out │
│  up1: Deconv + ResBlock → Final_out              │
│                                                  │
│  深层监督输出:                                     │
│  DS_out × 3 (不同尺度预测)                         │
│  Final_out (最终预测)                              │
└─────────────────────────────────────────────────┘
```

### 与 Baseline 的区别

| 对比项 | Baseline | Optimized |
|--------|----------|-----------|
| **上采样方式** | 三线性插值 | 转置卷积 (可学习) |
| **Skip Connection** | 相加 | 相加 |
| **深层监督 Head** | 无 | 3 个输出 Head |
| **模型类** | SegResNet | SegResNetDS |

---

## 三、三大优化技术详解

### 3.1 DiceFocalLoss

```
DiceFocalLoss = Dice Loss + Focal Loss

Focal Loss: L_focal = -α × (1-p)^γ × log(p)
- γ=2.0: 降低易分类样本权重，聚焦难分类样本
- 适合 BraTS 中严重的类别不平衡 (背景 >> 肿瘤)

总损失: L = 1.0 × L_dice + 1.0 × L_focal
```

### 3.2 EMA (指数移动平均)

```
训练时维护两套权重:
- 模型权重 θ: 正常梯度更新
- EMA 权重 θ_ema = decay × θ_ema + (1-decay) × θ

验证时使用 EMA 权重 → 更稳定
decay = 0.999
```

### 3.3 Deep Supervision (深层监督)

```
在多个解码层输出预测并计算损失:

总损失 = 0.5 × Loss_level1 + 0.3 × Loss_level2 + 0.2 × Loss_level3

作用:
- 帮助梯度更好地回传到浅层
- 加速训练收敛
- 提高深层网络的可训练性
```

---

## 四、训练配置

| 参数 | 值 |
|------|------|
| **损失函数** | DiceFocalLoss |
| **优化器** | AdamW |
| **学习率** | 1e-4 |
| **权重衰减** | 1e-5 |
| **学习率调度** | CosineAnnealingLR |
| **最大 Epochs** | 150 |
| **批次大小** | 2 |
| **Num Workers** | 4 |
| **EMA** | 开启 (decay=0.999) |
| **深层监督** | 开启 (dsdepth=3) |
| **训练器** | OptimizedTrainer |

---

## 五、模型参数量

| 指标 | 值 |
|------|------|
| **总参数量** | ~4.70M |
| **init_filters** | 16 |
| **深层监督深度** | 3 |
| **Mamba** | 无 (纯 CNN) |

---

## 六、实验结果

| 指标 | WT | TC | ET |
|------|------|------|------|
| **全局 Dice** | 0.9211 | 0.8842 | 0.8488 |
| **全局 HD95 (mm)** | 5.45 | 4.51 | 4.00 |
| **LW Dice WT** | 0.7858 | 0.8292 | 0.7629 |

---

## 七、文件结构

```
SegResNet-DiceFocal-AdamW-EMA-DS/
├── README.md              # 本文件
├── models/
│   └── __init__.py        # SegResNetDS (MONAI) + get_model 接口
├── frame/
│   └── train.py           # OptimizedTrainer (支持 EMA + DS)
└── pipeline/
    └── train.py           # 训练入口
```

---

## 八、训练命令

```bash
cd SegResNet-DiceFocal-AdamW-EMA-DS
python -m pipeline.train --cache --epochs 150 --batch 2 --ema
```

---

## 九、与 Baseline 对比

| 对比项 | Baseline (SegResNet) | Optimized (SegResNetDS) |
|--------|---------------------|------------------------|
| **损失函数** | DiceCE | DiceFocal |
| **EMA** | 关闭 | 开启 |
| **深层监督** | 关闭 | 开启 |
| **训练 Epochs** | 100 | 150 |
| **全局 Dice WT** | 0.9203 | 0.9211 |
| **全局 Dice TC** | 0.8892 | 0.8842 |
| **全局 HD95 WT** | 5.17 | 5.45 |
| **结论** | HD95 更优 | Dice WT 略优 |

优化版本在 Dice 指标上略有提升，但 HD95 边界指标略有下降。说明 EMA 和深层监督主要帮助整体分割精度，对边界精度的改善有限。
