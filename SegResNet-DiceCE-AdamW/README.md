# SegResNet-DiceCE-AdamW - Baseline 基线版本

> NVIDIA BraTS 冠军模型 SegResNet 作为基线，使用 DiceCELoss + AdamW 优化器

---

## 一、模型来源

| 项目 | 内容 |
|------|------|
| **模型** | SegResNet (MONAI 内置) |
| **论文** | 3D MRI brain tumor segmentation using autoencoder regularization (BraTS 2018 冠军方案) |
| **框架** | MONAI (Medical Open Network for AI) |
| **定位** | CNN 基线模型 (无 Mamba) |

---

## 二、核心架构

```
输入 [B, 4, 64, 64, 64]
    │
    ▼
┌─────────────────────────────────────────────────┐
│  convInit: Conv3d(4→16) + GroupNorm + ReLU      │
│  输出: [B, 16, 64, 64, 64]                       │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  down1: ResBlock×1 + Conv3d(s=2)                │
│  输出: [B, 32, 32, 32, 32]  ← skip1             │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  down2: ResBlock×2 + Conv3d(s=2)                │
│  输出: [B, 64, 16, 16, 16]  ← skip2             │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  down3: ResBlock×2 + Conv3d(s=2)                │
│  输出: [B, 128, 8, 8, 8]  ← skip3               │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  down4: ResBlock×4                              │
│  输出: [B, 256, 4, 4, 4]                         │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  解码器 (三线性插值上采样)                         │
│                                                  │
│  up4: Trilinear + Conv1x1 + skip3 + ResBlock×2  │
│  up3: Trilinear + Conv1x1 + skip2 + ResBlock×2  │
│  up2: Trilinear + Conv1x1 + skip1 + ResBlock×2  │
│  up1: Trilinear + Conv1x1 + ResBlock×2          │
│      ↓                                           │
│  Output: Conv3d(1) → [B, 4, 64, 64, 64]         │
└─────────────────────────────────────────────────┘
```

### 关键特点

- **GroupNorm**: 对小批次训练更稳定
- **三线性插值上采样**: 无可学习参数，简单高效
- **相加 Skip Connection**: 通道数不变，信息压缩
- **ResBlock**: 标准残差连接 (双卷积)

---

## 三、训练配置

| 参数 | 值 |
|------|------|
| **损失函数** | DiceCELoss (Dice + CrossEntropy) |
| **优化器** | AdamW |
| **学习率** | 1e-4 |
| **权重衰减** | 1e-5 |
| **学习率调度** | CosineAnnealingLR |
| **最大 Epochs** | 100 |
| **批次大小** | 2 |
| **Num Workers** | 4 |
| **EMA** | 关闭 |
| **深层监督** | 关闭 |
| **训练器** | BaselineTrainer |

---

## 四、模型参数量

| 指标 | 值 |
|------|------|
| **总参数量** | ~4.70M |
| **init_filters** | 16 |
| **Mamba** | 无 (纯 CNN) |

---

## 五、实验结果

| 指标 | WT | TC | ET |
|------|------|------|------|
| **全局 Dice** | 0.9203 | 0.8892 | 0.8516 |
| **全局 HD95 (mm)** | 5.17 | 4.36 | 3.93 |
| **LW Dice WT** | 0.8121 | 0.7815 | 0.7165 |

---

## 六、文件结构

```
SegResNet-DiceCE-AdamW/
├── README.md              # 本文件
├── models/
│   └── __init__.py        # SegResNet (MONAI) + get_model 接口
├── frame/
│   └── train.py           # BaselineTrainer 训练框架
└── pipeline/
    └── train.py           # 训练入口
```

---

## 七、训练命令

```bash
cd SegResNet-DiceCE-AdamW
python -m pipeline.train --cache --epochs 100 --batch 2
```

---

## 八、作为 Baseline 的意义

SegResNet 在本项目中作为 **CNN 基线** 的角色：

1. **性能参考**: 提供传统 CNN 方法的上界
2. **对比基准**: 与 Mamba 模型 (SegResMamba-Lite) 对比
3. **架构基础**: SegResMamba-Lite 的 LightResBlock 设计灵感来源于此
4. **稳定可靠**: 经过 BraTS 竞赛验证的成熟架构
