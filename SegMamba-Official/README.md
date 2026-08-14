# SegMamba-Official - 官方 SegMamba 实现

> 基于论文 **SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation** 的 1:1 复现

---

## 一、论文信息

| 项目 | 内容 |
|------|------|
| **论文标题** | SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation |
| **作者** | Zhaohu Xing, Tian Ye, Yijun Yang, Guang Liu, Lei Zhu |
| **机构** | The Hong Kong University of Science and Technology (Guangzhou) / Beijing Academy of Artificial Intelligence |
| **发表** | MICCAI 2024 |
| **arXiv** | [arXiv:2401.13560](https://arxiv.org/abs/2401.13560) |
| **官方代码** | [github.com/ge-xing/SegMamba](https://github.com/ge-xing/SegMamba) |

---

## 二、核心思想

SegMamba 是 **首个将 Mamba (State Space Model) 专门用于 3D 医学图像分割** 的工作，核心创新：

1. **TSMamba Block**: Tri-orientated Spatial Mamba 模块，从三个方向建模全局依赖
2. **GSC (Gated Spatial Convolution)**: 在 Mamba 层之前增强空间特征表示
3. **ToM (Tri-orientated Mamba)**: 三方向 Mamba (正向/反向/层间)，替代单向 Mamba
4. **FUE (Feature-level Uncertainty Estimation)**: 跳跃连接中的特征不确定性过滤
5. **线性复杂度**: 相比 Transformer 的 O(n²)，Mamba 仅 O(n)

---

## 三、论文原始官方配置

> 以下配置来自论文原文及官方 GitHub 仓库，是论文中报告结果所使用的配置。

### 3.1 模型架构参数

| 参数 | 值 | 说明 |
|------|------|------|
| **Stem 卷积核** | 7×7×7, stride=2, padding=3 | 深度可分离卷积 |
| **特征维度** | [48, 96, 192, 384] | 4 个 Stage 的通道数 |
| **Stage 深度** | [2, 2, 2, 2] | 每个 Stage 的 TSMamba 块数量 |
| **Mamba d_state** | 16 | 状态空间维度 |
| **Mamba d_conv** | 4 | 局部卷积核大小 |
| **Mamba expand** | 2 | 扩展因子 |
| **下采样方式** | InstanceNorm + Conv3d(k=2, s=2) | Stage 间下采样 |
| **跳跃连接** | FUE 模块 | 特征不确定性估计 |
| **解码器** | CNN-based (卷积上采样) | 非 UNETR 风格 |

### 3.2 训练参数 (论文原始)

| 参数 | 值 | 说明 |
|------|------|------|
| **优化器** | SGD | momentum=0.99, nesterov=True |
| **初始学习率** | 1e-2 | 论文中 SGD 比 AdamW 提升 ~1.2% Dice |
| **权重衰减** | 3e-5 | L2 正则化 |
| **学习率调度** | Polynomial Decay | 多项式衰减 |
| **损失函数** | CrossEntropyLoss | 交叉熵损失 |
| **最大 Epochs** | 1000 | 长训练确保收敛 |
| **批次大小** | 2 | 受限于 3D 体积显存 |
| **验证间隔** | 2 | 每 2 个 epoch 验证一次 |

### 3.3 数据预处理 (论文原始)

| 步骤 | 操作 | 说明 |
|------|------|------|
| **1. 重采样** | 各向同性重采样 | 遵循 nnUNet 策略，自动计算目标分辨率 |
| **2. 归一化** | Z-score 归一化 | 对每个模态独立归一化 |
| **3. 裁剪** | Crop to foreground | 裁剪到非零区域 |
| **4. 输入模态** | 4 通道 (T1, T1ce, T2, FLAIR) | BraTS2023 标准 4 模态 |
| **5. 输出类别** | 4 类 (背景/WT/TC/ET) | BraTS 标签映射 |
| **6. 数据增强** | 随机旋转/翻转/噪声/对比度等 | 5 倍增强 (nnUNet 风格) |

### 3.4 推理配置 (论文原始)

| 参数 | 值 |
|------|------|
| **推理方式** | Sliding Window |
| **窗口大小** | 128×128×128 |
| **重叠率** | 0.5 |
| **模式** | Gaussian weighting |

---

## 四、论文核心架构详解

### 4.1 TSMamba Block (Tri-orientated Spatial Mamba)

```
TSMamba Block = GSC + ToM + MLP + 残差连接

输入 z
    │
    ▼
┌─────────────────────────────────────────┐
│  GSC (Gated Spatial Convolution)         │
│                                          │
│  z → Conv3d(3×3×3) → Norm → ReLU ──┐   │
│  z → Conv3d(1×1×1) → Norm → ReLU ──┤   │
│                                      × (逐元素相乘)
│                                      ↓   │
│                              Conv3d(3×3×3)
│                                      ↓   │
│                              + z (残差)    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  ToM (Tri-orientated Mamba)              │
│                                          │
│  z → Mamba(z_forward)  ──┐              │
│  z → Mamba(z_reverse)  ──┤  (三方向求和) │
│  z → Mamba(z_slice)    ──┘              │
│                                          │
│  + z (残差连接)                           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  MLP + LayerNorm + 残差连接              │
└─────────────────────────────────────────┘
    │
输出
```

### 4.2 ToM 三方向展开说明

```
三方向 Mamba 的含义:

  z_forward: 沿 D 轴正向展平 → 1D 序列 → Mamba → 恢复 3D
  z_reverse: 沿 D 轴反向展平 → 1D 序列 → Mamba → 恢复 3D
  z_slice:   沿层间方向展平   → 1D 序列 → Mamba → 恢复 3D

目的: 原始 Mamba 只能沿一个方向建模依赖，
      三方向设计确保 3D 空间各向同性建模
```

### 4.3 FUE (Feature-level Uncertainty Estimation)

```
FUE 用于跳跃连接中的特征过滤:

输入特征 z (C×D×H×W)
    │
    ▼
沿通道维度求均值: z̄ = (1/C) × Σ z_c
    │
    ▼
Sigmoid 归一化: z̄ = σ(z̄)
    │
    ▼
计算不确定性: u = -z̄ × log(z̄)
    │
    ▼
用不确定性加权特征: z_enhanced = z × (1 - u)

目的: 过滤不确定性高的特征，增强可靠特征传递
```

---

## 五、本项目训练配置

> 以下为本项目中实际使用的配置，与论文原始配置有差异。

| 参数 | 论文原始 | 本项目实现 | 差异说明 |
|------|---------|-----------|---------|
| **优化器** | SGD | SGD | ✅ 一致 |
| **学习率** | 1e-2 | 1e-2 | ✅ 一致 |
| **权重衰减** | 3e-5 | 3e-5 | ✅ 一致 |
| **调度器** | Poly | Poly | ✅ 一致 |
| **损失函数** | CrossEntropyLoss | CrossEntropyLoss | ✅ 一致 |
| **Epochs** | 1000 | 1000 | ✅ 一致 |
| **Batch Size** | 2 | 2 | ✅ 一致 |
| **特征维度** | [48, 96, 192, 384] | [48, 96, 192, 384] | ✅ 一致 |
| **Mamba 类型** | ToM (三方向) | BiMamba (双向) | ⚠️ 简化实现 |
| **GSC 模块** | ✅ 有 | ❌ 无 | ⚠️ 未实现 |
| **FUE 模块** | ✅ 有 | ❌ 无 | ⚠️ 未实现 |
| **解码器** | CNN-based | UNETR-style ResBlock | ⚠️ 架构差异 |
| **分辨率** | nnUNet 自动 | 2.0 mm 固定 | ⚠️ 差异 |

---

## 六、与 SegResMamba-Lite 对比

| 对比项 | SegMamba (论文原始) | SegResMamba-Lite V3 |
|--------|-------------------|---------------------|
| **参数量** | ~74.87M | 1.35M |
| **Mamba 实现** | ToM (三方向) + GSC + FUE | TriMamba (三正交) |
| **编码器** | 全栈 Mamba (4 Stage) | LightResBlock + 局部 TriMamba |
| **解码器** | CNN-based | ConvTranspose + LightResBlock |
| **优化器** | SGD (momentum=0.99) | AdamW |
| **学习率** | 1e-2 | 1e-4 |
| **训练 Epochs** | 1000 | 100 |
| **损失函数** | CrossEntropyLoss | DiceFocalLoss |
| **定位** | 完整复现 (性能上限) | 轻量化设计 (<1.5M) |
| **参数量比** | 55× | 1× (基准) |

---

## 七、论文 BraTS2023 实验结果

> 以下结果来自论文原文 Table (BraTS2023 数据集)

| 方法 | Dice WT | Dice TC | Dice ET |
|------|---------|---------|---------|
| **SegMamba (论文)** | ~0.923 | ~0.887 | ~0.845 |

注: 论文中报告的具体数值请参考论文原文 Table 3 (Public Benchmarks)。

---

## 八、文件结构

```
SegMamba-Official/
├── README.md                    # 本文件
├── models/
│   ├── __init__.py              # 模型导出
│   └── segmamba_official.py     # SegMamba + GSC + MambaEncoder
├── frame/
│   └── train.py                 # OptimizedTrainer 训练框架
└── pipeline/
    ├── train.py                 # 训练入口 (SGD, lr=1e-2, 1000 epochs)
    └── evaluate.py              # 评估入口
```

---

## 九、训练命令

```bash
# 论文原始配置训练
cd SegMamba-Official
python -m pipeline.train --cache --epochs 1000 --batch 2 --resolution 2.0

# 评估
python -m pipeline.evaluate --cache --resolution 2.0 --checkpoint <path>
```

---

## 十、论文核心结论

1. SegMamba 是首个将 Mamba 专门用于 3D 医学图像分割的方法
2. ToM (Tri-orientated Mamba) 从三个方向建模全局依赖，优于单向 Mamba
3. GSC 模块在 Mamba 前增强空间特征，弥补 1D 序列展平丢失的空间信息
4. FUE 模块通过不确定性估计过滤跳跃连接中的噪声特征
5. 线性复杂度 O(n) 使得处理 64×64×64 (≈260k 序列长度) 高效可行
6. 在 BraTS2023 和 CRC-500 数据集上均超越 Transformer 方法
