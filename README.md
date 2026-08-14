# SegResMamba-Lite

> 脑肿瘤分割轻量级 Mamba 模型 - MoA 框架完整演进

## 📋 项目概述

SegResMamba-Lite 是基于 Mamba 架构的脑肿瘤分割模型集合（V1-V8），包含 **MoA（Mixture of Attention）框架**的完整设计演进。所有模型参数量均控制在 **< 1.5M**。

## 🌿 分支说明

| 分支 | 用途 | 内容 | 本地目录 |
|------|------|------|----------|
| **`main`** | **项目主分支（默认）** | **本目录（仅 README）** | `Main/` |
| `experiments` | 完整实验源码 | V1-V8 全部代码 + 基线对比 | `Code/` |
| `webapp` | Web 应用 | 前后端 Docker 网页项目 | `WebAPP/` |
| `android` | Android 应用 | Gradle Android App | `Android/` |

## 🎯 快速开始（experiments 分支）

### 1. 安装依赖

```bash
pip install torch torchvision monai nibabel numpy pandas tqdm
```

### 2. 训练 V8（最终版本）

```bash
cd SegResMamba-Lite
python pipeline/train.py --version v8 --resolution 2.0 \
  --init_filters 20 --epochs 100 --batch 6 --workers 8 \
  --device cuda --cache --ema --loss_name DiceFocalLoss
```

### 3. 评估 V8

```bash
python pipeline/evaluate.py --version v8 --resolution 2.0 \
  --init_filters 20 --device cuda --cache --workers 8
```

## 📊 版本演进（V1 → V8）

| 版本 | 关键创新 | 参数量 | Dice | HD95 |
|------|---------|:-----:|:----:|:----:|
| V1 | 双向 Mamba + 跳跃连接 | 1.52M | 0.8887 | 4.72 |
| V2 | 多尺度 BiMamba（3 尺度） | 1.42M | 0.8885 | 4.63 |
| V3 | 双重 Bottleneck + SE | 1.51M | 0.8853 | 4.41 |
| V4 | 三重 Bottleneck + 策略 SE | 1.56M | 0.8880 | 5.04 |
| **V6** | **MoA 框架（共享 Mamba + 4×DConv 专家，α=0.5）** | **1.43M** | **0.8904** | **4.28** |
| V7 | MoA + Sigmoid α 自动学习 | 1.50M | 0.8913 | 4.54 |
| **V8** | **MoA + α 自由自适应（温度缩放）** | **1.43M** | **0.8868** | **4.78** |

## 🏗️ 仓库结构

```
Zviolin/SegResMamba-Lite
├── main 分支（Main/ 目录）
│ └── README.md（项目门面）
│
├── experiments 分支（Code/ 目录）
│ ├── SegResMamba-Lite/   # V1-V8 模型
│ │ ├── models/ (v1.py ~ v8.py)
│ │ ├── pipeline/ (train.py + evaluate.py)
│ │ └── frame/ (OptimizedTrainer)
│ ├── SegMamba-Official/  # 官方基线
│ ├── SegResNet-DiceCE-AdamW/    # CNN 基线
│ ├── SegResNet-DiceFocal-AdamW-EMA-DS/    # CNN 基线
│ ├── VMUNet-DiceCE-AdamW/       # 极轻量基线
│ ├── VSSUNet-DiceCE-AdamW/      # 极轻量基线
│ └── shared/             # 共享工具（data/losses/optim/utils）
│
├── webapp 分支（WebAPP/ 目录）
│ └── README.md（占位，待开发）
│
└── android 分支（Android/ 目录）
 └── README.md（占位，待开发）
```

## 🔬 核心创新（V6-V8 MoA 框架）

**MoA = Mixture of Attention**

```
Bottleneck 设计：
├── Shared Mamba 主干（4 个 Mamba 专家，不同 d_state + 扫描方向）
├── 4× DConv 专家（Depthwise + Pointwise 卷积，轻量级局部细化）
├── Soft Router（GAP → MLP → Softmax）
└── α 混合权重（Mamba vs DConv）：
 - V6: 固定 α=0.5
 - V7: Sigmoid 自动学习（α ≈ 0.8）
 - V8: Sigmoid + 温度缩放（学术最佳实践）
```

### V6 关键洞察

```
传统 MoA（SHMoAReg 式）：4 个独立 Mamba 专家 + Top-K 路由
问题：轻量级模型下容量不足 + 训练不稳定

V6 的"实用 MoA"：
├── 1 个共享 Mamba 主干（保留 V2 全局建模）
├── 4 个轻量 DConv 专家（局部细化）
└── 软权重路由（无稀疏、无崩溃）
```

## 📚 详细文档

- 完整 V1-V8 对比：`Code/SegResMamba-Lite/模型完整对比与分析.md`
- V1-V4 演进：`Code/SegResMamba-Lite/README.md`
- 版本差异详解：`Code/模型版本差异详解.md`
- 训练流程详解：`Code/训练流程详解_面向零基础.md`

## 🤝 贡献

参考 `experiments` 分支的各版本源码注释，了解每个版本的演进动机。

## 📄 引用

如果使用本项目，请引用：
- SegResMamba-Lite V8: MoA 框架 + 温度缩放 α
- SegResMamba-Lite V6: 实用 MoA 框架
- SegResMamba-Lite V2: 多尺度 BiMamba 基线

---

**License**: MIT | **Author**: Zviolin | **Created**: 2026