# SegResMamba-Lite

> 脑肿瘤分割轻量级 Mamba 模型 - MoA 框架完整演进

## 📋 项目概述

SegResMamba-Lite 是基于 Mamba 架构的脑肿瘤分割模型集合（V1–V12），包含 **MoA（Mixture of Attention）框架**的完整设计演进，**最终模型为 V10（论文级 token 级稀疏 MoA，1.42M 参数）**。所有模型参数量均控制在 **< 1.5M**。

## 🌿 分支说明

| 分支 | 用途 | 内容 | 本地目录 |
|------|------|------|----------|
| **`main`** | **项目主分支（默认）** | **本目录（仅 README）** | `Main/` |
| `experiments` | 完整实验源码 | V1–V12 全部代码 + 4 基线 + shared 共享库 | `Code/` |
| `webapp` | Web 应用 | 前后端 Docker 网页项目 | `WebAPP/` |
| `App` | Android 应用 | Gradle + Kotlin Android App | `APP/` |

## 🎯 快速开始（experiments 分支）

完整训练/评估命令见 `Code/SegResMamba-Lite/README.md` 第六节（必须显式 `--init_filters 20`）：

```bash
# 训练最终模型 V10（详见 experiments 分支 SegResMamba-Lite/README.md）
python SegResMamba-Lite/pipeline/train.py --version v10 --init_filters 20 --epochs 100 --batch 6 --workers 8 --resolution 2.0 --device cuda --cache

# 评估（生成全局 + lesion-wise 逐例 CSV）
python SegResMamba-Lite/pipeline/evaluate.py --version v10 --init_filters 20 --resolution 2.0 --device cuda --cache --workers 8
```

## 📊 版本演进（V1 → V12，官方 374mm 满罚口径，188 例测试集）

| 版本 | 关键创新 | 参数量 | Dice 均值 | HD95 均值 |
|------|---------|:-----:|:----:|:----:|
| V1 | 双向 Mamba + 跳跃连接 | 1.52M | 0.8886 | 9.96 |
| V2 | 多尺度 BiMamba（3 尺度） | 1.42M | 0.8885 | 8.56 |
| V3 | 双重 Bottleneck + SE | 1.51M | 0.8853 | 9.00 |
| V4 | 三重 Bottleneck + 策略 SE | 1.56M | 0.8880 | 9.62 |
| V6 | Soft MoA（4 异构 Mamba 专家 + 样本级软路由） | 1.46M | 0.8904 | 8.87 |
| V7 | MoA + DConv 专家 + Sigmoid α 自动学习 | 1.50M | 0.8912 | 9.12 |
| V8 | MoA + α 温度缩放（权重未留存，HD95 旧口径不可比） | 1.50M | 0.8860 | — |
| V9 | 真 MoE（Top-K=2 稀疏 + 4 DConv 专家 + 负载均衡） | 1.42M | 0.8893 | 7.34 |
| **V10** | **论文级 token 级稀疏 MoA（共享 K/V + Attention 专家）** | **1.42M** | **0.8878** | **7.90** |
| V11 | 边界感知混合 MoA（2 Attention + 2 DConv） | 1.44M | 0.8879 | 8.28 |
| V12 | V10 + 边界门控（探索性负结果） | 1.43M | — | — |

> V10 为**最终模型**：LW Dice 0.8304 全系列/全部模型最高（病灶级核心主张）；全局 HD95 均值 7.90 优于全部基线。V5 为未注册废弃草稿。完整数字与证据链见 `Code/SegResMamba-Lite/论文实验章节.md`。

## 🏗️ 仓库结构

```
Zviolin/SegResMamba-Lite
├── main 分支（Main/ 目录）
│ └── README.md（项目门面）
│
├── experiments 分支（Code/ 目录）
│ ├── SegResMamba-Lite/   # V1–V12 模型（最终 V10）
│ │ ├── models/ (v1.py ~ v12.py)
│ │ ├── pipeline/ (train.py + evaluate.py)
│ │ ├── frame/ (trainer)
│ │ ├── 实验记录总集.md / 论文实验章节.md
│ │ └── scripts/ + bench_efficiency.py（统计检验/效率基准/路由可视化）
│ ├── SegMamba-Official/  # 官方基线（66.90M 本地实测）
│ ├── SegResNet-DiceCE-AdamW/    # CNN 基线（4.70M）
│ ├── SegResNet-DiceFocal-AdamW-SWA-DS/    # CNN 基线（4.70M，SWA + 深监督）
│ ├── VMUNet-DiceCE-AdamW/       # 极轻量基线（0.52M）
│ ├── VSSUNet-DiceCE-AdamW/      # 极轻量基线（0.19M）
│ ├── LightSegMamba-DiceCE-AdamW-V3/   # 轻量 Mamba 探索（1.27M）
│ └── shared/             # 共享工具（data/models/metrics/losses/optim/inference/utils）
│
├── webapp 分支（WebAPP/ 目录）
│ └── README.md（占位，待开发）
│
└── App 分支（APP/ 目录）
 └── README.md（占位，待开发）
```

## 🔬 核心创新（MoA 框架三代演进 V6 → V12）

**MoA = Mixture of Attention**

```
三代演进：
├── 第二代 Soft MoA（V6–V8）：4 异构专家 + 样本级软路由 + α 融合
│   ├── V6: Soft MoA（4 Mamba 专家软路由，无 α）
│   ├── V7: + DConv 专家 + Sigmoid α 自动学习
│   └── V8: α 温度缩放
└── 第三代 稀疏 MoE / 论文级 MoA（V9–V11）
    ├── V9: Top-K=2 真稀疏 + 4 DConv 专家 + 负载均衡（Switch Transformer 风格）
    ├── V10: 论文级 token 稀疏 MoA（严格按 SHMoAReg arXiv:2509.20073：
    │        共享 K/V + Attention 专家 + token 级 Top-2 路由）★ 最终模型
    └── V11: 混合专家池 + Sobel 边界注意力 + 路由噪声
```

### 关键结论

```
全局 Dice 由骨干决定（全版本极差仅 0.0059），病灶级指标才反映 MoA 设计差异：
V10 LW Dice 0.8304 为全部模型最高 ——「轻量参数 × 病灶级 SOTA」同时成立。
```

## 📚 详细文档（experiments 分支 `Code/SegResMamba-Lite/`）

- 完整证据链：`实验记录总集.md`（版本演进/消融/统计检验）
- 论文实验章节：`论文实验章节.md`（主表 + 官方 374mm 口径）
- 训练/评估命令：`README.md` 第六节

## 🤝 贡献

参考 `experiments` 分支的各版本源码注释，了解每个版本的演进动机。

## 📄 引用

如果使用本项目，请引用：
- SegResMamba-Lite V10: 论文级 token 稀疏 MoA（最终模型）
- SegResMamba-Lite V9: Top-K 稀疏 MoE + 负载均衡
- MoA 框架设计参考：SHMoAReg (arXiv:2509.20073)

---

**License**: MIT | **Author**: Zviolin | **Created**: 2026