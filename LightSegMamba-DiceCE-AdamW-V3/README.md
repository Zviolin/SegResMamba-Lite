# LightSegMamba-DiceCE-AdamW-V3：脑肿瘤分割复现

> **论文**：Kim J, Kim J, Dharejo F A, Abbas Z, Lee S W.
> *Lightweight Mamba Model for 3D Tumor Segmentation in Automated Breast Ultrasounds*.
> **Mathematics (MDPI), 2025**.
> **数据集**：BraTS 2023 GLI（T1n / T1c / T2w / T2f 四模态，1251 例）

## 1. V3 相对 V2 的改动

| 改动点 | V2 | V3 |
|------|-----|-----|
| 默认 `base_channels` | 24（807K 参数，偏小） | **32（实测 1.27M，2026-10 代码口径）** |
| 参数量 CLI 可调 | 是 | 是（`--base-channels`） |
| CLI 风格 | 自动用缓存 | **严格对齐 MambaUNet：`--cache` 显式启用** |
| 训练规模切换 | 单一缓存目录，会覆盖 | **按 `--max-samples` 分目录：`persistent_cache_2.0_200/` vs `persistent_cache_2.0/`** |
| Trainer 接口 | 仅 `validate` | **完全对齐 MambaUNetTrainer：`validate` + `validate_verbose`** |
| 评估结果 CSV | 单文件 | **按 `max_samples` 区分（`_200` / `_full`）** |
| smoke_test | 仅参数量 | **新增显存占用测量、参数量参考表** |
| `_flip_last_dim` 注释 | 歧义（"沿最后一维"实为 dim=1） | **重命名为 `_flip_seq_dim`，注释清晰** |

## 2. 目录结构

```
LightSegMamba-DiceCE-AdamW-V3/
├── __init__.py
├── README.md
├── smoke_test.py                # 无数据模型烟雾测试
├── models/
│   ├── __init__.py
│   └── lightsegmamba.py         # Stem / DASPP / ToM / DASPPMamba / Decoder / 主网络
├── frame/
│   ├── __init__.py
│   └── train.py                 # LightSegMambaTrainer（DiceCE + AdamW + CosineLR + SWA 权重平均）
└── pipeline/
    ├── __init__.py
    ├── gen_cache.py             # 生成 PersistentDataset 缓存（支持 --max-samples）
    ├── train.py                 # 训练入口（CLI 对齐 MambaUNet）
    ├── evaluate.py              # 评估入口（含 lesion-wise）
    ├── logs/                    # 训练 / 评估日志（运行后生成）
    ├── models/                  # 模型权重保存目录（运行后生成）
    │   ├── 2.0mm_full_cuda/     # 全量训练保存目录
    │   └── 2.0mm_200_cuda/      # 200 例训练保存目录
    └── evaluation_results/      # 评估 CSV（运行后生成）
```

### 零基础速览：训练到底在做什么

整个训练就是「看图 → 画圈 → 对答案 → 调整」的循环，重复若干轮（每轮记 1 个 epoch）：

```
原始数据 → 预处理 → 模型预测 → 计算Loss → 反向传播 → 参数更新 ─┐
   ↑                                                      │
   └──── 验证循环 ←── 滑窗推理 ←─────────────────────────────┘
        （每 val_interval 轮一次，保存最佳权重）
```

- **预处理**：把每位病人的 4 张 MRI 统一方向（RAS）、重采样到 2.0mm、裁掉背景、归一化——相当于把所有考卷印成同一格式，模型才好学。
- **模型预测**：编码器把 64³ 立体图逐级压缩（空间变小、通道变多），像把一幅彩画浓缩成一本摘要书——知道「大概哪里有肿瘤」但丢了精确边界；解码器再逐级放大回原尺寸，并经跳跃连接把浅层「精细笔记」传回来补细节，最后输出每个体素属于 4 个类别（背景/NCR/ED/ET）的概率。本项目的 DASPPMamba 编码器在压缩路径中嵌入了 Tri-Oriented Mamba 三向序列扫描（见 §3）。
- **计算 Loss**：用 DiceCELoss（Dice 重合度 + 交叉熵）衡量「画的圈」与医生标注的差距，重合度越高 Loss 越小。
- **反向传播 + 参数更新**：`loss.backward()` 算出每个权重该往哪调；AdamW 按设定的学习率更新权重；CosineAnnealingLR 让学习率随训练余弦式衰减——先大步学、后小步修。
- **验证**：每 val_interval 轮，用滑动窗口（64³ 窗口、50% 重叠、高斯加权）扫过整个验证病例算 Dice/HD95；WT Dice 最佳的权重存为 `best_metric_model.pth`，`latest_checkpoint.pth` 每个 epoch 都存（中断可续训）。
- **AMP 混合精度**：部分计算用半精度浮点省显存提速，GradScaler 防小梯度下溢。

| 术语 | 含义 |
|------|------|
| epoch | 全部训练数据完整过一遍 |
| batch | 一次同时送入模型的病例数 |
| patch | 每次随机裁出的 64³ 立方小块 |
| checkpoint | 可恢复训练的完整状态（权重 + 优化器 + 调度器 + AMP） |

## 3. 模型设计

### 3.1 整体架构

```
输入 (B, 4, D, H, W)
   │
   ├── Stem: 7×7×7 Conv3d stride=2 + BN + SiLU ── skip0 (c1 = 32)
   │
   ├── Down1: 2×2×2 Conv3d stride=2
   ├── DASPPMamba-1 (DASPP + Tri-Oriented Mamba) ────────── skip1 (c2 = 64)
   │
   ├── Down2: 2×2×2 Conv3d stride=2
   ├── DASPPMamba-2 (DASPP + Tri-Oriented Mamba) ────────── bottleneck (c3 = 128)
   │
   ├── Decoder-1: ConvTranspose3d + cat(skip1) + 双卷积 (c2)
   ├── Decoder-2: ConvTranspose3d + cat(skip0) + 双卷积 (c1)
   │
   └── Output: 1×1×1 Conv3d → 4 通道 logits
              + trilinear 上采样到原图尺寸
```

### 3.2 DASPP（Deep Atrous Spatial Pyramid Pooling）

6 条并行路径，全部使用 **深度可分离卷积**（depthwise 3×3×3 + pointwise 1×1×1）：

| 路径 | 卷积 | 作用 |
|------|------|------|
| 1 | AdaptiveAvgPool3d + 1×1×1 DWS Conv | 全局上下文 |
| 2 | 1×1×1 DWS Conv | 细粒度局部 |
| 3 | 3×3×3 DWS Conv, dilation=3 | 中等感受野 |
| 4 | 3×3×3 DWS Conv, dilation=6 | 大感受野 |
| 5 | 3×3×3 DWS Conv, dilation=9 | 极大感受野 |
| 6 | 跳跃连接 | 保留输入 |

拼接 → 1×1×1 DWS Conv 调通道 → BN + SiLU → 残差连接。

### 3.3 Tri-Oriented Mamba（ToM）

沿 **x、y、z** 三个解剖方向各做一次双向 Mamba 扫描，三方向输出取平均。

> **实现说明**：原论文 SegMamba 用 state-spaces 自定义 fork 的 `mamba_ssm.Mamba`，支持 `bimamba_type="v3"`。本复现兼容**官方版 mamba_ssm（≥1.2 / 2.x）**——每个方向用两个 Mamba 分别做正向与反向扫描，按 0.5 + 0.5 加权融合，等价于双向扫描。

```
输入 (B, C, D, H, W)
   → permute 到 channels-last (B, D, H, W, C)
   → LayerNorm
   → 三方向 reshape (B*D*H, W, C) / (B*D*W, H, C) / (B*H*W, D, C)
   → 每个方向：[forward Mamba + flip → backward Mamba → flip] / 2
   → 三方向 reshape 回原空间布局
   → 取平均 → 1×1×1 DWS Conv 融合 → 残差
```

总参数量：每方向 2 个 Mamba × 3 方向 = **6 个 mamba_ssm.Mamba**。

### 3.4 参数量参考表

| base_channels | 参数量 | 说明 |
|---------------|--------|------|
| 24 | ~0.81M | V2 量级（偏小，可能欠拟合） |
| **32** | **~1.4M** | **V3 默认（4GB 显存安全，推荐）** |
| 38 | ~2.0M | 折中 |
| 44 | ~2.7M | 接近论文 3.08M（4GB 显存需 patch=32³） |

## 4. 安装与运行

### 4.1 环境依赖

```bash
# 推荐环境：mamba_sm120（已安装 mamba_ssm + causal_conv1d）
conda activate mamba_sm120

# 基础依赖
pip install torch==2.5.1+cu121 monai==1.5.2 numpy scipy scikit-learn nibabel
```

> **注意**：SRTP 环境未安装 `mamba_ssm`，必须切换到 `mamba_sm120` 等含 Mamba 的环境。

### 4.2 烟雾测试（验证模型结构，< 1 分钟）

```bash
cd g:\Codes\Python\SRTP\Essay\PythonFiles\Code\LightSegMamba-DiceCE-AdamW-V3

# 默认 base_channels=32
python smoke_test.py

# 测其他参数量
python smoke_test.py --base-channels 24   # V2 量级
python smoke_test.py --base-channels 44   # 论文量级
```

期望输出：参数量约 1.4M、前向输出 `(1, 4, 64, 64, 64)`、反向梯度正常、峰值显存 < 2GB。

### 4.3 生成数据缓存

V3 **按 `--max-samples` 区分缓存目录**，200 例与全量互不干扰：

```bash
# 200 例 2.0mm 缓存（快速验证，约 5-10 分钟）
python pipeline/gen_cache.py --resolution 2.0 --max-samples 200 --generate --cache

# 1251 例 2.0mm 缓存（全量，约 30-60 分钟）
python pipeline/gen_cache.py --resolution 2.0 --generate --cache

# 1251 例 1.0mm 缓存（全量 + 高精度，约 2-4 小时）
python pipeline/gen_cache.py --resolution 1.0 --generate --cache --workers 4
```

缓存目录约定（2026-10-07 全库统一定案）：
- 不传 `--max-samples` → `D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache_2.0\`（全量，主项目体系缓存）
- `--max-samples 200` → `D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache_2.0_200\`（该目录尚未生成，需先跑 gen_cache）
- 缓存父目录默认主项目体系 `D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache`，可用环境变量 `SRTP_CACHE_PARENT` 覆盖
- 注：历史训练（Mamba1/Mamba2 全量）使用旧默认 `Essay\Data\persistent_cache_2.0`（划分实现与主项目体系不同，test 集交集仅 21/188），历史数字为旧划分口径

### 4.4 训练

#### 200 例快速验证（推荐先跑）

```powershell
# PowerShell 单行命令（可直接粘贴运行）
python pipeline/train.py --model lightsegmamba --resolution 2.0 --cache --max-samples 200 --batch 1 --workers 0 --patch-size 64 64 64 --base-channels 32 --epochs 50 --val-interval 5 --device cuda
```

模型权重保存：`pipeline/models/2.0mm_200_cuda/`

#### 1251 例全量训练

```powershell
python pipeline/train.py --model lightsegmamba --resolution 2.0 --cache --batch 1 --workers 0 --patch-size 64 64 64 --base-channels 32 --epochs 100 --val-interval 5 --device cuda
```

模型权重保存：`pipeline/models/2.0mm_full_cuda/`

#### 加 SWA（可选，4GB 显存需谨慎；历史命名 EMA，等权滑动平均）

```powershell
python pipeline/train.py --resolution 2.0 --cache --max-samples 200 --batch 1 --patch-size 64 64 64 --base-channels 32 --epochs 50 --val-interval 5 --swa --device cuda
```

#### 断点续训（中断恢复）

训练每个 epoch 结束都会把完整训练状态写入权重目录下的 `latest_checkpoint.pth`（默认目录如 `pipeline/models/2.0mm_200_cuda/`、`pipeline/models/2.0mm_full_cuda/`）。训练中途中断后，**直接用原命令重跑即可**：启动时检测到该文件会自动从中恢复（模型权重、优化器、学习率调度器等状态），并从断点 epoch 继续训练，无需手动指定 checkpoint。续训时保持与原训练相同的权重目录——`--model_dir` 传过就传同一个值，没传过就继续不传（目录由 `--resolution` / `--max-samples` / `--device` 推导，这些参数与原训练一致即自动一致）。若想放弃断点从头重训，先删除或移走该目录下的 `latest_checkpoint.pth`。

### 4.5 评估

```powershell
# 评估 200 例训练的模型
python pipeline/evaluate.py --resolution 2.0 --cache --max-samples 200 --base-channels 32 --checkpoint pipeline/models/2.0mm_200_cuda/best_metric_model.pth --device cuda

# 评估全量训练的模型
python pipeline/evaluate.py --resolution 2.0 --cache --base-channels 32 --checkpoint pipeline/models/2.0mm_full_cuda/best_metric_model.pth --device cuda
```

结果 CSV：
- `pipeline/evaluation_results/metrics_lightsegmamba_v3_2.0mm_200.csv`
- `pipeline/evaluation_results/metrics_lightsegmamba_v3_2.0mm_200_lesion.csv`
- `pipeline/evaluation_results/metrics_lightsegmamba_v3_2.0mm_full.csv`
- `pipeline/evaluation_results/metrics_lightsegmamba_v3_2.0mm_full_lesion.csv`

## 5. CLI 参数完整列表

### train.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `lightsegmamba` | 模型名称 |
| `--resolution` | `2.0` | 重采样分辨率（mm） |
| `--cache` | False | 启用硬盘缓存（必须） |
| `--max-samples` | None（全量） | 限样本数（200=快速验证） |
| `--batch` | 1 | 批次大小（4GB 显存建议 1） |
| `--workers` | 0 | DataLoader worker 数（4GB 显存机器建议 0-2） |
| `--patch-size` | `64 64 64` | 训练 patch 尺寸 |
| `--base-channels` | 32 | Stem 通道数（决定参数量） |
| `--epochs` | 50 | 训练轮数 |
| `--val-interval` | 5 | 验证间隔 |
| `--lr` | 1e-4 | 初始学习率 |
| `--weight-decay` | 1e-5 | AdamW 权重衰减 |
| `--swa` | False | 启用权重平均（等权滑动平均，SWA 式；历史命名 EMA，CLI 已改名） |
| `--device` | auto | 设备（cuda/cpu） |
| `--seed` | 42 | 训练随机种子（--deterministic 时生效，不影响数据划分缓存） |
| `--deterministic` | False | 确定性训练：固定种子 + cudnn.deterministic + 确定性算法 + CUBLAS_WORKSPACE_CONFIG（不触碰 TF32，保持论文口径可比） |
| `--cache_parent` | `""` | 缓存父目录覆盖：命令行 > 环境变量 SRTP_CACHE_PARENT > 历史默认；最终目录 = `{parent}_{分辨率}[_{max_samples}]`（与 `--max-samples` 采样目录正交） |
| `--skip_nonfinite` | False | 跳过非有限 loss（NaN/Inf）的 batch：不 backward/不更新/不进平均权重，epoch 指标按有效 step 平均 |
| `--best_weights` | `raw` | best_metric_model.pth 保存来源：raw=原始权重（历史口径）/ avg=验证所用的平均权重（SWA，需配合 --swa，未启用时回退 raw） |

### evaluate.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | `pipeline/models/2.0mm_full_cuda/best_metric_model.pth` | 检查点路径 |
| `--no-lesion-wise` | False | 禁用 lesion-wise 指标 |
| `--cache_parent` | `""` | 缓存父目录覆盖：命令行 > 环境变量 SRTP_CACHE_PARENT > 历史默认；最终目录 = `{parent}_{分辨率}[_{max_samples}]`（与 `--max-samples` 采样目录正交） |
| `--data_root` | `""` | 原始 BraTS TrainingData 目录覆盖：命令行 > 环境变量 BRATS_DATA_ROOT > 历史默认；本项目评估为纯缓存驱动，此参数仅保持跨项目口径一致 |
| 其他 | - | 与 train.py 一致 |

> 以上 `--deterministic` / `--cache_parent` / `--skip_nonfinite` / `--best_weights`（及 evaluate 的 `--cache_parent` / `--data_root`）为 2026-10-08 跨卡实验包合并引入的可选开关，默认关闭时不带任何新参数的命令行为与历史逐位一致；换机迁移用命令行参数或环境变量重定向，不改变各项目历史数据划分绑定。

### gen_cache.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--generate` | False | 生成缓存文件 |
| `--cache` | False | 启用缓存（必须） |
| `--train-ratio` | 0.7 | 训练集比例 |
| `--val-ratio` | 0.15 | 验证集比例 |
| `--test-ratio` | 0.15 | 测试集比例 |
| `--seed` | 42 | 随机种子 |
| `--dtype` | `float16` | 缓存数据类型 |
| `--cache-format` | `numpy` | 缓存格式 |

## 6. 显存与时间预估（RTX 3050 4GB，Mamba1 版历史实测）

| 配置 | 峰值显存 | 单 epoch 耗时 |
|------|---------|--------------|
| 200 例 + 64³ patch + batch=1 + base=32 | ~1.8 GB | ~1-2 min |
| 200 例 + 96³ patch + batch=1 + base=32 | ~2.8 GB | ~3-4 min |
| 1251 例 + 64³ patch + batch=1 + base=32 | ~2.0 GB | ~8-10 min/epoch |
| 1251 例 + 64³ patch + batch=1 + base=44 | ~3.5 GB | ~12-15 min/epoch |
| 滑窗推理（test, roi=64³, sw_bs=4） | ~1.5 GB | ~10 s/case |

> 4GB 卡务必：`patch ≤ 64³`、`batch = 1`、`--base-channels ≤ 32`。开权重平均（`--swa`，等权滑动平均；历史命名 EMA）会额外占用一份模型副本。
> **2026-10 迁移至 RTX 5060 8GB（Mamba2 版）后**：标准配置 `batch=6 + workers=8`（对齐 v10 锚点基准）实测余量充足，首次迭代含 triton JIT 编译约 37 s 属正常；全量 100 epoch 训练 + 188 例评估已完成。

## 7. 数据约定

- 4 模态固定顺序：`t1n, t1c, t2w, t2f`（与 shared.data.dataloader.get_data_list 一致）
- 标签：BraTS 三通道（WT/TC/ET）+ 背景，由 `brats_label_mapping(label_4_to_3=True)` 转换
- 缓存目录（2026-10-07 统一默认主项目体系缓存）：
  - 全量：`D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache_2.0\`
  - 200 例：`D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache_2.0_200\`（需 gen_cache 生成）
  - 缓存父目录可用环境变量 `SRTP_CACHE_PARENT` 覆盖
  - 注：历史训练数字基于旧默认 `Essay\Data` 划分口径（与主项目体系划分不同）

## 8. 推荐使用流程

```bash
# Step 1: 烟雾测试，确认环境与模型
python smoke_test.py

# Step 2: 生成 200 例缓存
python pipeline/gen_cache.py --resolution 2.0 --max-samples 200 --generate --cache

# Step 3: 200 例快速训练（约 1-2 小时）
python pipeline/train.py --resolution 2.0 --cache --max-samples 200 --epochs 50 --device cuda

# Step 4: 评估 200 例训练结果
python pipeline/evaluate.py --resolution 2.0 --cache --max-samples 200 --checkpoint pipeline/models/2.0mm_200_cuda/best_metric_model.pth --device cuda

# Step 5: 若 200 例结果合理，生成全量缓存并训练
python pipeline/gen_cache.py --resolution 2.0 --generate --cache
python pipeline/train.py --resolution 2.0 --cache --epochs 100 --device cuda

# Step 6: 评估全量训练结果
python pipeline/evaluate.py --resolution 2.0 --cache --checkpoint pipeline/models/2.0mm_full_cuda/best_metric_model.pth --device cuda
```

## 9. 已知限制

- **必须用 mamba_ssm 环境**：SRTP 环境未装，需切换到 mamba_sm120
- **仅支持 4 模态**：如改 1 模态（单 T1），需把模型 `in_channels=1`
- **下采样深度与论文不同（有意设计）**：论文 bottleneck 为 192×D/4（总下采样 4×，Section 3.2）；本复现 stem 2 次 + encoder 间下采样 2 次（总 8×），bottleneck 在 D/8——4GB 显存约束下的适配，此前对比表未注明，特此补充
- **不使用 bimamba_type**：本实现用官方 mamba_ssm 的"每方向正反两次扫描按 0.5+0.5 融合"等价替代，无 bimamba_type 版本依赖（旧版此处写的 bimamba_type 依赖说明已过时）
- **权重平均（SWA）验证接口**：如启用 `--swa`（等权滑动平均，历史命名 EMA），验证走 trainer 内 `self.swa.averaged_model`；evaluate.py 加载权重时对旧命名（含 "ema"）与新命名（含 "swa"）文件均自动识别
- **全局 HD95 口径**：逐病例空预测记 0（`frame/train.py` 中 nan/inf→0 防护）；lesion-wise 口径为空预测 374 惩罚。两种口径数值实测一致（全量日志汇总 6.8723 ≈ CSV 均值 6.87），引用时注明所采用口径即可

## 10. 论文与本复现对比

| 项目 | 论文 | V3 复现 |
|------|------|--------|
| 数据集 | TDSC-ABUS 2023 (~200 例) | BraTS 2023 GLI (200/1251 例) |
| 模态数 | 1 | 4 |
| 输出类别 | 2 | 4 (BG/TC/WT/ET) |
| Encoder 阶段数 | 2 | 2 ✓ |
| Bottleneck 位置 | D/4（192ch，总 4× 下采样） | D/8（128ch，总 8× 下采样，4GB 显存适配） |
| DASPP 扩张率 | 3/6/9 | 3/6/9 ✓ |
| Stem kernel | 7×7×7 stride=2 | 7×7×7 stride=2 ✓ |
| DWS Conv | ✓ | ✓ |
| Tri-Oriented Mamba | ✓ (x/y/z) | ✓ (x/y/z) |
| 参数量 | 3.08M | **1.27M (base=32，Mamba2 版实测 1,268,720)** / 2.7M (base=44)；Mamba1 版 1,369,604 已随 Mamba1 路线归档 |
| Dice (论文基准) | 0.7985 (ABUS 2 类) | **WT 0.8913 / TC 0.8297 / ET 0.7607**（BraTS 4 类，Mamba2 版，188 例测试集；Mamba1 版 0.8967/0.8402/0.7791 已归档；数据集与类别数不同，不可直接比较） |

> 官方代码说明：原论文未随文发布开源代码（论文 Code Availability 仅提供数据集链接；
> 截至 2026-10 经 GitHub / GitLab / Gitee / Papers With Code 检索均无官方仓库），
> 本复现按论文 Section 3.2 描述实现；文中"原论文用 fork 版 mamba_ssm"系对 SegMamba
> 谱系用法的推断，非论文原文表述。ToM 三方向输出融合方式论文未写明，本实现采用
> 三方向平均，属合理假设。

## 11. 引用

```bibtex
@article{kim2025lightsegmamba,
  title={Lightweight Mamba Model for 3D Tumor Segmentation in Automated Breast Ultrasounds},
  author={Kim, JongNam and Kim, Jun and Dharejo, Fayaz Ali and Abbas, Zubair and Lee, Sang Won},
  journal={Mathematics},
  year={2025},
  publisher={MDPI}
}
```
