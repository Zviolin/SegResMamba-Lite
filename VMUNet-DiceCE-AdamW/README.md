# VMUNet-DiceCE-AdamW：UltraLight 纯 CNN 轻量分割基线

> **一句话定位**：名字沿用 UltraLight VM-UNet 谱系的超轻量 3D U 形网络，但本仓库实现为 **ResBlock 纯 CNN**（约 0.52M 参数，无 VSSBlock、无 Mamba），用于回答"不引入任何状态空间/门控结构、只用标准残差卷积能到什么水平"。
>
> **实验结果（188 例测试集，2.0mm）**：全局 Dice WT 0.9109 / TC 0.8829 / ET 0.8376

## 1. 项目定位

| 项目 | 内容 |
|------|------|
| **模型** | UltraLightVMUNet3D（ResBlock 纯 CNN，**无 Mamba / 无 VSSBlock**） |
| **参考实现** | 仓库内 `UltraLight_VM_UNet_Version/model.py`（名称沿用 VM-UNet 谱系，结构已按轻量化适配） |
| **参数量** | 524,044（约 0.52M；代码实测，历史文档中的 ~0.8M / ~0.3M 均不准确） |
| **损失 / 优化器** | DiceCELoss / AdamW（项目命名即来源于此） |
| **运行环境** | conda `SRTP`（无需 mamba_ssm / causal_conv1d） |
| **数据集** | BraTS 2023 GLI，1251 例（训练 875 / 验证 188 / 测试 188，seed=42） |
| **共享设施** | 与其余 6 个模型项目共用 `shared/` 公共库（数据加载、损失、优化器、滑窗推理、指标） |
| **训练器** | `UltraLightTrainer`（日志前缀 `train_ultralight_*`） |

在 7 个对比模型中的位置：

| 模型项目 | 参数量 | Mamba 实现 | 环境 |
|----------|--------|-----------|------|
| SegResNet-DiceCE-AdamW | 4.70M | -（CNN 基线） | SRTP |
| SegResNet-DiceFocal-AdamW-SWA-DS | 4.70M | -（CNN + SWA/深监督） | SRTP |
| VSSUNet-DiceCE-AdamW | ~0.19M | VSSBlock（卷积近似） | SRTP |
| **VMUNet-DiceCE-AdamW（本项目）** | **~0.52M** | **无（纯 CNN ResBlock）** | **SRTP** |
| SegMamba-Official | 66.90M（本地实测） | mamba_ssm（真正 Mamba） | mamba_sm120 |
| SegResMamba-Lite | ~1.4M | mamba_ssm（真正 Mamba） | mamba_sm120 |
| LightSegMamba-DiceCE-AdamW-V3 | ~1.27M | mamba_ssm（真正 Mamba） | mamba_sm120 |

## 2. 新手背景：数据集、任务与指标

- **BraTS 2023 GLI**：国际医学影像挑战赛数据集，1251 例脑胶质母细胞瘤患者的 3D MRI，任务是**脑肿瘤分割**——对每个体素（3D 像素）判断属于哪个区域。
- **四模态**：每位患者有 4 种 MRI 序列——t1n（T1 平扫）、t1c（T1 钆增强）、t2w（T2 加权）、t2f（T2 FLAIR）。4 张 3D 图像叠在一起作为模型的 4 通道输入。
- **三区域**：WT（Whole Tumor，整个肿瘤，含水肿）、TC（Tumor Core，肿瘤核心）、ET（Enhancing Tumor，钆增强的活跃肿瘤）。模型输出 4 通道 logits（背景 + 3 区域），评估时按区域组合计算。
- **Dice**：预测与医生标注的重叠率，0~1，越高越好。**HD95**：95 分位 Hausdorff 距离，衡量分割边界与真实边界的误差（毫米），越低越好。
- **Lesion-wise（LW）指标**：把肿瘤拆成一个个连通"病灶"逐个评分再汇总，比全局指标更能反映多病灶的漏检与误检；LW Dice 还会被假阳性病灶数惩罚。

### 零基础速览：训练到底在做什么

整个训练就是「看图 → 画圈 → 对答案 → 调整」的循环，重复若干轮（每轮记 1 个 epoch）：

```
原始数据 → 预处理 → 模型预测 → 计算Loss → 反向传播 → 参数更新 ─┐
   ↑                                                      │
   └──── 验证循环 ←── 滑窗推理 ←─────────────────────────────┘
        （每 val_interval 轮一次，保存最佳权重）
```

- **预处理**：把每位病人的 4 张 MRI 统一方向（RAS）、重采样到 2.0mm、裁掉背景、归一化——相当于把所有考卷印成同一格式，模型才好学。
- **模型预测**：编码器把 64³ 立体图逐级压缩（空间变小、通道变多），像把一幅彩画浓缩成一本摘要书——知道「大概哪里有肿瘤」但丢了精确边界；解码器再逐级放大回原尺寸，并经跳跃连接把浅层「精细笔记」传回来补细节，最后输出每个体素属于 4 个类别（背景/NCR/ED/ET）的概率。
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

## 3. 模型结构与创新点

### 3.1 整体架构（2 层下采样的 U 形结构）

```
输入 [B, 4, D, H, W]
    │
    ▼
input_proj: ConvBlock(4→8)                        → [B, 8, D, H, W]
    │
    ▼
enc1: ConvBlock(8→16) + ResBlock + MaxPool(2)     → [B, 16, D/2, H/2, W/2]  ── skip1
    │
    ▼
enc2: ConvBlock(16→32) + ResBlock + MaxPool(2)    → [B, 32, D/4, H/4, W/4]  ── skip2
    │
    ▼
bottleneck: ConvBlock(32→64) + ResBlock           → [B, 64, D/8, H/8, W/8]
    │
    ▼
dec2: ConvTranspose3d + cat(skip2) + ConvBlock + ResBlock → [B, 32, D/4, ...]
    │
    ▼
dec1: ConvTranspose3d + cat(skip1) + ConvBlock + ResBlock → [B, 16, D/2, ...]
    │
    ▼
output_proj: 1×1×1 Conv3d(16→4)                   → [B, 4, D, H, W]（4 类 logits）
```

其中 `ConvBlock = Conv3d(3³) → BatchNorm3d → SiLU`；跳连（skip）拼接保留空间细节；上采样用可学习的转置卷积。

### 3.2 核心模块 ResBlock（标准残差卷积）

```
输入 x ────────────────────────────────┐（残差）
    │                                  │
    ▼                                  │
Conv3d(3³) → BatchNorm3d → SiLU        │
    ▼                                  │
Conv3d(3³) → BatchNorm3d → Dropout     │
    ▼                                  │
(+ x) → SiLU ──────────────────────────┴─► 输出
```

### 3.3 设计要点逐条

1. **极简主干**：仅 2 层下采样（enc1 + enc2），bottleneck 通道数 64（base_filters=8 的 8 倍），全模型 0.52M 参数。
2. **纯 CNN、零特殊依赖**：无 VSSBlock 门控、无 SSM 扫描、无注意力，任何装了 PyTorch 的环境都能跑，也无需 mamba_ssm 内核。
3. **标准残差连接**：ResBlock 稳定深层梯度流，训练无需额外的归一化技巧。
4. **可学习上采样**：解码用 ConvTranspose3d 而非插值，恢复空间分辨率的同时保留可学习容量。
5. **实验对照价值**：与参数量相近的 VSSUNet（0.19M，门控卷积）和真正的 Mamba 模型对照，界定"轻量 CNN 的能力边界"。

## 4. 环境说明

| 项 | 值 |
|----|----|
| **Conda 环境** | `SRTP`（非 Mamba 模型，无需 mamba_ssm / causal_conv1d） |
| **激活命令** | `conda activate SRTP` |
| **显卡口径** | RTX 5060 Laptop GPU 8GB；模型仅 0.52M，历史训练在更小显存的旧卡上即可完成，8GB 下 `--batch 4` 显存余量充足 |
| **关键依赖** | torch、MONAI（Dice/HD95 指标与滑窗推理）、numpy、scipy、nibabel |

## 5. 数据缓存约定（2026-10-07 全库统一定案）

- 本项目**强制使用缓存**：不传 `--cache` 会直接报错（代码中 `raise ValueError("必须启用缓存 (--cache)")`）。
- 缓存目录拼接规则 `{父目录}_{分辨率}`，默认父目录硬编码在 `pipeline/train.py` 中：

| 分辨率 | 实际缓存目录 |
|--------|-------------|
| 2.0mm（本机现用） | `D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache_2.0` |
| 其他分辨率 r | `D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache_{r}` |

- **整机重定向**：设置环境变量 `SRTP_CACHE_PARENT` 可覆盖默认父目录（用于换机迁移，仅影响父目录，拼接规则不变）。
- 缓存内含 `split_info.json`（seed=42 的 875/188/188 划分）与 `train/`、`val/`、`test/` 三个子目录；训练读 train/val，评估读 test。
- **预处理入口**（在 `Code/` 根目录执行，全库共用一份缓存，生成一次即可）：

```bash
conda activate SRTP
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code
python shared/data/prepare_data.py --resolution 2.0 --generate --cache --workers 8
```

完整预处理参数（含 3.0mm 等其他分辨率选项）见 `shared/data/prepare_data.py` 的 argparse 定义。

## 6. 命令行

统一口径：`conda activate SRTP` 后在 `g:/Codes/Python/SRTP/Essay/PythonFiles/Code` 目录执行。

### 6.1 训练（1251 例全量，历史实测配置 batch=4 / workers=4）

```bash
python VMUNet-DiceCE-AdamW/pipeline/train.py --resolution 2.0 --cache --batch 4 --epochs 100 --val_interval 5 --device cuda
```

- 权重保存目录：`VMUNet-DiceCE-AdamW/pipeline/models/2.0mm_cuda/`
- 日志保存：`VMUNet-DiceCE-AdamW/pipeline/logs/train_ultralight_2.0mm_*.log`
- 每 5 个 epoch（`--val_interval`）验证一次，保存 `epoch_{N}.pth`；验证 WT Dice 创新高时更新 `best_metric_model.pth`。

### 6.2 恢复训练（断点续训）

**机制**：每个 epoch 结束后都会保存 `latest_checkpoint.pth`（含模型权重、优化器/调度器/GradScaler 状态、epoch、最佳指标、最后成功验证的 epoch）。启动训练时若该文件存在即自动恢复——

```bash
# 训练中断后，原命令重跑即可，自动从 latest_checkpoint.pth 续训
python VMUNet-DiceCE-AdamW/pipeline/train.py --resolution 2.0 --cache --batch 4 --epochs 100 --val_interval 5 --device cuda
```

- 恢复后从断点 epoch 继续，**已验证过的 epoch 不会重复验证**（依据 checkpoint 内的 `last_validated_epoch`）。
- 验证阶段 OOM 等异常不会丢失进度：训练状态已保存，下次运行会重新尝试验证。
- 兼容旧格式 checkpoint：若发现旧格式（仅权重），自动从编号最大的 `epoch_*.pth` 恢复权重，但优化器/调度器状态丢失。
- **想彻底重训**：先手动移走或删除 `pipeline/models/2.0mm_cuda/latest_checkpoint.pth`，否则会直接续训。

### 6.3 评估（--cache + --checkpoint）

```bash
# 默认加载 pipeline/models/2.0mm_cuda/best_metric_model.pth（--checkpoint 可省略，但建议显式写）
python VMUNet-DiceCE-AdamW/pipeline/evaluate.py --resolution 2.0 --cache --checkpoint "VMUNet-DiceCE-AdamW/pipeline/models/2.0mm_cuda/best_metric_model.pth" --device cuda --workers 4
```

- 结果 CSV（固定命名，见 §8）：`pipeline/evaluation_results/metrics_vm_unet_2.0mm.csv` 与 `metrics_vm_unet_2.0mm_lesion.csv`。
- ⚠️ `--checkpoint` 默认值**固定指向 2.0mm_cuda**：评估其他分辨率的权重时必须显式传 `--checkpoint`。
- ⚠️ `--resolution` 必须与缓存目录及权重的训练分辨率一致。

### 6.4 快速 2-epoch 冒烟测试

```bash
# 2 epoch 小样验证环境与数据链路；--model_dir 隔离输出目录，避免污染正式 checkpoint
python VMUNet-DiceCE-AdamW/pipeline/train.py --epochs 2 --workers 0 --resolution 2.0 --device cuda --cache --val_interval 1 --batch 1 --model_dir VMUNet-DiceCE-AdamW/pipeline/models/smoke_2.0_cuda --log_name smoke_ultralight
```

正常表现：能读到 875/188 样本数、跑完 2 个 epoch 并各验证一次、在 `smoke_2.0_cuda/` 下生成权重文件。

## 7. 完整 CLI 参数表（照抄 argparse）

### pipeline/train.py

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `vm_unet` | 模型名称 |
| `--resolution` | float | `1.0` | 数据分辨率（1.0 / 2.0 / 3.0 mm，决定缓存目录与 pixdim） |
| `--workers` | str | `auto` | 数据加载线程数（`auto` / 数字；auto 表示自动计算） |
| `--batch` | str | `4` | 批次大小（`auto` / 数字；auto 表示自动计算） |
| `--cache` | flag | False | 启用硬盘缓存（**必须**，否则报错） |
| `--swa` | flag | False | 启用权重滑动平均（SWA 等权；历史命名 EMA，CLI 已改名） |
| `--epochs` | int | `100` | 训练轮数 |
| `--val_interval` | int | `5` | 验证间隔（每 N 个 epoch 验证一次） |
| `--device` | str | None | 设备（cuda/cpu，None=自动检测） |
| `--model_dir` | str | `""` | 自定义权重保存目录（默认 `pipeline/models/{分辨率}mm_{设备}`） |
| `--log_name` | str | `""` | 自定义日志名称前缀（默认 `train_ultralight_{分辨率}mm`） |
| `--seed` | int | `42` | 训练随机种子（--deterministic 时生效，不影响数据划分缓存） |
| `--deterministic` | flag | False | 确定性训练：固定种子 + cudnn.deterministic + 确定性算法 + CUBLAS_WORKSPACE_CONFIG（不触碰 TF32，保持论文口径可比） |
| `--cache_parent` | str | `""` | 缓存父目录覆盖：命令行 > 环境变量 SRTP_CACHE_PARENT > 历史默认；最终目录 = `{parent}_{分辨率}` |
| `--skip_nonfinite` | flag | False | 跳过非有限 loss（NaN/Inf）的 batch：不 backward/不更新/不进平均权重，epoch 指标按有效 step 平均 |
| `--best_weights` | str | `raw` | best_metric_model.pth 保存来源：raw=原始权重（历史口径）/ avg=验证所用的平均权重（SWA，需 --swa，否则回退 raw） |

固定超参（写死在代码中，不可经 CLI 修改）：学习率 `1e-4`、权重衰减 `1e-5`、DiceCELoss、AdamW、CosineAnnealingLR（T_max=epochs）、混合精度（AMP autocast + GradScaler）、滑窗推理 `sw_batch_size=4, overlap=0.5, mode=gaussian`。

### pipeline/evaluate.py

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `vm_unet` | 模型名称 |
| `--resolution` | float | `1.0` | 数据分辨率（1.0 / 2.0 / 3.0 mm） |
| `--checkpoint` | str | `{项目根}/pipeline/models/2.0mm_cuda/best_metric_model.pth` | 检查点路径 |
| `--workers` | str | `auto` | 数据加载线程数（`auto` / 数字） |
| `--cache` | flag | False | 启用硬盘缓存（**必须**，否则报错） |
| `--device` | str | None | 设备（cuda/cpu，None=自动检测） |
| `--no_lesion_wise` | flag | False | 禁用 lesion-wise 指标 |
| `--runs` | int | `1` | 评估次数（>1 时输出均值±标准差与浮动百分比） |
| `--cache_parent` | str | `""` | 缓存父目录覆盖：命令行 > 环境变量 SRTP_CACHE_PARENT > 历史默认；最终目录 = `{parent}_{分辨率}` |
| `--data_root` | str | `""` | 原始 BraTS TrainingData 目录覆盖：命令行 > 环境变量 BRATS_DATA_ROOT > 历史默认 |

> 以上 `--deterministic` / `--cache_parent` / `--skip_nonfinite` / `--best_weights`（及 evaluate 的 `--cache_parent` / `--data_root`）为 2026-10-08 跨卡实验包合并引入的可选开关，默认关闭时不带任何新参数的命令行为与历史逐位一致；换机迁移用命令行参数或环境变量重定向，不改变各项目历史数据划分绑定。

## 8. 目录结构与输出产物

```
VMUNet-DiceCE-AdamW/
├── __init__.py
├── README.md                     # 本文件
├── models/
│   └── __init__.py               # UltraLightVMUNet3D / ResBlock / get_model
├── frame/
│   ├── __init__.py
│   └── train.py                  # UltraLightTrainer（损失/优化器/验证/lesion-wise 写 CSV）
└── pipeline/
    ├── __init__.py
    ├── train.py                  # 训练入口（含断点续训）
    ├── evaluate.py               # 评估入口（Dice/HD95 + lesion-wise）
    ├── logs/                     # 训练/评估日志（train_ultralight_*、eval_vm_unet_*）
    ├── models/                   # 权重输出（运行后生成）
    │   └── 2.0mm_cuda/           # {分辨率}mm_{设备} 命名
    │       ├── epoch_{N}.pth             # 每个 val_interval 保存一次
    │       ├── best_metric_model.pth     # 验证 WT Dice 最高的权重
    │       └── latest_checkpoint.pth     # 断点续训检查点（每 epoch 更新）
    └── evaluation_results/       # 评估 CSV（固定命名，重评覆盖）
        ├── metrics_vm_unet_2.0mm.csv            # 逐病例全局指标
        ├── metrics_vm_unet_2.0mm_lesion.csv     # 逐病例 lesion-wise 指标
        └── *_old.csv                            # 人工改名留档的历史结果
```

**CSV 命名规则**：`metrics_{model_name}_{分辨率}mm.csv` 与 `metrics_{model_name}_{分辨率}mm_lesion.csv`。前者列为 `CaseID, Dice_WT, Dice_TC, Dice_ET, HD95_WT, HD95_TC, HD95_ET`；后者为 lesion-wise 指标，且比 VSSUNet 版本**多出病灶计数列**（`GT_WT, FP_WT, GT_TC, FP_TC, GT_ET, FP_ET`，标注真值病灶数与假阳性病灶数）。

## 9. 实验结果（2026-09-30 评估，188 例测试集，2.0mm 缓存划分）

| 指标 | WT | TC | ET |
|------|------|------|------|
| **全局 Dice** | 0.9109 | 0.8829 | 0.8376 |
| **全局 HD95 (mm)** | 5.4930 | 4.8003 | 4.0896 |
| **LW Dice** | 0.8509 | 0.8384 | 0.7691 |
| **LW HD95 (mm)** | 30.4720 | 27.2911 | 41.2797 |

结果解读：

- **全局 Dice 与全局 HD95 在轻量组中最好**：三项全局 Dice 均高于 VSSUNet，全局 HD95 三区域均低于 VSSUNet。
- **LW Dice 优秀**：WT 0.8509，明显高于 VSSUNet 的 0.6692，病灶检测能力强。
- **LW HD95 异常高**（27~41mm）：与全局 HD95（4~5.5mm）严重背离，提示存在离群假阳性病灶拉高均值，引用该组数字时需注明口径。

## 10. 注意事项

1. **没有 Mamba，也没有 VSSBlock**：本模型是纯 CNN ResBlock（尽管名字带 "VSS"），请在 `SRTP` 环境运行，以本 README 为准。
2. **必须 `--cache`**：代码强制启用缓存，且要求缓存中已有 `split_info.json`（先跑 `shared/data/prepare_data.py`）。
3. **评估 CSV 固定命名会被覆盖**：`metrics_vm_unet_*.csv` 每次评估都会重写；需要留档时先手动改名（库内已有 `_old` 后缀留档的先例）。
4. **`--checkpoint` 默认固定指向 2.0mm_cuda**：评估其他分辨率权重必须显式传路径。
5. **单卡串行训练纪律**：一次只跑一个训练任务（8GB 显存 + D 盘缓存 IO 共享），并行多任务会互相拖慢并可能 OOM。
6. **冒烟测试务必带 `--model_dir`**：否则 2-epoch 测试会写入正式目录的 `latest_checkpoint.pth`，污染断点续训状态。
7. **`epoch_*.pth` 会随训练增多**：每 5 个 epoch 一个文件，长训练注意磁盘空间；只需最终结果时可保留 `best_metric_model.pth` 与 `latest_checkpoint.pth`。
8. **HD95 空边界口径**：逐病例空预测 vs 空标注记 0（完美），一侧为空记 374mm 满额惩罚，nan/inf 兜底为 374（与 lesion-wise 口径一致）。
9. **参数量以代码实测为准**：524,044（约 0.52M），历史文档中 ~0.8M / ~0.3M 的写法不再沿用。
