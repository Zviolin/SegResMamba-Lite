# SegResNet-DiceFocal-AdamW-SWA-DS：优化版 CNN 基线（DiceFocal + SWA + 深层监督）

> **定位**：在 Baseline（`SegResNet-DiceCE-AdamW`，纯 CNN）基础上换损失、加权重平均、加深层监督的**优化版基线**。
> **论文背景**：Myronenko A. *Automated 3D Segmentation of Brain Tumors using Deep Neural Networks*（BraTS 2019 冠军方案，MONAI `SegResNet` 出处）；DiceFocal = Dice + Focal 组合损失（Focal 项出自 Lin T.-Y. et al., *Focal Loss for Dense Object Detection*, ICCV 2017；组合实现为 MONAI `DiceFocalLoss`）。
> **数据集**：BraTS 2023 GLI（成人胶质瘤，T1n / T1c / T2w / T2f 四模态，1251 例）。

## 1. 项目定位与仓库关系

本项目是仓库 7 个分割项目中唯一的 **DiceFocal 优化版 CNN 基线**，不含任何 Mamba 组件：

| 项目 | 模型 | 与本项目关系 |
|------|------|-------------|
| SegResNet-DiceCE-AdamW | SegResNet（CNN） | **直接对照 Baseline**（相同骨干，DiceCE 损失、无 SWA/DS） |
| **本项目** | SegResNet（CNN） | DiceFocal 损失 + SWA 权重平均 + 深层监督开关 |
| VSSUNet-DiceCE-AdamW / VMUNet-DiceCE-AdamW | 轻量 VSS | 轻量化对照 |
| SegMamba-Official | SegMamba（Mamba，本地实测 66.90M） | 官方 Mamba 复现，性能上限参照 |
| SegResMamba-Lite / LightSegMamba-DiceCE-AdamW-V3 | 轻量 Mamba | 轻量化 Mamba 对照 |

## 2. 新手背景：任务与指标

- **任务**：3D 脑肿瘤分割。输入每位病人 4 个 MRI 模态（T1n 平扫、T1c 增强、T2w、T2f Flair），模型输出逐体素的肿瘤分区。
- **三个评估区域**：WT（全肿瘤：所有肿瘤区域）、TC（肿瘤核心：切除腔+增强+坏死）、ET（增强肿瘤：强化最活跃部分）。标签经 4 类 → 3 区映射（`brats_label_mapping`）后计算。
- **Dice**：预测与标注的重合度，0~1，越高越好。
- **HD95 (mm)**：预测边界与真实边界距离的第 95 百分位，越低越好；空预测/空标注按满额惩罚 374 mm 记（与 lesion-wise 口径一致），避免"漏检反而得 0 距离"的假优。
- **数据划分**：训练/验证/测试 = 70%/15%/15%（seed=42），划分固化在缓存目录的 `split_info.json` 中，训练与评估必须使用同一份缓存。

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
- **计算 Loss**：用 DiceFocalLoss（Dice 重合度 + Focal 难例加权）衡量「画的圈」与医生标注的差距，重合度越高 Loss 越小。
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

### 3.1 骨干：MONAI SegResNet

```
输入 (B, 4, 64, 64, 64)
  │
  ├─ conv_init: 3×3×3 Conv + InstanceNorm + ReLU ── 16 通道
  │
  ├─ 编码器 down1→down4：残差块 + 2×2×2 下采样，通道 16→32→64→128→256
  │
  ├─ 解码器 up4→up1：反卷积上采样（可学习）+ 跳跃连接 + 残差块，通道 256→16
  │
  └─ conv_out: 1×1×1 Conv → 4 通道 logits（背景 + 3 类肿瘤分区）
```

配置：`init_filters=16`、`dropout_prob=0.2`（MONAI 默认块数）。参数量约 4.7M（以训练启动日志打印为准）。

### 3.2 三项优化（相对 Baseline）

| 优化项 | CLI | 实现 | 作用 |
|--------|-----|------|------|
| DiceFocal 损失 | 默认（硬编码） | `shared/losses/dice_focal_loss.py`，`lambda_dice=1.0, lambda_focal=1.0, gamma=2.0` | Focal 项降权易分类体素、聚焦难样本，缓解背景远大于肿瘤的类别不平衡 |
| SWA 等权权重平均 | `--swa` | `shared/optim/swa.py` 的 `SWA` 类（包装 `torch.optim.swa_utils.AveragedModel`） | 平滑权重波动，验证与论文口径均用平均权重 |
| 深层监督 | `--ds` | `frame/train.py` 多尺度损失；`models/get_model` 支持 `SegResNetDS(dsdepth=3)` | 多尺度监督（代码现状见 §8 注意事项） |

**DiceFocal**：`L = 1.0×Dice + 1.0×Focal`，其中 `Focal = -(1-p)^γ·log(p)`（γ=2.0，softmax、squared_pred、smooth_dr=1e-6）。

**SWA 等权平均（历史命名"EMA"实为 SWA）**：

```
每次 optimizer.step() 后：
  θ_avg ← (θ_1 + θ_2 + … + θ_n) / n     ← 等权累计平均，非指数加权
验证 / 评估 / 论文口径 → 使用 θ_avg（averaged_model）
```

- 每次验证若刷新最佳 WT Dice，同时保存原始权重 `best_metric_model.pth` 与平均权重 `best_metric_swa_model.pth`。
- 历史命名说明：该实现的代码/CLI/checkpoint 键曾统一叫 "EMA"，实际语义一直是等权 SWA（decay 从未生效）；2026-10-05 起全面改名为 SWA，旧文件名 `best_metric_ema_model.pth`、旧 checkpoint 键 `ema_state_dict` 在断点恢复与评估端均兼容；真指数移动平均由 `shared/optim/ema.py` 承载，本项目不用。

**深层监督（--ds）**：`frame/train.py` 的损失计算在模型输出为多尺度列表时，按 `[1.0, 0.5, 0.25, 0.125]` 加权求和，低分辨率监督目标用最近邻插值对齐；`models/get_model` 提供 `SegResNetDS`（MONAI，`dsdepth=3`）输出多尺度预测。⚠️ 当前 `pipeline/train.py` 的实际行为见 §8 注意事项。

### 3.3 训练配置（pipeline/train.py 硬编码）

| 配置 | 值 |
|------|-----|
| 优化器 | AdamW，lr=1e-4，weight_decay=1e-5 |
| 学习率调度 | CosineAnnealingLR（T_max=epochs） |
| 混合精度 | `torch.amp.autocast` + `GradScaler`（cuda） |
| 推理 | 滑动窗口：sw_batch=4、overlap=0.5、gaussian 融合 |
| 验证指标 | WT/TC/ET 的 Dice 与 HD95，另有 lesion-wise（BraTS 2023 官方公式） |

## 4. 环境说明

- **Conda 环境**：`conda activate SRTP`（非 Mamba 项目专用环境；本项目无 mamba_ssm 依赖）。
- **显卡口径**：RTX 5060 Laptop 8GB。2.0mm 缓存 + 64³ patch 下 `--batch 2` 可跑；OOM 时降到 `--batch 1` 或 `--batch auto`（自动计算）。

## 5. 数据缓存约定（2026-10-07 全库统一定案）

- **训练/评估都必须带 `--cache`**，否则直接报错退出（代码强制要求已生成缓存）。
- 缓存根目录（代码默认）：`D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache`，按分辨率拼接后实际目录为 `persistent_cache_2.0`（2.0mm）。
- 整机迁移可用环境变量 `SRTP_CACHE_PARENT` 重定向父目录（代码内默认路径仅作兜底）。
- 注意仓库存在**两份划分不同的 2.0mm 缓存**（主项目体系 vs Essay\Data LightSegMamba 体系，同为 seed=42 但 test 集交集仅 21/188），本项目走主项目体系，**不可与 LightSegMamba 体系的评估数字直接对比**。
- 首次生成缓存（预处理入口）：

```bash
conda activate SRTP
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code
python shared/data/prepare_data.py --resolution 2.0 --generate --cache --workers 8
```

## 6. 命令行

### 6.1 训练

```powershell
conda activate SRTP
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code

# 推荐配置（论文口径：SWA + DS 均开启）；PowerShell 单行命令（可直接粘贴运行）
python SegResNet-DiceFocal-AdamW-SWA-DS/pipeline/train.py --resolution 2.0 --cache --epochs 150 --val_interval 5 --batch 2 --swa --ds --device cuda
```

### 6.2 恢复训练（断点续训）

训练**每个 epoch 结束后**都会把完整训练状态（模型权重、优化器、学习率调度器、GradScaler、SWA 平均权重、当前 epoch、最佳指标、最后验证 epoch）写入权重目录下的 `latest_checkpoint.pth`。

- **中断后恢复**：用与原训练**完全相同的命令**重新运行即可（默认权重目录由 `--resolution` 与 `--device` 推导为 `pipeline/models/2.0mm_cuda/`；若原训练传了 `--model_dir`，续训也必须传同一个值，保证找到同一份 `latest_checkpoint.pth`）。启动时检测到该文件会自动恢复并从断点 epoch 继续训练。
- **验证中断保护**：验证阶段失败（如 OOM）时异常被捕获、该次验证跳过，但训练状态已在验证前保存到断点，恢复训练从下一 epoch 继续，不会卡在同一处反复失败。
- **旧格式兼容**：只有旧 `epoch_N.pth` 权重时也能恢复（配合 `best_metric_swa_model.pth` / 旧名 `best_metric_ema_model.pth` 恢复平均权重），但优化器/调度器状态丢失，epoch 计数从 0 重新开始（权重不重训、轮数重计）。
- **放弃断点重头训**：先删除或移走 `pipeline/models/2.0mm_cuda/latest_checkpoint.pth`。

```powershell
# 例：中断后原命令重跑即自动续训（与 6.1 完全一致）
python SegResNet-DiceFocal-AdamW-SWA-DS/pipeline/train.py --resolution 2.0 --cache --epochs 150 --val_interval 5 --batch 2 --swa --ds --device cuda
```

### 6.3 评估

评估读取缓存目录 `test/` 子目录与 `split_info.json` 的测试集划分；**必须显式传 `--checkpoint`**（`--checkpoint` 文件名含 `swa` 或 `ema` 时，CSV 自动加 `_swa` / `_ema` 后缀区分口径）：

```powershell
# 评估平均权重（论文口径）
python SegResNet-DiceFocal-AdamW-SWA-DS/pipeline/evaluate.py --resolution 2.0 --cache --device cuda --checkpoint SegResNet-DiceFocal-AdamW-SWA-DS/pipeline/models/2.0mm_cuda/best_metric_swa_model.pth

# 评估原始权重（对照口径）
python SegResNet-DiceFocal-AdamW-SWA-DS/pipeline/evaluate.py --resolution 2.0 --cache --device cuda --checkpoint SegResNet-DiceFocal-AdamW-SWA-DS/pipeline/models/2.0mm_cuda/best_metric_model.pth
```

不传 `--checkpoint` 时使用代码内默认路径 `pipeline/models/2.0mm_cuda/best_metric_model.pth`。

### 6.4 快速冒烟测试（2 epoch）

```powershell
python SegResNet-DiceFocal-AdamW-SWA-DS/pipeline/train.py --resolution 2.0 --cache --epochs 2 --val_interval 1 --batch 1 --workers 0 --swa --ds --device cuda --model_dir SegResNet-DiceFocal-AdamW-SWA-DS/pipeline/models/smoke_test
```

用 `--model_dir` 把冒烟产物隔离到 `smoke_test` 目录，避免污染正式训练的断点文件。验证通过后删除该目录即可。

## 7. CLI 参数完整列表

### 7.1 pipeline/train.py

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `segresnet` | 模型名称 |
| `--resolution` | float | `1.0` | 数据分辨率（mm），决定缓存目录与权重目录后缀 |
| `--workers` | str | `auto` | 数据加载线程数（`auto`/数字） |
| `--batch` | str | `2` | 批次大小（`auto`/数字，`auto` 表示自动计算） |
| `--cache` | flag | False | 启用硬盘缓存（**必须**，否则报错退出） |
| `--swa` | flag | False | 启用 SWA 等权权重平均（历史命名 EMA，实际语义一直为等权 SWA） |
| `--ds` | flag | False | 启用深层监督（代码现状见 §8 注意事项） |
| `--epochs` | int | `150` | 训练轮数 |
| `--val_interval` | int | `5` | 验证间隔（每 N 个 epoch 验证一次） |
| `--device` | str | None | 设备（cuda/cpu，None=自动选择） |
| `--model_dir` | str | `""` | 自定义权重保存目录（空=默认 `pipeline/models/{分辨率}mm_{设备}`） |
| `--log_name` | str | `""` | 自定义日志名称前缀（空=默认 `train_optimized_{分辨率}mm`） |
| `--seed` | int | `42` | 训练随机种子（--deterministic 时生效，不影响数据划分缓存） |
| `--deterministic` | flag | False | 确定性训练：固定种子 + cudnn.deterministic + 确定性算法 + CUBLAS_WORKSPACE_CONFIG（不触碰 TF32，保持论文口径可比） |
| `--cache_parent` | str | `""` | 缓存父目录覆盖：命令行 > 环境变量 SRTP_CACHE_PARENT > 历史默认；最终目录 = `{parent}_{分辨率}` |
| `--skip_nonfinite` | flag | False | 跳过非有限 loss（NaN/Inf）的 batch：不 backward/不更新/不进平均权重，epoch 指标按有效 step 平均 |
| `--best_weights` | str | `raw` | best_metric_model.pth 保存来源：raw=原始权重（历史口径）/ avg=验证所用的平均权重（SWA，需 `--swa`，否则回退 raw） |

> 以上 `--deterministic` / `--cache_parent` / `--skip_nonfinite` / `--best_weights`（及 evaluate 的 `--cache_parent`/`--data_root`）为 2026-10-08 跨卡实验包合并引入的可选开关，默认关闭时不带任何新参数的命令行为与历史逐位一致；换机迁移用命令行参数或环境变量重定向，不改变各项目历史数据划分绑定。

### 7.2 pipeline/evaluate.py

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `segresnet` | 模型名称 |
| `--resolution` | float | `1.0` | 数据分辨率（mm） |
| `--checkpoint` | str | `<项目根>/pipeline/models/2.0mm_cuda/best_metric_model.pth` | 权重路径（文件名含 `swa`/`ema` 时 CSV 自动加后缀） |
| `--workers` | str | `auto` | 数据加载线程数（`auto`/数字） |
| `--cache` | flag | False | 启用硬盘缓存（**必须**） |
| `--device` | str | None | 设备（cuda/cpu，None=自动选择） |
| `--no_lesion_wise` | flag | False | 禁用 lesion-wise 指标 |
| `--runs` | int | `1` | 评估次数（>1 时额外报告均值±标准差，但逐病例 CSV 仅 runs=1 时写出） |
| `--cache_parent` | str | `""` | 缓存父目录覆盖：命令行 > 环境变量 SRTP_CACHE_PARENT > 历史默认；最终目录 = `{parent}_{分辨率}` |
| `--data_root` | str | `""` | 原始 BraTS TrainingData 目录覆盖：命令行 > 环境变量 BRATS_DATA_ROOT > 历史默认 |

## 8. 目录结构与输出产物

```
SegResNet-DiceFocal-AdamW-SWA-DS/
├── README.md
├── models/
│   └── __init__.py        # get_model：SegResNet / SegResNetDS(dsdepth=3)
├── frame/
│   └── train.py           # OptimizedTrainer：DiceFocal + AdamW + CosineLR + SWA + DS 损失
└── pipeline/
    ├── train.py           # 训练入口（CLI、断点续训）
    ├── evaluate.py        # 评估入口（Dice/HD95 + lesion-wise）
    ├── logs/              # 训练/评估日志：{前缀}_{时间戳}.log，前缀默认 train_optimized_{分辨率}mm / eval_segresnet_{分辨率}mm
    ├── models/            # 权重目录：{分辨率}mm_{设备}（如 2.0mm_cuda）
    │   ├── latest_checkpoint.pth      # 每 epoch 更新的断点（含优化器/调度器/SWA/最佳指标）
    │   ├── best_metric_model.pth      # 最佳 WT Dice 的原始权重
    │   ├── best_metric_swa_model.pth  # 最佳 WT Dice 的 SWA 平均权重（旧名 best_metric_ema_model.pth 兼容识别）
    │   └── epoch_N.pth                # 各验证点权重快照
    └── evaluation_results/
        ├── metrics_segresnet_2.0mm.csv           # 原始权重逐病例：CaseID,Dice_WT/TC/ET,HD95_WT/TC/ET
        ├── metrics_segresnet_2.0mm_lesion.csv    # lesion-wise 逐病例
        ├── metrics_segresnet_2.0mm_swa.csv       # checkpoint 名含 swa → 平均权重口径
        └── metrics_segresnet_2.0mm_ema.csv       # 历史口径（旧权重名含 ema 自动识别）
```

## 9. 实验参考结果（2.0mm，1251 例，本机历史评估）

> **口径区分**：下表为**原始权重**（`best_metric_model.pth`）评估结果；**论文口径**使用平均权重 `best_metric_swa_model.pth`（`metrics_segresnet_2.0mm_ema.csv`：全局 Dice 0.9130 / 0.8870 / 0.8118），两者不是同一份评估。

| 指标 | WT | TC | ET |
|------|------|------|------|
| 全局 Dice | 0.9211 | 0.8842 | 0.8488 |
| 全局 HD95 (mm) | 5.45 | 4.51 | 4.00 |
| Lesion-wise Dice | 0.7858 | 0.8292 | 0.7629 |

与 Baseline（SegResNet-DiceCE-AdamW）对比：Dice WT 略优（0.9211 vs 0.9203），HD95 WT 略差（5.45 vs 5.17）——权重平均与深层监督主要改善整体重合度，对边界精度改善有限。

## 10. 注意事项

1. **--ds 的代码现状**：`--ds` 会传入训练器（`frame/train.py` 的多尺度损失分支已实现，权重 1.0/0.5/0.25/0.125），但 `pipeline/train.py` 构建模型时调用 `get_model(model, device)` **未传** `use_deep_supervision`（默认 False → 构建**普通 SegResNet**，输出单张量），因此该损失分支当前不会触发——带不带 `--ds` 训练行为相同。历史实验与论文口径基于 `SegResNetDS(dsdepth=3)` 配置；如需真正启用 DS，需让训练入口把该开关传入 `get_model`。
2. **评估 CSV 覆盖**：结果文件名固定（`metrics_segresnet_{分辨率}mm[_swa|_ema].csv`），重复评估会直接覆盖旧 CSV；需要留档请先改名或备份。
3. **显存**：8GB 显存推荐 `--batch 2`（2.0mm/64³ patch）；验证阶段滑动窗口 + HD95 峰值更高，OOM 时先降 batch 或减小 patch。
4. **串行训练纪律**：单卡 8GB，同卡不要并行多个训练/评估任务，避免互抢显存导致 OOM 与断点损坏。
5. **缓存一致性**：训练与评估必须用同一份缓存（同一 `persistent_cache_2.0`）；跨体系缓存（LightSegMamba 体系）的 test 划分不同，数字不可混比。
6. **HD95 空边界口径**：空预测/空标注记 374mm 满额惩罚，inf/nan 兜底同值；这不是 bug，是为了不让"整区漏检"得到虚低的距离值。
