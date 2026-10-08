# SegResNet-DiceCE-AdamW：CNN 基线（Baseline）

> **一句话定位**：NVIDIA BraTS 2018 冠军方案 SegResNet（MONAI 内置实现）作为纯 CNN 基线，DiceCELoss + AdamW 训练，为 SRTP 全部 Mamba 项目提供 CNN 性能参照——**无任何 Mamba / Transformer 组件**。
>
> **数据集**：BraTS 2023 GLI（脑胶质母细胞瘤）训练集 1251 例，2.0mm 重采样，188 例测试集评估。

## 一、项目定位与同级项目关系

本项目是 SRTP 论文的 **Baseline 对照项目**：所有 Mamba 模型（SegResMamba-Lite、VMUNet、VSSUNet、SegMamba-Official、LightSegMamba 等）都与它对比，以回答"引入 Mamba 后到底带来多少收益"。同时它的 ResBlock 设计是主项目 SegResMamba-Lite 中 LightResBlock 的灵感来源。

| 项目 | 角色 | 与本项目关系 |
|------|------|--------------|
| **SegResNet-DiceCE-AdamW（本项目）** | CNN 基线（DiceCE 损失） | — |
| SegResNet-DiceFocal-AdamW-SWA-DS | CNN 基线（DiceFocal + SWA + 深监督变体） | 损失/训练策略对照 |
| SegResMamba-Lite | 自研主模型（Mamba 混合） | 主要对比对象；LightResBlock 源自本模型 ResBlock |
| VMUNet / VSSUNet-DiceCE-AdamW | 轻量 Mamba 对照 | 共用同一套 shared 训练/评估体系 |
| SegMamba-Official | 官方 SegMamba 复现 | 重量级对照 |
| shared/ | 公共库 | 数据加载、损失、优化器、评估指标全部复用 |

## 二、新手背景：任务与指标（通俗版）

- **任务**：给一张 3D 脑部 MRI，自动把肿瘤像素出来。每个病例有 **4 种模态**（同一部位的 4 种扫描参数）：t1n（平扫 T1）、t1c（增强 T1）、t2w（T2 加权）、t2f（FLAIR），拼成 4 通道输入。
- **分割区域**（BraTS 官方三区域，由类别标签组合而成）：
  - **WT**（Whole Tumor，整体肿瘤）：全部肿瘤区域
  - **TC**（Tumor Core，肿瘤核心）：核心坏死 + 强化部分
  - **ET**（Enhancing Tumor，强化肿瘤）：增强扫描下显影的活性部分
- **指标**：**Dice**（预测与真值重叠率，0~1，越高越好）；**HD95**（95 分位 Hausdorff 距离，衡量预测边界偏了多少，越低越好）。本项目**全局 HD95 为 3D mm 口径**（MONAI 3D 计算、传入 spacing）；lesion-wise 指标为 BraTS 官方口径。
- **数据划分**：train 875 / val 249（训练中验证）/ test 188（评估用），seed=42 固定，与全部项目一致。

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

## 三、模型结构

### 3.1 整体架构（MONAI SegResNet，init_filters=16，实测通道/尺寸）

```
输入 (B, 4, 64, 64, 64)
   │
convInit: Conv3d(4→16) + GroupNorm + ReLU                         (16 @ 64³)
   │
down1: 1× ResBlock（无下采样）────────────────────────────────── skip3 (16 @ 64³)
down2: Conv3d(s=2) → 32 + 2× ResBlock ───────────────────────── skip2 (32 @ 32³)
down3: Conv3d(s=2) → 64 + 2× ResBlock ───────────────────────── skip1 (64 @ 16³)
down4: Conv3d(s=2) → 128 + 4× ResBlock                          (128 @ 8³，bottleneck)
   │
up1: trilinear 上采样→16³ + (+) skip1 + 1× ResBlock             (64 @ 16³)
up2: trilinear 上采样→32³ + (+) skip2 + 1× ResBlock             (32 @ 32³)
up3: trilinear 上采样→64³ + (+) skip3 + 1× ResBlock             (16 @ 64³)
   │
conv_final: 1×1×1 Conv3d → 4 通道 logits (B, 4, 64, 64, 64)
```

> 结构为实测（`encode`/`decode` 前向输出）：编码 4 级 blocks `(1, 2, 2, 4)`（第一级不下采样），解码 3 级 blocks `(1, 1, 1)`；**跳跃连接为逐元素相加**（`up(x) + skip`，通道数不变），非拼接。旧版 README 中"down4 位于 4³/256、解码 4 级"的图与 MONAI 实现不符，已按实测修正。

### 3.2 关键特点

| 特点 | 说明 |
|------|------|
| ResBlock | 双 3×3×3 卷积 + 残差连接的标准编码器块 |
| GroupNorm | 归一化用 GroupNorm，对小 batch 训练更稳定 |
| 相加 skip | `up(x) + skip` 相加融合，不增加解码器通道 |
| trilinear 上采样 | 非可学习插值（`upsample_mode="nontrainable"`），简单省参 |
| dropout | `dropout_prob=0.2`（3D dropout 正则） |
| 参数量 | **4,702,244 ≈ 4.70M**（init_filters=16 实测；无 Mamba） |

`models/__init__.py` 的 `get_model` 还支持 `unet`（MONAI UNet）与 `swin_unetr`（MONAI SwinUNETR）两种备用模型，通过 `--model` 切换；本 README 的配置与结果均针对默认的 `segresnet`。

## 四、环境说明

| 项 | 说明 |
|----|------|
| **conda 环境** | `SRTP`（纯 CNN，无 Mamba 依赖） |
| **硬件口径** | RTX 5060 Laptop 8GB；2.0mm + batch=2 显存余量充足 |
| **关键依赖** | monai、torch（CUDA 12.8 系） |

```bash
conda activate SRTP
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code   # 后续命令均在此目录执行
```

## 五、数据缓存约定（2026-10-07 全库统一定案）

- **默认缓存父目录**：`D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache`
- **实际目录**：按分辨率拼接，2.0mm → `persistent_cache_2.0`
- **整机重定向**：环境变量 `SRTP_CACHE_PARENT` 可覆盖默认父目录
- **必须显式 `--cache`**：训练/评估均要求显式启用缓存（未生成会报错并提示先跑 prepare_data）
- **预处理命令**（生成缓存，shared/data/prepare_data.py）：

```bash
python shared/data/prepare_data.py --resolution 2.0 --generate --cache --workers 8
```

- 划分固定 seed=42（train 875 / val 249 / test 188），写入缓存目录 `split_info.json`，与主项目体系一致，跨项目数字可直接对比。

## 六、命令行

### 6.1 训练（正式口径：2.0mm / 100 epochs / batch=2）

```bash
conda activate SRTP
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code

python SegResNet-DiceCE-AdamW/pipeline/train.py --epochs 100 --workers 8 --resolution 2.0 --device cuda --cache --batch 2
```

> 注意 `--resolution` 代码默认值为 1.0，正式实验请显式传 `--resolution 2.0`（权重目录名随分辨率变化）。

### 6.2 恢复训练（断点续训）

**机制（与代码一致）**：每个 epoch 结束都把完整状态（模型权重 + 优化器 + 调度器 + AMP GradScaler + 当前 epoch + 历史最佳 + 已验证 epoch 记录）存入权重目录的 `latest_checkpoint.pth`。训练中断后，**原样重跑同一条训练命令**即可：

```bash
# 中断前用什么命令，中断后就原样重跑什么命令
python SegResNet-DiceCE-AdamW/pipeline/train.py --epochs 100 --workers 8 --resolution 2.0 --device cuda --cache --batch 2
```

启动时打印 `发现完整检查点：...latest_checkpoint.pth，正在恢复训练...`，自动从断点 epoch 继续训练：

- `last_validated_epoch` 机制：恢复后对"已成功验证过"的 epoch 跳过验证，避免重复；验证失败的 epoch 不写入该记录，下次运行会自动重试。
- 旧格式 checkpoint 兼容：若检测到不含 `model_state_dict` 的旧格式文件，自动改从目录内最新的 `epoch_N.pth` 恢复模型权重（优化器/调度器状态丢失）。
- 从零重训：先删除权重目录中的 `latest_checkpoint.pth`（或用 `--model_dir` 换新目录）。

### 6.3 评估

```powershell
# PowerShell 单行命令（可直接粘贴运行）
python SegResNet-DiceCE-AdamW/pipeline/evaluate.py --resolution 2.0 --device cuda --cache --workers 8 --checkpoint "SegResNet-DiceCE-AdamW/pipeline/models/2.0mm_cuda/best_metric_model.pth"
```

- `--checkpoint` 不传时默认取 `pipeline/models/2.0mm_cuda/best_metric_model.pth`（**注意默认目录固定为 2.0mm**，其他分辨率必须显式传路径）。
- 输出（`pipeline/evaluation_results/`）：
  - `metrics_segresnet_2.0mm.csv`——逐病例全局 Dice/HD95（WT/TC/ET 六列）
  - `metrics_segresnet_2.0mm_lesion.csv`——逐病例 lesion-wise 指标（BraTS 官方口径）
  - `--runs > 1` 时输出多次评估的均值 ± 标准差（此时不写 CSV，避免多次结果混写）

### 6.4 快速 2-epoch 冒烟测试

```bash
# 本机已生成 2.0mm 缓存（本机无 3.0mm 缓存，勿用 --resolution 3.0）
python SegResNet-DiceCE-AdamW/pipeline/train.py --epochs 2 --workers 0 --resolution 2.0 --device cuda --cache --val_interval 1 --batch 1
```

## 七、完整 CLI 参数表

### 7.1 train.py（python SegResNet-DiceCE-AdamW/pipeline/train.py）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `segresnet` | 模型名称（segresnet/unet/swin_unetr，见 models/\_\_init\_\_.py） |
| `--resolution` | float | `1.0` | 数据分辨率（mm）；正式实验显式传 2.0 |
| `--workers` | str | `auto` | `auto`=自动计算，或数字字符串（如 `8`） |
| `--batch` | str | `auto` | `auto`=自动计算，或数字字符串（如 `2`） |
| `--cache` | flag | False | 启用硬盘缓存（必须） |
| `--swa` | flag | False | 启用权重滑动平均（等权 SWA；历史命名 EMA，CLI 已改名） |
| `--epochs` | int | `100` | 训练轮数 |
| `--val_interval` | int | `5` | 每 N epoch 验证一次 |
| `--device` | str | `None` | 设备（cuda/cpu，None=自动） |
| `--model_dir` | str | `""` | 自定义权重目录（默认 `pipeline/models/{分辨率}mm_{设备}`） |
| `--log_name` | str | `""` | 自定义日志名前缀（默认 `train_baseline_{分辨率}mm`） |

> 训练超参（固定，无 CLI）：DiceCELoss、AdamW lr=1e-4、weight_decay=1e-5、CosineAnnealingLR、AMP 混合精度（GradScaler + autocast）。

### 7.2 evaluate.py（python SegResNet-DiceCE-AdamW/pipeline/evaluate.py）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `segresnet` | 模型名称（与训练一致） |
| `--resolution` | float | `1.0` | 数据分辨率（mm）；评估 2.0mm 实验显式传 2.0 |
| `--checkpoint` | str | `pipeline/models/2.0mm_cuda/best_metric_model.pth` | 权重路径（默认目录固定 2.0mm_cuda） |
| `--workers` | int | `8` | DataLoader worker 数（注意：与 train.py 的 str 类型不同，此处是 int） |
| `--cache` | flag | False | 启用硬盘缓存（必须） |
| `--device` | str | `None` | 设备（cuda/cpu） |
| `--no_lesion_wise` | flag | False | 禁用 lesion-wise 指标 |
| `--runs` | int | `1` | 评估次数（>1 时输出均值 ± 标准差） |

## 八、目录结构与输出产物

```
SegResNet-DiceCE-AdamW/
├── README.md                  # 本文件
├── models/
│   └── __init__.py            # get_model：SegResNet（默认）/ UNet / SwinUNETR
├── frame/
│   └── train.py               # BaselineTrainer（DiceCE + AdamW + CosineLR + AMP + 可选 SWA）
└── pipeline/
    ├── train.py               # 训练入口
    ├── evaluate.py            # 评估入口（全局 + lesion-wise）
    ├── models/                # 权重（运行后生成）：{分辨率}mm_{设备}/
    │   └── 2.0mm_cuda/        #   best_metric_model.pth / latest_checkpoint.pth / epoch_N.pth（每次验证保存）
    ├── logs/                  # 日志：train_baseline_{分辨率}mm_*.log
    └── evaluation_results/    # 评估 CSV：metrics_segresnet_{分辨率}mm.csv + _lesion.csv
```

## 九、实验结果（2.0mm，100 epochs，测试集 188 例，seed=42）

| 指标 | WT | TC | ET |
|------|------:|------:|------:|
| **全局 Dice** | 0.9203 | 0.8892 | 0.8516 |
| **全局 HD95 (mm)** | 5.17 | 8.29 | 13.73 |
| **Lesion-wise Dice** | 0.8121 | 0.7815 | 0.7165 |

> 全局 HD95 为 **3D mm 满罚口径**（2026-10-08 重评；逐例 CSV：空-空记 0、一空一非空记 374 mm 满额惩罚，TC 2 例 / ET 5 例，与其他基线项目及 lesion-wise 口径一致；上表取逐例 CSV 均值）。训练验证与日志汇总行为 MONAI 聚合口径（空边界 inf→0），两套数字含义不同勿混用。引用本表请连同口径一并注明。

## 十、注意事项

1. **作为 Baseline 的意义**：提供"成熟 CNN + 标准训练策略"的性能参照；SegResMamba-Lite 的 LightResBlock 灵感来源于本模型的 ResBlock + GroupNorm 设计。
2. **最佳模型选择标准**：训练验证以 **WT Dice** 为最佳标准（`metrics["dice_wt"]`，与主项目 SegResMamba-Lite 用三区域平均 Dice 的标准不同，复现实验时注意）。
3. **`--workers` / `--batch` 类型不一致**：train.py 中两者是字符串（`auto`/数字），evaluate.py 中 `--workers` 是 int——写脚本时不要混用。
4. **分辨率默认值**：train/evaluate 的 `--resolution` 默认都是 1.0，而评估的默认 checkpoint 目录固定为 `2.0mm_cuda`——正式实验一律显式传 `--resolution 2.0` 与明确的 `--checkpoint`。
5. **串行训练纪律**：单卡 8GB，同一时间只跑一个训练任务；训练与评估共用 GPU 时注意显存（评估滑窗推理 sw_batch_size=4、overlap=0.5、gaussian 加权）。
6. **评估 CSV 覆盖**：`metrics_segresnet_2.0mm*.csv` 为固定命名，每次评估会**重写覆盖**；需要留存的逐例结果请先备份或记录评估时间戳。
7. **空边界满罚（逐例 CSV，2026-10-08 口径修复）**：逐例 CSV 中空预测的 Dice 记 1.0；HD95 空-空记 0.0、一空一非空记 374 满额惩罚（`frame/train.py` 的 `_safe_hd95`，与 DiceFocal/VMUNet/VSSUNet 基线一致）；日志汇总行由 MONAI metric 聚合（空边界 inf→0），两套数字含义不同，勿混用。
