# SegMamba-Official：官方 SegMamba 复现（Mamba 系）

> **论文**：*SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation*. MICCAI 2024. arXiv:2401.13560（与代码注释一致的题录）。
> **官方代码**：https://github.com/ge-xing/SegMamba（本仓库按其公开实现复刻）。
> **数据集**：BraTS 2023 GLI（成人胶质瘤，T1n / T1c / T2w / T2f 四模态，1251 例）。

## 1. 项目定位与仓库关系

本项目是仓库 7 个分割项目中的**官方 SegMamba 复现**，也是 3 个 Mamba 项目中参数量最大的一个（本地实测 66.90M；官方论文报告 74.87M，计数口径差异，另两个轻量 Mamba 为自研）：

| 项目 | 模型 | 与本项目关系 |
|------|------|-------------|
| SegResNet-DiceCE-AdamW / SegResNet-DiceFocal-AdamW-SWA-DS | 纯 CNN | CNN 基线对照 |
| VSSUNet-DiceCE-AdamW / VMUNet-DiceCE-AdamW | 轻量 VSS | 轻量化对照 |
| **本项目** | SegMamba（66.90M 本地实测） | 官方 Mamba 复现，衡量 Mamba 路线的性能上限 |
| SegResMamba-Lite / LightSegMamba-DiceCE-AdamW-V3 | 轻量 Mamba（~1.4M） | 轻量化 Mamba，与本项目形成"上限 vs 可部署"对照 |

## 2. 新手背景：任务与指标

- **任务**：3D 脑肿瘤分割。输入每位病人 4 个 MRI 模态（T1n 平扫、T1c 增强、T2w、T2f Flair），输出逐体素的肿瘤分区，4 通道 logits（背景 + 3 类）。
- **三个评估区域**：WT（全肿瘤）、TC（肿瘤核心）、ET（增强肿瘤）；标签经 4 类 → 3 区映射（`brats_label_mapping`）后计算。
- **Dice**：预测与标注重合度（0~1，越高越好）；**HD95 (mm)**：边界距离第 95 百分位（越低越好，空预测/空标注记 374mm 满额惩罚）。
- **数据划分**：训练/验证/测试 = 70%/15%/15%（seed=42），固化在缓存目录 `split_info.json`；训练与评估必须用同一份缓存。

### 零基础速览：训练到底在做什么

整个训练就是「看图 → 画圈 → 对答案 → 调整」的循环，重复若干轮（每轮记 1 个 epoch）：

```
原始数据 → 预处理 → 模型预测 → 计算Loss → 反向传播 → 参数更新 ─┐
   ↑                                                      │
   └──── 验证循环 ←── 滑窗推理 ←─────────────────────────────┘
        （每 val_interval 轮一次，保存最佳权重）
```

- **预处理**：把每位病人的 4 张 MRI 统一方向（RAS）、重采样到 2.0mm、裁掉背景、归一化——相当于把所有考卷印成同一格式，模型才好学。
- **模型预测**：编码器把 64³ 立体图逐级压缩（空间变小、通道变多），像把一幅彩画浓缩成一本摘要书——知道「大概哪里有肿瘤」但丢了精确边界；解码器再逐级放大回原尺寸，并经跳跃连接把浅层「精细笔记」传回来补细节，最后输出每个体素属于 4 个类别（背景/NCR/ED/ET）的概率。本项目的 4 个 Stage 中每级都嵌入 Mamba 全序列扫描（见 §3）。
- **计算 Loss**：用 CrossEntropyLoss 衡量「画的圈」与医生标注的差距，分类越准 Loss 越小（论文官方口径）。
- **反向传播 + 参数更新**：`loss.backward()` 算出每个权重该往哪调；SGD 按设定的学习率更新权重；Poly 调度让学习率按多项式衰减（lr = base_lr × (1 − step/T)^0.9）——先大步学、后小步修。
- **验证**：每 val_interval 轮，用滑动窗口（64³ 窗口、50% 重叠、高斯加权）扫过整个验证病例算 Dice/HD95；WT Dice 最佳的权重存为 `best_metric_model.pth`，`latest_checkpoint.pth` 每个 epoch 都存（中断可续训）。
- **AMP 混合精度**：部分计算用半精度浮点省显存提速，GradScaler 防小梯度下溢。

| 术语 | 含义 |
|------|------|
| epoch | 全部训练数据完整过一遍 |
| batch | 一次同时送入模型的病例数 |
| patch | 每次随机裁出的 64³ 立方小块 |
| checkpoint | 可恢复训练的完整状态（权重 + 优化器 + 调度器 + AMP） |

## 3. 模型结构

### 3.1 整体架构（MambaEncoder + UNETR 风格解码器）

```
输入 (B, 4, D, H, W)
  │
  ├─ Stem: 7×7×7 Conv, stride=2 ───────────── 48 通道 ── enc1 跳跃
  │
  ├─ Stage1: GSC → Mamba×2 → InstanceNorm + MLP(channel 48) ── enc2 跳跃
  ├─ 下采样: InstanceNorm + 2×2×2 Conv stride=2
  ├─ Stage2: GSC → Mamba×2 → Norm + MLP ───── 96 通道 ── enc3 跳跃
  ├─ 下采样
  ├─ Stage3: GSC → Mamba×2 → Norm + MLP ───── 192 通道 ── enc4 跳跃
  ├─ 下采样
  ├─ Stage4: GSC → Mamba×2 → Norm + MLP ───── 384 通道
  │
  ├─ encoder5: UNETR ResBlock 384→768
  ├─ decoder5→decoder2: UNETR UpBlock（转置卷积上采样 + 对应跳跃融合）
  ├─ decoder1: UNETR ResBlock
  └─ out: 1×1×1 Conv → 4 通道 logits
```

- 4 个 Stage：特征维度 `[48, 96, 192, 384]`、深度 `[2, 2, 2, 2]`；**参数量 66,903,556 = 66.90M**（本地实例化实测；官方论文报告 74.87M 为论文计数口径）。
- Mamba 块：`shared/models/mamba.py` 的 `MambaLayer`——3D 特征展平为全序列 `(B, L=D×H×W, C)` → LayerNorm → 单方向 Mamba → 还原 3D → 残差。Mamba 参数 `d_state=16, d_conv=4, expand=2`。
- Mamba 后端优先级：`mamba_ssm.Mamba2`（CUDA 优化，**RTX 5060 支持**）→ MiniMamba（纯 PyTorch）→ `mamba_ssm.Mamba1`（RTX 5060 不支持其 kernel）→ 纯 PyTorch 慢速兜底。

### 3.2 GSC 模块（本仓库实现版）

每个 Stage 的 Mamba 扫描前先过 GSC 增强空间特征（两路卷积相加 + 投影 + 残差）：

```
x → 3×3×3 Conv + InstanceNorm + ReLU → 3×3×3 Conv + InstanceNorm + ReLU ──┐
x → 1×1×1 Conv + InstanceNorm + ReLU ────────────────────────────────────┤ 相加
                                     → 1×1×1 Conv + InstanceNorm + ReLU → + x（残差）
```

### 3.3 与论文官方实现的差异

| 模块 | 论文官方 | 本仓库实现 |
|------|---------|-----------|
| Mamba 扫描 | ToM：三方向（前向/反向/层间）Mamba | 单方向 Mamba2 全序列扫描（`MambaLayer`；`num_slices` 形参保留但未使用） |
| GSC | 门控空间卷积（乘性融合） | 两路卷积相加版 GSC（见 3.2） |
| FUE | 跳跃连接中的特征不确定性过滤 | **未实现**（跳跃特征直接进 UNETR UpBlock） |
| 解码器 | CNN 解码器 | MONAI UNETR 风格 ResBlock（`UnetrUpBlock`） |
| 分辨率 | nnUNet 自适应重采样 | 固定 2.0mm 缓存 |

## 4. 环境说明

- **Conda 环境**：`conda activate mamba_sm120_v3`（Mamba 项目专用环境，已装 `mamba_ssm` + `causal_conv1d`；SRTP 环境没有 Mamba 依赖，跑不了本项目）。
- **显卡口径**：RTX 5060 Laptop 8GB。本项目约 66.90M 参数，2.0mm 分辨率下 `--batch 1`~`2` 训练显存压力明显大于 CNN 基线；OOM 时先降 batch、必要时减小 patch。

## 5. 数据缓存约定（2026-10-07 全库统一定案）

- **训练/评估都必须带 `--cache`**，否则直接报错退出（代码强制要求已生成缓存）。
- 缓存根目录（代码默认）：`D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache`，按分辨率拼接后实际目录为 `persistent_cache_2.0`（2.0mm）。
- 整机迁移可用环境变量 `SRTP_CACHE_PARENT` 重定向父目录（代码内默认路径仅作兜底）。
- 仓库存在**两份划分不同的 2.0mm 缓存**（主项目体系 vs Essay\Data LightSegMamba 体系，test 集交集仅 21/188），本项目走**主项目体系**，评估数字不可与 LightSegMamba 体系直接对比。
- 首次生成缓存（预处理入口）：

```bash
conda activate mamba_sm120_v3
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code
python shared/data/prepare_data.py --resolution 2.0 --generate --cache --workers 8
```

## 6. 命令行

> ✅ **官方口径已恢复（2026-10-07）**：`pipeline/train.py` 硬编码的 `loss_name="CrossEntropyLoss"`、`optimizer_type="SGD"`、`scheduler_type="Poly"`（论文官方三件套）曾因 shared 组件重构不在注册表内，训练启动即抛 `ValueError`；现已把官方配置注册回 shared——`shared/losses/cross_entropy_loss.py`（one-hot 标签自动转索引的 `CrossEntropyLoss`）、`shared/optim/poly_scheduler.py`（`PolyLR`，lr = base_lr × (1 − step/T_max)^0.9）、`shared/optim` 的 `SGD`，三件套已通过前向反向与调度公式逐位验证。训练命令可直接执行；评估脚本（DiceFocalLoss + CosineAnnealingLR）本就不受影响。

### 6.1 训练

```powershell
conda activate mamba_sm120_v3
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code

# 推荐口径：显式传 --epochs 150；PowerShell 单行命令（可直接粘贴运行）
python SegMamba-Official/pipeline/train.py --resolution 2.0 --cache --epochs 150 --val_interval 5 --batch 2 --device cuda
```

不传 `--epochs` 时 argparse 默认 1000（论文官方口径）、`--val_interval` 默认 2，本机 8GB 显存与时间预算下建议显式传 150 / 5。

### 6.2 恢复训练（断点续训）

训练**每个 epoch 结束后**都会把完整训练状态（模型权重、优化器、调度器、GradScaler、SWA 平均权重、当前 epoch、最佳指标）写入权重目录的 `latest_checkpoint.pth`：

- **中断后恢复**：用与原训练**完全相同的命令**重新运行即可（默认权重目录由 `--resolution` 与 `--device` 推导为 `pipeline/models/2.0mm_cuda/`；若原训练传了 `--model_dir`，续训也必须传同一个值）。启动时检测到该文件会自动恢复并从断点 epoch 继续训练。
- **验证中断保护**：验证阶段失败（如 OOM）时异常被捕获、该次验证跳过，训练状态已在验证前保存到断点，恢复训练从下一 epoch 继续。
- **放弃断点重头训**：先删除或移走 `pipeline/models/2.0mm_cuda/latest_checkpoint.pth`。

```powershell
# 例：中断后原命令重跑即自动续训（与 6.1 完全一致）
python SegMamba-Official/pipeline/train.py --resolution 2.0 --cache --epochs 150 --val_interval 5 --batch 2 --device cuda
```

### 6.3 评估

评估读取缓存目录 `test/` 子目录与 `split_info.json` 的测试集划分；**必须显式传 `--checkpoint`**（argparse 默认 None，不加载权重的评估无意义）：

```powershell
python SegMamba-Official/pipeline/evaluate.py --resolution 2.0 --cache --device cuda --checkpoint SegMamba-Official/pipeline/models/2.0mm_cuda/best_metric_model.pth
```

checkpoint 文件名含 `swa`（或旧命名 `ema`）时，结果 CSV 自动加 `_swa` / `_ema` 后缀区分口径。

### 6.4 快速冒烟测试（2 epoch）

```powershell
python SegMamba-Official/pipeline/train.py --resolution 2.0 --cache --epochs 2 --val_interval 1 --batch 1 --workers 0 --device cuda --model_dir SegMamba-Official/pipeline/models/smoke_test
```

用 `--model_dir` 把冒烟产物隔离到 `smoke_test` 目录，避免污染正式训练断点。

## 7. CLI 参数完整列表

### 7.1 pipeline/train.py

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `segmamba` | 模型名称 |
| `--resolution` | float | `2.0` | 数据分辨率（mm），决定缓存目录与权重目录后缀 |
| `--workers` | str | `auto` | 数据加载线程数（`auto`/数字） |
| `--batch` | str | `2` | 批次大小（`auto`/数字，`auto` 表示自动计算） |
| `--cache` | flag | False | 启用硬盘缓存（**必须**，否则报错退出） |
| `--swa` | flag | False | 启用 SWA 等权权重平均（历史命名 EMA，实际语义一直为等权 SWA） |
| `--epochs` | int | `1000` | 训练轮数（论文官方 1000；本机推荐显式传 150） |
| `--val_interval` | int | `2` | 验证间隔（论文官方 2；本机推荐显式传 5） |
| `--device` | str | None | 设备（cuda/cpu，None=自动选择） |
| `--model_dir` | str | `""` | 自定义权重保存目录（空=默认 `pipeline/models/{分辨率}mm_{设备}`） |
| `--log_name` | str | `""` | 自定义日志名称前缀（空=默认 `train_official_{分辨率}mm`） |

注：`train()` 函数签名中保留 `use_deep_supervision` 形参，但 **argparse 未暴露对应开关**，命令行无法启用深层监督（恒为 False）。

### 7.2 pipeline/evaluate.py

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `segmamba` | 模型名称 |
| `--resolution` | float | `1.0` | 数据分辨率（mm），评估请显式传 `--resolution 2.0` |
| `--checkpoint` | str | None | 权重路径（**必须显式传**；文件名含 `swa`/`ema` 决定 CSV 后缀） |
| `--init_filters` | int | `12` | 初始滤波器数量（模型固定 feat_size=[48,96,192,384]，**该参数实际被忽略**，保留仅为兼容） |
| `--workers` | str | `auto` | 数据加载线程数（`auto`/数字） |
| `--cache` | flag | False | 启用硬盘缓存（**必须**） |
| `--device` | str | None | 设备（cuda/cpu，None=自动选择） |
| `--no_lesion_wise` | flag | False | 禁用 lesion-wise 指标 |
| `--runs` | int | `1` | 评估次数（>1 时额外报告均值±标准差，但逐病例 CSV 仅 runs=1 时写出） |

## 8. 目录结构与输出产物

```
SegMamba-Official/
├── README.md
├── models/
│   └── segmamba_official.py   # SegMamba：MambaEncoder（GSC + Mamba×2×4 stage）+ UNETR 解码器
├── frame/
│   └── train.py               # OptimizedTrainer：损失/优化器工厂 + SWA + AMP + 滑窗验证
└── pipeline/
    ├── train.py               # 训练入口（官方 SGD 配置、断点续训）
    ├── evaluate.py            # 评估入口（Dice/HD95 + lesion-wise）
    ├── logs/                  # 训练/评估日志：{前缀}_{时间戳}.log，前缀默认 train_official_{分辨率}mm / eval_segmamba_{分辨率}mm
    ├── models/                # 权重目录：{分辨率}mm_{设备}（如 2.0mm_cuda）
    │   ├── latest_checkpoint.pth      # 每 epoch 更新的断点（含优化器/调度器/SWA/最佳指标）
    │   ├── best_metric_model.pth      # 最佳 WT Dice 的原始权重
    │   ├── best_metric_swa_model.pth  # 最佳 WT Dice 的 SWA 平均权重（--swa 时；旧名 best_metric_ema_model.pth 兼容识别）
    │   └── epoch_N.pth                # 各验证点权重快照
    └── evaluation_results/
        ├── metrics_segmamba_2.0mm.csv          # 原始权重逐病例：CaseID,Dice_WT/TC/ET,HD95_WT/TC/ET
        ├── metrics_segmamba_2.0mm_lesion.csv   # lesion-wise 逐病例
        └── metrics_segmamba_2.0mm_swa.csv      # checkpoint 名含 swa → 平均权重口径
```

## 9. 注意事项

1. **官方口径注册（2026-10-07 已修复）**：`CrossEntropyLoss` / `SGD` / `Poly` 三件套已注册回 shared 组件（详见 §6.1 开头说明），训练可直接启动，评估不受影响。历史日志 max_epochs=1000 的记录对应早期版本的 shared 组件，如需对齐历史口径请按 §7.1 参数表显式传参。
2. **评估 CSV 覆盖**：结果文件名固定（`metrics_segmamba_{分辨率}mm[_swa|_ema].csv`），重复评估直接覆盖旧 CSV；需要留档请先改名或备份。
3. **--checkpoint 必传**：评估默认 `None`，不传时模型是随机初始化权重，跑完的指标无意义。
4. **显存**：66.90M 参数 + 2.0mm 体数据，`--batch 2` 起步，OOM 降 `--batch 1`；同卡串行训练，勿并行多任务。
5. **SWA 口径**：`--swa` 启用 `shared/optim/swa.py` 的等权平均（非指数加权，历史命名 EMA 已于 2026-10-05 全面改名为 SWA）；旧权重名 `best_metric_ema_model.pth` 与旧 checkpoint 键在恢复/评估端均兼容。
6. **缓存一致性**：训练与评估必须用同一份 `persistent_cache_2.0`（主项目体系）；与 LightSegMamba 体系缓存的 test 划分不同，数字不可混比。
7. **论文数字口径**：论文报告的 BraTS2023 结果（Dice WT/TC/ET ≈ 0.92x/0.88x/0.84x）基于官方完整实现（ToM + GSC + FUE + nnUNet 预处理）；本仓库为简化复现（见 §3.3 差异表），结果预期与论文不严格可比。
