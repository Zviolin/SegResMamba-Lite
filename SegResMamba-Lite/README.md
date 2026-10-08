# SegResMamba-Lite：轻量化混合 Mamba 脑肿瘤分割（主项目）

> **一句话定位**：受 SegMamba (MICCAI 2024, arXiv:2401.13560) 启发的轻量化 3D 分割网络——CNN 骨干 + 关键层双向 Mamba，用约 1/50 的参数量（1.4M 级 vs 官方 74.87M）达到与其相当的性能；后续演进到 MoA / MoE 专家混合架构（V1–V12，共三代 + P2 探索）。
>
> **数据集**：BraTS 2023 GLI（脑胶质母细胞瘤）训练集 1251 例，2.0mm 重采样，188 例测试集评估。

## 一、项目定位与同级项目关系

本项目是 SRTP 论文的**主项目**（自研模型），围绕一个 1.4M 参数约束下的轻量混合架构，迭代了三代设计（纯 Mamba 混合 → MoA 软路由 → 稀疏 MoE/MoA）。同级项目：

| 项目 | 角色 | 与本项目关系 |
|------|------|--------------|
| **SegResMamba-Lite（本项目）** | 自研主模型（Mamba 混合） | — |
| SegResNet-DiceCE-AdamW | CNN 基线（纯卷积，无 Mamba） | 性能对照基准；本项目 LightResBlock 灵感来源 |
| SegResNet-DiceFocal-AdamW-SWA-DS | CNN 基线（DiceFocal 变体） | 对照基准 |
| VMUNet / VSSUNet-DiceCE-AdamW | 轻量 Mamba 对照模型 | 同参数量级对照 |
| SegMamba-Official | 官方 SegMamba 复现（本地实测 66.90M；论文报告 74.87M） | 本项目的轻量化对标对象 |
| LightSegMamba-DiceCE-AdamW-V3 | 论文复现（轻量 Mamba） | 同代轻量方法对照 |
| shared/ | 公共库 | 数据加载、损失、Mamba 后端、评估指标全部复用 |

## 二、新手背景：任务与指标（通俗版）

- **任务**：给一张 3D 脑部 MRI，自动把肿瘤像素出来。每个病例有 **4 种模态**（同一部位的 4 种不同扫描参数）：t1n（平扫 T1）、t1c（增强 T1）、t2w（T2 加权）、t2f（FLAIR），拼成 4 通道输入。
- **分割区域**（BraTS 官方三区域，由 3 个类别标签组合而成）：
  - **WT**（Whole Tumor，整体肿瘤）：全部肿瘤区域
  - **TC**（Tumor Core，肿瘤核心）：核心坏死 + 强化部分
  - **ET**（Enhancing Tumor，强化肿瘤）：增强扫描下显影的活性部分
- **指标**：
  - **Dice**：预测与真值的重叠率（0~1，越高越好，可理解为"分割准不准"）
  - **HD95**：95 分位 Hausdorff 距离，衡量"预测边界偏了多少"（越低越好）。注意本项目**全局 HD95 是 2D voxel 历史口径**（历史数字均为 voxel 单位，详见注意事项 §10.3）；lesion-wise HD95 为 3D mm（BraTS 官方口径）。
- **数据划分**：训练 875 / 验证 249（训练中验证）/ 测试 188（评估用），由缓存 `split_info.json` 固定 seed=42，所有项目一致。

### 零基础速览：训练到底在做什么

整个训练就是「看图 → 画圈 → 对答案 → 调整」的循环，重复若干轮（每轮记 1 个 epoch）：

```
原始数据 → 预处理 → 模型预测 → 计算Loss → 反向传播 → 参数更新 ─┐
   ↑                                                      │
   └──── 验证循环 ←── 滑窗推理 ←─────────────────────────────┘
        （每 val_interval 轮一次，保存最佳权重）
```

- **预处理**：把每位病人的 4 张 MRI 统一方向（RAS）、重采样到 2.0mm、裁掉背景、归一化——相当于把所有考卷印成同一格式，模型才好学。
- **模型预测**：编码器把 64³ 立体图逐级压缩（空间变小、通道变多），像把一幅彩画浓缩成一本摘要书——知道「大概哪里有肿瘤」但丢了精确边界；解码器再逐级放大回原尺寸，并经跳跃连接把浅层「精细笔记」传回来补细节，最后输出每个体素属于 4 个类别（背景/NCR/ED/ET）的概率。本项目在压缩到最深一层时，用 Mamba 序列扫描（V 系演进见下文）替换了普通卷积 bottleneck。
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

## 三、模型结构与创新点

### 3.1 总体架构

```
输入 (B, 4, 64, 64, 64)                 （通道数按实验 init_filters=20 计）
   │
Stem: LightResBlock ─────────────────────────────── skip0 (20 @ 64³)
   │
down1: Conv3d(s=2) → 40 @ 32³
       LightResBlock + BiMamba @32³ ─────────────── skip1
   │
down2: Conv3d(s=2) → 80 @ 16³
       LightResBlock + BiMamba @16³ ─────────────── skip2
   │
down3: Conv3d(s=2) → 80 @ 8³
   │
Bottleneck @8³（版本差异所在）：
   V1–V4:  BiMamba ×1~×3（逐代加深）
   V6–V8:  Soft MoA（4 Mamba / +4 DConv 专家，样本级软路由）
   V9:     Top-2 稀疏 DConv MoE + 负载均衡
   V10:    论文级 MoA（共享 K/V + Attention 专家 + token 级 Top-K）
   V11:    边界感知混合 MoA（2 Attn + 2 DConv + 边界注意力 + 路由噪声）
   │
Decoder: ConvTranspose3d 上采样 ×3（16³→32³→64³）+ skip 拼接
   │
Output: 1×1×1 Conv3d → 4 通道 logits（BG/WT 组成类/TC 组成类/ET）
```

> 各版本 Mamba 放置位置、专家配置有差异，见 §3.3 版本对照表。

### 3.2 与官方 SegMamba 的核心差异

| 对比项 | 官方 SegMamba | SegResMamba-Lite |
|--------|--------------|------------------|
| 设计理念 | 全栈 Mamba，所有层用 Mamba | 混合架构：LightResBlock 骨干 + 关键层 Mamba |
| 参数量 | 74.87M | 1.42M – 1.56M（**约 1/50**） |
| 扫描方向 | ToM 三方向，3 个独立模块 | BiMamba 双向：1 个模块 + `flip`，计算量 2× 而非 3× |
| 特征维度 | [48, 96, 192, 384] + hidden 768 | 实验 init=20/24 → [20/24, 40/48, 80/96] |
| Mamba 参数 | d_state=16, d_conv=4, expand=2 | d_state=8, d_conv=2, expand=2 |
| 优化器 | SGD + 1000 epochs | AdamW (lr=1e-4, wd=1e-5) + CosineAnnealingLR + 100 epochs |
| 损失函数 | CrossEntropyLoss | DiceFocalLoss |

未采用官方 GSC / ToM / FUE 模块（ToM 简化为 BiMamba，GSC 用 LightResBlock 替代）。底层 Mamba 实现共用 `shared/models/mamba.py`，按优先级自动选后端：`mamba_ssm.Mamba2 (CUDA) > MiniMamba (纯 PyTorch) > mamba_ssm.Mamba1 > PurePyTorchMamba`。

### 3.3 版本演进对照表（V1–V12）

`models/__init__.py` 注册 v1–v4、v6–v12 共 **11 个可用版本**；v5 为未写完的废弃草稿（未注册）。参数量均为训练日志值（Mamba2SSM 后端）：

| 版本 | 代际 | Bottleneck 核心 | 实验 init | 参数量 | 权重 | 日志 |
|------|------|----------------|:---:|---:|:---:|:---:|
| V1 | 一 | BiMamba ×1（2 尺度） | **24** | 1.52M | ✅ | ✅ |
| V2 | 一 | BiMamba ×1（3 尺度编码器） | **20** | 1.42M | ✅ | ✅ |
| V3 | 一 | 双重 BiMamba + 全量 SE + 解码器 Mamba | 20 | 1.51M | ✅ | ✅ |
| V4 | 一 | 三重 BiMamba + 双解码器 Mamba + 策略 SE | 20 | 1.56M | ✅ | ✅ |
| V5 | — | 未完成草稿（未注册，勿用） | — | — | ❌ | ❌ |
| V6 | 二 | Soft MoA：4 Mamba 专家样本级软路由（无 α） | **20** | 1.46M | ✅ | ✅ |
| V7 | 二 | +4 DConv 专家，Sigmoid α 自动学习 | 20 | 1.50M | ✅ | ✅ |
| V8 | 二 | 结构同 V7，α 改温度缩放 sigmoid(α_raw/τ) | 20 | 1.50M | ❌ 未留存 | ✅ |
| V9 | 三 | Top-2 稀疏 DConv MoE + 负载均衡 | **20** | 1.42M | ✅ | ✅ |
| V10 | 三 | 论文级 MoA：共享 K/V + Attention 专家 + token 级 Top-2 | **20** | 1.42M（二版） | ✅ | ✅ |
| V11 | 三 | 边界感知混合 MoA（2 Attn + 2 DConv + 固定边界核门控） | 20 | 1.44M | ✅ | ✅ |
| V12 | P2 | V10 + 边界门控（路由前增强） | 20 | 1.43M | ✅ | ✅ |

> V12 已于 2026-09 完成训练（训练日志 total 1,428,194 / trainable 1,426,034，val best Dice 0.9016 @ epoch 90）；测试集 188 例评估已完成：全局 Dice 均值 0.8912、LW Dice 均值 0.8086（LW HD95 6.80）——未达预设升级标准（LW > 0.8304），为探索性负结果，V10 保持最终模型（详见 [实验记录总集.md](./实验记录总集.md) Part B §2.3 / Part D）。

> **锚点口径**：V10 为论文锚点版本（第二版 Attention 专家，1,422,734 参数），锚点训练口径为 **batch=6 / workers=8 / epochs=100 / init_filters=20 / 2.0mm / seed=42**。所有实验均通过 CLI 显式传 `--init_filters`（V1=24、V2–V12=20），代码签名默认值（26/22/20，且训练与评估的 v1 默认不同：训练 26 / 评估 24）不对应任何实验结果。
>
> 逐版本详细分析见 [实验记录总集.md](./实验记录总集.md)（Part C）。

### 3.4 创新点逐条

1. **轻量混合架构**：LightResBlock（瓶颈通道 min(in,out) 的双 3³ 卷积残差块）做局部建模，仅在 32³/16³/8³ 关键尺度放双向 Mamba 做长程建模，1.4M 参数达到官方 1/50 参数量的同等全局 Dice。
2. **BiMamba**：单个共享 Mamba 模块 + 三维翻转实现正/反两向扫描，替代官方 ToM 三模块（省 1/3 扫描开销）。
3. **专家混合三代演进**（V6–V11）：软路由（V6）→ 可学习融合系数 α（V7）→ 温度缩放（V8）→ Top-K 稀疏 MoE + 负载均衡（V9，Switch Transformer 风格）→ 论文级 MoA（V10，严格按 SHMoAReg arXiv:2509.20073：共享 K/V + 独有 query/输出投影 + token 级稀疏路由）→ 边界感知混合专家（V11，Sobel 风格固定 26 邻域中心差分核 + 可学习门控 + Noisy Top-K Gating）。
4. **V10 内置消融开关**：expert_type（attention/mamba/dconv）、route_granularity（token/sample）、share_kv、route_noise、decoder_moa，全部默认值 = 锚点行为，支撑论文消融实验。

## 四、环境说明

| 项 | 说明 |
|----|------|
| **conda 环境** | `mamba_sm120_v3`（当前统一使用；历史 V10/V11 锚点日志产生于 `mamba_sm120`，两环境均含 mamba_ssm + causal_conv1d） |
| **禁止环境** | SRTP（PyTorch nightly 在 RTX 5060 上 GPU 训练 backward 极慢，勿用） |
| **硬件口径** | RTX 5060 Laptop 8GB；锚点配置 batch=6 + workers=8 显存余量充足 |
| **关键依赖** | mamba_ssm（CUDA 后端，延迟数字才有效）、MONAI、torch 2.8.0+cu129 系 |

```bash
conda activate mamba_sm120_v3
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code   # 后续命令均在此目录执行
```

## 五、数据缓存约定（2026-10-07 全库统一定案）

- **默认缓存父目录**：`D:\Codes\SRTP\BraTS\DATA\BraTS-GLI\TrainingDATA\persistent_cache`
- **实际目录**：按分辨率拼接，2.0mm → `persistent_cache_2.0`（即 `...\TrainingDATA\persistent_cache_2.0`）
- **整机重定向**：环境变量 `SRTP_CACHE_PARENT` 可覆盖默认父目录（换机器时统一迁移用）
- **必须显式 `--cache`**：训练/评估均要求显式启用缓存并确保已生成（未生成会报错并提示先跑 prepare_data）
- **预处理命令**（生成缓存，shared/data/prepare_data.py）：

```bash
python shared/data/prepare_data.py --resolution 2.0 --generate --cache --workers 8
```

- **划分固定**：train 875 / val 249 / test 188，seed=42，写入缓存目录 `split_info.json`；训练随机种子 `--seed`（默认 42）不影响数据划分
- 注意：本机存在两份划分实现不同的 persistent_cache_2.0（D 盘主项目体系与 Essay\Data LightSegMamba 体系，test 集交集仅 21/188），本项目使用 **D 盘主项目体系**，跨体系数字不可直接对比

## 六、命令行

### 6.1 训练（V10 锚点口径示例）

```bash
conda activate mamba_sm120_v3
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code

# V10 锚点：batch=6 / workers=8 / epochs=100 / init_filters=20 / 2.0mm
python SegResMamba-Lite/pipeline/train.py --version v10 --init_filters 20 --batch 6 --resolution 2.0 --epochs 100 --val_interval 5 --workers 8 --device cuda --swa --cache
```

其他版本把 `--version` 换成 v1/v2/v3/v4/v6/v7/v8/v9/v11/v12 即可（V1 记得 `--init_filters 24`，其余 20）。V7/V8 复现 EMA 对照消融时加 `--weight_mode ema`。

### 6.2 恢复训练（断点续训）

**机制（与代码一致）**：每个 epoch 结束都会把完整状态（模型权重 + 优化器 + 调度器 + 当前 epoch + 历史最佳 Dice）存到权重目录的 `latest_checkpoint.pth`。训练中断后，**原样重跑同一条训练命令**即可：

```bash
# 中断前用什么命令，中断后就原样重跑什么命令
python SegResMamba-Lite/pipeline/train.py --version v10 --init_filters 20 --batch 6 --resolution 2.0 --epochs 100 --val_interval 5 --workers 8 --device cuda --swa --cache
```

启动时会打印 `发现完整检查点：...latest_checkpoint.pth，正在恢复训练...`，自动从 `epoch+1` 继续，并还原历史最佳指标（防止续训后首次验证以次优权重覆盖 `best_metric_model.pth`）。

注意事项：

- 恢复位置由权重目录决定：`--model_dir` > 默认目录（`--run_tag` 会给目录名追加 `_tag` 后缀）。**消融变体（A2/B1/S1 等）必须带相同的 `--run_tag`**，否则找不到自己的断点（默认目录是锚点的）。
- `--torch.compile` 不影响恢复：保存/恢复均取 `_orig_mod` 的原始 state_dict。
- 从零重训而非续训时，先删除权重目录中的 `latest_checkpoint.pth`（或改用新的 `--run_tag`）。

### 6.3 评估

```powershell
# 模板：消融开关参数（--top_k/--num_experts/--expert_type/--route_granularity/--share_kv/--decoder_moa）
# 必须与训练命令逐一对齐；--run_tag 用于自动定位权重目录；route_noise 仅影响训练，评估无需传
# PowerShell 单行命令（可直接粘贴运行）
python SegResMamba-Lite/pipeline/evaluate.py --version v10 --init_filters 20 --resolution 2.0 --device cuda --cache --workers 8 --checkpoint "SegResMamba-Lite/pipeline/models/2.0mm_v10_cuda/best_metric_model.pth"

# 简写：不传 --checkpoint 时自动取 pipeline/models/2.0mm_{version}_cuda[_{run_tag}]/best_metric_model.pth
python SegResMamba-Lite/pipeline/evaluate.py --version v10 --init_filters 20 --resolution 2.0 --device cuda --cache --workers 8 --run_tag A3
```

输出（`pipeline/evaluation_results/`）：

- `metrics_segresmamba_lite_2.0mm.csv`——逐病例全局 Dice/HD95（WT/TC/ET 六列）；权重文件名含 `swa`/`ema` 时自动追加 `_swa`/`_ema` 后缀
- `metrics_segresmamba_lite_2.0mm_lesion.csv`——逐病例 lesion-wise 指标（BraTS 官方口径，3D mm）
- `--runs > 1` 时输出多次评估的均值 ± 标准差（lesion CSV 逐轮加 `_run{n}` 后缀）

### 6.4 快速 2-epoch 冒烟测试

```bash
python SegResMamba-Lite/pipeline/train.py --version v10 --init_filters 20 --epochs 2 --workers 2 --resolution 2.0 --device cuda --cache --val_interval 1 --batch 1
```

### 6.5 消融实验与工具脚本

V10/V12 消融命令全集（A2/A3/AB4/B1/B2/C1/C2/D1/E1–E3/F1/G1/H1/S1/S2）见 [实验记录总集.md](./实验记录总集.md)（Part F·§九）；要点：每条命令带 `--run_tag {编号}` 隔离权重与日志（`pipeline/models/2.0mm_v10_cuda_{TAG}`），batch 4 组（AB4/B2/D1/G1）与其余 batch 6 组分属不同同口径组。

```powershell
# 参数量自检（输出取决于当前 Mamba 后端；实验值以训练日志为准）
python SegResMamba-Lite/check_params.py

# 效率基准（参数量/FLOPs/延迟：自研 V2–V12 + 3 基线，64³ batch=1）
python SegResMamba-Lite/bench_efficiency.py            # 正式口径：预热 10 + 计时 100
python SegResMamba-Lite/bench_efficiency.py --quick    # 冒烟：预热 2 + 计时 10

# 逐例配对统计检验（Wilcoxon + Holm；CSV 为逐例评估产物，注意会被重评覆盖）
python SegResMamba-Lite/scripts/stats_tests.py --csv_a <本项目逐例CSV> --csv_b <对照项目逐例CSV> --lesion_csv_a <...> --lesion_csv_b <...> --label_a Lite --label_b <对照> --out stats_xxx.md

# MoA 路由可视化（v10/v12）
python SegResMamba-Lite/scripts/visualize_routing.py --version v10 --init_filters 20 --num_experts 4 --top_k 2 --checkpoint SegResMamba-Lite/pipeline/models/2.0mm_v10_cuda/best_metric_model.pth --input <病例.npy> --output_dir routing_vis
```

## 七、完整 CLI 参数表

### 7.1 train.py（python SegResMamba-Lite/pipeline/train.py）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `segresmamba_lite` | 模型名称 |
| `--version` | str | `v2` | 模型版本（v1/v2/v3/v4/v6/v7/v8/v9/v10/v11/v12） |
| `--resolution` | float | `2.0` | 数据分辨率（mm） |
| `--init_filters` | int | `None` | 初始滤波器数；None 时按版本回退（v1=26, v2=22, v3/v4/v7=18, v6/v8/其余=20）。**实验一律显式指定（V1=24，V2–V12=20）** |
| `--use_attention` / `--no_attention` | flag | True | 启用/禁用注意力 |
| `--d_state` | int | `8` | Mamba 状态维度 |
| `--expand` | int | `2` | Mamba 扩展因子 |
| `--num_experts` | int | `4` | MoA 专家数量（V6–V12） |
| `--top_k` | int | `2` | Top-K 激活专家数（仅 V10/V12） |
| `--expert_type` | str | `attention` | V10/V12 专家类型（attention/mamba/dconv） |
| `--route_granularity` | str | `token` | V10/V12 路由粒度（token/sample） |
| `--share_kv` | int | `1` | V10/V12 共享 K/V（1=共享，0=每专家独立） |
| `--route_noise` | float | `0.0` | V10/V12 训练期路由噪声 σ（F1 用 0.05） |
| `--decoder_moa` | int | `0` | V10/V12：16³ 解码器侧追加 MoA（1=启用，G1） |
| `--seed` | int | `42` | 训练随机种子（不影响数据划分） |
| `--use_deep_supervision` / `--no_deep_supervision` | flag | False | 深层监督开关 |
| `--epochs` | int | `100` | 训练轮数 |
| `--val_interval` | int | `5` | 每 N epoch 验证一次 |
| `--batch_size` / `--batch` | int | `None` | 批次大小（None=由 dataloader 自动；`--batch` 为简写）。锚点显式传 6 |
| `--workers` | int | `None` | DataLoader worker 数（None=自动）。锚点显式传 8 |
| `--cache` | flag | False | 启用硬盘缓存（必须） |
| `--swa` / `--use_swa` / `--no_swa` | flag | True | 权重平均（等权 SWA）默认开启；`--no_swa` 关闭 |
| `--weight_mode` | str | `swa` | 权重平均模式：swa=等权（历史口径）/ ema=真指数移动平均（V7/V8 对照复现，decay=0.999） |
| `--compile` | flag | False | torch.compile 编译加速 |
| `--device` | str | `None` | 设备（cuda/cpu，None=自动） |
| `--loss_name` | str | `DiceFocalLoss` | 损失（DiceCELoss/DiceFocalLoss/DiceLoss/V6LossSimple） |
| `--v6_alpha` | float | `0.5` | V6Loss：Dice 权重 |
| `--v6_beta` | float | `0.3` | V6Loss：Focal 权重 |
| `--v6_gamma` | float | `0.2` | V6Loss：Boundary 权重 |
| `--lb_weight` | float | `0.01` | V9 MoE 负载均衡损失权重（0=关闭） |
| `--run_tag` | str | `""` | 消融标签：权重/日志目录追加 `_tag` 后缀隔离变体 |
| `--model_dir` | str | `""` | 自定义权重目录（优先于 --run_tag；默认 `pipeline/models/{分辨率}mm_{版本}_{设备}[_{run_tag}]`） |
| `--log_name` | str | `""` | 自定义日志名前缀（默认 `train_{版本}_{分辨率}mm[_{run_tag}]`） |
| `--deterministic` | flag | False | 确定性训练：固定种子 + cudnn.deterministic + 确定性算法 + CUBLAS_WORKSPACE_CONFIG（不触碰 TF32，保持论文口径可比） |
| `--cache_parent` | str | `""` | 缓存父目录覆盖：命令行 > 环境变量 SRTP_CACHE_PARENT > 历史默认；最终目录 = `{parent}_{分辨率}` |
| `--skip_nonfinite` | flag | False | 跳过非有限 loss（NaN/Inf）的 batch：不 backward/不更新/不进平均权重，epoch 指标按有效 step 平均 |
| `--best_weights` | str | `raw` | best_metric_model.pth 保存来源：raw=原始权重（历史口径）/ avg=验证所用的平均权重（SWA/EMA，跟随 `--weight_mode`；权重平均关闭（`--no_swa`）时回退 raw） |

> 训练超参（固定，无 CLI）：AdamW lr=1e-4、weight_decay=1e-5、CosineAnnealingLR、alpha_raw/temperature 类参数 10× 学习率。`--d_conv`、`--v6_focal_alpha/--v6_focal_gamma` 未暴露到 CLI。
>
> 以上 `--deterministic` / `--cache_parent` / `--skip_nonfinite` / `--best_weights`（及 evaluate 的 `--cache_parent`/`--data_root`）为 2026-10-08 跨卡实验包合并引入的可选开关，默认关闭时不带任何新参数的命令行为与历史逐位一致；换机迁移用命令行参数或环境变量重定向，不改变各项目历史数据划分绑定。

### 7.2 evaluate.py（python SegResMamba-Lite/pipeline/evaluate.py）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `segresmamba_lite` | 模型名称 |
| `--version` | str | `v2` | 模型版本（v1–v12，同训练） |
| `--resolution` | float | `2.0` | 数据分辨率（mm） |
| `--checkpoint` | str | `None` | 权重路径；None 时自动取 `--model_dir` 优先，否则 `pipeline/models/2.0mm_{version}_cuda[_{run_tag}]/best_metric_model.pth` |
| `--init_filters` | int | `None` | None 时按版本回退（v1=**24**，v2=22, v3/v4/v7=18, v6/v8/其余=20；注意与训练的 v1 回退值 26 不同） |
| `--d_state` | int | `8` | Mamba 状态维度 |
| `--expand` | int | `2` | Mamba 扩展因子 |
| `--no_attention` | flag | False | 禁用注意力（默认启用） |
| `--num_experts` | int | `4` | MoA 专家数量 |
| `--top_k` | int | `2` | Top-K 专家数（V10/V12，须与训练一致） |
| `--expert_type` | str | `attention` | V10/V12 专家类型（与训练一致） |
| `--route_granularity` | str | `token` | V10/V12 路由粒度（与训练一致） |
| `--share_kv` | int | `1` | V10/V12 共享 K/V（与训练一致） |
| `--decoder_moa` | int | `0` | V10/V12 解码器侧 MoA（与训练一致） |
| `--deep_supervision` / `--no_deep_supervision` | flag | False | 深层监督（默认关闭，与训练默认一致） |
| `--workers` | str | `auto` | `auto`=自动，或数字（注意本文件里是字符串类型） |
| `--run_tag` | str | `""` | 消融标签：默认 checkpoint 路径追加 `_tag`（与训练对应） |
| `--model_dir` | str | `""` | 自定义权重目录（优先于 --run_tag，默认取其下 `best_metric_model.pth`） |
| `--cache` | flag | False | 启用硬盘缓存（必须） |
| `--device` | str | `None` | 设备（cuda/cpu） |
| `--no_lesion_wise` | flag | False | 禁用 lesion-wise 指标 |
| `--runs` | int | `1` | 评估次数（>1 时输出均值 ± 标准差） |
| `--visualize_attention` | flag | False | V6：打印 MoA 注意力权重统计 |
| `--cache_parent` | str | `""` | 缓存父目录覆盖：命令行 > 环境变量 SRTP_CACHE_PARENT > 历史默认；最终目录 = `{parent}_{分辨率}` |
| `--data_root` | str | `""` | 原始 BraTS TrainingData 目录覆盖：命令行 > 环境变量 BRATS_DATA_ROOT > 历史默认 |

## 八、目录结构与输出产物

```
SegResMamba-Lite/
├── README.md                      # 本文件
├── 实验记录总集.md                 # 研究文档合集（Part A–F：架构/指标/演进/消融/统计/设计存档）
├── check_params.py                # 参数量统计脚本
├── bench_efficiency.py            # 效率基准（参数量/FLOPs/延迟）
├── models/
│   ├── __init__.py                # 版本注册（v1–v4, v6–v12）
│   ├── v1.py ~ v4.py              # 第一代：纯 Mamba 混合
│   ├── v5.py                      # 未完成草稿（未注册，勿用）
│   ├── v6.py ~ v8.py              # 第二代：MoA
│   ├── v9.py ~ v11.py             # 第三代：稀疏 MoE / 论文级 MoA / 边界感知
│   └── v12.py                     # P2 探索：V10 + 边界门控（已训练，测试集评估待回填）
├── frame/train.py                 # OptimizedTrainer（SWA/EMA、深层监督、负载均衡接入）
├── scripts/
│   ├── stats_tests.py             # 统计检验（Wilcoxon + Holm）
│   └── visualize_routing.py       # MoA 路由可视化
└── pipeline/
    ├── train.py                   # 训练入口
    ├── evaluate.py                # 评估入口（全局 + lesion-wise）
    ├── models/                    # 权重（运行后生成）：{分辨率}mm_{版本}_{设备}[_{run_tag}]/
    │   └── 2.0mm_v10_cuda/        #   best_metric_model.pth / latest_checkpoint.pth / epoch_N.pth（每 10 epoch）
    ├── logs/                      # 日志：train_{版本}_{分辨率}mm[_{run_tag}]_*.log
    └── evaluation_results/        # 评估 CSV：metrics_segresmamba_lite_{分辨率}mm[_swa|_ema].csv + _lesion.csv
```

## 九、实验结果（BraTS-GLI，2.0mm，100 epochs，测试集 188 例，seed=42）

### 9.1 全局指标（V1–V11）

| 模型 | 参数量 | WT Dice | TC Dice | ET Dice | WT HD95 | TC HD95 | ET HD95 | Dice 均值 | HD95 均值 |
|------|-------:|--------:|--------:|--------:|--------:|--------:|--------:|:--------:|:--------:|
| SegResNet (DiceFocal 基线) | 4.70M | 0.9211 | 0.8842 | 0.8488 | 5.45 | 4.51 | 4.00 | 0.8847 | 4.65 |
| VSSUNet | 0.19M | 0.9106 | 0.8710 | 0.8344 | 6.30 | 6.89 | 4.69 | 0.8720 | 5.96 |
| VMUNet | 0.52M | 0.9109 | 0.8829 | 0.8376 | 5.49 | 4.80 | 4.09 | 0.8771 | 4.79 |
| **V1** | 1.52M | **0.9257** | 0.8825 | **0.8578** | 5.49 | 4.85 | 3.81 | 0.8887 | 4.72 |
| **V2** | 1.42M | 0.9242 | 0.8847 | 0.8565 | 5.37 | 4.83 | 3.70 | 0.8885 | 4.63 |
| **V3** | 1.51M | 0.9217 | 0.8817 | 0.8524 | **5.02** | **4.62** | **3.59** | 0.8853 | 4.41 |
| **V4** | 1.56M | 0.9229 | 0.8850 | 0.8560 | 5.85 | 5.18 | 4.08 | 0.8880 | 5.04 |
| **V6** | 1.46M | 0.9249 | 0.8884 | 0.8578 | 5.34 | 4.05 | 3.43 | 0.8904 | **4.28** |
| **V7** | 1.50M | 0.9233 | **0.8933** | 0.8572 | 5.22 | 4.43 | 3.96 | **0.8913** | 4.54 |
| **V8** | 1.50M | 0.9216 | 0.8820 | 0.8545 | 5.61 | 4.58 | 3.89 | 0.8860 | 4.69 |
| **V9** | 1.42M | 0.9245 | 0.8867 | 0.8566 | 5.87 | 4.35 | 3.97 | 0.8893 | 4.73 |
| **V10（Attention 版）** | 1.42M | 0.9227 | 0.8860 | 0.8547 | 5.65 | 4.35 | 3.89 | 0.8878 | 4.63 |
| **V11** | 1.44M | 0.9215 | 0.8851 | 0.8570 | 6.53 | 4.54 | 3.96 | 0.8879 | 5.01 |

> 全局 HD95 为 **2D voxel 历史口径**（非 mm），引用时勿写成 3D mm。表中基线行（SegResNet-DiceFocal 等）数字摘自旧版对比表，其评估口径未逐项复核，跨项目精确对比请以各自项目 README 注明的口径为准。V10 第一版（Mamba 专家，1.46M）：Dice 0.8894 / HD95 5.05，因病灶级能力弱被 Attention 版取代。

### 9.2 Lesion-wise 指标（BraTS 官方口径，3D mm）

| 模型 | WT Dice | TC Dice | ET Dice | Dice 均值 | HD95 均值 |
|------|--------:|--------:|--------:|:--------:|:--------:|
| VMUNet | 0.8509 | 0.8384 | 0.7691 | 0.8194 | 33.01 |
| VSSUNet | 0.6692 | 0.7543 | 0.6901 | 0.7045 | 5.66 |
| V1 | 0.8249 | 0.8155 | 0.7616 | 0.8007 | 8.05 |
| V2 | 0.8034 | 0.8332 | 0.7803 | 0.8056 | 7.46 |
| V3 | 0.7875 | 0.8323 | 0.7695 | 0.7964 | 7.20 |
| V4 | 0.8075 | 0.8189 | 0.7652 | 0.7972 | 7.39 |
| V6 | 0.7659 | 0.8455 | 0.7851 | 0.7988 | 7.28 |
| V7 | 0.7726 | 0.8375 | 0.7701 | 0.7934 | 7.76 |
| V8 | 0.7521 | 0.8119 | 0.7552 | 0.7731 | 6.09 |
| **V9** | 0.7839 | 0.8365 | 0.7774 | 0.7993 | 6.06 |
| **V10（Attention 版）** | **0.8410** | **0.8516** | **0.7987** | **0.8304** | 6.39 |
| **V11** | 0.8009 | 0.8468 | 0.7914 | 0.8130 | 6.80 |

### 9.3 核心结论

1. **参数量效率**：V2/V9/V10 仅 1.42M、V11 1.44M（均 <1.5M 合规），以官方 SegMamba 约 1/52、SegResNet 约 1/3.3 的参数进入性能第一梯队。
2. **全局 Dice 最优**：V7 0.8913（TC Dice 0.8933 为全系列单区最高）；全局 HD95（voxel 口径）最优：V6 4.28。
3. **MoA（V6–V8）整体优于纯 Mamba（V1–V4）**：平均 Dice 0.8892 vs 0.8876。
4. **第三代的收益集中在病灶级**：V10 Attention 版 LW Dice 0.8304 为自研系列最高（超过 VMUNet 的 0.8194，且 LW HD95 6.39 vs VMUNet 33.01）；V9 LW HD95 6.06 为自研系列最优（全表最小为 VSSUNet 5.66）。
5. **失败教训**：V4 三重 Bottleneck 过拟合（HD95 4.41→5.04）；策略性放置 SE 失效，SE 需一致使用；V10 第一版 Mamba 专家在 8³ 短序列上分工趋同，稀疏丢弃信息得不偿失。

## 十、注意事项

1. **环境**：必须用 `mamba_sm120_v3`（含 mamba_ssm CUDA 后端）；SRTP 环境 GPU 训练 backward 极慢，勿用。FLOPs 统计依赖 thop（未装则显示 N/A）。
2. **显存**：锚点配置 batch=6 在 RTX 5060 8GB 显存余量充足；G1（decoder_moa）等大配置需降到 batch=4（显存贴顶 paging）；首次迭代含 triton JIT 编译约 37s 属正常。
3. **HD95 口径**：全局 HD95 为 **2D voxel 历史口径**（`frame/train.py` 的 `_safe_hd95` 对 (1,D,H,W) 4 维输入不传 spacing，MONAI 按 2D 解释；空边界记 374 满额惩罚，inf/nan 兜底 374）——历史全部全局 HD95 数字均为该口径，引用时勿写成 3D mm；lesion-wise HD95 才是 3D mm（官方口径）。与其他项目（如 SegResNet 全局 HD95 为 3D mm 口径）对比时注意口径差异。
4. **串行训练纪律**：单卡 8GB，同一时间只跑一个训练任务；消融变体务必带 `--run_tag`，防止误恢复锚点权重。
5. **评估 CSV 覆盖**：`metrics_segresmamba_lite_2.0mm*.csv` 为固定命名，每次评估会**重写覆盖**（当前内容为最后评估的版本）；引用特定版本逐例数据前先重跑该版本评估或从 logs 确认评估时间戳。
6. **最佳模型选择标准**：训练验证以**三区域平均 Dice（dice_avg）**为最佳标准（注意与 SegResNet 基线的 WT Dice 标准不同）。
7. **init_filters 纪律**：实验必须显式 `--init_filters`（V1=24、V2–V12=20），签名默认值不对应任何实验结果；评估时 v1 回退默认 24 与训练回退默认 26 不同，评估论文 V1 权重靠默认值即可命中。
8. **V8 权重未留存**：无 `2.0mm_v8_cuda` 目录，α≈0.5005 仅见旧实验记录，无法从 checkpoint 复核。
9. **V1 权重目录为历史命名**：实际目录是 `2.0mm_cuda-v1`（连字符后置），与默认评估路径模式 `2.0mm_{version}_cuda` 不匹配——评估 V1 必须显式传 `--checkpoint SegResMamba-Lite/pipeline/models/2.0mm_cuda-v1/best_metric_model.pth`。

## 引用

```bibtex
@inproceedings{xing2024segmamba,
  title={SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation},
  author={Xing, Zhaohu and Ye, Sixiang and Yang, Yang and Liu, Guang and Zhu, Lei},
  booktitle={MICCAI},
  year={2024}
}
```
