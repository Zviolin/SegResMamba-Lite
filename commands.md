# 模型训练与评估命令

## 环境说明

| 模型类型 | 环境名称 | 激活命令 | 说明 |
|----------|----------|----------|------|
| **非Mamba模型** | SRTP | `conda activate SRTP` | SegResNet、VSSUNet、VMUNet 等 |
| **Mamba模型** | mamba_sm120 | `conda activate mamba_sm120` | SegMamba-Official、SegResMamba-Lite 等 |

### 环境切换

```bash
# 非Mamba模型 (SegResNet、VSSUNet、VMUNet等)
conda activate SRTP

# Mamba模型 (SegMamba-Official、SegResMamba-Lite等)
conda activate mamba_sm120
```

---

## 一、数据预处理

### 命令

```bash
python shared/data/prepare_data.py [参数]
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--resolution` | float | 2.0 | 数据分辨率 (1.0 / 2.0 / 3.0 mm) |
| `--cache` | flag | False | 生成缓存后启用缓存模式 |
| `--generate` | flag | False | 生成新的数据缓存 |
| `--train` | float | 0.7 | 训练集比例 |
| `--val` | float | 0.15 | 验证集比例 |
| `--test` | float | 0.15 | 测试集比例 |
| `--seed` | int | 42 | 随机种子 (固定划分) |
| `--workers` | int | 8 | 数据加载线程数 |
| `--samples` | int | None | 最大样本数 (None=使用全部) |
| `--dtype` | str | float16 | 缓存数据类型 (float16/float32) |
| `--cache-format` | str | numpy | 缓存格式 (numpy/tensor) |

### 示例

```bash
# 激活环境
conda activate SRTP

# 生成 2.0mm 分辨率缓存（推荐，平衡速度和精度）
python shared/data/prepare_data.py --resolution 2.0 --generate --cache --workers 8

# 生成 3.0mm 分辨率缓存，使用 50 个样本
python shared/data/prepare_data.py --resolution 3.0 --generate --cache --samples 50 --workers 4

# 使用 float32 精度（适合CPU训练）
python shared/data/prepare_data.py --resolution 2.0 --generate --cache --dtype float32 --workers 8

# 使用 tensor 格式缓存（纯PyTorch场景）
python shared/data/prepare_data.py --resolution 2.0 --generate --cache --cache-format tensor --workers 8

# 完整参数示例
python shared/data/prepare_data.py --resolution 2.0 --generate --cache --train 0.7 --val 0.15 --test 0.15 --seed 42 --dtype float16 --cache-format numpy --workers 8
```

---

## 二、训练命令

> ⚠️ **注意**：运行前请确保已激活正确的环境
> - 非Mamba模型：`conda activate SRTP`
> - Mamba模型：`conda activate mamba_sm120`

### 2.1 SegResNet-DiceCE-AdamW (Baseline)

**环境**：SRTP

#### 命令

```bash
conda activate SRTP
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code
python SegResNet-DiceCE-AdamW/pipeline/train.py [参数]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | segresnet | 模型名称 |
| `--resolution` | float | 1.0 | 数据分辨率 (1.0 / 2.0 / 3.0 mm) |
| `--workers` | int | 8 | 数据加载线程数 |
| `--batch` | int | 2 | 批次大小 |
| `--cache` | flag | False | 使用数据缓存 |
| `--ema` | flag | False | 启用 EMA |
| `--epochs` | int | 100 | 训练轮数 |
| `--val_interval` | int | 5 | 验证间隔 (每N个epoch验证一次) |
| `--device` | str | None | 设备类型 (cuda/cpu，None=自动) |

#### 示例

```bash
# GPU 训练 2 epoch
python SegResNet-DiceCE-AdamW/pipeline/train.py --epochs 2 --workers 0 --resolution 3.0 --device cuda --cache --val_interval 1 --batch 1

# CPU 训练
python SegResNet-DiceCE-AdamW/pipeline/train.py --epochs 2 --workers 0 --resolution 3.0 --device cpu --cache --val_interval 1 --batch 1
```

---

### 2.2 VSSUNet-DiceCE-AdamW

**环境**：SRTP

#### 命令

```bash
conda activate SRTP
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code
python VSSUNet-DiceCE-AdamW/pipeline/train.py [参数]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | vss_unet | 模型名称 |
| `--resolution` | float | 1.0 | 数据分辨率 (1.0 / 2.0 / 3.0 mm) |
| `--workers` | int | 8 | 数据加载线程数 |
| `--batch` | int | 2 | 批次大小 |
| `--cache` | flag | False | 使用数据缓存 |
| `--ema` | flag | False | 启用 EMA |
| `--epochs` | int | 100 | 训练轮数 |
| `--val_interval` | int | 5 | 验证间隔 |
| `--device` | str | None | 设备类型 (cuda/cpu) |

#### 示例

```bash
# GPU 训练 2 epoch
python VSSUNet-DiceCE-AdamW/pipeline/train.py --epochs 2 --workers 0 --resolution 3.0 --device cuda --cache --val_interval 1 --batch 1

# CPU 训练
python VSSUNet-DiceCE-AdamW/pipeline/train.py --epochs 2 --workers 0 --resolution 3.0 --device cpu --cache --val_interval 1 --batch 1
```

---

### 2.3 SegResNet-DiceFocal-AdamW-EMA-DS (Optimized)

**环境**：SRTP

#### 命令

```bash
conda activate SRTP
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code
python SegResNet-DiceFocal-AdamW-EMA-DS/pipeline/train.py [参数]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | segresnet | 模型名称 |
| `--resolution` | float | 1.0 | 数据分辨率 (1.0 / 2.0 / 3.0 mm) |
| `--workers` | int | 8 | 数据加载线程数 |
| `--batch` | int | 2 | 批次大小 |
| `--cache` | flag | False | 使用数据缓存 |
| `--ema` | flag | False | 启用 EMA (默认开启) |
| `--ds` | flag | False | 启用深层监督 (默认开启) |
| `--epochs` | int | 150 | 训练轮数 |
| `--val_interval` | int | 5 | 验证间隔 |
| `--device` | str | None | 设备类型 (cuda/cpu) |

#### 示例

```bash
# GPU 训练 2 epoch
python SegResNet-DiceFocal-AdamW-EMA-DS/pipeline/train.py --epochs 2 --workers 0 --resolution 3.0 --device cuda --cache --val_interval 1 --batch 1

# CPU 训练
python SegResNet-DiceFocal-AdamW-EMA-DS/pipeline/train.py --epochs 2 --workers 0 --resolution 3.0 --device cpu --cache --val_interval 1 --batch 1
```

---

### 2.4 VMUNet-DiceCE-AdamW (UltraLight)

**环境**：SRTP

#### 命令

```bash
conda activate SRTP
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code
python VMUNet-DiceCE-AdamW/pipeline/train.py [参数]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | vm_unet | 模型名称 |
| `--resolution` | float | 1.0 | 数据分辨率 (1.0 / 2.0 / 3.0 mm) |
| `--workers` | int | 8 | 数据加载线程数 |
| `--batch` | int | 2 | 批次大小 |
| `--cache` | flag | False | 使用数据缓存 |
| `--ema` | flag | False | 启用 EMA |
| `--epochs` | int | 100 | 训练轮数 |
| `--val_interval` | int | 5 | 验证间隔 |
| `--device` | str | None | 设备类型 (cuda/cpu) |

#### 示例

```bash
# GPU 训练 2 epoch
python VMUNet-DiceCE-AdamW/pipeline/train.py --epochs 2 --workers 0 --resolution 3.0 --device cuda --cache --val_interval 1 --batch 1

# CPU 训练
python VMUNet-DiceCE-AdamW/pipeline/train.py --epochs 2 --workers 0 --resolution 3.0 --device cpu --cache --val_interval 1 --batch 1
```

---

### 2.5 SegMamba-Official（官方SegMamba）

**环境**：mamba_sm120 ⚠️

#### 命令

```bash
conda activate mamba_sm120
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code
python SegMamba-Official/pipeline/train.py [参数]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | segmamba | 模型名称 |
| `--resolution` | float | 2.0 | 数据分辨率 (1.0 / 2.0 / 3.0 mm) |
| `--workers` | int | 8 | 数据加载线程数 |
| `--batch` | int | 2 | 批次大小 |
| `--cache` | flag | False | 使用数据缓存 |
| `--ema` | flag | False | 启用 EMA |
| `--epochs` | int | 150 | 训练轮数 |
| `--val_interval` | int | 5 | 验证间隔 |
| `--device` | str | None | 设备类型 (cuda/cpu) |

#### 示例

```bash
# GPU 训练（推荐配置）
python SegMamba-Official/pipeline/train.py --epochs 150 --workers 8 --resolution 2.0 --device cuda --cache --val_interval 5 --batch 2

# 测试训练（2个epoch）
python SegMamba-Official/pipeline/train.py --epochs 2 --workers 4 --resolution 2.0 --device cuda --cache --val_interval 1 --batch 1
```

#### 说明

- **使用真正的 mamba_ssm 包**，需要安装 `mamba-ssm`
- 参数量约 30M，需要较多显存
- 推荐使用 batch_size=2，搭配 float16 缓存

---

### 2.6 SegResMamba-Lite（轻量化SegResMamba）

**环境**：mamba_sm120 ⚠️

#### 命令

```bash
conda activate mamba_sm120
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code
python SegResMamba-Lite/pipeline/train.py [参数]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | segresmamba_lite | 模型名称 |
| `--resolution` | float | 2.0 | 数据分辨率 (1.0 / 2.0 / 3.0 mm) |
| `--workers` | int | 4 | 数据加载线程数 |
| `--batch` | int | 4 | 批次大小 |
| `--cache` | flag | False | 使用数据缓存 |
| `--ema` | flag | False | 启用 EMA |
| `--epochs` | int | 100 | 训练轮数 |
| `--val_interval` | int | 5 | 验证间隔 |
| `--device` | str | None | 设备类型 (cuda/cpu) |
| `--init_filters` | int | 8 | 初始滤波器数量 (8=1.39M) |

#### 示例

```bash
# GPU 训练（推荐配置）
python SegResMamba-Lite/pipeline/train.py --epochs 100 --workers 4 --resolution 2.0 --device cuda --cache --val_interval 5 --batch 4 --ema

# 使用默认参数训练
python SegResMamba-Lite/pipeline/train.py --resolution 2.0 --cache --ema

# 测试训练（2个epoch）
python SegResMamba-Lite/pipeline/train.py --epochs 2 --workers 4 --resolution 2.0 --device cuda --cache --val_interval 1 --batch 1

# 优化训练 - 增大模型容量
python SegResMamba-Lite/pipeline/train.py --init_filters 16 --cache --ema --epochs 150 --val_interval 5 --compile
```

#### 说明

- **使用真正的 mamba_ssm 包**，但参数量控制在 1.4M 以内
- 包含三个核心创新：双向VSS、CNN+Mamba并行融合、特征回流
- init_filters=8 时参数量约 1.39M，满足<1.5M要求

---

## 三、评估命令

> ⚠️ **注意**：运行前请确保已激活正确的环境
> - 非Mamba模型：`conda activate SRTP`
> - Mamba模型：`conda activate mamba_sm120`

### 3.1 SegResNet-DiceCE-AdamW

**环境**：SRTP

#### 命令

```bash
conda activate SRTP
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code
python SegResNet-DiceCE-AdamW/pipeline/evaluate.py [参数]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | segresnet | 模型名称 |
| `--resolution` | float | 1.0 | 数据分辨率 (1.0 / 2.0 / 3.0 mm) |
| `--checkpoint` | str | 自动 | 检查点路径 (默认使用best_metric_model.pth) |
| `--workers` | int | 8 | 数据加载线程数 |
| `--cache` | flag | False | 使用数据缓存 |
| `--device` | str | None | 设备类型 (cuda/cpu) |
| `--no_lesion_wise` | flag | False | 禁用 lesion-wise 指标 |

#### 示例

```bash
# GPU 评估 (使用默认 best_metric_model.pth)
python SegResNet-DiceCE-AdamW/pipeline/evaluate.py --resolution 3.0 --device cuda --cache --workers 0

# CPU 评估
python SegResNet-DiceCE-AdamW/pipeline/evaluate.py --resolution 3.0 --device cpu --cache --workers 0

# 指定 checkpoint 路径
python SegResNet-DiceCE-AdamW/pipeline/evaluate.py --resolution 3.0 --device cuda --cache --workers 0 --checkpoint "SegResNet-DiceCE-AdamW/pipeline/models/3.0mm_cuda/best_metric_model.pth"

# 禁用 lesion-wise 评估
python SegResNet-DiceCE-AdamW/pipeline/evaluate.py --resolution 3.0 --device cuda --cache --workers 0 --no_lesion_wise
```

---

### 3.2 VSSUNet-DiceCE-AdamW

**环境**：SRTP

#### 命令

```bash
conda activate SRTP
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code
python VSSUNet-DiceCE-AdamW/pipeline/evaluate.py [参数]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | vss_unet | 模型名称 |
| `--resolution` | float | 1.0 | 数据分辨率 (1.0 / 2.0 / 3.0 mm) |
| `--checkpoint` | str | 自动 | 检查点路径 |
| `--workers` | int | 8 | 数据加载线程数 |
| `--cache` | flag | False | 使用数据缓存 |
| `--device` | str | None | 设备类型 (cuda/cpu) |
| `--no_lesion_wise` | flag | False | 禁用 lesion-wise 指标 |

#### 示例

```bash
# GPU 评估
python VSSUNet-DiceCE-AdamW/pipeline/evaluate.py --resolution 3.0 --device cuda --cache --workers 0

# 指定 checkpoint 路径
python VSSUNet-DiceCE-AdamW/pipeline/evaluate.py --resolution 3.0 --device cuda --cache --workers 0 --checkpoint "VSSUNet-DiceCE-AdamW/pipeline/models/3.0mm_cuda/best_metric_model.pth"
```

---

### 3.3 SegResNet-DiceFocal-AdamW-EMA-DS

**环境**：SRTP

#### 命令

```bash
conda activate SRTP
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code
python SegResNet-DiceFocal-AdamW-EMA-DS/pipeline/evaluate.py [参数]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | segresnet | 模型名称 |
| `--resolution` | float | 1.0 | 数据分辨率 (1.0 / 2.0 / 3.0 mm) |
| `--checkpoint` | str | 自动 | 检查点路径 |
| `--workers` | int | 8 | 数据加载线程数 |
| `--cache` | flag | False | 使用数据缓存 |
| `--device` | str | None | 设备类型 (cuda/cpu) |
| `--no_lesion_wise` | flag | False | 禁用 lesion-wise 指标 |

#### 示例

```bash
# GPU 评估
python SegResNet-DiceFocal-AdamW-EMA-DS/pipeline/evaluate.py --resolution 3.0 --device cuda --cache --workers 0

# 指定 checkpoint 路径
python SegResNet-DiceFocal-AdamW-EMA-DS/pipeline/evaluate.py --resolution 3.0 --device cuda --cache --workers 0 --checkpoint "SegResNet-DiceFocal-AdamW-EMA-DS/pipeline/models/3.0mm_cuda/best_metric_model.pth"
```

---

### 3.4 VMUNet-DiceCE-AdamW

**环境**：SRTP

#### 命令

```bash
conda activate SRTP
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code
python VMUNet-DiceCE-AdamW/pipeline/evaluate.py [参数]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | vm_unet | 模型名称 |
| `--resolution` | float | 1.0 | 数据分辨率 (1.0 / 2.0 / 3.0 mm) |
| `--checkpoint` | str | 自动 | 检查点路径 |
| `--workers` | int | 8 | 数据加载线程数 |
| `--cache` | flag | False | 使用数据缓存 |
| `--device` | str | None | 设备类型 (cuda/cpu) |
| `--no_lesion_wise` | flag | False | 禁用 lesion-wise 指标 |

#### 示例

```bash
# GPU 评估
python VMUNet-DiceCE-AdamW/pipeline/evaluate.py --resolution 3.0 --device cuda --cache --workers 0

# 指定 checkpoint 路径
python VMUNet-DiceCE-AdamW/pipeline/evaluate.py --resolution 3.0 --device cuda --cache --workers 0 --checkpoint "VMUNet-DiceCE-AdamW/pipeline/models/3.0mm_cuda/best_metric_model.pth"
```

---

### 3.5 SegMamba-Official

**环境**：mamba_sm120 ⚠️

#### 命令

```bash
conda activate mamba_sm120
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code
python SegMamba-Official/pipeline/evaluate.py [参数]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | segmamba | 模型名称 |
| `--resolution` | float | 2.0 | 数据分辨率 (1.0 / 2.0 / 3.0 mm) |
| `--checkpoint` | str | 自动 | 检查点路径 |
| `--workers` | int | 8 | 数据加载线程数 |
| `--cache` | flag | False | 使用数据缓存 |
| `--device` | str | None | 设备类型 (cuda/cpu) |
| `--no_lesion_wise` | flag | False | 禁用 lesion-wise 指标 |

#### 示例

```bash
# GPU 评估
python SegMamba-Official/pipeline/evaluate.py --resolution 2.0 --device cuda --cache --workers 8

# 指定 checkpoint 路径
python SegMamba-Official/pipeline/evaluate.py --resolution 2.0 --device cuda --cache --workers 8 --checkpoint "SegMamba-Official/pipeline/models/2.0mm_cuda/best_metric_model.pth"
```

---

### 3.6 SegResMamba-Lite

**环境**：mamba_sm120 ⚠️

#### 命令

```bash
conda activate mamba_sm120
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code
python SegResMamba-Lite/pipeline/evaluate.py [参数]
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | segresmamba_lite | 模型名称 |
| `--resolution` | float | 2.0 | 数据分辨率 (1.0 / 2.0 / 3.0 mm) |
| `--checkpoint` | str | 自动 | 检查点路径 |
| `--workers` | int | 8 | 数据加载线程数 |
| `--cache` | flag | False | 使用数据缓存 |
| `--device` | str | None | 设备类型 (cuda/cpu) |
| `--no_lesion_wise` | flag | False | 禁用 lesion-wise 指标 |

#### 示例

```bash
# GPU 评估
python SegResMamba-Lite/pipeline/evaluate.py --resolution 2.0 --device cuda --cache --workers 8

# 指定 checkpoint 路径
python SegResMamba-Lite/pipeline/evaluate.py --resolution 2.0 --device cuda --cache --workers 8 --checkpoint "SegResMamba-Lite/pipeline/models/2.0mm_cuda/best_metric_model.pth"
```

---

## 四、版本对照表

| 版本目录 | 模型 | Loss | Optimizer | Mamba实现 | 参数量 | 特性 | 环境 |
|----------|------|------|-----------|-----------|-------|------|------|
| SegResNet-DiceCE-AdamW | SegResNet | Dice+CE | AdamW | - | ~2.6M | Baseline | **SRTP** |
| VSSUNet-DiceCE-AdamW | VSSUNet | Dice+CE | AdamW | VSSBlock(轻量) | ~0.2M | 轻量化VSS架构（非真正Mamba）| **SRTP** |
| SegResNet-DiceFocal-AdamW-EMA-DS | SegResNet | Dice+Focal | AdamW | - | ~2.6M | EMA + Deep Supervision | **SRTP** |
| VMUNet-DiceCE-AdamW | VMUNet (VSS) | Dice+CE | AdamW | VSSBlock(轻量) | ~0.3M | 超轻量化 | **SRTP** |
| **SegMamba-Official** | SegMamba | Dice+CE | AdamW | **mamba_ssm(真正)** | ~30M | 官方SegMamba | **mamba_sm120** ⚠️ |
| **SegResMamba-Lite** | SegResMamba | Dice+Focal | AdamW | **mamba_ssm(真正)** | ~1.4M | 真正Mamba + 轻量化 | **mamba_sm120** ⚠️ |

---

## 五、注意事项

1. **环境选择**：运行前必须先激活正确的 Conda 环境
   - 非Mamba模型：必须使用 `conda activate SRTP`
   - Mamba模型：必须使用 `conda activate mamba_sm120`

2. **数据缓存**：首次运行需要生成缓存，使用 `--generate --cache` 参数，后续运行只需 `--cache`

3. **模型存储**：GPU 和 CPU 训练使用不同的模型目录
   - GPU: `models/3.0mm_cuda/`
   - CPU: `models/3.0mm_cpu/`

4. **断点续训**：训练中途中断后，重新运行相同命令会自动从 `latest_checkpoint.pth` 恢复

5. **ROI Size**：自动从缓存数据推断，无需手动设置

6. **Lesion-wise 指标**：默认启用，使用 `--no_lesion_wise` 可禁用

---

## 六、快速开始

### 非Mamba模型（SegResNet、VSSUNet、VMUNet等）

```bash
# 1. 激活环境
conda activate SRTP
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code

# 2. 生成缓存 (首次，推荐配置)
python shared/data/prepare_data.py --resolution 2.0 --generate --cache --workers 8

# 3. 训练 Baseline
python SegResNet-DiceCE-AdamW/pipeline/train.py --epochs 100 --workers 8 --resolution 2.0 --device cuda --cache --batch 2

# 4. 评估
python SegResNet-DiceCE-AdamW/pipeline/evaluate.py --resolution 2.0 --device cuda --cache --workers 8
```

### Mamba模型（SegMamba-Official、SegResMamba-Lite等）

```bash
# 1. 激活环境
conda activate mamba_sm120
cd g:/Codes/Python/SRTP/Essay/PythonFiles/Code

# 2. 生成缓存 (首次，推荐配置)
python shared/data/prepare_data.py --resolution 2.0 --generate --cache --workers 8

# 3. 训练官方 SegMamba (需要安装 mamba-ssm)
python SegMamba-Official/pipeline/train.py --epochs 150 --workers 8 --resolution 2.0 --device cuda --cache --batch 2

# 4. 训练轻量化 SegResMamba (推荐)
python SegResMamba-Lite/pipeline/train.py --epochs 150 --workers 8 --resolution 2.0 --device cuda --cache --batch 2 --ema

# 5. 评估
python SegResMamba-Lite/pipeline/evaluate.py --resolution 2.0 --device cuda --cache --workers 8
```

### 模型选择建议

| 场景 | 推荐模型 | 理由 | 环境 |
|------|---------|------|------|
| **基准对比** | SegResNet-DiceCE-AdamW | Baseline，无Mamba依赖 | **SRTP** |
| **轻量级部署** | VSSUNet-DiceCE-AdamW | VSSBlock，~0.2M参数 | **SRTP** |
| **追求最佳性能** | SegMamba-Official | 真正Mamba，~30M参数 | **mamba_sm120** |
| **平衡性能与效率** | SegResMamba-Lite | 真正Mamba，~1.4M参数 | **mamba_sm120** |