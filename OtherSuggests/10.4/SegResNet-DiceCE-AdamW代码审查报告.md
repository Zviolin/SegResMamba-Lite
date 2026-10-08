# SegResNet-DiceCE-AdamW 代码审查报告

审查范围：`Code/SegResNet-DiceCE-AdamW`（pipeline/train.py、pipeline/evaluate.py、frame/train.py、models/__init__.py）及其依赖的 shared 模块（dataloader、ema、checkpoint、losses、metrics、inference）。审查基线为参考实现 MambaUNet-DiceCE-AdamW，并对照 `pipeline/logs/` 下的真实训练/评估日志与 `evaluation_results/` 中的逐病例 CSV 交叉验证。

## 一、问题总览

| 编号 | 等级 | 位置 | 问题简述 | 实际影响 | 触发状态 |
|------|------|------|----------|----------|----------|
| P1 | 严重 | frame/train.py#L63, L107, L160 | EMA 接口与 shared/optim/ema.py 不匹配 | 启用 `--ema` 训练必然抛 AttributeError 崩溃 | 未触发（默认 use_ema=False） |
| P2 | 严重 | pipeline/evaluate.py#L125-L141, L182 | 不加 `--cache` 时评估流程拿到空数据集并崩溃 | 默认参数运行 evaluate 直接 TypeError | 未触发（历史运行均带 --cache） |
| P3 | 严重 | pipeline/evaluate.py#L268 | 默认 checkpoint 路径指向不存在的 3.0mm 目录 | 默认参数加载权重必报 FileNotFoundError | 未触发（历史运行显式传了路径） |
| P4 | 严重 | pipeline/evaluate.py#L208 | `--runs > 1` 时 lesion-wise CSV 多轮追加不重置 | 汇总指标混入重复行，结果错误 | 未触发（历史运行 runs=1） |
| P5 | 中等 | frame/train.py#L145-L150 | 训练期验证未过滤空标签病例的 NaN/inf | dice 聚合为 nan 后最佳模型永远不再保存 | 未触发（日志无 nan，30 例验证集未出现双空病例） |
| P6 | 中等 | pipeline/train.py#L287 | `--workers` CLI 默认 8，与函数默认 0 不一致 | Windows 下 8 个 worker 进程常驻约 2.5–5GB 内存，挤占本就紧张的可用内存 | 已发生（0506/213021 长跑用了 8，跑完但内存余量极薄） |

## 二、严重问题详情

### P1：EMA 接口不匹配（启用 --ema 必崩）

frame/train.py 的 BaselineTrainer 调用了 `self.ema.register()`（L63）、`self.ema.apply_shadow()`（L107、L194）、`self.ema.restore()`（L160、L340），但共享模块 shared/optim/ema.py 中的 EMA 类只提供 `update / validate / get_state_dict / load_state_dict` 四个方法，没有上述三个方法。一旦命令行传入 `--ema`，训练会在第一次验证时抛出 `AttributeError`。

三个连带问题：其一，`AveragedModel(model, avg_fn=None)` 默认是 SWA 等权平均而非指数衰减，即使接口补齐，语义也不是 EMA；其二，latest_checkpoint 里没有保存 EMA 权重，断点续训会丢失 EMA 状态；其三，该问题在共享模块层面存在，同样影响 MambaUNet 等全部版本。由于基线训练默认 `use_ema=False`，正常基线流程不受影响，属于"潜伏"问题。

### P2：evaluate 不加 --cache 必然崩溃

`use_disk_cache=False`（CLI 默认值）时，`test_cache_dir=None`，随后 `get_val_dataloader([], ..., cache_dir=None)` 因 cache_dir 为空不会回填数据列表，得到一个空数据集；紧接着 `auto_roi_size_from_cache(None, pixdim)` 内部执行 `os.path.abspath(None)` 抛出 TypeError，评估在推理开始前就崩溃。其中无缓存分支还包含一段按 85% 分位切测试集的死代码（`test_files` 计算后从未使用），且该切分不带 seed=42，违反项目数据划分约束。修复方向：与 train.py 保持一致，未启用缓存时直接 raise 并给出明确提示，删除死分支。

### P3：默认 checkpoint 路径失效

`__main__` 中的默认检查点路径为 `pipeline/models/3.0mm/best_metric_model.pth`，但训练实际保存目录是 `models/2.0mm_cuda/`（train.py#L180 按 `{分辨率}mm_{设备}` 命名），项目中并不存在 3.0mm 目录；同时 CLI 默认 `--resolution 1.0` 与 3.0mm 路径自相矛盾。按默认参数运行评估时 `torch.load` 必报 FileNotFoundError。参考版本 MambaUNet 存在同样的模式，属于继承下来的历史遗留。

### P4：多轮评估时 lesion-wise 结果被污染

L208 的三元表达式实际等效于"只要 lesion_wise=True 就传入文件名"。CSV 表头在循环外只写一次，每轮 run 向同一文件追加 N 行；runs>1 时 `aggregate_lesion_results` 会对 2N 行重复数据取平均，全局 lesion-wise 指标混入多轮重复样本，结果失真。修复方向：runs>1 时每轮重置 CSV，或仅对第一轮做文件级汇总。

## 三、中等问题详情

### P5：训练期验证未过滤 NaN

DiceMetric 默认 `ignore_empty=False`，逐病例计算后求均值。当某病例某区域（常见于 ET）GT 为空且预测也为空时，交集与并集均为 0，Dice 为 nan；HD95 对空集合返回 inf。nan 具有传染性，30 例中只要 1 例双空，aggregate 后 `dice_wt/tc/et` 整体变 nan，导致 train.py#L243 的 `metrics["dice_wt"] > best_metric` 永远为 False，最佳模型从该轮起不再保存。逐病例评估路径（validate_verbose L262-L267）有 nan→1.0、inf→0.0 的兜底，但训练验证路径 validate() 完全没有，且 validate_verbose 返回的全局值同样来自未过滤的 aggregate，存在不一致。经查证，2.0mm 验证集日志与评估 CSV 均未出现 nan 或兜底值，说明该雷尚未踩中，但换数据划分或分辨率时有触发风险。

### P6：--workers 默认 8 与 16GB 内存环境不匹配

函数签名默认 `num_workers=0`（L35），但 CLI 默认 8（L287）并永远覆盖函数值。Windows 下 DataLoader 多进程为 spawn 模式，每个 worker 重新 import torch+MONAI+numpy，单进程常驻约 300–600MB，8 个 worker 光进程底座即 2.5–5GB；且 num_workers>0 时 dataloader.py#L165-L176 自动开启 persistent_workers 与 prefetch_factor=4，队列再吃数百 MB。日志显示训练启动时系统可用内存仅 5.2–6.0GB，叠加后余量极薄。而数据全部走 numpy 磁盘缓存（单样本约 17MB，毫秒级读取），batch_size=1 下 2 个 worker 已足够，8 个纯属内存浪费。历史日志确认 0506 与 213021 两次长跑均以 workers=8 跑完，属于"能跑但依赖内存余量"的状态。

## 四、确认无问题的部分

| 检查项 | 结论 |
|--------|------|
| 断点续训逻辑（含 last_validated_epoch 防重复验证、旧格式兼容） | 正确 |
| 数据划分从 split_info.json 读取（seed=42 固定） | 符合项目约束 |
| DiceCELoss 配置（softmax + to_onehot_y + squared_pred） | 与 4 类输出匹配 |
| CosineAnnealingLR 逐 epoch 调度、恢复后状态还原 | 正确 |
| SegResNet init_filters=16 | 适配 4GB 显存 |
| 滑动窗口推理（gaussian 权重 + AMP） | 正确 |
| 标签映射 4→3 在训练/验证两侧一致 | 正确 |
| sys.path 双根插入（shared 与 models 解析） | 正确 |

## 五、修复建议优先级

| 优先级 | 问题 | 建议动作 | 涉及范围 |
|--------|------|----------|----------|
| 1 | P2 | evaluate 无缓存时直接 raise，删除死分支 | 仅本版本 |
| 2 | P3 | 默认 checkpoint 改为按 resolution+device 动态拼接 | 仅本版本 |
| 3 | P4 | runs>1 时每轮重置 lesion CSV | 仅本版本 |
| 4 | P5 | validate() 逐病例计算并按 BraTS 惯例双空计 1.0，HD95 inf 计 0.0 | 本版本 frame/train.py（共享问题，建议各版本同步） |
| 5 | P6 | CLI --workers 默认改为 0 或 2 | 仅本版本 |
| 6 | P1 | 补齐 EMA 的 register/apply_shadow/restore 或改用统一实现，并保存 EMA 权重进 checkpoint | shared 模块，影响全部版本，需统一评估后修改 |
