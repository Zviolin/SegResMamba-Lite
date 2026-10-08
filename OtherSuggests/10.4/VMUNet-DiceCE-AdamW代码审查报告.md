# VMUNet-DiceCE-AdamW 代码审查报告

审查日期：2026-10-04

审查范围：本版本目录下的 `pipeline/train.py`、`pipeline/evaluate.py`、`frame/train.py`、`models/__init__.py`，并核对了所依赖的 `shared` 模块（dataloader、checkpoint、logger、lesion_metric、aggregate_lesion、sliding_window、postprocess）的接口签名与行为。

总体结论：代码整体结构清晰，训练主流程（缓存加载 → 训练 → 验证 → 断点续训）和评估主流程（缓存加载 → 滑窗推理 → 全局/逐病例/lesion-wise 指标）功能完整，历史上已成功跑通训练与评估。本次共发现 6 个问题，其中 1 个可能实际影响当前工作流（问题 5），其余均属于特定场景触发或死代码问题。

## 问题影响评估总表

影响评估结合了实际使用方式：训练与评估均固定启用 `--cache`、评估时总是显式指定 `--checkpoint`、评估次数 runs=1、训练未启用 `--ema`。在此工作流下，多数问题处于"休眠"状态，仅对不了解脚本使用前提的裸跑场景构成陷阱。

| 编号 | 问题 | 触发条件 | 对当前工作流的影响 | 实际风险 | 处理建议 |
|------|------|----------|--------------------|----------|----------|
| 1 | evaluate.py 无缓存时崩溃或空跑 | 运行评估时不带 `--cache` | 不触发（固定使用 `--cache`） | 低（防御性缺失） | 对齐 train.py 加显式报错，删除死代码 |
| 2 | 默认检查点路径不存在 | 裸跑 evaluate.py 且不传 `--checkpoint` | 不触发（总是显式传参） | 低（模板遗留） | 修正默认路径为 2.0mm_cuda |
| 3 | runs>1 时 lesion 结果跨轮污染 | 评估使用 `--runs` 大于 1 | 不触发（runs=1） | 中（一旦触发结果错误且难察觉） | 多次评估前必须修复 |
| 4 | 验证失败后无法补验证 | 训练中某轮验证异常后断点续训 | 影响极小（后续轮次可覆盖） | 低 | 可无视，或验证失败时保存 `epoch` 而非 `epoch+1` |
| 5 | 全局指标 NaN 污染 | 测试集中存在 ET 真实为空且预测也为空的病例 | **可能实际发生**，日志中出现 nan、HD95 avg 失真 | **中高（唯一可能实际发生）** | **建议优先修复**：聚合时过滤或清洗 NaN |
| 6 | EMA 状态未存入 checkpoint | 训练启用 `--ema` 且断点续训 | 不触发（未启用 EMA） | 低 | 可无视，或 checkpoint 增加 ema_state_dict |

## 问题清单

**问题 1：evaluate.py 不带 --cache 时必然崩溃或空跑（严重级，当前工作流不触发）**

evaluate.py 的数据加载设计只支持缓存模式，但缺少 train.py 中的显式保护。当 `use_disk_cache=False` 时，`current_cache_dir = None`，随后传入 `get_val_dataloader` 的是空文件列表加 `cache_dir=None`，由于 dataloader 的回填逻辑要求 `cache_dir` 非空才会从 `split_info.json` 读取样本列表，最终得到一个零样本的空数据集；同时 `auto_roi_size_from_cache(None)` 内部调用 `os.path.abspath(None)` 会直接抛出 TypeError。也就是说，不带 `--cache` 运行评估脚本时，要么在读到评估之前崩溃，要么（若绕过崩溃）对零个样本进行评估，产出只有表头的空 CSV。evaluate.py 中 L139 处的 85% 无缓存划分分支因此是永远走不通的死代码。当前工作流固定使用 `--cache`，此问题不会触发，属于防御性缺失；对齐 train.py 的做法（无缓存时直接 `raise ValueError` 并删除死代码）即可修复。

**问题 2：默认检查点路径不存在（中等级，当前工作流不触发）**

evaluate.py 中 argparse 的 `--checkpoint` 默认值硬编码为 `pipeline/models/3.0mm/best_metric_model.pth`，而实际模型目录为 `models/2.0mm_cuda/`。裸跑 `python evaluate.py` 会因路径不存在而 FileNotFoundError。由于实际运行时总是显式传入 `--checkpoint`，该默认值从未被使用，因此对当前工作流无影响；修正默认路径为 `models/2.0mm_cuda/best_metric_model.pth` 即可消除隐患（注：MambaUNet 参考项目存在同样的模板遗留问题）。

**问题 3：runs > 1 时 lesion-wise 结果跨 run 污染（中等级，当前工作流 runs=1 不触发）**

evaluate.py L208 的三元表达式存在逻辑错误：当 `runs > 1` 且 `lesion_wise=True` 时仍将 lesion 结果文件传入 `validate_verbose`。该函数以追加模式逐病例写入 CSV，且每轮结束时调用 `aggregate_lesion_results()` 汇总整个文件的内容，导致第 2 轮的汇总把第 1 轮的行也计入，均值被稀释，结果错误。全局指标文件（results_file）在 `runs > 1` 时正确地传了 None，lesion 文件应做同样处理；若需要多次评估的 lesion 结果，应每轮写入独立文件。

**问题 4：验证失败后不会重新尝试验证，注释与实际行为矛盾（低级别，影响极小）**

train.py 中当某轮验证失败（如显存不足抛异常）时，代码故意不更新 `last_validated_epoch`，注释承诺"下次运行时会重新尝试验证"。但 checkpoint 仍以 `"epoch": epoch + 1` 保存训练进度，恢复训练后循环从 epoch+1 开始，失败验证的那个 epoch 不在循环范围内，永远不会被补验证；同时该 epoch 的权重也未保存（保存语句位于 try 块内验证成功之后）。实际影响很小：后续仍有大量验证机会，最佳模型的发现不受影响。修复方式是验证失败时保存 `"epoch": epoch`（而非 epoch+1），或接受现状并修正注释。

**问题 5：全局指标未清洗 NaN，空 ET 病例会污染整轮结果（中等级，当前工作流可能实际发生）**

BraTS 数据中部分病例的增强肿瘤（ET）真实标签为空。计算指标时，若某病例预测 ET 与真实 ET 同时为空，Dice 与 HD95 数学上为 0÷0，MONAI 返回 NaN；聚合时 `aggregate()` 取 mean，单个 NaN 即令整轮的 `dice_et`/`hd95_et` 变为 NaN，HD95 avg 随之失真。`validate_verbose` 中逐病例写 CSV 前已做 NaN 清洗（L258-263，Dice 空对空记 1.0、HD95 记 0.0），但全局聚合与训练时验证的 `validate()` 路径均未做同样处理。由于最佳模型选择仅依赖 `dice_wt`，模型选择不受影响，但验证日志的可信度受损。这是唯一可能在实际运行中出现的问题，可通过在聚合后对 NaN 做与逐病例一致的清洗来修复，或在聚合前使用过滤 NaN 的均值。

**问题 6：EMA 状态未存入 checkpoint（低级别，当前工作流不触发）**

train.py 的 checkpoint 仅保存 model/optimizer/scheduler/scaler 四项状态，未包含 EMA 的 shadow 权重。若启用 `--ema` 训练，断点续训读档后 EMA 需从当前权重重新 register，历史平滑信息断裂，EMA 的连续性被破坏。当前训练未启用 EMA，此问题不触发；修复方式是在 checkpoint 中增加 `ema_state_dict` 并在恢复时相应加载。

## 优先级建议

按当前工作流（固定 2.0mm 分辨率、启用缓存、显式指定 checkpoint、runs=1、未启用 EMA）校准后：问题 5 是唯一可能实际发生的问题，影响评估与验证日志的可信度，建议优先处理；问题 1、2、3、6 均不触发，可作为防御性加固择机修复；问题 4 影响极小可无视。

若进行代码修改，按工作区规则需同步检查并更新 README.md 中的相关描述。
