"""
LightSegMamba-DiceCE-AdamW-V3

V3 相对 V2 的改进：
- 默认 base_channels=32（约 1.4M 参数，实测 1,369,604，比 V2 807K 容量翻倍）
- 严格对齐 MambaUNet-DiceCE-AdamW 的 CLI 风格（--cache/--batch/--workers/--patch-size/--base-channels）
- 训练规模通过 --max-samples 命令行可选（200 例快速验证 / 1251 例全量）
- batch_size 与 num_workers 完全 CLI 可调
"""
