"""
【LightSegMamba 训练 / 评估 pipeline V3】

入口：
- pipeline/gen_cache.py:  生成 PersistentDataset 缓存（支持 --max-samples）
- pipeline/train.py:      训练入口（CLI 对齐 MambaUNet 风格）
- pipeline/evaluate.py:   评估入口（含 lesion-wise）
"""
