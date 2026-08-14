"""
检查模型参数量（支持 V1-V4）
"""
import sys
import os

# 添加路径
version_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, version_root)

from models import get_model

print("=" * 60)
print("SegResMamba-Lite 各版本参数统计")
print("=" * 60)

# 检查所有版本
versions = ['v1', 'v2', 'v3', 'v4']
init_filters_list = [26, 22, 20, 20]  # 各版本默认值

for version, init_filters in zip(versions, init_filters_list):
    # 创建模型
    model = get_model(version=version, init_filters=init_filters, use_deep_supervision=False, device='cpu')
    
    # 计算参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'=' * 60}")
    print(f"SegResMamba-Lite {version.upper()} (init_filters={init_filters})")
    print(f"{'=' * 60}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"总参数量 (M): {total_params/1e6:.2f}M")
    print(f"是否 <1.5M: {'✅ 是' if total_params < 1.5e6 else '❌ 否'}")
    print(f"{'=' * 60}")
    
    # 分层统计
    print("\n各模块参数量:")
    print("-" * 40)
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        if params > 0:
            print(f"{name}: {params:,} ({params/1e6:.2f}M)")
