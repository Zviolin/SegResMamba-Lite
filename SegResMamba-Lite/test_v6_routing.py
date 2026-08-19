"""V6 稀疏 MoA 路由工作验证"""
import sys
sys.path.insert(0, r'g:/Codes/Python/SRTP/Essay/PythonFiles/Code')
sys.path.insert(0, r'g:/Codes/Python/SRTP/Essay/PythonFiles/Code/SegResMamba-Lite')

from models.v6 import get_model
import torch

model = get_model(init_filters=18, device='cpu')

# 多个 batch 验证路由
print("=" * 60)
print("不同 batch 的路由分析")
print("=" * 60)

for i in range(3):
    x = torch.randn(2, 4, 64, 64, 64)  # batch=2
    weights = model.get_moa_route_weights(x)
    print(f"\nBatch {i+1}:")
    for b in range(2):
        w = weights[b].tolist()
        top2_idx = torch.topk(weights[b], 2).indices.tolist()
        top2_w = torch.topk(weights[b], 2).values.tolist()
        print(f"  Sample {b}: 权重={[round(ww,4) for ww in w]}")
        print(f"  Top-2 专家: {top2_idx}, 权重: {[round(ww,4) for ww in top2_w]}")

# 单 batch 验证梯度
print("\n" + "=" * 60)
print("单 batch 路由-梯度一致性验证")
print("=" * 60)

x = torch.randn(1, 4, 64, 64, 64, requires_grad=True)
out = model(x)
loss = out.mean()
loss.backward()

# 获取路由选择
weights = model.get_moa_route_weights(x)
top2_idx = torch.topk(weights[0], 2).indices.tolist()
top2_w = torch.topk(weights[0], 2).values.tolist()

print(f"\n路由 Top-2 专家: {top2_idx}")
print(f"路由 Top-2 权重: {[round(w, 4) for w in top2_w]}")

print(f"\n各专家梯度（应该只有 Top-2 的有梯度）:")
for i, expert in enumerate(model.moa_8.experts):
    grad_sum = sum(p.grad.abs().sum().item() for p in expert.parameters() if p.grad is not None)
    expected = "✓ 被选中" if i in top2_idx else "✗ 未选中"
    print(f"  Expert {i}: 梯度={grad_sum:.6f}  {expected}")

# 验证：被选中的专家梯度 > 0
print(f"\n稀疏路由验证:")
for i in top2_idx:
    g = sum(p.grad.abs().sum().item() for p in model.moa_8.experts[i].parameters() if p.grad is not None)
    print(f"  Expert {i} 梯度 = {g:.6f} {'> 0 ✓' if g > 0 else '= 0 ✗'}")
for i in range(4):
    if i not in top2_idx:
        g = sum(p.grad.abs().sum().item() for p in model.moa_8.experts[i].parameters() if p.grad is not None)
        print(f"  Expert {i} (未选中) 梯度 = {g:.6f} {'= 0 ✓ (稀疏生效)' if g == 0 else '> 0 ✗ (未稀疏)'}")
