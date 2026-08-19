"""V9 快速验证脚本"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from models.v9 import get_model

model = get_model(init_filters=20, device='cpu')
total = sum(p.numel() for p in model.parameters())
print(f"V9 参数量: {total:,} ({total/1e6:.3f}M)")

x = torch.randn(1, 4, 64, 64, 64)
model.eval()
with torch.no_grad():
    out = model(x)
print(f"输入: {x.shape}, 输出: {out.shape}")

weights = model.get_moe_route_weights(x)
print(f"MoE 路由权重: {[round(w, 4) for w in weights[0].tolist()]}")
print(f"权重和: {weights[0].sum().item():.4f}")

topk_idx, topk_w = model.get_moe_topk_indices(x)
print(f"Top-K 索引: {topk_idx[0].tolist()}")
print(f"Top-K 权重: {[round(w, 4) for w in topk_w[0].tolist()]}")

# Backward test
loss = out.sum()
loss.backward()
print("Backward OK")

# 负载均衡损失测试
route_weights = model.get_moe_route_weights(x)
lb_loss = model.moe_8.load_balancing_loss(route_weights)
print(f"负载均衡损失: {lb_loss.item():.6f}")