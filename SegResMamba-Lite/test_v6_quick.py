"""V6 快速测试"""
import sys
sys.path.insert(0, r'g:/Codes/Python/SRTP/Essay/PythonFiles/Code')
sys.path.insert(0, r'g:/Codes/Python/SRTP/Essay/PythonFiles/Code/SegResMamba-Lite')

from models.v6 import get_model
import torch

# 1. 模型创建
model = get_model(init_filters=18, device='cpu')
total = sum(p.numel() for p in model.parameters())
print(f'V6 参数量 (init_filters=18): {total:,} ({total/1e6:.3f}M)')
print(f'符合 <1.5M 限制: {total < 1.5e6}')

# 2. 前向
x = torch.randn(1, 4, 64, 64, 64)
model.eval()
with torch.no_grad():
    out = model(x)
print(f'Forward OK: input={x.shape}, output={out.shape}')
print(f'输出和输入形状一致: {out.shape == x.shape}')

# 3. 路由权重
weights = model.get_moa_route_weights(x)
print(f'路由权重: {[round(w, 4) for w in weights[0].tolist()]}')
print(f'权重和: {weights[0].sum().item():.4f}')

topk_idx = torch.topk(weights[0], 2).indices.tolist()
print(f'Top-2 专家: {topk_idx}')
print(f'Top-2 权重: {round(weights[0][topk_idx[0]].item(), 4)}, {round(weights[0][topk_idx[1]].item(), 4)}')
print(f'Top-2 权重和: {round((weights[0][topk_idx[0]] + weights[0][topk_idx[1]]).item(), 4)}')

# 4. 反向
loss = ((out - torch.randn_like(out)) ** 2).mean()
loss.backward()
print(f'Backward OK: loss={loss.item():.4f}')

# 5. 验证 init_filters=20 是否也能 <1.5M
model20 = get_model(init_filters=20, device='cpu')
total20 = sum(p.numel() for p in model20.parameters())
print(f'V6 参数量 (init_filters=20): {total20:,} ({total20/1e6:.3f}M)')
print(f'符合 <1.5M 限制: {total20 < 1.5e6}')
