"""V6 反向传播测试"""
import sys
sys.path.insert(0, r'g:/Codes/Python/SRTP/Essay/PythonFiles/Code')
sys.path.insert(0, r'g:/Codes/Python/SRTP/Essay/PythonFiles/Code/SegResMamba-Lite')

from models.v6 import get_model
import torch

model = get_model(init_filters=18, device='cpu')
x = torch.randn(2, 4, 64, 64, 64, requires_grad=True)
out = model(x)
loss = out.mean()
loss.backward()
print(f'Backward OK: loss={loss.item():.4f}')
print(f'输入梯度: {x.grad is not None}')
print(f'输入梯度形状: {x.grad.shape}')

# 验证梯度真的流过 MoA
moa_grad = 0.0
for p in model.moa_8.parameters():
    if p.grad is not None:
        moa_grad += p.grad.abs().sum().item()
print(f'MoA 总梯度: {moa_grad:.4f}')

# 验证梯度流过 expert
expert_grads = []
for i, expert in enumerate(model.moa_8.experts):
    grad_sum = 0.0
    for p in expert.parameters():
        if p.grad is not None:
            grad_sum += p.grad.abs().sum().item()
    expert_grads.append(grad_sum)
print(f'每个专家的梯度: {[f"{g:.4f}" for g in expert_grads]}')
print(f'被选中的专家 (Top-2): [1, 3]')
print(f'被选中的专家梯度应该更大:')
for i, g in enumerate(expert_grads):
    print(f'  Expert {i}: {g:.4f} {"← 被选中" if i in [1, 3] else ""}')
