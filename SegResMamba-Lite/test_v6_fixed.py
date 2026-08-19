"""V6 修复后的稳定性测试"""
import sys
sys.path.insert(0, r'g:/Codes/Python/SRTP/Essay/PythonFiles/Code')
sys.path.insert(0, r'g:/Codes/Python/SRTP/Essay/PythonFiles/Code/SegResMamba-Lite')

from models.v6 import get_model
import torch

model = get_model(init_filters=20, device='cpu')

# 1. 训练模式测试
print("=" * 60)
print("训练模式测试（验证 soft weight 路径）")
print("=" * 60)
model.train()
x = torch.randn(2, 4, 64, 64, 64, requires_grad=True)
out = model(x)
loss = out.mean()
loss.backward()
print(f"Forward OK: input={x.shape}, output={out.shape}")
print(f"输入梯度: {x.grad is not None}")
print(f"损失: {loss.item():.4f}")

# 检查所有专家梯度（应该都有，非零）
print("\n所有专家梯度（验证全专家激活）:")
for i, expert in enumerate(model.moa_8.experts):
    g = sum(p.grad.abs().sum().item() for p in expert.parameters() if p.grad is not None)
    print(f"  Expert {i}: 梯度={g:.6f}")

# 路由器梯度
router_g = sum(p.grad.abs().sum().item() for p in model.moa_8.router.parameters() if p.grad is not None)
print(f"路由器梯度: {router_g:.6f}")

# 2. 验证模式测试
print("\n" + "=" * 60)
print("验证模式测试")
print("=" * 60)
model.eval()
with torch.no_grad():
    x = torch.randn(1, 4, 64, 64, 64)
    out = model(x)
    weights = model.get_moa_route_weights(x)
print(f"推理 OK: input={x.shape}, output={out.shape}")
print(f"路由权重: {[round(w, 4) for w in weights[0].tolist()]}")
print(f"权重和: {weights[0].sum().item():.4f}")

# 3. 多次 step 验证梯度稳定
print("\n" + "=" * 60)
print("多次 step 验证（稳定性）")
print("=" * 60)
model.train()
model.zero_grad()
for step in range(3):
    x = torch.randn(2, 4, 64, 64, 64)
    out = model(x)
    loss = out.mean()
    # 模拟 loss 反向
    loss.backward()
    total_grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print(f"Step {step+1}: loss={loss.item():.4f}, 总梯度={total_grad:.6f}")
    model.zero_grad()
