"""V8 真正自由 α 测试脚本"""
import sys
sys.path.insert(0, r'g:/Codes/Python/SRTP/Essay/PythonFiles/Code')
sys.path.insert(0, r'g:/Codes/Python/SRTP/Essay/PythonFiles/Code/SegResMamba-Lite')

import torch
from frame.train import OptimizedTrainer
from models import get_model

model = get_model(version='v8', init_filters=20, device='cuda')

trainer = OptimizedTrainer(
    model=model,
    device='cuda',
    loss_name='DiceFocalLoss',
    lr=1e-4,
    max_epochs=100,
    use_ema=True,
)

print('OK OptimizedTrainer 初始化成功')
print(f'Optimizer param_groups: {len(trainer.optimizer.param_groups)}')
for i, pg in enumerate(trainer.optimizer.param_groups):
    lr = pg['lr']
    n = len(pg['params'])
    print(f'  Group {i}: lr={lr:.6f}, params={n}')

# 模拟一步训练
x = torch.randn(2, 4, 64, 64, 64).cuda()
target = torch.randn(2, 4, 64, 64, 64).cuda()

trainer.optimizer.zero_grad()
out = model(x)
loss = ((out - target) ** 2).mean()
loss.backward()
trainer.optimizer.step()

print('OK 一步训练成功')
ar = float(model.moa_8.alpha_raw)
a = float(model.get_alpha())
print(f'alpha_raw: {ar:.6f}, alpha: {a:.6f}')

# 模拟 10 步训练看 alpha 移动
print()
print('=== 模拟 10 步训练（看 α 是否自由移动）===')
for step in range(10):
    x = torch.randn(2, 4, 64, 64, 64).cuda()
    target = torch.randn(2, 4, 64, 64, 64).cuda()
    trainer.optimizer.zero_grad()
    out = model(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    trainer.optimizer.step()
    if trainer.ema:
        trainer.ema.update()
    ar = float(model.moa_8.alpha_raw)
    a = float(model.get_alpha())
    print(f'Step {step+1}: alpha_raw = {ar:.6f}, alpha = {a:.6f}')