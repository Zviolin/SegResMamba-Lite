# SegResMamba-Lite - 轻量化 SegResMamba

> 受 SegMamba 启发的轻量化混合架构：CNN 骨干 + 关键层 Mamba，用 1/55 的参数量达到与 SegMamba 相当的性能

---

## 一、设计动机

SegMamba 论文 (arXiv:2401.13560, MICCAI 2024) 首次将 Mamba 用于 3D 医学图像分割，取得了优异的性能，但存在以下问题：

1. **参数量过大**: 74.87M，消费级 GPU 难以训练
2. **全栈 Mamba**: 每个 Stage 都用 Mamba，计算开销高
3. **三方向 Mamba (ToM)**: 三个方向的 Mamba 进一步增加计算量
4. **训练成本高**: 需要 1000 epochs，SGD 优化器，训练时间长

**SegResMamba-Lite 的设计目标**: 在 <1.5M 参数量限制下，通过混合架构设计，达到与 SegMamba 相当的性能。

---

## 二、与官方 SegMamba 的核心差异

### 2.1 架构理念

| 对比项 | 官方 SegMamba | SegResMamba-Lite |
|--------|--------------|------------------|
| **设计理念** | "全栈 Mamba" - 所有层用 Mamba | "混合架构" - CNN 骨干 + 关键层 Mamba |
| **参数量** | ~74.87M | 1.35M-1.56M |
| **轻量化程度** | 重型 | **55x 更轻** |
| **Mamba 覆盖** | 每个 Stage 都有 Mamba | 仅在关键尺度 (32³, 16³, 8³) |
| **特征维度** | [48, 96, 192, 384, 768] | [20, 40, 80] |

### 2.2 Mamba 实现差异

| 对比项 | 官方 SegMamba (ToM) | SegResMamba-Lite (BiMamba) |
|--------|---------------------|---------------------------|
| **方向数** | 三方向 (正向/反向/层间) | 双向 (正向/反向) |
| **实现方式** | 3 个独立 Mamba 模块 | 1 个 Mamba + 空间翻转 |
| **计算量** | 3x | 2x |
| **性能** | 更强 | 接近 (实验验证) |

**官方 ToM (Tri-orientated Mamba):**
```python
# 三个方向的 Mamba
z_forward = Mamba(z)           # 正向
z_reverse = Mamba(z[::-1])     # 反向
z_slice = Mamba(z_transposed)  # 层间方向
output = z_forward + z_reverse + z_slice
```

**SegResMamba-Lite BiMamba (双向 Mamba):**
```python
class BiMambaLayer(nn.Module):
    def forward(self, x):
        # 正向 Mamba
        x_forward = self.mamba(x)
        # 反向 Mamba：翻转空间维度
        x_rev = torch.flip(x, dims=[2, 3, 4])
        x_backward_rev = self.mamba(x_rev)
        x_backward = torch.flip(x_backward_rev, dims=[2, 3, 4])
        # 双向融合
        return x_forward + x_backward
```

**简化理由**:
- 三方向计算量大 (3x)，双向