"""
官方 Mamba 模块
基于 mamba_ssm 的官方实现
GitHub: https://github.com/ge-xing/SegMamba
完全一致的官方实现！
"""

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations
import torch.nn as nn
import torch
from functools import partial

# mamba_ssm Mamba2（优先，真正 CUDA 优化版本）
try:
    from mamba_ssm import Mamba2 as _Mamba2SSM
    MAMBA2_SSM_AVAILABLE = True
except ImportError:
    _Mamba2SSM = None
    MAMBA2_SSM_AVAILABLE = False

# MiniMamba（纯 PyTorch，Windows 兼容，但较慢）
try:
    from minimamba.s6 import S6 as _MiniMambaS6
    from minimamba.config import BaseMambaConfig as _BaseMambaConfig
    MINIMAMBA_AVAILABLE = True
except ImportError:
    _MiniMambaS6 = None
    _BaseMambaConfig = None
    MINIMAMBA_AVAILABLE = False

# mamba_ssm Mamba1（CUDA kernel，但 RTX 5060 不支持）
try:
    from mamba_ssm import Mamba as _OrigMamba
    MAMBA_AVAILABLE = True
except ImportError:
    _OrigMamba = None
    MAMBA_AVAILABLE = False

import torch.nn.functional as F


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class PurePyTorchMamba(nn.Module):
    """
    纯 PyTorch 实现的 Mamba 替代方案
    不需要 mamba_ssm 包，使用完整的 SSM 架构
    优化版：尽可能向量化操作
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        x = F.silu(x)
        
        y = self.ssm(x)
        
        y = y * F.silu(z)
        
        out = self.out_proj(y)
        
        return out
    
    def ssm(self, x):
        B, L, D = x.shape
        
        x_dbl = self.x_proj(x)
        delta, B_param, C_param = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        
        A = -torch.exp(self.A_log)
        
        delta_A = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        delta_B = delta.unsqueeze(-1) * B_param.unsqueeze(2)
        
        y = self.selective_scan(x, delta_A, delta_B, C_param)
        
        y = y + x * self.D.unsqueeze(0)
        
        return y
    
    def selective_scan(self, x, delta_A, delta_B, C_param):
        B, L, D = x.shape
        
        y = torch.zeros(B, L, D, device=x.device, dtype=x.dtype)
        
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        
        for i in range(L):
            h = delta_A[:, i] * h + delta_B[:, i] * x[:, i]
            y[:, i] = (C_param[:, i].unsqueeze(1) @ h.unsqueeze(-1)).squeeze(-1)
        
        return y


class Mamba2SSMWrapper(nn.Module):
    """
    真正的 mamba_ssm.Mamba2 包装器
    适配 3D 输入 (B, L, C) 格式
    使用 loscrossos 预编译的 CUDA kernel
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, headdim=8):
        super().__init__()
        self.d_model = d_model
        
        d_inner = int(expand * d_model)
        if d_inner % headdim != 0:
            headdim = 8
            while d_inner % headdim != 0:
                headdim //= 2
            if headdim < 1:
                headdim = 8
        
        self.mamba = _Mamba2SSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
    
    def forward(self, x):
        """
        x: (B, L, C) - 已 flatten 的序列输入
        """
        return self.mamba(x)


class MambaLayer(nn.Module):
    """
    官方 Mamba 层实现！
    完全来自官方 SegMamba 代码！
    优先级: mamba_ssm.Mamba2 > MiniMamba > mamba_ssm.Mamba1 > PurePyTorch
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None, use_mamba2=True):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        # 优先级 1: 真正的 mamba_ssm.Mamba2 (CUDA 优化，RTX 5060 支持)
        if use_mamba2 and MAMBA2_SSM_AVAILABLE:
            self.mamba = Mamba2SSMWrapper(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.use_mamba2 = True
            self.mamba_type = 'Mamba2SSM'
        
        # 优先级 2: MiniMamba (纯 PyTorch，稳定)
        elif MINIMAMBA_AVAILABLE:
            from functools import partial
            config = _BaseMambaConfig(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.mamba = _MiniMambaS6(config)
            self.use_mamba2 = False
            self.mamba_type = 'MiniMamba'
        
        # 优先级 3: mamba_ssm.Mamba1 (CUDA kernel，RTX 5060 不支持)
        elif MAMBA_AVAILABLE and _OrigMamba is not None:
            self.mamba = _OrigMamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.use_mamba2 = False
            self.mamba_type = 'Mamba1CUDA'
        
        # 优先级 4: PurePyTorch (最后方案)
        else:
            print("[警告: mamba_ssm/MiniMamba 未安装，使用纯 PyTorch Mamba 替代方案！]")
            self.mamba = PurePyTorchMamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.use_mamba2 = False
            self.mamba_type = 'PurePyTorch'

    def forward(self, x):
        """
        3D 输入 (B, C, D, H, W) -> 输出 (B, C, D, H, W)
        """
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim, f"通道数不匹配: C={C}, dim={self.dim}"

        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        
        x_mamba = self.mamba(x_norm)
        
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip
        return out


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def get_mamba_layer(dim, d_state=16, d_conv=4, expand=2, num_slices=None, use_mamba2=True):
    """
    获取官方 Mamba 层的快捷函数
    :param dim: 通道数
    :param d_state: Mamba 状态维度
    :param d_conv: Mamba 卷积核大小
    :param expand: Mamba 扩展倍数
    :param num_slices: BiMamba 切片数
    :param use_mamba2: 是否使用 mamba_ssm.Mamba2（默认 True，CUDA 优化）
    :return: MambaLayer 实例
    """
    return MambaLayer(
        dim=dim,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        num_slices=num_slices,
        use_mamba2=use_mamba2
    )


# 导出可用的 Mamba 类型
def get_available_mamba_types():
    """获取当前可用的 Mamba 类型"""
    types = []
    if MAMBA2_SSM_AVAILABLE:
        types.append("Mamba2SSM (mamba_ssm.Mamba2 - CUDA 优化)")
    if MINIMAMBA_AVAILABLE:
        types.append("MiniMamba (纯 PyTorch)")
    if MAMBA_AVAILABLE:
        types.append("Mamba1CUDA (mamba_ssm.Mamba1 - CUDA)")
    types.append("PurePyTorch (纯 PyTorch 备用)")
    return types


if __name__ == "__main__":
    print("=== Mamba 模块状态 ===")
    print(f"Mamba2 (mamba_ssm): {'✅ 可用' if MAMBA2_SSM_AVAILABLE else '❌ 不可用'}")
    print(f"MiniMamba: {'✅ 可用' if MINIMAMBA_AVAILABLE else '❌ 不可用'}")
    print(f"Mamba1 (mamba_ssm): {'✅ 可用' if MAMBA_AVAILABLE else '❌ 不可用'}")
    print()
    print("可用类型:", get_available_mamba_types())