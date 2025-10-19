"""
基于 PyTorch 的散射神经网络模型。

该模块实现了一个简单的光学前向模型：
- 输入图像张量并将其转换为幅度场
- 乘以一个可学习的逐像素相位项
- 展平后与一个固定的、随机生成的复数传输矩阵相乘
- 返回复数输出（或可选返回强度）

示例
-------
```python
import torch
from scatterNeuralNetwork import ScatterNeuralNetwork

H_in, W_in = 128, 128
H_out, W_out = 32, 32
model = ScatterNeuralNetwork(input_hw=(H_in, W_in), output_hw=(H_out, W_out), return_intensity=False)

x = torch.rand(4, 3, H_in, W_in)  # 一批 RGB 图像，数值范围 [0, 1]
y = model(x)  # 复数张量，形状为 [4, H_out, W_out]
```
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


class ScatterNeuralNetwork(nn.Module):
    """具有可学习相位和固定复数传输矩阵的散射神经网络。

    参数
    ----------
    input_hw:
        期望输入空间尺寸的二元组 (H_in, W_in)。
    output_hw:
        期望输出空间尺寸的二元组 (H_out, W_out)，前向输出为 [B, H_out, W_out]。
    normalize_input:
        若为 True，按样本在空间维度上做 min-max，将幅度归一化到 [0, 1]。
    sqrt_amplitude:
        若为 True，将输入视为强度并通过开方转换为幅度；
        若为 False，直接使用（经夹紧）的数值作为幅度。
    return_intensity:
        若为 True，返回 |y|^2（实数）；若为 False，返回复数 y。
    phase_init:
        相位的初始化策略，取值为 {"zeros", "uniform"}。当为 "uniform" 时，
        在区间 [0, 2π) 上均匀初始化。
    tmatrix_scale:
        在按 sqrt(in_dim) 归一化之前，用于实部/虚部正态初始化的标准差缩放因子。
    seed:
        可选的随机数种子，用于确定性地初始化传输矩阵。
    device, dtype:
        可选的 torch 设备与 dtype，应用于参数/缓冲区。
    """

    def __init__(
        self,
        input_hw: Tuple[int, int],
        output_hw: Tuple[int, int],
        *,
        normalize_input: bool = True,
        sqrt_amplitude: bool = True,
        return_intensity: bool = False,
        phase_init: str = "uniform",
        tmatrix_scale: float = 1.0,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        if not isinstance(input_hw, Sequence) or len(input_hw) != 2:
            raise ValueError("input_hw must be a (H_in, W_in) tuple")
        if not isinstance(output_hw, Sequence) or len(output_hw) != 2:
            raise ValueError("output_hw must be a (H_out, W_out) tuple")

        self.height, self.width = int(input_hw[0]), int(input_hw[1])
        self.output_height, self.output_width = int(output_hw[0]), int(output_hw[1])
        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError("output_hw must be positive")
        self.output_dim = self.output_height * self.output_width
        self.normalize_input = bool(normalize_input)
        self.sqrt_amplitude = bool(sqrt_amplitude)
        self.return_intensity = bool(return_intensity)
        self.eps: float = 1e-8

        in_dim = self.height * self.width

        # 可学习的逐像素相位（在 batch 维度上广播），单位：弧度
        # 使用 float32 存储相位，避免 torch.polar 在半精度上的算子缺失
        phase = torch.empty(1, 1, self.height, self.width, device=device, dtype=torch.float32)
        if phase_init == "zeros":
            nn.init.zeros_(phase)
        elif phase_init == "uniform":
            nn.init.uniform_(phase, a=0.0, b=2.0 * math.pi)
        else:
            raise ValueError("phase_init must be one of {'zeros', 'uniform'}")
        self.phase = nn.Parameter(phase)  # shape [1, 1, H, W]

        # 作为 buffer 注册的固定随机复数传输矩阵
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))
        else:
            gen = None

        # 用半精度生成实部/虚部，从而构成 complex32 的传输矩阵
        real = torch.randn(self.output_dim, in_dim, generator=gen, device=device, dtype=torch.float32)
        imag = torch.randn(self.output_dim, in_dim, generator=gen, device=device, dtype=torch.float32)

        # 用 in_dim 归一化方差以保持数值尺度合理
        scale = float(tmatrix_scale) / math.sqrt(in_dim)
        real = real * scale
        imag = imag * scale

        # 固定使用 complex64（cfloat）存储传输矩阵
        transmission_matrix = torch.complex(real, imag).to(torch.complex64)
        self.register_buffer("transmission_matrix", transmission_matrix, persistent=True)

    @torch.no_grad()
    def reset_transmission_matrix(self, *, seed: Optional[int] = None, tmatrix_scale: float = 1.0) -> None:
        """使用新的随机种子重新初始化固定的传输矩阵。

        这不会影响梯度，因为该矩阵是不可训练的 buffer。
        """
        in_dim = self.height * self.width
        device = self.transmission_matrix.device
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))
        else:
            gen = None
        # 重置时保持与当前缓冲区一致（此处为 complex64 → 实/虚为 float32）
        real = torch.randn(self.output_dim, in_dim, generator=gen, device=device, dtype=torch.float32)
        imag = torch.randn(self.output_dim, in_dim, generator=gen, device=device, dtype=torch.float32)
        scale = float(tmatrix_scale) / math.sqrt(in_dim)
        real = real * scale
        imag = imag * scale
        tm = torch.complex(real, imag).to(self.transmission_matrix.dtype)
        self.transmission_matrix.copy_(tm)

    

    @torch.no_grad()
    def set_transmission_matrix(self, tm: torch.Tensor) -> None:
        """从外部张量设置固定的复数传输矩阵。

        需要一个形状为 [H_out*W_out, H_in*W_in] 的复数张量。提供的张量将被移动到
        本模块的设备/数据类型，并拷贝到已注册的缓冲区 `transmission_matrix` 中。
        """
        if tm.ndim != 2:
            raise ValueError("Transmission matrix must be 2D [output_dim, in_dim]")

        in_dim = self.height * self.width
        expected_shape = (self.output_dim, in_dim)
        if tuple(tm.shape) != expected_shape:
            raise ValueError(
                f"Expected transmission matrix shape {expected_shape} but got {tuple(tm.shape)}"
            )

        if not torch.is_complex(tm):
            raise TypeError("Transmission matrix must be a complex tensor (e.g., complex64)")

        tm = tm.to(device=self.transmission_matrix.device, dtype=self.transmission_matrix.dtype)
        self.transmission_matrix.copy_(tm)

    
    
    def _image_to_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """将输入图像转换为大致位于 [0, 1] 的幅度张量。
        - 若为 3 通道，使用标准 RGB 权重转换为亮度；
        - 若为其他通道数，在通道维上取平均；
        - 可选：按样本进行 min-max 归一化；
        - 可选：对强度开方以转换为幅度。
        """
        if x.ndim != 4:
            raise ValueError("Input must be a 4D tensor [B, C, H, W]")
        if x.shape[-2] != self.height or x.shape[-1] != self.width:
            raise ValueError(f"Input spatial size must be ({self.height}, {self.width}) but got ({x.shape[-2]}, {x.shape[-1]})")

        x = x.float()
        b, c, _, _ = x.shape

        if c == 3:
            # 亮度权重
            weights = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
            amp = (x * weights).sum(dim=1, keepdim=True)
        elif c == 1:
            amp = x
        else:
            amp = x.mean(dim=1, keepdim=True)

        if self.normalize_input:
            min_val = amp.amin(dim=(2, 3), keepdim=True)
            max_val = amp.amax(dim=(2, 3), keepdim=True)
            amp = (amp - min_val) / (max_val - min_val + self.eps)

        amp = torch.clamp(amp, min=0.0)
        if self.sqrt_amplitude:
            amp = torch.sqrt(amp + self.eps)

        return amp  # 形状 [B, 1, H, W]，实数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数
        ----------
        x:
            形状为 [B, C, H, W] 的输入图像张量。H 与 W 必须与 `input_hw` 一致。

        返回
        -------
        torch.Tensor
            - 当 `return_intensity` 为 False：形状为 [B, H_out, W_out] 的复数张量；
            - 当 `return_intensity` 为 True：形状为 [B, H_out, W_out] 的实数张量（|y|^2）。
        """
        B, C, H, W = x.shape
        amplitude = self._image_to_amplitude(x)  # [B, 1, H, W]

        # 构建复数场：amplitude * exp(i * phase)
        # polar(abs, angle) -> complex
        # torch.polar 对 half CUDA 未实现，这里在极坐标构造处用 float32，然后再按需降精度
        field_complex = torch.polar(amplitude.float(), self.phase)  # [B, 1, H, W], complex64
        # 若传输矩阵为 complex32，则将场降到 complex32 以节省显存
        if self.transmission_matrix.dtype == getattr(torch, "complex32", torch.complex64):
            field_complex = field_complex.to(getattr(torch, "complex32", torch.complex64))

        # 展平空间维度
        b = field_complex.shape[0]
        field_vec = field_complex.reshape(b, -1)  # [B, H*W], complex

        # 避免 ComplexHalf 矩阵乘法：分离实部/虚部，用实数 matmul 组合
        tm = self.transmission_matrix
        tm_real = tm.real
        tm_imag = tm.imag
        fv_real = field_vec.real.to(tm_real.dtype)
        fv_imag = field_vec.imag.to(tm_real.dtype)
        y_real = fv_real @ tm_real.transpose(0, 1) - fv_imag @ tm_imag.transpose(0, 1)
        y_imag = fv_real @ tm_imag.transpose(0, 1) + fv_imag @ tm_real.transpose(0, 1)
        y = torch.complex(y_real, y_imag).to(tm.dtype)

        if self.return_intensity:
            # 强度：|y|^2 = real^2 + imag^2，然后重塑为二维输出
            y_real = y.real.pow(2) + y.imag.pow(2)
            return y_real.view(B, self.output_height, self.output_width)
        return y.view(B, self.output_height, self.output_width)


__all__ = ["ScatterNeuralNetwork"]


