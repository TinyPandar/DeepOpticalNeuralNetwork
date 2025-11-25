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
import torch.distributed as dist
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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from OGN.modules import FreeSpaceProp
from OpticalConvLayer import OpticalConvLayer   
from TiledConvLayer import TiledConvLayer


class _TPAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_local: torch.Tensor, rows_per_rank: int, output_dim: int, world_size: int) -> torch.Tensor:
        ctx.rows_per_rank = int(rows_per_rank)
        ctx.output_dim = int(output_dim)
        ctx.world_size = int(world_size)
        # Pad local columns to rows_per_rank for even concat
        B = y_local.shape[0]
        local_cols = y_local.shape[1]
        ctx.local_cols = int(local_cols)
        pad_cols = rows_per_rank - local_cols
        if pad_cols > 0:
            pad = torch.zeros(B, pad_cols, dtype=y_local.dtype, device=y_local.device)
            y_pad = torch.cat([y_local, pad], dim=1)
        else:
            y_pad = y_local
        # Gather real/imag separately for safety
        y_r = y_pad.real.contiguous()
        y_i = y_pad.imag.contiguous()
        gather_r = [torch.empty_like(y_r) for _ in range(world_size)]
        gather_i = [torch.empty_like(y_i) for _ in range(world_size)]
        dist.all_gather(gather_r, y_r)
        dist.all_gather(gather_i, y_i)
        y_full_r = torch.cat(gather_r, dim=1)[:, :output_dim]
        y_full_i = torch.cat(gather_i, dim=1)[:, :output_dim]
        return torch.complex(y_full_r, y_full_i)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not dist.is_available() or not dist.is_initialized() or ctx.world_size == 1:
            return grad_output, None, None, None
        rank = dist.get_rank()
        start = rank * ctx.rows_per_rank
        end = min(start + ctx.rows_per_rank, ctx.output_dim)
        grad_local = grad_output[:, start:end]
        return grad_local, None, None, None


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
    tmatrix_compute_dtype:
        传输计算时使用的实数精度（用于分解为实/虚两路 matmul 的路径）。
        可为 None/torch.float16/torch.bfloat16；为 None 时使用复数 matmul（complex64）。
    seed:
        可选的随机数种子，用于确定性地初始化传输矩阵。
    device, dtype:
        可选的 torch 设备与 dtype，应用于参数/缓冲区。
    activation:
        激活函数类型，可选值：{"abs", "relu", "leaky_relu", "sigmoid", "tanh", "elu", "softplus"}。
        默认为 "abs"（绝对值函数）。
    activation_params:
        激活函数的参数字典，例如 {"negative_slope": 0.01} 用于 LeakyReLU。
    normalize_negative:
        若为 True，对激活函数输出的负数部分进行正则化处理。
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
        tmatrix_compute_dtype: Optional[torch.dtype] = None,
        num_layers: int = 1,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        # Tensor-parallel options
        tp_enabled: bool = False,
        tp_rank: int = 0,
        tp_world_size: int = 1,
        # 激活函数选项
        activation: str = "none",
        activation_params: Optional[dict] = None,
        normalize_negative: bool = False,
        
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
        # 前向传播重复层数（相位调制+传输+幅度更新的次数）
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        self.num_layers: int = int(num_layers)

        # 激活函数配置
        valid_activations = {"abs", "relu", "leaky_relu", "sigmoid", "tanh", "elu", "softplus", "none"}
        if activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}, but got '{activation}'")
        self.activation: str = activation
        self.activation_params: dict = activation_params or {}
        self.normalize_negative: bool = bool(normalize_negative)

        in_dim = self.height * self.width

        # 设置传输矩阵的计算精度（仅影响前向中的 matmul 计算，不改变存储精度）
        if tmatrix_compute_dtype is not None and tmatrix_compute_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("tmatrix_compute_dtype must be one of {None, torch.float16, torch.bfloat16}")
        self.tmatrix_compute_dtype: Optional[torch.dtype] = tmatrix_compute_dtype

        # Tensor parallel config
        self.tp_enabled: bool = bool(tp_enabled and (tp_world_size or 1) > 1)
        self.tp_rank: int = int(tp_rank)
        self.tp_world_size: int = int(tp_world_size) if int(tp_world_size) > 0 else 1
        # 每 rank 均分行数（最后一块可能更短）；用于 all_gather 时对齐 padding
        self.rows_per_rank: int = (self.output_dim + self.tp_world_size - 1) // max(self.tp_world_size, 1)
        if self.tp_enabled:
            start = self.tp_rank * self.rows_per_rank
            end = min(start + self.rows_per_rank, self.output_dim)
        else:
            start, end = 0, self.output_dim
        self._row_start: int = start
        self._row_end: int = end

        # 可学习的逐像素相位（在 batch 维度上广播），单位：弧度
        # 为每一层分配独立的相位参数；使用 float32 存储以避免半精度构造复数的不支持
        phases: list = []
        for _ in range(self.num_layers):
            # 采用二维相位形状，便于与 [B, 1, H, W] 的幅度直接广播
            phase_l = torch.empty(1, 1, self.height, self.width, device=device, dtype=torch.float32)
            if phase_init == "zeros":
                nn.init.zeros_(phase_l)
            elif phase_init == "uniform":
                nn.init.uniform_(phase_l, a=0.0, b=2.0 * math.pi)
            else:
                raise ValueError("phase_init must be one of {'zeros', 'uniform'}")
            phases.append(nn.Parameter(phase_l))
        self.phases = nn.ParameterList(phases)  # 每层一个形状为 [1, 1, H, W] 的相位参数

        # 作为 buffer 注册的固定随机复数传输矩阵（或其本地 shard）
        if seed is not None:
            gen = torch.Generator(device=device)
            # 不同 rank 使用不同 seed，确保 shard 可复现
            rank_seed = int(seed) + int(self.tp_rank)
            gen.manual_seed(rank_seed)
        else:
            gen = None

        local_rows = self._row_end - self._row_start
        # 生成本地 shard 的实部/虚部（float32），随后与 in_dim 缩放以保持方差尺度，再组成 complex64 存储
        real = torch.randn(local_rows, in_dim, generator=gen, device=device, dtype=torch.float32)
        imag = torch.randn(local_rows, in_dim, generator=gen, device=device, dtype=torch.float32)

        # 用 in_dim 归一化方差以保持数值尺度合理
        scale = float(tmatrix_scale) / math.sqrt(in_dim)
        real = real * scale
        imag = imag * scale

        # 固定使用 complex64（cfloat）存储传输矩阵（计算时可按需降精度到 fp16/bf16 实数）
        transmission_matrix = torch.complex(real, imag).to(torch.complex64)
        self.register_buffer("transmission_matrix", transmission_matrix, persistent=True)

        # self.prop1 = FreeSpaceProp(wlength_vc=5.2e-7, 
        #                                 ridx_air=1.0,
        #                                 total_x_num=self.height, total_y_num=self.width,
        #                                 dx=8e-6, dy=8e-6,
        #                                 prop_z= 0.05)

        # self.prop2 = FreeSpaceProp(wlength_vc=5.2e-7, 
        #                                 ridx_air=1.0,
        #                                 total_x_num=self.output_height, total_y_num=self.output_width,
        #                                 dx=8e-6 , dy=8e-6 ,
        #                                 prop_z= 0.01)
        tiling = 2     # 3x3 阵列
        tile_sz = 400    # 拼接后每个块 64x64
        kern_sz = 5    # 实际可训练区域 50x50
        # self.conv = OpticalConvLayer(
        #                                 coherent=True,          # 使用干涉
        #                                 amplitude_mask=False,   # 相位调制元件
        #                                 r_NA=0.8, wavelength=633e-9,
        #                                 activation=torch.relu
        #                             )
        self.conv1 = TiledConvLayer(tiling_factor=tiling, tile_size=tile_sz, kernel_size=kern_sz, nonneg=True)
        self.conv2 = TiledConvLayer(tiling_factor=tiling, tile_size=tile_sz, kernel_size=kern_sz, nonneg=True)
        self.conv3 = TiledConvLayer(tiling_factor=tiling, tile_size=tile_sz, kernel_size=kern_sz, nonneg=True)
        self.conv4 = TiledConvLayer(tiling_factor=tiling, tile_size=tile_sz, kernel_size=kern_sz, nonneg=True)
        self.conv5 = TiledConvLayer(tiling_factor=tiling, tile_size=tile_sz, kernel_size=kern_sz, nonneg=True)

    @torch.no_grad()
    def reset_transmission_matrix(self, *, seed: Optional[int] = None, tmatrix_scale: float = 1.0) -> None:
        """使用新的随机种子重新初始化固定的传输矩阵。

        这不会影响梯度，因为该矩阵是不可训练的 buffer。
        """
        in_dim = self.height * self.width
        device = self.transmission_matrix.device
        if seed is not None:
            gen = torch.Generator(device=device)
            rank_seed = int(seed) + int(self.tp_rank)
            gen.manual_seed(rank_seed)
        else:
            gen = None
        # 重置当前 rank 的 shard（此处为 complex64 → 实/虚为 float32）
        local_rows = self._row_end - self._row_start
        real = torch.randn(local_rows, in_dim, generator=gen, device=device, dtype=torch.float32)
        imag = torch.randn(local_rows, in_dim, generator=gen, device=device, dtype=torch.float32)
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

    def _apply_activation(self, y: torch.Tensor) -> torch.Tensor:
        """应用选择的激活函数到复数输出 y。
        
        参数
        ----------
        y:
            复数张量，形状为 [B, 1, H, W]
            
        返回
        -------
        torch.Tensor
            实数张量，形状为 [B, 1, H, W]
        """
        # 获取幅度作为基础值
        amplitude = torch.abs(y)**2
        
        # 根据选择的激活函数进行处理
        if self.activation == "abs":
            return amplitude
            
        elif self.activation == "relu":
            return F.relu(amplitude)
            
        elif self.activation == "leaky_relu":
            negative_slope = self.activation_params.get("negative_slope", 0.01)
            return F.leaky_relu(amplitude, negative_slope=negative_slope)
            
        elif self.activation == "sigmoid":
            return torch.sigmoid(amplitude)
            
        elif self.activation == "tanh":
            return torch.tanh(amplitude)
            
        elif self.activation == "elu":
            alpha = self.activation_params.get("alpha", 1.0)
            return F.elu(amplitude, alpha=alpha)
            
        elif self.activation == "softplus":
            beta = self.activation_params.get("beta", 1.0)
            threshold = self.activation_params.get("threshold", 20.0)
            return F.softplus(amplitude, beta=beta, threshold=threshold)
        
        else:
            # 默认使用绝对值
            return amplitude

    def _normalize_negative_values(self, amplitude: torch.Tensor) -> torch.Tensor:
        """对激活函数输出的负数部分进行正则化处理。
        
        参数
        ----------
        amplitude:
            实数张量，可能包含负数
            
        返回
        -------
        torch.Tensor
            正则化后的实数张量，所有值都非负
        """
        # 找到负数部分
        negative_mask = amplitude < 0
        
        if not negative_mask.any():
            # 如果没有负数，直接返回
            return amplitude
            
        # 对负数部分进行正则化处理
        # 方法1：将负数映射到正数范围（例如使用绝对值）
        # 方法2：将负数缩放到 [0, 1] 范围
        # 这里使用方法2：将整个张量缩放到 [0, 1] 范围
        
        # 获取最小值和最大值
        min_val = amplitude.min()
        max_val = amplitude.max()
        
        # 如果所有值都是负数，特殊处理
        if max_val <= 0:
            # 将所有负数映射到 [0, 1] 范围
            normalized = (amplitude - min_val) / (max_val - min_val + self.eps)
        else:
            # 正常情况：将整个范围缩放到 [0, 1]
            normalized = (amplitude - min_val) / (max_val - min_val + self.eps)
        
        return normalized

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


        # 卷积
        # amplitude = self.conv(field)  # [B, 1, H, W]

        # 激活函数
        # amplitude = self._apply_activation(field)  # [B, 1, H, W]

        # 卷积
        field = torch.polar(amplitude.float(),torch.zeros_like(amplitude).to(amplitude.device))  # [B, 1, H, W] complex64
        field = self.conv1(field)  # [B, 1, H, W]
        amplitude = torch.abs(field)  # [B, 1, H, W]
        field = torch.polar(amplitude.float(),torch.zeros_like(amplitude).to(amplitude.device))  # [B, 1, H, W] complex64
        field = self.conv2(field)  # [B, 1, H, W]
        amplitude = torch.abs(field)  # [B, 1, H, W]
        field = torch.polar(amplitude.float(),torch.zeros_like(amplitude).to(amplitude.device))  # [B, 1, H, W] complex64
        field = self.conv3(field)  # [B, 1, H, W]
        amplitude = torch.abs(field)  # [B, 1, H, W]
        #再卷两层
        field = torch.polar(amplitude.float(),torch.zeros_like(amplitude).to(amplitude.device))  # [B, 1, H, W] complex64
        field = self.conv4(field)  # [B, 1, H, W]
        amplitude = torch.abs(field)  # [B, 1, H, W]
        field = torch.polar(amplitude.float(),torch.zeros_like(amplitude).to(amplitude.device))  # [B, 1, H, W] complex64
        field = self.conv5(field)  # [B, 1, H, W]
        amplitude = torch.abs(field)  # [B, 1, H, W]

        for l in range(self.num_layers):
            # print(f"layer {l}")
            # 构建复数场：amplitude * exp(i * phase_l)
            # torch.polar 对 half CUDA 未实现，这里在极坐标构造处用 float32，然后再按需降精度
            field = torch.polar(amplitude.float(), self.phases[l])  # [B, 1, H, W] complex64
            # 若传输矩阵为 complex32，则将场降到 complex32 以节省显存
            if self.transmission_matrix.dtype == getattr(torch, "complex32", torch.complex64):
                field = field.to(getattr(torch, "complex32", torch.complex64))


            # field = self.conv3(field)  # [B, 1, H, W]
            
            # 散射介质前自由衍射（保持四维）
            # field = self.prop1(field)  # [B, 1, H, W]

            # 展平到向量后进行传输矩阵乘法
            field_vec = field.reshape(B, -1)  # [B, H*W]


            # 复数矩阵乘法：
            # - 若未指定 tmatrix_compute_dtype，则直接用复数 matmul（complex64）
            # - 否则分解为实/虚两次实数 matmul，并在所选 dtype 下计算，以避免精度不匹配
            tm = self.transmission_matrix  # [local_rows, in_dim] 或 [output_dim, in_dim]
            if self.tmatrix_compute_dtype is None:
                y_local = field_vec @ tm.transpose(0, 1)  # [B, local_rows]
            else:
                rdtype = self.tmatrix_compute_dtype
                x_r = field_vec.real.to(rdtype)
                x_i = field_vec.imag.to(rdtype)
                tm_r = tm.real.to(rdtype)
                tm_i = tm.imag.to(rdtype)
                y_real = x_r @ tm_r.transpose(0, 1) - x_i @ tm_i.transpose(0, 1)
                y_imag = x_r @ tm_i.transpose(0, 1) + x_i @ tm_r.transpose(0, 1)
                y_local = torch.complex(y_real.to(torch.float32), y_imag.to(torch.float32))

            # 张量并行：使用自定义 autograd-safe all_gather 组装完整输出
            if self.tp_enabled and dist.is_available() and dist.is_initialized() and self.tp_world_size > 1:
                y = _TPAllGather.apply(
                    y_local,
                    self.rows_per_rank,
                    self.output_dim,
                    self.tp_world_size,
                )
            else:
                y = y_local

            # 输出面自由衍射
            y = y.reshape(B, 1, self.output_height, self.output_width)
            # y = self.prop2(y)

            # 幅度作为下一层的输入，应用选择的激活函数
            amplitude = self._apply_activation(y)
            
            # 如果启用了负数正则化，对负数部分进行处理
            if self.normalize_negative:
                amplitude = self._normalize_negative_values(amplitude)
            
            if (self.output_height, self.output_width) != (self.height, self.width):
                amplitude = F.interpolate(
                    amplitude,
                    size=(self.height, self.width),
                    mode="bilinear",
                    align_corners=False,
                )
            

        if self.return_intensity:
            # 强度：|y|^2 = real^2 + imag^2，然后重塑为二维输出
            y_real = y.real.pow(2) + y.imag.pow(2)
            return y_real.view(B, self.output_height, self.output_width)
        return y.view(B, self.output_height, self.output_width)


__all__ = ["ScatterNeuralNetwork"]


