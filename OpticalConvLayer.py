import torch
import torch.nn as nn
import torch.fft
import numpy as np
import torch.nn.functional as F


# === 辅助函数 ===
def fftshift2d(x):
    for dim in (-2, -1):
        n = x.size(dim)
        x = torch.roll(x, shifts=n // 2, dims=dim)
    return x

def ifftshift2d(x):
    for dim in (-2, -1):
        n = x.size(dim)
        x = torch.roll(x, shifts=-(n // 2), dims=dim)
    return x

def complex_exp(x):
    return torch.cos(x) + 1j * torch.sin(x)

def psf_from_atf(atf):
    """Compute PSF from ATF via inverse FFT."""
    psfc = fftshift2d(torch.fft.ifft2(ifftshift2d(atf), dim=(-2, -1)))
    psf = psfc.abs() ** 2
    psf = psf / psf.sum()
    return psf

def fft_conv2d(img, psf):
    """Frequency domain convolution with circular boundary conditions."""
    B, C, H, W = img.shape
    psf_shape = psf.shape
    img_fft = torch.fft.fft2(fftshift2d(img), dim=(-2, -1))
    psf_fft = torch.fft.fft2(ifftshift2d(psf), s=(H, W), dim=(-2, -1))
    out = torch.fft.ifft2(img_fft * psf_fft, dim=(-2, -1))
    out = ifftshift2d(out).real
    return out


# === Optical Convolution Layer ===
class OpticalConvLayer(nn.Module):
    def __init__(self,
                 hm_reg_scale=1e-4,
                 r_NA=1.0,
                 n=1.48,
                 wavelength=532e-9,
                 activation=None,
                 coherent=False,
                 amplitude_mask=False,
                 zernike=False,
                 fourier=True,
                 binarymask=False,
                 n_modes=1024,
                 freq_range=0.5,
                 zernike_file=None,
                 binary_mask_np=None,
                 name="optical_conv"):
        super().__init__()

        self.name = name
        self.hm_reg_scale = hm_reg_scale
        self.r_NA = r_NA
        self.n = n
        self.wavelength = wavelength
        self.activation = activation
        self.coherent = coherent
        self.amplitude_mask = amplitude_mask
        self.zernike = zernike
        self.fourier = fourier
        self.binarymask = binarymask
        self.freq_range = freq_range
        self.binary_mask_np = binary_mask_np
        self.zernike_file = zernike_file
        self.n_modes = n_modes

        # mask 参数占位符
        self.mask = None

    def build_mask(self, shape):
        """生成可训练的光学相位/振幅掩膜。"""
        H, W = shape[-2], shape[-1]
        if self.fourier:
            # 傅里叶系数相位掩膜
            mask = self.build_fourier_mask(shape)
        elif self.amplitude_mask:
            # 幅度掩膜（0~1区间）
            mask = nn.Parameter(torch.rand(1, 1, H, W) * 1e-4)
        else:
            # 相位掩膜（零中心）
            mask = nn.Parameter(torch.rand(1, 1, H, W) * 1e-4)
        return mask

    def build_fourier_mask(self, shape):
        """生成傅里叶系数相位掩膜。"""
        H, W = shape[-2], shape[-1]
        
        # 计算傅里叶系数尺寸
        freq_height = int(H * self.freq_range)
        freq_width = int(W * self.freq_range)
        
        # 创建傅里叶系数（实部和虚部）
        fourier_real = nn.Parameter(torch.zeros(1, 1, freq_height, freq_width) * 1e-4)
        fourier_imag = nn.Parameter(torch.zeros(1, 1, freq_height, freq_width) * 1e-4)
        
        # 注册参数
        self.register_parameter('fourier_real', fourier_real)
        self.register_parameter('fourier_imag', fourier_imag)
        
        # 构建复数傅里叶系数
        fourier_coeffs = torch.complex(fourier_real, fourier_imag)
        
        # 零填充到原始尺寸
        padding_height_1 = (H - freq_height) // 2
        padding_height_2 = H - freq_height - padding_height_1
        padding_width_1 = (W - freq_width) // 2
        padding_width_2 = W - freq_width - padding_width_1
        
        fourier_padded = F.pad(fourier_coeffs, 
                              [padding_width_1, padding_width_2, 
                               padding_height_1, padding_height_2])
        
        # 逆傅里叶变换得到高度图
        fourier_shifted = ifftshift2d(fourier_padded)
        height_map = torch.fft.ifft2(fourier_shifted, dim=(-2, -1))
        height_map = torch.real(height_map)
        
        # 转换为相位掩膜
        phase_mask = 2 * np.pi / self.wavelength * height_map
        
        return phase_mask

    def forward(self, input_field):
        """
        input_field: (B, 1, H, W), 实数或复数张量
        """
        B, C, H, W = input_field.shape

        if self.mask is None:
            self.mask = self.build_mask(input_field.shape).to(input_field.device)

        # === 计算传递函数 (ATF) ===
        atf = torch.ones((1, 1, H, W), dtype=torch.complex64, device=input_field.device)
        phase_term = 2 * np.pi / self.wavelength * self.mask
        atf = complex_exp(phase_term)

        # 附加二值掩膜
        if self.binarymask and self.binary_mask_np is not None:
            binary_mask = torch.tensor(self.binary_mask_np, dtype=torch.float32, device=input_field.device)
            binary_mask = binary_mask.unsqueeze(0).unsqueeze(0)
            atf = atf * binary_mask

        # === 得到 PSF ===
        psf = psf_from_atf(atf)
        psf = psf.unsqueeze(1)  # (1,1,H,W)
        psf = psf.to(torch.float32)

        # === 执行卷积 ===
        if self.coherent:
            # 复数卷积
            field = torch.fft.fft2(fftshift2d(input_field.to(torch.complex64)), dim=(-2, -1))
            out_field = torch.fft.ifft2(field * atf, dim=(-2, -1))
            output = out_field.abs() ** 2
        else:
            # 非相干卷积：I_out = I_in * PSF
            output = fft_conv2d(input_field, psf)

        # === 激活函数 ===
        if self.activation is not None:
            output = self.activation(output)

        return output
