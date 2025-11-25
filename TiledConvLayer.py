import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np

class TiledConvLayer(nn.Module):
    def __init__(self, tiling_factor, tile_size, kernel_size, nonneg=False):
        """
        初始化拼接卷积层
        Args:
            tiling_factor (int): 网格的行/列数 (例如 3 表示 3x3 的阵列)
            tile_size (int): 每个拼接块的目标尺寸 (对应原代码的 tile_size)
            kernel_size (int): 可训练核的实际尺寸
            nonneg (bool): 是否强制 PSF 为非负 (用于非相干光模拟)
        """
        super(TiledConvLayer, self).__init__()
        
        self.tiling_factor = tiling_factor
        self.tile_size = tile_size
        self.kernel_size = kernel_size
        self.nonneg = nonneg
        
        # 1. 初始化可训练的 Kernels
        # 原代码是列表的列表，这里使用一个 4D Tensor 管理更高效
        # Shape: [Grid_H, Grid_W, Kernel_H, Kernel_W]
        self.kernels = nn.Parameter(
            torch.randn(tiling_factor, tiling_factor, kernel_size, kernel_size)
        )
        
        # 使用 Xavier Uniform 初始化 (对应 tf.contrib.layers.xavier_initializer)
        nn.init.xavier_uniform_(self.kernels)
        # self.kernels.data = self._generate_oriented_kernels(2)

    def get_psf(self):
        """构建拼接后的大 PSF (对应原代码中的 stitching 逻辑)"""
        
        # A. 约束处理 (Non-negative constraint)
        kernels = self.kernels
        if self.nonneg:
            kernels = torch.abs(kernels)
            
        # B. 计算填充 (Padding)
        # TF: pad_one (ceil), pad_two (floor) -> PyTorch pad 顺序是 (左, 右, 上, 下)
        diff = self.tile_size - self.kernel_size
        pad_left = diff // 2           # np.floor behavior
        pad_right = diff - pad_left    # np.ceil behavior
        # 注意：原 TF 代码 pad_one 是 ceil (大), pad_two 是 floor (小)
        # 只要左右加起来等于 diff 即可。
        
        # 对每个小核进行填充
        # F.pad 的输入如果是 4D，padding tuple 对最后两维生效
        kernels_pad = F.pad(kernels, (pad_left, pad_right, pad_left, pad_right))
        
        # C. 拼接 (Stitching)
        # kernels_pad shape: [TF, TF, Tile, Tile]
        # 我们需要将其变为 [TF*Tile, TF*Tile]
        
        # 方法：先按行拼接 (dim=3 也就是宽), 再按列拼接 (dim=2 也就是高)
        # 这里的逻辑等同于 TF 的嵌套 concat
        rows = [torch.cat([kernels_pad[i, j] for j in range(self.tiling_factor)], dim=1) 
                for i in range(self.tiling_factor)]
        psf = torch.cat(rows, dim=0)
        
        return psf

    def fft_conv2d(self, img, psf):
        """
        基于 FFT 的二维卷积实现 (支持复数输入)
        img shape: [B, C, H, W] (可以是复数)
        psf shape: [H_psf, W_psf] (通常是实数或复数)
        """
        b, c, h, w = img.shape
        h_psf, w_psf = psf.shape
        
        # 1. 确定卷积后的尺寸
        fft_h = h + h_psf - 1
        fft_w = w + w_psf - 1
        
        # 2. 频域转换
        # --- 修改点 1: 使用 fft2 而不是 rfft2 ---
        # rfft2 只能处理实数，fft2 可以处理复数或实数
        img_fft = torch.fft.fft2(img, s=(fft_h, fft_w))
        
        # 对 PSF 进行 FFT
        psf_tensor = psf.view(1, 1, h_psf, w_psf) 
        
        # --- 修改点 2: PSF 也使用 fft2 ---
        # 即使 psf 是实数参数，为了和 img_fft 进行复数乘法，也统一转为复数谱
        psf_fft = torch.fft.fft2(psf_tensor, s=(fft_h, fft_w))
        
        # 3. 频域相乘 (复数乘法)
        out_fft = img_fft * psf_fft
        
        # 4. 逆变换
        # --- 修改点 3: 使用 ifft2 而不是 irfft2 ---
        # 结果仍然是复数
        out = torch.fft.ifft2(out_fft, s=(fft_h, fft_w))
        
        # 5. 裁剪结果 (Crop to 'same' size)
        start_h = (h_psf - 1) // 2
        start_w = (w_psf - 1) // 2
        
        output = out[:, :, start_h : start_h + h, start_w : start_w + w]
        
        # 注意：此时 output 是复数 (Complex)。
        # 如果后续层需要实数强度 (Intensity)，请在外部做 abs(output)**2
        
        return output

    def shrink_and_pad(self, img_batch, target_size=150):
        """
        将输入图片缩小到 target_size，然后居中填充回原始尺寸。
        (支持 ComplexFloat 复数输入)
        """
        b, c, h, w = img_batch.shape
        
        # --- 修改点 1: 处理复数插值 ---
        if img_batch.is_complex():
            # 复数不能直接 interpolate，需拆分实虚部
            real_small = F.interpolate(img_batch.real, size=(target_size, target_size), mode='area')
            imag_small = F.interpolate(img_batch.imag, size=(target_size, target_size), mode='area')
            small_img = torch.complex(real_small, imag_small)
        else:
            # 如果是实数，直接缩小
            small_img = F.interpolate(img_batch, size=(target_size, target_size), mode='area')
        
        # 2. 创建全黑背景 (Canvas)
        # zeros_like 会自动处理复数类型，生成 0+0j
        padded_img = torch.zeros_like(img_batch)
        
        # 3. 计算中心位置
        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2
        end_h = start_h + target_size
        end_w = start_w + target_size
        
        # 4. 将小图粘贴到中心
        padded_img[:, :, start_h:end_h, start_w:end_w] = small_img
        
        return padded_img

    def forward(self, x):
        """
        Args:
            x: 输入图像 [Batch, Channel, Height, Width]
        """
        # 1. 生成巨大的 PSF
        psf = self.get_psf()
        
        # 2. (可选) 模拟 TF 中的 padding 逻辑
        # 原代码计算了 input_img_pad 但没用它。
        # 这里如果不加 pad，FFT 卷积会自动处理边界，通常比手动 pad 更好。
        
        # 3. 执行卷积
        x = self.shrink_and_pad(x, target_size=self.tile_size)

        output_img = self.fft_conv2d(x, psf)
        
        return output_img

    def get_regularization_loss(self):
        """
        对应原代码: tf.contrib.layers.apply_regularization
        在训练 loop 中调用此函数并加到 total loss
        """
        psf = self.get_psf()
        # 默认 L2 正则
        return torch.sum(psf ** 2)


    def _generate_oriented_kernels(self, width):
        """
        内部辅助方法：生成一组旋转角度不同的线段核
        """
        tf = self.tiling_factor
        ks = self.kernel_size
        
        # 准备临时 Tensor
        kernels = torch.zeros(tf, tf, ks, ks)
        
        # 创建坐标网格
        x = np.linspace(-ks//2, ks//2, ks)
        X, Y = np.meshgrid(x, x)
        
        total_kernels = tf * tf
        count = 0
        
        for i in range(tf):
            for j in range(tf):
                # 计算角度 (均匀分布在 0 到 180 度之间)
                angle_deg = (count / total_kernels) * 180
                theta = np.radians(angle_deg)
                
                # 坐标旋转计算距离
                dist_from_line = -X * np.sin(theta) + Y * np.cos(theta)
                
                # 生成高斯线条
                line_img = np.exp(-(dist_from_line**2) / (2 * width**2))
                
                # 填入
                kernels[i, j] = torch.from_numpy(line_img).float()
                count += 1
                
        return kernels