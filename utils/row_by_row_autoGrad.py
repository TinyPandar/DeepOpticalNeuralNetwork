import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# from prVBEM_torch import pr_vbem_torch as pr_vbem
# from prVBEM import prVBEM_pytorch as pr_vbem
from torch.func import vmap
import time
import math
from typing import Optional, Union, Tuple

# --- 1. 仿真参数设置 ---
# 矩阵维度
# N_ROWS = 32*32 # 要并行处理的散射矩阵行数 输出维度
# N = 140*140     # 独立控制的数量 (信号维度)
# M = 32*32*5    # 输入场的数量 (测量次数)
N_ROWS = 100
N = 100
M = 400
BINARY=False

CUDA_STRING = 'cuda:1'

# 算法参数
ALPHA = 0.001       # 学习率 (α)
BETA = 0.8          # 动量因子 (β)
LAMBDA = 15.0       # 正则化系数 (λ)

# 仿真控制
NUM_TRIALS = 100            # 独立试验次数
MAX_ITERATIONS = 500        # 最大迭代次数
CONVERGENCE_THRESHOLD = 0.9997 # 收敛阈值
BATCH_SIZE = 100
PADDING_SHAKE = False

# FISTA 参数
FISTA_LAMBDA = 0.05
FISTA_MAX_ITERS = MAX_ITERATIONS
FISTA_TOL = 1e-6
# FISTA_DEVICE = 'cpu'

# 测试的信噪比 (SNR) 列表
# 使用 None 代表无噪声情况
SNRS = [None, 1000, 100, 10, 1]
# SNRS = [None]

# --- GPU/CPU 设备设置 ---
device = torch.device(CUDA_STRING if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print(f"Using device: {device}")

# --- 选择要运行/展示的算法 ---
ENABLE = {
    'autograd': True,
    'raf21_tm_auto': True,
    'wf_auto': True,
    'wf_reg_auto': True,
    'ggs1_auto': True,
    'ggs2_1_auto': True,
}

# --- 2. 辅助函数 (PyTorch版本) ---

def to_cpu_or_none(x):
    return x.detach().to('cpu') if (x is not None and torch.is_tensor(x)) else x

def to_numpy_or_none(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().to('cpu').numpy()
    return np.asarray(x)


def align_global_phase(est, ref):
    """
    使 est 和 ref 的总体相位对齐，便于计算误差 (PyTorch版本)。
    est: 估计的张量, 形状 (N_ROWS, N)
    ref: 参考的张量, 形状 (N_ROWS, N)
    """
    # 逐行计算相位差
    # (est * ref.conj()).sum(dim=1) 计算逐行点积
    phi = torch.angle((est * ref.conj()).sum(dim=1, keepdim=True))
    return est * torch.exp(1j * (-phi))

def generate_data(snr=None):
    """
    生成仿真所需的数据，并统一形状 (PyTorch版本)。

    参数:
        snr (float, optional): 信噪比。如果为 None，则不添加噪声。

    返回:
        tuple: 包含 (基准真相 X_true, 探测矩阵 P, 测量强度 I) 的元组。
               - X_true: 形状为 (N_ROWS, N) 的复数张量
               - P: 形状为 (M, N) 的复数张量
               - I: 形状为 (N_ROWS, M) 的实数张量
    """
    # 生成基准真相的散射矩阵 X_true, 形状 (N_ROWS, N)
    X_true = torch.randn(N_ROWS, N, dtype=torch.cfloat, device=device) / np.sqrt(2)
    row_norms = torch.linalg.norm(X_true, dim=1, keepdim=True) + 1e-12
    X_true = X_true / row_norms * (row_norms.mean())  # 或直接归一到 1

    # 生成探测矩阵 P, 形状 (M, N)
    # P = torch.randn(M, N, dtype=torch.complex64, device=device) / np.sqrt(2)
        

    # 随机二值(DMD)
    if BINARY:
        P = (2 * torch.randint(0, 2, (M, N), device=device, dtype=torch.int8) - 1).to(torch.float32).to(torch.cfloat)
    else:
        P = torch.randn(M, N, dtype=torch.complex64, device=device) / np.sqrt(2)

    # 这里假设每次

    if PADDING_SHAKE:
        
        # h_half, w_half = viewport_height // 2, viewport_width // 2
        landscape = int(np.sqrt(N))
        # M, ls, ls
        P = P.reshape(-1, landscape, landscape)
        # 随机平移偏移 (整数)
        shift_x = torch.randint(low=-landscape//4, high=landscape//4+1, size=(M,))
        shift_y = torch.randint(low=-landscape//4, high=landscape//4+1, size=(M,))
        # 基础方框大小 (左上 & 右下)
        box_size = landscape // 2  # 例如固定成 landscape/2 大小的方框

        masks = torch.zeros_like(P,dtype=torch.bool)

        for i in range(M):
            # 方框左上角坐标（加随机偏移）
            x1 = max(0, landscape//4 + shift_x[i].item())
            y1 = max(0, landscape//4 + shift_y[i].item())

            # 方框右下角坐标
            x2 = min(landscape, x1 + box_size)
            y2 = min(landscape, y1 + box_size)

            # 生成 mask
            masks[i, x1:x2, y1:y2] = 1
        
        P = P * masks
        P = P.reshape(M,N)


    # 计算无噪声的输出场和强度
    # (N_ROWS, N) @ (N, M) -> (N_ROWS, M)
    E_true = X_true @ P.T
    I_true = torch.abs(E_true)**2

    # 根据 SNR 添加高斯白噪声
    if snr is not None:
        # 逐行计算信号功率
        signal_power = torch.mean(I_true, dim=1, keepdim=True)
        noise_power = signal_power / snr
        noise_std = torch.sqrt(noise_power)
        noise = noise_std * torch.randn_like(I_true)
        I = I_true + noise
        I[I < 0] = 0  # 强度不能为负
    else:
        I = I_true

    return X_true, P, I

def correlation_coefficient(X1, X2):
    """
    计算两个复数张量之间的相关系数绝对值 (PyTorch版本)。
    X1, X2: 形状为 (N_ROWS, N)
    返回: 形状为 (N_ROWS,) 的相关系数值
    """
    norm1 = torch.linalg.norm(X1, dim=1)
    norm2 = torch.linalg.norm(X2, dim=1)
    
    # 防止除以零
    mask = (norm1 != 0) & (norm2 != 0)
    corr = torch.zeros(X1.shape[0], device=device)
    
    # 仅计算范数不为零的行
    if mask.any():
        # (X1[mask] * X2[mask].conj()).sum(dim=1) 是逐行点积
        dot_product = (X1[mask] * X2[mask].conj()).sum(dim=1)
        corr[mask] = torch.abs(dot_product) / (norm1[mask] * norm2[mask])
        
    return corr

# ---------------------------
# 谱初始化（Spectral Init）(PyTorch版本)
# ---------------------------
def spectral_init(P, y):
    """
    谱方法初始化 (PyTorch版本)。
    P: 探测矩阵, 形状 (M, N)
    y: 测量强度, 形状 (N_ROWS, M)
    返回: a0 (初始估计), 形状 (N_ROWS, N)
    """
    M, N = P.shape
    N_ROWS = y.shape[0]
    
    # 构造 Y = (1/M) * P^H * diag(y_row) * P
    # P: (M, N), y: (N_ROWS, M), P.T.conj(): (N, M)
    # Correct einsum string for batched operation:
    # nk: P.T.conj() -> (N, M)
    # ik: y -> (N_ROWS, M)
    # kj: P -> (M, N)
    # result inj -> (N_ROWS, N, N)
    Y = torch.einsum('nk,ik,kj->inj', P.T.conj(), y, P) / M
    
    # 特征值分解找最大特征向量
    # linalg.eigh 在批处理模式下工作
    vals, vecs = torch.linalg.eigh(Y)
    v = vecs[..., -1] # 取每个矩阵的最大特征向量
    
    # 幅度缩放
    # (N_ROWS, N) @ (N, M) -> (N_ROWS, M)
    z = v @ P.T
    # 逐行计算均值
    mean_y = torch.mean(y, dim=1, keepdim=True)
    mean_z_sq = torch.mean(torch.abs(z)**2, dim=1, keepdim=True)
    amp = torch.sqrt(mean_y / (mean_z_sq + 1e-12))
    
    return amp * v

def spectral_init_vmap(P, y):
    """
    谱方法初始化 (PyTorch版本) - 使用 vmap 进行向量化以优化内存 (修正版)。
    
    P: 探测矩阵, 形状 (M, N)
    y: 测量强度, 形状 (N_ROWS, M)
    返回: a0 (初始估计), 形状 (N_ROWS, N)
    """
    M, N = P.shape

    # 1. 定义一个处理单行 y_row 的函数
    #    【修正】: 将 P 作为显式参数传入
    def process_single_row(p_matrix, y_row):
        # p_matrix 的形状是 (M, N)
        # y_row 的形状是 (M,)
        
        # 构造 Y_single, 形状 (N, N)
        # 使用广播机制 (broadcasting) 比 einsum 更直接
        Y_single = (p_matrix.T.conj() * y_row.unsqueeze(0)) @ p_matrix / M
        
        # 特征值分解
        vals, vecs = torch.linalg.eigh(Y_single)
        v = vecs[..., -1] # 最大特征向量, (N,)
        
        # 幅度缩放
        z = v @ p_matrix.T # (N,) @ (N, M) -> (M,)
        
        mean_y = torch.mean(y_row)
        mean_z_sq = torch.mean(torch.abs(z)**2)
        amp = torch.sqrt(mean_y / (mean_z_sq + 1e-12))
        
        return amp * v

    # 2. 使用 vmap 对该函数进行向量化
    # in_dims=(None, 0) 表示：
    #   - p_matrix (第一个参数) 不进行批处理 (None)，每次都传递完整的P
    #   - y_row (第二个参数) 沿着第 0 维进行批处理 (0)，每次传递y的一行
    batched_process = vmap(process_single_row, in_dims=(None, 0), out_dims=0)
    
    # 3. 调用向量化后的函数，现在参数数量匹配了
    a0 = batched_process(P, y)
    
    return a0



def phase_conjugation_contrast( est_TM: torch.Tensor,transmission_matrix: torch.Tensor) -> torch.Tensor:
    """
    通过高效的矩阵运算，一次性计算在每个输出点上聚焦时的PBR。
    这可以看作是 correct_realistic_pcc 的并行化版本。

    Args:
        transmission_matrix (torch.Tensor): 传输矩阵 T, 形状为 (M, N)。

    Returns:
        torch.Tensor: 一个包含 M 个PBR值的向量，每个值对应一个输出焦点的效果。
    """
    M, N = transmission_matrix.shape

    # 1. 准备所有 M 个最优输入波前
    # 第 m 行是用于在输出点 m 聚焦的最优输入波前
    # 这个矩阵的形状是 (M, N)
    optimal_inputs_all = est_TM.conj()
    optimal_inputs_all = torch.exp(1j * torch.angle(optimal_inputs_all))
    

    # 2. 并行模拟 M 次传播过程
    # 我们将 M 个输入波前（optimal_inputs_all 的 M 个行向量）
    # 分别通过系统 T 进行传播。
    # T @ optimal_inputs_all.T 的数学含义是：
    # 结果矩阵 E 的第 (i, j) 个元素，代表用“目标为j”的波前输入后，在“位置i”得到的场。
    # T(M,N) @ optimal_inputs_all.T(N,M) -> E(M,M)
    E = transmission_matrix @ optimal_inputs_all.T

    # 3. 计算强度矩阵
    # I 的第 (i, j) 列，代表第 j 次聚焦实验在所有 M 个输出点上的光强分布
    I = torch.abs(E)**2

    # 4. 并行计算所有 M 次实验的PBR
    # 所有的焦点 (peaks) 正好是强度矩阵 I 的对角线
    # peak_j = I[j, j]
    peaks = torch.diagonal(I, 0)

    if M <= 1:
        return torch.tensor([float('inf')])

    # 对于第 j 次实验（第 j 列），背景光强是该列所有元素之和减去焦点光强
    background_sum_per_experiment = I.sum(dim=0) - peaks
    background_mean_per_experiment = background_sum_per_experiment / (M - 1)

    pbr_vector = peaks / (background_mean_per_experiment + 1e-12)

    return pbr_vector

# --- 3. 算法实现 (Autograd版本) ---

# ---------------------------
# Wirtinger Flow 主循环 (并行版)
# ---------------------------
def wirtinger_flow(P, y, iters=300, step=0.8, init=None):
    """
    Wirtinger Flow 算法 (PyTorch并行版)。
    P: 探测矩阵, 形状 (M, N)
    y: 测量强度, 形状 (N_ROWS, M)
    """
    M, N = P.shape
    a = init.clone() if init is not None else spectral_init(P, y)

    a_his = [a.clone()]
    y_sqrt = torch.sqrt(torch.clamp(y, min=0.0))

    for t in range(1, iters + 1):
        # (N_ROWS, N) @ (N, M) -> (N_ROWS, M)
        z = a @ P.T
        mag = torch.abs(z) + 1e-12
        
        # 梯度计算: grad = (1/M) * [ (1 - sqrt(y)/|z|) * z ] @ P.conj()
        residual = (1.0 - y_sqrt / mag) * z
        # (N_ROWS, M) @ (M, N) -> (N_ROWS, N)
        grad = residual @ P.conj() / M

        # 步长衰减
        eta = step * (1 - t / iters) + 0.05
        a -= eta * grad

        # 检查收敛 (每行独立检查)
        a_his.append(a.clone())
        if t > 3:
            corr_conv = correlation_coefficient(a, a_his[-4])
            if torch.all(corr_conv > CONVERGENCE_THRESHOLD):
                return t, a
    
    return iters, a

# ---------------------------
# 带正则项的 Wirtinger Flow (并行版)
# ---------------------------
def wirtinger_flow_gaussian(P, y, iters=300, step=0.8, init=None):
    """
    带高斯先验正则项的 Wirtinger Flow 算法 (PyTorch并行版)。
    P: 探测矩阵, 形状 (M, N)
    y: 测量强度, 形状 (N_ROWS, M)
    """
    M, N = P.shape
    a = init.clone() if init is not None else spectral_init(P, y)
    
    a_his = [a.clone()]
    y_sqrt = torch.sqrt(torch.clamp(y, min=0.0))

    for t in range(1, iters + 1):
        # (N_ROWS, N) @ (N, M) -> (N_ROWS, M)
        z = a @ P.T
        mag = torch.abs(z) + 1e-12
        
        residual = (1.0 - y_sqrt / mag) * z
        # 数据项梯度 + 正则项梯度
        # (N_ROWS, M) @ (M, N) -> (N_ROWS, N)
        grad = (residual @ P.conj() + 2 * LAMBDA * a) / M

        eta = step * (1 - t / iters) + 0.05
        a -= eta * grad

        a_his.append(a.clone())
        if t > 3:
            corr_conv = correlation_coefficient(a, a_his[-4])
            if torch.all(corr_conv > CONVERGENCE_THRESHOLD):
                return t, a

    return iters, a


# GGS 1
# ---------------------------
# 带正则项的 Wirtinger Flow (并行版)
# ---------------------------
def GGS1(P, y, iters=300, step=0.8, init=None):
    """
    Generalized GS 1
    P: 探测矩阵, 形状 (M, N)
    y: 测量强度, 形状 (M, N)
    """
    M, N = P.shape
    y = y.T
    a = init.clone() if init is not None else spectral_init(P, y)
    # 测试过随机初始化,效果不好只有0.5
    
    a_his = [a.clone()]
    y_sqrt = torch.sqrt(torch.clamp(y, min=0.0))
    P_pinv = torch.linalg.pinv(P)

    for t in range(1, iters + 1):
        # (N_ROWS, N) @ (N, M) -> (N_ROWS, M)
        E = P @ a.T
        E = y_sqrt * torch.exp(1j * torch.angle(E))
        
        a = (P_pinv @ E).T

        a_his.append(a.clone())
        if t > 3:
            corr_conv = correlation_coefficient(a, a_his[-4])
            if torch.all(corr_conv > CONVERGENCE_THRESHOLD):
                return t, a

    return iters, a

# GGS 21
# ---------------------------
# 带正则项的 Wirtinger Flow (并行版)
# ---------------------------
def GGS2_1(P, y, iters=300, step=0.8, init=None):
    """
    Generalized GS 2-1
    P: 探测矩阵, 形状 (M, N)
    y: 测量强度, 形状 (M, N)
    """
    M, N = P.shape
    y = y.T
    a = init.clone() if init is not None else spectral_init(P, y)
    # 测试过随机初始化,效果不好只有0.5
    
    a_his = [a.clone()]
    y_sqrt = torch.sqrt(torch.clamp(y, min=0.0))
    P_pinv = torch.linalg.pinv(P)
    pow_n = 2

    for t in range(1, iters + 1):
        # (N_ROWS, N) @ (N, M) -> (N_ROWS, M)
        if(t > (iters*2) / 3):
            pow_n = pow_n - ( 1 / (iters / 2) )
            
        E = P @ a.T
        E = torch.pow(y_sqrt, pow_n) * torch.exp(1j * torch.angle(E))
        
        a = (P_pinv @ E).T

        a_his.append(a.clone())
        if t > 3:
            corr_conv = correlation_coefficient(a, a_his[-4])
            if torch.all(corr_conv > CONVERGENCE_THRESHOLD):
                return t, a

    return iters, a

def pr_vbem(y_mag, D, opt=None):
    """
    PyTorch implementation of the prVBEM algorithm for phase retrieval, optimized for batch processing.

    This function implements the phase recovery algorithm described in:
    "Phase recovery from a Bayesian point of view: the variational approach" by A. Drémeau and F. Krzakala.

    Args:
        y_mag (torch.Tensor): A batch of observation magnitudes.
                              Shape: (B, M), dtype: torch.float32.
                              B is batch size, M is the number of measurements.
        D (torch.Tensor): The dictionary or sensing matrix.
                          Shape: (M, N), dtype: torch.complex64.
                          N is the signal dimension.
        opt (dict, optional): A dictionary of options.
            .var_a (float): Variance of the non-zero coefficients in x.
            .var_n (float): Initial variance of Gaussian noise n (default: 1e-8).
            .x0 (torch.Tensor): Initial guess for the signal x (default: computed via pseudoinverse).
                                 Shape: (B, N, 1) or (N, 1).
            .niter (int): Maximum number of iterations (default: 500).
            .flag_est_n (str): If 'on', noise variance is estimated (default: 'on').
            .flag_cv (str): Convergence criterion, 'KL' for Kullback-Leibler (default: 'KL').
            .pas_est (float): Step size for noise variance adjustment (default: 0.1).

    Returns:
        tuple:
            - torch.Tensor: The recovered signal batch `x_hat`. Shape: (B, N), dtype: torch.complex64.
            - torch.Tensor: The final Kullback-Leibler divergence for each item in the batch.
                            Shape: (B, 1, 1), dtype: torch.float32.
    """
    # --- 1. Setup and Initialization ---
    device = D.device
    dtype = D.dtype
    float_dtype = torch.float32

    M, N = D.shape
    B = y_mag.shape[0]

    # Reshape y_mag for matrix operations to (B, M, 1)
    y_mag = y_mag.view(B, M, 1).to(float_dtype)

    # --- 2. Default Parameters ---
    if opt is None:
        opt = {}

    # Create an initial guess for x0 if not provided
    with torch.no_grad():
        rand_phase = torch.exp(1j * 2 * torch.pi * torch.rand(B, M, 1, device=device, dtype=dtype))
        y_complex_init = y_mag * rand_phase
        x0_default = torch.linalg.pinv(D) @ y_complex_init
        var_a_default = torch.max(torch.abs(x0_default)**2).item()

    # Get options or use defaults
    var_a = torch.tensor(opt.get('var_a', var_a_default), device=device, dtype=float_dtype)
    var_n_init = torch.tensor(opt.get('var_n', 1e-8), device=device, dtype=float_dtype)
    x0 = opt.get('x0', x0_default).clone().to(dtype)
    if x0.dim() == 2: # Expand single x0 to batch size
        x0 = x0.unsqueeze(2).expand(B, -1, -1)
    
    flag_est_n = opt.get('flag_est_n', 'on')
    niter = opt.get('niter', 500)
    pas_est = opt.get('pas_est', 0.1)
    flag_cv = opt.get('flag_cv', 'KL')

    # Ensure var_n is not zero and make it broadcastable for batch operations
    var_n = torch.clamp(var_n_init, min=1e-6).view(1, 1, 1).expand(B, -1, -1)


    # --- 3. Initialize Variables for Iteration ---
    moy_x = x0
    ybar = y_complex_init # Initial estimate for phase-corrected y

    # Pre-compute H = D'D and its diagonal
    H = D.mH @ D
    H_diag = torch.diag(H).real.view(1, N, 1)

    # Initial calculations using broadcasted operations
    z = D.mH @ ybar
    var_x = (var_a * var_n) / (var_n + var_a * H_diag) # Broadcasts to (B, N, 1)
    
    var_n_true = var_n.clone()
    
    KLdiv_old = torch.full((B, 1, 1), 100000.0, device=device, dtype=float_dtype)
    # Mask to track which items in the batch are still iterating
    is_iterating = torch.ones(B, dtype=torch.bool, device=device)
    KLdiv = KLdiv_old.clone()

    # --- 4. Iterative Process ---
    for compt in range(niter+1):
        if not is_iterating.any():
            break # Stop if all items in the batch have converged

        # --- Estimation of var_n (Noise Variance) ---
        if flag_est_n == 'on' or flag_est_n == 'on_off':
            with torch.no_grad():
                Dm = D @ moy_x
                term1 = torch.sum(torch.abs(Dm)**2, dim=1, keepdim=True)
                term2 = torch.sum(var_x * H_diag, dim=1, keepdim=True)
                term3 = torch.sum(y_mag**2, dim=1, keepdim=True)
                term4 = -2 * torch.real(torch.sum(torch.conj(ybar) * Dm, dim=1, keepdim=True))
                var_n_new = (1.0 / M) * (term1 + term2 + term3 + term4)
                # Update var_n only for items that are still iterating
                current_var_n = torch.clamp(var_n_new.real, min=1e-9).view(B, 1, 1)
                var_n = torch.where(is_iterating.view(B,1,1), current_var_n, var_n)

        # --- Vectorized Update of q(x) ---
        # This vectorized operation replaces the inner for-loop over dim_s in MATLAB
        val_tmp = z - (H @ moy_x) + (H_diag * moy_x)
        
        update_factor_moy = var_a / (var_n + var_a * H_diag)
        moy_x_new = update_factor_moy * val_tmp
        
        update_factor_var = (var_n * var_a) / (var_n + var_a * H_diag)
        var_x_new = update_factor_var

        # Update moy_x and var_x only for iterating items
        moy_x = torch.where(is_iterating.view(B,1,1), moy_x_new, moy_x)
        var_x = torch.where(is_iterating.view(B,1,1), var_x_new, var_x)
        
        # --- Update Posterior of Phase (ybar) ---
        Dm = D @ moy_x
        t = y_mag * Dm
        t_abs = torch.abs(t)
        
        # Avoid division by zero for phase calculation
        phi = t / torch.clamp(t_abs, min=1e-20)
        
        bessel_arg = (2.0 / var_n) * t_abs
        I0 = torch.special.i0(bessel_arg)
        I1 = torch.special.i1(bessel_arg)

        fac_bessel = I1 / torch.clamp(I0, min=1e-20)
        fac_bessel[torch.isnan(fac_bessel)] = 1.0 # Handle potential NaNs

        ybar_new = y_mag * phi * fac_bessel
        ybar = torch.where(is_iterating.view(B,1,1), ybar_new, ybar)
        
        # Re-calculate z with the new ybar
        z = D.mH @ ybar

        # --- 5. Convergence Checks (using KL Divergence) ---
        if flag_cv == 'KL':
            with torch.no_grad():
                # Recalculate Dm for KLdiv
                Dm = D @ moy_x
                
                # Batched KL Divergence Calculation
                term_in_paren = torch.sum(torch.abs(Dm)**2, dim=1, keepdim=True) + \
                                torch.sum(var_x * H_diag, dim=1, keepdim=True) + \
                                torch.sum(y_mag**2, dim=1, keepdim=True) - \
                                2 * torch.real(torch.sum(torch.conj(ybar) * Dm, dim=1, keepdim=True))
                frst = M * torch.log(var_n) + (1.0 / var_n) * term_in_paren
                scnd = N * torch.log(var_a) + (1.0 / var_a) * torch.sum(var_x + torch.abs(moy_x)**2, dim=1, keepdim=True)
                sxth = torch.sum(torch.log(torch.clamp(var_x, min=1e-20)), dim=1, keepdim=True)
                vec_tmp = t_abs * fac_bessel
                svth = (2.0 / var_n) * torch.sum(vec_tmp, dim=1, keepdim=True) - torch.sum(torch.log(torch.clamp(I0, min=1e-20)), dim=1, keepdim=True)

                KLdiv_new = frst + scnd - sxth + svth
                KLdiv = torch.where(is_iterating.view(B,1,1), KLdiv_new, KLdiv)

                diff = KLdiv_old - KLdiv
                
                # Identify which items have converged
                converged_mask = (diff.abs() < 1e-7).squeeze() & is_iterating
                
                # For converged items, check if var_n has grown too large
                var_n_too_large_mask = (var_n > 1.1 * var_n_true).squeeze()
                
                # Apply adjustment to items that converged but have large var_n
                adjust_var_n_mask = converged_mask & var_n_too_large_mask
                if adjust_var_n_mask.any():
                    var_n[adjust_var_n_mask, ...] -= pas_est * (var_n[adjust_var_n_mask, ...] - var_n_true[adjust_var_n_mask, ...])
                    flag_est_n = 'off' # Turn off estimation for all subsequent steps

                # Stop iterating for items that converged and have stable var_n
                stop_mask = converged_mask & ~var_n_too_large_mask
                is_iterating[stop_mask] = False
                
                KLdiv_old = KLdiv.clone()

    return moy_x.squeeze(2), compt

def raf21_tm(
    P: torch.Tensor,
    y: torch.Tensor, 
    iters: int = 300,
    step: float = 0.8,
    init: Optional[torch.Tensor] = None,
    ratio: float = 2/3,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[int, torch.Tensor]:
    """
    重加权振幅流（Reweighted Amplitude Flow）模型用于恢复传输矩阵 TM (PyTorch 版本)。
    前向模型为 Y = abs(A * P.T)^2。给定 Y 和 P，函数返回 A 的估计值。
    此版本适配了逐行（row-by-row）数据格式，并恢复了算法关键的初始化步骤。

    Args:
        P (torch.Tensor): 已知的探测/模式矩阵 P，形状为 (M, N)。
        y (torch.Tensor): 测量矩阵 Y，形状为 (N_ROWS, M)。
        iters (int): 迭代次数。
        step (float): 梯度下降的步长 (mu)。
        init (torch.Tensor, optional): 矩阵 A 的初始猜测，形状为 (N_ROWS, N)。
                                       如果为 None，则使用 RAF 特定的初始化方法。
        ratio (float): 用于切换梯度公式的迭代比例。
        device (str or torch.device, optional): 运行计算的设备 ('cpu' or 'cuda')。
                                                 如果为 None，则自动检测 GPU。

    Returns:
        一个元组，包含:
        - iters (int): 执行的总迭代次数。
        - A_est (torch.Tensor): 估计的矩阵 A，形状为 (N_ROWS, N)。
    """
    # --- 1. 设置 ---
    
    # 确定计算设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # 确保输入张量在正确的设备上并具有正确的类型
    P = P.to(device, dtype=torch.cfloat)
    y = y.to(device, dtype=torch.float32)

    # 从输入中获取维度 - 匹配逐行格式
    N_ROWS, M = y.shape
    M_check, N = P.shape
    assert M == M_check, f"维度不匹配: y 有 {M} 个测量值, 而 P 有 {M_check}"
    
    # 其他算法常量
    npower_iter = 30  # 幂迭代次数
    beta = 5.0
    gamma = 0.5
    eps = 1e-30  # 用于数值稳定的小常数
    
    ymag = torch.sqrt(torch.clamp(y, min=0.0))
    
    # --- 2. 初始化 ---
    
    if init is not None:
        A_est = init.clone().to(device, dtype=torch.cfloat)
    else:
        # --- 关键步骤: 使用原始的 RAF 初始化逻辑 ---
        # 注意: 与原始 MATLAB 脚本相比，此处的逻辑进行了转置以匹配 A_est 的 (N_ROWS, N) 形状。
        
        # 随机初始化 A_est，形状为 (N_ROWS, N)
        A_est = (torch.randn(N_ROWS, N, dtype=torch.cfloat) / math.sqrt(2)).to(device)
        A_est = A_est / torch.linalg.norm(A_est, ord=2, dim=1, keepdim=True)
        
        # 准备加权最大相关性初始化
        # normest 需要有 (N_ROWS, 1) 的形状来缩放 A 的每一行
        normest = torch.sqrt(torch.mean(y, dim=1, keepdim=True))
        ysort, _ = torch.sort(y, dim=1)  # 沿着 M 维度排序
        ythresh_idx = int(round(M / 1.3)) - 1
        ythresh = ysort[:, ythresh_idx].unsqueeze(1)
        ind = (y >= ythresh).to(torch.float32)

        # 加权最大相关性初始化 (幂迭代)
        weights = ymag.pow(gamma) * ind
        for _ in range(npower_iter):
            # A_est shape: (N_ROWS, N)
            # P shape: (M, N) -> P.T shape: (N, M)
            # weights shape: (N_ROWS, M)
            # P.conj() shape: (M, N)
            # (A_est @ P.T) -> (N_ROWS, M)
            # (weights * (...)) @ P.conj() -> (N_ROWS, N)
            A_est = (weights * (A_est @ P.T)) @ P.conj()
            A_est = A_est / (torch.linalg.norm(A_est, ord=2, dim=1, keepdim=True) + eps)
        
        # 应用缩放
        A_est = normest * A_est
    
    # --- 3. 梯度下降循环 ---
    a_his = []
    for t in range(iters):
        # 前向模型: z = A @ P.T, 形状 (N_ROWS, M)
        z = A_est @ P.T
        abs_z = torch.abs(z) + eps
        phase = z / abs_z
        
        # RAF2-1 加权
        weight = abs_z / (abs_z + beta * ymag)
        
        # 根据迭代阶段计算梯度
        if (t + 1) <= math.ceil(ratio * iters):
            # 第一阶段: 使用测量的平方
            residual = (z - ymag.pow(2) * phase) * weight
        else:
            # 第二阶段: 使用线性的测量值
            residual = (z - ymag * phase) * weight
        
        # 梯度: (N_ROWS, M) @ (M, N) -> (N_ROWS, N)
        grad = residual @ P.conj() / M
        
        # 更新
        A_est -= step * grad

        a_his.append(A_est.clone())

        if t > 3:
            corr_conv = correlation_coefficient(A_est, a_his[-4])
            if torch.all(corr_conv > CONVERGENCE_THRESHOLD):
                return t, A_est
        
    
    # --- 4. 最终处理 (从原始版本恢复) ---
    inf_ind = torch.abs(A_est) > 1e10
    if torch.any(inf_ind):
        A_est[inf_ind] = torch.exp(1j * torch.angle(A_est[inf_ind]))

    mean_A = torch.mean(A_est)
    std_val = torch.std(torch.abs(A_est - mean_A), unbiased=False)
    A_est = A_est / (std_val + eps)
    
    return t, A_est
 
# ---------------------------
# Autograd-based amplitude loss optimizer (Adam)
# ---------------------------
def autograd_amp_solver(
    P: torch.Tensor,
    y: torch.Tensor,
    iters: int = 300,
    lr: float = 1e-2,
    init: Optional[torch.Tensor] = None,
    l1: float = 0.0,
    l2: float = 0.0,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[int, torch.Tensor]:
    """
    使用 PyTorch autograd 和 Adam 直接最小化振幅误差，恢复 A 使得 |A @ P.T| ≈ sqrt(y)。

    Args:
        P: (M, N) complex64/complex32
        y: (N_ROWS, M) float
        iters: 最大优化步数
        lr: Adam 学习率
        init: (N_ROWS, N) complex 初始值；默认使用 spectral_init
        l1: L1 正则权重（对幅度）
        l2: L2 正则权重（对幅度平方）
        device: 计算设备

    Returns:
        (iters_used, A_est)
    """
    if device is None:
        device = P.device
    P = P.to(device, dtype=torch.cfloat)
    y = y.to(device, dtype=torch.float32)

    N_ROWS, M = y.shape
    M_check, N = P.shape
    assert M == M_check, "P 与 y 尺寸不匹配"

    with torch.no_grad():
        if init is None:
            a0 = spectral_init(P, y)
        else:
            a0 = init
    X_re = torch.nn.Parameter(a0.real.detach().to(device=device, dtype=torch.float32).clone())
    X_im = torch.nn.Parameter(a0.imag.detach().to(device=device, dtype=torch.float32).clone())

    optimizer = torch.optim.Adam([X_re, X_im], lr=lr)
    eps = 1e-20
    y_sqrt = torch.sqrt(torch.clamp(y, min=0.0))

    snap_history = []
    last_corr = None

    for t in range(1, iters + 1):
        optimizer.zero_grad(set_to_none=True)

        X = (X_re + 1j * X_im).to(torch.cfloat)
        z = X @ P.T  # (N_ROWS, M)
        amp = torch.abs(z)
        loss_data = torch.mean((amp - y_sqrt) ** 2)

        # Regularization on magnitude of X
        mag_X = torch.sqrt(torch.clamp(X_re**2 + X_im**2, min=eps))
        loss_reg = l1 * torch.mean(mag_X) + l2 * torch.mean(mag_X ** 2)
        loss = loss_data + loss_reg

        loss.backward()
        optimizer.step()

        if t % 4 == 0:
            with torch.no_grad():
                X_cur = (X_re + 1j * X_im).to(torch.cfloat)
                if len(snap_history) > 0:
                    corr_conv = correlation_coefficient(X_cur, snap_history[-1])
                    last_corr = corr_conv.min().item()
                    if torch.all(corr_conv > CONVERGENCE_THRESHOLD):
                        return t, X_cur
                snap_history.append(X_cur.detach().clone())

    with torch.no_grad():
        X_final = (X_re + 1j * X_im).to(torch.cfloat)
    return iters, X_final

# ---------------------------
# Autograd-based RAF (Adam)
# ---------------------------
def autograd_raf_solver(
    P: torch.Tensor,
    y: torch.Tensor,
    iters: int = 300,
    lr: float = 1e-2,
    init: Optional[torch.Tensor] = None,
    ratio: float = 2/3,
    beta: float = 5.0,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[int, torch.Tensor]:
    """
    使用 Adam 最小化 RAF 残差的平方损失；保持与 raf21_tm 相同的两阶段残差与权重形式。
    """
    if device is None:
        device = P.device
    P = P.to(device, dtype=torch.cfloat)
    y = y.to(device, dtype=torch.float32)

    N_ROWS, M = y.shape
    M_check, N = P.shape
    assert M == M_check, "P 与 y 尺寸不匹配"

    with torch.no_grad():
        if init is None:
            a0 = spectral_init(P, y)
        else:
            a0 = init
    X_re = torch.nn.Parameter(a0.real.detach().to(device=device, dtype=torch.float32).clone())
    X_im = torch.nn.Parameter(a0.imag.detach().to(device=device, dtype=torch.float32).clone())

    optimizer = torch.optim.Adam([X_re, X_im], lr=lr)
    eps = 1e-30
    ymag = torch.sqrt(torch.clamp(y, min=0.0))

    snap_history = []

    for t in range(1, iters + 1):
        optimizer.zero_grad(set_to_none=True)

        X = (X_re + 1j * X_im).to(torch.cfloat)
        z = X @ P.T
        abs_z = torch.abs(z) + eps
        phase = z / abs_z
        weight = abs_z / (abs_z + beta * ymag)

        if (t) <= math.ceil(ratio * iters):
            residual = (z - ymag.pow(2) * phase) * weight
        else:
            residual = (z - ymag * phase) * weight

        loss = torch.mean(torch.abs(residual) ** 2)
        loss.backward()
        optimizer.step()

        if t % 4 == 0:
            with torch.no_grad():
                X_cur = (X_re + 1j * X_im).to(torch.cfloat)
                if len(snap_history) > 0:
                    corr_conv = correlation_coefficient(X_cur, snap_history[-1])
                    if torch.all(corr_conv > CONVERGENCE_THRESHOLD):
                        return t, X_cur
                snap_history.append(X_cur.detach().clone())

    with torch.no_grad():
        X_final = (X_re + 1j * X_im).to(torch.cfloat)
    return iters, X_final

# ---------------------------
# Autograd-based Wirtinger Flow (Adam)
# ---------------------------
def autograd_wf_solver(
    P: torch.Tensor,
    y: torch.Tensor,
    iters: int = 300,
    lr: float = 1e-2,
    init: Optional[torch.Tensor] = None,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[int, torch.Tensor]:
    if device is None:
        device = P.device
    P = P.to(device, dtype=torch.cfloat)
    y = y.to(device, dtype=torch.float32)

    N_ROWS, M = y.shape
    M_check, N = P.shape
    assert M == M_check, "P 与 y 尺寸不匹配"

    with torch.no_grad():
        if init is None:
            a0 = spectral_init(P, y)
        else:
            a0 = init
    X_re = torch.nn.Parameter(a0.real.detach().to(device=device, dtype=torch.float32).clone())
    X_im = torch.nn.Parameter(a0.imag.detach().to(device=device, dtype=torch.float32).clone())

    optimizer = torch.optim.Adam([X_re, X_im], lr=lr)
    y_sqrt = torch.sqrt(torch.clamp(y, min=0.0))
    eps = 1e-20
    snaps = []

    for t in range(1, iters + 1):
        optimizer.zero_grad(set_to_none=True)
        X = (X_re + 1j * X_im).to(torch.cfloat)
        z = X @ P.T
        amp = torch.abs(z)
        loss = torch.mean((amp - y_sqrt) ** 2)
        loss.backward()
        optimizer.step()

        if t % 4 == 0:
            with torch.no_grad():
                X_cur = (X_re + 1j * X_im).to(torch.cfloat)
                if len(snaps) > 0:
                    corr_conv = correlation_coefficient(X_cur, snaps[-1])
                    if torch.all(corr_conv > CONVERGENCE_THRESHOLD):
                        return t, X_cur
                snaps.append(X_cur.detach().clone())

    with torch.no_grad():
        X_final = (X_re + 1j * X_im).to(torch.cfloat)
    return iters, X_final

# ---------------------------
# Autograd-based WF with Gaussian regularization (Adam)
# ---------------------------
def autograd_wf_reg_solver(
    P: torch.Tensor,
    y: torch.Tensor,
    iters: int = 300,
    lr: float = 1e-2,
    lambda_reg: float = 15.0,
    init: Optional[torch.Tensor] = None,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[int, torch.Tensor]:
    if device is None:
        device = P.device
    P = P.to(device, dtype=torch.cfloat)
    y = y.to(device, dtype=torch.float32)

    with torch.no_grad():
        if init is None:
            a0 = spectral_init(P, y)
        else:
            a0 = init
    X_re = torch.nn.Parameter(a0.real.detach().to(device=device, dtype=torch.float32).clone())
    X_im = torch.nn.Parameter(a0.imag.detach().to(device=device, dtype=torch.float32).clone())

    optimizer = torch.optim.Adam([X_re, X_im], lr=lr)
    y_sqrt = torch.sqrt(torch.clamp(y, min=0.0))
    eps = 1e-20
    snaps = []

    for t in range(1, iters + 1):
        optimizer.zero_grad(set_to_none=True)
        X = (X_re + 1j * X_im).to(torch.cfloat)
        z = X @ P.T
        amp = torch.abs(z)
        data_loss = torch.mean((amp - y_sqrt) ** 2)
        mag_X2 = X_re**2 + X_im**2
        reg_loss = lambda_reg * torch.mean(mag_X2)
        loss = data_loss + reg_loss
        loss.backward()
        optimizer.step()

        if t % 4 == 0:
            with torch.no_grad():
                X_cur = (X_re + 1j * X_im).to(torch.cfloat)
                if len(snaps) > 0:
                    corr_conv = correlation_coefficient(X_cur, snaps[-1])
                    if torch.all(corr_conv > CONVERGENCE_THRESHOLD):
                        return t, X_cur
                snaps.append(X_cur.detach().clone())

    with torch.no_grad():
        X_final = (X_re + 1j * X_im).to(torch.cfloat)
    return iters, X_final

# ---------------------------
# Autograd-based GGS1 (Adam)
# ---------------------------
def autograd_ggs1_solver(
    P: torch.Tensor,
    y: torch.Tensor,
    iters: int = 300,
    lr: float = 1e-2,
    init: Optional[torch.Tensor] = None,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[int, torch.Tensor]:
    if device is None:
        device = P.device
    P = P.to(device, dtype=torch.cfloat)
    y = y.to(device, dtype=torch.float32)

    with torch.no_grad():
        a0 = spectral_init(P, y)
    X_re = torch.nn.Parameter(a0.real.detach().to(device=device, dtype=torch.float32).clone())
    X_im = torch.nn.Parameter(a0.imag.detach().to(device=device, dtype=torch.float32).clone())

    optimizer = torch.optim.Adam([X_re, X_im], lr=lr)
    y_sqrt_T = torch.sqrt(torch.clamp(y, min=0.0)).T
    snaps = []

    for t in range(1, iters + 1):
        optimizer.zero_grad(set_to_none=True)
        X = (X_re + 1j * X_im).to(torch.cfloat)
        E = P @ X.T  # (M, N_ROWS)
        target = y_sqrt_T * torch.exp(1j * torch.angle(E))
        residual = E - target
        loss = torch.mean(torch.abs(residual) ** 2)
        loss.backward()
        optimizer.step()

        if t % 4 == 0:
            with torch.no_grad():
                X_cur = (X_re + 1j * X_im).to(torch.cfloat)
                if len(snaps) > 0:
                    corr_conv = correlation_coefficient(X_cur, snaps[-1])
                    if torch.all(corr_conv > CONVERGENCE_THRESHOLD):
                        return t, X_cur
                snaps.append(X_cur.detach().clone())

    with torch.no_grad():
        X_final = (X_re + 1j * X_im).to(torch.cfloat)
    return iters, X_final

# ---------------------------
# Autograd-based GGS2-1 (Adam)
# ---------------------------
def autograd_ggs2_1_solver(
    P: torch.Tensor,
    y: torch.Tensor,
    iters: int = 300,
    lr: float = 1e-2,
    init: Optional[torch.Tensor] = None,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[int, torch.Tensor]:
    if device is None:
        device = P.device
    P = P.to(device, dtype=torch.cfloat)
    y = y.to(device, dtype=torch.float32)

    with torch.no_grad():
        a0 = spectral_init(P, y)
    X_re = torch.nn.Parameter(a0.real.detach().to(device=device, dtype=torch.float32).clone())
    X_im = torch.nn.Parameter(a0.imag.detach().to(device=device, dtype=torch.float32).clone())

    optimizer = torch.optim.Adam([X_re, X_im], lr=lr)
    y_sqrt_T = torch.sqrt(torch.clamp(y, min=0.0)).T
    snaps = []

    for t in range(1, iters + 1):
        optimizer.zero_grad(set_to_none=True)
        X = (X_re + 1j * X_im).to(torch.cfloat)
        E = P @ X.T  # (M, N_ROWS)
        # schedule pow_n from 2 down to 1 after 2/3 iters
        if t <= (2 * iters) // 3:
            pow_n = 2.0
        else:
            frac = (t - (2 * iters) / 3.0) / (iters / 2.0)
            pow_n = max(1.0, 2.0 - frac)
        target = torch.pow(y_sqrt_T, pow_n) * torch.exp(1j * torch.angle(E))
        residual = E - target
        loss = torch.mean(torch.abs(residual) ** 2)
        loss.backward()
        optimizer.step()

        if t % 4 == 0:
            with torch.no_grad():
                X_cur = (X_re + 1j * X_im).to(torch.cfloat)
                if len(snaps) > 0:
                    corr_conv = correlation_coefficient(X_cur, snaps[-1])
                    if torch.all(corr_conv > CONVERGENCE_THRESHOLD):
                        return t, X_cur
                snaps.append(X_cur.detach().clone())

    with torch.no_grad():
        X_final = (X_re + 1j * X_im).to(torch.cfloat)
    return iters, X_final
# --- 4. 主仿真循环 ---

def run_simulation():
    """运行完整的仿真流程。"""
    results = {
        'snrs': [snr if snr is not None else np.inf for snr in SNRS],
        'wf': {'iters': [], 'accs': [], 'times': []},
        'wf_reg': {'iters': [], 'accs': [], 'times': []},
        'ggs1': {'iters': [], 'accs': [], 'times': []},
        'ggs2_1': {'iters': [], 'accs': [], 'times': []},
        'fista': {'iters': [], 'accs': [], 'times': []},
        'pr_vbem': {'iters': [], 'accs': [], 'times': []},
        'raf21_tm': {'iters': [], 'accs': [], 'times': []},
        'autograd': {'iters': [], 'accs': [], 'times': []},
        'raf21_tm_auto': {'iters': [], 'accs': [], 'times': []},
        'wf_auto': {'iters': [], 'accs': [], 'times': []},
        'wf_reg_auto': {'iters': [], 'accs': [], 'times': []},
        'ggs1_auto': {'iters': [], 'accs': [], 'times': []},
        'ggs2_1_auto': {'iters': [], 'accs': [], 'times': []},
        'focus': {'wf': [], 'wf_reg': [], 'ggs1': [], 'ggs2_1': [], 'fista': [], 'pr_vbem': [], 'raf21_tm': [], 'autograd': []},
        'last_trial': None
    }

    for snr_idx, snr in enumerate(SNRS):
        print(f"\n--- 正在测试 SNR = {snr if snr is not None else '无噪声'} ---")

        # 用于存储每个 trial 的结果
        wf_iters_trial, wf_accs_trial, wf_focus_trial, wf_times_trial = [], [], [], []
        wf_reg_iters_trial, wf_reg_accs_trial, wf_reg_focus_trial, wf_reg_times_trial = [], [], [], []
        ggs1_iters_trial, ggs1_accs_trial, ggs1_focus_trial, ggs1_times_trial = [], [], [], []
        ggs2_1_iters_trial, ggs2_1_accs_trial, ggs2_1_focus_trial, ggs2_1_times_trial = [], [], [], []
        fista_iters_trial, fista_accs_trial, fista_focus_trial, fista_times_trial = [], [], [], []
        pr_vbem_iters_trial, pr_vbem_accs_trial, pr_vbem_focus_trial, pr_vbem_times_trial = [], [], [], []
        raf21_tm_iters_trial, raf21_tm_accs_trial, raf21_tm_focus_trial, raf21_tm_times_trial = [], [], [], []
        autograd_iters_trial, autograd_accs_trial, autograd_focus_trial, autograd_times_trial = [], [], [], []
        raf21_tm_auto_iters_trial, raf21_tm_auto_accs_trial, raf21_tm_auto_focus_trial, raf21_tm_auto_times_trial = [], [], [], []
        wf_auto_iters_trial, wf_auto_accs_trial, wf_auto_focus_trial, wf_auto_times_trial = [], [], [], []
        wf_reg_auto_iters_trial, wf_reg_auto_accs_trial, wf_reg_auto_focus_trial, wf_reg_auto_times_trial = [], [], [], []
        ggs1_auto_iters_trial, ggs1_auto_accs_trial, ggs1_auto_focus_trial, ggs1_auto_times_trial = [], [], [], []
        ggs2_1_auto_iters_trial, ggs2_1_auto_accs_trial, ggs2_1_auto_focus_trial, ggs2_1_auto_times_trial = [], [], [], []

        for trial_idx in tqdm(range(NUM_TRIALS), desc="独立试验进度"):
            # 生成数据，数据已在指定设备上
            X_true_full, P, I_full = generate_data(snr)
            
            # 使用谱方法进行初始化
            # a0 = spectral_init(P.to('cpu'), I.to('cpu')).to(device)
            # a0 = spectral_init_vmap(P, I)

            for batch_idx in range(0, N_ROWS, BATCH_SIZE):
                a0 = torch.randn(min(BATCH_SIZE, N_ROWS - batch_idx), N, device=device) + 1j * torch.randn(min(BATCH_SIZE, N_ROWS - batch_idx), N, device=device)
                # P = P_full[batch_idx, min(batch_idx+BATCH_SIZE, N_ROWS)]
                I = I_full[batch_idx : min(batch_idx+BATCH_SIZE, N_ROWS), :]
                X_true = X_true_full[batch_idx: min(batch_idx+BATCH_SIZE, N_ROWS), :]
                # FISTA 已移除（仅保留 autograd 算法）
                # 运行 Wirtinger Flow
                if ENABLE.get('wf', False):
                    t0 = time.perf_counter()
                    iters_wf, X_final_wf = wirtinger_flow(P, I, iters=MAX_ITERATIONS, step=0.9, init=a0)
                    t1 = time.perf_counter()
                    X_align_wf = align_global_phase(X_final_wf, X_true)
                    acc_wf = correlation_coefficient(X_align_wf, X_true)
                    focus_wf = phase_conjugation_contrast(X_align_wf, X_true)
                    # 记录平均迭代次数和平均精度
                    wf_iters_trial.append(iters_wf)
                    wf_accs_trial.extend(acc_wf.cpu().numpy())
                    wf_focus_trial.append(focus_wf.cpu().numpy().mean())
                    wf_times_trial.append(t1 - t0)
                # 运行带正则项的 Wirtinger Flow
                if ENABLE.get('wf_reg', False):
                    t0 = time.perf_counter()
                    iters_wf_reg, X_final_wf_reg = wirtinger_flow_gaussian(P, I, iters=MAX_ITERATIONS, step=0.9, init=a0)
                    t1 = time.perf_counter()
                    X_align_wf_reg = align_global_phase(X_final_wf_reg, X_true)
                    acc_wf_reg = correlation_coefficient(X_align_wf_reg, X_true)
                    focus_wf_reg = phase_conjugation_contrast(X_align_wf_reg, X_true)
                    wf_reg_iters_trial.append(iters_wf_reg)
                    wf_reg_accs_trial.extend(acc_wf_reg.cpu().numpy())
                    wf_reg_focus_trial.append(focus_wf_reg.cpu().numpy().mean())
                    wf_reg_times_trial.append(t1 - t0)
                # 运行 GGS 1
                if ENABLE.get('ggs1', False):
                    t0 = time.perf_counter()
                    iters_ggs1, X_final_ggs1 = GGS1(P, I, iters=MAX_ITERATIONS, step=0.9, init=a0)
                    t1 = time.perf_counter()
                    X_align_ggs1 = align_global_phase(X_final_ggs1, X_true)
                    acc_ggs1 = correlation_coefficient(X_align_ggs1, X_true)
                    focus_ggs1 = phase_conjugation_contrast(X_align_ggs1, X_true)
                
                    ggs1_iters_trial.append(iters_ggs1)
                    ggs1_accs_trial.extend(acc_ggs1.cpu().numpy())
                    ggs1_focus_trial.append(focus_ggs1.cpu().numpy().mean())
                    ggs1_times_trial.append(t1 - t0)
                
                # 运行 GGS 2-1
                if ENABLE.get('ggs2_1', False):
                    t0 = time.perf_counter()
                    iters_ggs2_1, X_final_ggs2_1 = GGS2_1(P, I, iters=MAX_ITERATIONS, step=0.9, init=a0)
                    t1 = time.perf_counter()
                    X_align_ggs2_1 = align_global_phase(X_final_ggs2_1, X_true)
                    acc_ggs2_1 = correlation_coefficient(X_align_ggs2_1, X_true)
                    focus_ggs2_1 = phase_conjugation_contrast(X_align_ggs2_1, X_true)
                    ggs2_1_iters_trial.append(iters_ggs2_1)
                    ggs2_1_accs_trial.extend(acc_ggs2_1.cpu().numpy())
                    ggs2_1_focus_trial.append(focus_ggs2_1.cpu().numpy().mean())
                    ggs2_1_times_trial.append(t1 - t0)
                # 运行PR-VBEM（批处理：I 形状为 (N_ROWS, M)）
                if ENABLE.get('pr_vbem', False):
                    t0 = time.perf_counter()
                    opts = {
                        # 'var_a': 1e-3,
                        # 'var_n': 1e-8,
                        'x0': a0,
                        # 'pas_est': 0.1,
                        # 'flag_est_n': 'on',
                        # 'flag_cv': 'KL',
                        'niter': MAX_ITERATIONS,
                    }
                    # opts = None
                    X_final_pr_vbem, iters_vbem = pr_vbem(I, P, opt=opts)
                    t1 = time.perf_counter()
                    X_align_pr_vbem = align_global_phase(X_final_pr_vbem, X_true).to(torch.complex64).to(device=device)
                    acc_pr_vbem = correlation_coefficient(X_align_pr_vbem, X_true)
                    focus_pr_vbem = phase_conjugation_contrast(X_align_pr_vbem, X_true)
                    pr_vbem_iters_trial.append(iters_vbem)
                    pr_vbem_accs_trial.extend(acc_pr_vbem.cpu().numpy())
                    pr_vbem_focus_trial.append(focus_pr_vbem.cpu().numpy().mean())
                    pr_vbem_times_trial.append(t1 - t0)
                
                # 运行 RAF21-TM
                if ENABLE.get('raf21_tm', False):
                    t0 = time.perf_counter()
                    iters_raf21_tm, X_final_raf21_tm = raf21_tm(P, I, iters=MAX_ITERATIONS, 
                                                                step=0.8, init=a0, ratio=2/3, 
                                                                 device=device)
                    t1 = time.perf_counter()
                    X_align_raf21_tm = align_global_phase(X_final_raf21_tm, X_true)
                    acc_raf21_tm = correlation_coefficient(X_align_raf21_tm, X_true)
                    focus_raf21_tm = phase_conjugation_contrast(X_align_raf21_tm, X_true)
                    raf21_tm_iters_trial.append(iters_raf21_tm)
                    raf21_tm_accs_trial.extend(acc_raf21_tm.cpu().numpy())
                    raf21_tm_focus_trial.append(focus_raf21_tm.cpu().numpy().mean())
                    raf21_tm_times_trial.append(t1 - t0)
                # 运行 Autograd Adam 优化器
                if ENABLE.get('autograd', False):
                    t0 = time.perf_counter()
                    iters_auto, X_final_auto = autograd_amp_solver(P, I, iters=MAX_ITERATIONS, lr=1e-2, init=a0, l1=0.0, l2=0.0, device=device)
                    t1 = time.perf_counter()
                    X_align_auto = align_global_phase(X_final_auto, X_true)
                    acc_auto = correlation_coefficient(X_align_auto, X_true)
                    focus_auto = phase_conjugation_contrast(X_align_auto, X_true)
                    autograd_iters_trial.append(iters_auto)
                    autograd_accs_trial.extend(acc_auto.cpu().numpy())
                    autograd_focus_trial.append(focus_auto.cpu().numpy().mean())
                    autograd_times_trial.append(t1 - t0)
                # 运行 RAF21-TM Autograd 版本
                if ENABLE.get('raf21_tm_auto', False):
                    t0 = time.perf_counter()
                    iters_raf_auto, X_final_raf_auto = autograd_raf_solver(P, I, iters=MAX_ITERATIONS, lr=1e-2, init=a0, ratio=2/3, beta=5.0, device=device)
                    t1 = time.perf_counter()
                    X_align_raf_auto = align_global_phase(X_final_raf_auto, X_true)
                    acc_raf_auto = correlation_coefficient(X_align_raf_auto, X_true)
                    focus_raf_auto = phase_conjugation_contrast(X_align_raf_auto, X_true)
                    raf21_tm_auto_iters_trial.append(iters_raf_auto)
                    raf21_tm_auto_accs_trial.extend(acc_raf_auto.cpu().numpy())
                    raf21_tm_auto_focus_trial.append(focus_raf_auto.cpu().numpy().mean())
                    raf21_tm_auto_times_trial.append(t1 - t0)
                # 运行 WF Autograd 版本
                if ENABLE.get('wf_auto', False):
                    t0 = time.perf_counter()
                    iters_wf_auto, X_final_wf_auto = autograd_wf_solver(P, I, iters=MAX_ITERATIONS, lr=1e-2, init=a0, device=device)
                    t1 = time.perf_counter()
                    X_align_wf_auto = align_global_phase(X_final_wf_auto, X_true)
                    acc_wf_auto = correlation_coefficient(X_align_wf_auto, X_true)
                    focus_wf_auto = phase_conjugation_contrast(X_align_wf_auto, X_true)
                    wf_auto_iters_trial.append(iters_wf_auto)
                    wf_auto_accs_trial.extend(acc_wf_auto.cpu().numpy())
                    wf_auto_focus_trial.append(focus_wf_auto.cpu().numpy().mean())
                    wf_auto_times_trial.append(t1 - t0)
                # 运行 WF-Reg Autograd 版本
                if ENABLE.get('wf_reg_auto', False):
                    t0 = time.perf_counter()
                    iters_wf_reg_auto, X_final_wf_reg_auto = autograd_wf_reg_solver(P, I, iters=MAX_ITERATIONS, lr=1e-2, lambda_reg=LAMBDA, init=a0, device=device)
                    t1 = time.perf_counter()
                    X_align_wf_reg_auto = align_global_phase(X_final_wf_reg_auto, X_true)
                    acc_wf_reg_auto = correlation_coefficient(X_align_wf_reg_auto, X_true)
                    focus_wf_reg_auto = phase_conjugation_contrast(X_align_wf_reg_auto, X_true)
                    wf_reg_auto_iters_trial.append(iters_wf_reg_auto)
                    wf_reg_auto_accs_trial.extend(acc_wf_reg_auto.cpu().numpy())
                    wf_reg_auto_focus_trial.append(focus_wf_reg_auto.cpu().numpy().mean())
                    wf_reg_auto_times_trial.append(t1 - t0)
                # 运行 GGS1 Autograd 版本
                if ENABLE.get('ggs1_auto', False):
                    t0 = time.perf_counter()
                    iters_ggs1_auto, X_final_ggs1_auto = autograd_ggs1_solver(P, I, iters=MAX_ITERATIONS, lr=1e-2, init=a0, device=device)
                    t1 = time.perf_counter()
                    X_align_ggs1_auto = align_global_phase(X_final_ggs1_auto, X_true)
                    acc_ggs1_auto = correlation_coefficient(X_align_ggs1_auto, X_true)
                    focus_ggs1_auto = phase_conjugation_contrast(X_align_ggs1_auto, X_true)
                    ggs1_auto_iters_trial.append(iters_ggs1_auto)
                    ggs1_auto_accs_trial.extend(acc_ggs1_auto.cpu().numpy())
                    ggs1_auto_focus_trial.append(focus_ggs1_auto.cpu().numpy().mean())
                    ggs1_auto_times_trial.append(t1 - t0)
                # 运行 GGS2-1 Autograd 版本
                if ENABLE.get('ggs2_1_auto', False):
                    t0 = time.perf_counter()
                    iters_ggs2_1_auto, X_final_ggs2_1_auto = autograd_ggs2_1_solver(P, I, iters=MAX_ITERATIONS, lr=1e-2, init=a0, device=device)
                    t1 = time.perf_counter()
                    X_align_ggs2_1_auto = align_global_phase(X_final_ggs2_1_auto, X_true)
                    acc_ggs2_1_auto = correlation_coefficient(X_align_ggs2_1_auto, X_true)
                    focus_ggs2_1_auto = phase_conjugation_contrast(X_align_ggs2_1_auto, X_true)
                    ggs2_1_auto_iters_trial.append(iters_ggs2_1_auto)
                    ggs2_1_auto_accs_trial.extend(acc_ggs2_1_auto.cpu().numpy())
                    ggs2_1_auto_focus_trial.append(focus_ggs2_1_auto.cpu().numpy().mean())
                    ggs2_1_auto_times_trial.append(t1 - t0)
                

                

        if ENABLE.get('wf', False):
            results['wf']['iters'].append(wf_iters_trial)
            results['wf']['accs'].append(wf_accs_trial)
            results['focus']['wf'].append(wf_focus_trial)
        if ENABLE.get('wf_reg', False):
            results['wf_reg']['iters'].append(wf_reg_iters_trial)
            results['wf_reg']['accs'].append(wf_reg_accs_trial)
            results['focus']['wf_reg'].append(wf_reg_focus_trial)
        if ENABLE.get('ggs1', False):
            results['ggs1']['iters'].append(ggs1_iters_trial)
            results['ggs1']['accs'].append(ggs1_accs_trial)
            results['focus']['ggs1'].append(ggs1_focus_trial)
        if ENABLE.get('ggs2_1', False):
            results['ggs2_1']['iters'].append(ggs2_1_iters_trial)
            results['ggs2_1']['accs'].append(ggs2_1_accs_trial)
            results['focus']['ggs2_1'].append(ggs2_1_focus_trial)
        if ENABLE.get('fista', False):
            results['fista']['iters'].append(fista_iters_trial)
            results['fista']['accs'].append(fista_accs_trial)
            results['focus']['fista'].append(fista_focus_trial)
        if ENABLE.get('pr_vbem', False):
            results['pr_vbem']['iters'].append(pr_vbem_iters_trial)
            results['pr_vbem']['accs'].append(pr_vbem_accs_trial)
            results['focus']['pr_vbem'].append(pr_vbem_focus_trial)
        if ENABLE.get('raf21_tm', False):
            results['raf21_tm']['iters'].append(raf21_tm_iters_trial)
            results['raf21_tm']['accs'].append(raf21_tm_accs_trial)
            results['focus']['raf21_tm'].append(raf21_tm_focus_trial)
        if ENABLE.get('autograd', False):
            results['autograd']['iters'].append(autograd_iters_trial)
            results['autograd']['accs'].append(autograd_accs_trial)
            results['focus']['autograd'].append(autograd_focus_trial)
        if ENABLE.get('raf21_tm_auto', False):
            results['raf21_tm_auto']['iters'].append(raf21_tm_auto_iters_trial)
            results['raf21_tm_auto']['accs'].append(raf21_tm_auto_accs_trial)
            # 共享 focus 字典，添加新键
            if 'raf21_tm_auto' not in results['focus']:
                results['focus']['raf21_tm_auto'] = []
            results['focus']['raf21_tm_auto'].append(raf21_tm_auto_focus_trial)
        if ENABLE.get('wf_auto', False):
            results['wf_auto']['iters'].append(wf_auto_iters_trial)
            results['wf_auto']['accs'].append(wf_auto_accs_trial)
            if 'wf_auto' not in results['focus']:
                results['focus']['wf_auto'] = []
            results['focus']['wf_auto'].append(wf_auto_focus_trial)
        if ENABLE.get('wf_reg_auto', False):
            results['wf_reg_auto']['iters'].append(wf_reg_auto_iters_trial)
            results['wf_reg_auto']['accs'].append(wf_reg_auto_accs_trial)
            if 'wf_reg_auto' not in results['focus']:
                results['focus']['wf_reg_auto'] = []
            results['focus']['wf_reg_auto'].append(wf_reg_auto_focus_trial)
        if ENABLE.get('ggs1_auto', False):
            results['ggs1_auto']['iters'].append(ggs1_auto_iters_trial)
            results['ggs1_auto']['accs'].append(ggs1_auto_accs_trial)
            if 'ggs1_auto' not in results['focus']:
                results['focus']['ggs1_auto'] = []
            results['focus']['ggs1_auto'].append(ggs1_auto_focus_trial)
        if ENABLE.get('ggs2_1_auto', False):
            results['ggs2_1_auto']['iters'].append(ggs2_1_auto_iters_trial)
            results['ggs2_1_auto']['accs'].append(ggs2_1_auto_accs_trial)
            if 'ggs2_1_auto' not in results['focus']:
                results['focus']['ggs2_1_auto'] = []
            results['focus']['ggs2_1_auto'].append(ggs2_1_auto_focus_trial)

    return results


# --- 5. 结果可视化 ---

def plot_results(results):
    """
    绘制仿真结果。
    """
    # 提取无噪声结果（按开关与可用性安全获取）
    noise_free_wf_iters = results['wf']['iters'][0] if ENABLE.get('wf', False) and len(results['wf']['iters']) > 0 else None
    noise_free_wf_accs = results['wf']['accs'][0] if ENABLE.get('wf', False) and len(results['wf']['accs']) > 0 else None
    noise_free_wf_reg_iters = results['wf_reg']['iters'][0] if ENABLE.get('wf_reg', False) and len(results['wf_reg']['iters']) > 0 else None
    noise_free_wf_reg_accs = results['wf_reg']['accs'][0] if ENABLE.get('wf_reg', False) and len(results['wf_reg']['accs']) > 0 else None
    noise_free_ggs1_iters = results['ggs1']['iters'][0] if ENABLE.get('ggs1', False) and len(results['ggs1']['iters']) > 0 else None
    noise_free_ggs1_accs = results['ggs1']['accs'][0] if ENABLE.get('ggs1', False) and len(results['ggs1']['accs']) > 0 else None
    noise_free_ggs2_1_iters = results['ggs2_1']['iters'][0] if ENABLE.get('ggs2_1', False) and len(results['ggs2_1']['iters']) > 0 else None
    noise_free_ggs2_1_accs = results['ggs2_1']['accs'][0] if ENABLE.get('ggs2_1', False) and len(results['ggs2_1']['accs']) > 0 else None
    noise_free_fista_iters = results['fista']['iters'][0] if ENABLE.get('fista', False) and len(results['fista']['iters']) > 0 else None
    noise_free_fista_accs = results['fista']['accs'][0] if ENABLE.get('fista', False) and len(results['fista']['accs']) > 0 else None
    noise_free_pr_vbem_iters = results['pr_vbem']['iters'][0] if ENABLE.get('pr_vbem', False) and len(results['pr_vbem']['iters']) > 0 else None
    noise_free_pr_vbem_accs = results['pr_vbem']['accs'][0] if ENABLE.get('pr_vbem', False) and len(results['pr_vbem']['accs']) > 0 else None
    noise_free_raf21_tm_iters = results['raf21_tm']['iters'][0] if ENABLE.get('raf21_tm', False) and len(results['raf21_tm']['iters']) > 0 else None
    noise_free_raf21_tm_accs = results['raf21_tm']['accs'][0] if ENABLE.get('raf21_tm', False) and len(results['raf21_tm']['accs']) > 0 else None
    noise_free_raf21_tm_auto_iters = results['raf21_tm_auto']['iters'][0] if ENABLE.get('raf21_tm_auto', False) and len(results['raf21_tm_auto']['iters']) > 0 else None
    noise_free_raf21_tm_auto_accs = results['raf21_tm_auto']['accs'][0] if ENABLE.get('raf21_tm_auto', False) and len(results['raf21_tm_auto']['accs']) > 0 else None
    noise_free_autograd_iters = results['autograd']['iters'][0] if ENABLE.get('autograd', False) and len(results['autograd']['iters']) > 0 else None
    noise_free_autograd_accs = results['autograd']['accs'][0] if ENABLE.get('autograd', False) and len(results['autograd']['accs']) > 0 else None

    trials_axis = np.arange(1, NUM_TRIALS + 1)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Comparison: Regularized vs. Standard Wirtinger Flow (PyTorch/GPU)', fontsize=16)

    # --- Fig 2a: 无噪声收敛效率 ---
    ax = axs[0, 0]
    if ENABLE.get('wf_reg', False):
        ax.plot(trials_axis, noise_free_wf_reg_iters, 'r-', label='WF with Regularization')
        ax.axhline(np.mean(noise_free_wf_reg_iters), color='r', linestyle=':', 
                   label=f'Mean: {np.mean(noise_free_wf_reg_iters):.1f}')
    if ENABLE.get('wf', False):
        ax.plot(trials_axis, noise_free_wf_iters, 'b-', label='Standard Wirtinger Flow')
        ax.axhline(np.mean(noise_free_wf_iters), color='b', linestyle=':', 
                   label=f'Mean: {np.mean(noise_free_wf_iters):.1f}')
    if ENABLE.get('ggs1', False):
        ax.plot(trials_axis, noise_free_ggs1_iters, 'm-', label='GGS 1')
        ax.axhline(np.mean(noise_free_ggs1_iters), color='m', linestyle=':', 
                   label=f'Mean: {np.mean(noise_free_ggs1_iters):.1f}')
    if ENABLE.get('ggs2_1', False):
        ax.plot(trials_axis, noise_free_ggs2_1_iters, 'g-', label='GGS 2-1')
        ax.axhline(np.mean(noise_free_ggs2_1_iters), color='g', linestyle=':', 
                   label=f'Mean: {np.mean(noise_free_ggs2_1_iters):.1f}')
    if ENABLE.get('fista', False):
        ax.plot(trials_axis, noise_free_fista_iters, 'c-', label='FISTA')
        ax.axhline(np.mean(noise_free_fista_iters), color='c', linestyle=':', 
                   label=f'Mean: {np.mean(noise_free_fista_iters):.1f}')
    if ENABLE.get('pr_vbem', False):
        ax.plot(trials_axis, noise_free_pr_vbem_iters, 'y-', label='PR-VBEM')
        ax.axhline(np.mean(noise_free_pr_vbem_iters), color='y', linestyle=':', 
                   label=f'Mean: {np.mean(noise_free_pr_vbem_iters):.1f}')
    if ENABLE.get('raf21_tm', False):
        ax.plot(trials_axis, noise_free_raf21_tm_iters, 'orange', label='RAF21-TM')
        ax.axhline(np.mean(noise_free_raf21_tm_iters), color='orange', linestyle=':', 
                   label=f'Mean: {np.mean(noise_free_raf21_tm_iters):.1f}')
    if ENABLE.get('raf21_tm_auto', False):
        ax.plot(trials_axis, noise_free_raf21_tm_auto_iters, 'brown', label='RAF21-TM-Auto')
        ax.axhline(np.mean(noise_free_raf21_tm_auto_iters), color='brown', linestyle=':', 
                   label=f'Mean: {np.mean(noise_free_raf21_tm_auto_iters):.1f}')
    if ENABLE.get('autograd', False):
        ax.plot(trials_axis, noise_free_autograd_iters, 'k-', label='Autograd-Adam')
        ax.axhline(np.mean(noise_free_autograd_iters), color='k', linestyle=':', 
                   label=f'Mean: {np.mean(noise_free_autograd_iters):.1f}')
    ax.set_title('(a) Convergence efficiency (Noise-free)')
    ax.set_xlabel('Trial index')
    ax.set_ylabel('Iterations')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- Fig 2b: 无噪声精度 (折线图) ---
    # 将所有精度结果按 trial 分组，并计算每个 trial 的平均值
    accs_wf_reg_per_trial = np.mean(np.array(noise_free_wf_reg_accs).reshape(NUM_TRIALS, N_ROWS), axis=1) if ENABLE.get('wf_reg', False) else None
    accs_wf_per_trial = np.mean(np.array(noise_free_wf_accs).reshape(NUM_TRIALS, N_ROWS), axis=1) if ENABLE.get('wf', False) else None
    accs_ggs1_per_trial = np.mean(np.array(noise_free_ggs1_accs).reshape(NUM_TRIALS, N_ROWS), axis=1) if ENABLE.get('ggs1', False) else None
    accs_ggs2_1_per_trial = np.mean(np.array(noise_free_ggs2_1_accs).reshape(NUM_TRIALS, N_ROWS), axis=1) if ENABLE.get('ggs2_1', False) else None
    accs_fista_per_trial = np.mean(np.array(noise_free_fista_accs).reshape(NUM_TRIALS, N_ROWS), axis=1) if ENABLE.get('fista', False) else None
    accs_pr_vbem_per_trial = np.mean(np.array(noise_free_pr_vbem_accs).reshape(NUM_TRIALS, N_ROWS), axis=1) if ENABLE.get('pr_vbem', False) else None
    accs_raf21_tm_per_trial = np.mean(np.array(noise_free_raf21_tm_accs).reshape(NUM_TRIALS, N_ROWS), axis=1) if ENABLE.get('raf21_tm', False) else None
    accs_raf21_tm_auto_per_trial = np.mean(np.array(noise_free_raf21_tm_auto_accs).reshape(NUM_TRIALS, N_ROWS), axis=1) if ENABLE.get('raf21_tm_auto', False) else None
    accs_autograd_per_trial = np.mean(np.array(noise_free_autograd_accs).reshape(NUM_TRIALS, N_ROWS), axis=1) if ENABLE.get('autograd', False) else None

    ax = axs[0, 1]
    if ENABLE.get('wf_reg', False):
        ax.plot(trials_axis, accs_wf_reg_per_trial, 'r-', label='WF with Regularization')
        ax.axhline(np.mean(accs_wf_reg_per_trial), color='r', linestyle=':',
                   label=f'Mean: {np.mean(accs_wf_reg_per_trial):.4f}')
    if ENABLE.get('wf', False):
        ax.plot(trials_axis, accs_wf_per_trial, 'b-', label='Standard Wirtinger Flow')
        ax.axhline(np.mean(accs_wf_per_trial), color='b', linestyle=':',
                   label=f'Mean: {np.mean(accs_wf_per_trial):.4f}')
    if ENABLE.get('ggs1', False):
        ax.plot(trials_axis, accs_ggs1_per_trial, 'm-', label='GGS 1')
        ax.axhline(np.mean(accs_ggs1_per_trial), color='m', linestyle=':',
                   label=f'Mean: {np.mean(accs_ggs1_per_trial):.4f}')
    if ENABLE.get('ggs2_1', False):
        ax.plot(trials_axis, accs_ggs2_1_per_trial, 'g-', label='GGS 2-1')
        ax.axhline(np.mean(accs_ggs2_1_per_trial), color='g', linestyle=':',
                   label=f'Mean: {np.mean(accs_ggs2_1_per_trial):.4f}')
    if ENABLE.get('fista', False):
        ax.plot(trials_axis, accs_fista_per_trial, 'c-', label='FISTA')
        ax.axhline(np.mean(accs_fista_per_trial), color='c', linestyle=':',
                   label=f'Mean: {np.mean(accs_fista_per_trial):.4f}')
    if ENABLE.get('pr_vbem', False):
        ax.plot(trials_axis, accs_pr_vbem_per_trial, 'y-', label='PR-VBEM')
        ax.axhline(np.mean(accs_pr_vbem_per_trial), color='y', linestyle=':',
                   label=f'Mean: {np.mean(accs_pr_vbem_per_trial):.4f}')
    if ENABLE.get('raf21_tm', False):
        ax.plot(trials_axis, accs_raf21_tm_per_trial, 'orange', label='RAF21-TM')
        ax.axhline(np.mean(accs_raf21_tm_per_trial), color='orange', linestyle=':',
                   label=f'Mean: {np.mean(accs_raf21_tm_per_trial):.4f}')
    if ENABLE.get('raf21_tm_auto', False):
        ax.plot(trials_axis, accs_raf21_tm_auto_per_trial, 'brown', label='RAF21-TM-Auto')
        ax.axhline(np.mean(accs_raf21_tm_auto_per_trial), color='brown', linestyle=':',
                   label=f'Mean: {np.mean(accs_raf21_tm_auto_per_trial):.4f}')
    if ENABLE.get('autograd', False):
        ax.plot(trials_axis, accs_autograd_per_trial, 'k-', label='Autograd-Adam')
        ax.axhline(np.mean(accs_autograd_per_trial), color='k', linestyle=':',
                   label=f'Mean: {np.mean(accs_autograd_per_trial):.4f}')
    ax.set_title('(b) Accuracy (Noise-free)')
    ax.set_xlabel('Trial index')
    ax.set_ylabel('Average correlation coefficient')
    # ax.set_ylim(0.4, 1)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- 提取带噪声的结果 ---
    snrs_with_noise = results['snrs'][1:]
    
    wf_iters_mean = [np.mean(iters) for iters in results['wf']['iters'][1:]] if ENABLE.get('wf', False) else []
    wf_iters_std = [np.std(iters) for iters in results['wf']['iters'][1:]] if ENABLE.get('wf', False) else []
    wf_reg_iters_mean = [np.mean(iters) for iters in results['wf_reg']['iters'][1:]] if ENABLE.get('wf_reg', False) else []
    wf_reg_iters_std = [np.std(iters) for iters in results['wf_reg']['iters'][1:]] if ENABLE.get('wf_reg', False) else []
    ggs1_iters_mean = [np.mean(iters) for iters in results['ggs1']['iters'][1:]] if ENABLE.get('ggs1', False) else []
    ggs1_iters_std = [np.std(iters) for iters in results['ggs1']['iters'][1:]] if ENABLE.get('ggs1', False) else []
    ggs2_1_iters_mean = [np.mean(iters) for iters in results['ggs2_1']['iters'][1:]] if ENABLE.get('ggs2_1', False) else []
    ggs2_1_iters_std = [np.std(iters) for iters in results['ggs2_1']['iters'][1:]] if ENABLE.get('ggs2_1', False) else []
    fista_iters_mean = [np.mean(iters) for iters in results['fista']['iters'][1:]] if ENABLE.get('fista', False) else []
    fista_iters_std = [np.std(iters) for iters in results['fista']['iters'][1:]] if ENABLE.get('fista', False) else []
    pr_vbem_iters_mean = [np.mean(iters) for iters in results['pr_vbem']['iters'][1:]] if ENABLE.get('pr_vbem', False) else []
    pr_vbem_iters_std = [np.std(iters) for iters in results['pr_vbem']['iters'][1:]] if ENABLE.get('pr_vbem', False) else []
    raf21_tm_iters_mean = [np.mean(iters) for iters in results['raf21_tm']['iters'][1:]] if ENABLE.get('raf21_tm', False) else []
    raf21_tm_iters_std = [np.std(iters) for iters in results['raf21_tm']['iters'][1:]] if ENABLE.get('raf21_tm', False) else []
    raf21_tm_auto_iters_mean = [np.mean(iters) for iters in results['raf21_tm_auto']['iters'][1:]] if ENABLE.get('raf21_tm_auto', False) else []
    raf21_tm_auto_iters_std = [np.std(iters) for iters in results['raf21_tm_auto']['iters'][1:]] if ENABLE.get('raf21_tm_auto', False) else []
    autograd_iters_mean = [np.mean(iters) for iters in results['autograd']['iters'][1:]] if ENABLE.get('autograd', False) else []
    autograd_iters_std = [np.std(iters) for iters in results['autograd']['iters'][1:]] if ENABLE.get('autograd', False) else []

    wf_accs_mean = [np.mean(accs) for accs in results['wf']['accs'][1:]] if ENABLE.get('wf', False) else []
    wf_accs_std = [np.std(accs) for accs in results['wf']['accs'][1:]] if ENABLE.get('wf', False) else []
    wf_reg_accs_mean = [np.mean(accs) for accs in results['wf_reg']['accs'][1:]] if ENABLE.get('wf_reg', False) else []
    wf_reg_accs_std = [np.std(accs) for accs in results['wf_reg']['accs'][1:]] if ENABLE.get('wf_reg', False) else []
    ggs1_accs_mean = [np.mean(accs) for accs in results['ggs1']['accs'][1:]] if ENABLE.get('ggs1', False) else []
    ggs1_accs_std = [np.std(accs) for accs in results['ggs1']['accs'][1:]] if ENABLE.get('ggs1', False) else []
    ggs2_1_accs_mean = [np.mean(accs) for accs in results['ggs2_1']['accs'][1:]] if ENABLE.get('ggs2_1', False) else []
    ggs2_1_accs_std = [np.std(accs) for accs in results['ggs2_1']['accs'][1:]] if ENABLE.get('ggs2_1', False) else []
    fista_accs_mean = [np.mean(accs) for accs in results['fista']['accs'][1:]] if ENABLE.get('fista', False) else []
    fista_accs_std = [np.std(accs) for accs in results['fista']['accs'][1:]] if ENABLE.get('fista', False) else []
    pr_vbem_accs_mean = [np.mean(accs) for accs in results['pr_vbem']['accs'][1:]] if ENABLE.get('pr_vbem', False) else []
    pr_vbem_accs_std = [np.std(accs) for accs in results['pr_vbem']['accs'][1:]] if ENABLE.get('pr_vbem', False) else []
    raf21_tm_accs_mean = [np.mean(accs) for accs in results['raf21_tm']['accs'][1:]] if ENABLE.get('raf21_tm', False) else []
    raf21_tm_accs_std = [np.std(accs) for accs in results['raf21_tm']['accs'][1:]] if ENABLE.get('raf21_tm', False) else []
    raf21_tm_auto_accs_mean = [np.mean(accs) for accs in results['raf21_tm_auto']['accs'][1:]] if ENABLE.get('raf21_tm_auto', False) else []
    raf21_tm_auto_accs_std = [np.std(accs) for accs in results['raf21_tm_auto']['accs'][1:]] if ENABLE.get('raf21_tm_auto', False) else []
    autograd_accs_mean = [np.mean(accs) for accs in results['autograd']['accs'][1:]] if ENABLE.get('autograd', False) else []
    autograd_accs_std = [np.std(accs) for accs in results['autograd']['accs'][1:]] if ENABLE.get('autograd', False) else []

    # --- Fig 2c: 不同信噪比下的收敛效率 ---
    ax = axs[1, 0]
    if ENABLE.get('wf_reg', False):
        ax.errorbar(snrs_with_noise, wf_reg_iters_mean, yerr=wf_reg_iters_std, fmt='-o', 
                    color='r', label='WF with Regularization', capsize=5)
    if ENABLE.get('wf', False):
        ax.errorbar(snrs_with_noise, wf_iters_mean, yerr=wf_iters_std, fmt='-s', 
                    color='b', label='Standard Wirtinger Flow', capsize=5)
    if ENABLE.get('ggs1', False):
        ax.errorbar(snrs_with_noise, ggs1_iters_mean, yerr=ggs1_iters_std, fmt='-^', 
                    color='m', label='GGS 1', capsize=5)
    if ENABLE.get('ggs2_1', False):
        ax.errorbar(snrs_with_noise, ggs2_1_iters_mean, yerr=ggs2_1_iters_std, fmt='-d', 
                    color='g', label='GGS 2-1', capsize=5)
    if ENABLE.get('fista', False):
        ax.errorbar(snrs_with_noise, fista_iters_mean, yerr=fista_iters_std, fmt='-x', 
                    color='c', label='FISTA', capsize=5)
    if ENABLE.get('pr_vbem', False):
        ax.errorbar(snrs_with_noise, pr_vbem_iters_mean, yerr=pr_vbem_iters_std, fmt='-p', 
                    color='y', label='PR-VBEM', capsize=5)
    if ENABLE.get('raf21_tm', False):
        ax.errorbar(snrs_with_noise, raf21_tm_iters_mean, yerr=raf21_tm_iters_std, fmt='-*', 
                    color='orange', label='RAF21-TM', capsize=5)
    if ENABLE.get('raf21_tm_auto', False):
        ax.errorbar(snrs_with_noise, raf21_tm_auto_iters_mean, yerr=raf21_tm_auto_iters_std, fmt='-o', 
                    color='brown', label='RAF21-TM-Auto', capsize=5)
    if ENABLE.get('autograd', False):
        ax.errorbar(snrs_with_noise, autograd_iters_mean, yerr=autograd_iters_std, fmt='-o', 
                    color='k', label='Autograd-Adam', capsize=5)
    ax.set_title('(c) Convergence efficiency under different SNRs')
    ax.set_xlabel('SNR')
    ax.set_ylabel('Average iterations')
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- Fig 2d: 不同信噪比下的精度 ---
    ax = axs[1, 1]
    if ENABLE.get('wf_reg', False):
        ax.errorbar(snrs_with_noise, wf_reg_accs_mean, yerr=wf_reg_accs_std, fmt='-o', 
                    color='r', label='WF with Regularization', capsize=5)
    if ENABLE.get('wf', False):
        ax.errorbar(snrs_with_noise, wf_accs_mean, yerr=wf_accs_std, fmt='-s', 
                    color='b', label='Standard Wirtinger Flow', capsize=5)
    if ENABLE.get('ggs1', False):
        ax.errorbar(snrs_with_noise, ggs1_accs_mean, yerr=ggs1_accs_std, fmt='-^', 
                    color='m', label='GGS 1', capsize=5)
    if ENABLE.get('ggs2_1', False):
        ax.errorbar(snrs_with_noise, ggs2_1_accs_mean, yerr=ggs2_1_accs_std, fmt='-d', 
                    color='g', label='GGS 2-1', capsize=5)
    if ENABLE.get('fista', False):
        ax.errorbar(snrs_with_noise, fista_accs_mean, yerr=fista_accs_std, fmt='-x', 
                    color='c', label='FISTA', capsize=5)
    if ENABLE.get('pr_vbem', False):
        ax.errorbar(snrs_with_noise, pr_vbem_accs_mean, yerr=pr_vbem_accs_std, fmt='-p', 
                    color='y', label='PR-VBEM', capsize=5)
    if ENABLE.get('raf21_tm', False):
        ax.errorbar(snrs_with_noise, raf21_tm_accs_mean, yerr=raf21_tm_accs_std, fmt='-*', 
                    color='orange', label='RAF21-TM', capsize=5)  
    if ENABLE.get('raf21_tm_auto', False):
        ax.errorbar(snrs_with_noise, raf21_tm_auto_accs_mean, yerr=raf21_tm_auto_accs_std, fmt='-o', 
                    color='brown', label='RAF21-TM-Auto', capsize=5)
    if ENABLE.get('autograd', False):
        ax.errorbar(snrs_with_noise, autograd_accs_mean, yerr=autograd_accs_std, fmt='-o', 
                    color='k', label='Autograd-Adam', capsize=5)
    ax.set_title('(d) Accuracy under different SNRs')
    ax.set_xlabel('SNR')
    ax.set_ylabel('Average correlation coefficient')
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('results.png')


def plot_focus(results):
    """
    绘制相位共轭聚焦对比度：
    (1) 无噪声下每个 trial 的聚焦对比度曲线
    (2) 不同 SNR 下的平均±标准差
    """
    trials_axis = np.arange(1, NUM_TRIALS + 1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Phase Conjugation Focusing Contrast')

    # 无噪声（SNRS[0]）每 trial 对比度
    ax = axs[0]
    ax.plot(trials_axis, results['focus']['wf'][0][0:NUM_TRIALS], 'b-', label=f'WF: ') if ENABLE.get('wf', False) else None
    ax.plot(trials_axis, results['focus']['wf_reg'][0][0:NUM_TRIALS], 'r-', label=f'WF-Reg: ') if ENABLE.get('wf_reg', False) else None
    ax.plot(trials_axis, results['focus']['ggs1'][0][0:NUM_TRIALS], 'm-', label=f'GGS1: ') if ENABLE.get('ggs1', False) else None
    ax.plot(trials_axis, results['focus']['ggs2_1'][0][0:NUM_TRIALS], 'g-', label=f'GGS2-1: ') if ENABLE.get('ggs2_1', False) else None
    ax.plot(trials_axis, results['focus']['fista'][0][0:NUM_TRIALS], 'c-', label=f'FISTA: ') if ENABLE.get('fista', False) else None
    ax.plot(trials_axis, results['focus']['pr_vbem'][0][0:NUM_TRIALS], 'y-', label=f'PR-VBEM: ') if ENABLE.get('pr_vbem', False) else None
    ax.plot(trials_axis, results['focus']['raf21_tm'][0][0:NUM_TRIALS], 'orange', label=f'RAF21-TM: ') if ENABLE.get('raf21_tm', False) else None
    ax.plot(trials_axis, results['focus']['autograd'][0][0:NUM_TRIALS], 'k-', label=f'Autograd-Adam: ') if ENABLE.get('autograd', False) else None
    # 绘制平均值
    ax.axhline(np.mean(results['focus']['wf'][0][0:NUM_TRIALS]), color='b', linestyle=':') if ENABLE.get('wf', False) else None
    ax.axhline(np.mean(results['focus']['wf_reg'][0][0:NUM_TRIALS]), color='r', linestyle=':') if ENABLE.get('wf_reg', False) else None
    ax.axhline(np.mean(results['focus']['ggs1'][0][0:NUM_TRIALS]), color='m', linestyle=':') if ENABLE.get('ggs1', False) else None
    ax.axhline(np.mean(results['focus']['ggs2_1'][0][0:NUM_TRIALS]), color='g', linestyle=':') if ENABLE.get('ggs2_1', False) else None
    ax.axhline(np.mean(results['focus']['fista'][0][0:NUM_TRIALS]), color='c', linestyle=':') if ENABLE.get('fista', False) else None
    ax.axhline(np.mean(results['focus']['pr_vbem'][0][0:NUM_TRIALS]), color='y', linestyle=':') if ENABLE.get('pr_vbem', False) else None
    ax.axhline(np.mean(results['focus']['raf21_tm'][0][0:NUM_TRIALS]), color='orange', linestyle=':') if ENABLE.get('raf21_tm', False) else None
    ax.axhline(np.mean(results['focus']['autograd'][0][0:NUM_TRIALS]), color='k', linestyle=':') if ENABLE.get('autograd', False) else None
    ax.set_title('(a) Contrast per trial (Noise-free)')
    ax.set_xlabel('Trial index')
    ax.set_ylabel('Contrast')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # 不同 SNR 的均值±标准差
    ax = axs[1]
    snrs_with_noise = results['snrs'][1:]
    def stats(x):
        xs = np.array(x).reshape(len(SNRS), NUM_TRIALS)[1:]
        return xs.mean(axis=1), xs.std(axis=1)
    wf_mean, wf_std = stats(results['focus']['wf']) if ENABLE.get('wf', False) else (None, None)
    wf_reg_mean, wf_reg_std = stats(results['focus']['wf_reg']) if ENABLE.get('wf_reg', False) else (None, None)
    ggs1_mean, ggs1_std = stats(results['focus']['ggs1']) if ENABLE.get('ggs1', False) else (None, None)
    ggs2_1_mean, ggs2_1_std = stats(results['focus']['ggs2_1']) if ENABLE.get('ggs2_1', False) else (None, None)
    fista_mean, fista_std = stats(results['focus']['fista']) if ENABLE.get('fista', False) else (None, None)
    pr_vbem_mean, pr_vbem_std = stats(results['focus']['pr_vbem']) if ENABLE.get('pr_vbem', False) else (None, None)
    raf21_tm_mean, raf21_tm_std = stats(results['focus']['raf21_tm']) if ENABLE.get('raf21_tm', False) else (None, None)
    autograd_mean, autograd_std = stats(results['focus']['autograd']) if ENABLE.get('autograd', False) else (None, None)

    ax.errorbar(snrs_with_noise, wf_reg_mean, yerr=wf_reg_std, fmt='-o', color='r', label='WF-Reg', capsize=5) if ENABLE.get('wf_reg', False) else None
    ax.errorbar(snrs_with_noise, wf_mean, yerr=wf_std, fmt='-s', color='b', label='WF', capsize=5) if ENABLE.get('wf', False) else None
    ax.errorbar(snrs_with_noise, ggs1_mean, yerr=ggs1_std, fmt='-^', color='m', label='GGS1', capsize=5) if ENABLE.get('ggs1', False) else None
    ax.errorbar(snrs_with_noise, ggs2_1_mean, yerr=ggs2_1_std, fmt='-d', color='g', label='GGS2-1', capsize=5) if ENABLE.get('ggs2_1', False) else None
    ax.errorbar(snrs_with_noise, fista_mean, yerr=fista_std, fmt='-x', color='c', label='FISTA', capsize=5) if ENABLE.get('fista', False) else None
    ax.errorbar(snrs_with_noise, pr_vbem_mean, yerr=pr_vbem_std, fmt='-p', color='y', label='PR-VBEM', capsize=5) if ENABLE.get('pr_vbem', False) else None
    ax.errorbar(snrs_with_noise, raf21_tm_mean, yerr=raf21_tm_std, fmt='-*', color='orange', label='RAF21-TM', capsize=5) if ENABLE.get('raf21_tm', False) else None
    ax.errorbar(snrs_with_noise, autograd_mean, yerr=autograd_std, fmt='-o', color='k', label='Autograd-Adam', capsize=5) if ENABLE.get('autograd', False) else None
    ax.set_title('(b) Contrast across SNRs')
    ax.set_xlabel('SNR')
    ax.set_ylabel('Contrast')
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('focus.png')

# --- 6. 运行主程序 ---
simulation_results = run_simulation()

plot_results(simulation_results)

plot_focus(simulation_results)

