import torch
import torch.nn.functional as F


def _coords_to_class_index(coords: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """将坐标转换为类别索引。"""
    rows = torch.clamp(coords[:, 0].long(), 0, h - 1)
    cols = torch.clamp(coords[:, 1].long(), 0, w - 1)
    return rows * w + cols


def _make_gaussian_target(coords: torch.Tensor, h: int, w: int, sigma: float, device: torch.device) -> torch.Tensor:
    """创建高斯目标热图。"""
    peak = 10000
    b = coords.shape[0]
    ys = torch.arange(h, device=device, dtype=torch.float32).view(h, 1)
    xs = torch.arange(w, device=device, dtype=torch.float32).view(1, w)
    rows = coords[:, 0].float().view(b, 1, 1)
    cols = coords[:, 1].float().view(b, 1, 1)
    dy2 = (ys.view(1, h, 1) - rows) ** 2
    dx2 = (xs.view(1, 1, w) - cols) ** 2
    g = torch.exp(-(dy2 + dx2) / (2.0 * (sigma ** 2) + 1e-12))
    g = g / (g.amax(dim=(1, 2), keepdim=True) + 1e-12)
    return g * peak


def pbr_loss(pred: torch.Tensor, coords: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """计算 PBR (peak-to-background ratio) 的损失（负对数均值）。
    
    Args:
        pred: [B, H, W] 复数或实数张量；若为复数，内部转强度
        coords: [B, 2] 的长整型像素坐标 (row, col)
        eps: 数值稳定性参数
        
    Returns:
        标量损失
    """
    if torch.is_complex(pred):
        intensity = pred.real.pow(2) + pred.imag.pow(2)
    else:
        intensity = pred

    B, H, W = intensity.shape
    batch_idx = torch.arange(B, device=intensity.device)
    peaks = intensity[batch_idx, coords[:, 0], coords[:, 1]]  # [B]

    total_sum = intensity.view(B, -1).sum(dim=1)              # [B]
    background_sum = total_sum - peaks                        # [B]
    background_count = H * W - 1
    background_mean = background_sum / (background_count + eps)

    pbr = peaks / (background_mean + eps)                     # [B]
    loss = -torch.log(pbr + eps).mean()                       # 标量
    return loss


def cross_entropy_loss_flat(pred: torch.Tensor, coords: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """扁平化的交叉熵损失函数。"""
    if torch.is_complex(pred):
        logits = pred.real.pow(2) + pred.imag.pow(2)
    else:
        logits = pred
    b, h, w = logits.shape
    target = _coords_to_class_index(coords, h, w)
    logits = logits.view(b, h * w)
    return F.cross_entropy(logits, target, label_smoothing=float(label_smoothing))


def focal_loss_flat(pred: torch.Tensor, coords: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """扁平化的Focal损失函数。"""
    if torch.is_complex(pred):
        logits = pred.real.pow(2) + pred.imag.pow(2)
    else:
        logits = pred
    b, h, w = logits.shape
    target = _coords_to_class_index(coords, h, w)
    logits = logits.view(b, h * w)
    log_probs = F.log_softmax(logits, dim=1)
    log_pt = log_probs.gather(1, target.view(-1, 1)).squeeze(1)
    pt = log_pt.exp()
    loss = -float(alpha) * (1 - pt) ** float(gamma) * log_pt
    return loss.mean()


def mse_gaussian_loss(pred: torch.Tensor, coords: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """高斯目标热图的MSE损失。"""
    intensity = pred.real.pow(2) + pred.imag.pow(2) if torch.is_complex(pred) else pred
    b, h, w = intensity.shape
    target = _make_gaussian_target(coords, h, w, float(sigma), intensity.device)
    return F.mse_loss(intensity, target)


def nrmse_gaussian_loss(pred: torch.Tensor, coords: torch.Tensor, sigma: float = 1.5, eps: float = 1e-12) -> torch.Tensor:
    """高斯目标热图的归一化RMSE损失。
    
    Computes per-sample RMSE and normalizes by the per-sample target range (max-min).
    Returns the batch mean of NRMSE values.
    """
    intensity = pred.real.pow(2) + pred.imag.pow(2) if torch.is_complex(pred) else pred
    b, h, w = intensity.shape
    target = _make_gaussian_target(coords, h, w, float(sigma), intensity.device)
    diff = intensity - target
    mse_per_sample = diff.pow(2).view(b, -1).mean(dim=1)
    rmse_per_sample = torch.sqrt(mse_per_sample + eps)
    t_max = target.amax(dim=(1, 2))
    t_min = target.amin(dim=(1, 2))
    t_range = t_max - t_min
    nrmse_per_sample = rmse_per_sample / (t_range + eps)
    return nrmse_per_sample.mean()


def kl_div_loss(pred: torch.Tensor, coords: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """高斯目标热图的KL散度损失。"""
    intensity = pred.real.pow(2) + pred.imag.pow(2) if torch.is_complex(pred) else pred
    b, h, w = intensity.shape
    target_map = _make_gaussian_target(coords, h, w, float(sigma), intensity.device)
    out_logits = intensity.view(b, h * w)
    tgt_logits = target_map.view(b, h * w)
    return kl_divergence_loss(out_logits, tgt_logits)


def coord_mse_loss(pred: torch.Tensor, coords: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """计算模型输出光强最大值坐标与标签坐标的MSE损失。
    
    Args:
        pred: [B, H, W] 复数或实数张量；若为复数，内部转强度
        coords: [B, 2] 的长整型像素坐标 (row, col)，表示标签坐标
        eps: 数值稳定性参数
        
    Returns:
        标量损失值
    """
    if torch.is_complex(pred):
        intensity = pred.real.pow(2) + pred.imag.pow(2)
    else:
        intensity = pred
    
    b, h, w = intensity.shape
    
    # 创建网格坐标系统，用于计算期望坐标
    # 使用softmax来获得可导的"期望"坐标
    intensity_flat = intensity.view(b, -1)  # [B, H*W]
    
    # 应用softmax获得每个位置的权重
    weights = F.softmax(intensity_flat, dim=1)  # [B, H*W]
    
    # 创建网格坐标矩阵 [H*W, 2]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=intensity.device, dtype=torch.float32),
        torch.arange(w, device=intensity.device, dtype=torch.float32),
        indexing='ij'
    )
    grid_coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)  # [H*W, 2]
    
    # 计算期望坐标：加权平均 [B, 2]
    pred_coords = torch.matmul(weights, grid_coords)  # [B, 2]
    
    # 将标签坐标转换为浮点数
    target_coords = coords.float()  # [B, 2]
    
    # 计算坐标之间的MSE损失
    coord_diff = pred_coords - target_coords  # [B, 2]
    mse_per_sample = torch.mean(coord_diff.pow(2), dim=1)  # [B]
    
    return mse_per_sample.mean()  # 标量损失


def _single_loss_by_name(name: str, pred: torch.Tensor, coords: torch.Tensor, args) -> torch.Tensor:
    """根据名称选择单个损失函数。"""
    if name == "pbr":
        return pbr_loss(pred, coords)
    if name == "xent":
        return cross_entropy_loss_flat(pred, coords, label_smoothing=0.0)
    if name == "xent_smooth":
        return cross_entropy_loss_flat(pred, coords, label_smoothing=float(getattr(args, "label_smoothing", 0.1)))
    if name == "focal":
        return focal_loss_flat(
            pred,
            coords,
            alpha=float(getattr(args, "focal_alpha", 0.25)),
            gamma=float(getattr(args, "focal_gamma", 2.0)),
        )
    if name in ("nrmse"):
        return nrmse_gaussian_loss(pred, coords, sigma=float(getattr(args, "gauss_sigma", 1.5)))
    if name == "kl":
        return kl_div_loss(pred, coords, sigma=float(getattr(args, "gauss_sigma", 1.5)))
    if name == "coord_mse":
        return coord_mse_loss(pred, coords)
    raise ValueError(f"Unknown base loss: {name}")


def compute_loss_by_name(pred: torch.Tensor, coords: torch.Tensor, args) -> torch.Tensor:
    """根据名称计算损失函数，支持混合损失。"""
    name = getattr(args, "loss", "pbr")
    if name == "mix":
        loss_a_name = getattr(args, "mix_loss_a", "pbr")
        loss_b_name = getattr(args, "mix_loss_b", "nrmse")
        alpha = float(getattr(args, "mix_alpha", 0.5))
        alpha = max(0.0, min(1.0, alpha))
        loss_a = _single_loss_by_name(loss_a_name, pred, coords, args)
        loss_b = _single_loss_by_name(loss_b_name, pred, coords, args)
        return alpha * loss_a + (1.0 - alpha) * loss_b
    return _single_loss_by_name(name, pred, coords, args)


# 从utils.py导入kl_divergence_loss函数
from utils import kl_divergence_loss