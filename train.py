import os
import time
import argparse
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
try:
    from tqdm import tqdm
except ImportError:
    # Fallback for environments without tqdm
    def tqdm(iterable, *args, **kwargs):
        return iterable

from scatterNeuralNetwork import ScatterNeuralNetwork
from utils import kl_divergence_loss
from cnn_model import SimpleCNN, ResNetHeatmap
from loss_functions import compute_loss_by_name
try:
    from caltech_loader import CaltechSeqDataset
    from inria_loader import InriaPersonDataset
    from pennfudan_loader import PennFudanDataset
except Exception:
    CaltechSeqDataset = None
    InriaPersonDataset = None
    PennFudanDataset = None

H_in, W_in = 768, 768
H_out, W_out = 32, 32
channels = 3

# Augmentation defaults (configurable via CLI)
AUG_ENABLE = True
AUG_HFLIP_P = 0.5
AUG_COLOR_P = 0.8
AUG_BLUR_P = 0.2
AUG_NOISE_P = 0.2

# Caltech Pedestrians root directory (contains Train/, Test/, annotations/)
DATA_ROOT = "/home/limingfei/speckle/donn/datasets/caltectPedestrains"

# PennFudan dataset root directory
PENNFUDAN_ROOT = "/home/limingfei/speckle/donn/datasets/PennFudanPed"

# For MNIST
MNIST_ROOT = "/home/limingfei/speckle/donn/datasets/mnist"



def _prepare_image_bgr_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """将 OpenCV BGR 图像预处理为 [1, 3, H_in, W_in] 的 torch.float32 张量，范围 [0,1]。"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (W_in, H_in), interpolation=cv2.INTER_AREA)
    img_rgb = (img_rgb.astype(np.float32) / 255.0)
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H_in, W_in]
    return x


def _augment_image_and_centers(img_bgr: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply simple train-time augmentations to image and adjust centers accordingly.

    Geometric: horizontal flip
    Photometric: color jitter (HSV), Gaussian blur, Gaussian noise
    """
    aug_img = img_bgr
    aug_centers = centers.copy() if centers is not None else None

    H, W = aug_img.shape[:2]

    # Horizontal flip
    if AUG_HFLIP_P > 0 and np.random.rand() < AUG_HFLIP_P:
        aug_img = cv2.flip(aug_img, 1)
        if aug_centers is not None and aug_centers.size > 0:
            # x' = (W - 1) - x
            aug_centers[:, 0] = (W - 1) - aug_centers[:, 0]

    # Color jitter in HSV space
    if AUG_COLOR_P > 0 and np.random.rand() < AUG_COLOR_P:
        hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV)
        # OpenCV HSV ranges: H: [0,180], S: [0,255], V: [0,255]
        h_shift = np.random.randint(-10, 11)  # +/- 10
        s_scale = np.random.uniform(0.8, 1.2)
        v_scale = np.random.uniform(0.8, 1.2)

        h = hsv[:, :, 0].astype(np.int32)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)

        h = (h + h_shift) % 180
        s = np.clip(s * s_scale, 0, 255)
        v = np.clip(v * v_scale, 0, 255)

        hsv[:, :, 0] = h.astype(np.uint8)
        hsv[:, :, 1] = s.astype(np.uint8)
        hsv[:, :, 2] = v.astype(np.uint8)
        aug_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Gaussian blur
    if AUG_BLUR_P > 0 and np.random.rand() < AUG_BLUR_P:
        k = np.random.choice([3, 5])
        aug_img = cv2.GaussianBlur(aug_img, (k, k), 0)

    # Additive Gaussian noise
    if AUG_NOISE_P > 0 and np.random.rand() < AUG_NOISE_P:
        sigma = np.random.uniform(5.0, 12.0)
        noise = np.random.normal(0.0, sigma, size=aug_img.shape).astype(np.float32)
        tmp = aug_img.astype(np.float32) + noise
        aug_img = np.clip(tmp, 0, 255).astype(np.uint8)

    return aug_img, (aug_centers if aug_centers is not None else centers)


def _pick_and_scale_center(centers: np.ndarray, orig_h: int, orig_w: int) -> tuple | None:
    """Pick one target center and scale to output grid (row, col).

    Returns (row, col) in [H_out, W_out] or None if no centers.
    """
    if centers is None or centers.size == 0:
        return None
    cx = centers[:, 0]
    cy = centers[:, 1]
    rows = np.clip(np.round(cy / orig_h * H_out).astype(np.int64), 0, H_out - 1)
    cols = np.clip(np.round(cx / orig_w * W_out).astype(np.int64), 0, W_out - 1)
    return int(rows[0]), int(cols[0])


def _batch_iter_caltech(root: str, split: str, batch_size: int, label_filter: str = "person", only_single: bool = False):
    """Yield batches from Caltech; frames without labels are skipped.

    When only_single=True, keep frames that contain exactly one labeled person.
    """
    if CaltechSeqDataset is None:
        raise RuntimeError("CaltechSeqDataset not available")
    ds = CaltechSeqDataset(root, split=split, label_filter=label_filter)
    imgs, coords = [], []
    for img_bgr, centers in ds:
        # Train-time augmentation
        if split == "Train" and AUG_ENABLE:
            img_bgr, centers = _augment_image_and_centers(img_bgr, centers)
        # Single-person filtering (before picking/scaling)
        if only_single:
            if centers is None or centers.size == 0:
                continue
            # Expect centers as shape [N, 2]; require N == 1
            if getattr(centers, "shape", None) is None or len(centers.shape) != 2 or centers.shape[0] != 1:
                continue
        orig_h, orig_w = img_bgr.shape[:2]
        rc = _pick_and_scale_center(centers, orig_h, orig_w)
        if rc is None:
            continue
        x1 = _prepare_image_bgr_to_tensor(img_bgr)
        imgs.append(x1)
        coords.append(torch.tensor([rc[0], rc[1]]).unsqueeze(0))
        if len(imgs) == batch_size:
            x_batch = torch.cat(imgs, dim=0)
            y_batch = torch.cat(coords, dim=0).long()
            yield x_batch, y_batch
            imgs, coords = [], []
    if imgs:
        x_batch = torch.cat(imgs, dim=0)
        y_batch = torch.cat(coords, dim=0).long()
        yield x_batch, y_batch


def _batch_iter_inria(root: str, split: str, batch_size: int, only_single: bool = False):
    """Yield batches from INRIA; images without boxes are skipped.

    When only_single=True, keep frames that contain exactly one labeled person.
    """
    if InriaPersonDataset is None:
        raise RuntimeError("InriaPersonDataset not available")
    ds = InriaPersonDataset(root, split=split, include_negatives=False)
    imgs, coords = [], []
    for img_bgr, centers in ds:
        if split == "Train" and AUG_ENABLE:
            img_bgr, centers = _augment_image_and_centers(img_bgr, centers)
        if only_single:
            if centers is None or centers.size == 0:
                continue
            if getattr(centers, "shape", None) is None or len(centers.shape) != 2 or centers.shape[0] != 1:
                continue
        orig_h, orig_w = img_bgr.shape[:2]
        rc = _pick_and_scale_center(centers, orig_h, orig_w)
        if rc is None:
            continue
        x1 = _prepare_image_bgr_to_tensor(img_bgr)
        imgs.append(x1)
        coords.append(torch.tensor([rc[0], rc[1]]).unsqueeze(0))
        if len(imgs) == batch_size:
            x_batch = torch.cat(imgs, dim=0)
            y_batch = torch.cat(coords, dim=0).long()
            yield x_batch, y_batch
            imgs, coords = [], []
    if imgs:
        x_batch = torch.cat(imgs, dim=0)
        y_batch = torch.cat(coords, dim=0).long()
        yield x_batch, y_batch


def _batch_iter_pennfudan(root: str, split: str, batch_size: int, only_single: bool = False):
    """Yield batches from Penn-Fudan; images without instances are skipped.

    When only_single=True, keep frames that contain exactly one labeled person.

    Penn-Fudan has no official Train/Test split. The split arg is ignored but kept
    for API symmetry.
    """
    if PennFudanDataset is None:
        raise RuntimeError("PennFudanDataset not available")
    ds = PennFudanDataset(root, split=split)

    imgs, coords = [], []
    for img_bgr, centers in ds:
        # Optional train-time augmentation
        if split == "Train" and AUG_ENABLE:
            img_bgr, centers = _augment_image_and_centers(img_bgr, centers)
        # Only keep frames with exactly one instance when requested
        if only_single:
            if centers is None or centers.size == 0:
                continue
            if getattr(centers, "shape", None) is None or len(centers.shape) != 2 or centers.shape[0] != 1:
                continue
        orig_h, orig_w = img_bgr.shape[:2]
        rc = _pick_and_scale_center(centers, orig_h, orig_w)
        if rc is None:
            continue
        x1 = _prepare_image_bgr_to_tensor(img_bgr)
        imgs.append(x1)
        coords.append(torch.tensor([rc[0], rc[1]]).unsqueeze(0))
        if len(imgs) == batch_size:
            x_batch = torch.cat(imgs, dim=0)
            y_batch = torch.cat(coords, dim=0).long()
            yield x_batch, y_batch
            imgs, coords = [], []
    if imgs:
        x_batch = torch.cat(imgs, dim=0)
        y_batch = torch.cat(coords, dim=0).long()
        yield x_batch, y_batch


def _mnist_label_to_coord(label: int, h: int, w: int) -> tuple[int, int]:
    """Map digit label 0-9 to one of ten fixed 2D points on the output grid.

    We place 10 points evenly spaced on a circle within the grid.
    Returns (row, col) integer indices in [0..h-1], [0..w-1].
    """
    # Place on circle
    num = 10
    idx = int(label) % num
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    r = 0.35 * min(h, w)
    theta = 2.0 * np.pi * (idx / num)
    y = cy + r * np.sin(theta)
    x = cx + r * np.cos(theta)
    rr = int(np.clip(np.round(y), 0, h - 1))
    cc = int(np.clip(np.round(x), 0, w - 1))
    return rr, cc


def _batch_iter_mnist(root: str, split: str, batch_size: int, image_hw: tuple[int, int], out_hw: tuple[int, int]):
    """Yield MNIST batches as RGB images resized to H_in,W_in with 2D coord targets.

    split: one of {"Train","Test"}
    """
    import torchvision
    from torchvision import transforms

    h_in, w_in = int(image_hw[0]), int(image_hw[1])
    h_out, w_out = int(out_hw[0]), int(out_hw[1])

    tfm = transforms.Compose([
        transforms.Resize((h_in, w_in)),
        transforms.ToTensor(),  # [1,H,W] in [0,1]
    ])
    train_flag = (split.lower() == "train")
    ds = torchvision.datasets.MNIST(root=root, train=train_flag, transform=tfm, download=True)

    imgs, coords = [], []
    for img_tensor, label in ds:
        # Convert to 3-channel RGB-like by repeat
        if img_tensor.ndim == 3 and img_tensor.shape[0] == 1:
            img3 = img_tensor.repeat(3, 1, 1).unsqueeze(0)  # [1,3,H,W]
        else:
            img3 = img_tensor.unsqueeze(0)
        rr, cc = _mnist_label_to_coord(int(label), h_out, w_out)
        imgs.append(img3)
        coords.append(torch.tensor([rr, cc]).view(1, 2))
        if len(imgs) == batch_size:
            x_batch = torch.cat(imgs, dim=0)
            y_batch = torch.cat(coords, dim=0).long()
            yield x_batch, y_batch
            imgs, coords = [], []
    if imgs:
        x_batch = torch.cat(imgs, dim=0)
        y_batch = torch.cat(coords, dim=0).long()
        yield x_batch, y_batch


@torch.no_grad()
def _compute_intensity(pred: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(pred):
        return pred.real.pow(2) + pred.imag.pow(2)
    return pred


def _visualize_epoch_samples(model: ScatterNeuralNetwork, data_root: str, label_filter: str, epoch_index: int, writer: SummaryWriter | None, vis_samples: int = 6, heatmap_alpha: float = 0.45, dataset: str = "caltech", only_single: bool = False, tag_prefix: str = "") -> None:
    """Visualize a few labeled training frames and write a figure to TensorBoard."""
    model.eval()
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    if dataset == "caltech":
        if CaltechSeqDataset is None:
            return
        ds = CaltechSeqDataset(data_root, split="Train", label_filter=label_filter)
    elif dataset == "inria":
        if InriaPersonDataset is None:
            return
        ds = InriaPersonDataset(data_root, split="Train", include_negatives=False)
    elif dataset == "pennfudan":
        if PennFudanDataset is None:
            return
        ds = PennFudanDataset(data_root, split="Train")
    else:  # mnist
        # Build few samples from MNIST directly
        import torchvision
        from torchvision import transforms
        tfm = transforms.Compose([
            transforms.Resize((H_in, W_in)),
            transforms.ToTensor(),
        ])
        ds = torchvision.datasets.MNIST(root=data_root, train=True, transform=tfm, download=True)
    if dataset == "mnist":
        for img_t, label in ds:
            samples.append((img_t, int(label)))
            if len(samples) >= vis_samples:
                break
    else:
        for img_bgr, centers in ds:
            if centers is not None and centers.size > 0 and (not only_single or (getattr(centers, "shape", None) is not None and len(centers.shape) == 2 and centers.shape[0] == 1)):
                samples.append((img_bgr, centers))
            if len(samples) >= vis_samples:
                break

    if not samples:
        return

    rows = int(np.ceil(len(samples) / 3))
    cols = 3
    fig = plt.figure(figsize=(cols * 4, rows * 3.5))

    if dataset == "mnist":
        for i, (img_t, label) in enumerate(samples):
            img_rgb = (img_t.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
            orig_h, orig_w = img_rgb.shape[:2]
            with torch.no_grad():
                x_in = torch.from_numpy(img_rgb[..., ::-1].copy()).float() / 255.0
                x_in = x_in.permute(2, 0, 1).unsqueeze(0)
                x_in = F.interpolate(x_in, size=(H_in, W_in), mode="bilinear", align_corners=False)
                x_in = x_in.to(next(model.parameters()).device)
                pred = model(x_in)
            if torch.is_complex(pred):
                intensity = (pred[0].real ** 2 + pred[0].imag ** 2).detach().cpu().numpy()
            else:
                intensity = pred[0].detach().cpu().numpy()

            flat_idx = int(intensity.reshape(-1).argmax())
            peak_r = flat_idx // intensity.shape[1]
            peak_c = flat_idx % intensity.shape[1]
            pred_cy = float(peak_r) / H_out * orig_h
            pred_cx = float(peak_c) / W_out * orig_w

            heat = intensity
            heat = (heat - heat.min()) / (np.ptp(heat) + 1e-12)
            heat_u8 = (heat * 255.0).astype(np.uint8)
            heat_u8 = cv2.resize(heat_u8, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            heat_color_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
            vis = cv2.addWeighted(heat_color_rgb, heatmap_alpha, img_rgb, 1.0 - heatmap_alpha, 0)

            rr, cc = _mnist_label_to_coord(int(label), H_out, W_out)
            gt_cy = float(rr) / H_out * orig_h
            gt_cx = float(cc) / W_out * orig_w
            cv2.circle(vis, (int(gt_cx), int(gt_cy)), 6, (0, 255, 0), 2)
            cv2.drawMarker(vis, (int(pred_cx), int(pred_cy)), (255, 255, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=2)

            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(vis)
            ax.set_title(f"E{epoch_index} sample {i+1}")
            ax.axis("off")
    else:
        for i, (img_bgr, centers) in enumerate(samples):
            orig_h, orig_w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                x_in = _prepare_image_bgr_to_tensor(img_bgr)
            x_in = x_in.to(next(model.parameters()).device)
            pred = model(x_in)
            if torch.is_complex(pred):
                intensity = (pred[0].real ** 2 + pred[0].imag ** 2).detach().cpu().numpy()
            else:
                intensity = pred[0].detach().cpu().numpy()

            flat_idx = int(intensity.reshape(-1).argmax())
            peak_r = flat_idx // intensity.shape[1]
            peak_c = flat_idx % intensity.shape[1]
            pred_cy = float(peak_r) / H_out * orig_h
            pred_cx = float(peak_c) / W_out * orig_w

            heat = intensity
            heat = (heat - heat.min()) / (np.ptp(heat) + 1e-12)
            heat_u8 = (heat * 255.0).astype(np.uint8)
            heat_u8 = cv2.resize(heat_u8, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            heat_color_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
            vis = cv2.addWeighted(heat_color_rgb, heatmap_alpha, img_rgb, 1.0 - heatmap_alpha, 0)

            if centers is not None and centers.size > 0:
                cx, cy = centers[0]
                cv2.circle(vis, (int(cx), int(cy)), 6, (0, 255, 0), 2)
            cv2.drawMarker(vis, (int(pred_cx), int(pred_cy)), (255, 255, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=2)

            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(vis)
            ax.set_title(f"E{epoch_index} sample {i+1}")
            ax.axis("off")

        flat_idx = int(intensity.reshape(-1).argmax())
        peak_r = flat_idx // intensity.shape[1]
        peak_c = flat_idx % intensity.shape[1]
        pred_cy = float(peak_r) / H_out * orig_h
        pred_cx = float(peak_c) / W_out * orig_w

        heat = intensity
        heat = (heat - heat.min()) / (np.ptp(heat) + 1e-12)
        heat_u8 = (heat * 255.0).astype(np.uint8)
        heat_u8 = cv2.resize(heat_u8, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        heat_color_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        vis = cv2.addWeighted(heat_color_rgb, heatmap_alpha, img_rgb, 1.0 - heatmap_alpha, 0)

        if dataset == "mnist":
            rr, cc = _mnist_label_to_coord(int(label), H_out, W_out)
            gt_cy = float(rr) / H_out * orig_h
            gt_cx = float(cc) / W_out * orig_w
            cv2.circle(vis, (int(gt_cx), int(gt_cy)), 6, (0, 255, 0), 2)
        elif centers is not None and centers.size > 0:
            cx, cy = centers[0]
            cv2.circle(vis, (int(cx), int(cy)), 6, (0, 255, 0), 2)
        cv2.drawMarker(vis, (int(pred_cx), int(pred_cy)), (255, 255, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=2)

        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(vis)
        ax.set_title(f"E{epoch_index} sample {i+1}")
        ax.axis("off")

    fig.suptitle(f"Epoch {epoch_index}: GT (green) vs Pred Peak (white)")
    fig.tight_layout()
    if writer is not None:
        writer.add_figure(f"{tag_prefix}train/epoch_visualization", fig, global_step=epoch_index)
    plt.close(fig)


def train_caltech(args) -> None:
    # Distributed init
    is_dist = False
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        is_dist = True
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)

    # Allow overriding global input/output sizes for helper functions
    global H_in, W_in, H_out, W_out, DATA_ROOT
    H_in, W_in = int(args.h_in), int(args.w_in)
    H_out, W_out = int(args.h_out), int(args.w_out)
    DATA_ROOT = args.data_root

    # Configure augmentation globals from args
    global AUG_ENABLE, AUG_HFLIP_P, AUG_COLOR_P, AUG_BLUR_P, AUG_NOISE_P
    AUG_ENABLE = not bool(args.disable_aug)
    AUG_HFLIP_P = float(args.aug_hflip_p)
    AUG_COLOR_P = float(args.aug_color_p)
    AUG_BLUR_P = float(args.aug_blur_p)
    AUG_NOISE_P = float(args.aug_noise_p)

    # Map CLI dtype to torch dtype for transmission-matrix compute path
    if args.tmatrix_compute_dtype == "fp32":
        tmatrix_compute_dtype = None  # use complex64 direct matmul
    elif args.tmatrix_compute_dtype == "bf16":
        tmatrix_compute_dtype = torch.bfloat16
    else:
        tmatrix_compute_dtype = torch.float16

    # Enable tensor-parallel shards inside model when running distributed (scatter model only)
    tp_enabled = bool(args.tp and (world_size > 1))
    if args.model == "scatter":
        # 解析激活函数参数
        activation_params = {}
        if args.activation_params:
            try:
                activation_params = json.loads(args.activation_params)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse activation_params JSON: {e}")
                activation_params = {}
        
        model = ScatterNeuralNetwork(
            input_hw=(H_in, W_in),
            output_hw=(H_out, W_out),
            return_intensity=args.return_intensity,
            num_layers=int(args.num_layers),
            seed=42,
            tmatrix_compute_dtype=tmatrix_compute_dtype,
            tp_enabled=tp_enabled,
            tp_rank=(local_rank if tp_enabled else 0),
            tp_world_size=(world_size if tp_enabled else 1),
            activation=args.activation,
            activation_params=activation_params,
            normalize_negative=args.normalize_negative,
        ).to(device)
    elif args.model == "simple_cnn":
        model = SimpleCNN(input_hw=(H_in, W_in), output_hw=(H_out, W_out)).to(device)
        tp_enabled = False
    else:  # resnet
        model = ResNetHeatmap(
            input_hw=(H_in, W_in),
            output_hw=(H_out, W_out),
            variant=args.resnet_variant,
            pretrained=bool(args.resnet_pretrained),
        ).to(device)
        tp_enabled = False

    if is_dist:
        # Don't broadcast buffers (each rank holds different shard)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    # rank0 only logging
    is_rank0 = (not is_dist) or (dist.get_rank() == 0)
    writer = SummaryWriter(args.log_dir) if is_rank0 else None
    # Tag prefix for TensorBoard when using CNN-based models
    tag_prefix = ("cnn/" if args.model in ("simple_cnn", "resnet") else "")
    if is_dist and tp_enabled:
        dist.barrier(device_ids=[local_rank])
    # Enable GradScaler only when actually using fp16 compute; it's not needed for bf16/fp32
    use_fp16 = (device.type == "cuda" and args.tmatrix_compute_dtype == "fp16")
    scaler = GradScaler("cuda", enabled=use_fp16)

    model.train()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_loss_sum = 0.0
        epoch_batch_count = 0

        if args.dataset == "caltech":
            batch_iter = _batch_iter_caltech(DATA_ROOT, split="Train", batch_size=args.batch_size, label_filter=args.label_filter, only_single=args.only_single)
        elif args.dataset == "inria":
            batch_iter = _batch_iter_inria(DATA_ROOT, split="Train", batch_size=args.batch_size, only_single=args.only_single)
        elif args.dataset == "pennfudan":
            batch_iter = _batch_iter_pennfudan(DATA_ROOT, split="Train", batch_size=args.batch_size, only_single=args.only_single)
        else:  # mnist
            batch_iter = _batch_iter_mnist(DATA_ROOT, split="Train", batch_size=args.batch_size, image_hw=(H_in, W_in), out_hw=(H_out, W_out))
        t0 = time.time()
        
        # 创建进度条（仅在rank0进程显示）
        if is_rank0:
            # 计算总批次数量
            total_batches = args.max_train_batches if args.max_train_batches > 0 else float('inf')
            pbar = tqdm(batch_iter, desc=f"Epoch {epoch}/{args.epochs}", 
                       total=total_batches, leave=False, 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        else:
            pbar = batch_iter
        
        for b_idx, (x_batch, coords_batch) in enumerate(pbar, start=1):
            # Broadcast batch from rank0 so each rank sees same data
            x_batch = x_batch.to(device, non_blocking=True)
            coords_batch = coords_batch.to(device, non_blocking=True)
            if is_dist:
                dist.broadcast(x_batch, src=0)
                dist.broadcast(coords_batch, src=0)

            optimizer.zero_grad(set_to_none=True)
            # Autocast settings: for scatter, follow tmatrix compute dtype; for CNN, use same mapping
            if args.tmatrix_compute_dtype == "fp32":
                ac_enabled = False
                ac_dtype = None
            elif args.tmatrix_compute_dtype == "bf16":
                ac_enabled = (device.type == "cuda")
                ac_dtype = torch.bfloat16
            else:
                ac_enabled = (device.type == "cuda")
                ac_dtype = torch.float16

            with autocast("cuda", enabled=ac_enabled, dtype=ac_dtype if ac_enabled else None):
                pred = model(x_batch)
                loss = compute_loss_by_name(pred, coords_batch, args)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if writer is not None:
                writer.add_scalar(f"{tag_prefix}train/loss_batch", float(loss.detach().cpu()), global_step)
            global_step += 1

            epoch_loss_sum += float(loss.detach().cpu())
            epoch_batch_count += 1

            # 更新进度条显示
            if is_rank0:
                current_loss = loss.item()
                avg_loss = epoch_loss_sum / epoch_batch_count
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'avg_loss': f'{avg_loss:.4f}'
                })

            if args.max_train_batches > 0 and b_idx >= args.max_train_batches:
                break
        
        # 关闭进度条
        if is_rank0:
            pbar.close()

        epoch_time = time.time() - t0
        if epoch_batch_count == 0 and is_rank0:
            print(f"epoch {epoch} has no batches; check dataset and filters.")
            continue

        epoch_loss_avg = epoch_loss_sum / epoch_batch_count
        if is_rank0:
            print(f"[Epoch {epoch}/{args.epochs}] avg_loss={epoch_loss_avg:.6f} batches={epoch_batch_count} time={epoch_time:.1f}s")
            if writer is not None:
                writer.add_scalar(f"{tag_prefix}train/loss_epoch", epoch_loss_avg, epoch)

        # Visualization: when tensor-parallel is enabled (scatter model), all ranks must run forward
        if is_dist and tp_enabled:
            dist.barrier(device_ids=[local_rank])
        _visualize_epoch_samples(
            model.module if isinstance(model, DDP) else model,
            DATA_ROOT,
            args.label_filter,
            epoch,
            writer if is_rank0 else None,
            vis_samples=args.vis_samples,
            dataset=args.dataset,
            only_single=args.only_single,
            tag_prefix=tag_prefix,
        )
        if is_dist and tp_enabled:
            dist.barrier(device_ids=[local_rank])

        # Evaluation on Test split (average loss)
        model.eval()
        test_loss_sum = 0.0
        test_batch_count = 0
        with torch.no_grad():
            if args.dataset == "caltech":
                test_iter = _batch_iter_caltech(DATA_ROOT, split="Test", batch_size=args.batch_size, label_filter=args.label_filter, only_single=args.only_single)
            elif args.dataset == "inria":
                test_iter = _batch_iter_inria(DATA_ROOT, split="Test", batch_size=args.batch_size, only_single=args.only_single)
            elif args.dataset == "pennfudan":
                test_iter = _batch_iter_pennfudan(DATA_ROOT, split="Test", batch_size=args.batch_size, only_single=args.only_single)
            else:
                test_iter = _batch_iter_mnist(DATA_ROOT, split="Test", batch_size=args.batch_size, image_hw=(H_in, W_in), out_hw=(H_out, W_out))
            
            # 创建测试进度条（仅在rank0进程显示）
            if is_rank0:
                test_pbar = tqdm(test_iter, desc=f"Testing Epoch {epoch}", 
                               total=args.max_test_batches if args.max_test_batches > 0 else float('inf'),
                               leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            else:
                test_pbar = test_iter
            
            for tb_idx, (x_batch, coords_batch) in enumerate(test_pbar, start=1):
                x_batch = x_batch.to(device, non_blocking=True)
                coords_batch = coords_batch.to(device, non_blocking=True)
                if is_dist:
                    dist.broadcast(x_batch, src=0)
                    dist.broadcast(coords_batch, src=0)
                if args.tmatrix_compute_dtype == "fp32":
                    ac_enabled = False
                    ac_dtype = None
                elif args.tmatrix_compute_dtype == "bf16":
                    ac_enabled = (device.type == "cuda")
                    ac_dtype = torch.bfloat16
                else:
                    ac_enabled = (device.type == "cuda")
                    ac_dtype = torch.float16
                with autocast("cuda", enabled=ac_enabled, dtype=ac_dtype if ac_enabled else None):
                    pred = model(x_batch)
                    loss = compute_loss_by_name(pred, coords_batch, args)
                test_loss_sum += float(loss.detach().cpu())
                test_batch_count += 1
                
                # 更新测试进度条
                if is_rank0:
                    current_test_loss = loss.item()
                    avg_test_loss = test_loss_sum / test_batch_count
                    test_pbar.set_postfix({
                        'test_loss': f'{current_test_loss:.4f}',
                        'avg_test_loss': f'{avg_test_loss:.4f}'
                    })
                
                if args.max_test_batches > 0 and tb_idx >= args.max_test_batches:
                    break
            
            # 关闭测试进度条
            if is_rank0:
                test_pbar.close()

        if is_dist and tp_enabled:
            # Ensure all ranks finished eval loop before printing/saving
            dist.barrier(device_ids=[local_rank])
        if test_batch_count > 0:
            test_loss_avg = test_loss_sum / test_batch_count
            if is_rank0:
                print(f"[Epoch {epoch}/{args.epochs}] test_avg_loss={test_loss_avg:.6f} over {test_batch_count} batches")
                if writer is not None:
                    writer.add_scalar(f"{tag_prefix}test/loss_epoch", test_loss_avg, epoch)
        else:
            if is_rank0:
                print("No test batches produced (check Test split labels/filter).")

        # Save checkpoint periodically
        if is_rank0 and (epoch % int(args.save_every) == 0 or epoch == int(args.epochs)):
            ckpt = {
            "epoch": epoch,
            "model_state_dict": (model.module if isinstance(model, DDP) else model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss_avg": epoch_loss_avg,
            "test_loss_avg": (test_loss_sum / test_batch_count) if test_batch_count > 0 else None,
            "args": vars(args),
            }
            os.makedirs(args.ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(args.ckpt_dir, f"epoch_{epoch:03d}.pth")
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint to: {ckpt_path}")

        model.train()

    if writer is not None:
        writer.flush()
        writer.close()
    if is_dist:
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train model (scatter/simple_cnn/resnet) on Caltech/INRIA/PennFudan/MNIST")
    # Model selection
    parser.add_argument("--model", type=str, default="scatter", choices=["scatter", "simple_cnn", "resnet"], help="Model type")
    parser.add_argument("--resnet_variant", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"], help="When --model resnet")
    parser.add_argument("--resnet_pretrained", action="store_true", help="Use ImageNet-pretrained weights for ResNet")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--dataset", type=str, default="mnist", choices=["caltech", "inria", "pennfudan", "mnist"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_train_batches", type=int, default=-1, help=">0 to limit batches per epoch; -1 for full")
    parser.add_argument("--max_test_batches", type=int, default=100, help=">0 to limit test batches per epoch; -1 for full")
    parser.add_argument("--vis_samples", type=int, default=6)
    parser.add_argument("--label_filter", type=str, default="person")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=("cuda:1" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--log_dir", type=str, default=os.path.join(os.path.abspath(os.getcwd()), "runs", "caltech_full"))
    parser.add_argument("--ckpt_dir", type=str, default=os.path.join(os.path.abspath(os.getcwd()), "checkpoints", "caltech_full"))
    parser.add_argument("--h_in", type=int, default=H_in)
    parser.add_argument("--w_in", type=int, default=W_in)
    parser.add_argument("--h_out", type=int, default=H_out)
    parser.add_argument("--w_out", type=int, default=W_out)
    parser.add_argument("--num_layers", type=int, default=1, help="Times to apply (phase->TM->abs) per forward")
    parser.add_argument("--tmatrix_compute_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Compute dtype for transmission matmul; fp32 uses complex64 direct matmul")
    parser.add_argument("--tp", action="store_true", help="Enable tensor-parallel sharded transmission matrix across world_size GPUs")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs (also saves at final epoch)")
    # Loss configuration
    parser.add_argument("--loss", type=str, default="pbr", choices=["pbr", "xent", "xent_smooth", "focal", "mse", "nrmse", "kl", "coord_mse", "mix"], help="Which loss to use")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing for xent_smooth")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="Focal loss alpha")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--gauss_sigma", type=float, default=1.5, help="Sigma for Gaussian target in MSE/NRMSE/KL")
    # Mixed loss configuration
    parser.add_argument("--mix_loss_a", type=str, default="pbr", choices=["pbr", "xent", "xent_smooth", "focal", "mse", "nrmse", "kl", "coord_mse"], help="First loss when --loss mix")
    parser.add_argument("--mix_loss_b", type=str, default="mse", choices=["pbr", "xent", "xent_smooth", "focal", "mse", "nrmse", "kl", "coord_mse"], help="Second loss when --loss mix")
    parser.add_argument("--mix_alpha", type=float, default=0.5, help="Weight for first loss in mix: total = a*alpha + b*(1-alpha)")
    # Augmentation controls
    parser.add_argument("--disable_aug", action="store_true", help="Disable all train-time data augmentation")
    parser.add_argument("--aug_hflip_p", type=float, default=0.5, help="Probability of horizontal flip during training")
    parser.add_argument("--aug_color_p", type=float, default=0.8, help="Probability of color jitter during training")
    parser.add_argument("--aug_blur_p", type=float, default=0.2, help="Probability of Gaussian blur during training")
    parser.add_argument("--aug_noise_p", type=float, default=0.2, help="Probability of Gaussian noise during training")
    # Dataset filtering
    parser.add_argument("--only_single", action="store_true", help="Use only images with exactly one labeled person")
    # Activation function configuration
    parser.add_argument("--activation", type=str, default="abs", choices=["abs", "relu", "leaky_relu", "sigmoid", "tanh", "elu", "softplus"], help="Activation function for scatter model")
    parser.add_argument("--activation_params", type=str, default=None, help="JSON string for activation function parameters")
    parser.add_argument("--normalize_negative", action="store_true", help="Normalize negative values to [0,1] range")
    parser.add_argument("--return_intensity", action="store_true", help="Return intensity (real values) instead of complex field")
    args = parser.parse_args()

    if args.max_train_batches < 0:
        args.max_train_batches = 0
    if args.max_test_batches < 0:
        args.max_test_batches = 0
    # Organize outputs by model group (cnn vs scatter), dataset and timestamp
    ts = time.strftime("%Y%m%d-%H%M%S")
    model_group = ("cnn" if args.model in ("simple_cnn", "resnet") else "scatter")
    args.log_dir = os.path.join(args.log_dir, model_group, args.dataset, ts)
    args.ckpt_dir = os.path.join(args.ckpt_dir, model_group, args.dataset, ts)
    # main() runs before distributed init; just print the logdir once
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"TensorBoard logdir: {args.log_dir}")
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train_caltech(args)


if __name__ == "__main__":
    main()