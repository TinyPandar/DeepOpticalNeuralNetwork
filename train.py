import os
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from scatterNeuralNetwork import ScatterNeuralNetwork
from caltech_loader import CaltechSeqDataset
from inria_loader import InriaPersonDataset
from pennfudan_loader import PennFudanDataset

H_in, W_in = 768, 768
H_out, W_out = 64, 64
channels = 3

# Caltech Pedestrians root directory (contains Train/, Test/, annotations/)
DATA_ROOT = "/home/limingfei/speckle/donn/datasets/caltectPedestrains"

def pbr_loss(pred: torch.Tensor, coords: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """计算 PBR (peak-to-background ratio) 的损失（负对数均值）。
    pred: [B, H, W] 复数或实数张量；若为复数，内部转强度。
    coords: [B, 2] 的长整型像素坐标 (row, col)。
    返回标量损失。
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

def _prepare_image_bgr_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """将 OpenCV BGR 图像预处理为 [1, 3, H_in, W_in] 的 torch.float32 张量，范围 [0,1]。"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (W_in, H_in), interpolation=cv2.INTER_AREA)
    img_rgb = (img_rgb.astype(np.float32) / 255.0)
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H_in, W_in]
    return x


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


def _batch_iter_caltech(root: str, split: str, batch_size: int, label_filter: str = "person"):
    """Yield batches from Caltech; frames without labels are skipped."""
    ds = CaltechSeqDataset(root, split=split, label_filter=label_filter)
    imgs, coords = [], []
    for img_bgr, centers in ds:
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


def _batch_iter_inria(root: str, split: str, batch_size: int):
    """Yield batches from INRIA; images without boxes are skipped."""
    ds = InriaPersonDataset(root, split=split, include_negatives=False)
    imgs, coords = [], []
    for img_bgr, centers in ds:
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


def _batch_iter_pennfudan(root: str, split: str, batch_size: int):
    """Yield batches from Penn-Fudan; images without instances are skipped.

    Penn-Fudan has no official Train/Test split. The split arg is ignored but kept
    for API symmetry.
    """
    ds = PennFudanDataset(root, split=split)
    imgs, coords = [], []
    for img_bgr, centers in ds:
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


@torch.no_grad()
def _compute_intensity(pred: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(pred):
        return pred.real.pow(2) + pred.imag.pow(2)
    return pred


def _visualize_epoch_samples(model: ScatterNeuralNetwork, data_root: str, label_filter: str, epoch_index: int, writer: SummaryWriter, vis_samples: int = 6, heatmap_alpha: float = 0.45, dataset: str = "caltech") -> None:
    """Visualize a few labeled training frames and write a figure to TensorBoard."""
    model.eval()
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    if dataset == "caltech":
        ds = CaltechSeqDataset(data_root, split="Train", label_filter=label_filter)
    elif dataset == "inria":
        ds = InriaPersonDataset(data_root, split="Train", include_negatives=False)
    else:
        ds = PennFudanDataset(data_root, split="Train")
    for img_bgr, centers in ds:
        if centers is not None and centers.size > 0:
            samples.append((img_bgr, centers))
        if len(samples) >= vis_samples:
            break

    if not samples:
        return

    rows = int(np.ceil(len(samples) / 3))
    cols = 3
    fig = plt.figure(figsize=(cols * 4, rows * 3.5))

    for i, (img_bgr, centers) in enumerate(samples):
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            x_in = _prepare_image_bgr_to_tensor(img_bgr)
            # Ensure input is on the same device as the model to avoid device mismatch
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
            for (cx, cy) in centers:
                cv2.circle(vis, (int(cx), int(cy)), 6, (0, 255, 0), 2)
        cv2.drawMarker(vis, (int(pred_cx), int(pred_cy)), (255, 255, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=2)

        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(vis)
        ax.set_title(f"E{epoch_index} sample {i+1}")
        ax.axis("off")

    fig.suptitle(f"Epoch {epoch_index}: GT (green) vs Pred Peak (white)")
    fig.tight_layout()
    writer.add_figure("train/epoch_visualization", fig, global_step=epoch_index)
    plt.close(fig)


def train_caltech(args) -> None:
    device = torch.device(args.device)

    # Allow overriding global input/output sizes for helper functions
    global H_in, W_in, H_out, W_out, DATA_ROOT
    H_in, W_in = int(args.h_in), int(args.w_in)
    H_out, W_out = int(args.h_out), int(args.w_out)
    DATA_ROOT = args.data_root

    model = ScatterNeuralNetwork(
        input_hw=(H_in, W_in),
        output_hw=(H_out, W_out),
        return_intensity=False,
        seed=42,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    writer = SummaryWriter(args.log_dir)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    model.train()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_loss_sum = 0.0
        epoch_batch_count = 0

        if args.dataset == "caltech":
            batch_iter = _batch_iter_caltech(DATA_ROOT, split="Train", batch_size=args.batch_size, label_filter=args.label_filter)
        elif args.dataset == "inria":
            batch_iter = _batch_iter_inria(DATA_ROOT, split="Train", batch_size=args.batch_size)
        else:
            batch_iter = _batch_iter_pennfudan(DATA_ROOT, split="Train", batch_size=args.batch_size)
        t0 = time.time()
        for b_idx, (x_batch, coords_batch) in enumerate(batch_iter, start=1):
            x_batch = x_batch.to(device)
            coords_batch = coords_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(x_batch)
                loss = pbr_loss(pred, coords_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            writer.add_scalar("train/loss_batch", float(loss.detach().cpu()), global_step)
            global_step += 1

            epoch_loss_sum += float(loss.detach().cpu())
            epoch_batch_count += 1

            if b_idx % 20 == 0:
                print(f"epoch {epoch:02d} batch {b_idx:04d} loss {loss.item():.6f}")

            if args.max_train_batches > 0 and b_idx >= args.max_train_batches:
                break

        epoch_time = time.time() - t0
        if epoch_batch_count == 0:
            print(f"epoch {epoch} has no batches; check dataset and filters.")
            continue

        epoch_loss_avg = epoch_loss_sum / epoch_batch_count
        print(f"[Epoch {epoch}/{args.epochs}] avg_loss={epoch_loss_avg:.6f} batches={epoch_batch_count} time={epoch_time:.1f}s")
        writer.add_scalar("train/loss_epoch", epoch_loss_avg, epoch)

        _visualize_epoch_samples(model, DATA_ROOT, args.label_filter, epoch, writer, vis_samples=args.vis_samples, dataset=args.dataset)

        # Evaluation on Test split (average loss)
        model.eval()
        test_loss_sum = 0.0
        test_batch_count = 0
        with torch.no_grad():
            if args.dataset == "caltech":
                test_iter = _batch_iter_caltech(DATA_ROOT, split="Test", batch_size=args.batch_size, label_filter=args.label_filter)
            elif args.dataset == "inria":
                test_iter = _batch_iter_inria(DATA_ROOT, split="Test", batch_size=args.batch_size)
            else:
                test_iter = _batch_iter_pennfudan(DATA_ROOT, split="Test", batch_size=args.batch_size)
            for tb_idx, (x_batch, coords_batch) in enumerate(test_iter, start=1):
                x_batch = x_batch.to(device)
                coords_batch = coords_batch.to(device)
                with autocast("cuda", enabled=(device.type == "cuda")):
                    pred = model(x_batch)
                    loss = pbr_loss(pred, coords_batch)
                test_loss_sum += float(loss.detach().cpu())
                test_batch_count += 1
                if args.max_test_batches > 0 and tb_idx >= args.max_test_batches:
                    break

        if test_batch_count > 0:
            test_loss_avg = test_loss_sum / test_batch_count
            print(f"[Epoch {epoch}/{args.epochs}] test_avg_loss={test_loss_avg:.6f} over {test_batch_count} batches")
            writer.add_scalar("test/loss_epoch", test_loss_avg, epoch)
        else:
            print("No test batches produced (check Test split labels/filter).")

        # Save checkpoint at the end of each epoch
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
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

    writer.flush()
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train ScatterNeuralNetwork on Caltech/INRIA/PennFudan")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--dataset", type=str, default="pennfudan", choices=["caltech", "inria", "pennfudan"])
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
    args = parser.parse_args()

    if args.max_train_batches < 0:
        args.max_train_batches = 0
    if args.max_test_batches < 0:
        args.max_test_batches = 0
    # Organize outputs by dataset and timestamp to avoid collisions
    ts = time.strftime("%Y%m%d-%H%M%S")
    args.log_dir = os.path.join(args.log_dir, args.dataset, ts)
    args.ckpt_dir = os.path.join(args.ckpt_dir, args.dataset, ts)
    print(f"TensorBoard logdir: {args.log_dir}")
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train_caltech(args)


if __name__ == "__main__":
    main()