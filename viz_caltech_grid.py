import os
from typing import List, Tuple

import cv2
import numpy as np

from caltech_loader import CaltechSeqDataset


def draw_centers(image: np.ndarray, centers: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    vis = image.copy()
    if centers is not None and centers.size:
        for (cx, cy) in centers:
            cv2.circle(vis, (int(round(cx)), int(round(cy))), radius=3, color=color, thickness=-1)
    return vis


def make_grid(images: List[np.ndarray], rows: int = 3, cols: int = 3) -> np.ndarray:
    assert len(images) == rows * cols, "images count must equal rows*cols"
    # Ensure same size
    h, w = images[0].shape[:2]
    fixed = [cv2.resize(im, (w, h)) if im.shape[:2] != (h, w) else im for im in images]
    # Optional margins
    bordered = [
        cv2.copyMakeBorder(im, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        for im in fixed
    ]
    row_imgs = []
    for r in range(rows):
        row = cv2.hconcat(bordered[r * cols : (r + 1) * cols])
        row_imgs.append(row)
    grid = cv2.vconcat(row_imgs)
    return grid


def main() -> None:
    root = "/home/limingfei/speckle/donn/datasets/caltectPedestrains"
    ds = CaltechSeqDataset(root, split="Train", set_id=0, video_id=1, label_filter="people")

    annotated: List[np.ndarray] = []
    for i, (img, centers) in enumerate(ds):
        vis = draw_centers(img, centers)
        # annotate count
        cv2.putText(vis, f"centers: {len(centers)}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        annotated.append(vis)
        if len(annotated) >= 9:
            break

    if len(annotated) < 9:
        raise RuntimeError("Not enough frames to build a 3x3 grid")

    grid = make_grid(annotated[:9], rows=3, cols=3)
    out_path = "/home/limingfei/speckle/donn/caltech_grid_3x3.png"
    cv2.imwrite(out_path, grid)
    print(f"Saved grid to: {out_path}")


if __name__ == "__main__":
    main()


