import os
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


def _box_to_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def _parse_inria_annotation(txt_path: str) -> List[Tuple[float, float, float, float]]:
    """Parse an INRIA Person annotation .txt file into a list of [x, y, w, h] boxes.

    The INRIA .txt format varies slightly by mirror; we implement a robust parser:
    - Skip blank/comment lines
    - If the first non-empty token is an integer and the line has a single token,
      treat it as a header (object count)
    - For remaining lines, read the first 4 numeric tokens as x y w h
    - Ignore boxes with non-positive width/height
    """
    boxes: List[Tuple[float, float, float, float]] = []
    if not os.path.isfile(txt_path):
        return boxes

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines()]

    header_skipped = False
    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        parts = ln.replace(",", " ").split()
        if not parts:
            continue
        if not header_skipped and len(parts) == 1:
            # Likely an object-count header; skip once
            try:
                int(parts[0])
                header_skipped = True
                continue
            except Exception:
                pass
        # Collect first 4 numeric tokens
        nums: List[float] = []
        for p in parts:
            try:
                nums.append(float(p))
            except Exception:
                # stop at first non-numeric after collecting at least 4 fields
                if len(nums) >= 4:
                    break
        if len(nums) >= 4:
            x, y, w, h = nums[:4]
            if w > 0 and h > 0:
                boxes.append((float(x), float(y), float(w), float(h)))
    return boxes


def _find_candidate_annotation_paths(root: str, split: str, img_dir: str, img_filename: str) -> List[str]:
    stem, _ = os.path.splitext(img_filename)
    candidates: List[str] = []
    # 1) sibling "annotations" directory next to pos/
    sibling_ann = os.path.join(os.path.dirname(img_dir), "annotations", f"{stem}.txt")
    candidates.append(sibling_ann)
    # 2) root/annotations/(optional split)/
    candidates.append(os.path.join(root, "annotations", f"{stem}.txt"))
    candidates.append(os.path.join(root, "annotations", split, f"{stem}.txt"))
    # 3) root/Annotations variants (some mirrors use capital A)
    candidates.append(os.path.join(root, "Annotations", f"{stem}.txt"))
    candidates.append(os.path.join(root, "Annotations", split, f"{stem}.txt"))
    # 4) same directory as image
    candidates.append(os.path.join(img_dir, f"{stem}.txt"))
    # 5) root/split/annotations
    candidates.append(os.path.join(root, split, "annotations", f"{stem}.txt"))
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for p in candidates:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


class InriaPersonDataset:
    """
    Minimal iterable over INRIA Person Dataset positive images and annotations.

    Produces tuples (image_bgr: np.ndarray, centers: np.ndarray[K,2]).

    Expected directory layout (robust to minor variations):
    - root/Train/pos/*.png|*.jpg
    - root/Train/annotations/*.txt   (or sibling to pos/, or root/annotations...)
    - root/Test/pos/*.png|*.jpg
    - root/Test/annotations/*.txt
    """

    def __init__(
        self,
        root: str,
        split: str = "Train",
        include_negatives: bool = False,
    ) -> None:
        self.root = root
        self.split = split
        self.include_negatives = bool(include_negatives)

        self.split_dir = os.path.join(root, split)
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Split dir not found: {self.split_dir}")

        self.pos_dir = os.path.join(self.split_dir, "pos")
        self.neg_dir = os.path.join(self.split_dir, "neg")
        if not os.path.isdir(self.pos_dir):
            raise FileNotFoundError(f"Positive images dir not found: {self.pos_dir}")

        # Build index of positive images (negatives are optional and have no boxes)
        self.pos_images: List[str] = []
        for fn in sorted(os.listdir(self.pos_dir)):
            lf = fn.lower()
            if lf.endswith(".jpg") or lf.endswith(".jpeg") or lf.endswith(".png") or lf.endswith(".bmp"):
                self.pos_images.append(os.path.join(self.pos_dir, fn))

        self.neg_images: List[str] = []
        if self.include_negatives and os.path.isdir(self.neg_dir):
            for fn in sorted(os.listdir(self.neg_dir)):
                lf = fn.lower()
                if lf.endswith(".jpg") or lf.endswith(".jpeg") or lf.endswith(".png") or lf.endswith(".bmp"):
                    self.neg_images.append(os.path.join(self.neg_dir, fn))

        if not self.pos_images and not self.neg_images:
            raise RuntimeError("No images found for INRIA dataset. Check directory structure.")

    def __len__(self) -> int:
        return len(self.pos_images) + len(self.neg_images)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        # Iterate positives with annotations
        for img_path in self.pos_images:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            dirname = os.path.dirname(img_path)
            fname = os.path.basename(img_path)
            centers: List[Tuple[float, float]] = []
            # find annotation file
            for cand in _find_candidate_annotation_paths(self.root, self.split, dirname, fname):
                if os.path.isfile(cand):
                    boxes = _parse_inria_annotation(cand)
                    if boxes:
                        centers = [_box_to_center(b) for b in boxes]
                    break
            centers_arr = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
            yield img, centers_arr

        # Optionally iterate negatives (no annotations). Training loop may skip these.
        for img_path in self.neg_images:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            centers_arr = np.empty((0, 2), dtype=np.float32)
            yield img, centers_arr


__all__ = ["InriaPersonDataset"]


