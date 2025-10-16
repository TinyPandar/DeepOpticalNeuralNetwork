import os
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from scipy.io import loadmat


def _load_vbb(vbb_path: str):
    data = loadmat(vbb_path, squeeze_me=True, struct_as_record=False)
    return data["A"]


def _iter_frame_boxes(A) -> Iterator[Tuple[int, List[Tuple[float, float, float, float]], List[int]]]:
    """
    Yield per-frame list of boxes and ids from VBB struct A.

    Returns (frame_index, [x, y, w, h] list, [id] list)
    """
    n_frame = int(A.nFrame)
    for frame_idx in range(n_frame):
        L = A.objLists[frame_idx]
        if L is None:
            yield frame_idx, [], []
            continue
        if isinstance(L, np.ndarray):
            objs = L
        else:
            objs = np.array([L])
        boxes: List[Tuple[float, float, float, float]] = []
        ids: List[int] = []
        for O in objs:
            pos = np.asarray(O.pos).astype(float).ravel()
            if pos.size != 4:
                continue
            x, y, w, h = [float(v) for v in pos]
            if w <= 0 or h <= 0:
                continue
            boxes.append((x, y, w, h))
            ids.append(int(getattr(O, "id", 0)))
        yield frame_idx, boxes, ids


def _box_to_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


class CaltechSeqDataset:
    """
    Minimal iterable over Caltech Pedestrians .seq video and .vbb annotations.

    Produces tuples (image_bgr: np.ndarray, centers: np.ndarray[K,2]) per frame.
    """

    def __init__(
        self,
        root: str,
        split: str = "Train",
        set_id: Optional[int] = None,
        video_id: Optional[int] = None,
        label_filter: Optional[str] = None,
    ) -> None:
        """
        - root: dataset root folder that contains Train/, Test/, annotations/
        - split: "Train" or "Test"
        - set_id: if provided, restrict to setXX
        - video_id: if provided, restrict to VYYY
        - label_filter: if provided, use obj id -> A.objLbl mapping to keep only that label (e.g., "person"/"people")
        """
        self.root = root
        self.split = split
        self.set_id = set_id
        self.video_id = video_id
        self.label_filter = label_filter

        # Paths
        self.split_dir = os.path.join(root, split)
        # In this dataset dump annotations are nested as annotations/annotations/setXX/VYYY.vbb
        self.ann_root = os.path.join(root, "annotations", "annotations")

        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Split dir not found: {self.split_dir}")
        if not os.path.isdir(self.ann_root):
            # fallback to annotations root directly
            alt = os.path.join(root, "annotations")
            if os.path.isdir(alt):
                self.ann_root = alt
            else:
                raise FileNotFoundError(f"Annotations dir not found under {root}")

        self.index: List[Tuple[str, str]] = []  # list of (seq_path, vbb_path)

        # Build index
        for name in sorted(os.listdir(self.split_dir)):
            if not name.startswith("set"):
                continue
            if self.set_id is not None and name != f"set{self.set_id:02d}":
                continue
            set_dir = os.path.join(self.split_dir, name, name)
            if not os.path.isdir(set_dir):
                # Some dumps have Train/setXX/VYYY.seq without nested setXX
                set_dir = os.path.join(self.split_dir, name)
            if not os.path.isdir(set_dir):
                continue
            for fn in sorted(os.listdir(set_dir)):
                if not fn.endswith(".seq"):
                    continue
                vid = os.path.splitext(fn)[0]
                if self.video_id is not None and vid != f"V{self.video_id:03d}":
                    continue
                seq_path = os.path.join(set_dir, fn)
                vbb_path = os.path.join(self.ann_root, name, f"{vid}.vbb")
                if os.path.isfile(vbb_path):
                    self.index.append((seq_path, vbb_path))

        if not self.index:
            raise RuntimeError("No (seq,vbb) pairs found for the given filters.")

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for seq_path, vbb_path in self.index:
            A = _load_vbb(vbb_path)
            # id to label mapping from A.objLbl (1-based ids)
            id_to_lbl: Dict[int, str] = {}
            if hasattr(A, "objLbl") and isinstance(A.objLbl, (list, np.ndarray)):
                for i, lbl in enumerate(A.objLbl, start=1):
                    id_to_lbl[i] = str(lbl)

            # Open video
            cap = cv2.VideoCapture(seq_path)
            if not cap.isOpened():
                continue

            # Iterate frames
            for frame_idx, boxes, ids in _iter_frame_boxes(A):
                ok, frame = cap.read()
                if not ok:
                    break

                centers: List[Tuple[float, float]] = []
                for box, oid in zip(boxes, ids):
                    if self.label_filter is not None:
                        lbl = id_to_lbl.get(oid, "")
                        if self.label_filter.lower() not in lbl.lower():
                            continue
                    centers.append(_box_to_center(box))

                centers_arr = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
                yield frame, centers_arr

            cap.release()


def load_first_sample(
    root: str,
    split: str = "Train",
    set_id: Optional[int] = None,
    video_id: Optional[int] = None,
    label_filter: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    ds = CaltechSeqDataset(root, split=split, set_id=set_id, video_id=video_id, label_filter=label_filter)
    for img, centers in ds:
        return img, centers
    raise RuntimeError("Dataset yielded no frames")


