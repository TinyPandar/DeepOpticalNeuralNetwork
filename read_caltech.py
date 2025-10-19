import os
from typing import Optional

import numpy as np

from caltech_loader import load_first_sample, CaltechSeqDataset


def main() -> None:
    root = "/home/limingfei/speckle/donn/datasets/caltectPedestrains"
    # Example: iterate a few frames from set00/V001 and print centers
    ds = CaltechSeqDataset(root, split="Train", set_id=0, video_id=1, label_filter="people")
    for i, (img, centers) in enumerate(ds):
        h, w = img.shape[:2]
        print(f"frame {i}: image=({h},{w},3) centers.shape={centers.shape}")
        if centers.size:
            print("first few centers:", centers[: min(5, len(centers))])
        if i >= 4:
            break


if __name__ == "__main__":
    main()


