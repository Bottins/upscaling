"""Split e caricamento di dataset di immagini HR."""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


IMG_EXT = {".png", ".bmp", ".jpg", ".jpeg"}


def list_images(root: Path) -> List[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT)


def split_paths(paths: List[Path], ratios=(0.8, 0.1, 0.1), seed: int = 0
                ) -> Tuple[List[Path], List[Path], List[Path]]:
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    n = len(paths)
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)
    return paths[:n_tr], paths[n_tr:n_tr + n_va], paths[n_tr + n_va:]


def load_image(path: Path, size: Tuple[int, int] | None = None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size[1], size[0]), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # (H, W, 3) -> (3, H, W)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class HRImageDataset(Dataset):
    def __init__(self, paths: List[Path], hr_size: Tuple[int, int]):
        self.paths = list(paths)
        self.hr_size = hr_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return load_image(self.paths[idx], self.hr_size)
