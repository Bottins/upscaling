"""Estrae patch HR random da DIV2K e sintetizza coppie LR-HR con P = S.B."""
from __future__ import annotations
from pathlib import Path
from typing import List
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from degradation.operator import apply_P


IMG_EXT = {".png", ".bmp", ".jpg", ".jpeg"}


def list_hr_images(root: Path) -> List[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT)


def _load_rgb(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class PatchDataset(Dataset):
    """
    Restituisce (lr, hr) con hr di dimensione (3, patch, patch) e
    lr sintetizzata applicando l'operatore P (blur + downsample).

    Ottimizzazioni:
    - cache opzionale di tutte le HR in RAM
    - scelta random dell'immagine ad ogni sample
    - crop/augment su tensori già caricati
    """
    def __init__(
        self,
        paths: List[Path],
        patch: int = 96,
        scale: int = 4,
        blur_sigma: float = 1.0,
        samples_per_image: int = 8,
        augment: bool = True,
        cache_images: bool = True,
    ):
        self.paths = list(paths)
        self.patch = patch
        self.scale = scale
        self.sigma = blur_sigma
        self.spi = samples_per_image
        self.augment = augment
        self.cache_images = cache_images

        self.images: List[torch.Tensor] | None = None
        if self.cache_images:
            self.images = []
            for p in self.paths:
                img = _load_rgb(p)
                _, h, w = img.shape
                if h >= self.patch and w >= self.patch:
                    self.images.append(img)

            if not self.images:
                raise RuntimeError(
                    f"Nessuna immagine valida abbastanza grande per patch={self.patch}"
                )

            print(f"[patches] cache RAM attiva: {len(self.images)} immagini HR precaricate")
        else:
            print("[patches] cache RAM disattiva: caricamento on-the-fly")

    def __len__(self) -> int:
        return len(self.paths) * self.spi

    def _get_image(self, idx: int) -> torch.Tensor:
        if self.images is not None:
            # Meglio random puro così i worker non rileggono sempre gli stessi sample
            return self.images[random.randrange(len(self.images))]
        path = self.paths[idx % len(self.paths)]
        return _load_rgb(path)

    def _random_crop(self, img: torch.Tensor) -> torch.Tensor:
        _, H, W = img.shape
        p = self.patch
        if H < p or W < p:
            raise ValueError(f"Immagine {img.shape} piu' piccola della patch {p}")
        y = random.randint(0, H - p)
        x = random.randint(0, W - p)
        return img[:, y:y + p, x:x + p]

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            img = torch.flip(img, dims=[-1])
        if random.random() < 0.5:
            img = torch.flip(img, dims=[-2])
        k = random.randint(0, 3)
        if k:
            img = torch.rot90(img, k, dims=[-2, -1])
        return img.contiguous()

    def __getitem__(self, idx: int):
        img = self._get_image(idx)
        hr = self._random_crop(img)
        if self.augment:
            hr = self._augment(hr)

        with torch.no_grad():
            lr = apply_P(hr, self.sigma, self.scale).clamp(0, 1)

        return lr.contiguous(), hr.contiguous()