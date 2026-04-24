"""Costruisce griglie visive di confronto tra metodi."""
from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def _to_np(t: torch.Tensor) -> np.ndarray:
    return (t.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255
            ).astype(np.uint8)


def save_grid(results: Dict[str, Tuple[torch.Tensor, Dict[str, float]]],
              hr: torch.Tensor, lr: torch.Tensor, path: str,
              n_cols: int = 4, title: str = "Benchmark SR") -> None:
    """results: {name: (pred_hr, metrics)}. Aggiunge HR e LR come riferimenti."""
    H, W = hr.shape[-2:]
    panels = []
    # HR e LR (nearest-up) in testa
    lr_up = F.interpolate(lr.unsqueeze(0), size=(H, W),
                          mode="nearest").squeeze(0)
    panels.append(("HR (ground truth)", _to_np(hr), None))
    panels.append(("LR (nearest)", _to_np(lr_up), None))
    for name, (pred, metrics) in results.items():
        panels.append((name, _to_np(pred), metrics))

    n = len(panels)
    rows = (n + n_cols - 1) // n_cols
    pad = 8
    label_h = 36
    cell_h = H + label_h
    cell_w = W
    grid_h = rows * cell_h + (rows - 1) * pad + 48
    grid_w = n_cols * cell_w + (n_cols - 1) * pad
    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 245

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    draw.text((8, 8), title, fill=(0, 0, 0))

    for idx, (name, arr, metrics) in enumerate(panels):
        r, c = divmod(idx, n_cols)
        y0 = 48 + r * (cell_h + pad)
        x0 = c * (cell_w + pad)
        label = name
        if metrics is not None:
            label = (f"{name}  PSNR={metrics['psnr']:.2f}dB  "
                     f"SSIM={metrics['ssim']:.3f}")
        draw.text((x0 + 4, y0 + 4), label, fill=(0, 0, 0))
        img.paste(Image.fromarray(arr), (x0, y0 + label_h))
    img.save(path)


def markdown_table(results: Dict[str, Tuple[torch.Tensor, Dict[str, float]]]) -> str:
    lines = ["| Method | PSNR (dB) | SSIM |", "|---|---|---|"]
    for name, (_, m) in results.items():
        lines.append(f"| {name} | {m['psnr']:.2f} | {m['ssim']:.4f} |")
    return "\n".join(lines)
