"""Costruisce griglie visive di confronto tra metodi."""
from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt
from pathlib import Path

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
    for name, data_tuple in results.items():
        pred, metrics = data_tuple[0], data_tuple[1]
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
    for name, data_tuple in results.items():
        m = data_tuple[1] 
        lines.append(f"| {name} | {m['psnr']:.2f} | {m['ssim']:.4f} |")
    return "\n".join(lines)



def plot_loss_history(history: dict, title: str, save_path: str):
    """
    Crea un plot con due subplot:
    1. L'andamento del PSNR nel tempo.
    2. L'andamento delle singole loss (in scala logaritmica).
    """
    if not history:
        return
        
    epochs = range(len(history["psnr"]))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # --- Subplot 1: PSNR ---
    ax1.plot(epochs, history["psnr"], color='blue', label='PSNR (dB)')
    ax1.set_title(f"PSNR vs Epochs - {title}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("PSNR (dB)")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # --- Subplot 2: Termini di Loss ---
    for name, values in history.items():
        if name == "psnr":
            continue
        # Evitiamo di plottare la 'total' se vogliamo vedere solo i singoli termini, 
        # oppure la lasciamo in nero spesso.
        if name == "total":
            ax2.plot(epochs, values, color='black', linewidth=2, label='Total Loss', linestyle='--')
        else:
            ax2.plot(epochs, values, label=name)
            
    ax2.set_title(f"Loss Components vs Epochs (Log Scale)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss Value")
    ax2.set_yscale('log') # Scala logaritmica essenziale per vedere le loss fisiche
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()