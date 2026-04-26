"""Metriche di valutazione: PSNR, SSIM (implementazione senza dipendenze)."""
from __future__ import annotations
import math
from typing import Dict

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((pred.clamp(0, 1) - target.clamp(0, 1)) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def _gaussian_kernel(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    ax = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    g1d = g1d / g1d.sum()
    g2d = g1d[:, None] * g1d[None, :]
    return g2d


def ssim(pred: torch.Tensor, target: torch.Tensor,
         window_size: int = 11, sigma: float = 1.5,
         max_val: float = 1.0) -> float:
    """SSIM medio su canali (default Wang 2004). pred,target: (C,H,W) in [0,1]."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    C = pred.shape[1]
    k = _gaussian_kernel(window_size, sigma, pred.device, pred.dtype)
    k = k.expand(C, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    def _flt(x):
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x, k, groups=C)

    mu1 = _flt(pred)
    mu2 = _flt(target)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2
    sig1_sq = _flt(pred * pred) - mu1_sq
    sig2_sq = _flt(target * target) - mu2_sq
    sig12 = _flt(pred * target) - mu12

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    num = (2 * mu12 + C1) * (2 * sig12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2)
    return (num / den).mean().item()


def compute_all(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    return {"psnr": psnr(pred, gt), "ssim": ssim(pred, gt)}
