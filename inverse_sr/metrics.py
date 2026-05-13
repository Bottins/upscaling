from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((pred.clamp(0, 1) - target.clamp(0, 1)) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def _gaussian_kernel(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    axis = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    kernel_1d = torch.exp(-0.5 * (axis / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    return kernel_1d[:, None] * kernel_1d[None, :]


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    max_val: float = 1.0,
) -> float:
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    channels = pred.shape[1]

    kernel = _gaussian_kernel(window_size, sigma, pred.device, pred.dtype)
    kernel = kernel.expand(channels, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    def _filter(x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x, kernel, groups=channels)

    mu_pred = _filter(pred)
    mu_target = _filter(target)
    mu_pred_sq = mu_pred.square()
    mu_target_sq = mu_target.square()
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = _filter(pred * pred) - mu_pred_sq
    sigma_target_sq = _filter(target * target) - mu_target_sq
    sigma_pred_target = _filter(pred * target) - mu_pred_target

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    return (numerator / denominator).mean().item()


def compute_all(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
    }
