"""Operatore di degradazione P = S.B (blur gaussiano + sottocampionamento) e aggiunto P*."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def gaussian_kernel1d(sigma: float, truncate: float = 4.0, device=None, dtype=torch.float32) -> torch.Tensor:
    radius = max(1, int(truncate * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """img: (B, C, H, W) o (C, H, W)."""
    squeeze = img.dim() == 3
    if squeeze:
        img = img.unsqueeze(0)
    C = img.size(1)
    k = gaussian_kernel1d(sigma, device=img.device, dtype=img.dtype)
    kx = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    pad = k.numel() // 2
    img = F.pad(img, (pad, pad, pad, pad), mode="reflect")
    img = F.conv2d(img, kx, groups=C)
    img = F.conv2d(img, ky, groups=C)
    return img.squeeze(0) if squeeze else img


def downsample(img: torch.Tensor, scale: int) -> torch.Tensor:
    """Sottocampionamento di tipo stride (S): prende 1 pixel ogni 'scale'."""
    return img[..., ::scale, ::scale].contiguous()


def upsample_adjoint(img: torch.Tensor, scale: int, out_hw) -> torch.Tensor:
    """Aggiunto del sottocampionamento stride: zero-insertion."""
    if img.dim() == 3:
        C, h, w = img.shape
        out = torch.zeros(C, out_hw[0], out_hw[1], device=img.device, dtype=img.dtype)
        out[..., ::scale, ::scale] = img
        return out
    B, C, h, w = img.shape
    out = torch.zeros(B, C, out_hw[0], out_hw[1], device=img.device, dtype=img.dtype)
    out[..., ::scale, ::scale] = img
    return out


def apply_P(u_hr: torch.Tensor, sigma: float, scale: int) -> torch.Tensor:
    """P u = S ( B u )."""
    return downsample(gaussian_blur(u_hr, sigma), scale)


def apply_P_adjoint(y_lr: torch.Tensor, sigma: float, scale: int, hr_hw) -> torch.Tensor:
    """P* y = B^T ( S^T y ). Il kernel gaussiano e' simmetrico, quindi B^T = B."""
    return gaussian_blur(upsample_adjoint(y_lr, scale, hr_hw), sigma)


def dot_product_test(H: int, W: int, sigma: float, scale: int, device="cpu") -> float:
    """Test di coerenza: <Pu, v> ≈ <u, P*v>. Deve essere ~0 a meno dell'errore float."""
    torch.manual_seed(0)
    u = torch.randn(3, H, W, device=device)
    v = torch.randn(3, H // scale, W // scale, device=device)
    lhs = (apply_P(u, sigma, scale) * v).sum().item()
    rhs = (u * apply_P_adjoint(v, sigma, scale, (H, W))).sum().item()
    return abs(lhs - rhs) / (abs(lhs) + 1e-12)
