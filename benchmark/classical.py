"""Baseline numeriche classiche per SR: Bicubic, TV-denoise, TV-ROF (deblur+SR).

- Bicubic: semplice upsampling.
- TV-denoise: upsampling bicubico seguito da denoising TV (Chambolle) su ciascun
  canale. Non usa il modello di degradazione.
- TV-ROF (o TV-L2 inverso): minimizza  0.5||P u - y||^2 + lam * TV(u)
  via proximal gradient (FISTA) con operatore P = blur + downsample.
  Riferimento: Beck & Teboulle 2009, ``Fast gradient-based algorithms for
  constrained total variation image denoising and deblurring problems''.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from degradation.operator import apply_P, apply_P_adjoint


# ---------------------------------------------------------------------- utils
def _grad(u: torch.Tensor):
    """Forward differences con replicate al bordo. u: (C,H,W)."""
    gx = torch.zeros_like(u)
    gy = torch.zeros_like(u)
    gx[..., :, :-1] = u[..., :, 1:] - u[..., :, :-1]
    gy[..., :-1, :] = u[..., 1:, :] - u[..., :-1, :]
    return gx, gy


def _div(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    """Divergenza aggiunta di _grad (backward differences)."""
    dx = torch.zeros_like(px)
    dy = torch.zeros_like(py)
    dx[..., :, 1:-1] = px[..., :, 1:-1] - px[..., :, :-2]
    dx[..., :, 0] = px[..., :, 0]
    dx[..., :, -1] = -px[..., :, -2]
    dy[..., 1:-1, :] = py[..., 1:-1, :] - py[..., :-2, :]
    dy[..., 0, :] = py[..., 0, :]
    dy[..., -1, :] = -py[..., -2, :]
    return dx + dy


# -------------------------------------------------------------------- methods
def bicubic(lr: torch.Tensor, scale: int) -> torch.Tensor:
    H, W = lr.shape[-2] * scale, lr.shape[-1] * scale
    u = F.interpolate(lr.unsqueeze(0), size=(H, W),
                      mode="bicubic", align_corners=False).squeeze(0)
    return u.clamp(0, 1)


def tv_chambolle(img: torch.Tensor, lam: float = 0.1, n_iter: int = 100) -> torch.Tensor:
    """Denoising TV (Chambolle 2004) sull'input img (C,H,W). Lavora per-canale."""
    tau = 1.0 / 8.0
    x = img.clone()
    C, H, W = x.shape
    p = torch.zeros((C, 2, H, W), device=x.device, dtype=x.dtype)
    for _ in range(n_iter):
        # div p - img/lam
        div_p = _div(p[:, 0], p[:, 1])
        g = div_p - img / lam
        gx, gy = _grad(g)
        denom = 1.0 + tau * torch.sqrt(gx * gx + gy * gy)
        p[:, 0] = (p[:, 0] + tau * gx) / denom
        p[:, 1] = (p[:, 1] + tau * gy) / denom
    return (img - lam * _div(p[:, 0], p[:, 1])).clamp(0, 1)


def tv_denoise(lr: torch.Tensor, scale: int, lam: float = 0.05,
               n_iter: int = 100) -> torch.Tensor:
    """Bicubic + denoise TV. Semplice baseline 'classica'."""
    return tv_chambolle(bicubic(lr, scale), lam=lam, n_iter=n_iter)


def tv_rof_sr(lr: torch.Tensor, scale: int, sigma: float = 1.0,
              lam: float = 0.01, n_iter: int = 200,
              inner_tv: int = 10) -> torch.Tensor:
    """TV-ROF per inverse SR: min 0.5||P u - y||^2 + lam TV(u).

    Risolto con proximal gradient (ISTA): passo gradiente su fidelita' +
    prox TV (chambolle) su residuo.
    """
    H = lr.shape[-2] * scale
    W = lr.shape[-1] * scale
    u = bicubic(lr, scale).clone()
    # stima grossolana della Lipschitz: ||P^T P|| <= 1 con downsample medio
    step = 1.0
    for _ in range(n_iter):
        y_pred = apply_P(u, sigma, scale)
        grad = apply_P_adjoint(y_pred - lr, sigma, scale, (H, W))
        u = u - step * grad
        u = tv_chambolle(u.clamp(0, 1), lam=lam, n_iter=inner_tv)
    return u.clamp(0, 1)
