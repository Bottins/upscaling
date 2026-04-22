"""Termini di fedelta' ai dati."""
from __future__ import annotations
import torch

from .registry import register
from degradation.operator import apply_P
from data.single_image import make_coord_grid


@register("data_lr")
def data_lr_loss(net, y_lr: torch.Tensor, hr_hw, sigma: float, scale: int,
                 **_) -> torch.Tensor:
    """
    || P u_theta - y ||^2 su griglia HR completa.
    y_lr: (3, h, w).
    """
    H, W = hr_hw
    coords = make_coord_grid(H, W, device=y_lr.device)
    u = net(coords).view(H, W, 3).permute(2, 0, 1)            # (3, H, W)
    y_pred = apply_P(u, sigma, scale)
    return ((y_pred - y_lr) ** 2).mean()


@register("data_points")
def data_points_loss(net, data_coords: torch.Tensor, data_rgb: torch.Tensor,
                     **_) -> torch.Tensor:
    """
    Loss puntuale: valuta la rete nelle coordinate HR dei pixel LR riposizionati
    e confronta con i valori RGB LR. Ottimo nel regime single-image.
    """
    pred = net(data_coords)
    return ((pred - data_rgb) ** 2).mean()
