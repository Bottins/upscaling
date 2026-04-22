"""Supporto per training single-image: LR -> punti dati in coordinate HR."""
from __future__ import annotations
from typing import Tuple
import torch


def make_coord_grid(H: int, W: int, device="cpu") -> torch.Tensor:
    """Restituisce coordinate normalizzate in [0, 1]^2, shape (H*W, 2) -> (x, y)."""
    ys = torch.linspace(0.0, 1.0, H, device=device)
    xs = torch.linspace(0.0, 1.0, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


def lr_pixels_as_data_points(lr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converte un'immagine LR (3, h, w) in coppie (coords_hr, rgb).
    I pixel LR sono *riposizionati* nel dominio continuo [0,1]^2 ai centri di cella:
    coord_x = (i + 0.5) / w, coord_y = (j + 0.5) / h.
    Cosi' possono essere valutati dalla PINN che vive su coordinate HR.
    """
    assert lr.dim() == 3 and lr.size(0) == 3
    _, h, w = lr.shape
    ys = (torch.arange(h, device=lr.device).float() + 0.5) / h
    xs = (torch.arange(w, device=lr.device).float() + 0.5) / w
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)   # (h*w, 2)
    rgb = lr.permute(1, 2, 0).reshape(-1, 3)                          # (h*w, 3)
    return coords, rgb


def sample_collocation(n: int, device="cpu") -> torch.Tensor:
    """Punti casuali in [0,1]^2 per il residuo PDE."""
    return torch.rand(n, 2, device=device)


def sample_data_points(coords: torch.Tensor, rgb: torch.Tensor, n: int
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sottocampiona n coppie (coord, rgb) dal set totale LR."""
    if n >= coords.shape[0]:
        return coords, rgb
    idx = torch.randperm(coords.shape[0], device=coords.device)[:n]
    return coords[idx], rgb[idx]
