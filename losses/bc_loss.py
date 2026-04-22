"""Condizioni al bordo di Neumann: gradiente normale nullo sul bordo di [0,1]^2."""
from __future__ import annotations
import torch

from .registry import register
from pde.operators import channel_gradients


@register("bc_neumann")
def neumann_bc_loss(net, n_bc: int = 512, device="cpu",
                    coord_scale: float = 1.0, **_) -> torch.Tensor:
    t = torch.rand(n_bc, device=device)
    zeros = torch.zeros_like(t)
    ones = torch.ones_like(t)
    # 4 bordi: x=0, x=1, y=0, y=1
    left   = torch.stack([zeros, t], dim=-1)
    right  = torch.stack([ones,  t], dim=-1)
    bottom = torch.stack([t, zeros], dim=-1)
    top    = torch.stack([t, ones ], dim=-1)

    def _normal_grad(pts: torch.Tensor, axis: int) -> torch.Tensor:
        pts = pts.clone().requires_grad_(True)
        _, grad_u = channel_gradients(net, pts)           # (N, 3, 2)
        return grad_u[..., axis] / coord_scale            # (N, 3)

    r = _normal_grad(left,   0) ** 2
    r = r + _normal_grad(right,  0) ** 2
    r = r + _normal_grad(bottom, 1) ** 2
    r = r + _normal_grad(top,    1) ** 2
    return r.mean()
