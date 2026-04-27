"""Termini PDE: Perona-Malik scalare e diffusione anisotropa vettoriale.

NOTA: le coordinate sono in [0,1]^2, ma l'immagine ha H x W pixel, quindi lo
'spacing' fisico tra pixel e' 1/H. Il residuo div(D . grad u) calcolato in
coord [0,1]^2 e' H^2 volte quello in coord-pixel. Per ottenere magnitudini
O(1) e pesi interpretabili, normalizziamo dividendo per coord_scale^2 = H^2.
"""
from __future__ import annotations
import torch

from .registry import register
from pde.diffusion import perona_malik_residual, anisotropic_tensor_residual


@register("pde_perona_malik")
def pm_loss(net, collocation: torch.Tensor, pm_kappa: float = 0.05,
            coord_scale: float = 1.0, **_) -> torch.Tensor:
    coords = collocation.clone().requires_grad_(True)
    res = perona_malik_residual(net, coords, kappa=pm_kappa)
    res = res / (coord_scale ** 2)
    return (res ** 2).mean()


@register("pde_anisotropic")
def aniso_loss(net, collocation: torch.Tensor, eig_clip=(1e-3, 1.0),
               struct_eps: float = 1e-4, coord_scale: float = 1.0,
               **_) -> torch.Tensor:
    coords = collocation.clone().requires_grad_(True)
    res = anisotropic_tensor_residual(net, coords, eig_clip=eig_clip,
                                      struct_eps=struct_eps)
    res = res / (coord_scale ** 2)
    return (res ** 2).mean()


@register("pde_ZG")
def ZG_loss(net, collocation: torch.Tensor, pm_kappa: float = 0.05,
            coord_scale: float = 1.0, **_) -> torch.Tensor:
    coords = collocation.clone().requires_grad_(True)
    res = ZG_residual(net, coords, kappa=pm_kappa)
    res = res / (coord_scale ** 2)
    return (res ** 2).mean()
