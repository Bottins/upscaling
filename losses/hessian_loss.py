"""Regolarizzatori del 4° ordine contro l'effetto 'scalino' (staircasing).

Il TV e Perona-Malik favoriscono soluzioni piecewise-constant: su immagini
degradate da downsampling i bordi che nascono come rampe dolci vengono
quantizzati in gradini visibili (staircasing). I prior del 4° ordine
penalizzano le derivate seconde e promuovono rampe lineari a tratti.

- reg_hessian (LLT, Lysaker-Lundervold-Tai 2003):
    E(u) = int sqrt( u_xx^2 + 2 u_xy^2 + u_yy^2 + eps )
  Eulero-Lagrange e' un'equazione del 4° ordine in u.

- pde_aniso4 (You-Kaveh 2000 / 4th-order edge-aware):
    residuo di  u_t = -Delta( g(|Delta u|^2) * Delta u ),
  stazionario =>  Delta( g(.) Delta u ) = 0.
  Diffonde le derivate seconde (leviga rampe) senza spianare i bordi netti.
"""
from __future__ import annotations
import torch

from .registry import register
from pde.operators import channel_gradients, _grad


def _hessian_per_channel(net, coords: torch.Tensor, coord_scale: float):
    """Ritorna (u_xx, u_yy, u_xy, laplacian), tutti (N, C), in scala pixel."""
    u, grad_u = channel_gradients(net, coords)            # (N,C), (N,C,2)
    grad_u = grad_u / coord_scale
    C = u.shape[-1]
    uxx = torch.zeros_like(u); uyy = torch.zeros_like(u); uxy = torch.zeros_like(u)
    for c in range(C):
        gx = grad_u[:, c, 0]
        gy = grad_u[:, c, 1]
        ggx = _grad(gx, coords) / coord_scale             # (N, 2) = [uxx, uxy]
        ggy = _grad(gy, coords) / coord_scale             # (N, 2) = [uyx, uyy]
        uxx[:, c] = ggx[:, 0]
        uxy[:, c] = 0.5 * (ggx[:, 1] + ggy[:, 0])
        uyy[:, c] = ggy[:, 1]
    lap = uxx + uyy
    return uxx, uyy, uxy, lap


@register("reg_hessian")
def hessian_loss(net, collocation: torch.Tensor,
                 coord_scale: float = 1.0, eps: float = 1e-6,
                 **_) -> torch.Tensor:
    """LLT: TV dei gradienti -> favorisce rampe, rimuove staircasing."""
    coords = collocation.clone().requires_grad_(True)
    uxx, uyy, uxy, _ = _hessian_per_channel(net, coords, coord_scale)
    # norma Frobenius per pixel, media su canali e punti
    H = torch.sqrt(uxx ** 2 + 2 * uxy ** 2 + uyy ** 2 + eps)
    return H.mean()


@register("pde_aniso4")
def aniso4_loss(net, collocation: torch.Tensor,
                coord_scale: float = 1.0, kappa: float = 0.05,
                **_) -> torch.Tensor:
    """
    PDE anisotropa del 4° ordine (You-Kaveh edge-aware):
        residuo = Delta( g(|Delta u|^2) * Delta u )
    con  g(s) = 1 / (1 + s / kappa^2).  Liscia le rampe, preserva i bordi netti.
    """
    coords = collocation.clone().requires_grad_(True)
    _, _, _, lap = _hessian_per_channel(net, coords, coord_scale)   # (N, C)
    g = 1.0 / (1.0 + (lap ** 2) / (kappa ** 2))
    flow = g * lap                                                  # (N, C)
    # Delta(flow): dobbiamo derivare 2 volte ancora -> riciclo _grad per canale
    res_sq_total = 0.0
    C = flow.shape[-1]
    for c in range(C):
        g1 = _grad(flow[:, c], coords) / coord_scale                # (N, 2)
        gxx = _grad(g1[:, 0], coords)[:, 0] / coord_scale
        gyy = _grad(g1[:, 1], coords)[:, 1] / coord_scale
        r = gxx + gyy
        res_sq_total = res_sq_total + (r ** 2).mean()
    return res_sq_total / C
