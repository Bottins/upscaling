"""Prior e residui che promuovono bordi netti.

- reg_tv      : Total Variation vettoriale (gradienti sparsi)
- pde_shock   : shock filter di Osher-Rudin, residuo  sign(u_eta_eta) * |grad u|

Entrambi sono da COMBINARE con un termine dati (data_lr o data_points):
da soli tendono a soluzioni banali (costante / piecewise constant).
"""
from __future__ import annotations
import torch

from .registry import register
from pde.operators import channel_gradients, _grad


@register("reg_tv")
def tv_loss(net, collocation: torch.Tensor,
            coord_scale: float = 1.0, eps: float = 1e-6, **_) -> torch.Tensor:
    """
    TV_vettoriale(u) = E[ sqrt( sum_c ||grad u_c||^2 + eps ) ]
    Minimizzare TV favorisce gradienti sparsi -> bordi netti, interno piatto.
    """
    coords = collocation.clone().requires_grad_(True)
    _, grad_u = channel_gradients(net, coords)          # (N, 3, 2)
    grad_u = grad_u / coord_scale                        # in scala pixel
    mag = torch.sqrt((grad_u ** 2).sum(dim=(-2, -1)) + eps)   # (N,)
    return mag.mean()


@register("pde_shock")
def shock_loss(net, collocation: torch.Tensor,
               coord_scale: float = 1.0, eps: float = 1e-3, **_) -> torch.Tensor:
    """
    Residuo shock filter: r = sign_smooth(u_eta_eta) * |grad u|
    dove eta = grad u / |grad u| (direzione del gradiente) e u_eta_eta e' la
    seconda derivata lungo eta. sign_smooth = tanh( . / eps) per differenziabilita'.
    Minimizzare r^2 concentra la soluzione su piecewise constant con shock nette.
    Lavora per canale e media sui canali.
    """
    coords = collocation.clone().requires_grad_(True)
    u, grad_u = channel_gradients(net, coords)           # (N, 3), (N, 3, 2)
    grad_u = grad_u / coord_scale
    res_sq_total = 0.0
    for c in range(u.shape[-1]):
        gc = grad_u[:, c, :]                             # (N, 2)
        mag = torch.sqrt((gc ** 2).sum(dim=-1, keepdim=True) + 1e-8)  # (N,1)
        eta = gc / mag                                   # (N, 2)
        # Calcola Hessian . eta via doppio autograd: grad (grad u . eta)
        gc_eta = (gc * eta).sum(dim=-1)                  # (N,)
        g2 = _grad(gc_eta, coords) / coord_scale         # (N, 2)
        u_eta_eta = (g2 * eta).sum(dim=-1)               # (N,)
        s = torch.tanh(u_eta_eta / eps)
        r = s * mag.squeeze(-1)
        res_sq_total = res_sq_total + (r ** 2).mean()
    return res_sq_total / u.shape[-1]
