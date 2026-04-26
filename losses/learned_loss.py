"""Loss che ancora la PINN a un prior SR appreso (CNN pre-addestrato).

Idea: la PINN continua a minimizzare data_lr + PDE, ma in piu' e'
spinta verso l'output del CNN. Il PDE smussa eventuali artefatti del
CNN, il data_lr garantisce fedelta' all'operatore fisico, il prior
appreso porta la conoscenza del dataset DIV2K.
"""
from __future__ import annotations
import torch

from .registry import register
from data.single_image import make_coord_grid


@register("prior_sr")
def prior_sr_loss(net, hr_target: torch.Tensor, hr_hw,
                  prior_mask: torch.Tensor | None = None, **_) -> torch.Tensor:
    """
    ||u_theta - hr_target||^2 sulla griglia HR, dove hr_target e' la
    predizione del CNN appreso (precalcolata una volta). Opzionale mask
    per ridurre peso su regioni incerte.
    """
    H, W = hr_hw
    coords = make_coord_grid(H, W, device=hr_target.device)
    u = net(coords).view(H, W, 3).permute(2, 0, 1)
    if prior_mask is None:
        return ((u - hr_target) ** 2).mean()
    return ((u - hr_target) ** 2 * prior_mask).mean()
