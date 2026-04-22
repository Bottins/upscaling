"""Operatori differenziali via autograd per la PINN."""
from __future__ import annotations
from typing import Tuple
import torch


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Pubblica: gradiente di uno scalare (somma su batch) rispetto a inputs."""
    return _grad(outputs, inputs)


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    g = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]
    return g


def channel_gradients(net, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ritorna (u, grad_u) con:
        u       : (N, 3)
        grad_u  : (N, 3, 2)  -> [dR/dx, dR/dy, dG/dx, dG/dy, dB/dx, dB/dy]
    coords deve avere requires_grad=True.
    """
    assert coords.requires_grad, "coords deve avere requires_grad=True"
    u = net(coords)                     # (N, 3)
    grads = []
    for c in range(u.shape[-1]):
        g = _grad(u[:, c], coords)      # (N, 2)
        grads.append(g)
    grad_u = torch.stack(grads, dim=1)  # (N, 3, 2)
    return u, grad_u


def divergence(vec: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    vec: (N, 3, 2)  campo vettoriale per canale
    ritorna: (N, 3)  divergenza per canale
    """
    N, C, _ = vec.shape
    div = torch.zeros(N, C, device=coords.device, dtype=coords.dtype)
    for c in range(C):
        dvx = _grad(vec[:, c, 0], coords)[:, 0]   # d/dx
        dvy = _grad(vec[:, c, 1], coords)[:, 1]   # d/dy
        div[:, c] = dvx + dvy
    return div
