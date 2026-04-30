"""Residui PDE di diffusione che accoppiano i canali RGB."""
from __future__ import annotations
import torch
import math

from .operators import channel_gradients, divergence


def perona_malik_residual(net, coords: torch.Tensor, kappa: float = 0.05,) -> torch.Tensor:
    """
    div( g(|grad u|^2) * grad u ) = 0, con g(s) = 1 / (1 + s / kappa^2).
    Canali indipendenti (baseline).
    coords: (N, 2) con requires_grad=True.
    Ritorna: (N, 3) residuo per canale.
    """
    u, grad_u = channel_gradients(net, coords)            # (N,3), (N,3,2)
    g_sq = (grad_u ** 2).sum(dim=-1, keepdim=True)         # (N,3,1)
    g = 1.0 / (1.0 + g_sq / (kappa ** 2))                  # (N,3,1)
    flux = g * grad_u                                      # (N,3,2)
    return divergence(flux, coords)                        # (N,3)


def _structure_tensor(grad_u: torch.Tensor) -> torch.Tensor:
    """
    G(u) = sum_c grad u_c * grad u_c^T  (2x2 simmetrico)
    grad_u: (N, 3, 2) -> (N, 2, 2)
    """
    # (N, C, 2, 1) @ (N, C, 1, 2) = (N, C, 2, 2) -> somma canali
    gu = grad_u.unsqueeze(-1)                          # (N, 3, 2, 1)
    G = (gu @ gu.transpose(-1, -2)).sum(dim=1)         # (N, 2, 2)
    return G


def _diffusion_tensor(G: torch.Tensor, eig_clip=(1e-3, 1.0),
                      eps: float = 1e-4) -> torch.Tensor:
    """
    D = Q diag(g1, g2) Q^T, con g_i funzione decrescente degli autovalori di G.
    Aggiunge eps*I per stabilizzare la decomposizione quando G ~ 0.
    """
    # Simmetrizza + regolarizza per stabilita' numerica
    G = 0.5 * (G + G.transpose(-1, -2))
    eye = torch.eye(2, device=G.device, dtype=G.dtype).expand_as(G)
    G = G + eps * eye
    evals, evecs = torch.linalg.eigh(G)                # evals: (N, 2), evecs: (N, 2, 2)
    # funzione di diffusione: piu' diffusione dove l'autovalore e' piccolo
    g_vals = 1.0 / (1.0 + evals)
    lo, hi = eig_clip
    g_vals = g_vals.clamp(min=lo, max=hi)
    # D = Q diag(g) Q^T
    D = evecs @ torch.diag_embed(g_vals) @ evecs.transpose(-1, -2)
    return D                                           # (N, 2, 2)


def anisotropic_tensor_residual(net, coords: torch.Tensor, eig_clip=(1e-3, 1.0),
                                struct_eps: float = 1e-4) -> torch.Tensor:
    """
    div( D(u) * grad u_c ) = 0 per ogni canale c, con D tensore 2x2 condiviso.
    I canali sono accoppiati attraverso il tensore di struttura G(u).
    Ritorna: (N, 3)
    NB: per un vero smoothing gaussiano di G (G_sigma) serve un campionamento su griglia;
    qui usiamo G puntuale (sufficiente come baseline in modalita' collocation).
    """
    _, grad_u = channel_gradients(net, coords)         # (N, 3, 2)
    G = _structure_tensor(grad_u)                      # (N, 2, 2)
    D = _diffusion_tensor(G, eig_clip, eps=struct_eps) # (N, 2, 2)
    # flux_c = D @ grad_u_c  -> (N, 3, 2)
    flux = torch.einsum("nij,ncj->nci", D, grad_u)
    return divergence(flux, coords)                    # (N, 3)


def ZG_residual(net, coords: torch.Tensor, kappa: float = 0.05, F: float = 0.5, B: float = 0.1) -> torch.Tensor:
    """
    div( g(|grad u|) * grad u ) - bih( u ) = 0, con g(s) = 1 / (1 + s / kappa^2) -- g(s) di Perona-Malik
    Canali indipendenti (baseline).
    coords: (N, 2) con requires_grad=True.
    Ritorna: (N, 3) residuo per canale.
    """
    u, grad_u = channel_gradients(net, coords)            # (N,3), (N,3,2)
    g_mod = torch.linalg.norm(grad_u)                     # |grad u|
    g = 1.0 / (1.0 + g_mod / (kappa ** 2))                 # g( |grad u| )
    flux = g * grad_u                                      # (N,3,2)
    d_flux = F * divergence(flux, coords)                      # (N,3)
    d_grad_u = divergence(grad_u, coords)                   # (N,3)
    _, gd_grad_u = channel_gradients(net,coords)           # (N,3,2)
    bih_u = B * divergence(gd_grad_u,coords)                   # (N,3)
    return d_flux - bih_u                                  # (N,3)
