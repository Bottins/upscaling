from __future__ import annotations

import torch

from .degradation import gaussian_blur


def forward_grad(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gx = torch.zeros_like(u)
    gy = torch.zeros_like(u)
    gx[..., :, :-1] = u[..., :, 1:] - u[..., :, :-1]
    gy[..., :-1, :] = u[..., 1:, :] - u[..., :-1, :]
    return gx, gy


def divergence(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    dx = torch.zeros_like(px)
    dy = torch.zeros_like(py)
    dx[..., :, 1:-1] = px[..., :, 1:-1] - px[..., :, :-2]
    dx[..., :, 0] = px[..., :, 0]
    dx[..., :, -1] = -px[..., :, -2]
    dy[..., 1:-1, :] = py[..., 1:-1, :] - py[..., :-2, :]
    dy[..., 0, :] = py[..., 0, :]
    dy[..., -1, :] = -py[..., -2, :]
    return dx + dy


def gradient_magnitude(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    gx, gy = forward_grad(u)
    return torch.sqrt(gx.square() + gy.square() + eps)


def total_variation(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    gx, gy = forward_grad(u)
    mag = torch.sqrt((gx.square() + gy.square()).sum(dim=0) + eps)
    return mag.mean()


def rof_pde_loss(u: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    gx, gy = forward_grad(u)
    mag = torch.sqrt(gx.square() + gy.square() + eps ** 2)
    nx = gx / mag
    ny = gy / mag
    residual = divergence(nx, ny)
    return residual.square().mean()


def perona_malik_loss(u: torch.Tensor, kappa: float = 0.08) -> torch.Tensor:
    gx, gy = forward_grad(u)
    grad_sq = gx.square() + gy.square()
    conductance = 1.0 / (1.0 + grad_sq / (kappa ** 2))
    residual = divergence(conductance * gx, conductance * gy)
    return residual.square().mean()


def shock_filter_loss(u: torch.Tensor, eps: float = 5e-3) -> torch.Tensor:
    gx, gy = forward_grad(u)
    mag = torch.sqrt(gx.square() + gy.square() + 1e-8)
    eta_x = gx / mag
    eta_y = gy / mag

    gxx, gxy = forward_grad(gx)
    gyx, gyy = forward_grad(gy)
    cross = 0.5 * (gxy + gyx)
    u_eta_eta = eta_x * (gxx * eta_x + cross * eta_y) + eta_y * (cross * eta_x + gyy * eta_y)
    residual = torch.tanh(u_eta_eta / eps) * mag
    return residual.square().mean()


def flat_noise_loss(u: torch.Tensor, blur_sigma: float = 1.2, alpha: float = 120.0) -> torch.Tensor:
    smooth = gaussian_blur(u, sigma=blur_sigma)
    highpass = u - smooth
    grad_mag = gradient_magnitude(u).sum(dim=0, keepdim=True)
    flat_mask = torch.exp(-alpha * grad_mag.square())
    return (flat_mask * highpass.square()).mean()


def edge_sharpness_loss(u: torch.Tensor, blur_sigma: float = 0.8, beta: float = 40.0) -> torch.Tensor:
    smooth = gaussian_blur(u, sigma=blur_sigma)
    grad_mag = gradient_magnitude(smooth).sum(dim=0, keepdim=True)
    edge_mask = 1.0 - torch.exp(-beta * grad_mag.square())
    return -(edge_mask * grad_mag).mean()
