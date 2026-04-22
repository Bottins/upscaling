"""MLP con Random Fourier Features (Tancik et al. 2020)."""
from __future__ import annotations
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, mapping_size: int, scale: float):
        super().__init__()
        B = torch.randn(in_dim, mapping_size) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * torch.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class FourierMLP(nn.Module):
    def __init__(self, in_dim: int = 2, out_dim: int = 3, hidden: int = 256,
                 num_layers: int = 5, mapping_size: int = 128, scale: float = 10.0):
        super().__init__()
        self.ff = FourierFeatures(in_dim, mapping_size, scale)
        layers = [nn.Linear(2 * mapping_size, hidden), nn.GELU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(self.ff(coords)))
