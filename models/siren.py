"""SIREN: MLP con attivazioni sinusoidali (Sitzmann et al. 2020)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int, is_first: bool = False, w0: float = 30.0):
        super().__init__()
        self.w0 = w0
        self.is_first = is_first
        self.linear = nn.Linear(in_f, out_f)
        self._init_weights(in_f)

    def _init_weights(self, in_f: int) -> None:
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1.0 / in_f, 1.0 / in_f)
            else:
                bound = math.sqrt(6.0 / in_f) / self.w0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


class SIREN(nn.Module):
    def __init__(self, in_dim: int = 2, out_dim: int = 3, hidden: int = 256,
                 num_layers: int = 5, w0: float = 30.0):
        super().__init__()
        layers = [SineLayer(in_dim, hidden, is_first=True, w0=w0)]
        for _ in range(num_layers - 2):
            layers.append(SineLayer(hidden, hidden, w0=w0))
        final = nn.Linear(hidden, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden) / w0
            final.weight.uniform_(-bound, bound)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (..., 2) in [0,1]^2 -> RGB in circa [-1, 1], rimappato a [0,1]."""
        x = 2.0 * coords - 1.0
        y = self.net(x)
        return 0.5 * (y + 1.0)
