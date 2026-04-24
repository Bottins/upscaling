"""Piccolo CNN SR (EDSR-baseline stile) per prior appreso su DIV2K."""
from __future__ import annotations
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, ch: int, res_scale: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.body(x)


class PixelShuffleUp(nn.Module):
    def __init__(self, ch: int, scale: int):
        super().__init__()
        layers = []
        s = scale
        while s > 1:
            step = 2 if s % 2 == 0 else 3
            layers += [nn.Conv2d(ch, ch * step * step, 3, padding=1),
                       nn.PixelShuffle(step)]
            s //= step
        self.up = nn.Sequential(*layers)

    def forward(self, x):
        return self.up(x)


class SmallEDSR(nn.Module):
    """SR net con n_blocks residual blocks e upsampler pixel-shuffle."""
    def __init__(self, scale: int = 4, n_blocks: int = 8, ch: int = 64):
        super().__init__()
        self.head = nn.Conv2d(3, ch, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(n_blocks)],
                                  nn.Conv2d(ch, ch, 3, padding=1))
        self.up = PixelShuffleUp(ch, scale)
        self.tail = nn.Conv2d(ch, 3, 3, padding=1)
        self.scale = scale

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        x = self.head(lr)
        x = self.body(x) + x
        x = self.up(x)
        return self.tail(x)
