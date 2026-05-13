from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_coord_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(0.0, 1.0, steps=height, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, steps=width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).view(-1, 2)


def sample_image_at_coords(image: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    if image.dim() != 3:
        raise ValueError("image deve essere (C,H,W)")
    batch = image.unsqueeze(0)
    grid = coords.view(1, -1, 1, 2) * 2.0 - 1.0
    sampled = F.grid_sample(
        batch,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0, :, :, 0].transpose(0, 1)


class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, is_first: bool = False, w0: float = 30.0):
        super().__init__()
        self.is_first = is_first
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            in_features = self.linear.in_features
            if self.is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / self.w0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


class ResidualSIREN(nn.Module):
    def __init__(
        self,
        base_image: torch.Tensor,
        hidden_dim: int = 128,
        num_layers: int = 4,
        w0: float = 20.0,
        residual_scale: float = 0.15,
    ):
        super().__init__()
        if base_image.dim() != 3:
            raise ValueError("base_image deve essere (C,H,W)")
        self.register_buffer("base_image", base_image)
        self.residual_scale = residual_scale

        layers: list[nn.Module] = [SineLayer(2, hidden_dim, is_first=True, w0=w0)]
        for _ in range(num_layers - 2):
            layers.append(SineLayer(hidden_dim, hidden_dim, w0=w0))
        final = nn.Linear(hidden_dim, 3)
        with torch.no_grad():
            final.weight.zero_()
            final.bias.zero_()
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = 2.0 * coords - 1.0
        base = sample_image_at_coords(self.base_image, coords)
        residual = torch.tanh(self.net(x))
        out = base + self.residual_scale * residual
        # soft clamp into [0,1] preserving gradients near the boundary
        return 0.5 + 0.5 * torch.tanh(2.0 * (out - 0.5))

    def render(self, height: int, width: int) -> torch.Tensor:
        coords = make_coord_grid(
            height=height,
            width=width,
            device=self.base_image.device,
            dtype=self.base_image.dtype,
        )
        values = self(coords)
        return values.transpose(0, 1).view(3, height, width)


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def channel_gradients(model: nn.Module, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = model(coords)
    grads = []
    for channel in range(values.shape[-1]):
        grads.append(grad(values[:, channel], coords))
    grad_values = torch.stack(grads, dim=1)
    return values, grad_values


def divergence(field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    num_points, channels, _ = field.shape
    out = torch.zeros(num_points, channels, device=coords.device, dtype=coords.dtype)
    for channel in range(channels):
        dx = grad(field[:, channel, 0], coords)[:, 0]
        dy = grad(field[:, channel, 1], coords)[:, 1]
        out[:, channel] = dx + dy
    return out
