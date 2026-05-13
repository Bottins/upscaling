from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NoiseTruth:
    gaussian_std: float = 0.03
    laplace_scale: float = 0.02
    speckle_std: float = 0.05
    weights: tuple[float, float, float] = (0.45, 0.30, 0.25)


@dataclass(frozen=True)
class SyntheticDegradation:
    scale: int = 4
    sigma: float = 1.35
    noise: NoiseTruth = NoiseTruth()
    seed: int = 0


def gaussian_kernel1d(
    sigma: torch.Tensor | float,
    radius: int = 9,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), device=device, dtype=dtype)
    sigma = sigma.to(device=device, dtype=dtype).clamp_min(1e-3)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_blur(
    img: torch.Tensor,
    sigma: torch.Tensor | float,
    radius: int = 9,
) -> torch.Tensor:
    if torch.is_tensor(sigma):
        sigma_scalar = float(sigma.detach().cpu().item())
    else:
        sigma_scalar = float(sigma)
    if sigma_scalar <= 1e-3:
        return img

    squeeze = img.dim() == 3
    batch = img.unsqueeze(0) if squeeze else img
    channels = batch.shape[1]

    kernel_1d = gaussian_kernel1d(
        sigma=sigma,
        radius=radius,
        device=batch.device,
        dtype=batch.dtype,
    )
    kx = kernel_1d.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    ky = kernel_1d.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    pad = radius

    out = F.pad(batch, (pad, pad, pad, pad), mode="reflect")
    out = F.conv2d(out, kx, groups=channels)
    out = F.conv2d(out, ky, groups=channels)
    return out.squeeze(0) if squeeze else out


def downsample_stride(img: torch.Tensor, scale: int) -> torch.Tensor:
    return img[..., ::scale, ::scale].contiguous()


def apply_forward_model(
    hr: torch.Tensor,
    sigma: torch.Tensor | float,
    scale: int,
    radius: int = 9,
) -> torch.Tensor:
    return downsample_stride(gaussian_blur(hr, sigma=sigma, radius=radius), scale)


def _randn_like(tensor: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    return torch.randn(
        tensor.shape,
        generator=generator,
        device=tensor.device,
        dtype=tensor.dtype,
    )


def _rand_like(tensor: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    return torch.rand(
        tensor.shape,
        generator=generator,
        device=tensor.device,
        dtype=tensor.dtype,
    )


def _laplace_noise_like(
    tensor: torch.Tensor,
    scale: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    uniform = _rand_like(tensor, generator).clamp_(1e-6, 1.0 - 1e-6) - 0.5
    return -scale * torch.sign(uniform) * torch.log1p(-2.0 * uniform.abs())


def sample_noise_mixture(
    clean_lr: torch.Tensor,
    truth: NoiseTruth,
    seed: int,
) -> torch.Tensor:
    weights = torch.tensor(truth.weights, dtype=clean_lr.dtype, device=clean_lr.device)
    weights = weights / weights.sum()

    generator = torch.Generator(device=clean_lr.device.type)
    generator.manual_seed(seed)

    choices = torch.multinomial(weights, clean_lr.numel(), replacement=True, generator=generator)
    choices = choices.view_as(clean_lr)

    gaussian_noise = truth.gaussian_std * _randn_like(clean_lr, generator)
    laplace_noise = _laplace_noise_like(clean_lr, truth.laplace_scale, generator)
    speckle_noise = clean_lr * truth.speckle_std * _randn_like(clean_lr, generator)

    noise = torch.zeros_like(clean_lr)
    noise = torch.where(choices == 0, gaussian_noise, noise)
    noise = torch.where(choices == 1, laplace_noise, noise)
    noise = torch.where(choices == 2, speckle_noise, noise)
    return noise


def make_observation(
    hr: torch.Tensor,
    degradation: SyntheticDegradation,
) -> tuple[torch.Tensor, torch.Tensor]:
    clean_lr = apply_forward_model(hr, sigma=degradation.sigma, scale=degradation.scale)
    noisy_lr = (clean_lr + sample_noise_mixture(clean_lr, degradation.noise, degradation.seed)).clamp(0, 1)
    return clean_lr, noisy_lr
