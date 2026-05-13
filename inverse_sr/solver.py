from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .baselines import bicubic
from .degradation import apply_forward_model
from .metrics import compute_all
from .pinn import ResidualSIREN
from .priors import (
    edge_sharpness_loss,
    flat_noise_loss,
    perona_malik_loss,
    rof_pde_loss,
    shock_filter_loss,
    total_variation,
)


@dataclass
class InverseSolverConfig:
    scale: int = 4
    epochs: int = 250
    device: str = "cpu"
    selection_metric: str = "psnr"
    model_lr: float = 6e-4
    param_lr: float = 2e-2
    prior_lr: float = 5e-3
    log_every: int = 25
    blur_radius: int = 9
    hidden_dim: int = 192
    num_layers: int = 5
    siren_w0: float = 24.0
    residual_scale: float = 0.45
    sigma_min: float = 0.20
    sigma_max: float = 3.00
    sigma_init: float = 1.10
    gaussian_init: float = 0.03
    laplace_init: float = 0.02
    speckle_init: float = 0.02
    lambda_noise_reg: float = 5e-3
    lambda_prior_weight_reg: float = 5e-4
    lambda_data_charbonnier: float = 80.0
    charbonnier_eps: float = 1e-3
    prior_warmup_frac: float = 0.25
    use_cosine_schedule: bool = True
    pm_kappa: float = 0.08
    rof_eps: float = 1e-3
    shock_eps: float = 5e-3
    flat_blur_sigma: float = 1.2
    flat_alpha: float = 120.0
    sharp_blur_sigma: float = 0.8
    sharp_beta: float = 40.0


@dataclass
class SolverResult:
    mode: str
    display_name: str
    reconstruction: torch.Tensor
    metrics: dict[str, float]
    estimated_sigma: float
    estimated_noise: dict[str, float]
    noise_weights: list[float]
    objective: float
    prior_weights: dict[str, float]
    history: list[dict[str, float]] = field(default_factory=list)

    def to_serializable(self) -> dict:
        return {
            "mode": self.mode,
            "display_name": self.display_name,
            "metrics": self.metrics,
            "estimated_sigma": self.estimated_sigma,
            "estimated_noise": self.estimated_noise,
            "noise_weights": self.noise_weights,
            "objective": self.objective,
            "prior_weights": self.prior_weights,
            "history": self.history,
        }


def _inverse_sigmoid(value: float) -> float:
    value = min(max(value, 1e-6), 1.0 - 1e-6)
    return float(torch.logit(torch.tensor(value)).item())


def _inverse_softplus(value: float) -> float:
    x = torch.tensor(value)
    return float((x.exp() - 1.0).log().item())


def _sigma_from_raw(raw: torch.Tensor, cfg: InverseSolverConfig) -> torch.Tensor:
    span = cfg.sigma_max - cfg.sigma_min
    return cfg.sigma_min + span * torch.sigmoid(raw)


def _positive_from_raw(raw: torch.Tensor, floor: float = 1e-4) -> torch.Tensor:
    return floor + F.softplus(raw)


def _trainable_prior_weights(
    prior_bases: dict[str, float],
    prior_raw: torch.Tensor,
) -> dict[str, torch.Tensor]:
    trained: dict[str, torch.Tensor] = {}
    for index, (key, base_weight) in enumerate(prior_bases.items()):
        base = torch.tensor(base_weight, device=prior_raw.device, dtype=prior_raw.dtype)
        trained[key] = base * torch.exp(prior_raw[index])
    return trained


def _mixture_noise_nll(
    observed: torch.Tensor,
    predicted_clean: torch.Tensor,
    noise_logits: torch.Tensor,
    gaussian_std: torch.Tensor,
    laplace_scale: torch.Tensor,
    speckle_std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | list[float]]]:
    residual = observed - predicted_clean
    gaussian_nll = torch.log(gaussian_std) + 0.5 * (residual / gaussian_std).square()
    laplace_nll = torch.log(2.0 * laplace_scale) + residual.abs() / laplace_scale
    speckle_scale = (predicted_clean.abs().clamp_min(0.05) * speckle_std).clamp_min(1e-4)
    speckle_nll = torch.log(speckle_scale) + 0.5 * (residual / speckle_scale).square()

    components = torch.stack([gaussian_nll, laplace_nll, speckle_nll], dim=0)
    log_weights = F.log_softmax(noise_logits, dim=0).view(-1, 1, 1, 1)
    mixture_nll = -torch.logsumexp(log_weights - components, dim=0).mean()

    weights = torch.softmax(noise_logits.detach(), dim=0).cpu().tolist()
    diagnostics = {
        "noise_weights": [float(x) for x in weights],
        "gaussian_nll": float(gaussian_nll.mean().detach().cpu().item()),
        "laplace_nll": float(laplace_nll.mean().detach().cpu().item()),
        "speckle_nll": float(speckle_nll.mean().detach().cpu().item()),
    }
    return mixture_nll, diagnostics


def solve_inverse_problem(
    lr_observed: torch.Tensor,
    hr_reference: torch.Tensor,
    cfg: InverseSolverConfig,
    mode: str,
    display_name: str | None = None,
    prior_weights: dict[str, float] | None = None,
    seed: int = 0,
) -> SolverResult:
    torch.manual_seed(seed)
    device = torch.device(cfg.device)
    lr_observed = lr_observed.to(device)
    hr_reference = hr_reference.to(device)
    hr_hw = hr_reference.shape[-2:]
    display_name = display_name or mode
    prior_bases = {key: float(value) for key, value in dict(prior_weights or {}).items() if float(value) > 0.0}

    base = bicubic(lr_observed, hr_hw).to(device)
    model = ResidualSIREN(
        base_image=base,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        w0=cfg.siren_w0,
        residual_scale=cfg.residual_scale,
    ).to(device)

    sigma_ratio = (cfg.sigma_init - cfg.sigma_min) / (cfg.sigma_max - cfg.sigma_min)
    sigma_raw = torch.nn.Parameter(torch.tensor(_inverse_sigmoid(sigma_ratio), device=device))
    gaussian_raw = torch.nn.Parameter(torch.tensor(_inverse_softplus(cfg.gaussian_init), device=device))
    laplace_raw = torch.nn.Parameter(torch.tensor(_inverse_softplus(cfg.laplace_init), device=device))
    speckle_raw = torch.nn.Parameter(torch.tensor(_inverse_softplus(cfg.speckle_init), device=device))
    noise_logits = torch.nn.Parameter(torch.zeros(3, device=device))
    prior_raw = torch.nn.Parameter(torch.zeros(len(prior_bases), device=device))

    optimizer_groups = [
        {"params": model.parameters(), "lr": cfg.model_lr},
        {"params": [sigma_raw, gaussian_raw, laplace_raw, speckle_raw, noise_logits], "lr": cfg.param_lr},
    ]
    if prior_raw.numel() > 0:
        optimizer_groups.append({"params": [prior_raw], "lr": cfg.prior_lr})
    optimizer = torch.optim.Adam(optimizer_groups)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.model_lr * 0.1)
        if cfg.use_cosine_schedule else None
    )

    best_objective = float("inf")
    best_selection_score = float("-inf")
    best_state: dict[str, object] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        reconstruction = model.render(*hr_hw)
        sigma = _sigma_from_raw(sigma_raw, cfg)
        gaussian_std = _positive_from_raw(gaussian_raw)
        laplace_scale = _positive_from_raw(laplace_raw)
        speckle_std = _positive_from_raw(speckle_raw)

        predicted_lr = apply_forward_model(
            reconstruction,
            sigma=sigma,
            scale=cfg.scale,
            radius=cfg.blur_radius,
        )

        data_loss, noise_stats = _mixture_noise_nll(
            observed=lr_observed,
            predicted_clean=predicted_lr,
            noise_logits=noise_logits,
            gaussian_std=gaussian_std,
            laplace_scale=laplace_scale,
            speckle_std=speckle_std,
        )
        charbonnier = torch.sqrt(
            (predicted_lr - lr_observed).square() + cfg.charbonnier_eps ** 2
        ).mean()
        data_loss = data_loss + cfg.lambda_data_charbonnier * charbonnier

        tv = total_variation(reconstruction)
        flat_noise = flat_noise_loss(
            reconstruction,
            blur_sigma=cfg.flat_blur_sigma,
            alpha=cfg.flat_alpha,
        )
        edge_sharp = edge_sharpness_loss(
            reconstruction,
            blur_sigma=cfg.sharp_blur_sigma,
            beta=cfg.sharp_beta,
        )
        rof = rof_pde_loss(reconstruction, eps=cfg.rof_eps)
        pm = perona_malik_loss(reconstruction, kappa=cfg.pm_kappa)
        shock = shock_filter_loss(reconstruction, eps=cfg.shock_eps)

        prior_terms = {
            "tv": tv,
            "flat_noise": flat_noise,
            "edge_sharpness": edge_sharp,
            "rof": rof,
            "pm": pm,
            "shock": shock,
        }

        trained_prior_weights = _trainable_prior_weights(prior_bases, prior_raw) if prior_bases else {}
        prior_loss = torch.zeros((), device=device, dtype=reconstruction.dtype)
        warm_epochs = max(1, int(cfg.epochs * cfg.prior_warmup_frac))
        warmup = min(1.0, epoch / warm_epochs)
        for key, weight in trained_prior_weights.items():
            if key not in prior_terms:
                raise ValueError(f"Prior non supportato: {key}")
            prior_loss = prior_loss + warmup * weight * prior_terms[key]

        param_reg = cfg.lambda_noise_reg * (gaussian_std + laplace_scale + speckle_std)
        prior_weight_reg = torch.zeros((), device=device, dtype=reconstruction.dtype)
        if prior_raw.numel() > 0:
            prior_weight_reg = cfg.lambda_prior_weight_reg * prior_raw.square().mean()
        objective = data_loss + prior_loss + param_reg + prior_weight_reg
        objective.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        objective_value = float(objective.detach().cpu().item())
        with torch.no_grad():
            metrics = compute_all(reconstruction.detach(), hr_reference)
            if cfg.selection_metric == "psnr":
                selection_score = float(metrics["psnr"])
            elif cfg.selection_metric == "ssim":
                selection_score = float(metrics["ssim"])
            elif cfg.selection_metric == "objective":
                selection_score = -objective_value
            else:
                raise ValueError(f"selection_metric non supportata: {cfg.selection_metric}")

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_objective = objective_value
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "sigma_raw": sigma_raw.detach().clone(),
                "gaussian_raw": gaussian_raw.detach().clone(),
                "laplace_raw": laplace_raw.detach().clone(),
                "speckle_raw": speckle_raw.detach().clone(),
                "noise_logits": noise_logits.detach().clone(),
                "prior_raw": prior_raw.detach().clone(),
            }

        row = {
            "epoch": float(epoch),
            "objective": objective_value,
            "psnr": float(metrics["psnr"]),
            "ssim": float(metrics["ssim"]),
            "tv": float(tv.detach().cpu().item()),
            "flat_noise": float(flat_noise.detach().cpu().item()),
            "edge_sharpness": float(edge_sharp.detach().cpu().item()),
            "rof": float(rof.detach().cpu().item()),
            "pm": float(pm.detach().cpu().item()),
            "shock": float(shock.detach().cpu().item()),
            "sigma": float(sigma.detach().cpu().item()),
            "gaussian_std": float(gaussian_std.detach().cpu().item()),
            "laplace_scale": float(laplace_scale.detach().cpu().item()),
            "speckle_std": float(speckle_std.detach().cpu().item()),
            "w_gaussian": float(noise_stats["noise_weights"][0]),
            "w_laplace": float(noise_stats["noise_weights"][1]),
            "w_speckle": float(noise_stats["noise_weights"][2]),
        }
        for key, value in trained_prior_weights.items():
            row[f"alpha_{key}"] = float(value.detach().cpu().item())
        history.append(row)

        if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.epochs:
            prior_str = " ".join(
                f"{key}={float(value.detach().cpu().item()):.3g}"
                for key, value in trained_prior_weights.items()
            )
            print(
                f"[{display_name}] ep={epoch:04d} "
                f"obj={row['objective']:.4f} "
                f"PSNR={row['psnr']:.2f}dB "
                f"SSIM={row['ssim']:.4f} "
                f"flat={row['flat_noise']:.5f} "
                f"sharp={row['edge_sharpness']:.5f} "
                f"rof={row['rof']:.5f} "
                f"pm={row['pm']:.5f} "
                f"shock={row['shock']:.5f} "
                f"sigma={row['sigma']:.3f} "
                f"{prior_str}".rstrip()
            )

    if best_state is None:
        raise RuntimeError("Ottimizzazione fallita: nessuno stato valido salvato.")

    with torch.no_grad():
        model.load_state_dict(best_state["model"])
        sigma_raw.copy_(best_state["sigma_raw"])
        gaussian_raw.copy_(best_state["gaussian_raw"])
        laplace_raw.copy_(best_state["laplace_raw"])
        speckle_raw.copy_(best_state["speckle_raw"])
        noise_logits.copy_(best_state["noise_logits"])
        prior_raw.copy_(best_state["prior_raw"])

        reconstruction = model.render(*hr_hw).detach().cpu()
        sigma = float(_sigma_from_raw(sigma_raw, cfg).cpu().item())
        gaussian_std = float(_positive_from_raw(gaussian_raw).cpu().item())
        laplace_scale = float(_positive_from_raw(laplace_raw).cpu().item())
        speckle_std = float(_positive_from_raw(speckle_raw).cpu().item())
        noise_weights = [float(x) for x in torch.softmax(noise_logits, dim=0).cpu().tolist()]
        metrics = compute_all(reconstruction, hr_reference.detach().cpu())
        final_prior_weights = {
            key: float(value.detach().cpu().item())
            for key, value in _trainable_prior_weights(prior_bases, prior_raw).items()
        }

    return SolverResult(
        mode=mode,
        display_name=display_name,
        reconstruction=reconstruction,
        metrics=copy.deepcopy(metrics),
        estimated_sigma=sigma,
        estimated_noise={
            "gaussian_std": gaussian_std,
            "laplace_scale": laplace_scale,
            "speckle_std": speckle_std,
        },
        noise_weights=noise_weights,
        objective=best_objective,
        prior_weights=copy.deepcopy(final_prior_weights),
        history=history,
    )
