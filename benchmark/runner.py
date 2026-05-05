"""Benchmark runner per il confronto PINN vs TV / TV-ROF.

Strategia:
  1) Si calcola UNA VOLTA la baseline bicubica e si pre-addestra UNA VOLTA
     una rete PINN a replicarla (warm-start condiviso).
  2) Ogni config PINN parte da questo checkpoint e corre al massimo
     `max_epochs` epoche, con EARLY STOPPING a PSNR-patience `patience`.
  3) I metodi classici (TV, TV-ROF) lavorano direttamente sulla LR senza
     warm-start PINN e vengono valutati come riferimento.
"""
from __future__ import annotations
import copy
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from config import Config
from models import build_model
from losses import build_losses
from degradation.operator import apply_P
from data.single_image import (
    make_coord_grid,
    lr_pixels_as_data_points,
    sample_collocation,
    sample_data_points,
)
from benchmark.metrics import compute_all, psnr
from benchmark.classical import bicubic, tv_denoise, tv_rof_sr
from utils.device import setup_device
from utils.metrics import save_image


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower()).strip("_")
    return s or "trial"


# ---------------------------------------------------------------- warm-start
def _warmstart_bicubic(cfg: Config, lr: torch.Tensor, hr_hw,
                       device, steps: int = 2000) -> dict:
    """Pre-addestra la rete PINN a replicare il bicubic. Ritorna lo state_dict."""
    H, W = hr_hw
    target = F.interpolate(lr.unsqueeze(0), size=(H, W),
                           mode="bicubic", align_corners=False
                           ).clamp(0, 1).squeeze(0)
    net = build_model(cfg.model).to(device)
    coords = make_coord_grid(H, W, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    print(f"[warm-start] pre-training comune su bicubic per {steps} step...")
    for i in range(steps):
        pred = net(coords).view(H, W, 3).permute(2, 0, 1)
        loss = F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (i + 1) % (steps // 4) == 0:
            print(f"  [warm-start] step {i+1:5d}  mse={loss.item():.2e}")
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


# ---------------------------------------------------------------- PINN train
_HEAVY_TERMS = {"pde_aniso4", "reg_hessian"}   # 4 ordine: molta memoria


def _train_pinn(cfg: Config, hr: torch.Tensor, lr: torch.Tensor,
                terms: List[str], init_state: dict,
                max_epochs: int, patience: int, device) -> Tuple[torch.Tensor, dict]:
    """Addestra una PINN con i termini dati, partendo da init_state. Early stop
    sul PSNR vs HR (patience epoche senza miglioramento)."""
    H, W = hr.shape[-2:]
    net = build_model(cfg.model).to(device)
    net.load_state_dict({k: v.to(device) for k, v in init_state.items()})
    opt = torch.optim.Adam(net.parameters(), lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-6)
    loss_fns = build_losses(list(terms))
    data_coords, data_rgb = lr_pixels_as_data_points(lr)

    coord_scale = float(max(H, W))
    coords_full = make_coord_grid(H, W, device=device)

    # I termini del 4 ordine moltiplicano molto l'uso di memoria: riduciamo
    # i punti di collocation quando sono attivi.
    n_coll = cfg.train.n_collocation
    if any(t in _HEAVY_TERMS for t in terms):
        n_coll = max(512, n_coll // 4)
        print(f"  [mem] termini del 4 ordine attivi -> n_collocation={n_coll}")

    def _render():
        net.eval()
        with torch.no_grad():
            p = net(coords_full).view(H, W, 3).permute(2, 0, 1).clamp(0, 1)
        net.train()
        return p

    best_psnr = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    stale = 0
    w = cfg.loss.weights

    history = {"total": [], "psnr": []}
    for name in terms:
        history[name] = []

    def _pde_scale(ep):
        k0 = cfg.loss.pde_warmup_epochs
        kr = max(1, cfg.loss.pde_ramp_epochs)
        if ep < k0: return 0.0
        return min(1.0, (ep - k0) / kr)

    t0 = time.time()
    for ep in range(max_epochs):
        dc, dr = sample_data_points(data_coords, data_rgb, cfg.train.n_data_points)
        coll = sample_collocation(n_coll, device=device)
        kwargs = dict(
            y_lr=lr, hr_hw=(H, W),
            sigma=cfg.data.blur_sigma, scale=cfg.data.scale,
            data_coords=dc, data_rgb=dr, collocation=coll,
            pm_kappa=cfg.loss.pm_kappa, eig_clip=cfg.loss.eig_clip,
            struct_eps=cfg.loss.struct_eps, coord_scale=coord_scale,
            hr_target=None, n_bc=512, device=device,
        )
        pde_s = _pde_scale(ep)
        total = torch.zeros((), device=device)

        current_losses = {}

        for name, fn in loss_fns.items():
            val = fn(net, **kwargs)
            wt = w.get(name, 1.0)
            if name.startswith("pde_"):
                wt = wt * pde_s
            weighted_val = wt * val
            total = total + wt * val
            current_losses[name] = weighted_val.item()

        opt.zero_grad(set_to_none=True)
        total.backward()
        if cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.train.grad_clip)
        opt.step()

        # eval per early stopping (ogni epoca, e' veloce su immagine singola)
        p = psnr(_render(), hr)

        history["total"].append(total.item())
        history["psnr"].append(p)
        for name in terms:
            history[name].append(current_losses[name])

        if p > best_psnr + 1e-4:
            best_psnr = p
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if ep % 25 == 0 or ep == max_epochs - 1:
            print(f"  [ep {ep:4d}] loss={total.item():.3e}  "
                  f"PSNR={p:.2f}dB  best={best_psnr:.2f}dB  stale={stale}")

        if stale >= patience:
            print(f"  [early-stop] ep {ep}, no improvement for {patience} epochs  "
                  f"(best PSNR={best_psnr:.2f}dB)")
            break

        scheduler.step()

    net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  [pinn] elapsed {time.time() - t0:.1f}s  best PSNR={best_psnr:.2f}dB")
    return _render().detach(), history


# -------------------------------------------------------------------- entry
def run_benchmark(hr: torch.Tensor, cfg: Config, configs: List[dict],
                  max_epochs: int = 500, patience: int = 150,
                  warmstart_steps: int = 2000, verbose: bool = True,
                  cache_dir: Optional[str] = None,
                  force: bool = False,
                  on_trial_done=None,
                  ) -> Tuple[Dict[str, Tuple[torch.Tensor, dict]], torch.Tensor]:
    device = setup_device(cfg.train.device)
    hr = hr.to(device)
    H, W = hr.shape[-2:]
    with torch.no_grad():
        lr = apply_P(hr, cfg.data.blur_sigma, cfg.data.scale)
        if cfg.data.noise_std > 0:
            lr = lr + cfg.data.noise_std * torch.randn_like(lr)
        lr = lr.clamp(0, 1)

    # cartelle di cache (riprendi-da-dove-interrotto + snapshot visivi)
    cache_path = Path(cache_dir) if cache_dir else None
    trials_path = None
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)
        trials_path = cache_path.parent / "trials"
        trials_path.mkdir(parents=True, exist_ok=True)
        # salva anche HR e LR come riferimento visivo una volta
        save_image(hr.cpu(), str(trials_path / "_HR_ground_truth.png"))
        lr_up = F.interpolate(lr.unsqueeze(0), size=(H, W),
                              mode="nearest").squeeze(0).cpu()
        save_image(lr_up, str(trials_path / "_LR_nearest_up.png"))

    # warm-start condiviso (cachato su disco se possibile)
    warm_ckpt = cache_path / "warmstart.pt" if cache_path else None
    if warm_ckpt and warm_ckpt.exists() and not force:
        print(f"[warm-start] carico checkpoint esistente: {warm_ckpt}")
        init_state = torch.load(warm_ckpt, map_location="cpu")
    else:
        init_state = _warmstart_bicubic(cfg, lr, (H, W), device,
                                        steps=warmstart_steps)
        if warm_ckpt:
            torch.save(init_state, warm_ckpt)
            print(f"[warm-start] salvato in {warm_ckpt}")

    bic = bicubic(lr, cfg.data.scale).to(device)
    print(f"[warm-start] bicubic PSNR={psnr(bic, hr):.2f}dB")

    results: Dict[str, Tuple[torch.Tensor, dict]] = {}
    for spec in configs:
        name = spec["name"]
        method = spec["method"]
        extras = spec.get("extras", {})
        slug = _slug(name)

        # -------- resume: se il trial e' gia' in cache, saltalo -----------
        trial_ckpt = cache_path / f"{slug}.pt" if cache_path else None
        if trial_ckpt and trial_ckpt.exists() and not force:
            data = torch.load(trial_ckpt, map_location="cpu")
            history_data = data.get("history", None)
            results[name] = (data["pred"], data["metrics"], history_data)
            print(f"[skip] {name}: risultato in cache "
                  f"(PSNR={data['metrics']['psnr']:.2f}dB  "
                  f"SSIM={data['metrics']['ssim']:.4f})")
            if on_trial_done is not None:
                on_trial_done(results, lr.cpu())
            continue

        print(f"\n===== [benchmark] {name}  (method={method}) =====")

        if method == "classical":
            kind = spec["kind"]
            sc = cfg.data.scale
            if kind == "bicubic":
                pred = bicubic(lr, sc)
            elif kind == "tv_denoise":
                pred = tv_denoise(lr, sc,
                                  lam=extras.get("lam", 0.05),
                                  n_iter=extras.get("n_iter", 120))
            elif kind == "tv_rof":
                pred = tv_rof_sr(lr, sc, sigma=cfg.data.blur_sigma,
                                 lam=extras.get("lam", 0.01),
                                 n_iter=extras.get("n_iter", 200),
                                 inner_tv=extras.get("inner_tv", 10))
            else:
                raise ValueError(f"Unknown classical kind: {kind}")
        elif method == "pinn":
            pred, history = _train_pinn(cfg, hr, lr, spec["terms"], init_state,
                               max_epochs=max_epochs, patience=patience,
                               device=device)
        else:
            raise ValueError(f"Unknown method: {method}")

        pred = pred.to(device).clamp(0, 1)
        metrics = compute_all(pred, hr)
        pred_cpu = pred.cpu()
        if method == "pinn":
            results[name] = (pred_cpu, metrics, history)
        else:
            results[name] = (pred_cpu, metrics, None)

        print(f"  -> PSNR={metrics['psnr']:.2f}dB  SSIM={metrics['ssim']:.4f}")

        # ---- persisti subito: riprendi-da-dove-interrotto + snapshot PNG
        if trial_ckpt is not None:
            save_data = {"pred": pred_cpu, "metrics": metrics,
                        "spec": spec}
            if method == "pinn":
                save_data["history"] = history
            torch.save(save_data, trial_ckpt)
            save_image(pred_cpu,
                       str(trials_path / f"{slug}__"
                           f"psnr{metrics['psnr']:.2f}_"
                           f"ssim{metrics['ssim']:.3f}.png"))
        if on_trial_done is not None:
            on_trial_done(results, lr.cpu())
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results, lr.cpu()
