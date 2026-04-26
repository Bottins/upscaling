"""Training loop per PINN Super-Resolution (modalita' single-image e batch)."""
from __future__ import annotations
import os
from pathlib import Path

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
from utils.metrics import psnr, save_image, save_triptych
from utils.device import setup_device
from learned_prior.load import load_prior, predict_hr


class PINNTrainer:
    def __init__(self, cfg: Config, hr: torch.Tensor):
        """
        hr: (3, H, W) ground truth HR usato per generare la LR e per valutare.
        """
        self.cfg = cfg
        self.device = setup_device(cfg.train.device)
        self.hr = hr.to(self.device)
        self.H, self.W = hr.shape[-2:]

        # Genera y = P u + eta
        with torch.no_grad():
            self.lr = apply_P(self.hr, cfg.data.blur_sigma, cfg.data.scale)
            if cfg.data.noise_std > 0:
                self.lr = self.lr + cfg.data.noise_std * torch.randn_like(self.lr)
            self.lr = self.lr.clamp(0, 1)

        # Precalcola i "punti dati" riposizionati dai pixel LR
        self.data_coords, self.data_rgb = lr_pixels_as_data_points(self.lr)

        # Modello + ottimizzatore
        self.net = build_model(cfg.model).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.train.lr)

        # Loss selezionate dal config (pulito e modulare)
        self.loss_fns = build_losses(cfg.loss.terms)

        # Pre-calcola la predizione del prior appreso se richiesto
        self.hr_prior_target = None
        if "prior_sr" in cfg.loss.terms:
            prior_net, _ = load_prior(cfg.loss.prior_ckpt, self.device)
            self.hr_prior_target = predict_hr(prior_net, self.lr).detach()
            p_prior = psnr(self.hr_prior_target, self.hr)
            print(f"[prior] CNN prior PSNR vs HR: {p_prior:.2f}dB")

    # ---------------------------------------------------------------- utils
    def _bicubic_init(self, steps: int = 2000) -> None:
        """Pre-addestra la rete a replicare l'upsampling bicubico della LR."""
        target = F.interpolate(self.lr.unsqueeze(0),
                               size=(self.H, self.W),
                               mode="bicubic", align_corners=False
                               ).clamp(0, 1).squeeze(0)
        bicubic_psnr = psnr(target, self.hr)
        print(f"[init] bicubic baseline vs HR: PSNR={bicubic_psnr:.2f}dB")

        coords = make_coord_grid(self.H, self.W, device=self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        for i in range(steps):
            pred = self.net(coords).view(self.H, self.W, 3).permute(2, 0, 1)
            loss = F.mse_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if (i + 1) % (steps // 4) == 0:
                p = psnr(pred.detach().clamp(0, 1), self.hr)
                print(f"[init] step {i+1:5d}  fit_mse={loss.item():.2e}  "
                      f"PSNR_vs_HR={p:.2f}dB")

    def _pde_active(self, epoch: int) -> float:
        """Curriculum: abilita gradualmente il termine PDE (0 -> 1 su un ramp)."""
        k0 = self.cfg.loss.pde_warmup_epochs
        kr = max(1, self.cfg.loss.pde_ramp_epochs)
        if epoch < k0:
            return 0.0
        return min(1.0, (epoch - k0) / kr)

    # ---------------------------------------------------------------- step
    def _compute_loss(self, epoch: int):
        cfg = self.cfg
        w = cfg.loss.weights
        pde_scale = self._pde_active(epoch)

        # punti dati riposizionati per il termine 'data_points'
        dc, dr = sample_data_points(self.data_coords, self.data_rgb,
                                    cfg.train.n_data_points)
        # collocation per i termini PDE
        coll = sample_collocation(cfg.train.n_collocation, device=self.device)

        kwargs = dict(
            y_lr=self.lr, hr_hw=(self.H, self.W),
            sigma=cfg.data.blur_sigma, scale=cfg.data.scale,
            data_coords=dc, data_rgb=dr,
            collocation=coll,
            pm_kappa=cfg.loss.pm_kappa,
            eig_clip=cfg.loss.eig_clip,
            struct_eps=cfg.loss.struct_eps,
            coord_scale=float(max(self.H, self.W)),
            hr_target=self.hr_prior_target,
            n_bc=512, device=self.device,
        )

        total = torch.zeros((), device=self.device)
        logs = {}
        for name, fn in self.loss_fns.items():
            val = fn(self.net, **kwargs)
            weight = w.get(name, 1.0)
            if name.startswith("pde_"):
                weight = weight * pde_scale
            total = total + weight * val
            logs[name] = val.item()
        return total, logs

    # ---------------------------------------------------------------- fit
    def fit(self) -> None:
        cfg = self.cfg
        if cfg.train.init_from_bicubic:
            print("[init] pre-training su bicubic...")
            self._bicubic_init()

        Path(cfg.train.ckpt_dir).mkdir(parents=True, exist_ok=True)
        snap_dir = Path(cfg.train.snap_dir)
        snap_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(cfg.train.epochs):
            self.net.train()
            loss, logs = self._compute_loss(epoch)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.train.grad_clip and cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                               cfg.train.grad_clip)
            self.opt.step()

            if epoch % cfg.train.log_every == 0 or epoch == cfg.train.epochs - 1:
                metric = self.evaluate()
                msg = " | ".join(f"{k}={v:.3e}" for k, v in logs.items())
                print(f"[ep {epoch:5d}] total={loss.item():.3e} | {msg} "
                      f"| PSNR={metric['psnr']:.2f}dB")

            if (epoch % cfg.train.snapshot_every == 0
                    or epoch == cfg.train.epochs - 1):
                self._save_snapshot(snap_dir, epoch)

        self.save_outputs()

    def _save_snapshot(self, out_dir: Path, epoch: int) -> None:
        pred = self.render_hr()
        p = psnr(pred, self.hr)
        title = f"epoch {epoch:05d}  |  HR  |  LR (nearest)  |  pred  (PSNR={p:.2f}dB)"
        save_triptych(self.hr, self.lr, pred,
                      str(out_dir / f"compare_ep{epoch:05d}.png"),
                      title=title)

    # ---------------------------------------------------------------- eval
    @torch.no_grad()
    def render_hr(self) -> torch.Tensor:
        coords = make_coord_grid(self.H, self.W, device=self.device)
        pred = self.net(coords).view(self.H, self.W, 3).permute(2, 0, 1)
        return pred.clamp(0, 1)

    @torch.no_grad()
    def evaluate(self) -> dict:
        pred = self.render_hr()
        return {"psnr": psnr(pred, self.hr)}

    def save_outputs(self) -> None:
        out_dir = Path(self.cfg.train.ckpt_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_image(self.render_hr(), str(out_dir / "pred_hr.png"))
        save_image(self.lr, str(out_dir / "input_lr.png"))
        save_image(self.hr, str(out_dir / "gt_hr.png"))
        torch.save(self.net.state_dict(), out_dir / "model.pt")
        print(f"[save] output in {out_dir}")
