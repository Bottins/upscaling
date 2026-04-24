"""CLI: addestra il prior CNN su DIV2K.

Uso:
    python -m learned_prior.train --scale 4 --epochs 30
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.download import ensure_dataset
from learned_prior.patches import list_hr_images, PatchDataset
from learned_prior.model import SmallEDSR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="DIV2K_train_HR")
    ap.add_argument("--root", default="./datasets")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--patch", type=int, default=96)

    # turbo defaults
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n_blocks", type=int, default=8)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--spi", type=int, default=16,
                    help="patch sampled per image per epoch")
    ap.add_argument("--out", default="./checkpoints/prior_sr.pt")
    ap.add_argument("--workers", type=int, default=8)

    # nuove opzioni turbo
    ap.add_argument("--cache-images", action="store_true",
                    help="Precarica tutte le immagini HR in RAM")
    ap.add_argument("--no-cache-images", action="store_true",
                    help="Disattiva il precaricamento RAM")
    ap.add_argument("--prefetch", type=int, default=4,
                    help="prefetch_factor del DataLoader")
    ap.add_argument("--amp", action="store_true",
                    help="Abilita mixed precision")
    ap.add_argument("--compile", action="store_true",
                    help="Prova torch.compile sul modello")
    args = ap.parse_args()

    use_cache_images = True
    if args.no_cache_images:
        use_cache_images = False
    elif args.cache_images:
        use_cache_images = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[prior] device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    root = ensure_dataset(args.dataset, args.root)
    paths = list_hr_images(root)
    if not paths:
        raise RuntimeError(f"Nessuna immagine in {root}")
    print(f"[prior] {len(paths)} immagini HR in {root}")

    ds = PatchDataset(
        paths,
        patch=args.patch,
        scale=args.scale,
        blur_sigma=args.sigma,
        samples_per_image=args.spi,
        cache_images=use_cache_images,
    )

    dl_kwargs = dict(
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    if args.workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = args.prefetch

    dl = DataLoader(ds, **dl_kwargs)

    net = SmallEDSR(scale=args.scale, n_blocks=args.n_blocks, ch=args.ch).to(device)
    if device.type == "cuda":
        net = net.to(memory_format=torch.channels_last)

    if args.compile:
        try:
            net = torch.compile(net)
            print("[prior] torch.compile attivo")
        except Exception as e:
            print(f"[prior] torch.compile non disponibile: {e}")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    amp_enabled = args.amp and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(args.epochs):
        t0 = time.time()
        net.train()
        running, n = 0.0, 0

        for step, (lr_img, hr_img) in enumerate(dl, start=1):
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            if device.type == "cuda":
                lr_img = lr_img.contiguous(memory_format=torch.channels_last)
                hr_img = hr_img.contiguous(memory_format=torch.channels_last)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred = net(lr_img)
                loss = F.l1_loss(pred, hr_img)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item() * lr_img.size(0)
            n += lr_img.size(0)

        sched.step()
        avg = running / max(1, n)
        psnr = 10 * torch.log10(1.0 / torch.tensor(avg ** 2 + 1e-12)).item()
        dt = time.time() - t0

        print(
            f"[ep {ep+1:3d}/{args.epochs}] "
            f"L1={avg:.4f}  ~PSNR~{psnr:.2f}dB  "
            f"lr={sched.get_last_lr()[0]:.2e}  "
            f"{dt:.1f}s"
        )

    torch.save(
        {
            "state_dict": net.state_dict(),
            "scale": args.scale,
            "n_blocks": args.n_blocks,
            "ch": args.ch,
            "sigma": args.sigma,
        },
        args.out,
    )
    print(f"[prior] salvato in {args.out}")


if __name__ == "__main__":
    main()