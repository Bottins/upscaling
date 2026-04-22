"""Entry point. Uso:
    python main.py                    # single-image, SIREN + PM + data_lr + BC
    python main.py --loss data_points pde_anisotropic bc_neumann
"""
from __future__ import annotations
import argparse
from pathlib import Path

from config import Config
from data.download import ensure_dataset
from data.dataset import list_images, load_image
from training.trainer import PINNTrainer


def build_config(args) -> Config:
    cfg = Config()
    if args.loss:
        cfg.loss.terms = args.loss
    if args.model:
        cfg.model.kind = args.model
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.scale:
        cfg.data.scale = args.scale
    if args.image:
        cfg.data.image_name = args.image
    return cfg


def pick_image(cfg: Config):
    # 1) se image_name e' un path locale esistente, usalo direttamente
    p = Path(cfg.data.image_name)
    if p.is_file():
        return p
    # 2) altrimenti scarica/usa il dataset e cerca per nome
    name_map = {"Set5": "Set5", "DIV2K": "DIV2K_valid_HR"}
    ds_key = name_map.get(cfg.data.name, cfg.data.name)
    root = ensure_dataset(ds_key, cfg.data.root)
    imgs = list_images(root)
    if not imgs:
        raise RuntimeError(f"Nessuna immagine trovata in {root}")
    for q in imgs:
        if q.name.lower() == cfg.data.image_name.lower():
            return q
    return imgs[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", nargs="+", default=None,
                    help="Termini di loss attivi (es: data_points pde_perona_malik bc_neumann)")
    ap.add_argument("--model", choices=["siren", "fourier_mlp"], default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--scale", type=int, default=None)
    ap.add_argument("--image", type=str, default=None)
    args = ap.parse_args()

    cfg = build_config(args)
    print("[config] loss terms:", cfg.loss.terms)

    img_path = pick_image(cfg)
    print(f"[data] using image: {img_path}")
    hr = load_image(img_path, size=cfg.data.hr_size)

    trainer = PINNTrainer(cfg, hr)
    trainer.fit()


if __name__ == "__main__":
    main()
