"""Benchmark PINN vs metodi classici (Bicubic / TV / TV-ROF).

- Il warm-start bicubico e' calcolato una volta sola (condiviso).
- Ogni trial PINN parte da quel checkpoint e corre al massimo 500 epoche
  con early-stopping a patience 150 sul PSNR.

Uso:
    python main_benchmark.py
    python main_benchmark.py --image butterfly.png --max-epochs 500 --patience 150
    python main_benchmark.py --only TV aniso4        # sottoinsieme di config

Output in ./benchmark_results/:
    - comparison.png   (griglia visiva con PSNR/SSIM)
    - results.md       (tabella)
"""
from __future__ import annotations
import argparse
from pathlib import Path

from config import Config
from data.download import ensure_dataset
from data.dataset import list_images, load_image
from benchmark.runner import run_benchmark
from benchmark.visualize import save_grid, markdown_table


def default_configs():
    return [
        # -------- baseline classiche ------------------------------------
        {"name": "Bicubic", "method": "classical", "kind": "bicubic"},
        {"name": "TV (Chambolle)", "method": "classical",
         "kind": "tv_denoise", "extras": {"lam": 0.05, "n_iter": 120}},
        {"name": "TV-ROF (inverse)", "method": "classical",
         "kind": "tv_rof", "extras": {"lam": 0.01, "n_iter": 200,
                                      "inner_tv": 10}},
        # -------- PINN: termine dati da solo ----------------------------
        {"name": "PINN: data_lr", "method": "pinn",
         "terms": ["data_lr"]},
        # -------- PINN + PDE del 2 ordine -------------------------------
        {"name": "PINN + Perona-Malik", "method": "pinn",
         "terms": ["data_lr", "pde_perona_malik", "bc_neumann"]},
        {"name": "PINN + anisotropic tensor", "method": "pinn",
         "terms": ["data_lr", "pde_anisotropic", "bc_neumann"]},
        # -------- PINN + regolarizzatori di nitidezza -------------------
        {"name": "PINN + TV",    "method": "pinn",
         "terms": ["data_lr", "reg_tv"]},
        {"name": "PINN + shock", "method": "pinn",
         "terms": ["data_lr", "pde_shock"]},
        # -------- PINN + 4° ordine (anti-staircase) ---------------------
        {"name": "PINN + Hessian (LLT)", "method": "pinn",
         "terms": ["data_lr", "reg_hessian"]},
        {"name": "PINN + aniso4 (You-Kaveh)", "method": "pinn",
         "terms": ["data_lr", "pde_aniso4"]},
        # -------- combinazioni ------------------------------------------
        {"name": "PINN + aniso + LLT",    "method": "pinn",
         "terms": ["data_lr", "pde_anisotropic", "reg_hessian"]},
        {"name": "PINN + PM + shock",     "method": "pinn",
         "terms": ["data_lr", "pde_perona_malik", "pde_shock"]},
        {"name": "PINN + aniso + shock + LLT", "method": "pinn",
         "terms": ["data_lr", "pde_anisotropic", "pde_shock", "reg_hessian"]},
    ]


def pick_image(cfg: Config):
    p = Path(cfg.data.image_name)
    if p.is_file():
        return p
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--scale", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=500)
    ap.add_argument("--patience",   type=int, default=150)
    ap.add_argument("--warmstart-steps", type=int, default=2000)
    ap.add_argument("--n-collocation", type=int, default=None,
                    help="Numero punti PDE. Default da config (4096). Abbassa se OOM.")
    ap.add_argument("--device", type=str, default=None,
                    help="'cuda' | 'cpu' (default: da config, cuda se disponibile)")
    ap.add_argument("--out", type=str, default="./benchmark_results")
    ap.add_argument("--force", action="store_true",
                    help="Ignora la cache e rifa' tutti i trial da zero")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Esegui solo i config il cui nome contiene una di queste sottostringhe")
    args = ap.parse_args()

    cfg = Config()
    if args.image: cfg.data.image_name = args.image
    if args.scale: cfg.data.scale = args.scale
    if args.device: cfg.train.device = args.device
    if args.n_collocation: cfg.train.n_collocation = args.n_collocation

    img_path = pick_image(cfg)
    print(f"[benchmark] image: {img_path}")
    hr = load_image(img_path, size=cfg.data.hr_size)

    configs = default_configs()
    if args.only:
        configs = [c for c in configs
                   if any(s.lower() in c["name"].lower() for s in args.only)]
        print(f"[benchmark] filtered to {len(configs)} configs")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "cache"
    title = (f"SR benchmark  -  {img_path.name}  x{cfg.data.scale}  "
             f"(PINN {args.max_epochs}ep / patience {args.patience})")

    def _on_trial_done(partial_results, lr_t):
        # aggiorna griglia + tabella dopo ogni trial (utile se interrompi)
        save_grid(partial_results, hr, lr_t, str(out / "comparison.png"),
                  n_cols=4, title=title)
        (out / "results.md").write_text(markdown_table(partial_results),
                                        encoding="utf-8")

    results, lr = run_benchmark(hr, cfg, configs,
                                max_epochs=args.max_epochs,
                                patience=args.patience,
                                warmstart_steps=args.warmstart_steps,
                                cache_dir=str(cache_dir),
                                force=args.force,
                                on_trial_done=_on_trial_done)

    save_grid(results, hr, lr, str(out / "comparison.png"),
              n_cols=4, title=title)
    md = markdown_table(results)
    (out / "results.md").write_text(md, encoding="utf-8")

    print("\n===== RESULTS =====")
    print(md)
    print(f"\n[save] {out / 'comparison.png'}")
    print(f"[save] {out / 'results.md'}")


if __name__ == "__main__":
    main()
