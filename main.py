from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import re

import torch
import torch.nn.functional as F

from inverse_sr.baselines import bicubic
from inverse_sr.degradation import NoiseTruth, SyntheticDegradation, make_observation
from inverse_sr.io import ensure_dir, load_image, pick_image, save_comparison_strip, save_image
from inverse_sr.metrics import compute_all
from inverse_sr.solver import InverseSolverConfig, solve_inverse_problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark per ricostruzione SR con operatore blur+downsample ignoto, "
            "miscela di noise trainabile e prior PDE (Perona-Malik + shock)."
        )
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Path immagine HR oppure nome file da cercare in datasets/.")
    parser.add_argument("--dataset-root", type=str, default="datasets",
                        help="Cartella in cui cercare immagini se --image non e' un path valido.")
    parser.add_argument("--size", type=int, default=128,
                        help="Resize quadrato dell'immagine HR prima del benchmark.")
    parser.add_argument("--scale", type=int, default=2,
                        help="Fattore di downsampling.")
    parser.add_argument("--epochs", type=int, default=250,
                        help="Epoche di ottimizzazione per metodo.")
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto', 'cuda' o 'cpu'.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed globale per degradazione sintetica e inizializzazione.")
    parser.add_argument("--sigma-true", type=float, default=1.35,
                        help="Sigma reale del blur gaussiano usato per generare LR.")
    parser.add_argument("--gaussian-std", type=float, default=0.03,
                        help="Std reale del rumore gaussiano nella miscela.")
    parser.add_argument("--laplace-scale", type=float, default=0.02,
                        help="Scala reale del rumore Laplace nella miscela.")
    parser.add_argument("--speckle-std", type=float, default=0.05,
                        help="Std reale del rumore speckle nella miscela.")
    parser.add_argument("--noise-weights", type=float, nargs=3,
                        default=(0.45, 0.30, 0.25),
                        metavar=("W_GAUSS", "W_LAPLACE", "W_SPECKLE"),
                        help="Pesi reali della miscela di noise per generare LR.")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Root cartella di output.")
    parser.add_argument("--out", type=str, default=None,
                        help="Cartella run specifica, stile old/main_benchmark.py.")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Esegui solo i metodi il cui nome contiene una di queste stringhe.")
    return parser.parse_args()


def resolve_device(user_choice: str) -> str:
    if user_choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return user_choice


def default_configs() -> list[dict]:
    blind_common = {"flat_noise": 300.0, "edge_sharpness": 0.30}
    return [
        {"id": "bicubic", "name": "Bicubic", "kind": "bicubic"},
        {"id": "all_trainable", "name": "All PDE trainable", "kind": "inverse",
         "prior_weights": {**blind_common, "tv": 0.03, "rof": 0.012, "pm": 3.0, "shock": 1.0}},
        {"id": "tv_energy", "name": "TV energy", "kind": "inverse",
         "prior_weights": {**blind_common, "tv": 0.03}},
        {"id": "rof_pde", "name": "ROF PDE", "kind": "inverse",
         "prior_weights": {**blind_common, "rof": 0.015}},
        {"id": "pm_pde", "name": "Perona-Malik PDE", "kind": "inverse",
         "prior_weights": {**blind_common, "pm": 4.0}},
        {"id": "shock_pde", "name": "Shock PDE", "kind": "inverse",
         "prior_weights": {**blind_common, "shock": 1.5}},
        {"id": "rof_shock", "name": "ROF PDE + Shock", "kind": "inverse",
         "prior_weights": {**blind_common, "rof": 0.012, "shock": 1.0}},
        {"id": "pm_shock", "name": "Perona-Malik + Shock", "kind": "inverse",
         "prior_weights": {**blind_common, "pm": 3.0, "shock": 1.0}},
    ]


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def observed_title(scale: int) -> str:
    return "Observed degraded" if scale == 1 else "Observed LR"


def markdown_table(results: list[dict]) -> str:
    lines = [
        "| Metodo | PSNR | SSIM | Sigma stimato | Pesi noise |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for item in results:
        lines.append(
            f"| {item['name']} | {item['psnr']:.2f} | {item['ssim']:.4f} | "
            f"{item['sigma']} | {item['noise_weights']} |"
        )
    return "\n".join(lines)


def learned_prior_lines(methods_payload: dict) -> list[str]:
    lines = ["## Pesi prior appresi", ""]
    found = False
    for method_id, payload in methods_payload.items():
        prior_weights = payload.get("prior_weights")
        display_name = payload.get("display_name", method_id)
        if not prior_weights:
            continue
        found = True
        pretty = ", ".join(f"{key}={value:.4g}" for key, value in prior_weights.items())
        lines.append(f"- {display_name}: {pretty}")
    if not found:
        lines.append("- Nessun prior trainabile in questo run.")
    return lines


def save_outputs(
    out_dir: Path,
    hr: torch.Tensor,
    lr_clean: torch.Tensor,
    lr_observed: torch.Tensor,
    images: dict[str, torch.Tensor],
    comparison_entries: list[dict],
    results_table: str,
    results_payload: dict,
) -> None:
    ensure_dir(out_dir)
    save_image(hr, out_dir / "hr.png")
    save_image(lr_clean, out_dir / "lr_clean.png")
    save_image(lr_observed, out_dir / "lr_observed.png")

    for name, tensor in images.items():
        save_image(tensor, out_dir / f"{name}.png")

    save_comparison_strip(comparison_entries, out_dir / "comparison.png")
    (out_dir / "results.md").write_text(results_table, encoding="utf-8")
    with (out_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results_payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    image_path = pick_image(args.image, Path(args.dataset_root))
    hr = load_image(image_path, size=(args.size, args.size))

    degradation = SyntheticDegradation(
        scale=args.scale,
        sigma=args.sigma_true,
        noise=NoiseTruth(
            gaussian_std=args.gaussian_std,
            laplace_scale=args.laplace_scale,
            speckle_std=args.speckle_std,
            weights=tuple(args.noise_weights),
        ),
        seed=args.seed,
    )

    lr_clean, lr_observed = make_observation(hr, degradation)
    degraded_display = F.interpolate(
        lr_observed.unsqueeze(0),
        size=hr.shape[-2:],
        mode="nearest",
    ).squeeze(0).clamp(0, 1)
    degraded_metrics = compute_all(degraded_display, hr)
    bicubic_metrics = compute_all(bicubic(lr_observed, hr.shape[-2:]), hr)

    solver_cfg = InverseSolverConfig(
        scale=args.scale,
        epochs=args.epochs,
        device=device,
    )

    configs = default_configs()
    if args.only:
        selected = [
            config for config in configs
            if any(
                token.lower() in config["name"].lower() or token.lower() in config["id"].lower()
                for token in args.only
            )
        ]
        bicubic_cfg = next((config for config in configs if config["id"] == "bicubic"), None)
        configs = []
        if bicubic_cfg is not None:
            configs.append(bicubic_cfg)
        for config in selected:
            if config["id"] != "bicubic":
                configs.append(config)
        print(f"[benchmark] filtered to {len(configs)} methods (including Bicubic baseline)")
    if not configs:
        raise RuntimeError("Nessun metodo selezionato.")

    print(f"[data] image={image_path}")
    print(f"[device] {device}")
    print(
        "[truth] "
        f"sigma={degradation.sigma:.3f} "
        f"noise_weights={tuple(round(x, 3) for x in degradation.noise.weights)} "
        f"gaussian_std={degradation.noise.gaussian_std:.4f} "
        f"laplace_scale={degradation.noise.laplace_scale:.4f} "
        f"speckle_std={degradation.noise.speckle_std:.4f}"
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = Path(args.results_dir) / f"{image_path.stem}_x{args.scale}_{stamp}"
    run_dir = Path(args.out) if args.out else default_out

    table_rows: list[dict] = []
    saved_images: dict[str, torch.Tensor] = {}
    comparison_entries = [
        {
            "title": "Ground truth",
            "image": hr,
            "lines": [
                f"size {hr.shape[-1]}x{hr.shape[-2]}",
                f"scale x{args.scale}",
                f"sigma true {degradation.sigma:.2f}",
            ],
        },
        {
            "title": observed_title(args.scale),
            "image": degraded_display,
            "lines": [
                "degraded input",
                f"PSNR {degraded_metrics['psnr']:.2f} dB",
                f"SSIM {degraded_metrics['ssim']:.4f}",
            ],
        },
    ]
    payload = {
        "image": str(image_path),
        "device": device,
        "degradation_truth": asdict(degradation),
        "methods": {},
    }

    for config in configs:
        if config["kind"] == "bicubic":
            reconstruction = bicubic(lr_observed, hr.shape[-2:])
            metrics = bicubic_metrics
            payload["methods"][config["id"]] = {
                "display_name": config["name"],
                "metrics": metrics,
            }
            saved_images["bicubic"] = reconstruction
            table_rows.append({
                "name": config["name"],
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "sigma": "-",
                "noise_weights": "-",
            })
            comparison_entries.append({
                "title": "Bicubic",
                "image": reconstruction,
                "lines": [
                    f"PSNR {metrics['psnr']:.2f} dB",
                    f"SSIM {metrics['ssim']:.4f}",
                ],
                "history": [],
            })
            print(f"[bicubic]  PSNR={metrics['psnr']:.2f}dB  SSIM={metrics['ssim']:.4f}")
        else:
            result = solve_inverse_problem(
                lr_observed=lr_observed,
                hr_reference=hr,
                cfg=solver_cfg,
                mode=config["id"],
                display_name=config["name"],
                prior_weights=config.get("prior_weights", {}),
                seed=args.seed,
            )
            payload["methods"][config["id"]] = result.to_serializable()
            saved_images[slugify(config["name"])] = result.reconstruction.cpu()
            table_rows.append({
                "name": config["name"],
                "psnr": result.metrics["psnr"],
                "ssim": result.metrics["ssim"],
                "sigma": f"{result.estimated_sigma:.4f}",
                "noise_weights": tuple(round(x, 4) for x in result.noise_weights),
            })
            comparison_entries.append({
                "title": config["name"],
                "image": result.reconstruction.cpu(),
                "lines": [
                    f"PSNR {result.metrics['psnr']:.2f} dB",
                    f"SSIM {result.metrics['ssim']:.4f}",
                    f"sigma {result.estimated_sigma:.3f}",
                ],
                "history": result.history,
            })
            print(
                f"[{config['name']}] "
                f"PSNR={result.metrics['psnr']:.2f}dB  "
                f"SSIM={result.metrics['ssim']:.4f}"
            )

        report_lines = [
            "# Benchmark SR con operatore ignoto",
            "",
            f"- Immagine: `{image_path}`",
            f"- Device: `{device}`",
            f"- Scala: `{args.scale}`",
            f"- Sigma reale blur: `{degradation.sigma:.4f}`",
            (
                "- Noise reale: "
                f"weights={tuple(round(x, 4) for x in degradation.noise.weights)}, "
                f"gaussian_std={degradation.noise.gaussian_std:.4f}, "
                f"laplace_scale={degradation.noise.laplace_scale:.4f}, "
                f"speckle_std={degradation.noise.speckle_std:.4f}"
            ),
            "",
            markdown_table(table_rows),
        ]
        report_lines.extend([""] + learned_prior_lines(payload["methods"]))
        save_outputs(
            out_dir=run_dir,
            hr=hr,
            lr_clean=lr_clean,
            lr_observed=lr_observed,
            images=saved_images,
            comparison_entries=comparison_entries,
            results_table="\n".join(report_lines),
            results_payload=payload,
        )

    print("\n===== RESULTS =====")
    print(markdown_table(table_rows))
    print(f"\n[save] {run_dir / 'comparison.png'}")
    print(f"[save] {run_dir / 'results.md'}")


if __name__ == "__main__":
    main()
