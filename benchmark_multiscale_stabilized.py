from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from main import default_configs, resolve_device, slugify
from inverse_sr.baselines import bicubic
from inverse_sr.degradation import NoiseTruth, SyntheticDegradation, make_observation
from inverse_sr.io import ensure_dir, load_image, pick_image, save_comparison_strip, save_image
from inverse_sr.metrics import compute_all
from inverse_sr.solver_stabilized import (
    StabilizedInverseSolverConfig,
    solve_inverse_problem_stabilized,
)


@dataclass(frozen=True)
class Scenario:
    key: str
    scale: int
    sigma_true: float
    note: str


SCENARIOS = [
    Scenario(
        key="x1_deblur_denoise",
        scale=1,
        sigma_true=1.35,
        note="x1 restoration: no downsampling, but unknown Gaussian blur + noise.",
    ),
    Scenario(key="x2_sr", scale=2, sigma_true=1.35, note="x2 super-resolution with unknown Gaussian blur + noise."),
    Scenario(key="x4_sr", scale=4, sigma_true=1.35, note="x4 super-resolution with unknown Gaussian blur + noise."),
]


def observed_title(scale: int) -> str:
    return "Observed degraded" if scale == 1 else "Observed LR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental multiscale benchmark: reduces shared priors and freezes "
            "the forward model / trainable priors after warm-up."
        )
    )
    parser.add_argument("--image", type=str, default="butterfly.png")
    parser.add_argument("--dataset-root", type=str, default="datasets")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gaussian-std", type=float, default=0.03)
    parser.add_argument("--laplace-scale", type=float, default=0.02)
    parser.add_argument("--speckle-std", type=float, default=0.05)
    parser.add_argument("--noise-weights", type=float, nargs=3, default=(0.45, 0.30, 0.25))
    parser.add_argument("--common-prior-scale", type=float, default=0.25)
    parser.add_argument("--freeze-after-frac", type=float, default=0.25)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Subset of scenarios to run, using the keys defined in SCENARIOS.",
    )
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def _format_tuple(values: list[float] | tuple[float, ...]) -> str:
    return "(" + ", ".join(f"{value:.4g}" for value in values) + ")"


def stabilized_configs(common_prior_scale: float) -> list[dict]:
    configs: list[dict] = []
    for config in default_configs():
        copied = dict(config)
        if copied.get("kind") == "inverse":
            scaled = dict(copied.get("prior_weights", {}))
            if "flat_noise" in scaled:
                scaled["flat_noise"] *= common_prior_scale
            if "edge_sharpness" in scaled:
                scaled["edge_sharpness"] *= common_prior_scale
            copied["prior_weights"] = scaled
        configs.append(copied)
    return configs


def _comparison_entries(
    hr: torch.Tensor,
    observed_display: torch.Tensor,
    observed_metrics: dict[str, float],
    scenario: Scenario,
    method_entries: list[dict],
) -> list[dict]:
    entries = [
        {
            "title": "Ground truth",
            "image": hr,
            "lines": [
                f"size {hr.shape[-1]}x{hr.shape[-2]}",
                f"scale x{scenario.scale}",
                f"sigma true {scenario.sigma_true:.2f}",
            ],
        },
        {
            "title": observed_title(scenario.scale),
            "image": observed_display,
            "lines": [
                "degraded input",
                f"PSNR {observed_metrics['psnr']:.2f} dB",
                f"SSIM {observed_metrics['ssim']:.4f}",
            ],
        },
    ]
    entries.extend(method_entries)
    return entries


def _write_scale_report(
    scenario_dir: Path,
    scenario: Scenario,
    image_path: Path,
    device: str,
    degradation: SyntheticDegradation,
    rows: list[dict],
    methods_payload: dict,
    args: argparse.Namespace,
) -> None:
    lines = [
        f"# {scenario.key}",
        "",
        f"- Image: `{image_path}`",
        f"- Device: `{device}`",
        f"- Scale: `{scenario.scale}`",
        f"- Note: {scenario.note}",
        f"- Ground-truth blur sigma: `{degradation.sigma:.4f}`",
        f"- common_prior_scale: `{args.common_prior_scale}`",
        f"- freeze_after_frac: `{args.freeze_after_frac}`",
        (
            "- Ground-truth noise: "
            f"weights={_format_tuple(degradation.noise.weights)}, "
            f"gaussian_std={degradation.noise.gaussian_std:.4f}, "
            f"laplace_scale={degradation.noise.laplace_scale:.4f}, "
            f"speckle_std={degradation.noise.speckle_std:.4f}"
        ),
        "",
        "| Method | PSNR | Delta PSNR vs Bicubic | SSIM | Delta SSIM vs Bicubic | Estimated sigma | Noise weights |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    bic_psnr = next(row["psnr"] for row in rows if row["name"] == "Bicubic")
    bic_ssim = next(row["ssim"] for row in rows if row["name"] == "Bicubic")
    best_psnr = max(row["psnr"] for row in rows)
    best_ssim = max(row["ssim"] for row in rows)

    for row in rows:
        name = row["name"]
        if row["psnr"] == best_psnr:
            name += " [best PSNR]"
        if row["ssim"] == best_ssim:
            name += " [best SSIM]"
        lines.append(
            f"| {name} | {row['psnr']:.2f} | {row['psnr'] - bic_psnr:+.2f} | "
            f"{row['ssim']:.4f} | {row['ssim'] - bic_ssim:+.4f} | "
            f"{row['sigma']} | {row['noise_weights']} |"
        )

    lines.extend(["", "## Learned Prior Weights", ""])
    found = False
    for payload in methods_payload.values():
        prior_weights = payload.get("prior_weights")
        if not prior_weights:
            continue
        found = True
        pretty = ", ".join(f"{key}={value:.4g}" for key, value in prior_weights.items())
        lines.append(f"- {payload.get('display_name', 'method')}: {pretty}")
    if not found:
        lines.append("- No trainable priors in this scenario.")

    (scenario_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")


def _build_master_report(
    out_dir: Path,
    image_path: Path,
    size: int,
    epochs: int,
    device: str,
    scenario_results: dict[str, dict],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Stabilized Inverse PINN Multiscale Benchmark",
        "",
        f"- Image: `{image_path}`",
        f"- HR resize: `{size}x{size}`",
        f"- Epochs per method: `{epochs}`",
        f"- Device: `{device}`",
        f"- common_prior_scale: `{args.common_prior_scale}`",
        f"- freeze_after_frac: `{args.freeze_after_frac}`",
        "",
        "## Summary",
        "",
        "| Scenario | Best PSNR | Best SSIM | Bicubic PSNR | Bicubic SSIM | File |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]

    for scenario_key, payload in scenario_results.items():
        rows = payload["rows"]
        bic = next(row for row in rows if row["name"] == "Bicubic")
        best_psnr_row = max(rows, key=lambda row: row["psnr"])
        best_ssim_row = max(rows, key=lambda row: row["ssim"])
        lines.append(
            f"| {scenario_key} | {best_psnr_row['name']} ({best_psnr_row['psnr']:.2f}) | "
            f"{best_ssim_row['name']} ({best_ssim_row['ssim']:.4f}) | "
            f"{bic['psnr']:.2f} | {bic['ssim']:.4f} | "
            f"[results.md]({out_dir / scenario_key / 'results.md'}) |"
        )

    lines.extend(["", "## Scenario Comparison", ""])

    for scenario_key, payload in scenario_results.items():
        scenario = payload["scenario"]
        rows = payload["rows"]
        lines.extend([
            f"### {scenario_key}",
            "",
            f"- Note: {scenario.note}",
            f"- Comparison: [comparison.png]({out_dir / scenario_key / 'comparison.png'})",
            "",
            "| Method | PSNR | Delta PSNR vs Bicubic | SSIM | Delta SSIM vs Bicubic | Estimated sigma | Noise weights |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        bic_psnr = next(row["psnr"] for row in rows if row["name"] == "Bicubic")
        bic_ssim = next(row["ssim"] for row in rows if row["name"] == "Bicubic")
        for row in rows:
            lines.append(
                f"| {row['name']} | {row['psnr']:.2f} | {row['psnr'] - bic_psnr:+.2f} | "
                f"{row['ssim']:.4f} | {row['ssim'] - bic_ssim:+.4f} | "
                f"{row['sigma']} | {row['noise_weights']} |"
            )
        lines.extend(["", "Learned prior weights:", ""])
        found = False
        for method_id, method_payload in payload["methods"].items():
            prior_weights = method_payload.get("prior_weights")
            display_name = method_payload.get("display_name", method_id)
            if not prior_weights:
                continue
            found = True
            pretty = ", ".join(f"{key}={value:.4g}" for key, value in prior_weights.items())
            lines.append(f"- {display_name}: {pretty}")
        if not found:
            lines.append("- No trainable priors.")
        lines.append("")

    (out_dir / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    image_path = pick_image(args.image, Path(args.dataset_root))
    hr = load_image(image_path, size=(args.size, args.size))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path("results") / f"stabilized_benchmark_{image_path.stem}_{stamp}"
    ensure_dir(out_dir)

    methods = stabilized_configs(args.common_prior_scale)
    scenario_results: dict[str, dict] = {}
    selected_scenarios = SCENARIOS
    if args.scenarios:
        allowed = {item.lower() for item in args.scenarios}
        selected_scenarios = [scenario for scenario in SCENARIOS if scenario.key.lower() in allowed]
        if not selected_scenarios:
            available = ", ".join(scenario.key for scenario in SCENARIOS)
            raise RuntimeError(f"No valid scenario selected. Available: {available}")

    for scenario in selected_scenarios:
        print(f"\n===== {scenario.key} =====")
        scenario_dir = out_dir / scenario.key
        ensure_dir(scenario_dir)

        degradation = SyntheticDegradation(
            scale=scenario.scale,
            sigma=scenario.sigma_true,
            noise=NoiseTruth(
                gaussian_std=args.gaussian_std,
                laplace_scale=args.laplace_scale,
                speckle_std=args.speckle_std,
                weights=tuple(args.noise_weights),
            ),
            seed=args.seed,
        )
        lr_clean, lr_observed = make_observation(hr, degradation)

        observed_display = F.interpolate(
            lr_observed.unsqueeze(0),
            size=hr.shape[-2:],
            mode="nearest",
        ).squeeze(0).clamp(0, 1)
        observed_metrics = compute_all(observed_display, hr)
        bicubic_reconstruction = bicubic(lr_observed, hr.shape[-2:])
        bicubic_metrics = compute_all(bicubic_reconstruction, hr)

        save_image(hr, scenario_dir / "hr.png")
        save_image(lr_clean, scenario_dir / "lr_clean.png")
        save_image(lr_observed, scenario_dir / "lr_observed.png")

        rows: list[dict] = []
        methods_payload: dict[str, dict] = {}
        method_images: dict[str, torch.Tensor] = {}
        comparison_method_entries: list[dict] = []

        solver_cfg = StabilizedInverseSolverConfig(
            scale=scenario.scale,
            epochs=args.epochs,
            device=device,
            freeze_forward_after_frac=args.freeze_after_frac,
            freeze_prior_after_frac=args.freeze_after_frac,
        )

        for method in methods:
            if method["kind"] == "bicubic":
                reconstruction = bicubic_reconstruction
                metrics = bicubic_metrics
                methods_payload[method["id"]] = {
                    "display_name": method["name"],
                    "metrics": metrics,
                }
                method_images["bicubic"] = reconstruction
                rows.append({
                    "name": method["name"],
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "sigma": "-",
                    "noise_weights": "-",
                })
                comparison_method_entries.append({
                    "title": method["name"],
                    "image": reconstruction,
                    "lines": [
                        f"PSNR {metrics['psnr']:.2f} dB",
                        f"SSIM {metrics['ssim']:.4f}",
                    ],
                    "history": [],
                })
                save_image(reconstruction, scenario_dir / "bicubic.png")
                print(f"[{scenario.key}] Bicubic: PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f}")
                continue

            result = solve_inverse_problem_stabilized(
                lr_observed=lr_observed,
                hr_reference=hr,
                cfg=solver_cfg,
                mode=method["id"],
                display_name=method["name"],
                prior_weights=method.get("prior_weights", {}),
                seed=args.seed,
            )
            methods_payload[method["id"]] = result.to_serializable()
            image_key = slugify(method["name"])
            method_images[image_key] = result.reconstruction.cpu()
            rows.append({
                "name": method["name"],
                "psnr": result.metrics["psnr"],
                "ssim": result.metrics["ssim"],
                "sigma": f"{result.estimated_sigma:.4f}",
                "noise_weights": _format_tuple(result.noise_weights),
            })
            comparison_method_entries.append({
                "title": method["name"],
                "image": result.reconstruction.cpu(),
                "lines": [
                    f"PSNR {result.metrics['psnr']:.2f} dB",
                    f"SSIM {result.metrics['ssim']:.4f}",
                    f"sigma {result.estimated_sigma:.3f}",
                ],
                "history": result.history,
            })
            save_image(result.reconstruction.cpu(), scenario_dir / f"{image_key}.png")
            print(
                f"[{scenario.key}] {method['name']}: "
                f"PSNR={result.metrics['psnr']:.2f} SSIM={result.metrics['ssim']:.4f} "
                f"sigma={result.estimated_sigma:.3f}"
            )

        comparison_entries = _comparison_entries(
            hr=hr,
            observed_display=observed_display,
            observed_metrics=observed_metrics,
            scenario=scenario,
            method_entries=comparison_method_entries,
        )
        save_comparison_strip(comparison_entries, scenario_dir / "comparison.png")

        payload = {
            "scenario": asdict(scenario),
            "image": str(image_path),
            "device": device,
            "degradation_truth": asdict(degradation),
            "experiment": {
                "common_prior_scale": args.common_prior_scale,
                "freeze_after_frac": args.freeze_after_frac,
            },
            "rows": rows,
            "methods": methods_payload,
        }
        with (scenario_dir / "results.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        _write_scale_report(
            scenario_dir=scenario_dir,
            scenario=scenario,
            image_path=image_path,
            device=device,
            degradation=degradation,
            rows=rows,
            methods_payload=methods_payload,
            args=args,
        )
        scenario_results[scenario.key] = {
            "scenario": scenario,
            "rows": rows,
            "methods": methods_payload,
        }

    _build_master_report(
        out_dir=out_dir,
        image_path=image_path,
        size=args.size,
        epochs=args.epochs,
        device=device,
        scenario_results=scenario_results,
        args=args,
    )
    print(f"\n[done] report salvato in {out_dir}")


if __name__ == "__main__":
    main()
