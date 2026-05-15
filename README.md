# Inverse PINN for Blind Image Restoration and Super-Resolution

This repository implements a blind inverse-imaging pipeline based on:

- an implicit neural image representation
- a learnable blur and downsampling model
- trainable PDE-inspired regularization

The goal is to reconstruct a high-resolution image from a single degraded observation while jointly estimating unknown blur and heterogeneous noise.

The current synthetic benchmarks cover three settings:

- `x1_deblur_denoise`: blind deblurring plus denoising
- `x2_sr`: blind 2x super-resolution
- `x4_sr`: blind 4x super-resolution

At a high level, the method represents the unknown HR image with a coordinate MLP, passes it through a learnable degradation operator, compares the predicted LR image to the observed one, and regularizes the reconstruction with trainable PDE priors.

## Repository Layout

```text
.
|-- main.py
|-- benchmark_multiscale.py
|-- benchmark_multiscale_stabilized.py
|-- inverse_sr/
|   |-- pinn.py
|   |-- degradation.py
|   |-- priors.py
|   |-- solver.py
|   |-- solver_stabilized.py
|   |-- baselines.py
|   |-- metrics.py
|   `-- io.py
|-- datasets/
|   `-- Set5/Set5_HR/
|-- results/
|   |-- final_inverse_pinn_v1_size256/
|   `-- final_inverse_pinn_v1_size256_stabilized/
```

## Method Overview

### 1. Inverse problem

The synthetic observation model is:

```text
y = A_sigma_s(u) + noise
A_sigma_s(u) = downsample_s(gaussian_blur_sigma(u))
```

Where:

- `u` is the unknown HR image
- `y` is the observed degraded image
- `sigma` is the unknown Gaussian blur width
- `s` is the scale factor

When `scale = 1`, the problem becomes blind deblurring plus denoising.

Synthetic degradation is generated in [inverse_sr/degradation.py](inverse_sr/degradation.py), and the same operator family is re-estimated during inversion.

### 2. PINN-style image parameterization

The HR image is not optimized pixel by pixel. Instead, it is represented by a residual SIREN implemented in [inverse_sr/pinn.py](inverse_sr/pinn.py):

```text
u_theta(x) = soft_clamp(u_base(x) + rho * tanh(f_theta(2x - 1)))
```

Where:

- `u_base` is the bicubic upsampled observation
- `f_theta` is a sinusoidal coordinate network
- `rho` controls the residual amplitude
- `soft_clamp` keeps intensities inside a valid image range

Strictly speaking, this is an implicit neural representation with physics-inspired regularization, rather than a classical time-evolution PINN solver. The "PINN" aspect here comes from combining a continuous neural image model with PDE residual terms inside the inverse objective.

### 3. Blind noise model

The residual between the observed image and the predicted degraded image is modeled with a trainable mixture of:

- Gaussian noise
- Laplace noise
- signal-dependent speckle-like noise

The solver learns:

- mixture weights
- Gaussian standard deviation
- Laplace scale
- speckle scale

This logic is implemented in [inverse_sr/solver.py](inverse_sr/solver.py).

### 4. PDE and variational priors

The reconstruction is regularized with trainable prior families implemented in [inverse_sr/priors.py](inverse_sr/priors.py):

- `TV`: total variation energy
- `ROF PDE`: ROF-style PDE residual
- `Perona-Malik PDE`: anisotropic diffusion residual
- `Shock PDE`: shock-filter-inspired residual
- `flat_noise_loss`: penalizes high-frequency content in flat regions
- `edge_sharpness_loss`: rewards sharp gradients on edge regions

### 5. Joint optimization

The solver jointly estimates:

- image network parameters
- blur width
- noise scales and mixture weights
- active prior weights

The training objective combines:

- data fidelity under the trainable noise mixture
- a Charbonnier-style robust reconstruction term
- active PDE and variational priors
- noise regularization
- weight decay on trainable prior weights

In the default solver:

- prior terms are warmed up during the first `25%` of training
- the best checkpoint is selected by PSNR on the synthetic HR reference

Important benchmark note:
the HR image is used only for evaluation and model selection, not inside the inverse loss itself. That is acceptable for synthetic benchmarking, but it is not a fully reference-free protocol.

### 6. Stabilized variant

[inverse_sr/solver_stabilized.py](inverse_sr/solver_stabilized.py) adds a more conservative optimization schedule:

- common priors can be reduced with `--common-prior-scale`
- forward-model parameters and trainable prior weights can be frozen after a warm-up fraction using `--freeze-after-frac`

This variant is used by [benchmark_multiscale_stabilized.py](benchmark_multiscale_stabilized.py).

## Implemented Methods

The benchmark compares:

- `Bicubic`
- `All PDE trainable`
- `TV energy`
- `ROF PDE`
- `Perona-Malik PDE`
- `Shock PDE`
- `ROF PDE + Shock`
- `Perona-Malik + Shock`

For inverse methods, the blur parameter, noise mixture, and active prior weights are optimized jointly.

## Installation

Install the minimal dependencies with:

```bash
pip install -r requirements.txt
```

Current dependencies:

- `torch>=2.0`
- `numpy`
- `Pillow`

## Data

The repository already includes Set5 examples under `datasets/Set5/Set5_HR/`.

If no image is passed explicitly, the code falls back to `butterfly.png` when available.

## Usage

### Single-scale run

Runs one scenario with a chosen scale.

```bash
python main.py --image butterfly.png --size 256 --scale 4 --epochs 200 --out results/my_single_run
```

### Multiscale benchmark

Runs the default `x1`, `x2`, and `x4` benchmark suite.

```bash
python benchmark_multiscale.py --image butterfly.png --size 256 --epochs 200 --out results/my_multiscale_run
```

You can also restrict the benchmark to specific scenarios:

```bash
python benchmark_multiscale.py --image butterfly.png --scenarios x2_sr x4_sr --out results/my_subset_run
```

### Stabilized multiscale benchmark

Runs the same suite with conservative prior scaling and parameter freezing.

```bash
python benchmark_multiscale_stabilized.py --image butterfly.png --size 256 --epochs 200 --common-prior-scale 0.25 --freeze-after-frac 0.25 --out results/my_stabilized_run
```

## Outputs

Each scenario saves:

- `hr.png`
- `lr_clean.png`
- `lr_observed.png`
- one image per method
- `comparison.png`
- `results.md`
- `results.json`

The comparison strip also includes zoomed crops and optimization-history plots.

## Included Benchmark Results

The repository already contains a complete benchmark report at [results/final_inverse_pinn_v1_size256/benchmark_report.md](results/final_inverse_pinn_v1_size256/benchmark_report.md).

These results were obtained with:

- image: `datasets/Set5/Set5_HR/butterfly.png`
- HR resize: `256 x 256`
- epochs per method: `200`
- device: `cpu`
- true blur sigma: `1.35`
- true noise mixture weights: `(0.45, 0.30, 0.25)`
- true Gaussian std / Laplace scale / speckle std: `0.03 / 0.02 / 0.05`

### Summary table

| Scenario | Bicubic PSNR / SSIM | Best PSNR | Best SSIM | Gain over bicubic |
| --- | --- | --- | --- | --- |
| `x1_deblur_denoise` | `22.27 / 0.6733` | `ROF PDE + Shock` at `23.36 dB` | `TV energy` at `0.7472` | about `+1.09 dB` |
| `x2_sr` | `21.47 / 0.6847` | `All PDE trainable` at `23.10 dB` | `Perona-Malik PDE` at `0.7539` | about `+1.64 dB` |
| `x4_sr` | `17.97 / 0.5629` | `TV energy` at `21.08 dB` | `ROF PDE + Shock` at `0.6746` | about `+3.10 dB` |

### Per-scenario highlights

- `x1_deblur_denoise`: all inverse variants cluster around `23.35` to `23.36 dB`, well above bicubic. The best SSIM in the report is `0.7472`, and the learned blur estimate is around `0.715`.
- `x2_sr`: the best PSNR is `23.10 dB` and the best SSIM is `0.7539`, with the Perona-Malik PDE configuration delivering the strongest structure preservation.
- `x4_sr`: this is the hardest case and also the one with the largest improvement. Bicubic reaches `17.97 dB`, while the best inverse model reaches `21.08 dB`.

Overall, the hardest upsampling regime (`x4`) benefits the most from the inverse formulation, which is consistent with the fact that bicubic interpolation alone cannot recover unknown blur and mixed noise.

### Figures

Qualitative grid:

![Qualitative grid](results/final_inverse_pinn_v1_size256/paper_qualitative_grid.png)

Convergence summary:

![Convergence summary](results/final_inverse_pinn_v1_size256/paper_convergence_summary.png)

Scenario reports:

- [x1_deblur_denoise/results.md](results/final_inverse_pinn_v1_size256/x1_deblur_denoise/results.md)
- [x2_sr/results.md](results/final_inverse_pinn_v1_size256/x2_sr/results.md)
- [x4_sr/results.md](results/final_inverse_pinn_v1_size256/x4_sr/results.md)

## Experimental Notes

- The synthetic degradation model is known when generating data but treated as unknown during reconstruction.
- `scale = 1` is a deblurring and denoising problem because downsampling becomes the identity.
- The benchmark is reproducible from the scripts in this repository, but best-checkpoint selection currently relies on the ground-truth HR image.
- A stabilized variant of the same benchmark is also available in [results/final_inverse_pinn_v1_size256_stabilized](results/final_inverse_pinn_v1_size256_stabilized).
