# x4_sr

- Immagine: `C:\Users\alexq\Desktop\escience\datasets\Set5\Set5_HR\butterfly.png`
- Device: `cpu`
- Scala: `4`
- Nota: Super-resolution x4 con blur gaussiano ignoto + noise.
- Sigma reale blur: `1.3500`
- common_prior_scale: `0.25`
- freeze_after_frac: `0.25`
- Noise reale: weights=(0.45, 0.3, 0.25), gaussian_std=0.0300, laplace_scale=0.0200, speckle_std=0.0500

| Metodo | PSNR | Delta PSNR vs Bicubic | SSIM | Delta SSIM vs Bicubic | Sigma stimato | Noise weights |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Bicubic | 17.97 | +0.00 | 0.5629 | +0.0000 | - | - |
| All PDE trainable [best SSIM] | 21.06 | +3.08 | 0.6729 | +0.1100 | 0.7304 | (0.5786, 0.1404, 0.281) |
| TV energy [best PSNR] | 21.07 | +3.09 | 0.6659 | +0.1030 | 0.7186 | (0.5657, 0.131, 0.3033) |
| ROF PDE | 21.05 | +3.07 | 0.6713 | +0.1084 | 0.7244 | (0.5734, 0.1355, 0.2911) |
| Perona-Malik PDE | 21.06 | +3.09 | 0.6725 | +0.1095 | 0.7306 | (0.578, 0.1405, 0.2815) |
| Shock PDE | 21.05 | +3.07 | 0.6661 | +0.1032 | 0.7186 | (0.5638, 0.1312, 0.305) |
| ROF PDE + Shock | 21.05 | +3.08 | 0.6712 | +0.1083 | 0.7242 | (0.5717, 0.1356, 0.2927) |
| Perona-Malik + Shock | 21.03 | +3.06 | 0.6628 | +0.0999 | 0.7186 | (0.5643, 0.1311, 0.3046) |

## Pesi prior appresi

- All PDE trainable: flat_noise=57.85, edge_sharpness=0.09746, tv=0.02317, rof=0.009262, pm=2.31, shock=0.7705
- TV energy: flat_noise=56.46, edge_sharpness=0.09994, tv=0.02263
- ROF PDE: flat_noise=57.15, edge_sharpness=0.09869, rof=0.01144
- Perona-Malik PDE: flat_noise=57.85, edge_sharpness=0.09746, pm=3.08
- Shock PDE: flat_noise=56.46, edge_sharpness=0.09994, shock=1.128
- ROF PDE + Shock: flat_noise=57.15, edge_sharpness=0.09869, rof=0.009151, shock=0.7611
- Perona-Malik + Shock: flat_noise=56.46, edge_sharpness=0.09994, pm=2.253, shock=0.752