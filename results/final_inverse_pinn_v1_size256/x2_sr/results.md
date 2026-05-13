# x2_sr

- Immagine: `C:\Users\alexq\Desktop\escience\datasets\Set5\Set5_HR\butterfly.png`
- Device: `cpu`
- Scala: `2`
- Nota: Super-resolution x2 con blur gaussiano ignoto + noise.
- Sigma reale blur: `1.3500`
- Noise reale: weights=(0.45, 0.3, 0.25), gaussian_std=0.0300, laplace_scale=0.0200, speckle_std=0.0500

| Metodo | PSNR | Delta PSNR vs Bicubic | SSIM | Delta SSIM vs Bicubic | Sigma stimato | Noise weights |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Bicubic | 21.47 | +0.00 | 0.6847 | +0.0000 | - | - |
| All PDE trainable [best PSNR] | 23.10 | +1.64 | 0.7536 | +0.0689 | 0.6994 | (0.5561, 0.1213, 0.3226) |
| TV energy | 23.09 | +1.62 | 0.7538 | +0.0691 | 0.6956 | (0.554, 0.1192, 0.3268) |
| ROF PDE | 23.07 | +1.61 | 0.7529 | +0.0682 | 0.6916 | (0.5516, 0.1172, 0.3311) |
| Perona-Malik PDE [best SSIM] | 23.10 | +1.63 | 0.7539 | +0.0692 | 0.7119 | (0.5623, 0.1278, 0.31) |
| Shock PDE | 23.07 | +1.61 | 0.7533 | +0.0686 | 0.6955 | (0.5536, 0.1192, 0.3272) |
| ROF PDE + Shock | 23.08 | +1.62 | 0.7536 | +0.0688 | 0.6994 | (0.556, 0.1213, 0.3227) |
| Perona-Malik + Shock | 23.10 | +1.63 | 0.7537 | +0.0690 | 0.6955 | (0.554, 0.1192, 0.3268) |

## Pesi prior appresi

- All PDE trainable: flat_noise=223.8, edge_sharpness=0.4044, tv=0.0224, rof=0.008973, pm=2.24, shock=0.7449
- TV energy: flat_noise=222.5, edge_sharpness=0.4069, tv=0.02227
- ROF PDE: flat_noise=221.2, edge_sharpness=0.4094, rof=0.01109
- Perona-Malik PDE: flat_noise=227.8, edge_sharpness=0.3969, pm=3.041
- Shock PDE: flat_noise=222.5, edge_sharpness=0.4069, shock=1.111
- ROF PDE + Shock: flat_noise=223.8, edge_sharpness=0.4044, rof=0.008973, shock=0.7449
- Perona-Malik + Shock: flat_noise=222.5, edge_sharpness=0.4069, pm=2.228, shock=0.7406