# x2_sr

- Immagine: `C:\Users\alexq\Desktop\escience\datasets\Set5\Set5_HR\butterfly.png`
- Device: `cpu`
- Scala: `2`
- Nota: Super-resolution x2 con blur gaussiano ignoto + noise.
- Sigma reale blur: `1.3500`
- common_prior_scale: `0.25`
- freeze_after_frac: `0.25`
- Noise reale: weights=(0.45, 0.3, 0.25), gaussian_std=0.0300, laplace_scale=0.0200, speckle_std=0.0500

| Metodo | PSNR | Delta PSNR vs Bicubic | SSIM | Delta SSIM vs Bicubic | Sigma stimato | Noise weights |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Bicubic | 21.47 | +0.00 | 0.6847 | +0.0000 | - | - |
| All PDE trainable | 23.18 | +1.71 | 0.7482 | +0.0634 | 0.7077 | (0.5604, 0.1255, 0.3141) |
| TV energy | 23.17 | +1.71 | 0.7490 | +0.0643 | 0.7078 | (0.5603, 0.1255, 0.3142) |
| ROF PDE | 23.18 | +1.71 | 0.7498 | +0.0651 | 0.7078 | (0.5603, 0.1255, 0.3142) |
| Perona-Malik PDE [best PSNR] | 23.18 | +1.72 | 0.7494 | +0.0647 | 0.7079 | (0.5604, 0.1255, 0.3141) |
| Shock PDE | 23.18 | +1.71 | 0.7497 | +0.0650 | 0.7077 | (0.5605, 0.1255, 0.314) |
| ROF PDE + Shock | 23.17 | +1.70 | 0.7465 | +0.0618 | 0.7077 | (0.5605, 0.1255, 0.314) |
| Perona-Malik + Shock [best SSIM] | 23.18 | +1.71 | 0.7520 | +0.0673 | 0.7077 | (0.5603, 0.1255, 0.3142) |

## Pesi prior appresi

- All PDE trainable: flat_noise=56.6, edge_sharpness=0.09985, tv=0.02266, rof=0.009077, pm=2.267, shock=0.7537
- TV energy: flat_noise=56.6, edge_sharpness=0.09985, tv=0.02266
- ROF PDE: flat_noise=56.6, edge_sharpness=0.09985, rof=0.01135
- Perona-Malik PDE: flat_noise=56.6, edge_sharpness=0.09985, pm=3.023
- Shock PDE: flat_noise=56.6, edge_sharpness=0.09985, shock=1.131
- ROF PDE + Shock: flat_noise=56.6, edge_sharpness=0.09985, rof=0.009077, shock=0.7537
- Perona-Malik + Shock: flat_noise=56.6, edge_sharpness=0.09985, pm=2.267, shock=0.7537