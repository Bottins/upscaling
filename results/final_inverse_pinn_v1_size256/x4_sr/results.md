# x4_sr

- Immagine: `C:\Users\alexq\Desktop\escience\datasets\Set5\Set5_HR\butterfly.png`
- Device: `cpu`
- Scala: `4`
- Nota: Super-resolution x4 con blur gaussiano ignoto + noise.
- Sigma reale blur: `1.3500`
- Noise reale: weights=(0.45, 0.3, 0.25), gaussian_std=0.0300, laplace_scale=0.0200, speckle_std=0.0500

| Metodo | PSNR | Delta PSNR vs Bicubic | SSIM | Delta SSIM vs Bicubic | Sigma stimato | Noise weights |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Bicubic | 17.97 | +0.00 | 0.5629 | +0.0000 | - | - |
| All PDE trainable | 21.07 | +3.09 | 0.6742 | +0.1112 | 0.7408 | (0.5846, 0.1487, 0.2667) |
| TV energy [best PSNR] | 21.08 | +3.10 | 0.6710 | +0.1081 | 0.7186 | (0.5646, 0.1311, 0.3042) |
| ROF PDE | 21.06 | +3.08 | 0.6711 | +0.1082 | 0.7213 | (0.5695, 0.1333, 0.2972) |
| Perona-Malik PDE | 21.07 | +3.09 | 0.6726 | +0.1097 | 0.7187 | (0.5641, 0.1312, 0.3047) |
| Shock PDE | 21.06 | +3.08 | 0.6708 | +0.1079 | 0.7214 | (0.5687, 0.1334, 0.2979) |
| ROF PDE + Shock [best SSIM] | 21.07 | +3.10 | 0.6746 | +0.1116 | 0.7371 | (0.5835, 0.1458, 0.2707) |
| Perona-Malik + Shock | 21.07 | +3.09 | 0.6726 | +0.1096 | 0.7239 | (0.5732, 0.1356, 0.2913) |

## Pesi prior appresi

- All PDE trainable: flat_noise=235.7, edge_sharpness=0.3825, tv=0.0236, rof=0.009435, pm=2.354, shock=0.7849
- TV energy: flat_noise=225.9, edge_sharpness=0.3998, tv=0.02263
- ROF PDE: flat_noise=227.2, edge_sharpness=0.3973, rof=0.01137
- Perona-Malik PDE: flat_noise=225.8, edge_sharpness=0.3998, pm=3.005
- Shock PDE: flat_noise=227.2, edge_sharpness=0.3973, shock=1.135
- ROF PDE + Shock: flat_noise=234.3, edge_sharpness=0.3849, rof=0.009377, shock=0.78
- Perona-Malik + Shock: flat_noise=228.6, edge_sharpness=0.3948, pm=2.282, shock=0.7611