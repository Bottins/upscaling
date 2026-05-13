# x1_deblur_denoise

- Immagine: `C:\Users\alexq\Desktop\escience\datasets\Set5\Set5_HR\butterfly.png`
- Device: `cpu`
- Scala: `1`
- Nota: Restauro x1: nessun downsampling, ma blur gaussiano ignoto + noise.
- Sigma reale blur: `1.3500`
- common_prior_scale: `0.25`
- freeze_after_frac: `0.25`
- Noise reale: weights=(0.45, 0.3, 0.25), gaussian_std=0.0300, laplace_scale=0.0200, speckle_std=0.0500

| Metodo | PSNR | Delta PSNR vs Bicubic | SSIM | Delta SSIM vs Bicubic | Sigma stimato | Noise weights |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Bicubic | 22.27 | +0.00 | 0.6733 | +0.0000 | - | - |
| All PDE trainable | 23.44 | +1.17 | 0.7500 | +0.0767 | 0.6938 | (0.5908, 0.1267, 0.2825) |
| TV energy [best PSNR] | 23.45 | +1.18 | 0.7502 | +0.0768 | 0.6939 | (0.5908, 0.1267, 0.2825) |
| ROF PDE | 23.44 | +1.18 | 0.7500 | +0.0767 | 0.6939 | (0.5908, 0.1267, 0.2825) |
| Perona-Malik PDE | 23.44 | +1.18 | 0.7503 | +0.0769 | 0.6939 | (0.5908, 0.1267, 0.2825) |
| Shock PDE | 23.44 | +1.18 | 0.7503 | +0.0769 | 0.6938 | (0.5908, 0.1267, 0.2825) |
| ROF PDE + Shock | 23.44 | +1.17 | 0.7503 | +0.0769 | 0.6938 | (0.5908, 0.1267, 0.2825) |
| Perona-Malik + Shock [best SSIM] | 23.44 | +1.18 | 0.7505 | +0.0771 | 0.6938 | (0.5908, 0.1267, 0.2825) |

## Pesi prior appresi

- All PDE trainable: flat_noise=56.6, edge_sharpness=0.09985, tv=0.02267, rof=0.009082, pm=2.271, shock=0.7543
- TV energy: flat_noise=56.6, edge_sharpness=0.09985, tv=0.02267
- ROF PDE: flat_noise=56.6, edge_sharpness=0.09985, rof=0.01135
- Perona-Malik PDE: flat_noise=56.6, edge_sharpness=0.09985, pm=3.028
- Shock PDE: flat_noise=56.6, edge_sharpness=0.09985, shock=1.131
- ROF PDE + Shock: flat_noise=56.6, edge_sharpness=0.09985, rof=0.009082, shock=0.7543
- Perona-Malik + Shock: flat_noise=56.6, edge_sharpness=0.09985, pm=2.271, shock=0.7543