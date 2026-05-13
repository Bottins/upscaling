# x1_deblur_denoise

- Immagine: `C:\Users\alexq\Desktop\escience\datasets\Set5\Set5_HR\butterfly.png`
- Device: `cpu`
- Scala: `1`
- Nota: Restauro x1: nessun downsampling, ma blur gaussiano ignoto + noise.
- Sigma reale blur: `1.3500`
- Noise reale: weights=(0.45, 0.3, 0.25), gaussian_std=0.0300, laplace_scale=0.0200, speckle_std=0.0500

| Metodo | PSNR | Delta PSNR vs Bicubic | SSIM | Delta SSIM vs Bicubic | Sigma stimato | Noise weights |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Bicubic | 22.27 | +0.00 | 0.6733 | +0.0000 | - | - |
| All PDE trainable | 23.36 | +1.09 | 0.7471 | +0.0737 | 0.7206 | (0.5908, 0.1389, 0.2703) |
| TV energy [best SSIM] | 23.36 | +1.09 | 0.7472 | +0.0739 | 0.7152 | (0.591, 0.1363, 0.2727) |
| ROF PDE | 23.35 | +1.09 | 0.7472 | +0.0738 | 0.7152 | (0.5911, 0.1363, 0.2727) |
| Perona-Malik PDE | 23.36 | +1.09 | 0.7472 | +0.0739 | 0.7153 | (0.591, 0.1363, 0.2727) |
| Shock PDE | 23.35 | +1.09 | 0.7471 | +0.0738 | 0.7151 | (0.591, 0.1363, 0.2727) |
| ROF PDE + Shock [best PSNR] | 23.36 | +1.09 | 0.7471 | +0.0737 | 0.7206 | (0.5908, 0.1389, 0.2703) |
| Perona-Malik + Shock | 23.35 | +1.09 | 0.7471 | +0.0738 | 0.7152 | (0.591, 0.1363, 0.2727) |

## Pesi prior appresi

- All PDE trainable: flat_noise=233.2, edge_sharpness=0.3871, tv=0.02335, rof=0.009351, pm=2.338, shock=0.7771
- TV energy: flat_noise=231.8, edge_sharpness=0.3895, tv=0.02321
- ROF PDE: flat_noise=231.8, edge_sharpness=0.3895, rof=0.01162
- Perona-Malik PDE: flat_noise=231.8, edge_sharpness=0.3895, pm=3.099
- Shock PDE: flat_noise=231.8, edge_sharpness=0.3895, shock=1.159
- ROF PDE + Shock: flat_noise=233.2, edge_sharpness=0.3871, rof=0.009351, shock=0.7771
- Perona-Malik + Shock: flat_noise=231.8, edge_sharpness=0.3895, pm=2.325, shock=0.7724