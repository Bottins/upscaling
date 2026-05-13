# Benchmark Multiscala Inverse PINN

- Immagine: `C:\Users\alexq\Desktop\escience\datasets\Set5\Set5_HR\butterfly.png`
- Resize HR: `256x256`
- Epoche per metodo: `200`
- Device: `cpu`

## Sintesi

| Scenario | Best PSNR | Best SSIM | Bicubic PSNR | Bicubic SSIM | File |
| --- | --- | --- | ---: | ---: | --- |
| x1_deblur_denoise | ROF PDE + Shock (23.36) | TV energy (0.7472) | 22.27 | 0.6733 | [results.md](results\final_inverse_pinn_v1_size256\x1_deblur_denoise\results.md) |
| x2_sr | All PDE trainable (23.10) | Perona-Malik PDE (0.7539) | 21.47 | 0.6847 | [results.md](results\final_inverse_pinn_v1_size256\x2_sr\results.md) |
| x4_sr | TV energy (21.08) | ROF PDE + Shock (0.6746) | 17.97 | 0.5629 | [results.md](results\final_inverse_pinn_v1_size256\x4_sr\results.md) |

## Confronto Per Scenario

### x1_deblur_denoise

- Nota: Restauro x1: nessun downsampling, ma blur gaussiano ignoto + noise.
- Comparison: [comparison.png](results\final_inverse_pinn_v1_size256\x1_deblur_denoise\comparison.png)

| Metodo | PSNR | Delta PSNR vs Bicubic | SSIM | Delta SSIM vs Bicubic | Sigma stimato | Noise weights |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Bicubic | 22.27 | +0.00 | 0.6733 | +0.0000 | - | - |
| All PDE trainable | 23.36 | +1.09 | 0.7471 | +0.0737 | 0.7206 | (0.5908, 0.1389, 0.2703) |
| TV energy | 23.36 | +1.09 | 0.7472 | +0.0739 | 0.7152 | (0.591, 0.1363, 0.2727) |
| ROF PDE | 23.35 | +1.09 | 0.7472 | +0.0738 | 0.7152 | (0.5911, 0.1363, 0.2727) |
| Perona-Malik PDE | 23.36 | +1.09 | 0.7472 | +0.0739 | 0.7153 | (0.591, 0.1363, 0.2727) |
| Shock PDE | 23.35 | +1.09 | 0.7471 | +0.0738 | 0.7151 | (0.591, 0.1363, 0.2727) |
| ROF PDE + Shock | 23.36 | +1.09 | 0.7471 | +0.0737 | 0.7206 | (0.5908, 0.1389, 0.2703) |
| Perona-Malik + Shock | 23.35 | +1.09 | 0.7471 | +0.0738 | 0.7152 | (0.591, 0.1363, 0.2727) |

Pesi prior appresi:

- All PDE trainable: flat_noise=233.2, edge_sharpness=0.3871, tv=0.02335, rof=0.009351, pm=2.338, shock=0.7771
- TV energy: flat_noise=231.8, edge_sharpness=0.3895, tv=0.02321
- ROF PDE: flat_noise=231.8, edge_sharpness=0.3895, rof=0.01162
- Perona-Malik PDE: flat_noise=231.8, edge_sharpness=0.3895, pm=3.099
- Shock PDE: flat_noise=231.8, edge_sharpness=0.3895, shock=1.159
- ROF PDE + Shock: flat_noise=233.2, edge_sharpness=0.3871, rof=0.009351, shock=0.7771
- Perona-Malik + Shock: flat_noise=231.8, edge_sharpness=0.3895, pm=2.325, shock=0.7724

### x2_sr

- Nota: Super-resolution x2 con blur gaussiano ignoto + noise.
- Comparison: [comparison.png](results\final_inverse_pinn_v1_size256\x2_sr\comparison.png)

| Metodo | PSNR | Delta PSNR vs Bicubic | SSIM | Delta SSIM vs Bicubic | Sigma stimato | Noise weights |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Bicubic | 21.47 | +0.00 | 0.6847 | +0.0000 | - | - |
| All PDE trainable | 23.10 | +1.64 | 0.7536 | +0.0689 | 0.6994 | (0.5561, 0.1213, 0.3226) |
| TV energy | 23.09 | +1.62 | 0.7538 | +0.0691 | 0.6956 | (0.554, 0.1192, 0.3268) |
| ROF PDE | 23.07 | +1.61 | 0.7529 | +0.0682 | 0.6916 | (0.5516, 0.1172, 0.3311) |
| Perona-Malik PDE | 23.10 | +1.63 | 0.7539 | +0.0692 | 0.7119 | (0.5623, 0.1278, 0.31) |
| Shock PDE | 23.07 | +1.61 | 0.7533 | +0.0686 | 0.6955 | (0.5536, 0.1192, 0.3272) |
| ROF PDE + Shock | 23.08 | +1.62 | 0.7536 | +0.0688 | 0.6994 | (0.556, 0.1213, 0.3227) |
| Perona-Malik + Shock | 23.10 | +1.63 | 0.7537 | +0.0690 | 0.6955 | (0.554, 0.1192, 0.3268) |

Pesi prior appresi:

- All PDE trainable: flat_noise=223.8, edge_sharpness=0.4044, tv=0.0224, rof=0.008973, pm=2.24, shock=0.7449
- TV energy: flat_noise=222.5, edge_sharpness=0.4069, tv=0.02227
- ROF PDE: flat_noise=221.2, edge_sharpness=0.4094, rof=0.01109
- Perona-Malik PDE: flat_noise=227.8, edge_sharpness=0.3969, pm=3.041
- Shock PDE: flat_noise=222.5, edge_sharpness=0.4069, shock=1.111
- ROF PDE + Shock: flat_noise=223.8, edge_sharpness=0.4044, rof=0.008973, shock=0.7449
- Perona-Malik + Shock: flat_noise=222.5, edge_sharpness=0.4069, pm=2.228, shock=0.7406

### x4_sr

- Nota: Super-resolution x4 con blur gaussiano ignoto + noise.
- Comparison: [comparison.png](results\final_inverse_pinn_v1_size256\x4_sr\comparison.png)

| Metodo | PSNR | Delta PSNR vs Bicubic | SSIM | Delta SSIM vs Bicubic | Sigma stimato | Noise weights |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Bicubic | 17.97 | +0.00 | 0.5629 | +0.0000 | - | - |
| All PDE trainable | 21.07 | +3.09 | 0.6742 | +0.1112 | 0.7408 | (0.5846, 0.1487, 0.2667) |
| TV energy | 21.08 | +3.10 | 0.6710 | +0.1081 | 0.7186 | (0.5646, 0.1311, 0.3042) |
| ROF PDE | 21.06 | +3.08 | 0.6711 | +0.1082 | 0.7213 | (0.5695, 0.1333, 0.2972) |
| Perona-Malik PDE | 21.07 | +3.09 | 0.6726 | +0.1097 | 0.7187 | (0.5641, 0.1312, 0.3047) |
| Shock PDE | 21.06 | +3.08 | 0.6708 | +0.1079 | 0.7214 | (0.5687, 0.1334, 0.2979) |
| ROF PDE + Shock | 21.07 | +3.10 | 0.6746 | +0.1116 | 0.7371 | (0.5835, 0.1458, 0.2707) |
| Perona-Malik + Shock | 21.07 | +3.09 | 0.6726 | +0.1096 | 0.7239 | (0.5732, 0.1356, 0.2913) |

Pesi prior appresi:

- All PDE trainable: flat_noise=235.7, edge_sharpness=0.3825, tv=0.0236, rof=0.009435, pm=2.354, shock=0.7849
- TV energy: flat_noise=225.9, edge_sharpness=0.3998, tv=0.02263
- ROF PDE: flat_noise=227.2, edge_sharpness=0.3973, rof=0.01137
- Perona-Malik PDE: flat_noise=225.8, edge_sharpness=0.3998, pm=3.005
- Shock PDE: flat_noise=227.2, edge_sharpness=0.3973, shock=1.135
- ROF PDE + Shock: flat_noise=234.3, edge_sharpness=0.3849, rof=0.009377, shock=0.78
- Perona-Malik + Shock: flat_noise=228.6, edge_sharpness=0.3948, pm=2.282, shock=0.7611
