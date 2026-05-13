# Benchmark Multiscala Inverse PINN Stabilized

- Immagine: `C:\Users\alexq\Desktop\escience\datasets\Set5\Set5_HR\butterfly.png`
- Resize HR: `256x256`
- Epoche per metodo: `200`
- Device: `cpu`
- common_prior_scale: `0.25`
- freeze_after_frac: `0.25`

## Sintesi

| Scenario | Best PSNR | Best SSIM | Bicubic PSNR | Bicubic SSIM | File |
| --- | --- | --- | ---: | ---: | --- |
| x1_deblur_denoise | TV energy (23.45) | Perona-Malik + Shock (0.7505) | 22.27 | 0.6733 | [results.md](results\final_inverse_pinn_v1_size256_stabilized\x1_deblur_denoise\results.md) |
| x2_sr | Perona-Malik PDE (23.18) | Perona-Malik + Shock (0.7520) | 21.47 | 0.6847 | [results.md](results\final_inverse_pinn_v1_size256_stabilized\x2_sr\results.md) |
| x4_sr | TV energy (21.07) | All PDE trainable (0.6729) | 17.97 | 0.5629 | [results.md](results\final_inverse_pinn_v1_size256_stabilized\x4_sr\results.md) |

## Confronto Per Scenario

### x1_deblur_denoise

- Nota: Restauro x1: nessun downsampling, ma blur gaussiano ignoto + noise.
- Comparison: [comparison.png](results\final_inverse_pinn_v1_size256_stabilized\x1_deblur_denoise\comparison.png)

| Metodo | PSNR | Delta PSNR vs Bicubic | SSIM | Delta SSIM vs Bicubic | Sigma stimato | Noise weights |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Bicubic | 22.27 | +0.00 | 0.6733 | +0.0000 | - | - |
| All PDE trainable | 23.44 | +1.17 | 0.7500 | +0.0767 | 0.6938 | (0.5908, 0.1267, 0.2825) |
| TV energy | 23.45 | +1.18 | 0.7502 | +0.0768 | 0.6939 | (0.5908, 0.1267, 0.2825) |
| ROF PDE | 23.44 | +1.18 | 0.7500 | +0.0767 | 0.6939 | (0.5908, 0.1267, 0.2825) |
| Perona-Malik PDE | 23.44 | +1.18 | 0.7503 | +0.0769 | 0.6939 | (0.5908, 0.1267, 0.2825) |
| Shock PDE | 23.44 | +1.18 | 0.7503 | +0.0769 | 0.6938 | (0.5908, 0.1267, 0.2825) |
| ROF PDE + Shock | 23.44 | +1.17 | 0.7503 | +0.0769 | 0.6938 | (0.5908, 0.1267, 0.2825) |
| Perona-Malik + Shock | 23.44 | +1.18 | 0.7505 | +0.0771 | 0.6938 | (0.5908, 0.1267, 0.2825) |

Pesi prior appresi:

- All PDE trainable: flat_noise=56.6, edge_sharpness=0.09985, tv=0.02267, rof=0.009082, pm=2.271, shock=0.7543
- TV energy: flat_noise=56.6, edge_sharpness=0.09985, tv=0.02267
- ROF PDE: flat_noise=56.6, edge_sharpness=0.09985, rof=0.01135
- Perona-Malik PDE: flat_noise=56.6, edge_sharpness=0.09985, pm=3.028
- Shock PDE: flat_noise=56.6, edge_sharpness=0.09985, shock=1.131
- ROF PDE + Shock: flat_noise=56.6, edge_sharpness=0.09985, rof=0.009082, shock=0.7543
- Perona-Malik + Shock: flat_noise=56.6, edge_sharpness=0.09985, pm=2.271, shock=0.7543

### x2_sr

- Nota: Super-resolution x2 con blur gaussiano ignoto + noise.
- Comparison: [comparison.png](results\final_inverse_pinn_v1_size256_stabilized\x2_sr\comparison.png)

| Metodo | PSNR | Delta PSNR vs Bicubic | SSIM | Delta SSIM vs Bicubic | Sigma stimato | Noise weights |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Bicubic | 21.47 | +0.00 | 0.6847 | +0.0000 | - | - |
| All PDE trainable | 23.18 | +1.71 | 0.7482 | +0.0634 | 0.7077 | (0.5604, 0.1255, 0.3141) |
| TV energy | 23.17 | +1.71 | 0.7490 | +0.0643 | 0.7078 | (0.5603, 0.1255, 0.3142) |
| ROF PDE | 23.18 | +1.71 | 0.7498 | +0.0651 | 0.7078 | (0.5603, 0.1255, 0.3142) |
| Perona-Malik PDE | 23.18 | +1.72 | 0.7494 | +0.0647 | 0.7079 | (0.5604, 0.1255, 0.3141) |
| Shock PDE | 23.18 | +1.71 | 0.7497 | +0.0650 | 0.7077 | (0.5605, 0.1255, 0.314) |
| ROF PDE + Shock | 23.17 | +1.70 | 0.7465 | +0.0618 | 0.7077 | (0.5605, 0.1255, 0.314) |
| Perona-Malik + Shock | 23.18 | +1.71 | 0.7520 | +0.0673 | 0.7077 | (0.5603, 0.1255, 0.3142) |

Pesi prior appresi:

- All PDE trainable: flat_noise=56.6, edge_sharpness=0.09985, tv=0.02266, rof=0.009077, pm=2.267, shock=0.7537
- TV energy: flat_noise=56.6, edge_sharpness=0.09985, tv=0.02266
- ROF PDE: flat_noise=56.6, edge_sharpness=0.09985, rof=0.01135
- Perona-Malik PDE: flat_noise=56.6, edge_sharpness=0.09985, pm=3.023
- Shock PDE: flat_noise=56.6, edge_sharpness=0.09985, shock=1.131
- ROF PDE + Shock: flat_noise=56.6, edge_sharpness=0.09985, rof=0.009077, shock=0.7537
- Perona-Malik + Shock: flat_noise=56.6, edge_sharpness=0.09985, pm=2.267, shock=0.7537

### x4_sr

- Nota: Super-resolution x4 con blur gaussiano ignoto + noise.
- Comparison: [comparison.png](results\final_inverse_pinn_v1_size256_stabilized\x4_sr\comparison.png)

| Metodo | PSNR | Delta PSNR vs Bicubic | SSIM | Delta SSIM vs Bicubic | Sigma stimato | Noise weights |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Bicubic | 17.97 | +0.00 | 0.5629 | +0.0000 | - | - |
| All PDE trainable | 21.06 | +3.08 | 0.6729 | +0.1100 | 0.7304 | (0.5786, 0.1404, 0.281) |
| TV energy | 21.07 | +3.09 | 0.6659 | +0.1030 | 0.7186 | (0.5657, 0.131, 0.3033) |
| ROF PDE | 21.05 | +3.07 | 0.6713 | +0.1084 | 0.7244 | (0.5734, 0.1355, 0.2911) |
| Perona-Malik PDE | 21.06 | +3.09 | 0.6725 | +0.1095 | 0.7306 | (0.578, 0.1405, 0.2815) |
| Shock PDE | 21.05 | +3.07 | 0.6661 | +0.1032 | 0.7186 | (0.5638, 0.1312, 0.305) |
| ROF PDE + Shock | 21.05 | +3.08 | 0.6712 | +0.1083 | 0.7242 | (0.5717, 0.1356, 0.2927) |
| Perona-Malik + Shock | 21.03 | +3.06 | 0.6628 | +0.0999 | 0.7186 | (0.5643, 0.1311, 0.3046) |

Pesi prior appresi:

- All PDE trainable: flat_noise=57.85, edge_sharpness=0.09746, tv=0.02317, rof=0.009262, pm=2.31, shock=0.7705
- TV energy: flat_noise=56.46, edge_sharpness=0.09994, tv=0.02263
- ROF PDE: flat_noise=57.15, edge_sharpness=0.09869, rof=0.01144
- Perona-Malik PDE: flat_noise=57.85, edge_sharpness=0.09746, pm=3.08
- Shock PDE: flat_noise=56.46, edge_sharpness=0.09994, shock=1.128
- ROF PDE + Shock: flat_noise=57.15, edge_sharpness=0.09869, rof=0.009151, shock=0.7611
- Perona-Malik + Shock: flat_noise=56.46, edge_sharpness=0.09994, pm=2.253, shock=0.752
