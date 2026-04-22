"""Metriche standard per SR."""
from __future__ import annotations
import math
import torch


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def save_image(img: torch.Tensor, path: str) -> None:
    from PIL import Image
    import numpy as np
    arr = img.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def save_triptych(gt: torch.Tensor, lr: torch.Tensor, pred: torch.Tensor,
                  path: str, title: str | None = None) -> None:
    """Salva affiancati [HR ground truth | LR (nearest-up) | predizione]."""
    from PIL import Image, ImageDraw
    import numpy as np
    import torch.nn.functional as F

    H, W = gt.shape[-2:]
    lr_up = F.interpolate(lr.unsqueeze(0), size=(H, W),
                          mode="nearest").squeeze(0)

    def _to_np(t):
        return (t.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255
                ).astype(np.uint8)

    gap = 6
    panel = np.ones((H, gap, 3), dtype=np.uint8) * 255
    strip = np.concatenate([_to_np(gt), panel, _to_np(lr_up),
                            panel, _to_np(pred)], axis=1)

    if title:
        strip_h = strip.shape[0] + 24
        canvas = np.ones((strip_h, strip.shape[1], 3), dtype=np.uint8) * 255
        canvas[24:] = strip
        img = Image.fromarray(canvas)
        ImageDraw.Draw(img).text((8, 4), title, fill=(0, 0, 0))
    else:
        img = Image.fromarray(strip)
    img.save(path)
