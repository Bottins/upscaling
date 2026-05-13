from __future__ import annotations

import torch
import torch.nn.functional as F


def bicubic(lr: torch.Tensor, hr_hw: tuple[int, int]) -> torch.Tensor:
    squeeze = lr.dim() == 3
    batch = lr.unsqueeze(0) if squeeze else lr
    out = F.interpolate(batch, size=hr_hw, mode="bicubic", align_corners=False)
    out = out.clamp(0, 1)
    return out.squeeze(0) if squeeze else out
