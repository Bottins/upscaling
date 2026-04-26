"""Carica un prior SR addestrato e fornisce la sua predizione HR."""
from __future__ import annotations
from pathlib import Path
import torch

from learned_prior.model import SmallEDSR


def load_prior(ckpt_path: str, device) -> tuple[SmallEDSR, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    net = SmallEDSR(scale=ckpt["scale"], n_blocks=ckpt["n_blocks"],
                    ch=ckpt["ch"]).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, ckpt


@torch.no_grad()
def predict_hr(net: SmallEDSR, lr: torch.Tensor) -> torch.Tensor:
    """lr: (3, h, w) -> hr: (3, h*s, w*s) in [0,1]."""
    y = net(lr.unsqueeze(0)).squeeze(0).clamp(0, 1)
    return y
