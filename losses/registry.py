"""Registry estensibile per i termini di loss. Scelta pulita da config."""
from __future__ import annotations
from typing import Callable, Dict, List

import torch


LossFn = Callable[..., torch.Tensor]
LOSS_REGISTRY: Dict[str, LossFn] = {}


def register(name: str):
    def deco(fn: LossFn) -> LossFn:
        if name in LOSS_REGISTRY:
            raise ValueError(f"Loss '{name}' gia' registrata")
        LOSS_REGISTRY[name] = fn
        return fn
    return deco


def build_losses(names: List[str]) -> Dict[str, LossFn]:
    missing = [n for n in names if n not in LOSS_REGISTRY]
    if missing:
        raise ValueError(f"Loss sconosciute: {missing}. Disponibili: {list(LOSS_REGISTRY)}")
    return {n: LOSS_REGISTRY[n] for n in names}
