from .siren import SIREN
from .fourier_mlp import FourierMLP


def build_model(cfg):
    if cfg.kind == "siren":
        return SIREN(in_dim=2, out_dim=3, hidden=cfg.hidden_dim,
                     num_layers=cfg.num_layers, w0=cfg.siren_w0)
    if cfg.kind == "fourier_mlp":
        return FourierMLP(in_dim=2, out_dim=3, hidden=cfg.hidden_dim,
                          num_layers=cfg.num_layers,
                          mapping_size=cfg.fourier_mapping_size,
                          scale=cfg.fourier_scale)
    raise ValueError(f"Modello sconosciuto: {cfg.kind}")
