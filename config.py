"""Configurazione centrale del progetto."""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    root: str = "./datasets"
    name: str = "Set5"                  # "Set5" | "DIV2K"
    image_name: str = "butterfly.png"   # usata in single-image mode
    hr_size: Tuple[int, int] = (256, 256)
    scale: int = 4                      # fattore di downsampling
    blur_sigma: float = 1.0
    noise_std: float = 0.0


@dataclass
class ModelConfig:
    kind: str = "siren"                 # "siren" | "fourier_mlp"
    hidden_dim: int = 256
    num_layers: int = 5
    siren_w0: float = 30.0
    fourier_mapping_size: int = 128
    fourier_scale: float = 10.0


@dataclass
class LossConfig:
    # Pesi dei termini attivi; un peso=0 disattiva il termine
    terms: List[str] = field(default_factory=lambda: ["data_lr", "pde_perona_malik", "bc_neumann"])
    weights: dict = field(default_factory=lambda: {
        "data_lr":          1.0,
        "data_points":      1.0,
        # priori gentili: devono modellare il sottospazio libero (kernel di P),
        # non sovrastare il fit dati.
        "pde_perona_malik": 2e-2,
        "pde_anisotropic":  2e-2,
        "bc_neumann":       1e-3,
        "reg_tv":           2e-3,
        "pde_shock":        5e-3,
    })
    # Perona-Malik
    pm_kappa: float = 0.05
    # tensore di struttura
    struct_sigma: float = 1.0
    eig_clip: Tuple[float, float] = (1e-3, 1.0)
    struct_eps: float = 1e-4      # regolarizzazione di G prima di eigh
    # curriculum: il peso PDE sale linearmente da 0 a 1 fra [warmup, warmup+ramp]
    pde_warmup_epochs: int = 50
    pde_ramp_epochs:   int = 200


@dataclass
class TrainConfig:
    epochs: int = 2000
    lr: float = 1e-4
    device: str = "cuda"
    n_collocation: int = 4096           # punti per residuo PDE
    n_data_points: int = 2048           # punti dati riposizionati (single-image)
    init_from_bicubic: bool = True
    log_every: int = 50
    snapshot_every: int = 25     # salva confronto visivo ogni N epoche
    ckpt_dir: str = "./checkPoints"
    snap_dir: str = "./checkPoints/snapshots"
    grad_clip: float = 1.0        # norma massima del gradiente (stabilita')


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    single_image: bool = True
