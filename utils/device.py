"""Utility per selezionare il device e stampare diagnostica GPU."""
from __future__ import annotations
import torch


def setup_device(requested: str = "cuda") -> torch.device:
    """Sceglie il device, attiva le ottimizzazioni cuDNN/TF32 e stampa diagnostica.

    Su RTX 3050 (Ampere) abilitiamo:
      - cuDNN benchmark (seleziona i kernel piu' veloci per shape fisse)
      - TF32 in matmul e conv (perdita precisione trascurabile per SR)
    """
    if requested.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(requested)
        name = torch.cuda.get_device_name(dev)
        cap = torch.cuda.get_device_capability(dev)
        total_gb = torch.cuda.get_device_properties(dev).total_memory / 1024**3
        print(f"[device] CUDA OK -> {name} (sm_{cap[0]}{cap[1]}, "
              f"{total_gb:.1f} GB), torch {torch.__version__}, "
              f"CUDA {torch.version.cuda}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    else:
        dev = torch.device("cpu")
        if requested.startswith("cuda"):
            print("[device] CUDA non disponibile, fallback a CPU. "
                  "In WSL: verifica `nvidia-smi` e che torch sia il build cu* "
                  "(pip install torch --index-url https://download.pytorch.org/whl/cu121).")
        else:
            print("[device] uso CPU")
    return dev
