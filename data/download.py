"""Download dei dataset se assenti localmente, oppure fallback su singola immagine."""
from __future__ import annotations
import os
import tarfile
import zipfile
import urllib.request
from pathlib import Path


# Mirror funzionanti (2026). In caso di 404, lo script prova il successivo.
DATASETS = {
    "Set5": {
        "urls": [
            "https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/data/Set5_HR.tar.gz",
        ],
        "subdir": "Set5",
    },
    "DIV2K_valid_HR": {
        "urls": [
            "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
            "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
        ],
        "subdir": "DIV2K_valid_HR",
    },
}

# Singole immagini di test se nessun dataset e' scaricabile.
SAMPLE_IMAGES = [
    ("butterfly.png",
     "https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set5/image_SRF_4/img_003_SRF_4_HR.png"),
    ("baby.png",
     "https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Set5/image_SRF_4/img_001_SRF_4_HR.png"),
]


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {dst}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def _try_download(urls, dst: Path) -> bool:
    for url in urls:
        try:
            _download(url, dst)
            return True
        except Exception as e:
            print(f"[download] fallito {url}: {e}")
            if dst.exists():
                dst.unlink()
    return False


def _extract(archive: Path, out_dir: Path) -> None:
    print(f"[extract] {archive} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir)
    else:  # .tar.gz / .tgz / .tar
        with tarfile.open(archive) as t:
            t.extractall(out_dir)


def _has_images(d: Path) -> bool:
    if not d.exists():
        return False
    exts = {".png", ".bmp", ".jpg", ".jpeg"}
    return any(p.suffix.lower() in exts for p in d.rglob("*"))


def ensure_dataset(name: str, root: str | Path) -> Path:
    """Scarica ed estrae se possibile; altrimenti fallback su singole immagini."""
    if name not in DATASETS:
        raise ValueError(f"Dataset sconosciuto: {name}. Disponibili: {list(DATASETS)}")
    root = Path(root)
    target = root / DATASETS[name]["subdir"]
    if _has_images(target):
        return target

    urls = DATASETS[name]["urls"]
    # scegli estensione archivio dall'URL
    ext = ".tar.gz" if urls[0].endswith((".tar.gz", ".tgz")) else ".zip"
    archive = root / f"{name}{ext}"
    if archive.exists() or _try_download(urls, archive):
        try:
            _extract(archive, target)
            if _has_images(target):
                return target
        except Exception as e:
            print(f"[extract] fallita: {e}")

    # --- fallback: scarica qualche immagine singola -----------------------
    print("[fallback] download di immagini di test singole in", target)
    target.mkdir(parents=True, exist_ok=True)
    ok = False
    for fname, url in SAMPLE_IMAGES:
        dst = target / fname
        if dst.exists() or _try_download([url], dst):
            ok = True
    if not ok:
        raise RuntimeError(
            f"Impossibile scaricare {name}. Salva manualmente un'immagine HR "
            f"in {target} (es. butterfly.png) e rilancia."
        )
    return target
