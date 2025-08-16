import cv2, numpy as np, yaml
from pathlib import Path
from typing import List, Dict, Optional
from app.utils.align import align_to_master
from app.utils.ssim import ssim_metrics
from app.utils.color import deltaE_stats_with_lab
from app.utils.sharp import variance_of_laplacian

def pick_best_ref(imgs: List[np.ndarray]) -> np.ndarray:
    """Escoge como referencia la imagen más nítida (evita una ref borrosa/oblicua)."""
    best = None; best_sharp = -1
    for im in imgs:
        s = variance_of_laplacian(im)
        if s > best_sharp:
            best, best_sharp = im, s
    return best

def align_many_to_ref(imgs: List[np.ndarray], ref: np.ndarray) -> List[np.ndarray]:
    """Alinea a ref; DESCARTA las que no se puedan alinear (no reescalar 'a ojo')."""
    out = []
    for im in imgs:
        aligned, _ = align_to_master(im, ref)
        if aligned is not None:
            out.append(aligned)
    return out

def stack_median(images_bgr: List[np.ndarray]) -> np.ndarray:
    arr = np.stack(images_bgr, axis=0).astype(np.uint8)
    return np.median(arr, axis=0).astype(np.uint8)

def _percentile(vals: List[float], p: float) -> float:
    v = sorted(vals); k = (len(v)-1) * (p/100.0)
    f, c = int(np.floor(k)), int(np.ceil(k))
    if f == c: return float(v[f])
    return float(v[f]*(c-k) + v[c]*(k-f))

def derive_thresholds(samples_aligned: List[np.ndarray], master: np.ndarray) -> Dict[str, float]:
    ssims, dE_avgs, dE_maxs, sharps = [], [], [], []
    for img in samples_aligned:
        s, _ = ssim_metrics(img, master); ssims.append(s)
        cs = deltaE_stats_with_lab(img, master)
        dE_avgs.append(cs["dE_avg"]); dE_maxs.append(cs["dE_max"])
        sharps.append(variance_of_laplacian(img))
    return {
        "ssim_global_min": _percentile(ssims, 5),
        "deltaE_avg_max":  _percentile(dE_avgs, 95),
        "deltaE_max_max":  _percentile(dE_maxs, 95),
        "sharpness_min":   _percentile(sharps, 5),
        "n_samples":       len(samples_aligned)
    }

def write_master(master_path: Path, master_img: np.ndarray):
    master_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(master_path), master_img)

def update_config(cfg_path: Path, sku: str, th: Dict[str, float]):
    import yaml
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    cfg.setdefault("defaults", {})
    cfg.setdefault("skus", {})
    sku_cfg = cfg["skus"].get(sku, {}) or {}
    sku_cfg.update({
        "ssim_global_min": float(th["ssim_global_min"]),
        "deltaE_avg_max":  float(th["deltaE_avg_max"]),
        "deltaE_max_max":  float(th["deltaE_max_max"]),
        "sharpness_min":   float(th["sharpness_min"]),
    })
    cfg["skus"][sku] = sku_cfg
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
