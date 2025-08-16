import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "config.yaml"

def load_cfg():
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    defaults = data.get("defaults", {}) or {}
    skus = data.get("skus", {}) or {}
    return defaults, skus

def sku_cfg(sku: str):
    defaults, skus = load_cfg()
    out = defaults.copy()
    out.update(skus.get(sku, {}) or {})
    return out
