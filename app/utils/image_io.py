import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from io import BytesIO
import imagehash

def read_bgr_bytes(b: bytes):
    """Lee bytes de imagen -> BGR (corrigiendo rotación EXIF)."""
    if not b:
        raise ValueError("Archivo vacío")
    bio = BytesIO(b)
    im = Image.open(bio)
    im = ImageOps.exif_transpose(im)  # respeta Orientation
    im = im.convert("RGB")
    arr = np.array(im)[:, :, ::-1]  # RGB -> BGR
    return arr

def read_bgr(path: str | Path):
    """Lee imagen desde ruta -> BGR (corrigiendo rotación EXIF)."""
    p = str(path)
    try:
        im = Image.open(p)
        im = ImageOps.exif_transpose(im)  # respeta Orientation
        im = im.convert("RGB")
        arr = np.array(im)[:, :, ::-1]  # RGB -> BGR
        return arr
    except Exception:
        # Fallback a OpenCV si PIL falla
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"No se pudo leer {path}")
        return img

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def phash_distance(pth_img: str | Path, pth_master: str | Path) -> int:
    h1 = imagehash.phash(Image.open(pth_img).convert("RGB"))
    h2 = imagehash.phash(Image.open(pth_master).convert("RGB"))
    return h1 - h2
