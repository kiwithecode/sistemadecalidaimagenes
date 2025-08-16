import cv2
from pathlib import Path
from typing import List, Tuple
import json
from app.utils.image_io import read_bgr
from app.utils.align import align_to_master
from app.utils.ssim import ssim_metrics
from app.utils.color import deltaE_stats_with_lab
from app.utils.sharp import variance_of_laplacian
from app.utils.ocr import ocr_text_conf
from app.utils.barcode import decode_codes, match_expected_codes
from app.utils.textdiff import missing_tokens

def process_folder_for_training(folder_path: Path, master_path: Path, sku: str, cfg: dict, label: str) -> List[dict]:
    """
    Procesa una carpeta de imágenes (buenas o malas) y extrae métricas para entrenamiento.
    
    Args:
        folder_path: Ruta a la carpeta con imágenes
        master_path: Ruta a la imagen master
        sku: SKU del producto
        cfg: Configuración del SKU
        label: 'buena' o 'mala'
    
    Returns:
        Lista de diccionarios con métricas y etiquetas
    """
    if not folder_path.exists():
        print(f"[FOLDER] Carpeta no existe: {folder_path}")
        return []
    
    # Cargar master
    try:
        master = read_bgr(master_path)
    except Exception as e:
        print(f"[FOLDER] Error cargando master {master_path}: {e}")
        return []
    
    training_samples = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    for img_path in folder_path.iterdir():
        if not img_path.is_file() or img_path.suffix.lower() not in image_extensions:
            continue
            
        try:
            print(f"[FOLDER] Procesando {img_path.name} como '{label}'")
            
            # Cargar y alinear imagen
            src = read_bgr(img_path)
            aligned, H = align_to_master(src, master)
            
            if aligned is None:
                print(f"[FOLDER] No se pudo alinear {img_path.name}, saltando...")
                continue
            
            # Calcular métricas (igual que en batch_predict)
            
            # SSIM
            ssim_global, ssim_diff = ssim_metrics(aligned, master)
            
            # Color + LAB
            color_stats = deltaE_stats_with_lab(aligned, master)
            dE_avg = color_stats["dE_avg"]
            dE_max = color_stats["dE_max"]
            lab_diff = color_stats["lab_diff"]
            
            # Luminosidad
            delta_L = abs(lab_diff[0])
            
            # Nitidez
            sharp = variance_of_laplacian(aligned)
            
            # OCR
            ocr_text, ocr_conf = ocr_text_conf(aligned)
            text_misses = missing_tokens(ocr_text, cfg.get("expected_text", []))
            
            # Códigos
            codes_found = decode_codes(aligned) if bool(cfg.get("use_codes", False)) else []
            codes_ok, codes_miss = (True, [])
            if bool(cfg.get("use_codes", False)):
                codes_ok, codes_miss = match_expected_codes(codes_found, cfg.get("expected_codes", []))
            
            # Preparar métricas para ML
            metrics_dict = {
                "ssim_global": (ssim_global, float(cfg.get("ssim_global_min", 0.75))),
                "deltaE_avg": (dE_avg, float(cfg.get("deltaE_avg_max", 10.0))),
                "deltaE_max": (dE_max, float(cfg.get("deltaE_max_max", 100.0))),
                "sharpness": (sharp, float(cfg.get("sharpness_min", 50.0))),
                "lab_diff": (lab_diff, None),
            }
            
            if bool(cfg.get("use_ocr", False)):
                metrics_dict["ocr_conf"] = (ocr_conf, float(cfg.get("ocr_min_conf", 0.60)))
            
            extras_dict = {
                "text_missing": text_misses,
                "codes_missing": codes_miss,
                "codes_found": codes_found
            }
            
            # Añadir muestra de entrenamiento
            training_samples.append({
                "file": img_path.name,
                "sku": sku,
                "label": label,
                "metrics": metrics_dict,
                "extras": extras_dict,
                "source": "folder_training"
            })
            
        except Exception as e:
            print(f"[FOLDER] Error procesando {img_path.name}: {e}")
            continue
    
    print(f"[FOLDER] Procesadas {len(training_samples)} imágenes de {folder_path}")
    return training_samples

def load_training_from_folders(base_path: Path, sku: str, master_path: Path, cfg: dict) -> List[dict]:
    """
    Carga datos de entrenamiento desde carpetas organizadas como:
    base_path/
    ├── buenas/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── malas/
        ├── img3.jpg
        └── img4.jpg
    """
    all_samples = []
    
    # Procesar carpeta de buenas
    buenas_path = base_path / "buenas"
    if buenas_path.exists():
        buenas_samples = process_folder_for_training(buenas_path, master_path, sku, cfg, "buena")
        all_samples.extend(buenas_samples)
    else:
        print(f"[FOLDER] Carpeta 'buenas' no encontrada en {base_path}")
    
    # Procesar carpeta de malas
    malas_path = base_path / "malas"
    if malas_path.exists():
        malas_samples = process_folder_for_training(malas_path, master_path, sku, cfg, "mala")
        all_samples.extend(malas_samples)
    else:
        print(f"[FOLDER] Carpeta 'malas' no encontrada en {base_path}")
    
    return all_samples
