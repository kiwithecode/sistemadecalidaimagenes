from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Tuple
from pathlib import Path
import uuid, json, sys
import cv2
import numpy as np

from app.core.models import ImageResult, BatchResponse, Metric
from app.core.config import sku_cfg, load_cfg
from app.utils.image_io import read_bgr_bytes, read_bgr, ensure_dir, phash_distance
from app.utils.align import align_to_master
from app.utils.ssim import ssim_metrics, make_ssim_overlay
from app.utils.color import deltaE_stats_with_lab
from app.utils.sharp import variance_of_laplacian
from app.utils.ocr import ocr_text_conf
from app.utils.barcode import decode_codes, match_expected_codes
from app.utils.textdiff import missing_tokens
from app.utils.llm import explain_via_llm
from app.utils.explain import human_explanation
from app.utils.regions import boxes_from_diffmap, boxes_union, draw_boxes
from app.utils.pdf import draw_pdf
from app.utils.defect_annotator import DefectAnnotator, create_defect_explanation
from app.utils.ml_classifier import QCClassifier, load_training_data_from_reports
from app.utils.folder_trainer import load_training_from_folders
from app.web.router import router as web_router, mount_static

# Calibración
from app.utils.calibrate import (
    align_many_to_ref, stack_median, derive_thresholds, write_master, update_config
)

app = FastAPI(title="QC Wraps MVP")
app.include_router(web_router, prefix="/web")

# === RUTAS BASE DEL PROYECTO (asegúrate de ejecutar uvicorn desde la raíz del repo) ===
ROOT = Path(__file__).resolve().parents[1]   # .../qc-wraps
MASTERS = ROOT / "masters"                   # .../qc-wraps/masters
REPORTS = ROOT / "reports"
TEMP = ROOT / "temp"
ensure_dir(REPORTS)
ensure_dir(TEMP)

# servir assets temporales y estáticos web
app.mount("/temp", StaticFiles(directory=str(TEMP)), name="temp")
mount_static(app)

print(f"[QC] ROOT={ROOT}", file=sys.stderr)
print(f"[QC] MASTERS={MASTERS}", file=sys.stderr)

MASTER_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]

# Inicializar clasificador ML
ML_MODEL_PATH = ROOT / "models" / "qc_classifier.joblib"
ensure_dir(ML_MODEL_PATH.parent)
ml_classifier = QCClassifier(model_path=str(ML_MODEL_PATH))
if ML_MODEL_PATH.exists():
    print(f"[QC] Modelo ML cargado desde {ML_MODEL_PATH}")
else:
    print(f"[QC] No se encontró modelo ML en {ML_MODEL_PATH}")

# Inicializar anotador de defectos
defect_annotator = DefectAnnotator()

def find_sku_dir_case_insensitive(sku: str) -> Optional[Path]:
    """Devuelve la carpeta del SKU ignorando mayúsculas/minúsculas."""
    if not MASTERS.exists():
        return None
    for d in MASTERS.iterdir():
        if d.is_dir() and d.name.lower() == sku.lower():
            return d
    return None


def find_master_path(sku: str) -> Optional[Path]:
    """Busca master.* dentro del directorio del SKU (case-insensitive)."""
    sku_dir = find_sku_dir_case_insensitive(sku)
    if sku_dir is None:
        print(f"[QC] SKU '{sku}' no tiene carpeta en {MASTERS}", file=sys.stderr)
        return None
    for ext in MASTER_EXTS:
        candidate = sku_dir / f"master{ext}"
        if candidate.exists():
            print(f"[QC] Usando master: {candidate}", file=sys.stderr)
            return candidate
    print(f"[QC] No se encontró master.* en {sku_dir}", file=sys.stderr)
    return None


def pick_sku_auto(tmp_img_path: Path) -> Tuple[Optional[str], float]:
    """Autodetecta SKU comparando pHash contra cualquier master.* (case-insensitive)."""
    best, best_d = None, 1e9
    sku_dirs = [d for d in MASTERS.iterdir() if d.is_dir()] if MASTERS.exists() else []
    for sku_dir in sku_dirs:
        master_candidate = None
        for ext in MASTER_EXTS:
            cand = sku_dir / f"master{ext}"
            if cand.exists():
                master_candidate = cand
                break
        if master_candidate is None:
            continue
        try:
            d = phash_distance(tmp_img_path, master_candidate)
        except Exception:
            continue
        if d < best_d:
            best, best_d = sku_dir.name, d
    if best:
        print(f"[QC] Autodetect SKU -> {best} (pHash={best_d})", file=sys.stderr)
    return best, best_d


@app.post("/batch/predict", response_model=BatchResponse)
async def batch_predict(
    files: List[UploadFile] = File(...),
    sku: Optional[str] = Query(default=None, description="SKU o 'auto' para autodetección"),
    ssim_box_thresh: float = Query(0.35, ge=0.1, le=0.9),
    ssim_min_area: int = Query(400, ge=50, le=50000)
):
    try:
        # ---- Lote y carpetas
        lote = str(uuid.uuid4())
        lotedir = TEMP / lote
        ensure_dir(lotedir)

        # ---- Validar y guardar archivos subidos
        tmp_paths = []
        if not files:
            return JSONResponse(status_code=422, content={"error": "Debes enviar al menos un campo 'files'."})

        for f in files:
            # Validación basada en decodificación real (no dependemos de content_type)
            b = await f.read()
            if not b:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"El archivo '{f.filename}' llegó vacío (0 bytes). Si usas OneDrive, copia el archivo a una ruta local (no stub)."},
                )
            try:
                img = read_bgr_bytes(b)
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"No se pudo leer '{f.filename}' como imagen: {e}"},
                )
            # Guardar original en temp del lote
            img_out = lotedir / f.filename
            cv2.imwrite(str(img_out), img)
            tmp_paths.append(img_out)

        # ---- Determinar SKU
        if sku is None or sku.lower() == "auto":
            sku, dist = pick_sku_auto(tmp_paths[0])
            if sku is None:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No se pudo autodetectar SKU (no hay masters/<SKU>/master.*)"},
                )

        # ---- Cargar master (extensión flexible)
        master_path = find_master_path(sku)
        if master_path is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"No existe master para SKU {sku}. Coloca uno de: "
                             + ", ".join([f"master{e}" for e in MASTER_EXTS])
                },
            )
        try:
            master = read_bgr(master_path)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Error leyendo master: {e}"})

        # ---- Config
        defaults, _ = load_cfg()
        cfg = sku_cfg(sku)
        cfg_llm = defaults.get("llm", {})
        use_llm = bool(cfg_llm.get("enabled", False))

        results: List[ImageResult] = []
        pdf_items = []

        # ---- Procesar cada imagen
        for i, p in enumerate(tmp_paths):
            try:
                print(f"[DEBUG] Procesando imagen {i+1}/{len(tmp_paths)}: {p.name}", file=sys.stderr)
                src = read_bgr(p)
                filename = p.name
                img_out = p

                # Alineación
                aligned, H = align_to_master(src, master)
                if aligned is None:
                    explanation_text = "No se pudo alinear con la plantilla; posible desajuste de registro o rotación."
                    results.append(ImageResult(
                        file=p.name, sku=sku, status="mala", defects=["registro"], metrics=[],
                        lab={"lab_avg_img":[0,0,0],"lab_avg_master":[0,0,0],"lab_diff":[0,0,0]},
                        text_missing=[], codes_missing=[], codes_found=[], explanation=explanation_text, boxes=[]
                    ))
                    pdf_items.append({
                        "file": p.name, "status": "mala", "defects": ["registro"],
                        "metrics": [{"name":"alineacion","value":0.0,"threshold":1,"passed":False}],
                        "img_path": str(p), "aligned_path": str(p), "overlay_path": str(p), "annotated_path": str(p),
                        "explanation": explanation_text,
                        "lab": {"lab_avg_img":[0,0,0],"lab_avg_master":[0,0,0],"lab_diff":[0,0,0]},
                        "text_missing": [], "codes_missing": [], "codes_found": []
                    })
                    continue

                # SSIM y recuadros
                ssim_global, ssim_diff = ssim_metrics(aligned, master)
                ssim_pass = ssim_global >= float(cfg["ssim_global_min"])
                ssim_box_thresh = float(cfg.get("ssim_box_thresh", 0.5))
                ssim_min_area = int(cfg.get("ssim_min_area", 500))
                boxes_raw = boxes_from_diffmap(ssim_diff, thresh=ssim_box_thresh, min_area=ssim_min_area)
                boxes = boxes_union(boxes_raw, iou_threshold=0.25)
                boxed = draw_boxes(aligned, boxes, color=(0,0,255), thick=2)

                # Color + LAB
                color_stats = deltaE_stats_with_lab(aligned, master)
                dE_avg = color_stats["dE_avg"]; dE_max = color_stats["dE_max"]
                lab_avg_img = color_stats["lab_avg_img"]; lab_avg_master = color_stats["lab_avg_master"]; lab_diff = color_stats["lab_diff"]
                
                # Asegurar que todos los valores LAB son arrays numpy
                if isinstance(lab_diff, tuple):
                    lab_diff = np.array(lab_diff)
                if isinstance(lab_avg_img, tuple):
                    lab_avg_img = np.array(lab_avg_img)
                if isinstance(lab_avg_master, tuple):
                    lab_avg_master = np.array(lab_avg_master)
                
                # Validación adicional para luminosidad (detectar galletas oscuras/quemadas)
                delta_L = float(abs(lab_diff[0]))  # Diferencia absoluta en luminosidad
                luminosity_threshold = float(cfg.get("deltaL_max", 8.0))  # Umbral para ΔL
                luminosity_pass = delta_L <= luminosity_threshold
                
                color_pass = (dE_avg <= float(cfg["deltaE_avg_max"])) and (dE_max <= float(cfg["deltaE_max_max"])) and luminosity_pass

                # Nitidez
                sharp = variance_of_laplacian(aligned)
                sharp_pass = sharp >= float(cfg["sharpness_min"])

                # OCR
                ocr_text, ocr_conf = ocr_text_conf(aligned)
                text_misses = missing_tokens(ocr_text, cfg.get("expected_text", []))
                ocr_ok = True
                if bool(cfg.get("use_ocr", False)):
                    # Muy permisivo: confianza baja O al menos 50% de textos encontrados
                    min_conf = float(cfg.get("ocr_min_conf", 0.60))  # Reducido a 0.60
                    expected_texts = cfg.get("expected_text", [])
                    text_found_ratio = 1.0 if not expected_texts else (len(expected_texts) - len(text_misses)) / len(expected_texts)
                    ocr_ok = ocr_conf >= min_conf or text_found_ratio >= 0.5

                # Códigos
                codes_found = decode_codes(aligned) if bool(cfg.get("use_codes", False)) else []
                codes_ok, codes_miss = (True, [])
                if bool(cfg.get("use_codes", False)):
                    codes_ok, codes_miss = match_expected_codes(codes_found, cfg.get("expected_codes", []))
                    # Más permisivo: OK si encuentra al menos 1 código esperado
                    expected_codes = cfg.get("expected_codes", [])
                    if expected_codes and codes_found:
                        codes_ok = len(codes_miss) < len(expected_codes)  # Al menos 1 código encontrado

                # Defectos y status
                defects = []
                if not ssim_pass: defects.append("estructura/defecto")
                if not color_pass: 
                    if not luminosity_pass:
                        defects.append("luminosidad")  # Específico para galletas oscuras/quemadas
                    else:
                        defects.append("color")
                if not sharp_pass: defects.append("borroso")
                if bool(cfg.get("use_ocr", False)) and not ocr_ok: defects.append("texto")
                if bool(cfg.get("use_codes", False)) and not codes_ok: defects.append("codigo")

                # Métricas para el reporte
                metrics = {
                    "ssim_global": ssim_global, "deltaE_avg": dE_avg, "deltaE_max": dE_max,
                    "deltaL": delta_L, "sharpness": sharp, "ocr_confidence": ocr_conf,
                    "text_found": len(cfg.get("expected_text", [])) - len(text_misses), 
                    "text_expected": len(cfg.get("expected_text", [])),
                    "codes_found": len(codes_found), "codes_expected": len(cfg.get("expected_codes", []))
                }

                # Para LLM (incluye LAB)
                metrics_for_llm = {
                    "ssim_global": (ssim_global, float(cfg["ssim_global_min"])),
                    "deltaE_avg": (dE_avg, float(cfg["deltaE_avg_max"])),
                    "deltaE_max": (dE_max, float(cfg["deltaE_max_max"])),
                    "sharpness": (sharp, float(cfg["sharpness_min"])),
                    "lab_diff": (lab_diff, None),
                    "deltaL": (delta_L, float(cfg.get("deltaL_max", 8.0)))  # Add deltaL metric
                }
                if bool(cfg.get("use_ocr", False)):
                    metrics_for_llm["ocr_conf"] = (ocr_conf, float(cfg.get("ocr_min_conf", 0.85)))

                # Definir extras antes de usarlo
                extras = {
                    "bad_cells": [],
                    "text_missing": text_misses,
                    "codes_missing": codes_miss,
                    "codes_found": codes_found
                }

                # Clasificación usando ML si está disponible, sino lógica tradicional
                if ml_classifier.is_trained:
                    try:
                        ml_result = ml_classifier.predict(metrics_for_llm, extras)
                        status = ml_result['prediction']
                        ml_confidence = ml_result['confidence']
                        print(f"[ML] Predicción: {status} (confianza: {ml_confidence:.3f})", file=sys.stderr)
                    except Exception as e:
                        print(f"[ML] Error en predicción, usando lógica tradicional: {e}", file=sys.stderr)
                        # Fallback a lógica tradicional
                        critical_pass = ssim_pass and sharp_pass
                        secondary_failures = 0
                        if not color_pass: secondary_failures += 1
                        if bool(cfg.get("use_ocr", False)) and not ocr_ok: secondary_failures += 1
                        if bool(cfg.get("use_codes", False)) and not codes_ok: secondary_failures += 1
                        status = "buena" if (critical_pass and secondary_failures <= 1) else "mala"
                else:
                    # Lógica tradicional cuando no hay modelo ML
                    critical_pass = ssim_pass and sharp_pass
                    secondary_failures = 0
                    if not color_pass: secondary_failures += 1
                    if bool(cfg.get("use_ocr", False)) and not ocr_ok: secondary_failures += 1
                    if bool(cfg.get("use_codes", False)) and not codes_ok: secondary_failures += 1
                    status = "buena" if (critical_pass and secondary_failures == 0) else "mala"

                # Explicación
                if status == "mala":
                    try:
                        explanation_text = explain_via_llm(cfg_llm, defects, metrics_for_llm, extras) if use_llm else None
                    except Exception:
                        explanation_text = None
                else:
                    explanation_text = None

                # Convertir métricas a formato Pydantic
                metrics_list = [
                    Metric(name="ssim_global", value=ssim_global, threshold=float(cfg["ssim_global_min"]), passed=ssim_pass),
                    Metric(name="deltaE_avg", value=dE_avg, threshold=float(cfg["deltaE_avg_max"]), passed=color_pass),
                    Metric(name="deltaE_max", value=dE_max, threshold=float(cfg["deltaE_max_max"]), passed=color_pass),
                    Metric(name="deltaL", value=delta_L, threshold=None, passed=luminosity_pass),
                    Metric(name="sharpness", value=sharp, threshold=float(cfg["sharpness_min"]), passed=sharp_pass),
                    Metric(name="ocr_confidence", value=ocr_conf, threshold=float(cfg.get("ocr_min_conf", 0.85)), passed=ocr_ok)
                ]

                result_item = {
                    "file": p.name, "sku": sku, "status": status, "defects": defects,
                    "metrics": metrics_list,
                    "lab": {
                        "lab_avg_img": lab_avg_img.tolist(),
                        "lab_avg_master": lab_avg_master.tolist(),
                        "lab_diff": lab_diff.tolist()
                    },
                    "text_missing": text_misses, "codes_missing": codes_miss, "codes_found": codes_found,
                    "explanation": explanation_text or "",
                    "boxes": boxes
                }
                results.append(ImageResult(**result_item))

                # Guardar imagen alineada
                aligned_path = lotedir / f"aligned_{filename}.png"
                cv2.imwrite(str(aligned_path), aligned)

                # Generar overlay SSIM
                overlay_path = lotedir / f"overlay_{filename}.png"
                overlay_img = make_ssim_overlay(ssim_diff)
                cv2.imwrite(str(overlay_path), overlay_img)

                # Generar imagen anotada con defectos
                annotated_path = defect_annotator.save_annotated_image(
                    src, master, aligned, metrics, result_item, 
                    lotedir / f"defects_{filename}.png"
                )

                # Métricas en formato esperado por el PDF (lista de dicts)
                pdf_metrics = [
                    {"name": "ssim_global", "value": ssim_global, "threshold": float(cfg["ssim_global_min"]), "passed": ssim_pass},
                    {"name": "deltaE_avg",  "value": dE_avg,      "threshold": float(cfg["deltaE_avg_max"]),  "passed": color_pass},
                    {"name": "deltaE_max",  "value": dE_max,      "threshold": float(cfg["deltaE_max_max"]),  "passed": color_pass},
                    {"name": "deltaL",      "value": delta_L,     "threshold": float(cfg.get("deltaL_max", 8.0)), "passed": luminosity_pass},
                    {"name": "sharpness",   "value": sharp,       "threshold": float(cfg["sharpness_min"]),  "passed": sharp_pass},
                    {"name": "ocr_conf",    "value": ocr_conf,    "threshold": float(cfg.get("ocr_min_conf", 0.85)), "passed": ocr_ok},
                ]

                pdf_items.append({
                    "file": p.name, "status": status, "defects": defects,
                    "metrics": pdf_metrics,
                    "img_path": str(img_out),
                    "aligned_path": str(aligned_path),
                    "overlay_path": str(overlay_path),
                    "annotated_path": str(annotated_path),
                    "explanation": explanation_text or "",
                    "lab": {
                        "lab_avg_img": lab_avg_img.tolist(),
                        "lab_avg_master": lab_avg_master.tolist(),
                        "lab_diff": lab_diff.tolist()
                    },
                    "text_missing": text_misses, "codes_missing": [], "codes_found": codes_found,
                    "boxes": boxes
                })

            except Exception as img_error:
                import traceback
                img_traceback = traceback.format_exc()
                print(f"[ERROR] Error procesando imagen {i+1} ({p.name}):\n{img_traceback}", file=sys.stderr)
                # Agregar resultado de error para esta imagen
                results.append(ImageResult(
                    file=p.name, sku=sku, status="mala", defects=["error_procesamiento"], metrics=[],
                    lab={"lab_avg_img":[0,0,0],"lab_avg_master":[0,0,0],"lab_diff":[0,0,0]},
                    text_missing=[], codes_missing=[], codes_found=[], explanation=f"Error: {str(img_error)}", boxes=[]
                ))
                pdf_items.append({
                    "file": p.name, "status": "mala", "defects": ["error_procesamiento"],
                    "metrics": [],
                    "img_path": str(p), "aligned_path": str(p), "overlay_path": str(p), "annotated_path": str(p),
                    "explanation": f"Error: {str(img_error)}",
                    "lab": {"lab_avg_img":[0,0,0],"lab_avg_master":[0,0,0],"lab_diff":[0,0,0]},
                    "text_missing": [], "codes_missing": [], "codes_found": [], "boxes": []
                })

        # ---- PDF
        pdf_path = REPORTS / f"{lote}.pdf"
        draw_pdf(str(pdf_path), sku, lote, pdf_items)

        # ---- JSON del lote (para dashboard)
        lote_json = {
            "lote": lote, "sku": sku,
            "resultados": [r.model_dump() for r in results],
            "pdf": f"/download/{lote}"
        }
        (REPORTS / f"{lote}.json").write_text(json.dumps(lote_json, ensure_ascii=False, indent=2), encoding="utf-8")

        return BatchResponse(lote=lote, sku=sku, resultados=results, pdf=f"/download/{lote}")

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] Traceback completo:\n{error_details}", file=sys.stderr)
        return JSONResponse(
            status_code=500,
            content={"error": f"Error procesando lote: {str(e)}", "traceback": error_details}
        )


@app.post("/sku/calibrate")
async def sku_calibrate(
    files: List[UploadFile] = File(...),
    sku: str = Query(..., description="SKU a calibrar (usar mismo nombre de carpeta)")
):
    """
    Sube 5-20 imágenes 'buenas' del SKU. El sistema:
      - Alínea todas a la primera,
      - Construye master por mediana,
      - Deriva umbrales por percentiles,
      - Guarda masters/<SKU>/master.png,
      - Actualiza config.yaml con umbrales del SKU.
    """
    if not files or len(files) < 5:
        return JSONResponse(status_code=400, content={"error":"Sube al menos 5 imágenes 'buenas' para calibrar."})

    # Leer imágenes (autorotate EXIF lo hace read_bgr_bytes)
    imgs = []
    for f in files:
        b = await f.read()
        if not b:
            return JSONResponse(status_code=400, content={"error": f"Archivo vacío: {f.filename}"})
        try:
            img = read_bgr_bytes(b)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"No se pudo leer '{f.filename}' como imagen: {e}"})
        imgs.append(img)

    # Alinear todas a la primera
    ref = imgs[0]
    aligned_all = align_many_to_ref(imgs, ref)

    # Master por mediana
    master_img = stack_median(aligned_all)

    # Derivar umbrales desde tus "buenas"
    th = derive_thresholds(aligned_all, master_img)

    # Guardar master en masters/<SKU>/master.png
    sku_dir = MASTERS / sku
    ensure_dir(sku_dir)
    master_path = sku_dir / "master.png"
    write_master(master_path, master_img)

    # Actualizar config.yaml (en raíz del repo)
    CFG_PATH = ROOT / "config.yaml"
    update_config(CFG_PATH, sku, th)

    return {
        "sku": sku,
        "master_saved": str(master_path),
        "thresholds": {
            "ssim_global_min": round(th["ssim_global_min"], 3),
            "deltaE_avg_max": round(th["deltaE_avg_max"], 2),
            "deltaE_max_max": round(th["deltaE_max_max"], 2),
            "sharpness_min": round(th["sharpness_min"], 1),
            "n_samples": th["n_samples"]
        }
    }


@app.post("/ml/train")
async def train_ml_model():
    """Entrena el modelo ML usando todos los reportes JSON existentes."""
    try:
        # Cargar datos de entrenamiento desde reportes
        training_samples = load_training_data_from_reports(REPORTS)
        
        if len(training_samples) < 10:
            return JSONResponse(
                status_code=400,
                content={"error": f"Necesitas al menos 10 muestras. Encontradas: {len(training_samples)}"}
            )
        
        # Añadir muestras al clasificador
        for sample in training_samples:
            ml_classifier.add_training_sample(
                sample['metrics'], 
                sample['extras'], 
                sample['label']
            )
        
        # Entrenar modelo
        accuracy = ml_classifier.train()
        
        # Guardar modelo
        ml_classifier.save_model()
        
        # Estadísticas
        buenas = sum(1 for s in training_samples if s['label'] == 'buena')
        malas = len(training_samples) - buenas
        
        return {
            "message": "Modelo ML entrenado exitosamente",
            "accuracy": accuracy,
            "samples": {
                "total": len(training_samples),
                "buenas": buenas,
                "malas": malas
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error entrenando modelo: {str(e)}"}
        )


@app.post("/ml/label")
async def label_batch_results(
    lote_id: str = Query(..., description="ID del lote a etiquetar"),
    labels: dict = Body(..., description="Diccionario {filename: 'buena'/'mala'}")
):
    """Etiqueta manualmente los resultados de un lote para entrenamiento."""
    try:
        # Buscar el archivo JSON del lote
        lote_file = REPORTS / f"{lote_id}.json"
        if not lote_file.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"Lote {lote_id} no encontrado"}
            )
        
        # Cargar y actualizar etiquetas
        with open(lote_file, 'r', encoding='utf-8') as f:
            lote_data = json.load(f)
        
        updated = 0
        for resultado in lote_data.get('resultados', []):
            filename = resultado.get('file', '')
            if filename in labels:
                resultado['status'] = labels[filename]
                resultado['manually_labeled'] = True
                updated += 1
        
        # Guardar archivo actualizado
        with open(lote_file, 'w', encoding='utf-8') as f:
            json.dump(lote_data, f, ensure_ascii=False, indent=2)
        
        return {
            "message": f"Etiquetas actualizadas para {updated} imágenes",
            "lote": lote_id,
            "updated": updated
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error etiquetando lote: {str(e)}"}
        )


@app.get("/ml/status")
async def ml_model_status():
    """Obtiene el estado del modelo ML."""
    try:
        # Contar muestras de entrenamiento disponibles
        training_samples = load_training_data_from_reports(REPORTS)
        buenas = sum(1 for s in training_samples if s['label'] == 'buena')
        malas = len(training_samples) - buenas
        
        return {
            "model_trained": ml_classifier.is_trained,
            "model_path": str(ML_MODEL_PATH),
            "model_exists": ML_MODEL_PATH.exists(),
            "training_data": {
                "total_samples": len(training_samples),
                "buenas": buenas,
                "malas": malas,
                "ready_to_train": len(training_samples) >= 10
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error obteniendo estado: {str(e)}"}
        )


@app.post("/ml/train-from-folders")
async def train_from_folders(
    sku: str = Query(..., description="SKU del producto"),
    folder_path: str = Query(..., description="Ruta base con carpetas 'buenas' y 'malas'")
):
    """
    Entrena el modelo ML usando carpetas organizadas con imágenes buenas y malas.
    Estructura esperada:
    folder_path/
    ├── buenas/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── malas/
        ├── img3.jpg
        └── img4.jpg
    """
    try:
        from app.core.config import sku_cfg
        
        # Convertir ruta de Windows a WSL si es necesario
        if folder_path.startswith("C:/") or folder_path.startswith("C:\\"):
            folder_path = folder_path.replace("C:/", "/mnt/c/").replace("C:\\", "/mnt/c/").replace("\\", "/")
        
        # Validar carpeta base
        base_path = Path(folder_path)
        if not base_path.exists():
            return JSONResponse(
                status_code=400,
                content={"error": f"Carpeta no existe: {folder_path}"}
            )
        
        # Buscar master del SKU
        master_path = find_master_path(sku)
        if master_path is None:
            return JSONResponse(
                status_code=400,
                content={"error": f"No se encontró master para SKU {sku}"}
            )
        
        # Obtener configuración del SKU
        cfg = sku_cfg(sku)
        
        # Cargar datos de entrenamiento desde carpetas
        training_samples = load_training_from_folders(base_path, sku, master_path, cfg)
        
        if len(training_samples) < 10:
            return JSONResponse(
                status_code=400,
                content={"error": f"Necesitas al menos 10 muestras. Encontradas: {len(training_samples)}"}
            )
        
        # Añadir muestras al clasificador
        for sample in training_samples:
            ml_classifier.add_training_sample(
                sample['metrics'], 
                sample['extras'], 
                sample['label']
            )
        
        # Entrenar modelo
        accuracy = ml_classifier.train()
        
        # Guardar modelo
        ml_classifier.save_model()
        
        # Estadísticas
        buenas = sum(1 for s in training_samples if s['label'] == 'buena')
        malas = len(training_samples) - buenas
        
        return {
            "message": "Modelo ML entrenado desde carpetas",
            "accuracy": accuracy,
            "samples": {
                "total": len(training_samples),
                "buenas": buenas,
                "malas": malas
            },
            "sku": sku,
            "folder_path": folder_path
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error entrenando desde carpetas: {str(e)}"}
        )


@app.get("/download/{report_id}")
def download(report_id: str):
    path = REPORTS / f"{report_id}.pdf"
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "No existe ese reporte"})
    return FileResponse(str(path), media_type="application/pdf", filename=f"{report_id}.pdf")
