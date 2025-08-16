# QC Wraps — MVP

Sistema de control de calidad para empaques con FastAPI + OpenCV.

- Multi-SKU, thresholds por SKU desde `config.yaml`
- Alineación + SSIM + Color (LAB/ΔE) + Nitidez + OCR + Códigos de barras
- Clasificación con ML + fallback por umbrales
- PDF con overlays y anotaciones, dashboard web con paginación y borrado

---

## 1) Requisitos

### Windows (nativo)
- Python 3.10+ desde python.org
- Microsoft Visual C++ Redistributable (para paquetes científicos)
- Tesseract OCR para Windows (instalador):
  - https://github.com/UB-Mannheim/tesseract/wiki
  - Tras instalar, agrega a PATH la carpeta `Tesseract-OCR` (donde está `tesseract.exe`).

### WSL2 (Ubuntu) — recomendado para CV
```bash
sudo apt update && sudo apt install -y \
  python3.10 python3.10-venv python3-pip build-essential pkg-config \
  libgl1 libglib2.0-0 tesseract-ocr libtesseract-dev libjpeg-turbo-progs
```

---

## 2) Clonar y entorno

```bash
git clone https://github.com/kiwithecode/sistemadecalidaimagenes.git
cd sistemadecalidaimagenes  # carpeta del repo

# Crear entorno
python -m venv .venv

# Activar entorno
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# WSL/Ubuntu
source .venv/bin/activate

# Instalar dependencias
pip install --upgrade pip wheel
pip install -r requirements.txt
```

---

## 3) Estructura de carpetas

```
qc-wraps/
├─ app/
│  ├─ main.py                 # FastAPI
│  ├─ web/                    # Router, templates, estáticos
│  └─ utils/                  # CV, ML, PDF, etc.
├─ masters/                   # Imágenes maestras por SKU
├─ models/                    # Modelos ML entrenados
├─ pr/                        # Carpeta de trabajo / ejemplos
├─ reports/                   # .json y .pdf generados
├─ temp/                      # Imágenes intermedias por lote
├─ config.yaml                # Configuración global y por SKU
└─ requirements.txt
```

---

## 4) Configuración rápida

Edita `config.yaml` para:
- thresholds por SKU (SSIM, ΔE, nitidez, etc.)
- rutas a `masters/`, `reports/`, `temp/` (por defecto ya apuntan dentro del repo)
- opciones de PDF, OCR, códigos, etc.

Las rutas funcionan en Windows y WSL. Si usas WSL, coloca tus imágenes dentro del repo para evitar permisos en montajes de Windows.

---

## 5) Ejecutar el servidor

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Abre el dashboard:
- http://localhost:8000/web/

Características del dashboard (`app/web/`):
- Listado de lotes recientes (orden por fecha, paginación)
- Ver detalle de lote e imágenes (alineada, overlay, anotada)
- Descargar PDF del lote
- Borrar lote (JSON, PDF y carpeta `temp/`)

---

## 6) Flujos típicos

### A) Procesar un lote desde el dashboard
1. Ir a `/web/` y click en “+ Nuevo lote”.
2. Subir imágenes del SKU correspondiente.
3. El sistema alinea, calcula métricas, clasifica y genera:
   - 3 imágenes por muestra: `aligned_*.png`, `overlay_*.png`, `defects_*.png` en `temp/<lote>/`
   - Reporte `reports/<lote>.json` y `reports/<lote>.pdf`

### B) Entrenar modelo ML con carpetas organizadas
Estructura esperada:
```
carpeta/
├─ buenas/  # imágenes buenas
└─ malas/   # imágenes defectuosas
```
Llama al endpoint (ej. con curl o Postman):
```bash
curl -X POST "http://localhost:8000/ml/train-from-folders" \
  -H "Content-Type: application/json" \
  -d '{"root": "RUTA/ABSOLUTA/A/carpeta", "sku": "MI_SKU"}'
```
El modelo se guarda en `models/` y se usa automáticamente en predicciones.

---

## 7) Endpoints útiles

- `GET /web/` — Dashboard
- `GET /web/batch/{lote}` — Vista de lote
- `GET /web/image/{lote}/{filename}` — Vista de una imagen del lote
- `POST /web/delete/{lote}` — Borrar lote (JSON/PDF/temp)
- `POST /ml/train-from-folders` — Entrenar desde carpetas `buenas/` y `malas/`

Swagger/OpenAPI: `http://localhost:8000/docs`

---

## 8) Resolución de problemas

- __OpenCV no abre ventanas en servidor__: este es un backend web (no usa `cv2.imshow`). Asegúrate de ver imágenes en `/web/` o archivos en `temp/`.
- __Error de Tesseract no encontrado__: en Windows, instala Tesseract y agrega su carpeta al PATH. En WSL instala `tesseract-ocr`.
- __Imágenes giradas en PDF__: el sistema normaliza orientación con EXIF y re-guarda como PNG antes de insertar en PDF. Regenera el lote si venías de una versión anterior.
- __Permisos en WSL__: coloca las imágenes dentro del repo para evitar issues de permisos en montajes `/mnt/c`.

---

## 9) Desarrollo

Reinicia el servidor al cambiar Python:
```bash
uvicorn app.main:app --reload
```

Commits/Push estándar:
```bash
git add . && git commit -m "mensaje" && git push
```

---

## 10) Licencia

Privado / Interno MVP.
