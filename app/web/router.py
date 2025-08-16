from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
from jinja2 import Environment, FileSystemLoader, select_autoescape

router = APIRouter()
BASE = Path(__file__).resolve().parent

# ROOT = qc-wraps/
ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
TEMP = ROOT / "temp"

TEMPLATES = BASE / "templates"
STATIC = BASE / "static"

env = Environment(
    loader=FileSystemLoader(str(TEMPLATES)),
    autoescape=select_autoescape()
)

def mount_static(app):
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    items = []
    if REPORTS.exists():
        for p in sorted(REPORTS.glob("*.json"), reverse=True):
            try:
                js = json.loads(p.read_text(encoding="utf-8"))
                items.append({
                    "lote": js.get("lote"),
                    "sku": js.get("sku"),
                    "num": len(js.get("resultados", [])),
                    "pdf": js.get("pdf"),
                })
            except Exception:
                pass
    tpl = env.get_template("index.html")
    return tpl.render(items=items)

@router.get("/batch/{lote}", response_class=HTMLResponse)
async def batch_view(request: Request, lote: str):
    jpath = REPORTS / f"{lote}.json"
    if not jpath.exists():
        return HTMLResponse(f"<h1>No existe lote {lote}</h1>", status_code=404)
    data = json.loads(jpath.read_text(encoding="utf-8"))
    tpl = env.get_template("batch.html")

    # thumbs desde qc-wraps/temp/<lote>/
    thumbs = []
    tmp_dir = TEMP / lote
    for item in data.get("resultados", []):
        aligned_name = f"aligned_{item['file']}.png"
        thumbs.append({
            "file": item["file"],
            "status": item["status"],
            "thumb": f"/temp/{lote}/{aligned_name}",
            "view": f"/web/image/{lote}/{item['file']}"
        })
    return tpl.render(data=data, thumbs=thumbs, lote=lote)

@router.get("/image/{lote}/{filename}", response_class=HTMLResponse)
async def image_view(request: Request, lote: str, filename: str):
    jpath = REPORTS / f"{lote}.json"
    if not jpath.exists():
        return HTMLResponse(f"<h1>No existe lote {lote}</h1>", status_code=404)
    data = json.loads(jpath.read_text(encoding="utf-8"))
    item = next((r for r in data.get("resultados", []) if r["file"] == filename), None)
    if not item:
        return HTMLResponse(f"<h1>No existe imagen {filename} en lote {lote}</h1>", status_code=404)

    aligned = f"/temp/{lote}/aligned_{filename}.png"
    overlay = f"/temp/{lote}/overlay_{filename}.png"
    boxes = item.get("boxes", [])
    tpl = env.get_template("image.html")
    return tpl.render(
        lote=lote, filename=filename, status=item["status"],
        aligned=aligned, overlay=overlay, boxes=boxes, pdf=data.get("pdf")
    )
    
@router.get("/upload", response_class=HTMLResponse)
async def upload_view(request: Request):
    tpl = env.get_template("upload.html")
    return tpl.render()
