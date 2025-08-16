from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
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
async def index(request: Request, page: int = Query(1, ge=1), per_page: int = Query(10, ge=1, le=100)):
    items = []
    if REPORTS.exists():
        files = list(REPORTS.glob("*.json"))
        # ordenar por fecha de modificación (desc)
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files:
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

    total = len(items)
    start = (page - 1) * per_page
    end = start + per_page
    page_items = items[start:end]
    total_pages = (total + per_page - 1) // per_page if per_page else 1

    tpl = env.get_template("index.html")
    return tpl.render(items=page_items, page=page, per_page=per_page, total=total, total_pages=total_pages)

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

@router.post("/delete/{lote}")
async def delete_lote(lote: str):
    """Elimina un lote: JSON, PDF y carpeta temp."""
    deleted = {"json": False, "pdf": False, "temp": False}
    jpath = REPORTS / f"{lote}.json"
    ppath = REPORTS / f"{lote}.pdf"
    tdir = TEMP / lote
    try:
        if jpath.exists():
            jpath.unlink()
            deleted["json"] = True
        if ppath.exists():
            ppath.unlink()
            deleted["pdf"] = True
        if tdir.exists() and tdir.is_dir():
            # borrar contenido
            for child in tdir.iterdir():
                try:
                    if child.is_file():
                        child.unlink()
                    else:
                        # borrar subdirectorios recursivamente
                        import shutil
                        shutil.rmtree(child, ignore_errors=True)
                except Exception:
                    pass
            try:
                tdir.rmdir()
                deleted["temp"] = True
            except Exception:
                # si no está vacío por algún motivo, intentar forzar
                import shutil
                shutil.rmtree(tdir, ignore_errors=True)
                deleted["temp"] = True
        return JSONResponse({"ok": True, "deleted": deleted})
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e), "deleted": deleted})
