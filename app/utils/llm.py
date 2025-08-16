from __future__ import annotations
import json, requests

SYSTEM_PROMPT = (
    "Eres un experto en control de calidad de empaques. "
    "Responde SOLO con formato de tabla HTML limpia y concisa. "
    "Incluye: 1) Tabla de métricas con valores/umbrales, 2) Defectos principales, 3) Recomendaciones breves. "
    "Usa colores: rojo para fallidas, verde para aprobadas. "
    "Máximo 150 palabras total."
)

def _build_prompt(defects, metrics, extras):
    m = {}
    for k, vt in metrics.items():
        val, thr = vt
        if hasattr(val, "__len__") and k.startswith("lab_"):
            m[k] = {"value": list(val), "threshold": thr}
        else:
            m[k] = {"value": float(val), "threshold": (None if thr is None else float(thr))}
    payload = {
        "task": "explain_bad_sample",
        "defects": defects,
        "metrics": m,
        "extras": extras or {},
        "instructions": "Redacta 2–5 frases claras; cita métricas y umbrales si aplican."
    }
    return json.dumps(payload, ensure_ascii=False)

def explain_with_ollama(host, model, prompt, max_tokens=220):
    url = f"{host}/api/generate"
    data = {
        "model": model,
        "prompt": f"{SYSTEM_PROMPT}\nINPUT:\n{prompt}\nRESPUESTA:",
        "options": {"num_predict": max_tokens, "temperature": 0.25},
        "stream": False
    }
    r = requests.post(url, json=data, timeout=60)
    r.raise_for_status()
    return r.json().get("response","").strip()

def explain_via_llm(cfg_llm, defects, metrics, extras):
    host = cfg_llm.get("host","http://localhost:11434")
    model = cfg_llm.get("model","llama3.1:8b")
    prompt = _build_prompt(defects, metrics, extras)
    return explain_with_ollama(host, model, prompt, int(cfg_llm.get("max_tokens",220)))
