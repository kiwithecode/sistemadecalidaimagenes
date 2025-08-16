from typing import List, Dict, Optional
from pydantic import BaseModel

class Metric(BaseModel):
    name: str
    value: float
    threshold: Optional[float] = None
    passed: Optional[bool] = None

class ImageResult(BaseModel):
    file: str
    sku: str
    status: str
    defects: List[str]
    metrics: List[Metric]
    lab: Dict[str, list]
    text_missing: List[str]
    codes_missing: List[str]
    codes_found: List[Dict]
    explanation: str
    boxes: List[Dict] = []

class BatchResponse(BaseModel):
    lote: str
    sku: str
    resultados: List[ImageResult]
    pdf: str
