import cv2
import numpy as np
from typing import List, Dict

def boxes_from_diffmap(diff_map: np.ndarray, thresh: float = 0.35, min_area: int = 400) -> List[Dict]:
    dm = np.clip(diff_map, 0, 1)
    bw = (dm >= float(thresh)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    boxes = []
    for i in range(1, num):
        x,y,w,h,area = stats[i]
        if area < min_area: 
            continue
        roi = dm[y:y+h, x:x+w]
        score = float(roi.mean())
        boxes.append({"x":int(x), "y":int(y), "w":int(w), "h":int(h), "score":score})
    boxes.sort(key=lambda b: b["score"], reverse=True)
    return boxes

def boxes_union(boxes: List[Dict], iou_threshold: float = 0.2) -> List[Dict]:
    def iou(a,b):
        ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"]+a["w"], a["y"]+a["h"]
        bx1, by1, bx2, by2 = b["x"], b["y"], b["x"]+b["w"], b["y"]+b["h"]
        inter_x1, inter_y1 = max(ax1,bx1), max(ay1,by1)
        inter_x2, inter_y2 = min(ax2,bx2), min(ay2,by2)
        iw, ih = max(0, inter_x2-inter_x1), max(0, inter_y2-inter_y1)
        inter = iw*ih
        area_a = a["w"]*a["h"]; area_b = b["w"]*b["h"]
        union = area_a + area_b - inter + 1e-6
        return inter/union
    out = []
    for b in boxes:
        merged = False
        for o in out:
            if iou(b,o) >= iou_threshold:
                x1 = min(o["x"], b["x"]); y1 = min(o["y"], b["y"])
                x2 = max(o["x"]+o["w"], b["x"]+b["w"]); y2 = max(o["y"]+o["h"], b["y"]+b["h"])
                o.update({"x":x1, "y":y1, "w":x2-x1, "h":y2-y1, "score":max(o["score"], b["score"])})
                merged = True
                break
        if not merged:
            out.append(b.copy())
    return out

def draw_boxes(img_bgr: np.ndarray, boxes: List[Dict], color=(0,0,255), thick=2) -> np.ndarray:
    out = img_bgr.copy()
    for b in boxes:
        cv2.rectangle(out, (b["x"], b["y"]), (b["x"]+b["w"], b["y"]+b["h"]), color, thick)
    return out
