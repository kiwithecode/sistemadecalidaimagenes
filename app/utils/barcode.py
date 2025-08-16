import cv2
from pyzbar.pyzbar import decode, ZBarSymbol

def decode_codes(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    res = decode(gray, symbols=[ZBarSymbol.QRCODE, ZBarSymbol.EAN13, ZBarSymbol.CODE128, ZBarSymbol.CODE39])
    out = []
    for r in res:
        data = r.data.decode("utf-8", errors="ignore")
        sym = r.type
        out.append({"type": sym, "data": data, "rect": {"x": r.rect.left, "y": r.rect.top, "w": r.rect.width, "h": r.rect.height}})
    return out

def match_expected_codes(found, expected_patterns):
    if not expected_patterns:
        return True, []
    misses = []
    for pat in expected_patterns:
        if ":" in pat:
            typ, pref = pat.split(":", 1)
            ok = any((f["type"].upper()==typ.upper() and pref in f["data"]) for f in found)
        else:
            ok = any(pat in f["data"] for f in found)
        if not ok:
            misses.append(pat)
    return (len(misses)==0), misses
