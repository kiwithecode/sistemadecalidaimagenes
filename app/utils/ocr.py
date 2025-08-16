import pytesseract, cv2
from pytesseract import TesseractNotFoundError

def ocr_text_conf(img_bgr):
    try:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)
        confs = [float(c) for c in data.get("conf", []) if c not in ("-1", -1, None)]
        text = " ".join([w for w in data.get("text", []) if w])
        avg_conf = float(sum(confs)/len(confs)) if confs else 0.0
        return text, avg_conf/100.0
    except TesseractNotFoundError:
        return "", 0.0
