"""
Sistema de anotación visual de defectos para QC Wraps.
Genera imágenes con cuadros rojos y explicaciones humanas de los problemas encontrados.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from PIL import Image, ImageDraw, ImageFont
import json

class DefectAnnotator:
    """Anota defectos visualmente en imágenes con explicaciones humanas."""
    
    def __init__(self):
        self.defect_colors = {
            'color': (0, 0, 255),      # Rojo para problemas de color
            'sharpness': (255, 0, 0),  # Azul para problemas de nitidez
            'text': (0, 255, 255),     # Amarillo para problemas de texto
            'barcode': (255, 0, 255),  # Magenta para códigos
            'alignment': (0, 255, 0),  # Verde para alineación
            'general': (0, 0, 255)     # Rojo por defecto
        }
        
    def create_human_explanations(self, metrics: Dict, result: Dict) -> List[str]:
        """Genera explicaciones humanas de los defectos encontrados."""
        explanations = []
        
        # Verificar SSIM (similitud estructural)
        if 'ssim' in metrics:
            ssim_val = metrics['ssim']
            if isinstance(ssim_val, tuple):
                ssim_val = ssim_val[0] if len(ssim_val) > 0 else 0.0
            ssim_val = float(ssim_val)
            if ssim_val < 0.8:
                explanations.append(f"❌ Estructura diferente al modelo (similitud: {ssim_val:.1%})")
            
        # Verificar color
        if 'color_delta' in metrics:
            color_delta = metrics['color_delta']
            if isinstance(color_delta, tuple):
                color_delta = color_delta[0] if len(color_delta) > 0 else 0.0
            color_delta = float(color_delta)
            if color_delta > 10:
                explanations.append(f"🎨 Colores incorrectos (diferencia: {color_delta:.1f})")
            
        # Verificar nitidez
        if 'sharpness' in metrics:
            sharpness = metrics['sharpness']
            if isinstance(sharpness, tuple):
                sharpness = sharpness[0] if len(sharpness) > 0 else 0.0
            sharpness = float(sharpness)
            if sharpness < 50:
                explanations.append(f"📷 Imagen borrosa (nitidez: {sharpness:.1f})")
            
        # Verificar texto OCR
        if 'ocr_confidence' in metrics:
            ocr_conf = metrics['ocr_confidence']
            if isinstance(ocr_conf, tuple):
                ocr_conf = ocr_conf[0] if len(ocr_conf) > 0 else 0.0
            ocr_conf = float(ocr_conf)
            if ocr_conf < 80:
                explanations.append(f"📝 Texto ilegible o incorrecto (confianza: {ocr_conf:.1f}%)")
            
        # Verificar códigos de barras
        if 'barcode_found' in metrics and not metrics['barcode_found']:
            explanations.append("📊 Código de barras no detectado o ilegible")
            
        # Verificar texto faltante
        if 'missing_text' in metrics and metrics['missing_text']:
            missing = ', '.join(metrics['missing_text'])
            explanations.append(f"🔤 Texto faltante: {missing}")
            
        # Si no hay explicaciones específicas pero fue rechazada
        if not explanations and result.get('status') == 'mala':
            explanations.append("⚠️ Producto no cumple con los estándares de calidad")
            
        return explanations
        
    def find_defect_regions(self, img: np.ndarray, master: np.ndarray, 
                           aligned_img: np.ndarray, metrics: Dict) -> List[Dict]:
        """Identifica regiones con defectos para marcar con cuadros."""
        regions = []
        h, w = img.shape[:2]
        
        # Calcular mapa de diferencias
        if aligned_img is not None and master is not None:
            # Redimensionar si es necesario
            if aligned_img.shape != master.shape:
                aligned_img = cv2.resize(aligned_img, (master.shape[1], master.shape[0]))
                
            # Diferencia de color
            diff = cv2.absdiff(aligned_img, master)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Encontrar regiones con diferencias significativas
            _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Solo regiones significativas
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    regions.append({
                        'bbox': (x, y, w_box, h_box),
                        'type': 'color',
                        'description': 'Diferencia de color detectada'
                    })
        
        # Agregar regiones basadas en métricas específicas
        if 'sharpness' in metrics:
            sharpness = metrics['sharpness']
            if isinstance(sharpness, tuple):
                sharpness = sharpness[0] if len(sharpness) > 0 else 0.0
            sharpness = float(sharpness)
            if sharpness < 50:
                # Región central para problemas de nitidez
                regions.append({
                    'bbox': (w//4, h//4, w//2, h//2),
                    'type': 'sharpness',
                    'description': 'Imagen borrosa'
                })
            
        # Región para problemas de texto (parte inferior típicamente)
        if 'ocr_confidence' in metrics:
            ocr_conf = metrics['ocr_confidence']
            if isinstance(ocr_conf, tuple):
                ocr_conf = ocr_conf[0] if len(ocr_conf) > 0 else 0.0
            ocr_conf = float(ocr_conf)
            if ocr_conf < 80:
                regions.append({
                    'bbox': (w//10, h*3//4, w*8//10, h//5),
                    'type': 'text',
                    'description': 'Texto ilegible'
                })
            
        return regions
        
    def annotate_image(self, img: np.ndarray, master: np.ndarray, 
                      aligned_img: np.ndarray, metrics: Dict, 
                      result: Dict) -> np.ndarray:
        """Crea imagen anotada con cuadros rojos y explicaciones."""
        # Usar la imagen alineada como base si está disponible para mantener orientación consistente
        base = aligned_img if aligned_img is not None else img

        # Convertir a PIL para mejor manejo de texto
        img_pil = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Intentar cargar fuente, usar por defecto si falla
        try:
            font_large = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
            
        # Encontrar regiones con defectos
        defect_regions = self.find_defect_regions(base, master, aligned_img if aligned_img is not None else base, metrics)
        
        # Dibujar cuadros rojos en las regiones problemáticas
        for region in defect_regions:
            x, y, w, h = region['bbox']
            defect_type = region['type']
            color = self.defect_colors.get(defect_type, self.defect_colors['general'])
            
            # Dibujar rectángulo
            draw.rectangle([x, y, x+w, y+h], outline=color, width=4)
            
            # Etiqueta del defecto
            draw.text((x, y-25), region['description'], fill=color, font=font_small)
            
        # Generar explicaciones humanas
        explanations = self.create_human_explanations(metrics, result)
        
        # Área para explicaciones (parte superior)
        explanation_height = len(explanations) * 30 + 60
        img_height = img_pil.size[1]
        
        # Crear imagen expandida para explicaciones
        expanded_img = Image.new('RGB', (img_pil.size[0], img_height + explanation_height), 'white')
        expanded_img.paste(img_pil, (0, explanation_height))
        draw = ImageDraw.Draw(expanded_img)
        
        # Título
        status_color = (255, 0, 0) if result.get('status') == 'mala' else (0, 128, 0)
        status_text = "❌ PRODUCTO RECHAZADO" if result.get('status') == 'mala' else "✅ PRODUCTO APROBADO"
        draw.text((20, 10), status_text, fill=status_color, font=font_large)
        
        # Explicaciones
        y_offset = 50
        for explanation in explanations:
            draw.text((20, y_offset), explanation, fill=(50, 50, 50), font=font_small)
            y_offset += 30
            
        # Convertir de vuelta a OpenCV
        annotated_img = cv2.cvtColor(np.array(expanded_img), cv2.COLOR_RGB2BGR)
        
        return annotated_img
        
    def save_annotated_image(self, img: np.ndarray, master: np.ndarray,
                           aligned_img: np.ndarray, metrics: Dict,
                           result: Dict, output_path: Path) -> Path:
        """Guarda la imagen anotada con defectos marcados."""
        
        annotated = self.annotate_image(img, master, aligned_img, metrics, result)
        
        # Generar nombre de archivo
        annotated_path = output_path.parent / f"annotated_{output_path.name}"
        
        # Guardar imagen
        cv2.imwrite(str(annotated_path), annotated)
        
        return annotated_path

def create_defect_explanation(metrics: Dict, result: Dict) -> Dict:
    """Crea explicación estructurada de defectos para el reporte."""
    
    annotator = DefectAnnotator()
    explanations = annotator.create_human_explanations(metrics, result)
    
    return {
        'status': result.get('status', 'unknown'),
        'explanations': explanations,
        'summary': f"Producto {'RECHAZADO' if result.get('status') == 'mala' else 'APROBADO'}",
        'defect_count': len([e for e in explanations if '❌' in e or '🎨' in e or '📷' in e or '📝' in e or '📊' in e])
    }
