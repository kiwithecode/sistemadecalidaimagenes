from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.platypus.doctemplate import SimpleDocTemplate
from reportlab.platypus.flowables import Spacer, Image
from reportlab.platypus.paragraph import Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import cv2
import re

def clean_html_for_pdf(html_text):
    """Limpia HTML complejo para compatibilidad con ReportLab."""
    if not html_text:
        return ""
    
    # Normalizaciones rápidas (colores comunes)
    html_text = html_text.replace("color: verde", "color: green")
    html_text = html_text.replace("color: rojo", "color: red")

    # Reemplazar span con color soportado por font, y eliminar cualquier otro span
    html_text = re.sub(r'<span[^>]*style="color:\s*red;?"[^>]*>(.*?)</span>', r'<font color="red">\1</font>', html_text, flags=re.IGNORECASE)
    html_text = re.sub(r'<span[^>]*style="color:\s*green;?"[^>]*>(.*?)</span>', r'<font color="green">\1</font>', html_text, flags=re.IGNORECASE)
    # Eliminar spans restantes (incluyendo incompletos)
    html_text = re.sub(r'</?span[^>]*>', '', html_text, flags=re.IGNORECASE)

    # Soporte básico: convertir etiquetas personalizadas (tabla/celda) a líneas de texto
    replacements = {
        '<tabla>': '', '</tabla>': '', '<cabecera>': '', '</cabecera>': '',
        '<cuerpo>': '', '</cuerpo>': '', '<fila>': '', '</fila>': '\n',
        '<celda>': '', '</celda>': ' | ',
    }
    for k, v in replacements.items():
        html_text = html_text.replace(k, v)

    # Convertir tablas HTML a texto estructurado
    if '<table>' in html_text or '|' in html_text:
        # Si es una tabla markdown, convertir a texto simple
        lines = html_text.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('|') and line.endswith('|'):
                # Línea de tabla markdown
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if cells and not all(cell in ['---', ''] for cell in cells):
                    clean_lines.append(' | '.join(cells))
            elif line and not line.startswith('|'):
                clean_lines.append(line)
        # Unir con saltos de línea primero, luego escapar y convertir a <br/>
        text = '\n'.join(clean_lines)
        text = re.sub(r'<[^>]+>', '', text)
        # Escapar caracteres especiales para Paragraph
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        text = text.replace('\n', '<br/>')
        return text
    
    # Si no es tabla, eliminar cualquier etiqueta HTML no soportada dejando texto plano
    html_text = re.sub(r'<[^>]+>', '', html_text)
    # Normalizar espacios y saltos y escapar caracteres especiales
    html_text = re.sub(r'\s+\n', '\n', html_text).strip()
    html_text = html_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    # Preservar saltos de línea en Paragraph
    html_text = html_text.replace('\n', '<br/>')
    return html_text

def save_image(path, bgr):
    cv2.imwrite(str(path), bgr)

def save_overlay(path, base_bgr, heat_bgr, alpha=0.6):
    h, w = base_bgr.shape[:2]
    heat_res = cv2.resize(heat_bgr, (w, h))
    overlay = cv2.addWeighted(base_bgr, 1.0, heat_res, alpha, 0)
    cv2.imwrite(str(path), overlay)

def draw_pdf(pdf_path, sku, lote_id, items):
    """Genera PDF con formato tabular profesional usando platypus."""
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []
    
    # Título principal
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        textColor=colors.darkblue
    )
    story.append(Paragraph(f"<b>Reporte de Control de Calidad</b>", title_style))
    story.append(Paragraph(f"<b>SKU:</b> {sku} | <b>Lote:</b> {lote_id} | <b>Total:</b> {len(items)} imágenes", styles['Normal']))
    story.append(Spacer(1, 20))
    
    for it in items:
        # Encabezado de muestra
        status_color = colors.green if it['status'] == 'buena' else colors.red
        status_style = ParagraphStyle(
            'Status',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=status_color,
            spaceAfter=15
        )
        
        story.append(Paragraph(f"<b>Archivo:</b> {it['file']} | <b>Estado:</b> {it['status'].upper()}", status_style))
        
        # Imágenes de comparación
        try:
            # Crear tabla con imágenes lado a lado
            img_data = []
            img_row = []
            
            # Imagen alineada
            if it.get("img_path"):
                aligned_img = Image(it["img_path"], width=7*cm, height=7*cm)
                img_row.append(aligned_img)
            
            # Overlay de diferencias
            if it.get("overlay_path"):
                overlay_img = Image(it["overlay_path"], width=7*cm, height=7*cm)
                img_row.append(overlay_img)
            
            if img_row:
                img_data.append(img_row)
                
                # Etiquetas para las imágenes
                labels = ["Imagen Alineada", "Mapa de Diferencias"][:len(img_row)]
                img_data.append(labels)
                
                img_table = Table(img_data, colWidths=[7.5*cm] * len(img_row))
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
                    ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 1), (-1, 1), 10),
                    ('TOPPADDING', (0, 1), (-1, 1), 10),
                ]))
                
                story.append(img_table)
                story.append(Spacer(1, 20))
        except Exception as e:
            # Si hay error con imágenes, continuar sin ellas
            story.append(Paragraph(f"<i>Error cargando imágenes: {str(e)}</i>", styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Tabla de métricas
        metric_data = [['Métrica', 'Valor', 'Umbral', 'Estado']]
        
        metric_names = {
            "ssim_global": "Similitud Estructural",
            "deltaE_avg": "Color Promedio (ΔE)",
            "deltaE_max": "Color Máximo (ΔE)", 
            "sharpness": "Nitidez",
            "ocr_conf": "Confianza OCR"
        }
        
        for m in it["metrics"]:
            name = metric_names.get(m["name"], m["name"])
            
            # Formatear valor
            if m["name"] == "ssim_global":
                value = f"{m['value']:.1%}"
                threshold = f"≥{m['threshold']:.1%}" if m['threshold'] else "N/A"
            elif m["name"] in ["deltaE_avg", "deltaE_max"]:
                value = f"{m['value']:.1f}"
                threshold = f"≤{m['threshold']:.1f}" if m['threshold'] else "N/A"
            elif m["name"] == "ocr_conf":
                value = f"{m['value']:.1%}"
                threshold = f"≥{m['threshold']:.1%}" if m['threshold'] else "N/A"
            else:
                value = f"{m['value']:.0f}"
                threshold = f"≥{m['threshold']:.0f}" if m['threshold'] else "N/A"
            
            status = "✓ PASA" if m.get("passed") else "✗ FALLA"
            metric_data.append([name, value, threshold, status])
        
        # Crear tabla de métricas
        metrics_table = Table(metric_data, colWidths=[4*cm, 2.5*cm, 2.5*cm, 2.5*cm])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        # Colorear filas según estado
        for i, m in enumerate(it["metrics"], 1):
            if m.get("passed"):
                metrics_table.setStyle(TableStyle([('TEXTCOLOR', (3, i), (3, i), colors.green)]))
            else:
                metrics_table.setStyle(TableStyle([('TEXTCOLOR', (3, i), (3, i), colors.red)]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 15))
        
        # Información LAB
        lab = it.get("lab", {})
        if lab:
            img_lab = lab.get("lab_avg_img", [0,0,0])
            ref_lab = lab.get("lab_avg_master", [0,0,0])
            dlab = lab.get("lab_diff", [0,0,0])
            
            lab_data = [
                ['Parámetro', 'L*', 'a*', 'b*'],
                ['Imagen', f"{img_lab[0]:.1f}", f"{img_lab[1]:.1f}", f"{img_lab[2]:.1f}"],
                ['Master', f"{ref_lab[0]:.1f}", f"{ref_lab[1]:.1f}", f"{ref_lab[2]:.1f}"],
                ['Diferencia (Δ)', f"{dlab[0]:+.1f}", f"{dlab[1]:+.1f}", f"{dlab[2]:+.1f}"]
            ]
            
            lab_table = Table(lab_data, colWidths=[3*cm, 2*cm, 2*cm, 2*cm])
            lab_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            story.append(Paragraph("<b>Análisis de Color (CIELAB):</b>", styles['Heading3']))
            story.append(lab_table)
            story.append(Spacer(1, 15))
        
        # Defectos específicos
        defects_info = []
        if it.get("text_missing"):
            defects_info.append(f"<b>Textos faltantes:</b> {', '.join(it['text_missing'])}")
        if it.get("codes_missing"):
            defects_info.append(f"<b>Códigos faltantes:</b> {', '.join(it['codes_missing'])}")
        if it.get("codes_found"):
            codes_str = ", ".join([f"{cf['type']}: {cf['data'][:30]}..." for cf in it['codes_found'][:3]])
            defects_info.append(f"<b>Códigos encontrados:</b> {codes_str}")
        
        if defects_info:
            story.append(Paragraph("<b>Información Adicional:</b>", styles['Heading3']))
            for info in defects_info:
                story.append(Paragraph(f"• {info}", styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Explicación (convertir HTML a texto plano para ReportLab)
        explanation = it.get("explanation", "")
        if explanation:
            story.append(Paragraph("<b>Evaluación:</b>", styles['Heading3']))
            # Limpiar HTML complejo que ReportLab no puede manejar
            clean_explanation = clean_html_for_pdf(explanation)
            story.append(Paragraph(clean_explanation, styles['Normal']))
        
        story.append(Spacer(1, 30))
    
    doc.build(story)
