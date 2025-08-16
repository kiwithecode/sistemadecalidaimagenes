def human_explanation(defects, metrics: dict, extras=None) -> str:
    """Genera explicación profesional con formato de tabla HTML."""
    ex = extras or {}
    
    # Si no hay defectos, dar explicación positiva
    if not defects:
        return """
<div style="background: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 5px;">
    <h3 style="color: #155724; margin: 0;">✅ MUESTRA APROBADA</h3>
    <p style="margin: 5px 0;">La muestra cumple con todos los estándares de calidad establecidos.</p>
</div>
"""
    
    # Tabla de métricas
    table_rows = []
    metric_names = {
        "ssim_global": "Similitud Estructural",
        "deltaE_avg": "Color Promedio (ΔE)",
        "deltaE_max": "Color Máximo (ΔE)", 
        "sharpness": "Nitidez",
        "ocr_conf": "Confianza OCR"
    }
    
    for key, name in metric_names.items():
        if key in metrics:
            val, thresh = metrics[key]
            if key == "ssim_global":
                val_str = f"{val:.1%}"
                thresh_str = f"{thresh:.1%}"
                passed = val >= thresh
            elif key in ["deltaE_avg", "deltaE_max"]:
                val_str = f"{val:.1f}"
                thresh_str = f"≤{thresh:.1f}"
                passed = val <= thresh
            elif key == "ocr_conf":
                val_str = f"{val:.1%}"
                thresh_str = f"≥{thresh:.1%}"
                passed = val >= thresh
            else:
                val_str = f"{val:.0f}"
                thresh_str = f"≥{thresh:.0f}"
                passed = val >= thresh
            
            color = "#28a745" if passed else "#dc3545"
            status = "✓" if passed else "✗"
            
            table_rows.append(f"""
                <tr>
                    <td>{name}</td>
                    <td style="color: {color}; font-weight: bold;">{val_str}</td>
                    <td>{thresh_str}</td>
                    <td style="color: {color}; font-weight: bold;">{status}</td>
                </tr>
            """)
    
    # Defectos específicos
    defect_items = []
    if "texto" in defects:
        miss = ex.get("text_missing", [])
        if miss:
            defect_items.append(f"<li><strong>Textos faltantes:</strong> {', '.join(miss)}</li>")
    
    if "codigo" in defects:
        miss = ex.get("codes_missing", [])
        if miss:
            defect_items.append(f"<li><strong>Códigos faltantes:</strong> {', '.join(miss)}</li>")
    
    if "luminosidad" in defects:
        if "lab_diff" in metrics:
            dL = metrics["lab_diff"][0][0]  # Diferencia en L*
            defect_items.append(f"<li><strong>Problema de luminosidad:</strong> ΔL = {dL:+.1f} (galleta {'más oscura' if dL < 0 else 'más clara'} que el estándar)</li>")
    
    defects_html = ""
    if defect_items:
        defects_html = f"""
        <div style="margin-top: 15px;">
            <h4 style="color: #721c24; margin-bottom: 10px;">Defectos Específicos:</h4>
            <ul style="margin: 0; padding-left: 20px;">
                {''.join(defect_items)}
            </ul>
        </div>
        """
    
    return f"""
<div style="background: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 5px;">
    <h3 style="color: #721c24; margin: 0 0 15px 0;">❌ MUESTRA RECHAZADA</h3>
    
    <table style="width: 100%; border-collapse: collapse; margin-bottom: 10px;">
        <thead>
            <tr style="background: #e9ecef;">
                <th style="padding: 8px; text-align: left; border: 1px solid #dee2e6;">Métrica</th>
                <th style="padding: 8px; text-align: left; border: 1px solid #dee2e6;">Valor</th>
                <th style="padding: 8px; text-align: left; border: 1px solid #dee2e6;">Umbral</th>
                <th style="padding: 8px; text-align: center; border: 1px solid #dee2e6;">Estado</th>
            </tr>
        </thead>
        <tbody>
            {''.join(table_rows)}
        </tbody>
    </table>
    
    {defects_html}
</div>
"""
