# runner/pricing/modules/roof.py
def calculate_roof_details(roof_result_data: dict) -> dict:
    """
    Transformă output-ul complex din etapa 'roof' într-o listă simplă de items pentru pricing.
    Include cantități (mp, ml) pentru afișare corectă în PDF.
    """
    components = roof_result_data.get("components", {})
    inputs = roof_result_data.get("inputs", {})
    total = roof_result_data.get("roof_final_total_eur", 0.0)
    
    # Extragem cantitățile de bază
    area_roof = inputs.get("house_area_m2", 0.0)
    area_ceiling = inputs.get("ceiling_area_m2", 0.0)
    perimeter = inputs.get("perimeter_m", 0.0)
    
    items = []
    
    # 1. Baza (Structură)
    base = components.get("roof_base", {})
    if base:
        items.append({
            "category": "roof_base",
            "name": "Dachstruktur (Șarpantă)",
            "details": base.get("description", ""),
            "cost": base.get("average_total_eur", 0.0),
            "quantity": area_roof,
            "unit": "m²"
        })
        
    # 2. Tinichigerie
    sheet = components.get("sheet_metal", {})
    if sheet:
        # Lungimea calculată cu tot cu overhang
        len_m = sheet.get("perimeter_with_overhang_m", 0.0)
        items.append({
            "category": "roof_sheet_metal",
            "name": "Spenglerarbeiten (Tinichigerie)",
            "details": sheet.get("description", ""),
            "cost": sheet.get("total_eur", 0.0),
            "quantity": len_m,
            "unit": "ml"
        })
        
    # 3. Pereți Extra (Lucarne etc.)
    extra = components.get("extra_walls", {})
    if extra and extra.get("total_eur", 0) > 0:
        items.append({
            "category": "roof_extra_walls",
            "name": "Zusatzwände Dach",
            "details": extra.get("description", ""),
            "cost": extra.get("total_eur", 0.0),
            "quantity": perimeter,
            "unit": "ml"
        })
        
    # 4. Insulation
    ins = components.get("insulation", {})
    if ins:
        items.append({
            "category": "roof_insulation",
            "name": "Dämmung (Izolație)",
            "details": ins.get("description", ""),
            "cost": ins.get("total_eur", 0.0),
            "quantity": area_ceiling,
            "unit": "m²"
        })
        
    # 5. Material (Tiglă/Tablă)
    mat = components.get("material", {})
    if mat:
        items.append({
            "category": "roof_cover",
            "name": "Dacheindeckung (Învelitoare)",
            "details": mat.get("description", ""),
            "cost": mat.get("total_eur", 0.0),
            "quantity": area_roof,
            "unit": "m²"
        })

    return {
        "total_cost": total,
        "detailed_items": items
    }