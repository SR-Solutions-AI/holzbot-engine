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
    
    def _make_item(category: str, name_en: str, cost: float, quantity: float, unit: str) -> dict:
        """Build item with cost, quantity, and PDF-friendly area_m2/unit_price."""
        q = float(quantity) if quantity else 0.0
        c = float(cost) if cost else 0.0
        # For PDF: area_m2 used when unit is m²; for ml we use quantity as length_m
        area_m2 = round(q, 2) if unit == "m²" else 0.0
        unit_price = round((c / q), 2) if q > 0 else 0.0
        return {
            "category": category,
            "name": name_en,
            "details": "",
            "cost": round(c, 2),
            "quantity": q,
            "unit": unit,
            "area_m2": area_m2,
            "unit_price": unit_price,
        }

    # 1. Roof structure (Rafters/Carpentry)
    base = components.get("roof_base", {})
    if base:
        cost_val = base.get("average_total_eur", 0.0)
        items.append(_make_item(
            "roof_base",
            "Roof structure (Rafters/Carpentry)",
            cost_val,
            area_roof,
            "m²"
        ))
        
    # 2. Sheet metal / Flashing
    sheet = components.get("sheet_metal", {})
    if sheet:
        len_m = sheet.get("perimeter_with_overhang_m", 0.0)
        cost_val = sheet.get("total_eur", 0.0)
        items.append(_make_item(
            "roof_sheet_metal",
            "Sheet metal / Flashing",
            cost_val,
            len_m,
            "ml"
        ))
        
    # 3. Extra walls (Dormers etc.)
    extra = components.get("extra_walls", {})
    if extra and extra.get("total_eur", 0) > 0:
        cost_val = extra.get("total_eur", 0.0)
        items.append(_make_item(
            "roof_extra_walls",
            "Additional roof walls (dormers etc.)",
            cost_val,
            perimeter,
            "ml"
        ))
        
    # 4. Insulation
    ins = components.get("insulation", {})
    if ins:
        cost_val = ins.get("total_eur", 0.0)
        items.append(_make_item(
            "roof_insulation",
            "Roof insulation",
            cost_val,
            area_ceiling,
            "m²"
        ))
        
    # 5. Roof covering
    mat = components.get("material", {})
    if mat:
        cost_val = mat.get("total_eur", 0.0)
        items.append(_make_item(
            "roof_cover",
            "Roof covering",
            cost_val,
            area_roof,
            "m²"
        ))

    return {
        "total_cost": total,
        "detailed_items": items
    }