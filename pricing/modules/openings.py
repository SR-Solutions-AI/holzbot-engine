def calculate_openings_details(coeffs: dict, openings_list: list, material: str, frontend_data: dict | None = None) -> dict:
    items = []
    total = 0.0
    
    # Determinăm înălțimile și coeficientul pentru calitatea geamurilor din formular
    window_height_m = 1.25  # Default
    door_height_m = 2.05  # Default
    window_quality_multiplier = 1.0  # Default (2-fach verglast)
    
    if frontend_data:
        ferestre_usi = frontend_data.get("ferestreUsi", {})
        
        # Înălțime ferestre bazată pe bodentiefeFenster
        bodentiefe = ferestre_usi.get("bodentiefeFenster", "")
        if bodentiefe == "Nein":
            window_height_m = 1.0  # 1m pentru toate geamurile
        elif bodentiefe == "Ja – einzelne":
            window_height_m = 1.5  # 1.5m pentru fiecare geam
        elif bodentiefe == "Ja – mehrere / große Glasflächen":
            window_height_m = 2.0  # 2m pentru toate geamurile
        
        # Înălțime uși bazată pe turhohe
        turhohe = ferestre_usi.get("turhohe", "")
        if turhohe == "Erhöht / Sondermaß (2,2+ m)":
            door_height_m = 2.2  # 2.2m pentru uși înalte
        
        # Coeficient pentru calitatea geamurilor
        window_quality = ferestre_usi.get("windowQuality", "")
        if window_quality == "3-fach verglast":
            window_quality_multiplier = 1.25
        elif window_quality == "3-fach verglast, Passiv":
            window_quality_multiplier = 1.6
    
    for op in openings_list:
        obj_type = op.get("type", "unknown")
        width = float(op.get("width_m", 0.0))
        
        # Folosim înălțimile din formular
        height = door_height_m if "door" in obj_type else window_height_m
        area = width * height
        
        # Determinare categorie preț
        category_key = "windows_unit_prices_per_m2"
        is_exterior = True
        
        if "door" in obj_type:
            if op.get("status") == "exterior":
                category_key = "doors_exterior_unit_prices_per_m2"
                is_exterior = True
            else:
                category_key = "doors_interior_unit_prices_per_m2"
                is_exterior = False
        elif "window" in obj_type:
            category_key = "windows_unit_prices_per_m2"
            is_exterior = True
            
        # Preț unitar
        cat_prices = coeffs.get(category_key, {})
        price_per_m2 = cat_prices.get(material, 0.0)
        
        # Aplicăm coeficientul pentru calitatea geamurilor (doar pentru ferestre)
        if "window" in obj_type:
            price_per_m2 = price_per_m2 * window_quality_multiplier
        
        cost = area * price_per_m2
        total += cost
        
        # Item detaliat
        # Păstrăm status-ul original pentru uși (exterior/interior)
        door_status = op.get("status", "interior") if "door" in obj_type else None
        
        items.append({
            "id": op.get("id"),
            "name": f"{obj_type.replace('_', ' ').title()} #{op.get('id')}",
            "type": obj_type,
            "location": "Exterior" if is_exterior else "Interior",
            "status": door_status if door_status else ("exterior" if is_exterior else "interior"),  # Păstrăm status-ul pentru uși
            "material": material,
            "dimensions_m": f"{width:.2f} x {height:.2f}",
            "area_m2": round(area, 2),
            "unit_price": price_per_m2,
            "total_cost": round(cost, 2)
        })

    return {
        "total_cost": round(total, 2),
        "items": items
    }