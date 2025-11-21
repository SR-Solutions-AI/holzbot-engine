def calculate_openings_details(coeffs: dict, openings_list: list, material: str) -> dict:
    items = []
    total = 0.0
    
    for op in openings_list:
        obj_type = op.get("type", "unknown")
        width = float(op.get("width_m", 0.0))
        
        # Înălțime standard estimată (pentru calcul arie)
        height = 2.05 if "door" in obj_type else 1.25
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
        
        cost = area * price_per_m2
        total += cost
        
        # Item detaliat
        items.append({
            "id": op.get("id"),
            "name": f"{obj_type.replace('_', ' ').title()} #{op.get('id')}",
            "type": obj_type,
            "location": "Exterior" if is_exterior else "Interior",
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