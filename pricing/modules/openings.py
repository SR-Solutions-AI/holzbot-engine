def calculate_openings_details(coeffs: dict, openings_list: list, frontend_data: dict | None = None) -> dict:
    """
    Cost deschideri: arie = lățime × înălțime (înălțimea vine doar din formular, pentru calcul suprafață; nu afectează prețul).
    Uși: un preț €/m² interior, unul exterior. Ferestre: preț 2-fach sau 3-fach (din formular Fensterart).
    """
    items = []
    total = 0.0

    ferestre_usi = frontend_data.get("ferestreUsi", {}) if frontend_data else {}
    # Înălțimi doar pentru calculul ariei (nu pentru preț)
    window_height_m = 1.25
    door_height_m = 2.05

    if frontend_data and ferestre_usi:
        # Bodentiefe Fenster → înălțime ferestre (doar pentru suprafață)
        bodentiefe = ferestre_usi.get("bodentiefeFenster", "")
        if bodentiefe == "Nein":
            window_height_m = 1.0
        elif bodentiefe == "Ja – einzelne":
            window_height_m = 1.5
        elif bodentiefe == "Ja – mehrere / große Glasflächen":
            window_height_m = 2.0
        # Türhöhe → înălțime uși (doar pentru suprafață)
        turhohe = ferestre_usi.get("turhohe", "")
        if turhohe == "Erhöht / Sondermaß (2,2+ m)":
            door_height_m = 2.2
        elif turhohe == "Standard (2m)":
            door_height_m = 2.0

    # Prețuri: uși interior/exterior (€/m²), ferestre după Fensterart (2-fach / 3-fach)
    door_int_price = float(coeffs.get("door_interior_price_per_m2", 0))
    door_ext_price = float(coeffs.get("door_exterior_price_per_m2", 0))
    windows_prices = coeffs.get("windows_price_per_m2", {})
    window_quality = (ferestre_usi.get("windowQuality", "3-fach verglast") if frontend_data else "3-fach verglast")
    window_price_per_m2 = float(windows_prices.get(window_quality, windows_prices.get("3-fach verglast", 0)))

    for op in openings_list:
        obj_type = op.get("type", "unknown")
        width = float(op.get("width_m", 0.0))
        height = door_height_m if "door" in obj_type else window_height_m
        area = width * height

        if "door" in obj_type:
            is_exterior = op.get("status") == "exterior"
            price_per_m2 = door_ext_price if is_exterior else door_int_price
            door_status = op.get("status", "interior")
        else:
            is_exterior = True
            price_per_m2 = window_price_per_m2
            door_status = None

        cost = area * price_per_m2
        total += cost

        material_label = "Interior" if not is_exterior and "door" in obj_type else ("Exterior" if "door" in obj_type else window_quality)
        items.append({
            "id": op.get("id"),
            "name": f"{obj_type.replace('_', ' ').title()} #{op.get('id')}",
            "type": obj_type,
            "location": "Exterior" if is_exterior else "Interior",
            "status": door_status if door_status else ("exterior" if is_exterior else "interior"),
            "material": material_label,
            "width_m": round(width, 2),
            "height_m": round(height, 2),
            "dimensions_m": f"{width:.2f} x {height:.2f}",
            "area_m2": round(area, 2),
            "unit_price": price_per_m2,
            "total_cost": round(cost, 2)
        })

    return {
        "total_cost": round(total, 2),
        "items": items,
        "detailed_items": items,
    }
