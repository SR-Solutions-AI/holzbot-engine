import re


def _infer_window_height_from_width(width_m: float, obj_type: str = "") -> float:
    t = str(obj_type or "").lower()
    if "double" in t:
        return 2.0
    if width_m >= 2.0:
        return 2.0
    if width_m >= 1.35:
        return 1.5
    return 1.0


def _parse_height_from_label(label: str | None) -> float | None:
    if not label:
        return None
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*m", str(label).lower())
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except Exception:
        return None


def calculate_openings_details(coeffs: dict, openings_list: list, frontend_data: dict | None = None) -> dict:
    """
    Ferestre: cost = suprafață (m²) × preț €/m² (Fensterart din formular).
    Uși normale: cost = preț €/Stück după tipul Innen/Außen selectat (o bucată = o deschidere în plan).
    Garagentor: dacă „Garagentor gewünscht“ e bifat, cost = preț €/Stück; altfel 0 €, dar rămâne în listă (aceeași număr ca în editor).
    """
    items = []
    total = 0.0

    ferestre_usi = frontend_data.get("ferestreUsi", {}) if frontend_data else {}
    door_int_prices = coeffs.get("door_interior_prices", {}) or {}
    door_ext_prices = coeffs.get("door_exterior_prices", {}) or {}
    door_material_int = (ferestre_usi.get("doorMaterialInterior", "Standard") or "Standard").strip() if frontend_data else "Standard"
    door_material_ext = (ferestre_usi.get("doorMaterialExterior", "Standard") or "Standard").strip() if frontend_data else "Standard"
    door_price_int = float(door_int_prices.get(door_material_int, 0))
    door_price_ext = float(door_ext_prices.get(door_material_ext, 0))

    windows_prices = coeffs.get("windows_price_per_m2", {})
    window_quality = (ferestre_usi.get("windowQuality", "3-fach verglast") if frontend_data else "3-fach verglast")
    window_price_per_m2 = float(windows_prices.get(window_quality, windows_prices.get("3-fach verglast", 0)))
    sliding_prices = coeffs.get("sliding_door_prices_per_m2", {}) or {}
    sliding_type = (ferestre_usi.get("slidingDoorType", "Standard") if frontend_data else "Standard")
    sliding_price_per_m2 = float(sliding_prices.get(sliding_type, sliding_prices.get("Standard", 0)))

    garage_prices = coeffs.get("garage_door_prices", {}) or {}
    garage_type = (ferestre_usi.get("garageDoorType", "Sektionaltor Standard") if frontend_data else "Sektionaltor Standard")
    wants_garage_door = bool(ferestre_usi.get("garagentorGewuenscht")) if frontend_data else False
    garage_price_piece = float(
        garage_prices.get(garage_type, garage_prices.get("Sektionaltor Standard", 0)),
    )

    door_height_map = coeffs.get("door_height_m", {}) or {}
    door_height_label = (
        (ferestre_usi.get("doorHeightOption") or ferestre_usi.get("turhohe") or "Standard (2,01 m)")
        if frontend_data
        else "Standard (2,01 m)"
    )
    selected_door_height_m = float(door_height_map.get(door_height_label, 2.05))
    if door_height_label not in door_height_map:
        parsed_h = _parse_height_from_label(door_height_label)
        if parsed_h and parsed_h > 0:
            selected_door_height_m = parsed_h

    for op in openings_list:
        obj_type = op.get("type", "unknown")
        ot = str(obj_type).lower()
        width = float(op.get("width_m", 0.0))
        if ot == "garage_door":
            explicit_h = op.get("height_m")
            if explicit_h is not None and float(explicit_h) > 0:
                height = float(explicit_h)
            else:
                height = 2.1
        elif "window" in ot or ot == "sliding_door":
            explicit_h = op.get("height_m")
            if explicit_h is not None and float(explicit_h) > 0:
                height = float(explicit_h)
            elif ot == "sliding_door":
                height = 2.0
            else:
                height = _infer_window_height_from_width(width, obj_type)
        elif "door" in ot:
            explicit_h = op.get("height_m")
            if explicit_h is not None and float(explicit_h) > 0:
                height = float(explicit_h)
            else:
                height = selected_door_height_m
        else:
            explicit_h = op.get("height_m")
            if explicit_h is not None and float(explicit_h) > 0:
                height = float(explicit_h)
            else:
                height = _infer_window_height_from_width(width, obj_type)
        area = width * height

        if ot == "garage_door":
            # Immer eine Zeile pro Öffnung (wie Editor): ohne Wunsch-Flag Kosten 0, damit
            # breakdown/Angebot dieselbe Stückzahl wie measurements_plan.json haben.
            if wants_garage_door:
                cost = garage_price_piece
                price_factor = garage_price_piece
            else:
                cost = 0.0
                price_factor = 0.0
            unit_label = "€/Stück"
            material_label = garage_type
            door_status = "exterior"
            is_exterior = True
        elif "door" in ot and ot != "sliding_door":
            is_exterior = op.get("status") == "exterior"
            cost = door_price_ext if is_exterior else door_price_int
            unit_label = "€/Stück"
            price_factor = cost
            material_label = door_material_ext if is_exterior else door_material_int
            door_status = op.get("status", "interior")
        else:
            is_exterior = True
            if ot == "sliding_door":
                cost = area * sliding_price_per_m2
                price_factor = sliding_price_per_m2
                material_label = sliding_type
            else:
                cost = area * window_price_per_m2
                price_factor = window_price_per_m2
                material_label = window_quality
            unit_label = "€/m²"
            door_status = None

        total += cost

        items.append({
            "id": op.get("id"),
            "name": f"{str(obj_type).replace('_', ' ').title()} #{op.get('id')}",
            "type": obj_type,
            "location": "Exterior" if is_exterior else "Interior",
            "status": door_status if door_status else ("exterior" if is_exterior else "interior"),
            "material": material_label,
            "width_m": round(width, 2),
            "height_m": round(height, 2),
            "dimensions_m": f"{width:.2f} x {height:.2f}",
            "area_m2": round(area, 2),
            "unit_price": round(price_factor, 2),
            "price_unit": unit_label,
            "total_cost": round(cost, 2),
        })

    return {
        "total_cost": round(total, 2),
        "items": items,
        "detailed_items": items,
    }
