def calculate_finishes_details(coeffs: dict, area_int_net: float, area_ext_net: float, type_int: str, type_ext: str, floor_label: str = "") -> dict:
    price_int = coeffs["interior"].get(type_int, 0.0)
    price_ext = coeffs["exterior"].get(type_ext, 0.0)
    
    cost_int = area_int_net * price_int
    cost_ext = area_ext_net * price_ext
    
    total = cost_int + cost_ext
    
    # Adăugăm etajul la nume dacă este specificat
    floor_suffix = f" - {floor_label}" if floor_label else ""
    
    return {
        "total_cost": round(total, 2),
        "detailed_items": [
            {
                "category": "finish_interior",
                "name": f"Finisaj Interior ({type_int}){floor_suffix}",
                # explicit material for PDF tables
                "material": type_int,
                "area_m2": round(area_int_net, 2),
                "unit_price": price_int,
                "cost": round(cost_int, 2),
                "total_cost": round(cost_int, 2),  # Pentru compatibilitate
                "floor_label": floor_label  # Adăugăm etajul explicit
            },
            {
                "category": "finish_exterior",
                "name": f"Finisaj Exterior ({type_ext}){floor_suffix}",
                # explicit material for PDF tables
                "material": type_ext,
                "area_m2": round(area_ext_net, 2),
                "unit_price": price_ext,
                "cost": round(cost_ext, 2),
                "total_cost": round(cost_ext, 2),  # Pentru compatibilitate
                "floor_label": floor_label  # Adăugăm etajul explicit
            }
        ]
    }