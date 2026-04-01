def calculate_finishes_details(
    coeffs: dict,
    area_int_inner_net: float,
    area_int_outer_net: float,
    area_ext_net: float,
    type_int_inner: str,
    type_int_outer: str,
    type_ext: str,
    floor_label: str = "",
) -> dict:
    price_int_inner = coeffs["interior_inner"].get(type_int_inner, 0.0)
    price_int_outer = coeffs["interior_outer"].get(type_int_outer, 0.0)
    price_ext = coeffs["exterior"].get(type_ext, 0.0)

    cost_int_inner = area_int_inner_net * price_int_inner
    cost_int_outer = area_int_outer_net * price_int_outer
    cost_ext = area_ext_net * price_ext

    total = cost_int_inner + cost_int_outer + cost_ext

    # Adăugăm etajul la nume dacă este specificat
    floor_suffix = f" - {floor_label}" if floor_label else ""

    return {
        "total_cost": round(total, 2),
        "detailed_items": [
            {
                "category": "finish_interior_inner",
                "name": f"Innenausbau Innenwände ({type_int_inner}){floor_suffix}",
                # explicit material for PDF tables
                "material": type_int_inner,
                "area_m2": round(area_int_inner_net, 2),
                "unit_price": price_int_inner,
                "cost": round(cost_int_inner, 2),
                "total_cost": round(cost_int_inner, 2),  # Pentru compatibilitate
                "floor_label": floor_label  # Adăugăm etajul explicit
            },
            {
                "category": "finish_interior_outer",
                "name": f"Innenausbau Außenwände ({type_int_outer}){floor_suffix}",
                "material": type_int_outer,
                "area_m2": round(area_int_outer_net, 2),
                "unit_price": price_int_outer,
                "cost": round(cost_int_outer, 2),
                "total_cost": round(cost_int_outer, 2),
                "floor_label": floor_label
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