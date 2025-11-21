def calculate_finishes_details(coeffs: dict, area_int_net: float, area_ext_net: float, type_int: str, type_ext: str) -> dict:
    price_int = coeffs["interior"].get(type_int, 0.0)
    price_ext = coeffs["exterior"].get(type_ext, 0.0)
    
    cost_int = area_int_net * price_int
    cost_ext = area_ext_net * price_ext
    
    total = cost_int + cost_ext
    
    return {
        "total_cost": round(total, 2),
        "detailed_items": [
            {
                "category": "finish_interior",
                "name": f"Finisaj Interior ({type_int})",
                "area_m2": round(area_int_net, 2),
                "unit_price": price_int,
                "cost": round(cost_int, 2)
            },
            {
                "category": "finish_exterior",
                "name": f"Finisaj Exterior ({type_ext})",
                "area_m2": round(area_ext_net, 2),
                "unit_price": price_ext,
                "cost": round(cost_ext, 2)
            }
        ]
    }