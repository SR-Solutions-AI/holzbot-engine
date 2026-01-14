def calculate_walls_details(coeffs: dict, area_int_net: float, area_ext_net: float, system: str, prefab_type: str) -> dict:
    base_prices = coeffs["base_unit_prices"].get(system, {"interior": 0.0, "exterior": 0.0})
    modifier = coeffs["prefabrication_modifiers"].get(prefab_type, 1.0)
    
    final_price_int = base_prices["interior"] * modifier
    final_price_ext = base_prices["exterior"] * modifier
    
    cost_int = area_int_net * final_price_int
    cost_ext = area_ext_net * final_price_ext
    
    return {
        "total_cost": round(cost_int + cost_ext, 2),
        "detailed_items": [
            {
                "category": "walls_structure_int",
                "name": f"Pereți Interiori ({system} - {prefab_type})",
                # explicit fields for PDF tables
                "material": system,
                "construction_mode": prefab_type,
                "area_m2": round(area_int_net, 2),
                "unit_price": round(final_price_int, 2),
                "cost": round(cost_int, 2)
            },
            {
                "category": "walls_structure_ext",
                "name": f"Pereți Exteriori ({system} - {prefab_type})",
                # explicit fields for PDF tables
                "material": system,
                "construction_mode": prefab_type,
                "area_m2": round(area_ext_net, 2),
                "unit_price": round(final_price_ext, 2),
                "cost": round(cost_ext, 2)
            }
        ]
    }