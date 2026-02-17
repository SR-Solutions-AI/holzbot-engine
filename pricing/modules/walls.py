def calculate_walls_details(coeffs: dict, area_int_net: float, area_ext_net: float, system: str) -> dict:
    """Pereți portanți: preț din sistem (fără grad prefabricare). Factorii acces șantier și teren se aplică la structura totală în calculator."""
    base_prices = coeffs["base_unit_prices"].get(system, {"interior": 0.0, "exterior": 0.0})
    price_int = base_prices["interior"]
    price_ext = base_prices["exterior"]

    cost_int = area_int_net * price_int
    cost_ext = area_ext_net * price_ext

    return {
        "total_cost": round(cost_int + cost_ext, 2),
        "detailed_items": [
            {
                "category": "walls_structure_int",
                "name": f"Pereți Interiori ({system})",
                "material": system,
                "construction_mode": system,
                "area_m2": round(area_int_net, 2),
                "unit_price": round(price_int, 2),
                "cost": round(cost_int, 2)
            },
            {
                "category": "walls_structure_ext",
                "name": f"Pereți Exteriori ({system})",
                "material": system,
                "construction_mode": system,
                "area_m2": round(area_ext_net, 2),
                "unit_price": round(price_ext, 2),
                "cost": round(cost_ext, 2)
            }
        ]
    }