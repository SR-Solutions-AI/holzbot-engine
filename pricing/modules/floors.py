def calculate_floors_details(coeffs: dict, floor_area: float, ceiling_area: float) -> dict:
    price_floor = coeffs.get("floor_coefficient_per_m2", 0.0)
    price_ceiling = coeffs.get("ceiling_coefficient_per_m2", 0.0)
    
    cost_floor = floor_area * price_floor
    cost_ceiling = ceiling_area * price_ceiling
    
    return {
        "total_cost": round(cost_floor + cost_ceiling, 2),
        "detailed_items": [
            {
                "category": "floor_structure",
                "name": "Structură Planșeu/Podea",
                "area_m2": round(floor_area, 2),
                "unit_price": price_floor,
                "cost": round(cost_floor, 2)
            },
            {
                "category": "ceiling_structure",
                "name": "Structură Tavan",
                "area_m2": round(ceiling_area, 2),
                "unit_price": price_ceiling,
                "cost": round(cost_ceiling, 2)
            }
        ]
    }