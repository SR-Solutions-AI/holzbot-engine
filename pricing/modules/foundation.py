def calculate_foundation_details(coeffs: dict, foundation_area_m2: float, type_foundation: str) -> dict:
    if foundation_area_m2 <= 0:
        return {"total_cost": 0.0, "detailed_items": []}
        
    price = coeffs["unit_price_per_m2"].get(type_foundation, 0.0)
    cost = foundation_area_m2 * price
    
    return {
        "total_cost": round(cost, 2),
        "detailed_items": [
            {
                "category": "foundation",
                "name": f"Fundație ({type_foundation})",
                "area_m2": round(foundation_area_m2, 2),
                "unit_price": price,
                "cost": round(cost, 2)
            }
        ]
    }