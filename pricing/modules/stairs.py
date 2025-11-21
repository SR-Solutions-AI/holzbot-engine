# new/runner/pricing/modules/stairs.py
from __future__ import annotations

def calculate_stairs_details(coeffs: dict, total_floors: int) -> dict:
    """
    Calculează costul scărilor bazat pe numărul de etaje.
    Formula: (Număr Etaje - 1) * Preț Unitar.
    
    Exemplu:
      - 1 etaj (Parter) -> 0 scări
      - 2 etaje (P+1) -> 1 scară
      - 3 etaje (P+2) -> 2 scări
    """
    # Calculăm numărul de scări necesare
    num_stairs = max(0, total_floors - 1)
    
    if num_stairs == 0:
        return {"total_cost": 0.0, "detailed_items": []}
    
    # Prețuri unitare din JSON
    price_per_unit = float(coeffs.get("price_per_stair_unit", 4500.0))
    price_railing = float(coeffs.get("railing_price_per_stair", 800.0))
    
    full_price_per_stair = price_per_unit + price_railing
    total_cost = num_stairs * full_price_per_stair
    
    return {
        "total_cost": round(total_cost, 2),
        "detailed_items": [
            {
                "category": "stairs_structure",
                "name": "Scară interioară completă (Structură + Finisaj)",
                "details": "Include trepte, contratrepte și structură rezistență",
                "quantity": num_stairs,
                "unit": "buc",
                "unit_price": price_per_unit,
                "cost": round(num_stairs * price_per_unit, 2)
            },
            {
                "category": "stairs_railing",
                "name": "Balustradă scară",
                "details": "Mână curentă și elemente siguranță",
                "quantity": num_stairs,
                "unit": "buc",
                "unit_price": price_railing,
                "cost": round(num_stairs * price_railing, 2)
            }
        ]
    }