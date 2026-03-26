# new/runner/pricing/modules/stairs.py
from __future__ import annotations


def calculate_stairs_details(coeffs: dict, total_floors: int, stairs_floors_count: int | None = None, stair_type: str | None = None) -> dict:
    """
    Calculează costul scărilor bazat pe numărul de etaje.
    Formula: (Număr Etaje - 1) * Preț Unitar.
    
    Exemplu:
      - 1 etaj (Parter) -> 0 scări
      - 2 etaje (P+1) -> 1 scară
      - 3 etaje (P+2) -> 2 scări
    """
    # Numărul de scări: etaje care conțin efectiv "stairs" (dacă e disponibil), altfel fallback clasic.
    num_stairs = int(stairs_floors_count) if isinstance(stairs_floors_count, int) and stairs_floors_count >= 0 else max(0, total_floors - 1)
    
    if num_stairs == 0:
        return {"total_cost": 0.0, "detailed_items": []}
    
    # Prețuri unitare din JSON. Tipul selectat în formular poate suprascrie prețul structurii.
    stair_type = (stair_type or "Standard").strip() or "Standard"
    type_price_map = coeffs.get("stair_type_price_map") or {}
    if isinstance(type_price_map, dict) and stair_type in type_price_map:
        price_per_unit = float(type_price_map.get(stair_type, 4500.0))
    else:
        price_per_unit = float(coeffs.get("price_per_stair_unit", 4500.0))
    price_railing = float(coeffs.get("railing_price_per_stair", 800.0))
    
    full_price_per_stair = price_per_unit + price_railing
    total_cost = num_stairs * full_price_per_stair
    
    return {
        "total_cost": round(total_cost, 2),
        "detailed_items": [
            {
                "category": "stairs_structure",
                "name": "Scară interioară (pe bucată)",
                "details": f"Tip selectat: {stair_type}",
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