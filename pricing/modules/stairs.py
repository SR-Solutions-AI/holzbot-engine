# new/runner/pricing/modules/stairs.py
from __future__ import annotations


def calculate_stairs_details(
    coeffs: dict,
    total_floors: int,
    stairs_floors_count: int | None = None,
    stair_type: str | None = None,
    stairs_total_count: int | None = None,
) -> dict:
    """
    Cost scări: număr bucăți × (preț structură + balustradă).
    Prioritate: `stairs_total_count` = număr total deschideri „stairs” din editor (detections_review),
    apoi `stairs_floors_count` (etaje cu cel puțin o scară din măsurători raster),
    altfel max(0, total_floors - 1).
    """
    if isinstance(stairs_total_count, int) and stairs_total_count >= 0:
        num_stairs = stairs_total_count
    elif isinstance(stairs_floors_count, int) and stairs_floors_count >= 0:
        num_stairs = int(stairs_floors_count)
    else:
        num_stairs = max(0, total_floors - 1)
    
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