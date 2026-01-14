# new/runner/area/aggregator.py
from __future__ import annotations

from typing import List


def aggregate_multi_plan_areas(plan_results: List[dict]) -> dict:
    """
    Agregare rezultate multi-plan cu sumă inteligentă.
    
    Logica:
    - Fundație: doar ground_floor (NU se sumează)
    - Acoperiș: doar top_floor (NU se sumează)
    - Pereți: SE SUMEAZĂ toate planurile
    - Podele: SE SUMEAZĂ toate planurile
    - Tavane: SE SUMEAZĂ toate planurile
    """
    
    total_walls_interior_gross = 0.0
    total_walls_interior_net = 0.0
    total_walls_exterior_gross = 0.0
    total_walls_exterior_net = 0.0
    
    total_floor = 0.0
    total_ceiling = 0.0
    
    foundation = 0.0
    roof = 0.0
    
    for plan in plan_results:
        walls = plan["walls"]
        surfaces = plan["surfaces"]
        
        # Pereți (sumează)
        total_walls_interior_gross += walls["interior"]["gross_area_m2"]
        total_walls_interior_net += walls["interior"]["net_area_m2"]
        total_walls_exterior_gross += walls["exterior"]["gross_area_m2"]
        total_walls_exterior_net += walls["exterior"]["net_area_m2"]
        
        # Podele/Tavane (sumează)
        total_floor += surfaces["floor_m2"]
        total_ceiling += surfaces["ceiling_m2"]
        
        # Fundație (doar ground_floor)
        if surfaces.get("foundation_m2"):
            foundation = surfaces["foundation_m2"]
        
        # Acoperiș (doar top_floor)
        if surfaces.get("roof_m2"):
            roof = surfaces["roof_m2"]
    
    return {
        "total_plans": len(plan_results),
        "walls": {
            "interior": {
                "gross_total_m2": round(total_walls_interior_gross, 2),
                "net_total_m2": round(total_walls_interior_net, 2)
            },
            "exterior": {
                "gross_total_m2": round(total_walls_exterior_gross, 2),
                "net_total_m2": round(total_walls_exterior_net, 2)
            }
        },
        "surfaces": {
            "foundation_m2": round(foundation, 2),
            "floor_total_m2": round(total_floor, 2),
            "ceiling_total_m2": round(total_ceiling, 2),
            "roof_m2": round(roof, 2)
        },
        "breakdown_by_plan": plan_results
    }