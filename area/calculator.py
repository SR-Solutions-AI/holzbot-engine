# new/runner/area/calculator.py
from __future__ import annotations

import json
from typing import Dict

from .config import (
    STANDARD_WALL_HEIGHT_M,
    STANDARD_DOOR_HEIGHT_M,
    STANDARD_WINDOW_HEIGHT_M,
    WALL_THICKNESS_EXTERIOR_M,
    WALL_THICKNESS_INTERIOR_M
)


def calculate_areas_for_plan(
    plan_id: str,
    floor_type: str,
    area_net_m2: float,       # Input direct din CubiCasa
    area_gross_m2: float,     # Input direct din CubiCasa
    walls_measurements: dict,
    openings_all: list,
    stairs_area_m2: float | None,
    is_single_plan: bool
) -> dict:
    """
    Folosește direct valorile de arie (Net și Gross) furnizate, fără a le recalcula 
    pe baza grosimii pereților.
    """
    
    # 1. PEREȚI (Lungimi și Arii Verticale)
    # Acestea rămân calculate deoarece CubiCasa dă doar lungimea liniară, 
    # iar noi avem nevoie de suprafața verticală (mp) pentru deviz.
    avg = walls_measurements.get("estimations", {}).get("average_result", {})
    interior_length_m = float(avg.get("interior_meters", 0.0))
    exterior_length_m = float(avg.get("exterior_meters", 0.0))
    
    interior_walls_gross_m2 = interior_length_m * STANDARD_WALL_HEIGHT_M
    exterior_walls_gross_m2 = exterior_length_m * STANDARD_WALL_HEIGHT_M
    
    # Calculăm footprint-ul doar informativ (nu îl mai folosim la ariile casei)
    walls_footprint_m2 = (
        (interior_length_m * WALL_THICKNESS_INTERIOR_M) +
        (exterior_length_m * WALL_THICKNESS_EXTERIOR_M)
    )
    
    # 2. DESCHIDERI (Pentru scăderea din pereți)
    area_windows_m2 = 0.0
    area_doors_interior_m2 = 0.0
    area_doors_exterior_m2 = 0.0
    counts = {"windows": 0, "doors_interior": 0, "doors_exterior": 0}
    
    for opening in openings_all:
        obj_type = opening.get("type", "").lower()
        width_m = float(opening.get("width_m", 0.0))
        if width_m <= 0: continue
        
        if "window" in obj_type:
            area = width_m * STANDARD_WINDOW_HEIGHT_M
            area_windows_m2 += area
            counts["windows"] += 1
        elif "door" in obj_type:
            area = width_m * STANDARD_DOOR_HEIGHT_M
            status = opening.get("status", "").lower()
            if status == "exterior":
                area_doors_exterior_m2 += area
                counts["doors_exterior"] += 1
            else:
                area_doors_interior_m2 += area
                counts["doors_interior"] += 1

    # Arii pereți nete (Vertical)
    exterior_walls_net_m2 = max(0.0, exterior_walls_gross_m2 - area_windows_m2 - area_doors_exterior_m2)
    interior_walls_net_m2 = max(0.0, interior_walls_gross_m2 - area_doors_interior_m2)
    
    # ==========================================
    # 3. SUPRAFEȚE ORIZONTALE (Direct Assignment)
    # ==========================================
    
    # A. PODEA / TAVAN (Folosim NET Area)
    # Scădem scara doar dacă nu e parter (sau dacă e single plan considerăm totul util, 
    # depinde de logica dorită. De obicei scara se scade din etaj).
    
    if floor_type == "ground_floor" or is_single_plan:
        floor_m2 = area_net_m2
    else:
        # La etaj scădem golul scării
        floor_m2 = max(0.0, area_net_m2 - (stairs_area_m2 or 0.0))
        
    ceiling_m2 = floor_m2
    
    # B. FUNDAȚIE / ACOPERIȘ (Folosim GROSS Area)
    # Se aplică doar la nivelurile relevante
    has_foundation = (floor_type == "ground_floor") or is_single_plan
    foundation_m2 = area_gross_m2 if has_foundation else 0.0
    
    has_roof = (floor_type == "top_floor") or is_single_plan
    roof_m2 = area_gross_m2 if has_roof else 0.0
    
    # ==========================================
    # 4. REZULTAT
    # ==========================================
    return {
        "plan_id": plan_id,
        "floor_type": floor_type,
        "is_single_plan": is_single_plan,
        
        # Informativ
        "input_net_area_m2": round(area_net_m2, 2),
        "input_gross_area_m2": round(area_gross_m2, 2),
        "walls_footprint_computed_m2": round(walls_footprint_m2, 2),
        "stairs_area_m2": round(stairs_area_m2, 2) if stairs_area_m2 else None,
        
        "walls": {
            "interior": {
                "length_m": round(interior_length_m, 2),
                "gross_area_m2": round(interior_walls_gross_m2, 2),
                "openings_area_m2": round(area_doors_interior_m2, 2),
                "net_area_m2": round(interior_walls_net_m2, 2)
            },
            "exterior": {
                "length_m": round(exterior_length_m, 2),
                "gross_area_m2": round(exterior_walls_gross_m2, 2),
                "openings_area_m2": round(area_windows_m2 + area_doors_exterior_m2, 2),
                "net_area_m2": round(exterior_walls_net_m2, 2)
            }
        },
        
        "surfaces": {
            "foundation_m2": round(foundation_m2, 2) if foundation_m2 > 0 else None,
            "floor_m2": round(floor_m2, 2),
            "ceiling_m2": round(ceiling_m2, 2),
            "roof_m2": round(roof_m2, 2) if roof_m2 > 0 else None,
        }
    }