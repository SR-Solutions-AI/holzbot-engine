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
    floor_type: str,  # ground_floor | top_floor | intermediate
    house_area_m2: float,
    walls_measurements: dict,
    openings_all: list,
    stairs_area_m2: float | None,
    is_single_plan: bool
) -> dict:
    """
    Calculează toate ariile pentru UN singur plan.
    Include logică specială pentru Single Plan (are și fundație și acoperiș).
    """
    
    # ==========================================
    # EXTRAGERE DATE PEREȚI
    # ==========================================
    avg = walls_measurements.get("estimations", {}).get("average_result", {})
    interior_length_m = float(avg.get("interior_meters", 0.0))
    exterior_length_m = float(avg.get("exterior_meters", 0.0))
    
    # ==========================================
    # CALCUL ARII PEREȚI BRUT (Vertical)
    # ==========================================
    interior_walls_gross_m2 = interior_length_m * STANDARD_WALL_HEIGHT_M
    exterior_walls_gross_m2 = exterior_length_m * STANDARD_WALL_HEIGHT_M
    
    # ==========================================
    # CALCUL AMPRENTĂ PEREȚI (Orizontal - Footprint)
    # ==========================================
    walls_footprint_m2 = (
        (interior_length_m * WALL_THICKNESS_INTERIOR_M) +
        (exterior_length_m * WALL_THICKNESS_EXTERIOR_M)
    )
    
    # ==========================================
    # CALCUL ARII DESCHIDERI
    # ==========================================
    area_windows_m2 = 0.0
    area_doors_interior_m2 = 0.0
    area_doors_exterior_m2 = 0.0
    
    counts = {
        "windows": 0,
        "doors_interior": 0,
        "doors_exterior": 0
    }
    
    for opening in openings_all:
        obj_type = opening.get("type", "").lower()
        status = opening.get("status", "").lower()
        width_m = float(opening.get("width_m", 0.0))
        
        if width_m <= 0:
            continue
        
        # Determinăm înălțimea
        if "window" in obj_type:
            height_m = STANDARD_WINDOW_HEIGHT_M
            area = width_m * height_m
            area_windows_m2 += area
            counts["windows"] += 1
        
        elif "door" in obj_type:
            height_m = STANDARD_DOOR_HEIGHT_M
            area = width_m * height_m
            
            if status == "exterior":
                area_doors_exterior_m2 += area
                counts["doors_exterior"] += 1
            else:
                area_doors_interior_m2 += area
                counts["doors_interior"] += 1
    
    # ==========================================
    # CALCUL ARII PEREȚI NET (Vertical)
    # ==========================================
    exterior_walls_net_m2 = max(0.0, exterior_walls_gross_m2 - area_windows_m2 - area_doors_exterior_m2)
    interior_walls_net_m2 = max(0.0, interior_walls_gross_m2 - area_doors_interior_m2)
    
    # ==========================================
    # LOGICA SUPRAFEȚE UTILE (PODEA / TAVAN)
    # ==========================================
    
    # Dacă e single plan, e logic "ground_floor", deci raw_floor_area = house_area_m2
    # Dacă e multi-plan, la etaje se scade scara.
    if floor_type == "ground_floor" or is_single_plan:
        raw_floor_area = house_area_m2
    else:
        raw_floor_area = house_area_m2 - (stairs_area_m2 or 0.0)
        
    useful_floor_area = max(0.0, raw_floor_area - walls_footprint_m2)
    
    floor_m2 = useful_floor_area
    ceiling_m2 = useful_floor_area
    
    # ==========================================
    # LOGICA FUNDAȚIE & ACOPERIȘ (CORRECTED)
    # ==========================================
    
    # Are fundație dacă e parter SAU dacă e singurul plan din proiect
    has_foundation = (floor_type == "ground_floor") or is_single_plan
    foundation_m2 = house_area_m2 if has_foundation else 0.0
    
    # Are acoperiș dacă e ultimul etaj SAU dacă e singurul plan din proiect
    has_roof = (floor_type == "top_floor") or is_single_plan
    roof_m2 = house_area_m2 if has_roof else 0.0
    
    # ==========================================
    # STRUCTURĂ REZULTAT
    # ==========================================
    result = {
        "plan_id": plan_id,
        "floor_type": floor_type,
        "is_single_plan": is_single_plan,
        "house_area_m2": round(house_area_m2, 2),
        "walls_footprint_m2": round(walls_footprint_m2, 2),
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
                "windows_area_m2": round(area_windows_m2, 2),
                "doors_area_m2": round(area_doors_exterior_m2, 2),
                "net_area_m2": round(exterior_walls_net_m2, 2)
            }
        },
        
        "openings_counts": counts,
        
        "surfaces": {
            "foundation_m2": round(foundation_m2, 2) if foundation_m2 > 0 else None,
            "floor_m2": round(floor_m2, 2),
            "ceiling_m2": round(ceiling_m2, 2),
            "roof_m2": round(roof_m2, 2) if roof_m2 > 0 else None,
        },
        
        "notes": {
            "floor_calculation": f"Area ({raw_floor_area:.1f}) - Walls Footprint ({walls_footprint_m2:.1f})",
            "foundation": "Include (Single Plan / Ground)" if has_foundation else "N/A",
            "roof": "Include (Single Plan / Top)" if has_roof else "N/A"
        }
    }
    
    return result