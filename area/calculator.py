# new/runner/area/calculator.py
from __future__ import annotations

import json
from typing import Dict

from .config import (
    STANDARD_WALL_HEIGHT_M,
    STANDARD_DOOR_HEIGHT_M,
    STANDARD_WINDOW_HEIGHT_M,
    WALL_THICKNESS_EXTERIOR_M,
    WALL_THICKNESS_INTERIOR_M,
    WALL_HEIGHT_EXTRA_STRUCTURE_AND_EXT_FINISH_M,
)

STRUCTURE_OPENING_AREA_THRESHOLD_M2 = 2.50


def _infer_window_height_from_width(width_m: float, obj_type: str = "") -> float:
    """
    Height heuristic used when no explicit height is available from editor/raster.
    """
    t = str(obj_type or "").lower()
    if "double" in t:
        return 2.0
    if width_m >= 2.0:
        return 2.0
    if width_m >= 1.35:
        return 1.5
    return 1.0


def calculate_areas_for_plan(
    plan_id: str,
    floor_type: str,
    area_net_m2: float,       # Input direct din CubiCasa
    area_gross_m2: float,     # Input direct din CubiCasa
    walls_measurements: dict,
    openings_all: list,
    stairs_area_m2: float | None,
    is_single_plan: bool,
    frontend_data: dict | None = None,
    is_top_floor: bool = False,
    floor_height_m_by_option: dict | None = None,
) -> dict:
    """
    Folosește direct valorile de arie (Net și Gross) furnizate, fără a le recalcula 
    pe baza grosimii pereților.
    """
    
    # 1. PEREȚI (Lungimi și Arii Verticale)
    # Acestea rămân calculate deoarece CubiCasa dă doar lungimea liniară, 
    # iar noi avem nevoie de suprafața verticală (mp) pentru deviz.
    avg = walls_measurements.get("estimations", {}).get("average_result", {})
    
    # Lungimi pentru finisaje (din outline)
    interior_length_m_finish = float(avg.get("interior_meters", 0.0))  # Outline verde
    exterior_length_m = float(avg.get("exterior_meters", 0.0))  # Outline albastru
    
    # Lungimi pentru structură (din skeleton)
    interior_length_m_structure = float(avg.get("interior_meters_structure", avg.get("interior_meters", 0.0)))  # Skeleton interior
    # Exterior rămâne același (outline albastru) pentru structură
    
    # Determinăm înălțimea pereților: din Preisdatenbank (floor_height_m_by_option) sau fallback din eticheta din formular
    wall_height_m = STANDARD_WALL_HEIGHT_M  # Default
    if frontend_data:
        inaltime_etaje = frontend_data.get("inaltimeEtaje", "")
        if floor_height_m_by_option and inaltime_etaje and inaltime_etaje in floor_height_m_by_option:
            wall_height_m = float(floor_height_m_by_option[inaltime_etaje])
        elif "Komfort" in inaltime_etaje or "2,70" in inaltime_etaje:
            wall_height_m = 2.70
        elif "Hoch" in inaltime_etaje or "2,85" in inaltime_etaje:
            wall_height_m = 2.85
        else:
            wall_height_m = 2.50

        # Pentru ultimul etaj (mansardă), folosim înălțimea pereților mansardei dacă e specificată
        if is_top_floor:
            inaltime_pereti_mansarda = frontend_data.get("inaltimePeretiMansarda")
            if inaltime_pereti_mansarda and float(inaltime_pereti_mansarda) > 0:
                wall_height_m = float(inaltime_pereti_mansarda)

    _h_extra = float(WALL_HEIGHT_EXTRA_STRUCTURE_AND_EXT_FINISH_M)
    # Finisaj interior: strict înălțimea din formular. Structură int/ext + finisaj exterior: +18 cm.
    h_finish_interior = wall_height_m
    h_structure_and_ext_finish = wall_height_m + _h_extra
    
    # Arii brute pentru finisaje (folosim outline)
    interior_finish_exterior_length_m = max(0.0, exterior_length_m)
    interior_finish_interior_length_m = max(0.0, interior_length_m_finish - interior_finish_exterior_length_m)
    interior_walls_gross_m2_finish = interior_length_m_finish * h_finish_interior
    interior_finish_interior_walls_gross_m2 = interior_finish_interior_length_m * h_finish_interior
    interior_finish_exterior_walls_gross_m2 = interior_finish_exterior_length_m * h_finish_interior
    exterior_walls_gross_m2 = exterior_length_m * h_structure_and_ext_finish
    
    # Arii brute pentru structură (skeleton interior; înălțime cu același adaos ca structura exterioară)
    interior_walls_gross_m2_structure = interior_length_m_structure * h_structure_and_ext_finish
    
    # Calculăm footprint-ul doar informativ (nu îl mai folosim la ariile casei)
    # Folosim lungimea skeleton pentru structură interior
    walls_footprint_m2 = (
        (interior_length_m_structure * WALL_THICKNESS_INTERIOR_M) +
        (exterior_length_m * WALL_THICKNESS_EXTERIOR_M)
    )
    
    # 2. DESCHIDERI (Pentru scăderea din pereți)
    area_windows_m2 = 0.0
    area_doors_interior_m2 = 0.0
    area_doors_exterior_m2 = 0.0
    area_windows_structure_m2 = 0.0
    area_doors_interior_structure_m2 = 0.0
    area_doors_exterior_structure_m2 = 0.0
    counts = {"windows": 0, "doors_interior": 0, "doors_exterior": 0}
    
    # Determinăm înălțimile; pentru ferestre NU mai folosim bodentiefe din formular.
    # Prioritate: height_m din detecție/editor, fallback pe euristică din lățime.
    window_height_m_default = STANDARD_WINDOW_HEIGHT_M  # fallback final
    
    for opening in openings_all:
        obj_type = opening.get("type", "").lower()
        width_m = float(opening.get("width_m", 0.0))
        if width_m <= 0:
            continue
        
        if "window" in obj_type:
            explicit_h = opening.get("height_m")
            if explicit_h is not None and float(explicit_h) > 0:
                win_h = float(explicit_h)
            else:
                win_h = _infer_window_height_from_width(width_m, obj_type) or window_height_m_default
            area = width_m * win_h
            area_windows_m2 += area
            if area >= STRUCTURE_OPENING_AREA_THRESHOLD_M2:
                area_windows_structure_m2 += area
            counts["windows"] += 1
        elif obj_type == "garage_door":
            explicit_h = opening.get("height_m")
            gh = float(explicit_h) if explicit_h is not None and float(explicit_h) > 0 else 2.1
            area = width_m * gh
            area_doors_exterior_m2 += area
            if area >= STRUCTURE_OPENING_AREA_THRESHOLD_M2:
                area_doors_exterior_structure_m2 += area
            counts["doors_exterior"] += 1
        elif "door" in obj_type:
            explicit_h = opening.get("height_m")
            dh = float(explicit_h) if explicit_h is not None and float(explicit_h) > 0 else STANDARD_DOOR_HEIGHT_M
            area = width_m * dh
            status = opening.get("status", "").lower()
            if status == "exterior":
                area_doors_exterior_m2 += area
                if area >= STRUCTURE_OPENING_AREA_THRESHOLD_M2:
                    area_doors_exterior_structure_m2 += area
                counts["doors_exterior"] += 1
            else:
                area_doors_interior_m2 += area
                if area >= STRUCTURE_OPENING_AREA_THRESHOLD_M2:
                    area_doors_interior_structure_m2 += area
                counts["doors_interior"] += 1

    # Arii pereți nete (Vertical)
    # Pentru exterior: folosim outline (pentru finisaje și structură)
    exterior_openings_m2_finish = area_windows_m2 + area_doors_exterior_m2
    exterior_openings_m2_structure = area_windows_structure_m2 + area_doors_exterior_structure_m2
    exterior_walls_net_m2 = max(0.0, exterior_walls_gross_m2 - exterior_openings_m2_finish)
    
    # Pentru interior finisaje: folosim outline verde
    interior_walls_net_m2_finish = max(0.0, interior_walls_gross_m2_finish - area_doors_interior_m2)
    interior_finish_interior_walls_net_m2 = max(0.0, interior_finish_interior_walls_gross_m2 - area_doors_interior_m2)
    # Innenausbau Außenwände: same interior finish height, but subtract only structural exterior openings (>2.5 m²).
    interior_finish_exterior_walls_net_m2 = max(0.0, interior_finish_exterior_walls_gross_m2 - exterior_openings_m2_structure)
    
    # Pentru interior structură: folosim skeleton (nu scădem deschiderile pentru structură)
    # Structura pereților interiori se calculează pe baza skeleton-ului, fără să scădem deschiderile
    interior_walls_net_m2_structure = max(0.0, interior_walls_gross_m2_structure - area_doors_interior_structure_m2)
    exterior_walls_net_m2_structure = max(0.0, exterior_walls_gross_m2 - exterior_openings_m2_structure)
    
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
        
        # Înălțime pereți (folosită la calcul, pentru PDF / rapoarte)
        "wall_height_m": round(wall_height_m, 2),
        
        # Informativ
        "input_net_area_m2": round(area_net_m2, 2),
        "input_gross_area_m2": round(area_gross_m2, 2),
        "walls_footprint_computed_m2": round(walls_footprint_m2, 2),
        "stairs_area_m2": round(stairs_area_m2, 2) if stairs_area_m2 else None,
        
        "walls": {
            "interior": {
                "length_m": round(interior_length_m_finish, 2),  # Pentru finisaje (outline)
                "length_m_structure": round(interior_length_m_structure, 2),  # Pentru structură (skeleton)
                "gross_area_m2": round(interior_walls_gross_m2_finish, 2),  # Pentru finisaje
                "gross_area_m2_structure": round(interior_walls_gross_m2_structure, 2),  # Pentru structură
                "finish_interior_walls_length_m": round(interior_finish_interior_length_m, 2),
                "finish_exterior_walls_length_m": round(interior_finish_exterior_length_m, 2),
                "gross_area_m2_finish_interior_walls": round(interior_finish_interior_walls_gross_m2, 2),
                "gross_area_m2_finish_exterior_walls": round(interior_finish_exterior_walls_gross_m2, 2),
                "openings_area_m2": round(area_doors_interior_m2, 2),
                "openings_area_m2_structure": round(area_doors_interior_structure_m2, 2),
                "net_area_m2": round(interior_walls_net_m2_finish, 2),  # Pentru finisaje
                "net_area_m2_structure": round(interior_walls_net_m2_structure, 2),  # Pentru structură
                "net_area_m2_finish_interior_walls": round(interior_finish_interior_walls_net_m2, 2),
                "net_area_m2_finish_exterior_walls": round(interior_finish_exterior_walls_net_m2, 2),
            },
            "exterior": {
                "length_m": round(exterior_length_m, 2),
                "gross_area_m2": round(exterior_walls_gross_m2, 2),
                "openings_area_m2": round(exterior_openings_m2_finish, 2),
                "openings_area_m2_structure": round(exterior_openings_m2_structure, 2),
                "net_area_m2": round(exterior_walls_net_m2, 2),
                "net_area_m2_structure": round(exterior_walls_net_m2_structure, 2)
            }
        },
        
        "surfaces": {
            "foundation_m2": round(foundation_m2, 2) if foundation_m2 > 0 else None,
            "floor_m2": round(floor_m2, 2),
            "ceiling_m2": round(ceiling_m2, 2),
            "roof_m2": round(roof_m2, 2) if roof_m2 > 0 else None,
        }
    }