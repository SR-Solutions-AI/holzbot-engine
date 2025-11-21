# new/runner/pricing/calculator.py
from __future__ import annotations
import json
from pathlib import Path

from .config import (
    FINISH_COEFFS_FILE,
    FOUNDATION_COEFFS_FILE,
    OPENINGS_PRICES_FILE,
    SYSTEM_PREFAB_FILE,
    AREA_COEFFS_FILE,
    ELECTRICITY_COEFFS_FILE,
    HEATING_COEFFS_FILE,
    VENTILATION_COEFFS_FILE,
    SEWAGE_COEFFS_FILE,
    STAIRS_COEFFS_FILE
)

from .modules.finishes import calculate_finishes_details
from .modules.foundation import calculate_foundation_details
from .modules.openings import calculate_openings_details
from .modules.walls import calculate_walls_details
from .modules.floors import calculate_floors_details
from .modules.roof import calculate_roof_details
from .modules.utilities import calculate_utilities_details
from .modules.stairs import calculate_stairs_details


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_pricing_for_plan(
    area_data: dict,
    openings_data: list,
    frontend_input: dict,
    roof_data: dict | None = None,
    total_floors: int = 1,      # ✨ NOU
    is_ground_floor: bool = False # ✨ NOU
) -> dict:
    """
    Calculează toate costurile pentru un plan.
    """
    
    # 1. LOAD COEFICIENȚI
    finish_coeffs = load_json(FINISH_COEFFS_FILE)
    foundation_coeffs = load_json(FOUNDATION_COEFFS_FILE)
    openings_coeffs = load_json(OPENINGS_PRICES_FILE)
    system_coeffs = load_json(SYSTEM_PREFAB_FILE)
    area_coeffs = load_json(AREA_COEFFS_FILE)
    electricity_coeffs = load_json(ELECTRICITY_COEFFS_FILE)
    heating_coeffs = load_json(HEATING_COEFFS_FILE)
    ventilation_coeffs = load_json(VENTILATION_COEFFS_FILE)
    sewage_coeffs = load_json(SEWAGE_COEFFS_FILE)
    stairs_coeffs = load_json(STAIRS_COEFFS_FILE)

    # 2. EXTRAGE ARII
    walls_data = area_data.get("walls", {})
    w_int_net = float(walls_data.get("interior", {}).get("net_area_m2", 0.0))
    w_ext_net = float(walls_data.get("exterior", {}).get("net_area_m2", 0.0))
    
    surfaces = area_data.get("surfaces", {})
    foundation_area = float(surfaces.get("foundation_m2") or 0.0)
    floor_area = float(surfaces.get("floor_m2") or 0.0)
    ceiling_area = float(surfaces.get("ceiling_m2") or 0.0)

    # 3. PREFERINȚE UTILIZATOR
    sist_constr = frontend_input.get("sistemConstructiv", {})
    mat_finisaj = frontend_input.get("materialeFinisaj", {})
    performanta = frontend_input.get("performanta", {})
    
    system_constructie = sist_constr.get("tipSistem", "HOLZRAHMEN").upper()
    prefab_type = sist_constr.get("gradPrefabricare", "PANOURI").upper()
    foundation_type = sist_constr.get("tipFundatie", "Placă")
    
    finish_int = mat_finisaj.get("finisajInterior", "Tencuială")
    finish_ext = mat_finisaj.get("fatada", "Tencuială")
    material_tamplarie = mat_finisaj.get("tamplarie", "PVC")
    
    energy_level = performanta.get("nivelEnergetic", "Standard")
    heating_type = performanta.get("incalzire", "Gaz")
    has_ventilation = performanta.get("ventilatie", False)
    
    # 4. CALCULE COMPONENTE
    
    cost_walls = calculate_walls_details(
        system_coeffs, w_int_net, w_ext_net,
        system=system_constructie, prefab_type=prefab_type
    )
    
    cost_finishes = calculate_finishes_details(
        finish_coeffs, w_int_net, w_ext_net,
        type_int=finish_int, type_ext=finish_ext
    )
    
    cost_foundation = calculate_foundation_details(
        foundation_coeffs, foundation_area,
        type_foundation=foundation_type
    )
    
    cost_openings = calculate_openings_details(
        openings_coeffs, openings_data,
        material=material_tamplarie
    )
    
    cost_floors_ceilings = calculate_floors_details(
        area_coeffs, floor_area, ceiling_area
    )

    if roof_data:
        cost_roof = calculate_roof_details(roof_data)
    else:
        cost_roof = {"total_cost": 0.0, "detailed_items": []}
    
    cost_utilities = calculate_utilities_details(
        electricity_coeffs,
        heating_coeffs,
        ventilation_coeffs,
        sewage_coeffs,
        total_floor_area_m2=floor_area,
        energy_level=energy_level,
        heating_type=heating_type,
        has_ventilation=has_ventilation,
        has_sewage=True 
    )

    # ✨ CALCUL SCĂRI (DOAR DACĂ ESTE PARTER)
    # Asta asigură că dacă avem 3 planuri, scările (3-1=2) sunt adăugate o singură dată la total,
    # atașate de planul parterului.
    if is_ground_floor:
        cost_stairs = calculate_stairs_details(stairs_coeffs, total_floors)
    else:
        cost_stairs = {"total_cost": 0.0, "detailed_items": []}

    # 5. TOTAL
    total_plan_cost = (
        cost_walls["total_cost"] +
        cost_finishes["total_cost"] +
        cost_foundation["total_cost"] +
        cost_openings["total_cost"] +
        cost_floors_ceilings["total_cost"] +
        cost_roof["total_cost"] +
        cost_utilities["total_cost"] +
        cost_stairs["total_cost"]
    )

    # 6. RETURNARE
    return {
        "total_cost_eur": round(total_plan_cost, 2),
        "currency": "EUR",
        "breakdown": {
            "foundation": cost_foundation,
            "structure_walls": cost_walls,
            "floors_ceilings": cost_floors_ceilings,
            "roof": cost_roof,
            "openings": cost_openings,
            "finishes": cost_finishes,
            "utilities": cost_utilities,
            "stairs": cost_stairs
        },
        "inputs": {
            "referinta": frontend_input.get("referinta", ""),
            "system_constructie": system_constructie,
            "prefab_type": prefab_type,
            "foundation_type": foundation_type,
            "finish_interior": finish_int,
            "finish_exterior": finish_ext,
            "material_tamplarie": material_tamplarie,
            "energy_level": energy_level,
            "heating_type": heating_type,
            "has_ventilation": has_ventilation,
            "total_floors_in_project": total_floors
        }
    }