# pricing/jobs.py
from __future__ import annotations
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List

from config.settings import load_plan_infos, PlansListError, PlanInfo, get_run_dir
from .calculator import calculate_pricing_for_plan
from .db_loader import fetch_pricing_parameters
from .modes import get_pricing_mode
from area.calculator import calculate_areas_for_plan

STAGE_NAME = "pricing"

def _apply_allowed_categories(result: dict, allowed: list[str] | None) -> dict:
    """
    Filters pricing breakdown based on allowed categories coming from offer_types.allowed_pricing_categories.
    DB categories: foundation, structure, roof, floors, openings, finishes, utilities, stairs
    Result keys:    foundation, structure_walls, roof, floors_ceilings, openings, finishes, utilities, stairs
    """
    if not allowed:
        return result

    map_to_breakdown_key = {
        "foundation": "foundation",
        "structure": "structure_walls",
        "roof": "roof",
        "floors": "floors_ceilings",
        "openings": "openings",
        "finishes": "finishes",
        "utilities": "utilities",
        "stairs": "stairs",
    }

    bd = result.get("breakdown") or {}
    if not isinstance(bd, dict):
        return result

    keep_keys = {map_to_breakdown_key.get(a) for a in allowed}
    keep_keys.discard(None)

    new_bd: dict = {}
    total = 0.0
    for k, v in bd.items():
        if k not in keep_keys:
            continue
        new_bd[k] = v
        try:
            total += float(v.get("total_cost", 0.0))
        except Exception:
            pass

    # produce a new result dict (don't mutate original)
    out = dict(result)
    out["breakdown"] = new_bd
    out["total_cost_eur"] = round(total, 2)
    return out

@dataclass
class PricingJobResult:
    plan_id: str
    work_dir: Path
    success: bool
    message: str
    total_cost: float = 0.0
    result_data: dict | None = None

def _run_for_single_plan(
    run_id: str, 
    plan: PlanInfo, 
    plan_index: int,
    frontend_data: dict, 
    total_plans: int,
    pricing_coeffs: dict,
    all_plans: List[PlanInfo] = None,  # Lista tuturor planurilor pentru a calcula floor_idx
    basement_plan_index: int | None = None,  # Index (0-based) al planului ales ca beci; None = fără beci dedicat
) -> PricingJobResult:
    work_dir = plan.stage_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    roof_json = work_dir.parent.parent / "roof" / plan.plan_id / "roof_estimation.json"
    
    # ✅ FOLOSIM EXCLUSIV DATE DIN RASTERSCAN (FĂRĂ dependență de CubiCasa, FĂRĂ fallback)
    scale_dir = work_dir.parent.parent / "scale" / plan.plan_id
    raster_room_scales = scale_dir / "cubicasa_steps" / "raster_processing" / "walls_from_coords" / "room_scales.json"
    raster_openings = scale_dir / "cubicasa_steps" / "raster_processing" / "walls_from_coords" / "openings_measurements.json"
    raster_walls_measurements = scale_dir / "cubicasa_steps" / "raster_processing" / "walls_from_coords" / "walls_measurements.json"
    
    area_data = None
    openings_data_from_raster = []  # Inițializăm înainte de blocul if
    
    # ✅ PRIORITATE: Construim openings_data din raster dacă există (independenți de area_data)
    if raster_openings.exists():
        try:
            with open(raster_openings, "r", encoding="utf-8") as f:
                openings_measurements_data = json.load(f)
            
            # Structura: openings_measurements_data['openings'] este o listă de openings
            openings_list = openings_measurements_data.get("openings", [])
            
            # Convertim formatul din openings_measurements.json la formatul așteptat
            for opening in openings_list:
                opening_type = opening.get("type", "")
                width_m = opening.get("width_m", 0.0)
                
                if width_m > 0 and opening_type:
                    # Normalizăm tipul (door/double_door -> door, window/double_window -> window)
                    normalized_type = opening_type
                    if "door" in opening_type.lower():
                        normalized_type = "door" if "double" not in opening_type.lower() else "double_door"
                    elif "window" in opening_type.lower():
                        normalized_type = "window" if "double" not in opening_type.lower() else "double_window"
                    
                    # Determinăm status-ul (exterior/interior)
                    status = opening.get("status", "interior")
                    if not isinstance(status, str):
                        status = "interior"
                    status = status.lower()
                    
                    openings_data_from_raster.append({
                        "type": normalized_type,
                        "width_m": float(width_m),
                        "status": status
                    })
            
            print(f"       ✅ Folosesc {len(openings_data_from_raster)} openings din raster_processing")
        except Exception as e:
            import traceback
            print(f"       ⚠️ Eroare la citirea openings_measurements.json: {e}")
            traceback.print_exc()
    
    # ✅ PRIORITATE: Folosim datele din raster dacă există (chiar dacă areas_calculated.json există)
    # RasterScan oferă date mai precise, deci le priorităm
    use_raster_data = False  # ✅ Inițializăm variabila
    if raster_room_scales.exists():
        try:
            with open(raster_room_scales, "r", encoding="utf-8") as f:
                room_scales_data = json.load(f)
            
            total_area_m2 = room_scales_data.get('total_area_m2', 0.0)
            if total_area_m2 > 0:
                # ✅ Folosim DOAR walls_measurements din RasterScan (FĂRĂ dependență de CubiCasa)
                walls_measurements = {
                    "estimations": {
                        "average_result": {
                            "interior_meters": 0.0,
                            "exterior_meters": 0.0,
                            "interior_meters_structure": 0.0
                        }
                    }
                }
                
                if raster_walls_measurements.exists():
                    try:
                        with open(raster_walls_measurements, "r", encoding="utf-8") as f:
                            walls_measurements = json.load(f)
                        print(f"       ✅ Folosesc walls_measurements din RasterScan (walls_measurements.json)")
                    except Exception as e:
                        print(f"       ⚠️ Eroare la citirea walls_measurements.json: {e}")
                else:
                    print(f"       ⚠️ walls_measurements.json nu există, folosesc valori default (0.0)")
                
                # Încercăm să citim floor_type din metadata
                floor_type = "ground_floor"  # Default
                job_root = work_dir.parent.parent.parent.parent  # Navigăm la job_root
                plan_metadata_dir = job_root / "plan_metadata"
                if plan_metadata_dir.exists():
                    # Căutăm metadata pentru acest plan
                    for meta_file in plan_metadata_dir.glob("*.json"):
                        if meta_file.name != "_floor_classification_summary.json":
                            try:
                                with open(meta_file, "r", encoding="utf-8") as f:
                                    meta_data = json.load(f)
                                floor_class = meta_data.get("floor_classification", {})
                                if floor_class.get("floor_type"):
                                    floor_type = floor_class["floor_type"]
                                    break
                            except Exception:
                                pass
                
                # ✅ Construim area_data complet folosind calculate_areas_for_plan
                # Aceasta va construi structura completă cu walls, surfaces, etc.
                is_single_plan = (total_plans == 1)
                is_top_floor_plan = ("top" in floor_type.lower() or "mansard" in floor_type.lower())
                
                area_data = calculate_areas_for_plan(
                    plan_id=plan.plan_id,
                    floor_type=floor_type,
                    area_net_m2=float(total_area_m2),
                    area_gross_m2=float(total_area_m2),
                    walls_measurements=walls_measurements,
                    openings_all=openings_data_from_raster,
                    stairs_area_m2=None,  # Nu avem date despre scări din raster
                    is_single_plan=is_single_plan,
                    frontend_data=frontend_data,
                    is_top_floor=is_top_floor_plan
                )
                
                use_raster_data = True
                print(f"       ✅ Folosesc datele din raster_processing pentru pricing: {total_area_m2:.2f} m² (floor_type: {floor_type})")
        except Exception as e:
            print(f"       ⚠️ Eroare la citirea room_scales.json: {e}")
    
    # Dacă nu am putut folosi raster, încercăm areas_calculated.json
    if not use_raster_data:
        area_json = work_dir / "areas_calculated.json"
        if not area_json.exists():
            return PricingJobResult(plan.plan_id, work_dir, False, "Missing areas_calculated.json")
        
        try:
            with open(area_json, "r", encoding="utf-8") as f: 
                area_data = json.load(f)
        except Exception as e:
            return PricingJobResult(plan.plan_id, work_dir, False, f"Error reading areas_calculated.json: {e}")
    
    # Extragem datele comune (pentru ambele cazuri: raster sau areas_calculated.json)
    try:
        floor_type = area_data.get("floor_type", "unknown")
        is_ground_floor = ("ground" in floor_type or "parter" in floor_type) or (total_plans == 1)
        is_top_floor = ("top" in floor_type.lower() or "mansard" in floor_type.lower())

        # ✅ PRIORITATE: Folosim openings din raster dacă există (FĂRĂ fallback)
        openings_data = []
        if len(openings_data_from_raster) > 0:
            # openings_data_from_raster a fost deja construit mai sus (din RasterScan)
            openings_data = openings_data_from_raster
            print(f"       ✅ Folosesc {len(openings_data)} openings din raster_processing pentru pricing")
        else:
            # Dacă nu avem openings din RasterScan, folosim lista goală (nu mai există fallback)
            openings_data = []
            print(f"       ⚠️ Nu am openings din RasterScan, folosesc lista goală")
            
        roof_data = None
        if roof_json.exists():
            with open(roof_json, "r", encoding="utf-8") as f: roof_data = json.load(f)
        
        # Calculăm floor_idx pentru etajele intermediare
        # Numărăm câte etaje intermediare sunt înainte de acest plan
        intermediate_floor_index = 0
        if not is_ground_floor and not is_top_floor and all_plans:
            # Este etaj intermediar - numărăm câte etaje intermediare sunt înainte
            for idx, p in enumerate(all_plans):
                if idx >= plan_index:
                    break
                # Verificăm tipul etajului pentru planul anterior din plan_metadata (FĂRĂ fallback la areas_calculated.json)
                job_root = work_dir.parent.parent.parent.parent
                plan_metadata_dir = job_root / "plan_metadata"
                if plan_metadata_dir.exists():
                    for meta_file in plan_metadata_dir.glob("*.json"):
                        if meta_file.name != "_floor_classification_summary.json":
                            try:
                                with open(meta_file, "r", encoding="utf-8") as f:
                                    prev_meta_data = json.load(f)
                                prev_floor_class = prev_meta_data.get("floor_classification", {})
                                prev_floor_type = prev_floor_class.get("floor_type", "unknown").lower()
                                prev_is_ground = ("ground" in prev_floor_type or "parter" in prev_floor_type)
                                prev_is_top = ("top" in prev_floor_type or "mansard" in prev_floor_type)
                                if not prev_is_ground and not prev_is_top:
                                    intermediate_floor_index += 1
                                    break
                            except Exception:
                                pass
        
        is_basement_plan = basement_plan_index is not None and plan_index == basement_plan_index
        has_dedicated_basement_plan = basement_plan_index is not None
        result = calculate_pricing_for_plan(
            area_data=area_data, 
            openings_data=openings_data, 
            frontend_input=frontend_data, 
            pricing_coeffs=pricing_coeffs,
            roof_data=roof_data,
            total_floors=total_plans,
            is_ground_floor=is_ground_floor,
            plan_index=plan_index,
            intermediate_floor_index=intermediate_floor_index,  # Indexul etajului intermediar (1, 2, 3, etc.)
            is_basement_plan=is_basement_plan,
            has_dedicated_basement_plan=has_dedicated_basement_plan,
        )

        allowed = frontend_data.get("allowed_pricing_categories")
        if isinstance(allowed, list):
            result = _apply_allowed_categories(result, allowed)
        
        out_file = work_dir / "pricing_raw.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        return PricingJobResult(
            plan_id=plan.plan_id, 
            work_dir=work_dir, 
            success=True, 
            message=f"Cost brut: {result['total_cost_eur']:,.0f} EUR", 
            total_cost=result['total_cost_eur'],
            result_data=result
        )
    except Exception as e:
        return PricingJobResult(plan.plan_id, work_dir, False, str(e))

def run_pricing_for_run(run_id: str, max_parallel: int | None = None, frontend_data_override: dict = None) -> List[PricingJobResult]:
    try:
        plans = load_plan_infos(run_id, stage_name=STAGE_NAME)
    except PlansListError as e:
        return []
        
    frontend_data = frontend_data_override if frontend_data_override is not None else {}
    
    # Tenant slug is required (sent by API). No hard-coded fallbacks.
    tenant_slug = frontend_data.get("tenant_slug")
    if not tenant_slug:
        raise ValueError("Missing required frontend_data.tenant_slug (sent by API)")

    calc_mode = frontend_data.get("calc_mode") or "default"
    pricing_mode = get_pricing_mode(calc_mode)
    frontend_data = pricing_mode.normalize_frontend_input(frontend_data)
    
    try:
        print(f"🌍 [{STAGE_NAME}] Fetching pricing params for Tenant: {tenant_slug} (mode={pricing_mode.key})...")
        pricing_coeffs = fetch_pricing_parameters(tenant_slug, calc_mode=calc_mode)
    except Exception as e:
        print(f"❌ [{STAGE_NAME}] DB Error: {e}")
        return []

    total_plans = len(plans)
    basement_plan_index = None
    try:
        run_dir = get_run_dir(run_id)
        bp_file = run_dir / "basement_plan_id.json"
        if bp_file.exists():
            bp_data = json.loads(bp_file.read_text(encoding="utf-8"))
            basement_plan_index = bp_data.get("basement_plan_index")
            if basement_plan_index is not None:
                print(f"   📋 [PRICING] Beci dedicat: plan index {basement_plan_index}")
    except Exception as e:
        print(f"   ⚠️ [PRICING] Nu am putut încărca basement_plan_id.json: {e}")

    max_parallel = max_parallel or min(os.cpu_count() or 4, total_plans)
    results = []
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(_run_for_single_plan, run_id, plan, idx, frontend_data, total_plans, pricing_coeffs, plans, basement_plan_index): plan 
            for idx, plan in enumerate(plans)
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(f"   {'✅' if res.success else '❌'} {res.plan_id}: {res.message}")
            
    return results