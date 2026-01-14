# pricing/jobs.py
from __future__ import annotations
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List

from config.settings import load_plan_infos, PlansListError, PlanInfo
from .calculator import calculate_pricing_for_plan
from .db_loader import fetch_pricing_parameters
from .modes import get_pricing_mode

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
    all_plans: List[PlanInfo] = None  # Lista tuturor planurilor pentru a calcula floor_idx
) -> PricingJobResult:
    work_dir = plan.stage_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    area_json = work_dir.parent.parent / "area" / plan.plan_id / "areas_calculated.json"
    openings_json = work_dir.parent.parent / "measure_objects" / plan.plan_id / "openings_all.json"
    roof_json = work_dir.parent.parent / "roof" / plan.plan_id / "roof_estimation.json"
    
    if not area_json.exists():
        return PricingJobResult(plan.plan_id, work_dir, False, "Missing areas_calculated.json")
    
    try:
        with open(area_json, "r", encoding="utf-8") as f: area_data = json.load(f)
        floor_type = area_data.get("floor_type", "unknown")
        is_ground_floor = ("ground" in floor_type or "parter" in floor_type) or (total_plans == 1)
        is_top_floor = ("top" in floor_type.lower() or "mansard" in floor_type.lower())

        openings_data = []
        if openings_json.exists():
            with open(openings_json, "r", encoding="utf-8") as f: openings_data = json.load(f)
            
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
                # Verificăm tipul etajului pentru planul anterior
                prev_area_json = p.stage_work_dir.parent.parent / "area" / p.plan_id / "areas_calculated.json"
                if prev_area_json.exists():
                    with open(prev_area_json, "r", encoding="utf-8") as f:
                        prev_area_data = json.load(f)
                    prev_floor_type = prev_area_data.get("floor_type", "unknown").lower()
                    prev_is_ground = ("ground" in prev_floor_type or "parter" in prev_floor_type)
                    prev_is_top = ("top" in prev_floor_type or "mansard" in prev_floor_type)
                    if not prev_is_ground and not prev_is_top:
                        intermediate_floor_index += 1
        
        result = calculate_pricing_for_plan(
            area_data=area_data, 
            openings_data=openings_data, 
            frontend_input=frontend_data, 
            pricing_coeffs=pricing_coeffs,
            roof_data=roof_data,
            total_floors=total_plans,
            is_ground_floor=is_ground_floor,
            plan_index=plan_index,
            intermediate_floor_index=intermediate_floor_index  # Indexul etajului intermediar (1, 2, 3, etc.)
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
    max_parallel = max_parallel or min(os.cpu_count() or 4, total_plans)
    results = []
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(_run_for_single_plan, run_id, plan, idx, frontend_data, total_plans, pricing_coeffs, plans): plan 
            for idx, plan in enumerate(plans)
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(f"   {'✅' if res.success else '❌'} {res.plan_id}: {res.message}")
            
    return results