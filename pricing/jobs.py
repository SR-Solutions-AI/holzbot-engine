# new/runner/pricing/jobs.py
from __future__ import annotations
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List

from config.settings import load_plan_infos, PlansListError, PlanInfo
from .calculator import calculate_pricing_for_plan

STAGE_NAME = "pricing"

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
    frontend_data: dict, 
    total_plans: int
) -> PricingJobResult:
    
    work_dir = plan.stage_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Inputs din etapele anterioare
    area_json = work_dir.parent.parent / "area" / plan.plan_id / "areas_calculated.json"
    openings_json = work_dir.parent.parent / "measure_objects" / plan.plan_id / "openings_all.json"
    roof_json = work_dir.parent.parent / "roof" / plan.plan_id / "roof_estimation.json"
    
    if not area_json.exists():
        return PricingJobResult(plan.plan_id, work_dir, False, "Missing areas_calculated.json")
    
    try:
        with open(area_json, "r", encoding="utf-8") as f: area_data = json.load(f)
        
        # Identificăm dacă e parter (folosind clasificarea din area_data sau din metadata)
        # area_data["floor_type"] vine din etapa area
        floor_type = area_data.get("floor_type", "unknown")
        is_ground_floor = ("ground" in floor_type or "parter" in floor_type)
        
        # Dacă avem un singur plan, îl tratăm ca parter pentru a pune scările (dacă formula permite, deși 1-1=0)
        if total_plans == 1:
            is_ground_floor = True

        openings_data = []
        if openings_json.exists():
            with open(openings_json, "r", encoding="utf-8") as f: openings_data = json.load(f)
            
        roof_data = None
        if roof_json.exists():
            with open(roof_json, "r", encoding="utf-8") as f: roof_data = json.load(f)
        
        # Calculăm prețul
        result = calculate_pricing_for_plan(
            area_data=area_data, 
            openings_data=openings_data, 
            frontend_input=frontend_data, 
            roof_data=roof_data,
            total_floors=total_plans,     # ✨ Pasăm numărul total
            is_ground_floor=is_ground_floor # ✨ Pasăm dacă e parter
        )
        
        out_file = work_dir / "pricing_raw.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        msg = f"Cost brut: {result['total_cost_eur']:,.0f} EUR"
        
        return PricingJobResult(
            plan_id=plan.plan_id, 
            work_dir=work_dir, 
            success=True, 
            message=msg, 
            total_cost=result['total_cost_eur'],
            result_data=result
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return PricingJobResult(plan.plan_id, work_dir, False, str(e))

def run_pricing_for_run(run_id: str, max_parallel: int | None = None, frontend_data_override: dict = None) -> List[PricingJobResult]:
    try:
        plans = load_plan_infos(run_id, stage_name=STAGE_NAME)
    except PlansListError as e:
        print(f"❌ [{STAGE_NAME}] {e}")
        return []
        
    # Folosim datele pasate din orchestrator, altfel {}
    frontend_data = frontend_data_override if frontend_data_override is not None else {}
    
    total_plans = len(plans)
    print(f"\n💰 [{STAGE_NAME}] Calcul costuri (brut) pentru {total_plans} planuri...")
    
    max_parallel = max_parallel or min(os.cpu_count() or 4, total_plans)
    results = []
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(_run_for_single_plan, run_id, plan, frontend_data, total_plans): plan 
            for plan in plans
        }
        
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if res.success:
                print(f"   ✅ {res.plan_id}: {res.message}")
            else:
                print(f"   ❌ {res.plan_id}: {res.message}")
            
    return results