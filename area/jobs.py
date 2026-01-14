# new/runner/area/jobs.py
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

from config.settings import (
    load_plan_infos,
    PlansListError,
    PlanInfo,
    OUTPUT_ROOT, 
)

from .calculator import calculate_areas_for_plan
from .aggregator import aggregate_multi_plan_areas

STAGE_NAME = "area"

@dataclass
class AreaJobResult:
    plan_id: str
    work_dir: Path
    success: bool
    message: str
    result_data: dict | None = None

def _run_for_single_plan(
    run_id: str, 
    index: int, 
    total: int, 
    plan: PlanInfo,
    is_single_plan: bool,
    frontend_data: dict | None = None,
    is_top_floor_override: bool = False
) -> AreaJobResult:
    work_dir = plan.stage_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # --- RESURSE ---
    original_name = plan.plan_image.stem
    output_root = work_dir.parent.parent
    job_root = output_root.parent.parent / "jobs" / run_id
    metadata_file = job_root / "plan_metadata" / f"{original_name}.json"
    
    # Fișiere input
    perimeter_dir = work_dir.parent.parent / "perimeter" / plan.plan_id
    walls_json = perimeter_dir / "walls_measurements_gemini.json"
    
    measure_dir = work_dir.parent.parent / "measure_objects" / plan.plan_id
    openings_json = measure_dir / "openings_all.json"
    measurements_json = measure_dir / "openings_measurements_gemini.json"
    
    # CUBICASA CACHE
    scale_dir = work_dir.parent.parent / "scale" / plan.plan_id
    cubicasa_cache = scale_dir / "cubicasa_result.json"

    if not walls_json.exists():
        return AreaJobResult(plan.plan_id, work_dir, False, f"Missing walls: {walls_json.name}")
    if not openings_json.exists():
        return AreaJobResult(plan.plan_id, work_dir, False, f"Missing openings: {openings_json.name}")
    
    try:
        print(f"[{STAGE_NAME}] ({index}/{total}) {plan.plan_id} → calculate areas (Full CubiCasa Direct)...", flush=True)
        
        # Variabile pentru arii (Net și Gross)
        area_net_m2 = 0.0
        area_gross_m2 = 0.0
        source_type = "unknown"
        
        # 1. EXTRAGERE ARII DIN CUBICASA (PRIORITATE MAXIMĂ)
        if cubicasa_cache.exists():
            try:
                with open(cubicasa_cache, "r", encoding="utf-8") as f:
                    cc_data = json.load(f)
                
                metrics = cc_data.get("measurements", {}).get("metrics", {})
                
                # Extragem valorile direct
                val_net = float(metrics.get("area_indoor_m2", 0.0))
                val_gross = float(metrics.get("gross_area_m2", 0.0)) # Presupunem cheia "gross_area_m2"
                
                # Dacă CubiCasa nu a dat gross, dar a dat net, putem face un fallback minim (sau lăsăm 0)
                # Dar conform cerinței, le folosim pe ambele.
                
                if val_net > 0:
                    area_net_m2 = val_net
                    area_gross_m2 = val_gross
                    source_type = "cubicasa_direct"
                    print(f"       🏠 CubiCasa Values: Net={area_net_m2:.2f}m², Gross={area_gross_m2:.2f}m²")
                    
            except Exception as e:
                print(f"       ⚠️ Failed to read CubiCasa cache: {e}")

        # 2. FALLBACK LA METADATA (DOAR DACĂ NU AVEM CUBICASA)
        if area_net_m2 <= 0:
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                estimated = meta.get("floor_classification", {}).get("estimated_area_m2", 0.0)
                # La fallback considerăm estimarea ca fiind Gross (vechea logică) sau Net? 
                # O tratăm ca Gross pentru siguranță și Net-ul îl derivăm (sau invers).
                # Pentru simplitate, setăm ambele la fel în fallback.
                area_gross_m2 = estimated
                area_net_m2 = estimated # Aproximare grosolană
                source_type = "metadata_fallback"
                print(f"       ℹ️ Using Metadata Area: {area_gross_m2:.2f} m²")

        if area_gross_m2 <= 0 and area_net_m2 <= 0:
            return AreaJobResult(plan.plan_id, work_dir, False, "Nu am putut determina ariile.")

        # 3. DATE SUPLIMENTARE
        floor_type = "unknown"
        is_top_floor = is_top_floor_override  # Folosim override-ul din run_area_for_run
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
                floor_type = meta_data.get("floor_classification", {}).get("floor_type", "unknown")
                # Dacă nu avem override, folosim floor_type din metadata
                if not is_top_floor:
                    is_top_floor = (floor_type == "top_floor")

        with open(walls_json, "r", encoding="utf-8") as f: walls_data = json.load(f)
        with open(openings_json, "r", encoding="utf-8") as f: openings_data = json.load(f)
        
        stairs_area_m2 = None
        if measurements_json.exists():
            with open(measurements_json, "r", encoding="utf-8") as f:
                meas_data = json.load(f)
            stairs_meas = meas_data.get("measurements", {}).get("stairs")
            if stairs_meas:
                stairs_area_m2 = float(stairs_meas.get("total_area_m2", 0.0))
        
        # 4. CALCULATOR (Cu valorile explicite)
        result = calculate_areas_for_plan(
            plan_id=plan.plan_id,
            floor_type=floor_type,
            area_net_m2=area_net_m2,       # <--- Trimitem Net
            area_gross_m2=area_gross_m2,   # <--- Trimitem Gross
            walls_measurements=walls_data,
            openings_all=openings_data,
            stairs_area_m2=stairs_area_m2,
            is_single_plan=is_single_plan,
            frontend_data=frontend_data,
            is_top_floor=is_top_floor
        )
        
        result["meta"] = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "stage": STAGE_NAME,
            "area_source": source_type
        }
        
        output_file = work_dir / "areas_calculated.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return AreaJobResult(
            plan_id=plan.plan_id, 
            work_dir=work_dir, 
            success=True, 
            message=f"Net: {area_net_m2:.1f}m², Gross: {area_gross_m2:.1f}m²", 
            result_data=result
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return AreaJobResult(plan.plan_id, work_dir, False, f"Eroare: {e}")

def run_area_for_run(run_id: str, max_parallel: int | None = None, frontend_data: dict | None = None) -> List[AreaJobResult]:
    try:
        plans: List[PlanInfo] = load_plan_infos(run_id, stage_name=STAGE_NAME)
    except PlansListError as e:
        print(f"❌ [{STAGE_NAME}] {e}")
        return []
    
    total = len(plans)
    is_single_plan = (total == 1)
    
    print(f"\n📌 [{STAGE_NAME}] {total} planuri - Mod Direct CubiCasa")
    
    if max_parallel is None:
        cpu_count = os.cpu_count() or 4
        max_parallel = min(cpu_count, total)
    
    results: List[AreaJobResult] = []
    
    # Determinăm care plan este ultimul etaj (pentru mansardă)
    lista_etaje = frontend_data.get("listaEtaje", []) if frontend_data else []
    has_mansarda = any(e == "mansarda" for e in lista_etaje)
    # Ultimul plan din listă este top_floor dacă avem mansardă
    top_floor_plan_id = plans[-1].plan_id if (has_mansarda and len(plans) > 0) else None
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(_run_for_single_plan, run_id, idx, total, plan, is_single_plan, frontend_data, plan.plan_id == top_floor_plan_id): plan
            for idx, plan in enumerate(plans, start=1)
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            status = "✅" if res.success else "❌"
            print(f"{status} [{STAGE_NAME}] {res.plan_id} → {res.message}", flush=True)
    
    successful_results = [r.result_data for r in results if r.success and r.result_data]
    if len(successful_results) > 1:
        print(f"\n📊 Agregare rezultate multi-plan...")
        summary = aggregate_multi_plan_areas(successful_results)
        output_root = OUTPUT_ROOT / run_id / STAGE_NAME
        output_root.mkdir(parents=True, exist_ok=True)
        with open(output_root / "areas_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return results