# new/runner/measure_objects/jobs.py
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List

from config.settings import (
    load_plan_infos,
    PlansListError,
    PlanInfo,
)

from .calculator import calculate_widths_from_detections
from .aggregate import create_openings_all


STAGE_NAME = "measure_objects"


@dataclass
class MeasureObjectsJobResult:
    plan_id: str
    work_dir: Path
    success: bool
    message: str


def _run_for_single_plan(run_id: str, index: int, total: int, plan: PlanInfo) -> MeasureObjectsJobResult:
    """
    Calculează lățimile obiectelor + arii scări DIN DETECȚII (bbox × meters_per_pixel).
    """
    work_dir = plan.stage_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Input files
    count_objects_dir = work_dir.parent.parent / "count_objects" / plan.plan_id
    detections_all_json = count_objects_dir / "detections_all.json"
    
    scale_dir = work_dir.parent.parent / "scale" / plan.plan_id
    scale_json = scale_dir / "scale_result.json"
    
    exterior_doors_dir = work_dir.parent.parent / "exterior_doors" / plan.plan_id
    exterior_doors_json = exterior_doors_dir / "exterior_doors.json"
    
    # Verificări
    if not detections_all_json.exists():
        return MeasureObjectsJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=False,
            message=f"Nu găsesc {detections_all_json.name}"
        )
    
    if not scale_json.exists():
        return MeasureObjectsJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=False,
            message=f"Nu găsesc {scale_json.name}"
        )
    
    try:
        print(
            f"[{STAGE_NAME}] ({index}/{total}) {plan.plan_id} → calculate widths + stairs area + aggregate "
            f"(cwd={work_dir})",
            flush=True,
        )
        
        # ==========================================
        # STEP 1: CALCUL LĂȚIMI + ARII (din bbox + scale)
        # ==========================================
        measurements = calculate_widths_from_detections(detections_all_json, scale_json)
        
        # Salvează openings_measurements_gemini.json (păstrăm numele pentru compatibilitate)
        measurements_output = work_dir / "openings_measurements_gemini.json"
        with open(measurements_output, "w", encoding="utf-8") as f:
            json.dump(measurements, f, indent=2, ensure_ascii=False)
        
        print(f"       📄 Salvat: {measurements_output.name}")
        
        # ==========================================
        # STEP 2: AGREGARE openings_all.json
        # ==========================================
        openings_all_output = work_dir / "openings_all.json"
        
        total_openings = create_openings_all(
            detections_all_json,
            measurements_output,
            exterior_doors_json,
            openings_all_output
        )
        
        print(f"       📄 Salvat: {openings_all_output.name} ({total_openings} obiecte)")
        
        # ==========================================
        # SUMMARY MESSAGE
        # ==========================================
        measured = measurements.get("measurements", {})
        summary_parts = []
        
        for obj_type in ["door", "double_door", "window", "double_window"]:
            if obj_type in measured:
                meas = measured[obj_type]
                width = meas.get("real_width_meters")
                count = meas.get("count_measured")
                if width:
                    summary_parts.append(f"{obj_type}={width:.3f}m(n={count})")
        
        # Adaugă scări dacă există
        if "stairs" in measured:
            stairs = measured["stairs"]
            area = stairs.get("total_area_m2")
            count = stairs.get("count_measured")
            if area:
                summary_parts.append(f"stairs={area:.2f}m²(n={count})")
        
        message = (
            f"Calculat: {', '.join(summary_parts)} | "
            f"openings_all.json: {total_openings} obiecte"
        )
        
        return MeasureObjectsJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=True,
            message=message
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return MeasureObjectsJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=False,
            message=f"Eroare: {e}"
        )


def run_measure_objects_for_run(run_id: str, max_parallel: int | None = None) -> List[MeasureObjectsJobResult]:
    """
    Punct de intrare pentru etapa „measure_objects".
    
    Calculează:
    - Lățimi uși/ferestre: MAX(bbox_width, bbox_height) × meters_per_pixel
    - Arii scări: width × height × (meters_per_pixel)²
    
    Output-uri:
      new/runner/output/<RUN_ID>/measure_objects/<plan_id>/
        ├─ openings_measurements_gemini.json  ← Lățimi + arii calculate
        └─ openings_all.json                   ← Lista agregată
    """
    try:
        plans: List[PlanInfo] = load_plan_infos(run_id, stage_name=STAGE_NAME)
    except PlansListError as e:
        print(f"❌ [{STAGE_NAME}] {e}")
        return []
    
    total = len(plans)
    print(f"\n📌 [{STAGE_NAME}] {total} planuri găsite pentru RUN_ID={run_id}\n", flush=True)
    
    if max_parallel is None:
        cpu_count = os.cpu_count() or 4
        # Calcul pur matematic, putem paraleliza agresiv
        max_parallel = min(cpu_count, total)
    
    print(f"⚙️  [{STAGE_NAME}] rulez cu max_parallel = {max_parallel}\n", flush=True)
    
    results: List[MeasureObjectsJobResult] = []
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(
                _run_for_single_plan,
                run_id,
                idx,
                total,
                plan,
            ): plan
            for idx, plan in enumerate(plans, start=1)
        }
        
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            status = "✅" if res.success else "❌"
            print(
                f"{status} [{STAGE_NAME}] {res.plan_id} → {res.message[:250]}",
                flush=True,
            )
    
    failed = [r for r in results if not r.success]
    if failed:
        print(f"\n⚠️ [{STAGE_NAME}] unele planuri au eșuat:")
        for r in failed:
            print(f"   - {r.plan_id}: {r.message[:300]}")
    else:
        print(f"\n✅ [{STAGE_NAME}] toate planurile au trecut etapa measure_objects.")
    
    # Afișare rezumat măsurări
    print(f"\n{'─'*70}")
    print("📏 LĂȚIMI + ARII CALCULATE:")
    print(f"{'─'*70}")
    for r in results:
        if r.success:
            print(f"  {r.plan_id}: {r.message}")
    print(f"{'─'*70}\n")
    
    return results