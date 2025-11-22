# new/runner/area/jobs.py
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

# --- 1. Importăm OUTPUT_ROOT din setări ---
from config.settings import (
    load_plan_infos,
    PlansListError,
    PlanInfo,
    OUTPUT_ROOT, 
)

from .calculator import calculate_areas_for_plan
from .aggregator import aggregate_multi_plan_areas

# IMPORTUL NOULUI MODUL
from .gemini_area import estimate_house_area_with_gemini


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
    is_single_plan: bool
) -> AreaJobResult:
    """
    Calculează ariile pentru un singur plan.
    Integrat cu Gemini Area Estimation.
    """
    work_dir = plan.stage_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # 1. LOCATE RESOURCES
    # ==========================================
    
    # Original name for metadata (fallback)
    original_name = plan.plan_image.stem
    output_root = work_dir.parent.parent
    job_root = output_root.parent.parent / "jobs" / run_id
    
    metadata_file = job_root / "plan_metadata" / f"{original_name}.json"
    
    # Walls measurements (Geometry)
    perimeter_dir = work_dir.parent.parent / "perimeter" / plan.plan_id
    walls_json = perimeter_dir / "walls_measurements_gemini.json"
    
    # Openings (Windows/Doors)
    measure_dir = work_dir.parent.parent / "measure_objects" / plan.plan_id
    openings_json = measure_dir / "openings_all.json"
    measurements_json = measure_dir / "openings_measurements_gemini.json"
    
    # Scale File (NECESAR PENTRU GEMINI AREA)
    scale_dir = work_dir.parent.parent / "scale" / plan.plan_id
    scale_json = scale_dir / "scale_result.json"

    # Validări fișiere critice
    if not walls_json.exists():
        return AreaJobResult(plan.plan_id, work_dir, False, f"Missing walls: {walls_json.name}")
    if not openings_json.exists():
        return AreaJobResult(plan.plan_id, work_dir, False, f"Missing openings: {openings_json.name}")
    
    try:
        print(
            f"[{STAGE_NAME}] ({index}/{total}) {plan.plan_id} → calculate areas...",
            flush=True,
        )
        
        # ==========================================
        # 2. CALCULATE HOUSE AREA (GEMINI INTEGRATION)
        # ==========================================
        house_area_m2 = 0.0
        gemini_area_result = {}
        
        # Încercăm întâi cu Gemini folosind scriptul tău
        if scale_json.exists() and os.getenv("GEMINI_API_KEY"):
            try:
                print(f"       🤖 Calling Gemini Area Estimation for {plan.plan_id}...")
                gemini_area_result = estimate_house_area_with_gemini(
                    image_path=plan.plan_image,
                    scale_json_path=scale_json
                )
                
                # Extragem valoarea finală calculată de AI
                est = gemini_area_result.get("surface_estimation", {})
                house_area_m2 = float(est.get("final_area_m2", 0.0))
                
                # Salvăm rezultatul detaliat al AI-ului (pt. debug/încredere)
                gemini_out_file = work_dir / "house_area_gemini.json"
                with open(gemini_out_file, "w", encoding="utf-8") as f:
                    json.dump(gemini_area_result, f, indent=2, ensure_ascii=False)
                
                print(f"       ✅ Gemini Area: {house_area_m2:.2f} m² (Method: {est.get('method_used')})")

            except Exception as e:
                print(f"       ⚠️ Gemini Area Failed: {e}. Falling back to metadata.")
        
        # Fallback: Metadata
        if house_area_m2 <= 0:
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                house_area_m2 = meta.get("floor_classification", {}).get("estimated_area_m2", 0.0)
                print(f"       ℹ️ Using Metadata Area: {house_area_m2:.2f} m²")
            else:
                return AreaJobResult(plan.plan_id, work_dir, False, "Nu am putut determina aria casei (nici Gemini, nici Metadata).")

        # ==========================================
        # 3. LOAD OTHER DATA
        # ==========================================
        
        # Floor Type (tot din metadata, clasificarea rămâne valabilă)
        floor_type = "unknown"
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                 floor_type = json.load(f).get("floor_classification", {}).get("floor_type", "unknown")

        with open(walls_json, "r", encoding="utf-8") as f:
            walls_data = json.load(f)
        
        with open(openings_json, "r", encoding="utf-8") as f:
            openings_data = json.load(f)
        
        # Stairs
        stairs_area_m2 = None
        if measurements_json.exists():
            with open(measurements_json, "r", encoding="utf-8") as f:
                meas_data = json.load(f)
            stairs_meas = meas_data.get("measurements", {}).get("stairs")
            if stairs_meas:
                stairs_area_m2 = float(stairs_meas.get("total_area_m2", 0.0))
        
        # ==========================================
        # 4. EXECUTE CALCULATOR
        # ==========================================
        
        result = calculate_areas_for_plan(
            plan_id=plan.plan_id,
            floor_type=floor_type,
            is_single_plan=is_single_plan,
            house_area_m2=house_area_m2, # Valoarea nouă de la Gemini
            walls_measurements=walls_data,
            openings_all=openings_data,
            stairs_area_m2=stairs_area_m2
        )
        
        # Add metadata & source info
        result["meta"] = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "stage": STAGE_NAME,
            "area_source": "gemini_hybrid" if gemini_area_result else "metadata_fallback",
            "confidence": gemini_area_result.get("confidence", "unknown")
        }
        
        # ==========================================
        # 5. SAVE RESULT
        # ==========================================
        
        output_file = work_dir / "areas_calculated.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"       📄 Salvat: {output_file.name}")
        
        # Summary message
        walls = result["walls"]
        surfaces = result["surfaces"]
        msg_parts = [
            f"Total: {house_area_m2:.0f}m²",
            f"Floor: {surfaces['floor_m2']:.1f}m²",
            f"Walls Ext: {walls['exterior']['net_area_m2']:.1f}m²"
        ]
        return AreaJobResult(plan.plan_id, work_dir, True, " | ".join(msg_parts), result_data=result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return AreaJobResult(plan.plan_id, work_dir, False, f"Eroare: {e}")


def run_area_for_run(run_id: str, max_parallel: int | None = None) -> List[AreaJobResult]:
    """
    Punct de intrare pentru etapa „area" (calcul arii complete).
    """
    try:
        plans: List[PlanInfo] = load_plan_infos(run_id, stage_name=STAGE_NAME)
    except PlansListError as e:
        print(f"❌ [{STAGE_NAME}] {e}")
        return []
    
    total = len(plans)
    is_single_plan = (total == 1)
    
    print(f"\n📌 [{STAGE_NAME}] {total} plan{'uri' if total > 1 else ''} găsit{'e' if total > 1 else ''} pentru RUN_ID={run_id}")
    
    if max_parallel is None:
        cpu_count = os.cpu_count() or 4
        max_parallel = min(cpu_count, total)
    
    results: List[AreaJobResult] = []
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(_run_for_single_plan, run_id, idx, total, plan, is_single_plan): plan
            for idx, plan in enumerate(plans, start=1)
        }
        
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            status = "✅" if res.success else "❌"
            print(f"{status} [{STAGE_NAME}] {res.plan_id} → {res.message[:150]}", flush=True)
    
    # AGREGARE
    successful_results = [r.result_data for r in results if r.success and r.result_data]
    if len(successful_results) > 1:
        print(f"\n📊 Agregare rezultate multi-plan...")
        summary = aggregate_multi_plan_areas(successful_results)
        
        # --- 2. CORECTEAZĂ CALEA AICI FOLOSIND OUTPUT_ROOT ---
        output_root = OUTPUT_ROOT / run_id / STAGE_NAME
        output_root.mkdir(parents=True, exist_ok=True) # Asigurăm că folderul există
        
        summary_file = output_root / "areas_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"       📄 Summary salvat: {summary_file}")

    return results