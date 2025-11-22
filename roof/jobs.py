# new/runner/roof/jobs.py
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
from config.frontend_loader import load_frontend_data_for_run

from .calculator import calculate_roof_price


STAGE_NAME = "roof" 


@dataclass
class RoofJobResult:
    plan_id: str
    work_dir: Path
    success: bool
    message: str
    result_data: dict | None = None


def _load_floor_metadata(job_root: Path, original_name: str) -> dict | None:
    """Încarcă metadata pentru a determina floor_type."""
    metadata_file = job_root / "plan_metadata" / f"{original_name}.json"
    if not metadata_file.exists():
        return None
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _run_for_single_plan(
    run_id: str,
    index: int,
    total: int,
    plan: PlanInfo,
    frontend_data: dict | None,
    total_floors: int,
    job_root: Path,
) -> RoofJobResult:
    """
    Calculează prețul acoperișului.
    FORCE UPDATE: Dacă total_floors == 1, se calculează OBLIGATORIU acoperișul.
    """
    work_dir = plan.stage_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # STEP 1: Determinare Tip Etaj (Blindată)
    # ==========================================
    
    original_name = plan.plan_image.stem
    metadata = _load_floor_metadata(job_root, original_name)
    
    is_top_floor = False
    floor_type = "unknown"
    
    # 1. Citire Metadata (doar informativ inițial)
    if metadata:
        floor_class = metadata.get("floor_classification", {})
        floor_type = floor_class.get("floor_type", "unknown").lower()
        
        # Detectare standard
        is_top_floor = any(keyword in floor_type for keyword in ["top", "roof", "attic", "mansarda"])

    # 2. 🔥 REGULA DE AUR (FORCE SINGLE PLAN) 🔥
    # Această verificare este scoasă în afara oricărui if anterior.
    # Dacă e singurul plan din run, are acoperiș. Punct.
    if total_floors == 1:
        if not is_top_floor:
            print(f"       🔥🔥 FORCE ROOF: Single plan detected (Type was '{floor_type}'). Forcing is_top_floor=True.")
        is_top_floor = True

    # Dacă TOT nu e top floor (caz multi-etaj unde e parter/inter), ieșim
    if not is_top_floor:
        result = {
            "plan_id": plan.plan_id,
            "floor_type": floor_type,
            "is_top_floor": False,
            "roof_area_sqm": 0.0,
            "roof_final_total_eur": 0.0,
            "note": f"Not top floor ({floor_type}) & total_floors={total_floors} -> no roof"
        }
        
        output_file = work_dir / "roof_estimation.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return RoofJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=True,
            message=f"Nu e top floor ({floor_type}) → skip acoperiș",
            result_data=result
        )
    
    # ==========================================
    # STEP 2: Încarcă date pentru CALCUL
    # ==========================================
    
    # INPUT FILES
    area_dir = work_dir.parent.parent / "area" / plan.plan_id
    area_json = area_dir / "areas_calculated.json"
    
    if not area_json.exists():
        return RoofJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=False,
            message=f"Nu găsesc {area_json.name}"
        )
    
    perimeter_dir = work_dir.parent.parent / "perimeter" / plan.plan_id
    perimeter_json = perimeter_dir / "walls_measurements_gemini.json"
    
    try:
        print(
            f"[{STAGE_NAME}] ({index}/{total}) {plan.plan_id} → calculate roof price...",
            flush=True
        )
        
        with open(area_json, "r", encoding="utf-8") as f:
            area_data = json.load(f)
        
        surfaces = area_data.get("surfaces", {})
        
        # Aria acoperișului
        roof_area_m2 = surfaces.get("roof_m2") 
        ceiling_area_m2 = surfaces.get("ceiling_m2")
        
        # Fallback logică agresivă pentru Single Plan / Unknown
        if (roof_area_m2 is None or roof_area_m2 <= 0):
            # Încercăm ceiling
            if ceiling_area_m2 and ceiling_area_m2 > 0:
                 roof_area_m2 = ceiling_area_m2 * 1.35 # Adăugăm pantă + streașină estimată
                 print(f"       ⚠️  roof_m2 lipsă. Estimat din ceiling ({ceiling_area_m2}) * 1.35 = {roof_area_m2:.2f}")
            else:
                # Încercăm gross area
                gross = surfaces.get("floor_gross_m2") or area_data.get("house_area_m2")
                if gross and gross > 0:
                    roof_area_m2 = gross * 1.35
                    ceiling_area_m2 = gross # Aproximare pentru tavan
                    print(f"       ⚠️  roof_m2 & ceiling lipsă. Estimat din gross ({gross}) = {roof_area_m2:.2f}")

        if roof_area_m2 is None or roof_area_m2 <= 0:
             return RoofJobResult(
                plan_id=plan.plan_id,
                work_dir=work_dir,
                success=False,
                message="Eroare critică: Aria (roof_m2) este 0 chiar și după fallback."
            )
        
        # Fallback final pentru ceiling (necesar la izolație)
        if ceiling_area_m2 is None:
            ceiling_area_m2 = roof_area_m2 / 1.35

        # Perimetrul
        perimeter_m = None
        if perimeter_json.exists():
            with open(perimeter_json, "r", encoding="utf-8") as f:
                perim_data = json.load(f)
            perimeter_m = perim_data.get("estimations", {}).get("average_result", {}).get("total_perimeter_meters")
        
        # ==========================================
        # STEP 3: Extrage input de la frontend
        # ==========================================
        
        roof_type_user = "Două ape"
        material_user = "Țiglă"
        
        if frontend_data:
            roof_type_user = frontend_data.get("tipAcoperis", roof_type_user)
            material_user = frontend_data.get("materialAcoperis", material_user)
        
        # ==========================================
        # STEP 4: CALCUL PREȚ ACOPERIȘ
        # ==========================================
        
        result = calculate_roof_price(
            house_area_m2=float(roof_area_m2),
            ceiling_area_m2=float(ceiling_area_m2),
            perimeter_m=float(perimeter_m) if perimeter_m else None,
            roof_type_user=roof_type_user,
            material_user=material_user,
            frontend_data=frontend_data,
            total_floors=total_floors
        )
        
        result["plan_id"] = plan.plan_id
        result["floor_type"] = floor_type
        result["is_top_floor"] = True # Confirmăm în output
        
        # ==========================================
        # STEP 5: Salvează rezultatul
        # ==========================================
        
        output_file = work_dir / "roof_estimation.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        roof_info = result["inputs"]["roof_type"]
        final_cost = result["roof_final_total_eur"]
        
        return RoofJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=True,
            message=f"Tip: {roof_info.get('matched_name_de', 'N/A')} | Arie: {roof_area_m2:.1f}m² | Cost: {final_cost:,.0f} EUR",
            result_data=result
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return RoofJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=False,
            message=f"Eroare: {e}"
        )

def run_roof_for_run(run_id: str, max_parallel: int | None = None) -> List[RoofJobResult]:
    """
    Punct de intrare pentru etapa „roof" (calcul acoperiș).
    
    LOGICA:
    - Doar planurile TOP FLOOR au acoperiș calculat
    - Ground floor → cost = 0 (cu excepția cazului când e singurul etaj)
    
    Output-uri:
      new/runner/output/<RUN_ID>/roof/<plan_id>/roof_estimation.json
    """
    try:
        plans = load_plan_infos(run_id, stage_name=STAGE_NAME)
    except PlansListError as e:
        print(f"❌ [{STAGE_NAME}] {e}")
        return []
    
    total = len(plans)
    
    # Locația job_root pentru metadata
    from config.settings import JOBS_ROOT
    job_root = None
    for jdir in JOBS_ROOT.glob("*"):
        if jdir.is_dir() and run_id in jdir.name:
            job_root = jdir
            break
    
    if job_root is None:
        job_root = JOBS_ROOT / run_id
    
    # Încarcă frontend data
    frontend_data = load_frontend_data_for_run(run_id, job_root)
    
    print(f"\n⚙️  [{STAGE_NAME}] Calcul acoperiș pentru {total} plan{'uri' if total > 1 else ''} (total_floors={total})...")
    
    if max_parallel is None:
        cpu_count = os.cpu_count() or 4
        max_parallel = min(cpu_count, total)
    
    results: List[RoofJobResult] = []
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(
                _run_for_single_plan,
                run_id,
                idx,
                total,
                plan,
                frontend_data,
                total,
                job_root,
            ): plan
            for idx, plan in enumerate(plans, start=1)
        }
        
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            status = "✅" if res.success else "❌"
            print(f"{status} [{STAGE_NAME}] {res.plan_id} → {res.message}")
    
    # ==========================================
    # REZUMAT FINAL
    # ==========================================
    
    print(f"\n{'─'*70}")
    print("🏠 REZUMAT ACOPERIȘ:")
    print(f"{'─'*70}")
    
    total_roof_cost = 0.0
    for r in results:
        if r.success and r.result_data:
            cost = r.result_data.get("roof_final_total_eur", 0)
            total_roof_cost += cost
            
            if r.result_data.get("is_top_floor"):
                area = r.result_data.get("inputs", {}).get("house_area_m2", 0)
                print(f"  ✅ {r.plan_id}: {area:.1f} m² → {cost:,.0f} EUR")
            else:
                print(f"  ℹ️  {r.plan_id}: Nu e top floor → 0 EUR")
    
    print(f"{'─'*70}")
    print(f"💰 TOTAL ACOPERIȘ: {total_roof_cost:,.2f} EUR")
    print(f"{'─'*70}\n")
    
    return results