# new/runner/perimeter/jobs.py
from __future__ import annotations

import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

# ✅ IMPORT CORECT
from config.settings import load_plan_infos, PlanInfo, get_output_root_for_run

# ✅ IMPORT NOU: CubiCasa Detector
from cubicasa_detector.jobs import run_cubicasa_for_plan

STAGE_NAME = "perimeter"


@dataclass
class PerimeterJobResult:
    """Rezultatul procesării unui plan în etapa de măsurare perimetru."""
    plan_id: str
    work_dir: Path
    success: bool
    message: str
    exterior_meters: float | None = None
    interior_meters: float | None = None


def _run_for_single_plan(
    run_id: str, 
    index: int, 
    total: int, 
    plan: PlanInfo
) -> PerimeterJobResult:
    """
    Rulează măsurarea perimetrului pentru un singur plan folosind CubiCasa.
    
    Încearcă mai întâi să refolosească rezultatul de la etapa 'scale' (cache).
    Dacă nu există, rulează CubiCasa din nou.
    """
    work_dir = plan.stage_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Verifică dacă avem deja rezultatul CubiCasa de la scale
    scale_dir = work_dir.parent.parent / "scale" / plan.plan_id
    cubicasa_cache = scale_dir / "cubicasa_result.json"
    
    try:
        print(
            f"[{STAGE_NAME}] ({index}/{total}) {plan.plan_id} → perimeter from CubiCasa",
            flush=True,
        )
        
        # Încercăm să refolosim rezultatul de la scale (dacă există)
        if cubicasa_cache.exists():
            print(f"       ♻️  Refolosesc rezultatul CubiCasa din cache", flush=True)
            with open(cubicasa_cache, "r", encoding="utf-8") as f:
                cubicasa_result = json.load(f)
        else:
            # Rulăm CubiCasa din nou
            print(f"       🔄 Rulare CubiCasa (cache lipsește)", flush=True)
            cubicasa_result = run_cubicasa_for_plan(
                plan_image=plan.plan_image,
                output_dir=work_dir
            )
            
            # Salvăm pentru cache (pentru viitor)
            cubicasa_cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cubicasa_cache, "w", encoding="utf-8") as f:
                json.dump(cubicasa_result, f, indent=2, ensure_ascii=False)
        
        # Extragem măsurătorile
        measurements = cubicasa_result["measurements"]["metrics"]
        
        # Adaptăm la formatul așteptat de pipeline
        result = {
            "scale_meters_per_pixel": measurements["scale_m_per_px"],
            "estimations": {
                "by_cubicasa": {
                    "interior_meters": measurements["walls_int_m"],
                    "exterior_meters": measurements["walls_ext_m"],
                    "total_perimeter_meters": measurements["walls_ext_m"],
                    "method_notes": "CubiCasa AI + Gemini optimization"
                },
                # average_result e folosit de pricing și alte module downstream
                "average_result": {
                    "interior_meters": measurements["walls_int_m"],
                    "exterior_meters": measurements["walls_ext_m"],
                    "total_perimeter_meters": measurements["walls_ext_m"]
                }
            },
            "confidence": "high",
            "verification_notes": f"Calculat cu CubiCasa. Indoor area: {measurements['area_indoor_m2']:.2f} m²"
        }
        
        # Adaugă metadata
        result["meta"] = {
            "plan_id": plan.plan_id,
            "plan_image": str(plan.plan_image),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "stage": STAGE_NAME
        }
        
        # Salvează (ACELAȘI FORMAT ca înainte)
        output_file = work_dir / "walls_measurements_gemini.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        message = (
            f"Interior: {measurements['walls_int_m']:.1f}m, "
            f"Exterior: {measurements['walls_ext_m']:.1f}m"
        )
        
        print(
            f"✅ [{STAGE_NAME}] {plan.plan_id}: {message}",
            flush=True
        )
        
        return PerimeterJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=True,
            message=message,
            exterior_meters=measurements["walls_ext_m"],
            interior_meters=measurements["walls_int_m"]
        )
        
    except Exception as e:
        error_msg = f"Eroare CubiCasa pentru {plan.plan_id}: {e}"
        print(f"❌ [{STAGE_NAME}] {error_msg}", flush=True)
        traceback.print_exc()
        
        return PerimeterJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=False,
            message=error_msg,
            exterior_meters=None,
            interior_meters=None
        )


def run_perimeter_for_run(run_id: str, max_workers: int = 4) -> List[PerimeterJobResult]:
    """
    Rulează măsurarea perimetrului pentru toate planurile din run.
    
    Folosește ThreadPoolExecutor pentru paralelizare.
    """
    print(f"\n{'='*60}")
    print(f"[{STAGE_NAME}] Starting perimeter measurement for run: {run_id}")
    print(f"{'='*60}\n")
    
    plans = load_plan_infos(run_id, STAGE_NAME)
    
    if not plans:
        print(f"⚠️ [{STAGE_NAME}] No plans found for run {run_id}")
        return []
    
    results: List[PerimeterJobResult] = []
    total = len(plans)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_for_single_plan, run_id, idx, total, plan): plan
            for idx, plan in enumerate(plans, start=1)
        }
        
        for future in as_completed(futures):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                plan = futures[future]
                print(f"❌ [{STAGE_NAME}] Unexpected error for {plan.plan_id}: {e}")
                traceback.print_exc()
                results.append(
                    PerimeterJobResult(
                        plan_id=plan.plan_id,
                        work_dir=plan.stage_work_dir,
                        success=False,
                        message=f"Unexpected error: {e}",
                        exterior_meters=None,
                        interior_meters=None
                    )
                )
    
    # Summary
    successful = sum(1 for r in results if r.success)
    failed = total - successful
    
    print(f"\n{'='*60}")
    print(f"[{STAGE_NAME}] Perimeter Measurement Complete")
    print(f"  ✅ Success: {successful}/{total}")
    if failed > 0:
        print(f"  ❌ Failed: {failed}/{total}")
    print(f"{'='*60}\n")
    
    return results