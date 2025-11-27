# new/runner/scale/jobs.py
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

# ✅ IMPORT CubiCasa
from cubicasa_detector.jobs import run_cubicasa_for_plan

STAGE_NAME = "scale"


# ✅ CUSTOM JSON ENCODER pentru Path
class PathEncoder(json.JSONEncoder):
    """Convertește Path objects → string pentru JSON serialization."""
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


@dataclass
class ScaleJobResult:
    """Rezultatul procesării unui plan în etapa de detectare a scării."""
    plan_id: str
    work_dir: Path
    success: bool
    message: str
    meters_per_pixel: float | None = None


def _run_for_single_plan(
    run_id: str, 
    index: int, 
    total: int, 
    plan: PlanInfo
) -> ScaleJobResult:
    """
    Rulează detecția scării pentru un singur plan folosind CubiCasa + Gemini.
    """
    work_dir = plan.stage_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    output_file = work_dir / "scale_result.json"
    
    try:
        print(
            f"[{STAGE_NAME}] ({index}/{total}) {plan.plan_id} → CubiCasa scale detection",
            flush=True,
        )
        
        # 🔥 FOLOSIM CUBICASA
        cubicasa_result = run_cubicasa_for_plan(
            plan_image=plan.plan_image,
            output_dir=work_dir
        )
        
        scale_info = cubicasa_result["scale_result"]
        meters_per_pixel = float(scale_info["meters_per_pixel"])
        
        # Adaptăm output-ul la formatul așteptat de pipeline
        result = {
            "meters_per_pixel": meters_per_pixel,
            "method": "cubicasa_gemini",
            "confidence": "high" if scale_info["rooms_used"] >= 3 else "medium",
            "rooms_analyzed": scale_info["rooms_used"],
            "optimization_info": scale_info["optimization_info"],
            "per_room_details": scale_info.get("per_room", [])
        }
        
        # Adaugă metadata
        result["meta"] = {
            "plan_id": plan.plan_id,
            "plan_image": str(plan.plan_image),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "stage": STAGE_NAME
        }
        
        # Salvează rezultatul (ACELAȘI FORMAT ca înainte)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # ✅ Salvează și rezultatul complet CubiCasa pentru cache
        # FOLOSEȘTE PathEncoder pentru a converti Path → string
        cache_file = work_dir / "cubicasa_result.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cubicasa_result, f, indent=2, ensure_ascii=False, cls=PathEncoder)
        
        print(
            f"✅ [{STAGE_NAME}] {plan.plan_id}: "
            f"{meters_per_pixel:.6f} m/px ({scale_info['rooms_used']} camere)",
            flush=True
        )
        
        return ScaleJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=True,
            message=f"Scară detectată: {meters_per_pixel:.6f} m/px ({scale_info['rooms_used']} camere)",
            meters_per_pixel=meters_per_pixel
        )
        
    except Exception as e:
        error_msg = f"Eroare CubiCasa pentru {plan.plan_id}: {e}"
        print(f"❌ [{STAGE_NAME}] {error_msg}", flush=True)
        traceback.print_exc()
        
        return ScaleJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=False,
            message=error_msg,
            meters_per_pixel=None
        )


def run_scale_detection_for_run(run_id: str, max_workers: int = 4) -> List[ScaleJobResult]:
    """
    Rulează detecția scării pentru toate planurile din run.
    """
    print(f"\n{'='*60}")
    print(f"[{STAGE_NAME}] Starting scale detection for run: {run_id}")
    print(f"{'='*60}\n")
    
    plans = load_plan_infos(run_id, STAGE_NAME)
    
    if not plans:
        print(f"⚠️ [{STAGE_NAME}] No plans found for run {run_id}")
        return []
    
    results: List[ScaleJobResult] = []
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
                    ScaleJobResult(
                        plan_id=plan.plan_id,
                        work_dir=plan.stage_work_dir,
                        success=False,
                        message=f"Unexpected error: {e}",
                        meters_per_pixel=None
                    )
                )
    
    # Summary
    successful = sum(1 for r in results if r.success)
    failed = total - successful
    
    print(f"\n{'='*60}")
    print(f"[{STAGE_NAME}] Scale Detection Complete")
    print(f"  ✅ Success: {successful}/{total}")
    if failed > 0:
        print(f"  ❌ Failed: {failed}/{total}")
    print(f"{'='*60}\n")
    
    return results