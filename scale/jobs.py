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

# ⚠️ NU mai folosim CubiCasa - doar RasterScan este folosit

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
    Rulează detecția scării pentru un singur plan.
    Folosește DOAR scale-ul calculat de RasterScan (nu mai există fallback la CubiCasa).
    """
    work_dir = plan.stage_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    output_file = work_dir / "scale_result.json"
    
    try:
        print(
            f"[{STAGE_NAME}] ({index}/{total}) {plan.plan_id} → Scale detection",
            flush=True,
        )
        
        # ✅ Verificăm dacă există scale calculat de RasterScan
        # Calea corectă: walls_from_coords/room_scales.json (nu rooms/room_scales.json)
        raster_room_scales_path = work_dir / "cubicasa_steps" / "raster_processing" / "walls_from_coords" / "room_scales.json"
        
        if raster_room_scales_path.exists():
            try:
                with open(raster_room_scales_path, "r", encoding="utf-8") as f:
                    raster_data = json.load(f)
                
                weighted_m_px = raster_data.get("weighted_average_m_px")
                if weighted_m_px and weighted_m_px > 0:
                    print(f"       ✅ Folosesc scale-ul din RasterScan: {weighted_m_px:.9f} m/px", flush=True)
                    
                    # Construim rezultatul în același format ca CubiCasa
                    room_scales = raster_data.get("room_scales", {})
                    rooms_used = len(room_scales)
                    
                    result = {
                        "meters_per_pixel": float(weighted_m_px),
                        "method": "raster_scan_gemini",
                        "confidence": "high" if rooms_used >= 3 else "medium",
                        "rooms_analyzed": rooms_used,
                        "optimization_info": {
                            "method": "weighted_average",
                            "rooms_count": rooms_used
                        },
                        "per_room_details": [
                            {
                                "room_id": str(room_id),
                                "room_name": room_data.get("room_name", "Unknown"),
                                "area_m2": room_data.get("area_m2", 0.0),
                                "m_px": room_data.get("m_px", 0.0)
                            }
                            for room_id, room_data in room_scales.items()
                        ]
                    }
                    
                    # Adaugă metadata
                    result["meta"] = {
                        "plan_id": plan.plan_id,
                        "plan_image": str(plan.plan_image),
                        "generated_at": datetime.utcnow().isoformat() + "Z",
                        "stage": STAGE_NAME
                    }
                    
                    # Salvează rezultatul
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    
                    print(
                        f"✅ [{STAGE_NAME}] {plan.plan_id}: "
                        f"{weighted_m_px:.6f} m/px ({rooms_used} camere) [RasterScan]",
                        flush=True
                    )
                    
                    return ScaleJobResult(
                        plan_id=plan.plan_id,
                        work_dir=work_dir,
                        success=True,
                        message=f"Scară detectată: {weighted_m_px:.6f} m/px ({rooms_used} camere) [RasterScan]",
                        meters_per_pixel=float(weighted_m_px)
                    )
            except Exception as e:
                print(f"       ⚠️ Eroare la citirea room_scales.json: {e}", flush=True)
        
        # ⚠️ NU mai folosim fallback la CubiCasa - doar RasterScan este folosit
        error_msg = f"Nu există scale calculat de RasterScan pentru {plan.plan_id}"
        print(f"❌ [{STAGE_NAME}] {error_msg}", flush=True)
        
        return ScaleJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=False,
            message=error_msg,
            meters_per_pixel=None
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