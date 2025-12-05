# new/runner/count_objects/jobs.py
from __future__ import annotations

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List

from config.settings import (
    load_plan_infos,
    PlansListError,
    PlanInfo,
)

from .detector import run_hybrid_detection

STAGE_NAME = "count_objects"


@dataclass
class CountObjectsJobResult:
    plan_id: str
    work_dir: Path
    success: bool
    message: str


def _run_for_single_plan(
    run_id: str, 
    index: int, 
    total: int, 
    plan: PlanInfo,
    total_plans: int
) -> CountObjectsJobResult:
    """
    Rulează Count Objects folosind fișierul detections.json generat în pasul anterior.
    """
    work_dir = plan.stage_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Identificăm directoarele sursă
    # Pasul anterior (detections) a rulat aici:
    detections_step_dir = work_dir.parent.parent / "detections" / plan.plan_id
    
    plan_jpg = detections_step_dir / "plan.jpg"
    detections_json = detections_step_dir / "export_objects" / "detections.json"
    
    # Directorul de export curent
    exports_dir = detections_step_dir / "export_objects" / "exports"
    
    # Validări
    if not plan_jpg.exists():
        # Fallback: poate e în folderul curent?
        if (work_dir / "plan.jpg").exists():
            plan_jpg = work_dir / "plan.jpg"
        else:
            return CountObjectsJobResult(plan.plan_id, work_dir, False, f"Lipsă plan.jpg (căutat în {detections_step_dir})")

    # Configurare API (pentru scări și verificări)
    api_key = os.getenv("ROBOFLOW_API_KEY", "")
    roboflow_config = {
        "api_key": api_key,
        "workspace": os.getenv("ROBOFLOW_WORKSPACE", "blueprint-recognition"),
        "project": os.getenv("ROBOFLOW_PROJECT", "house-plan-uwkew"),
        "version": int(os.getenv("ROBOFLOW_VERSION", "5"))
    }
    
    try:
        print(f"[{STAGE_NAME}] Processing {plan.plan_id}...", flush=True)
        
        # 2. ÎNCĂRCARE CANDIDAȚI DE PE DISC
        candidates = []
        if detections_json.exists():
            try:
                with open(detections_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    candidates = data.get("predictions", [])
                print(f"       📂 Loaded {len(candidates)} candidates from disk.", flush=True)
            except Exception as e:
                print(f"       ⚠️ Error reading detections.json: {e}", flush=True)
        else:
            print(f"       ⚠️ detections.json not found at {detections_json}. Logic will fallback/fail.", flush=True)

        # 3. RULARE HYBRID DETECTION
        # Trimitem calea către JSON (sau lista încărcată)
        print(f"       🧠 Starting Validation Logic (Template + AI)...", flush=True)
        
        success, message = run_hybrid_detection(
            plan_image=plan_jpg,
            exports_dir=exports_dir,
            output_dir=work_dir,
            roboflow_config=roboflow_config,
            total_plans=total_plans,
            external_predictions=candidates # Aici intră datele de pe disc
        )
        
        return CountObjectsJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=success,
            message=message
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return CountObjectsJobResult(plan.plan_id, work_dir, False, str(e))


def run_count_objects_for_run(run_id: str, max_parallel: int | None = None) -> List[CountObjectsJobResult]:
    try:
        plans: List[PlanInfo] = load_plan_infos(run_id, stage_name=STAGE_NAME)
    except PlansListError as e:
        print(f"❌ [{STAGE_NAME}] {e}")
        return []
    
    total = len(plans)
    if max_parallel is None:
        cpu_count = os.cpu_count() or 4
        max_parallel = max(2, min(cpu_count, total))
    
    results = []
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(_run_for_single_plan, run_id, idx, total, plan, total): plan
            for idx, plan in enumerate(plans, start=1)
        }
        for fut in as_completed(futures):
            results.append(fut.result())
            
    return results