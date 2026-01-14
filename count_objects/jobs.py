# file: engine/count_objects/jobs.py
from __future__ import annotations

import os
import json
import traceback
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
    # 1. Configurare directoare curente
    work_dir = plan.stage_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Identificare resurse din etapele ANTERIOARE
    # Structura: runner/output/<RUN_ID>/<STAGE>/<PLAN_ID>
    run_output_root = work_dir.parent.parent
    
    # A. Din DETECTIONS (pentru JSON-ul cu candidați)
    detections_dir = run_output_root / "detections" / plan.plan_id
    plan_jpg = detections_dir / "plan.jpg"
    detections_json = detections_dir / "export_objects" / "detections.json"
    
    # B. Din SCALE (pentru masca outdoor)
    scale_dir = run_output_root / "scale" / plan.plan_id
    outdoor_mask_path = scale_dir / "cubicasa_steps" / "03_outdoor_mask.png"
    
    # Directorul unde salvăm noi rezultatele finale
    exports_dir = detections_dir / "export_objects" / "exports" # Salvăm tot în structura detections pentru consistență
    
    # Validări fișiere critice
    if not plan_jpg.exists():
        # Încercăm fallback local
        if (work_dir / "plan.jpg").exists():
            plan_jpg = work_dir / "plan.jpg"
        else:
            return CountObjectsJobResult(plan.plan_id, work_dir, False, "Lipsă plan.jpg (nici în detections, nici local)")

    # Configurare API Roboflow (pentru scări, dacă e nevoie)
    api_key = os.getenv("ROBOFLOW_API_KEY", "")
    roboflow_config = {
        "api_key": api_key,
        "workspace": os.getenv("ROBOFLOW_WORKSPACE", "blueprint-recognition"),
        "project": os.getenv("ROBOFLOW_PROJECT", "house-plan-uwkew"),
        "version": int(os.getenv("ROBOFLOW_VERSION", "5"))
    }
    
    try:
        print(f"[{STAGE_NAME}] Processing {plan.plan_id}...", flush=True)
        
        # 3. ÎNCĂRCARE DATE DE PE DISC
        candidates = []
        if detections_json.exists():
            try:
                with open(detections_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    candidates = data.get("predictions", [])
                print(f"       📂 Loaded {len(candidates)} candidates from {detections_json.name}", flush=True)
            except Exception as e:
                print(f"       ⚠️ JSON Corrupt: {e}", flush=True)
        else:
            print(f"       ⚠️ detections.json missing at {detections_json}", flush=True)

        # 4. RULARE VALIDARE
        print(f"       🧠 Starting Hybrid Validation...", flush=True)
        if outdoor_mask_path.exists():
            print(f"       🏞️  Found outdoor mask: {outdoor_mask_path.name}", flush=True)
        
        success, message = run_hybrid_detection(
            plan_image=plan_jpg,
            exports_dir=exports_dir,
            output_dir=work_dir,
            roboflow_config=roboflow_config,
            total_plans=total_plans,
            external_predictions=candidates,   # <--- Aici intră lista citită de pe disc
            outdoor_mask_path=outdoor_mask_path # <--- Aici intră masca
        )
        
        # 5. NOTIFICARE UI (Live Feed) - Imaginea Portocalie
        orange_img = work_dir / "final_orange.jpg"
        if success and orange_img.exists():
            # Mesaj special pentru orchestrator/backend
            print(f">>> UI:STAGE:{STAGE_NAME}|IMG:{str(orange_img)}", flush=True)
        
        return CountObjectsJobResult(plan.plan_id, work_dir, success, message)
    
    except Exception as e:
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