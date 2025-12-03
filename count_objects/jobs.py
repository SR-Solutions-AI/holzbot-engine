# new/runner/count_objects/jobs.py
from __future__ import annotations

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

from .detector import run_hybrid_detection

# ✅ IMPORTĂM MODULUL DE SLICING (pe care l-am modificat anterior)
# Asigură-te că fișierul detections/roboflow_import.py este cel actualizat!
from detections.roboflow_import import run_roboflow_import


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
    """Rulează Slicing + Hybrid Detection pentru un singur plan."""
    work_dir = plan.stage_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Directorul unde se află imaginea originală plan.jpg
    detections_dir = work_dir.parent.parent / "detections" / plan.plan_id
    plan_jpg = detections_dir / "plan.jpg"
    
    if not plan_jpg.exists():
        return CountObjectsJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=False,
            message=f"Nu găsesc plan.jpg în {detections_dir}"
        )
    
    exports_dir = detections_dir / "export_objects" / "exports"
    
    if not exports_dir.exists():
        return CountObjectsJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=False,
            message=f"Nu găsesc exports_dir în {exports_dir}"
        )
    
    # Configurare Roboflow
    api_key = os.getenv("ROBOFLOW_API_KEY", "")
    workspace = os.getenv("ROBOFLOW_WORKSPACE", "blueprint-recognition")
    project = os.getenv("ROBOFLOW_PROJECT", "house-plan-uwkew")
    version_str = os.getenv("ROBOFLOW_VERSION", "5")
    
    if not api_key:
        return CountObjectsJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=False,
            message="ROBOFLOW_API_KEY lipsește"
        )

    roboflow_config = {
        "api_key": api_key,
        "workspace": workspace,
        "project": project,
        "version": int(version_str)
    }
    
    try:
        print(
            f"[{STAGE_NAME}] ({index}/{total}) {plan.plan_id} → Processing... "
            f"(total_plans={total_plans}, cwd={work_dir})",
            flush=True,
        )
        
        # =========================================================================
        # PASUL 1: RULARE SLICING (Pentru a prinde obiecte mici/multe)
        # =========================================================================
        print(f"       🚀 [SLICING] Running Roboflow Slicing first...", flush=True)
        
        # Configurare env pentru slicing
        slicing_env = {
            "ROBOFLOW_API_KEY": api_key,
            "ROBOFLOW_PROJECT": project,
            "ROBOFLOW_VERSION": version_str,
            # Setăm confidence mic ca să luăm tot, detectorul va filtra ulterior
            "ROBOFLOW_CONFIDENCE": "15" 
        }
        
        # Apelăm funcția de import (modificată să returneze lista)
        # Atenție: folderul de lucru pentru slicing este cel unde e plan.jpg (detections_dir)
        slicing_success, slicing_predictions = run_roboflow_import(
            env=slicing_env, 
            work_dir=detections_dir 
        )
        
        external_preds = None
        if slicing_success and isinstance(slicing_predictions, list):
            print(f"       ✅ [SLICING] Success! Found {len(slicing_predictions)} raw candidates.", flush=True)
            external_preds = slicing_predictions
        else:
            print(f"       ⚠️ [SLICING] Failed or no data. Falling back to standard detection.", flush=True)

        # =========================================================================
        # PASUL 2: RULARE HYBRID DETECTION (Validare + Scări + Gemini)
        # =========================================================================
        print(f"       🧠 [HYBRID] validating candidates...", flush=True)
        
        success, message = run_hybrid_detection(
            plan_image=plan_jpg,
            exports_dir=exports_dir,
            output_dir=work_dir,
            roboflow_config=roboflow_config,
            total_plans=total_plans,
            # Trimitem predicțiile din slicing către detector
            external_predictions=external_preds 
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
        return CountObjectsJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=False,
            message=f"Eroare: {e}"
        )


def run_count_objects_for_run(run_id: str, max_parallel: int | None = None) -> List[CountObjectsJobResult]:
    """Punct de intrare pentru etapa „count_objects" (hybrid detection)."""
    try:
        plans: List[PlanInfo] = load_plan_infos(run_id, stage_name=STAGE_NAME)
    except PlansListError as e:
        print(f"❌ [{STAGE_NAME}] {e}")
        return []
    
    total = len(plans)
    print(f"\n📌 [{STAGE_NAME}] {total} planuri găsite pentru RUN_ID={run_id}\n", flush=True)
    
    if max_parallel is None:
        cpu_count = os.cpu_count() or 4
        max_parallel = max(2, min(cpu_count, total))
    
    print(f"⚙️  [{STAGE_NAME}] rulez cu max_parallel = {max_parallel}\n", flush=True)
    
    results: List[CountObjectsJobResult] = []
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(
                _run_for_single_plan,
                run_id,
                idx,
                total,
                plan,
                total  # TRANSMITE total_plans
            ): plan
            for idx, plan in enumerate(plans, start=1)
        }
        
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            status = "✅" if res.success else "❌"
            print(
                f"{status} [{STAGE_NAME}] {res.plan_id} "
                f"({res.work_dir}) → {res.message[:200]}",
                flush=True,
            )
    
    failed = [r for r in results if not r.success]
    if failed:
        print(f"\n⚠️ [{STAGE_NAME}] unele planuri au eșuat:")
        for r in failed:
            print(f"   - {r.plan_id}: {r.message[:300]}")
    else:
        print(f"\n✅ [{STAGE_NAME}] toate planurile au trecut etapa count_objects.")
    
    return results