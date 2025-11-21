# new/runner/scale/jobs.py
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

from ..config.settings import (
    load_plan_infos,
    PlansListError,
    PlanInfo,
)

from .openai_scale import detect_scale_with_openai


STAGE_NAME = "scale"


@dataclass
class ScaleJobResult:
    plan_id: str
    work_dir: Path
    success: bool
    message: str
    meters_per_pixel: float | None


def _run_for_single_plan(run_id: str, index: int, total: int, plan: PlanInfo) -> ScaleJobResult:
    """
    Detectează scala pentru un singur plan folosind GPT-4o.
    """
    work_dir = plan.stage_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = work_dir / "scale_result.json"
    
    try:
        print(
            f"[{STAGE_NAME}] ({index}/{total}) {plan.plan_id} → scale detection "
            f"(cwd={work_dir})",
            flush=True,
        )
        
        # Apel AI pentru detectare scară
        result = detect_scale_with_openai(plan.plan_image)
        
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
        
        meters_per_pixel = float(result["meters_per_pixel"])
        
        return ScaleJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=True,
            message=f"Scară detectată: {meters_per_pixel:.6f} m/px",
            meters_per_pixel=meters_per_pixel
        )
    
    except Exception as e:
        return ScaleJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=False,
            message=f"Eroare: {e}",
            meters_per_pixel=None
        )


def run_scale_detection_for_run(run_id: str, max_parallel: int | None = None) -> List[ScaleJobResult]:
    """
    Punct de intrare pentru etapa „scale" (scale detection).
    
    Toate output-urile se vor regăsi în:
      new/runner/output/<RUN_ID>/scale/<plan_id>/scale_result.json
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
        # Scale detection e I/O bound (API calls), putem fi mai agresivi
        max_parallel = min(cpu_count * 2, total, 10)  # max 10 concurrent
    
    print(f"⚙️  [{STAGE_NAME}] rulez cu max_parallel = {max_parallel}\n", flush=True)
    
    results: List[ScaleJobResult] = []
    
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
                f"{status} [{STAGE_NAME}] {res.plan_id} "
                f"({res.work_dir}) → {res.message}",
                flush=True,
            )
    
    failed = [r for r in results if not r.success]
    if failed:
        print(f"\n⚠️ [{STAGE_NAME}] unele planuri au eșuat:")
        for r in failed:
            print(f"   - {r.plan_id}: {r.message[:300]}")
    else:
        print(f"\n✅ [{STAGE_NAME}] toate planurile au trecut etapa scale detection.")
    
    # Afișare rezumat scale-uri detectate
    print(f"\n{'─'*70}")
    print("📏 SCALE-URI DETECTATE:")
    print(f"{'─'*70}")
    for r in results:
        if r.success and r.meters_per_pixel:
            print(f"  {r.plan_id}: {r.meters_per_pixel:.6f} m/pixel")
    print(f"{'─'*70}\n")
    
    return results