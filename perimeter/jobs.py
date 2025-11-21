# new/runner/perimeter/jobs.py
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

from .gemini_measure import measure_perimeter_with_gemini
from .config import (
    MIN_INTERIOR_WALLS_M,
    MAX_INTERIOR_WALLS_M,
    MIN_EXTERIOR_WALLS_M,
    MAX_EXTERIOR_WALLS_M,
    MIN_PERIMETER_M,
    MAX_PERIMETER_M
)


STAGE_NAME = "perimeter"


@dataclass
class PerimeterJobResult:
    plan_id: str
    work_dir: Path
    success: bool
    message: str


def _run_for_single_plan(run_id: str, index: int, total: int, plan: PlanInfo) -> PerimeterJobResult:
    """
    Măsoară lungimile pereților pentru un singur plan.
    """
    work_dir = plan.stage_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Input files
    scale_dir = work_dir.parent.parent / "scale" / plan.plan_id
    scale_json = scale_dir / "scale_result.json"
    
    # Verificări
    if not scale_json.exists():
        return PerimeterJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=False,
            message=f"Nu găsesc {scale_json.name}"
        )
    
    try:
        print(
            f"[{STAGE_NAME}] ({index}/{total}) {plan.plan_id} → measure walls "
            f"(cwd={work_dir})",
            flush=True,
        )
        
        # Load scale data
        with open(scale_json, "r", encoding="utf-8") as f:
            scale_data = json.load(f)
        
        # Call GPT-4o
        result = measure_perimeter_with_gemini(plan.plan_image, scale_data)
        
        # Validare rezultate
        avg = result["estimations"]["average_result"]
        int_m = float(avg["interior_meters"])
        ext_m = float(avg["exterior_meters"])
        per_m = float(avg["total_perimeter_meters"])
        
        warnings = []
        
        if not (MIN_INTERIOR_WALLS_M <= int_m <= MAX_INTERIOR_WALLS_M):
            warnings.append(f"Pereți interiori în afara intervalului: {int_m:.1f}m")
        
        if not (MIN_EXTERIOR_WALLS_M <= ext_m <= MAX_EXTERIOR_WALLS_M):
            warnings.append(f"Pereți exteriori în afara intervalului: {ext_m:.1f}m")
        
        if not (MIN_PERIMETER_M <= per_m <= MAX_PERIMETER_M):
            warnings.append(f"Perimetru în afara intervalului: {per_m:.1f}m")
        
        if per_m > ext_m:
            warnings.append(f"Perimetru ({per_m:.1f}m) > Pereți ext ({ext_m:.1f}m) - INVALID")
        
        # Add metadata + warnings
        result["meta"] = {
            "plan_id": plan.plan_id,
            "plan_image": str(plan.plan_image),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "stage": STAGE_NAME
        }
        
        if warnings:
            result["warnings"] = warnings
        
        # Save result
        output_file = work_dir / "walls_measurements_gemini.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"       📄 Salvat: {output_file.name}")
        
        # Summary message
        message = (
            f"Interior: {int_m:.1f}m, Exterior: {ext_m:.1f}m, Perimetru: {per_m:.1f}m"
        )
        
        if warnings:
            message += f" | ⚠️  {len(warnings)} warnings"
        
        return PerimeterJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=True,
            message=message
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return PerimeterJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=False,
            message=f"Eroare: {e}"
        )


def run_perimeter_for_run(run_id: str, max_parallel: int | None = None) -> List[PerimeterJobResult]:
    """
    Punct de intrare pentru etapa „perimeter" (măsurare lungimi pereți).
    
    Output-uri:
      new/runner/output/<RUN_ID>/perimeter/<plan_id>/walls_measurements_gemini.json
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
        # Perimeter = API calls, limităm concurrent-ul
        max_parallel = min(4, total)
    
    print(f"⚙️  [{STAGE_NAME}] rulez cu max_parallel = {max_parallel}\n", flush=True)
    
    results: List[PerimeterJobResult] = []
    
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
        print(f"\n✅ [{STAGE_NAME}] toate planurile au trecut etapa perimeter.")
    
    # Afișare rezumat
    print(f"\n{'─'*70}")
    print("📏 LUNGIMI PEREȚI MĂSURATE:")
    print(f"{'─'*70}")
    for r in results:
        if r.success:
            print(f"  {r.plan_id}: {r.message}")
    print(f"{'─'*70}\n")
    
    return results