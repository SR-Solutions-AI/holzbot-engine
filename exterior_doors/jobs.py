# new/runner/exterior_doors/jobs.py
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

from .pipeline import run_exterior_doors_for_plan

STAGE_NAME = "exterior_doors"

@dataclass
class ExteriorDoorsJobResult:
    plan_id: str
    work_dir: Path
    success: bool
    message: str

def _run_for_single_plan(run_id: str, index: int, total: int, plan: PlanInfo) -> ExteriorDoorsJobResult:
    """
    Pentru exterior_doors, intrările sunt:
      - plan.jpg din stage 'detections'
      - detections_all.json din stage 'count_objects'
    Output-urile se scriu în stage_work_dir (exterior_doors/<plan_id>/).
    """
    work_dir = plan.stage_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    # plan.jpg a fost copiat în stage 'detections'
    detections_dir = work_dir.parent.parent / "detections" / plan.plan_id
    plan_jpg = detections_dir / "plan.jpg"
    if not plan_jpg.exists():
        return ExteriorDoorsJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=False,
            message=f"Nu găsesc plan.jpg în {detections_dir}"
        )

    # detections_all.json e produs de stage 'count_objects'
    count_dir = work_dir.parent.parent / "count_objects" / plan.plan_id
    detections_all = count_dir / "detections_all.json"
    if not detections_all.exists():
        return ExteriorDoorsJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=False,
            message=f"Nu găsesc detections_all.json în {count_dir}"
        )

    try:
        print(
            f"[{STAGE_NAME}] ({index}/{total}) {plan.plan_id} → flood BLUE + classify "
            f"(cwd={work_dir})",
            flush=True,
        )

        ok, msg = run_exterior_doors_for_plan(
            plan_image=plan_jpg,
            detections_all_json=detections_all,
            work_dir=work_dir
        )

        return ExteriorDoorsJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=ok,
            message=msg
        )

    except Exception as e:
        return ExteriorDoorsJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=False,
            message=f"Eroare: {e}"
        )

def run_exterior_doors_for_run(run_id: str, max_parallel: int | None = None) -> List[ExteriorDoorsJobResult]:
    """
    Punct de intrare pentru etapa „exterior_doors”.
    Output-urile se vor regăsi în:
      new/runner/output/<RUN_ID>/exterior_doors/<plan_id>/
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
        max_parallel = max(2, min(cpu_count, total))

    print(f"⚙️  [{STAGE_NAME}] rulez cu max_parallel = {max_parallel}\n", flush=True)

    results: List[ExteriorDoorsJobResult] = []

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
                f"({res.work_dir}) → {res.message[:200]}",
                flush=True,
            )

    failed = [r for r in results if not r.success]
    if failed:
        print(f"\n⚠️ [{STAGE_NAME}] unele planuri au eșuat:")
        for r in failed:
            print(f"   - {r.plan_id}: {r.message[:300]}")
    else:
        print(f"\n✅ [{STAGE_NAME}] toate planurile au trecut etapa exterior_doors.")

    return results
