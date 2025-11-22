# new/runner/detections/jobs.py
from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from config.settings import (
    load_plan_infos,
    PlansListError,
    PlanInfo,
)

from .roboflow_import import run_roboflow_import
from .object_crops import run_object_crops


STAGE_NAME = "detections"


@dataclass
class DetectionJobResult:
    plan_id: str
    work_dir: Path
    success: bool
    message: str


def _prepare_workdir(plan: PlanInfo) -> Path:
    """
    Copiază imaginea de plan în directorul de lucru al etapei ca 'plan.jpg'.
    """
    work_dir = plan.stage_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    dest = work_dir / "plan.jpg"
    if not dest.exists():
        shutil.copy2(plan.plan_image, dest)

    return work_dir


def _build_env(run_id: str, plan_id: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    env["PLAN_ID"] = plan_id
    return env


def _run_for_single_plan(run_id: str, index: int, total: int, plan: PlanInfo) -> DetectionJobResult:
    work_dir = _prepare_workdir(plan)
    env = _build_env(run_id, plan.plan_id)

    try:
        print(
            f"[{STAGE_NAME}] ({index}/{total}) {plan.plan_id} → roboflow_import "
            f"(cwd={work_dir})",
            flush=True,
        )
        ok, msg = run_roboflow_import(env, work_dir)
        if not ok:
            return DetectionJobResult(
                plan_id=plan.plan_id,
                work_dir=work_dir,
                success=False,
                message=msg,
            )

        print(
            f"[{STAGE_NAME}] ({index}/{total}) {plan.plan_id} → object_crops "
            f"(cwd={work_dir})",
            flush=True,
        )
        ok, msg = run_object_crops(env, work_dir)
        if not ok:
            return DetectionJobResult(
                plan_id=plan.plan_id,
                work_dir=work_dir,
                success=False,
                message=msg,
            )

        return DetectionJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=True,
            message="OK",
        )

    except Exception as e:
        return DetectionJobResult(
            plan_id=plan.plan_id,
            work_dir=work_dir,
            success=False,
            message=f"Eroare neașteptată: {e}",
        )


def run_detections_for_run(run_id: str, max_parallel: int | None = None) -> List[DetectionJobResult]:
    """
    Punct de intrare pentru etapa „detections" (object examples).

    Toate output-urile se vor regăsi în:
      new/runner/output/<RUN_ID>/detections/<plan_id>/...
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

    results: List[DetectionJobResult] = []

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
        print(f"\n✅ [{STAGE_NAME}] toate planurile au trecut etapa detections.")

    return results