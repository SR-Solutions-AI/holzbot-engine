# new/runner/config/settings.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

# Rădăcina folderului "runner" (cel din screenshot)
RUNNER_ROOT: Path = Path(__file__).resolve().parents[1]

# Rădăcina proiectului – presupunem că e cu un nivel mai sus de "runner"
# (acolo unde ai export_objects/, count_objects/, runs/, etc.)
PROJECT_ROOT: Path = RUNNER_ROOT

# Folderul cu rulari (input-uri, planuri etc.)
# ex: <project_root>/runs/<RUN_ID>/
RUNS_ROOT: Path = PROJECT_ROOT / "runs"

# Toate output-urile pipeline-ului merg aici:
#   new/runner/output/<RUN_ID>/<stage_name>/<plan_id>/...
OUTPUT_ROOT: Path = RUNNER_ROOT / "output"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Folderul de "job-uri" interne ale orchestratorului (segmenter, etc.)
# ex: <project_root>/jobs/segmentation_job_YYYYmmdd_HHMMSS/
JOBS_ROOT: Path = PROJECT_ROOT / "jobs"
JOBS_ROOT.mkdir(parents=True, exist_ok=True)


class PlansListError(RuntimeError):
    """Erori legate de plans_list.json sau de structura planurilor."""


@dataclass
class PlanInfo:
    """
    Reprezintă un plan (o planșă) de intrare.

    - source_path: path-ul brut citit din plans_list.json
    - plan_image: path-ul imaginii sursă (jpg/png)
    - plan_id: identificator frumos (plan_01_living, etc.)
    - stage_work_dir: directorul în care rulează scripturile pentru etapa curentă
    """
    source_path: Path
    plan_image: Path
    plan_id: str
    stage_work_dir: Path


def _slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "plan"


def get_run_dir(run_id: str) -> Path:
    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists():
        raise PlansListError(f"Nu există folderul de run: {run_dir}")
    return run_dir


def get_output_root_for_run(run_id: str) -> Path:
    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_plan_infos(run_id: str, stage_name: str) -> List[PlanInfo]:
    """
    Încarcă planurile din runs/<RUN_ID>/plans_list.json și construiește
    câte un PlanInfo pentru fiecare.

    Acceptă:
      - căi către directoare care conțin un plan.jpg
      - sau căi directe către imagini (jpg/png).

    stage_name este ceva de genul "segmenter", "detections" etc. și va fi folosit
    pentru a construi path-ul de lucru:

      new/runner/output/<RUN_ID>/<stage_name>/<plan_id>/
    """
    run_dir = get_run_dir(run_id)
    plans_json = run_dir / "plans_list.json"

    if not plans_json.exists():
        raise PlansListError(
            f"Nu găsesc {plans_json}. Asigură-te că pasul anterior a rulat."
        )

    try:
        data = json.loads(plans_json.read_text(encoding="utf-8"))
    except Exception as e:
        raise PlansListError(f"Eroare la citirea {plans_json}: {e}") from e

    raw_plans = data.get("plans") or []
    if not raw_plans:
        raise PlansListError(f"{plans_json} nu conține niciun plan ('plans' e gol).")

    out_root = get_output_root_for_run(run_id)
    stage_root = out_root / stage_name
    stage_root.mkdir(parents=True, exist_ok=True)

    plan_infos: List[PlanInfo] = []

    for idx, item in enumerate(raw_plans, start=1):
        src = Path(item)

        if src.is_dir():
            # caz: pentru fiecare plan ai deja un folder cu plan.jpg înăuntru
            plan_img = src / "plan.jpg"
        else:
            # caz: în plans_list.json ai calea directă către imagine (jpg/png)
            plan_img = src

        if not plan_img.exists():
            raise PlansListError(
                f"Pentru intrarea #{idx} din {plans_json}, nu găsesc imaginea de plan: {plan_img}"
            )

        nice_name = _slugify(plan_img.stem)
        plan_id = f"plan_{idx:02d}_{nice_name}"

        stage_work_dir = stage_root / plan_id
        stage_work_dir.mkdir(parents=True, exist_ok=True)

        plan_infos.append(
            PlanInfo(
                source_path=src,
                plan_image=plan_img,
                plan_id=plan_id,
                stage_work_dir=stage_work_dir,
            )
        )

    return plan_infos


def build_job_root(job_id: str | None = None, prefix: str = "job") -> Path:
    """
    Creează un director de job sub PROJECT_ROOT/jobs.

    Dacă job_id nu este dat, generează automat unul de forma
    '<prefix>_YYYYmmdd_HHMMSS'.
    """
    if job_id is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"{prefix}_{ts}"

    job_root = JOBS_ROOT / job_id
    job_root.mkdir(parents=True, exist_ok=True)
    return job_root
