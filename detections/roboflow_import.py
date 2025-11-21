# new/runner/detections/roboflow_import.py
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Tuple, Dict

import requests


def run_roboflow_import(env: Dict[str, str], work_dir: Path) -> Tuple[bool, str]:
    """
    Importă detecții de la Roboflow pentru un plan.
    - env: environment complet (conține RUN_ID / PLAN_ID + Roboflow credentials)
    - work_dir: directorul în care se află plan.jpg și unde vrem să salvăm detections.json
    """
    API_KEY = os.getenv("ROBOFLOW_API_KEY", "").strip()
    PROJECT = os.getenv("ROBOFLOW_PROJECT", "house-plan-uwkew").strip()
    VERSION = os.getenv("ROBOFLOW_VERSION", "5").strip()
    CONF = int(os.getenv("ROBOFLOW_CONFIDENCE", "50"))
    OVERLAP = int(os.getenv("ROBOFLOW_OVERLAP", "30"))

    if not API_KEY:
        return False, "ROBOFLOW_API_KEY lipsește din environment"

    plan_jpg = work_dir / "plan.jpg"
    if not plan_jpg.exists():
        return False, f"Nu găsesc plan.jpg în {work_dir}"

    # Endpoint Roboflow
    url = f"https://detect.roboflow.com/{PROJECT}/{VERSION}"
    params = {
        "api_key": API_KEY,
        "confidence": CONF,
        "overlap": OVERLAP
    }

    print(f"  🔍 Roboflow API: {url} (conf={CONF}, overlap={OVERLAP})")
    start = time.time()

    try:
        with open(plan_jpg, "rb") as f:
            files = {"file": ("plan.jpg", f, "image/jpeg")}
            r = requests.post(url, params=params, files=files, timeout=120)
    except Exception as e:
        return False, f"Request eșuat: {e}"

    elapsed = time.time() - start

    if r.status_code != 200:
        return False, f"Roboflow HTTP {r.status_code}: {r.text[:400]}"

    try:
        result = r.json()
    except Exception:
        return False, f"Răspuns non-JSON: {r.text[:400]}"

    preds = result.get("predictions", [])
    print(f"  ✅ {len(preds)} detecții în {elapsed:.2f}s")

    # Salvează detections.json în export_objects/ (ca în scriptul vechi)
    detections_dir = work_dir / "export_objects"
    detections_dir.mkdir(parents=True, exist_ok=True)
    detections_file = detections_dir / "detections.json"

    detections_file.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    return True, f"{len(preds)} detecții salvate în {detections_file.relative_to(work_dir)}"