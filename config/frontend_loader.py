from __future__ import annotations
from pathlib import Path
import json
from typing import Dict

from .settings import RUNNER_ROOT, JOBS_ROOT


def load_frontend_data_for_run(run_id: str, job_root: Path | None = None) -> Dict:
    """
    Încarcă datele frontend pentru un run specific.
    
    Logica:
    1. Load fallback/default din RUNNER_ROOT
    2. Caută job_root bazat pe run_id (sau folosește cel furnizat)
    3. Load job specific (override)
    
    Args:
        run_id: ID-ul run-ului
        job_root: Path optional către job root (pentru a evita căutarea)
    
    Returns:
        Dict cu datele merged
    """
    data = {}
    
    # 1. Load Fallback/Default din RUNNER_ROOT
    for fname in ["fallback_frontend_data.json", "frontend_data.json"]:
        fpath = RUNNER_ROOT / fname
        if fpath.exists():
            try:
                print(f"🔹 Loading base config from: {fname}")
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                break  # Ne oprim la primul găsit
            except Exception as e:
                print(f"⚠️ Error loading base config {fname}: {e}")
    
    # 2. Găsește job_root dacă nu e furnizat (preferăm path-ul exact unde API-ul scrie pdf_company/pdf_assets)
    if job_root is None:
        possible_paths = [
            JOBS_ROOT / run_id,
            JOBS_ROOT / f"segmentation_job_{run_id}",
        ]
        for path in possible_paths:
            if path.exists() and path.is_dir():
                job_root = path
                break
        if job_root is None:
            for jdir in JOBS_ROOT.glob("*"):
                if jdir.is_dir() and run_id in jdir.name:
                    job_root = jdir
                    break
    
    # 3. Load Job Specific (override) - CRITIC
    if job_root:
        job_file = job_root / "frontend_data.json"
        if job_file.exists():
            try:
                print(f"🔹 Loading JOB config from: {job_file}")
                with open(job_file, "r", encoding="utf-8") as f:
                    job_data = json.load(f)
                data.update(job_data)  # Merge: valorile din job suprascriu baza
                print(f"✅ Config merged. Keys: {list(data.keys())}")
            except Exception as e:
                print(f"⚠️ Error loading job config: {e}")
        else:
            print(f"⚠️ Job specific config NOT found at: {job_file}")
    else:
        print(f"⚠️ No job_root found for run_id: {run_id}")
    
    return data