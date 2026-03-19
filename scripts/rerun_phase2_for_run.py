#!/usr/bin/env python3
"""Re-rulare Phase 2 (room_scales cu per-crop Gemini) pentru un run existent."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ENGINE = Path(__file__).resolve().parents[1]
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))
os.chdir(_ENGINE)

def _load_dotenv():
    for p in (_ENGINE / ".env", Path.cwd() / ".env", Path.home() / ".env"):
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            except Exception:
                pass
            break

def main():
    _load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY lipsește. Setează variabila sau .env.")
        return 1
    run_id = sys.argv[1] if len(sys.argv) > 1 else "ae488c37-11d8-4a30-81c3-1c0e997ade59"
    from config.settings import load_plan_infos
    from cubicasa_detector.jobs import run_cubicasa_phase2
    plans = load_plan_infos(run_id, "scale")
    if not plans:
        print(f"Nu s-au găsit planuri pentru run {run_id}")
        return 1
    for plan in plans:
        print(f"Rulare Phase 2 pentru {plan.plan_id} ...")
        run_cubicasa_phase2(plan.stage_work_dir)
        print(f"OK {plan.plan_id}")
    print("Gata.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
