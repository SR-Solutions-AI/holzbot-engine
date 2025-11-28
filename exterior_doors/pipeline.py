# new/runner/exterior_doors/pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple

from .classify import classify_exterior_doors

def run_exterior_doors_for_plan(
    plan_image: Path,
    detections_all_json: Path,
    work_dir: Path
) -> Tuple[bool, str]:
    """
    Pipeline: CubiCasa Mask + Distance Classification
    """
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Găsește masca CubiCasa
        run_dir = work_dir.parent.parent
        plan_name = work_dir.name
        
        cubicasa_mask = run_dir / "scale" / plan_name / "cubicasa_steps" / "03_outdoor_mask.png"
        
        if not cubicasa_mask.exists():
            return False, f"CubiCasa mask missing at: {cubicasa_mask}"
            
        print(f"       ♻️  Using CubiCasa Mask: {cubicasa_mask.name}")
        
        # 2. Clasifică folosind scriptul robust (classify.py)
        out_json, out_img = classify_exterior_doors(
            plan_image=plan_image,
            outdoor_mask_path=cubicasa_mask,
            detections_all_json=detections_all_json,
            out_dir=work_dir
        )
        
        return True, f"OK | json={out_json.name}, img={out_img.name}"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Error: {e}"