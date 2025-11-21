# new/runner/exterior_doors/pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple

from .flood_blue import compute_blue_mask
from .classify import classify_exterior_doors


def run_exterior_doors_for_plan(
    plan_image: Path,
    detections_all_json: Path,
    work_dir: Path
) -> Tuple[bool, str]:
    """
    Rulează pipeline-ul pentru un singur plan:
      1) Flood BLUE (exterior)
      2) Clasificare uși după distanță la BLUE
    
    Returns:
        (success, message cu path-uri către toate outputs)
    """
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Generează blue mask
        blue_mask_path, blue_overlay_path = compute_blue_mask(plan_image, work_dir)
        
        # Step 2: Clasifică uși
        out_json, out_overlay, out_flood_marked = classify_exterior_doors(
            plan_image, 
            blue_mask_path, 
            detections_all_json, 
            work_dir
        )
        
        # ✅ Check dacă a returnat None (eroare în classify)
        if out_json is None or out_overlay is None or out_flood_marked is None:
            return False, "Eroare la clasificarea ușilor (returnare None)"
        
        msg = (
            f"OK | "
            f"blue_mask={blue_mask_path.name}, "
            f"doors_json={out_json.name}, "
            f"overlay={out_overlay.name}, "
            f"flood_marked={out_flood_marked.name}"
        )
        return True, msg
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Error: {e}"