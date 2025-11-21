# new/runner/detections/object_crops.py
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, Dict, List

# Import crop scripts (le vom crea mai jos)
from .crop_scripts import crop_door, crop_double_door, crop_window, crop_double_window


def run_object_crops(env: Dict[str, str], work_dir: Path) -> Tuple[bool, str]:
    """
    Rulează crop-urile pentru toate tipurile de obiecte detectate.
    - env: environment complet
    - work_dir: directorul în care avem plan.jpg și export_objects/detections.json
    """
    plan_jpg = work_dir / "plan.jpg"
    detections_json = work_dir / "export_objects" / "detections.json"
    exports_dir = work_dir / "export_objects" / "exports"

    if not plan_jpg.exists():
        return False, f"Nu găsesc plan.jpg în {work_dir}"

    if not detections_json.exists():
        return False, f"Nu găsesc detections.json în {work_dir / 'export_objects'}"

    exports_dir.mkdir(parents=True, exist_ok=True)

    # Scripturile de crop (funcții, nu subprocess)
    crop_functions = [
        ("door", crop_door.process),
        ("double_door", crop_double_door.process),
        ("window", crop_window.process),
        ("double_window", crop_double_window.process),
    ]

    results: List[Tuple[str, str, str]] = []

    print(f"  ✂️  Rulez {len(crop_functions)} crop scripts în paralel...")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                func,
                image_path=plan_jpg,
                json_path=detections_json,
                out_root=exports_dir
            ): name
            for name, func in crop_functions
        }

        for fut in as_completed(futures):
            name = futures[fut]
            try:
                count = fut.result()
                results.append((name, "OK", f"{count} crops"))
                print(f"    ✅ {name}: {count} crops")
            except Exception as e:
                results.append((name, "ERR", str(e)))
                print(f"    ❌ {name}: {e}")

    # Verifică dacă au fost generate crop-uri
    all_crops = list(exports_dir.rglob("*.png")) + list(exports_dir.rglob("*.jpg"))
    total_crops = len(all_crops)

    summary = "\n".join([f"  {name}: {status} ({msg})" for name, status, msg in results])

    if total_crops == 0:
        return False, f"Niciun crop generat!\n{summary}"

    return True, f"{total_crops} crop-uri generate în {exports_dir.relative_to(work_dir)}\n{summary}"