#!/usr/bin/env python3
"""
Rulează strict doar partea walls_from_coords (inclusiv balcon/terasă/wintergarden):
generate_walls_from_room_coordinates cu masca 02_ai_walls_closed.png existentă.

Utilizare:
  cd holzbot-engine && python scripts/run_walls_from_coords_only.py [RUN_ID] [PLAN_ID]
  Exemplu: python scripts/run_walls_from_coords_only.py 7cf2b2f7-ee15-418f-9fb1-8e2a673cf656 plan_01_cluster_1
  Sau:     RUN_ID=9f4602b9-e2e9-4f5a-aa31-8064a7969aef python scripts/run_walls_from_coords_only.py
"""

from pathlib import Path
import sys
import os
import cv2

ENGINE_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ENGINE_ROOT / "output"


def main():
    run_id = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RUN_ID", "7cf2b2f7-ee15-418f-9fb1-8e2a673cf656")).strip()
    plan_id = (sys.argv[2] if len(sys.argv) > 2 else os.environ.get("PLAN_ID", "plan_01_cluster_1")).strip()

    plan_dir = OUTPUT_ROOT / run_id / "scale" / plan_id
    steps_dir = plan_dir / "cubicasa_steps"
    raster_dir = steps_dir / "raster"

    original_path = steps_dir / "00_original.png"
    walls_path = steps_dir / "02_ai_walls_closed.png"

    if not plan_dir.is_dir():
        print(f"Nu există: {plan_dir}")
        return 1
    if not original_path.exists():
        print(f"Lipsește: {original_path}")
        return 1
    if not walls_path.exists():
        print(f"Lipsește: {walls_path}")
        return 1
    if not (raster_dir / "response.json").exists():
        print(f"Lipsește: {raster_dir / 'response.json'}")
        return 1

    original_img = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    walls_mask = cv2.imread(str(walls_path), cv2.IMREAD_GRAYSCALE)
    if original_img is None or walls_mask is None:
        print("Eroare la citire 00_original.png sau 02_ai_walls_closed.png")
        return 1

    # Aliniere dimensiuni dacă e cazul
    if walls_mask.shape[:2] != original_img.shape[:2]:
        walls_mask = cv2.resize(walls_mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    from cubicasa_detector.raster_processing import generate_walls_from_room_coordinates

    print(f"Run walls_from_coords (inclusiv balcon): run_id={run_id}, plan_id={plan_id}")
    result = generate_walls_from_room_coordinates(
        original_img,
        {},
        raster_dir,
        str(steps_dir),
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        initial_walls_mask_1px=walls_mask,
    )
    if not result:
        print("Done. result=None")
        return 1
    # Regenerează detections_review_data.json (și overlay-urile) ca în editor să apară corect tipurile (window/stairs/garage_door)
    try:
        from cubicasa_detector.raster_api import save_detections_review_image
        path_base, _path_rooms, _path_doors = save_detections_review_image(raster_dir)
        if path_base:
            print("Regenerat detections_review_data.json + detections_review_*.png (tipuri pentru editor).")
        else:
            print("Avertisment: save_detections_review_image nu a returnat imagine (09_interior/brute_steps pot lipsi).")
    except Exception as e:
        print(f"Avertisment: regenerare detections_review: {e}")
    print("Done. result=ok")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ENGINE_ROOT))
    exit(main())
