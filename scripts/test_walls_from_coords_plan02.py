#!/usr/bin/env python3
"""Testează generate_walls_from_room_coordinates pentru plan_02 (crop) cu modificările coverage."""

from pathlib import Path
import json
import cv2

# Paths pentru job 389c1bc6, plan_02
ENGINE_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ENGINE_ROOT / "output" / "389c1bc6-2273-42e0-9350-a353c1b63bce"
PLAN_DIR = OUTPUT_ROOT / "scale" / "plan_02_cluster_7"
STEPS_DIR = PLAN_DIR / "cubicasa_steps"
RASTER_DIR = STEPS_DIR / "raster"


def main():
    original_path = STEPS_DIR / "00_original.png"
    config_path = RASTER_DIR / "brute_force_best_config.json"
    if not original_path.exists() or not config_path.exists():
        print(f"Missing: original={original_path.exists()}, config={config_path.exists()}")
        return 1

    original_img = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    with open(config_path) as f:
        best_config = json.load(f)

    # Ajustare best_config: position e listă [x,y], detector-ul poate folosi tuple
    if "position" in best_config and isinstance(best_config["position"], list):
        best_config["position"] = tuple(best_config["position"])

    from cubicasa_detector.raster_processing import generate_walls_from_room_coordinates

    print("Run generate_walls_from_room_coordinates (plan_02, crop)...")
    result = generate_walls_from_room_coordinates(
        original_img,
        best_config,
        RASTER_DIR,
        str(STEPS_DIR),
        gemini_api_key=None,
    )
    print("Done.", "result=", "ok" if result else "None")
    return 0 if result else 1


if __name__ == "__main__":
    exit(main())
