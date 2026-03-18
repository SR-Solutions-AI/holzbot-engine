#!/usr/bin/env python3
"""
Test local pentru asocierea camerelor (poligoane 09_interior <-> room_scales după arie).
Rulează fluxul din raster_api (extragere poligoane + salvare detections_review_data),
apoi verifică că roomType atribuit corespunde cu matching-ul pe arie.

Utilizare:
  cd holzbot-engine
  .venv/bin/python scripts/test_room_association.py [path]

  path = director cubicasa_steps (ex: output/RUN_ID/scale/plan_01_cluster_1/cubicasa_steps)
        sau director raster (ex: .../cubicasa_steps/raster).
  Dacă lipsește, folosește un run din output/ (primul găsit cu raster + room_scales).
"""

from pathlib import Path
import json
import sys
from typing import Optional

import cv2
import numpy as np

ENGINE_ROOT = Path(__file__).resolve().parent.parent


def find_default_raster_dir() -> Optional[Path]:
    """Găsește un raster_dir valid sub output/ (primul plan cu raster + room_scales)."""
    out = ENGINE_ROOT / "output"
    if not out.exists():
        return None
    for run_dir in sorted(out.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        scale = run_dir / "scale"
        if not scale.exists():
            continue
        for plan_dir in scale.iterdir():
            if not plan_dir.is_dir():
                continue
            raster = plan_dir / "cubicasa_steps" / "raster"
            rs = plan_dir / "cubicasa_steps" / "raster_processing" / "walls_from_coords" / "room_scales.json"
            if raster.exists() and rs.exists():
                return raster
    return None


def main() -> int:
    if len(sys.argv) >= 2:
        p = Path(sys.argv[1]).resolve()
        if (p / "raster").exists():
            raster_dir = p / "raster"
        elif (p.parent / "raster_processing").exists():
            raster_dir = p
        else:
            raster_dir = p
    else:
        raster_dir = find_default_raster_dir()
        if not raster_dir:
            print("Nu s-a găsit niciun run cu raster + room_scales. Da path explicit.")
            print("Ex: .venv/bin/python scripts/test_room_association.py output/RUN_ID/scale/plan_01_cluster_1/cubicasa_steps")
            return 1
        print(f"Folosesc: {raster_dir}")

    # 1) Rulează fluxul real
    sys.path.insert(0, str(ENGINE_ROOT))
    from cubicasa_detector.raster_api import save_detections_review_image

    print("Rulez save_detections_review_image(...)")
    path_base, _path_rooms, _path_doors = save_detections_review_image(raster_dir)
    if not path_base:
        print("Eroare: save_detections_review_image nu a returnat imagine de bază (lipsesc fișiere în raster?).")
        return 1

    # 2) Încarcă rezultatul și room_scales
    review_path = raster_dir / "detections_review_data.json"
    room_scales_path = raster_dir.parent / "raster_processing" / "walls_from_coords" / "room_scales.json"
    interior_path = raster_dir.parent / "raster_processing" / "walls_from_coords" / "09_interior.png"

    if not review_path.exists():
        print(f"Lipsește: {review_path}")
        return 1
    with open(review_path, "r", encoding="utf-8") as f:
        review = json.load(f)
    req_w = review.get("imageWidth") or 0
    req_h = review.get("imageHeight") or 0
    rooms = review.get("rooms") or []

    if not rooms:
        print("detections_review_data.json: 0 camere. Verifică 09_interior.png și fluxul de extragere.")
        return 1

    w_orig, h_orig = req_w, req_h
    if interior_path.exists():
        img = cv2.imread(str(interior_path))
        if img is not None and img.size > 0:
            h_orig, w_orig = img.shape[:2]

    rs = {}
    if room_scales_path.exists():
        with open(room_scales_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rs = data.get("room_scales") or data.get("rooms") or {}

    # 3) Calculează arii (în spațiul original) pentru fiecare room din review
    scale_area = (w_orig * h_orig) / (req_w * req_h) if (req_w and req_h) else 1.0
    poly_areas = []
    for r in rooms:
        pts = r.get("points") or []
        if len(pts) < 3:
            poly_areas.append(0.0)
            continue
        arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        area_req = float(cv2.contourArea(arr))
        poly_areas.append(area_req * scale_area)

    # 4) Ordine după arie: poligoane vs room_scales
    room_indices_sorted = sorted(
        (k for k in rs.keys() if str(k).isdigit()),
        key=lambda k: (rs.get(k) or rs.get(str(k)) or {}).get("area_px") or 0,
    )
    poly_order_by_area = sorted(range(len(poly_areas)), key=lambda i: poly_areas[i])

    print("\n--- Poligoane (din detections_review_data) sortate după arie (orig) ---")
    for rank, idx in enumerate(poly_order_by_area):
        area_orig = poly_areas[idx]
        room_type = rooms[idx].get("roomType") or "Raum"
        n_pts = len(rooms[idx].get("points") or [])
        print(f"  rank {rank}: poly_idx={idx}  area_orig={area_orig:.0f}  roomType={room_type!r}  points={n_pts}")

    print("\n--- room_scales sortate după area_px ---")
    for rank, k in enumerate(room_indices_sorted):
        r = rs.get(k) or rs.get(str(k)) or {}
        area_px = r.get("area_px") or 0
        room_type = r.get("room_type") or "Raum"
        print(f"  rank {rank}: key={k!r}  area_px={area_px}  room_type={room_type!r}")

    # 5) Verificare: la matching pe arie, poly_order_by_area[rank] ar trebui să aibă roomType =
    #    room_scales[room_indices_sorted[rank]].room_type
    print("\n--- Verificare asociere pe arie ---")
    ok = True
    for rank in range(min(len(poly_order_by_area), len(room_indices_sorted))):
        poly_idx = poly_order_by_area[rank]
        room_key = room_indices_sorted[rank]
        expected = (rs.get(room_key) or rs.get(str(room_key)) or {}).get("room_type") or "Raum"
        actual = rooms[poly_idx].get("roomType") or "Raum"
        match = "OK" if actual == expected else "MISMATCH"
        if actual != expected:
            ok = False
        print(f"  rank {rank}: polygon {poly_idx} -> expected {expected!r}  actual {actual!r}  [{match}]")

    if ok and len(poly_order_by_area) == len(room_indices_sorted):
        print("\n✅ Asociere pe arie: OK (toate etichetele corespund ordinii după arie).")
    elif not ok:
        print("\n❌ Asociere: unele etichete nu corespund matching-ului pe arie.")
    else:
        print(f"\n⚠️ Număr diferit: {len(rooms)} poligoane vs {len(room_indices_sorted)} intrări room_scales.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
