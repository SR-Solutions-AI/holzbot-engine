#!/usr/bin/env python3
"""
Generează o imagine: blueprint + poligoane camere + etichete (roomType) peste fiecare cameră.
Pentru verificare vizuală că asocierea etichetelor este corectă.

Utilizare:
  cd holzbot-engine
  .venv/bin/python scripts/draw_room_labels_overlay.py [path]

  path = director cubicasa_steps sau .../cubicasa_steps/raster.
  Salvează: raster/detections_review_rooms_with_labels.png
"""

from pathlib import Path
import json
import sys
from typing import Optional

import cv2
import numpy as np

ENGINE_ROOT = Path(__file__).resolve().parent.parent


def find_default_raster_dir() -> Optional[Path]:
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
            if raster.exists() and (raster / "detections_review_data.json").exists():
                return raster
    return None


def _room_color_bgr(i: int):
    import colorsys
    n = 48
    h = (i * 137) % n / n
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


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
            print("Nu s-a găsit niciun run cu detections_review_data.json. Da path explicit.")
            return 1
        print(f"Folosesc: {raster_dir}")

    review_path = raster_dir / "detections_review_data.json"
    base_path = raster_dir / "detections_review_base.png"
    if not review_path.exists():
        print(f"Lipsește: {review_path} – rulează mai întâi save_detections_review_image (ex: test_room_association.py).")
        return 1
    if not base_path.exists():
        print(f"Lipsește: {base_path}")
        return 1

    with open(review_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rooms = data.get("rooms") or []
    if not rooms:
        print("Nicio cameră în detections_review_data.json.")
        return 1

    base = cv2.imread(str(base_path))
    if base is None:
        print("Nu s-a putut încărca imaginea de bază.")
        return 1
    overlay = base.copy()
    h_img, w_img = overlay.shape[:2]

    for i, room in enumerate(rooms):
        pts_raw = room.get("points") or []
        if len(pts_raw) < 3:
            continue
        pts = np.array(pts_raw, dtype=np.int32).reshape(-1, 1, 2)
        color = _room_color_bgr(i)
        # fill semi-transparent
        fill_layer = overlay.copy()
        cv2.fillPoly(fill_layer, [pts], color)
        overlay = cv2.addWeighted(fill_layer, 0.35, overlay, 0.65, 0).astype(np.uint8)
        cv2.polylines(overlay, [pts], True, color, 2)
        # label: roomName (nume în germană de la Gemini) sau roomType la centroid
        M = cv2.moments(pts)
        if M["m00"] and M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            label = (room.get("roomName") or room.get("room_name") or room.get("roomType") or "Raum").strip() or "Raum"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale_font = max(0.5, min(1.2, w_img / 1200))
            thick = max(1, int(scale_font * 2))
            (tw, th), _ = cv2.getTextSize(label, font, scale_font, thick)
            # background rectangle pentru lizibilitate
            pad = 4
            x1 = max(0, cx - tw // 2 - pad)
            y1 = max(0, cy - th // 2 - pad)
            x2 = min(w_img, cx + tw // 2 + pad)
            y2 = min(h_img, cy + th // 2 + pad)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                overlay, label, (x1 + pad, y2 - pad),
                font, scale_font, (255, 255, 255), thick, cv2.LINE_AA,
            )

    out_path = raster_dir / "detections_review_rooms_with_labels.png"
    if not cv2.imwrite(str(out_path), overlay):
        print("Eroare la scrierea imaginii.")
        return 1
    print(f"Salvat: {out_path}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ENGINE_ROOT))
    sys.exit(main())
