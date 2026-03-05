#!/usr/bin/env python3
"""
Re-generează 01_walls_from_coords.png strict din 00_flood_fill_lines_removed.png:
- Pereți = pixeli albi (BGR 255,255,255) din viz
- Skeletonizare 1px
- Salvare 01_walls_from_coords.png

Utilizare:
  cd holzbot-engine && python scripts/rerun_01_walls_from_flood_lines_removed.py [RUN_ID] [PLAN_ID]
  Exemplu: python scripts/rerun_01_walls_from_flood_lines_removed.py 90286a83-fe9a-47c8-83c1-549b6770e8f6 plan_02_cluster_2
"""

from pathlib import Path
import sys
import os
import cv2
import numpy as np

ENGINE_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ENGINE_ROOT / "output"


def main():
    run_id = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RUN_ID", "")).strip()
    plan_id = (sys.argv[2] if len(sys.argv) > 2 else os.environ.get("PLAN_ID", "")).strip()
    if not run_id or not plan_id:
        print("Usage: python rerun_01_walls_from_flood_lines_removed.py RUN_ID PLAN_ID")
        return 1

    output_dir = OUTPUT_ROOT / run_id / "scale" / plan_id / "cubicasa_steps" / "raster_processing" / "walls_from_coords"
    lines_removed_path = output_dir / "00_flood_fill_lines_removed.png"
    out_path = output_dir / "01_walls_from_coords.png"

    if not lines_removed_path.exists():
        print(f"Lipsește: {lines_removed_path}")
        return 1

    img = cv2.imread(str(lines_removed_path), cv2.IMREAD_COLOR)
    if img is None:
        print("Eroare la citire 00_flood_fill_lines_removed.png")
        return 1

    # În viz: alb (255,255,255) = pereți care rămân; roșu = eliminate; verde = flood
    # Extragem doar pixeli albi = pereți care rămân
    b, g, r = cv2.split(img)
    walls_mask = ((b == 255) & (g == 255) & (r == 255)).astype(np.uint8) * 255
    n_wall = int(np.sum(walls_mask > 0))
    print(f"Pixeli pereți (alb din viz): {n_wall}")

    if n_wall == 0:
        print("Nu există pixeli albi (pereți) în 00_flood_fill_lines_removed.png")
        segments_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        cv2.imwrite(str(out_path), segments_img)
        print(f"Salvat 01_walls_from_coords.png (gol)")
        return 0

    # Skeletonizare 1px
    try:
        from skimage.morphology import skeletonize
        binary_1px = (walls_mask > 0).astype(np.uint8)
        skel = skeletonize(binary_1px.astype(bool))
        walls_1px = (skel.astype(np.uint8)) * 255
        print("Skeleton 1px aplicat.")
    except Exception as e:
        print(f"Skeleton eșuat: {e}, salvez masca fără skeleton.")
        walls_1px = walls_mask

    segments_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    segments_img[walls_1px > 0] = [255, 255, 255]
    # Margine neagră 20px SUPLIMENTARĂ: mărim poza, conținutul rămâne în centru
    border_px = 20
    h, w = img.shape[0], img.shape[1]
    H, W = h + 2 * border_px, w + 2 * border_px
    out_img = np.zeros((H, W, 3), dtype=np.uint8)
    out_img[border_px : border_px + h, border_px : border_px + w] = segments_img
    cv2.imwrite(str(out_path), out_img)
    print(f"Salvat: {out_path} (cu margine {border_px}px suplimentară, {W}x{H})")
    print(f"Salvat: {out_path}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ENGINE_ROOT))
    exit(main())
