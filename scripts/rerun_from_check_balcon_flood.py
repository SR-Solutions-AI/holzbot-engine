#!/usr/bin/env python3
"""
Re-generează de la check_balcon/flood_fill.png în colo: 00_flood_fill_cleanup_viz → 01_walls_from_coords.

check_balcon/flood_fill.png este salvat de save_check_garage_image: pereți albi, flood cyan (BGR 255,200,100 blend).
Din ea extragem: masca de pereți (alb) și masca de flood (cyan), apoi același pipeline ca la strip.

Utilizare:
  cd holzbot-engine && python scripts/rerun_from_check_balcon_flood.py [RUN_ID] [PLAN_ID]
"""

from pathlib import Path
import sys
import cv2
import numpy as np

ENGINE_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ENGINE_ROOT / "output"


def main():
    run_id = (sys.argv[1] if len(sys.argv) > 1 else __import__("os").environ.get("RUN_ID", "")).strip()
    plan_id = (sys.argv[2] if len(sys.argv) > 2 else __import__("os").environ.get("PLAN_ID", "")).strip()
    if not run_id or not plan_id:
        print("Usage: python rerun_from_check_balcon_flood.py RUN_ID PLAN_ID")
        return 1

    output_dir = OUTPUT_ROOT / run_id / "scale" / plan_id / "cubicasa_steps" / "raster_processing" / "walls_from_coords"
    flood_path = output_dir / "check_balcon" / "flood_fill.png"

    if not flood_path.exists():
        print(f"Lipsește: {flood_path}")
        return 1

    img = cv2.imread(str(flood_path), cv2.IMREAD_COLOR)
    if img is None:
        print("Eroare la citire check_balcon/flood_fill.png")
        return 1

    b, g, r = cv2.split(img)
    # Pereți = alb (save_check_garage_image păstrează alb unde nu e overlay)
    accepted_wall_segments_mask = ((b >= 250) & (g >= 250) & (r >= 250)).astype(np.uint8) * 255
    # Flood = cyan din save_check_garage_image: BGR (255,200,100)*0.6 pe negru = (153,120,60); toleranță mică
    flood_any = np.zeros_like(accepted_wall_segments_mask)
    flood_color = (np.abs(b.astype(np.int32) - 153) <= 15) & (np.abs(g.astype(np.int32) - 120) <= 15) & (np.abs(r.astype(np.int32) - 60) <= 15)
    flood_any[flood_color & (accepted_wall_segments_mask == 0)] = 255

    h_orig, w_orig = accepted_wall_segments_mask.shape[:2]
    n_wall = int(np.sum(accepted_wall_segments_mask > 0))
    n_flood = int(np.sum(flood_any > 0))
    print(f"check_balcon/flood_fill.png: {n_wall} pixeli pereți, {n_flood} pixeli flood")

    # 00_flood_fill_cleanup_viz.png – verde = flood, alb = pereți
    flood_cleanup_viz = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    flood_cleanup_viz[flood_any > 0] = [0, 180, 0]
    flood_cleanup_viz[accepted_wall_segments_mask > 0] = [255, 255, 255]
    cv2.imwrite(str(output_dir / "00_flood_fill_cleanup_viz.png"), flood_cleanup_viz)
    print("Salvat: 00_flood_fill_cleanup_viz.png (verde = flood, alb = pereți)")

    # Eliminare pixeli pereți cu ≥2 vecini flood opuși, <3 vecini pereți
    removed_total = 0
    for _round in range(5000):
        wall_coords = np.argwhere(accepted_wall_segments_mask > 0)
        if wall_coords.size == 0:
            break
        walls_to_remove = np.zeros((h_orig, w_orig), dtype=np.uint8)
        for idx in range(len(wall_coords)):
            y, x = int(wall_coords[idx, 0]), int(wall_coords[idx, 1])
            flood_n = 1 if y > 0 and flood_any[y - 1, x] > 0 else 0
            flood_s = 1 if y < h_orig - 1 and flood_any[y + 1, x] > 0 else 0
            flood_e = 1 if x < w_orig - 1 and flood_any[y, x + 1] > 0 else 0
            flood_w = 1 if x > 0 and flood_any[y, x - 1] > 0 else 0
            if not ((flood_n and flood_s) or (flood_e and flood_w)):
                continue
            n_wall_neigh = sum(
                1 for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dx or dy) and 0 <= y + dy < h_orig and 0 <= x + dx < w_orig and accepted_wall_segments_mask[y + dy, x + dx] != 0
            )
            if n_wall_neigh >= 3:
                continue
            walls_to_remove[y, x] = 255
        round_removed = int(np.sum(walls_to_remove > 0))
        if _round == 0:
            lines_removed_viz = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
            lines_removed_viz[flood_any > 0] = [0, 180, 0]
            lines_removed_viz[accepted_wall_segments_mask > 0] = [255, 255, 255]
            lines_removed_viz[walls_to_remove > 0] = [0, 0, 255]
            cv2.imwrite(str(output_dir / "00_flood_fill_lines_removed.png"), lines_removed_viz)
            print("Salvat: 00_flood_fill_lines_removed.png (roșu = eliminate, alb = rămân)")
            white_only = (accepted_wall_segments_mask > 0) & (walls_to_remove == 0)
            flood_test = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
            flood_test[white_only] = [255, 255, 255]
            cv2.imwrite(str(output_dir / "00_flood_test.png"), flood_test)
            print("Salvat: 00_flood_test.png")
        if round_removed == 0:
            break
        accepted_wall_segments_mask[walls_to_remove > 0] = 0
        removed_total += round_removed
        if _round == 0:
            break
    print(f"Eliminat {removed_total} pixeli (linii suplimentare).")

    # Skeletonizare 1px
    try:
        from skimage.morphology import skeletonize
        binary_1px = (accepted_wall_segments_mask > 0).astype(np.uint8)
        skel = skeletonize(binary_1px.astype(bool))
        walls_1px = (skel.astype(np.uint8)) * 255
        print("Skeleton 1px aplicat.")
    except Exception as e:
        print(f"Skeleton eșuat: {e}, folosesc masca așa cum e.")
        walls_1px = accepted_wall_segments_mask

    # 01_walls_from_coords.png cu margine 20px
    segments_img = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    segments_img[walls_1px > 0] = [255, 255, 255]
    border_px = 20
    H, W = h_orig + 2 * border_px, w_orig + 2 * border_px
    out_img = np.zeros((H, W, 3), dtype=np.uint8)
    out_img[border_px : border_px + h_orig, border_px : border_px + w_orig] = segments_img
    cv2.imwrite(str(output_dir / "01_walls_from_coords.png"), out_img)
    print(f"Salvat: 01_walls_from_coords.png (margine {border_px}px, {W}x{H})")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ENGINE_ROOT))
    exit(main())
