#!/usr/bin/env python3
"""
Re-generează de la 00_flood_fill_cleanup_viz (în culori) până la 01_walls_from_coords,
pornind de la masca din terasa_balcon_strip (walls_without_terasa_balcon.png).

Pași:
  1. Încarcă terasa_balcon_strip/walls_without_terasa_balcon.png (alb = pereți)
  2. Flood fill din colțuri/margini → exterior
  3. Salvează 00_flood_fill_cleanup_viz.png (verde = exterior, alb = pereți)
  4. Elimină pixelii de perete cu ≥2 vecini flood opuși (N-S sau E-W), <3 vecini pereți
  5. Salvează 00_flood_fill_lines_removed.png (verde, alb, roșu = eliminate)
  6. Salvează 00_flood_test.png (doar pereții care rămân)
  7. Skeletonizare 1px
  8. Salvează 01_walls_from_coords.png (cu margine neagră 20px)

Utilizare:
  cd holzbot-engine && python scripts/rerun_from_flood_cleanup_after_strip.py [RUN_ID] [PLAN_ID]
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
        print("Usage: python rerun_from_flood_cleanup_after_strip.py RUN_ID PLAN_ID")
        return 1

    output_dir = OUTPUT_ROOT / run_id / "scale" / plan_id / "cubicasa_steps" / "raster_processing" / "walls_from_coords"
    strip_dir = output_dir / "terasa_balcon_strip"
    mask_path = strip_dir / "walls_without_terasa_balcon.png"

    if not mask_path.exists():
        print(f"Lipsește: {mask_path}")
        return 1

    img = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if img is None:
        print("Eroare la citire walls_without_terasa_balcon.png")
        return 1

    # Alb = pereți
    b, g, r = cv2.split(img)
    accepted_wall_segments_mask = ((b == 255) & (g == 255) & (r == 255)).astype(np.uint8) * 255
    h_orig, w_orig = accepted_wall_segments_mask.shape[:2]
    total_non_black = int(np.sum(accepted_wall_segments_mask > 0))
    print(f"Pixeli pereți (din terasa_balcon_strip): {total_non_black}")

    flood_base = np.where(accepted_wall_segments_mask > 0, 0, 255).astype(np.uint8)
    seeds = [(0, 0), (w_orig - 1, 0), (0, h_orig - 1), (w_orig - 1, h_orig - 1)]
    step = max(1, min(50, w_orig // 10, h_orig // 10))
    for px in range(0, w_orig, step):
        seeds.append((px, 0))
        seeds.append((px, h_orig - 1))
    for py in range(0, h_orig, step):
        seeds.append((0, py))
        seeds.append((w_orig - 1, py))
    flood_any = np.zeros((h_orig, w_orig), dtype=np.uint8)
    for cx, cy in seeds:
        if cy >= h_orig or cx >= w_orig or flood_base[cy, cx] != 255:
            continue
        region_mask = np.zeros((h_orig + 2, w_orig + 2), dtype=np.uint8)
        cv2.floodFill(flood_base.copy(), region_mask, (cx, cy), 128, None, None, cv2.FLOODFILL_MASK_ONLY | 4)
        flood_any[region_mask[1:-1, 1:-1] > 0] = 255

    # 00_flood_fill_cleanup_viz.png – ÎN CULORI: verde = exterior, alb = pereți
    flood_cleanup_viz = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    flood_cleanup_viz[flood_any > 0] = [0, 180, 0]
    flood_cleanup_viz[accepted_wall_segments_mask > 0] = [255, 255, 255]
    cv2.imwrite(str(output_dir / "00_flood_fill_cleanup_viz.png"), flood_cleanup_viz)
    print("Salvat: 00_flood_fill_cleanup_viz.png (verde = exterior, alb = pereți)")

    # Eliminare pixeli cu ≥2 vecini flood opuși, <3 vecini pereți
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
            n_wall = sum(
                1 for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dx or dy) and 0 <= y + dy < h_orig and 0 <= x + dx < w_orig and accepted_wall_segments_mask[y + dy, x + dx] != 0
            )
            if n_wall >= 3:
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

    # 01_walls_from_coords.png cu margine 20px suplimentară
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
