#!/usr/bin/env python3
"""
Test rapid: verifică că pasul „flood fill din 4 colțuri” nu mai elimină
pixelii de la colțurile pereților (L, T). Rulează fără workflow complet.

Necesită: numpy, opencv-python (același env ca engine-ul).
Din holzbot-engine: python scripts/test_walls_flood_no_holes.py
"""
import sys
from pathlib import Path

try:
    import numpy as np
    import cv2
except ImportError as e:
    print("Necesită numpy și opencv-python. Activează venv-ul proiectului.")
    sys.exit(2)


def make_L_mask(h: int, w: int) -> np.ndarray:
    """Mască 1px: L (colț stânga-sus)."""
    out = np.zeros((h, w), dtype=np.uint8)
    out[2, 2:8] = 255
    out[2:8, 2] = 255
    return out


def flood_remove_without_junctions(mask: np.ndarray) -> np.ndarray:
    """Replică logica din raster_processing: flood din 4 colțuri, elimină doar non-joncțiuni."""
    h, w = mask.shape
    flood_base = (255 - mask).astype(np.uint8)
    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    flood_any = np.zeros((h, w), dtype=np.uint8)
    for cx, cy in corners:
        if flood_base[cy, cx] == 255:
            region_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            img_copy = flood_base.copy()
            cv2.floodFill(img_copy, region_mask, (cx, cy), 128, None, None, cv2.FLOODFILL_MASK_ONLY | 4)
            flood_any[region_mask[1:-1, 1:-1] > 0] = 255
    walls_to_remove = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0:
                continue
            n_flood = sum(
                1 for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= y + dy < h and 0 <= x + dx < w and flood_any[y + dy, x + dx] > 0
            )
            if n_flood < 2:
                continue
            n_wall = sum(
                1 for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dx or dy) and 0 <= y + dy < h and 0 <= x + dx < w and mask[y + dy, x + dx] > 0
            )
            if n_wall >= 2:
                continue
            walls_to_remove[y, x] = 255
    out = mask.copy()
    out[walls_to_remove > 0] = 0
    return out


def main():
    # L 1px: colț la (2,2)
    mask = make_L_mask(12, 12)
    corner_val_before = int(mask[2, 2])
    result = flood_remove_without_junctions(mask)
    corner_val_after = int(result[2, 2])
    if corner_val_before == 255 and corner_val_after == 255:
        print("OK: pixel colț (2,2) păstrat după flood-removal.")
        return 0
    print(f"FAIL: colț (2,2) era {corner_val_before}, după eliminare e {corner_val_after} (trebuia 255).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
