# file: engine/cubicasa_detector/interior_exterior.py
"""
Module pentru detectarea zonelor interior și exterior.
Conține funcții pentru generarea măștilor indoor_mask și outdoor_mask.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def detect_interior_exterior_zones(
    walls_mask: np.ndarray,
    h_orig: int,
    w_orig: int,
    steps_dir: str = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detectează zonele interior și exterior bazate pe masca de pereți.
    
    Args:
        walls_mask: Masca pereților (255 = perete, 0 = spațiu liber)
        h_orig: Înălțimea imaginii originale
        w_orig: Lățimea imaginii originale
        steps_dir: Director pentru salvarea imaginilor de debug (opțional)
    
    Returns:
        Tuple: (indoor_mask, outdoor_mask) - ambele ca măști binare (255 = zonă, 0 = rest)
    """
    print("   🌊 Analizez zonele...")
    
    # Kernel repair pentru restul procesării
    min_dim = min(h_orig, w_orig) 
    rep_k = max(3, int(min_dim * 0.005))
    if rep_k % 2 == 0: rep_k += 1
    kernel_repair = cv2.getStructuringElement(cv2.MORPH_RECT, (rep_k, rep_k))
    
    walls_thick = cv2.dilate(walls_mask, kernel_repair, iterations=3)
    
    h_pad, w_pad = h_orig + 2, w_orig + 2
    pad_walls = np.zeros((h_pad, w_pad), dtype=np.uint8)
    pad_walls[1:h_orig+1, 1:w_orig+1] = walls_thick
    
    inv_pad_walls = cv2.bitwise_not(pad_walls)
    flood_mask = np.zeros((h_pad+2, w_pad+2), dtype=np.uint8)
    cv2.floodFill(inv_pad_walls, flood_mask, (0, 0), 128)
    
    outdoor_mask = (inv_pad_walls[1:h_orig+1, 1:w_orig+1] == 128).astype(np.uint8) * 255
    
    kernel_grow = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    free_space = cv2.bitwise_not(walls_mask)
    
    for _ in range(30):
        outdoor_mask = cv2.bitwise_and(cv2.dilate(outdoor_mask, kernel_grow), free_space)
    
    total_space = cv2.bitwise_not(walls_mask)
    occupied = outdoor_mask
    indoor_mask = cv2.subtract(total_space, occupied)
    
    if steps_dir:
        cv2.imwrite(str(Path(steps_dir) / "03_outdoor_mask.png"), outdoor_mask)
        cv2.imwrite(str(Path(steps_dir) / "03_indoor_mask.png"), indoor_mask)
        print(f"      💾 Salvat: 03_outdoor_mask.png și 03_indoor_mask.png")
    
    return indoor_mask, outdoor_mask
