# file: engine/cubicasa_detector/measurements.py
"""
Module pentru calcularea măsurătorilor din planuri.
Conține funcții pentru calcularea lungimilor pereților, ariilor și conversiile în metri.
"""

from __future__ import annotations

import cv2
import numpy as np
from .wall_repair import get_strict_1px_outline


def calculate_measurements(
    walls_mask: np.ndarray,
    indoor_mask: np.ndarray,
    outdoor_mask: np.ndarray,
    m_px: float,
    steps_dir: str = None
) -> dict:
    """
    Calculează măsurătorile din plan (lungimi pereți, arii).
    
    Args:
        walls_mask: Masca pereților (255 = perete, 0 = spațiu liber)
        indoor_mask: Masca zonei interioare (255 = interior, 0 = rest)
        outdoor_mask: Masca zonei exterioare (255 = exterior, 0 = rest)
        m_px: Metri per pixel (scara)
        steps_dir: Director pentru salvarea imaginilor de debug (opțional)
    
    Returns:
        Dict cu măsurătorile în pixeli și metri
    """
    print("   📏 Calculez măsurători...")
    
    outline = get_strict_1px_outline(walls_mask)
    kernel_grow = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    touch_zone = cv2.dilate(outdoor_mask, kernel_grow, iterations=2)
    
    outline_ext_mask = cv2.bitwise_and(outline, touch_zone)
    outline_int_mask = cv2.subtract(outline, outline_ext_mask)
    
    # Calculăm lungimea pereților exteriori (din outline)
    px_len_skeleton_ext = int(np.count_nonzero(outline_ext_mask))
    
    # Lungimea structurii pereților interiori (folosim outline interior)
    px_len_skeleton_structure_int = int(np.count_nonzero(outline_int_mask))
    
    # Lungimi din outline (pentru finisaje)
    px_len_ext = int(np.count_nonzero(outline_ext_mask))
    px_len_int = int(np.count_nonzero(outline_int_mask))
    
    # Arii
    px_area_indoor = int(np.count_nonzero(indoor_mask))
    px_area_total = int(np.count_nonzero(cv2.bitwise_not(outdoor_mask)))

    # Conversii în metri
    walls_ext_m = px_len_ext * m_px  # Pentru finisaje
    walls_int_m = px_len_int * m_px  # Pentru finisaje
    
    # Lungimi din skeleton (pentru structură)
    walls_skeleton_ext_m = px_len_skeleton_ext * m_px
    walls_skeleton_structure_int_m = px_len_skeleton_structure_int * m_px  # Structură pereți interiori (din outline)
    
    area_indoor_m2 = px_area_indoor * (m_px ** 2)
    area_total_m2 = px_area_total * (m_px ** 2)

    measurements = {
        "pixels": {
            "walls_len_ext": int(px_len_ext),
            "walls_len_int": int(px_len_int),
            "walls_skeleton_ext": int(px_len_skeleton_ext),
            "walls_skeleton_structure_int": int(px_len_skeleton_structure_int),
            "indoor_area": int(px_area_indoor),
            "total_area": int(px_area_total)
        },
        "metrics": {
            "scale_m_per_px": float(m_px),
            "walls_ext_m": float(round(walls_ext_m, 2)),  # Pentru finisaje
            "walls_int_m": float(round(walls_int_m, 2)),  # Pentru finisaje
            "walls_skeleton_ext_m": float(round(walls_skeleton_ext_m, 2)),
            "walls_skeleton_structure_int_m": float(round(walls_skeleton_structure_int_m, 2)),  # Pentru structură pereți interiori
            "area_indoor_m2": float(round(area_indoor_m2, 2)),
            "area_total_m2": float(round(area_total_m2, 2))
        }
    }

    print(f"   🏠 Arie Indoor: {area_indoor_m2:.2f} m²")
    print(f"   📏 Lungimi pereți:")
    print(f"      - Exterior (outline): {walls_ext_m:.2f} m (pentru finisaje)")
    print(f"      - Interior (outline): {walls_int_m:.2f} m (pentru finisaje)")
    print(f"      - Skeleton exterior (din outline): {walls_skeleton_ext_m:.2f} m")
    print(f"      - Structură interior (din outline): {walls_skeleton_structure_int_m:.2f} m")
    
    return measurements
