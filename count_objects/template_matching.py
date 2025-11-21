# new/runner/count_objects/template_matching.py
from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import (
    SCALES, 
    ROTATION_ANGLES, 
    MAX_TEMPLATE_WORKERS, 
    MAX_DETECTION_WORKERS,
    STAIRS_OVERLAP_THRESHOLD,
    DETECTION_OVERLAP_THRESHOLD
)


def _match_single_rotation(crop: np.ndarray, template: dict, scale: float, angle: int) -> float:
    """Verifică o singură combinație (rotație + template + scale)."""
    try:
        h, w = crop.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        
        cos, sin = abs(M[0,0]), abs(M[0,1])
        new_w, new_h = int(h*sin + w*cos), int(h*cos + w*sin)
        
        M[0,2] += (new_w/2) - (w/2)
        M[1,2] += (new_h/2) - (h/2)
        
        rotated_crop = cv2.warpAffine(crop, M, (new_w, new_h), borderValue=255)
        
        resized = cv2.resize(
            rotated_crop,
            (int(template["image"].shape[1]*scale), int(template["image"].shape[0]*scale))
        )
        
        result = cv2.matchTemplate(resized, template["image"], cv2.TM_CCOEFF_NORMED)
        return cv2.minMaxLoc(result)[1]
    except Exception:
        return 0.0


def match_with_rotation(crop: np.ndarray, templates: list[dict], scales: list[float] = SCALES) -> float:
    """Template matching cu rotație crop + scale variations - PARALELIZAT COMPLET."""
    if not templates:
        return 0.0
    
    tasks = []
    for angle in ROTATION_ANGLES:
        for template in templates:
            for scale in scales:
                tasks.append((crop, template, scale, angle))
    
    best_sim = 0.0
    max_workers = min(MAX_TEMPLATE_WORKERS, len(tasks))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_match_single_rotation, c, t, s, a): None
            for c, t, s, a in tasks
        }
        
        for future in as_completed(futures):
            try:
                sim = future.result()
                best_sim = max(best_sim, sim)
            except Exception:
                pass
    
    return best_sim


def overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Calculează IoU între două bounding boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    
    dx = min(ax2, bx2) - max(ax1, bx1)
    dy = min(ay2, by2) - max(ay1, by1)
    
    if dx <= 0 or dy <= 0:
        return 0.0
    
    area_overlap = dx * dy
    area_a = (ax2 - ax1) * (ay2 - ay1)
    
    return area_overlap / area_a if area_a > 0 else 0.0


def process_detections_parallel(
    detections: List[dict],
    gray_image: np.ndarray,
    templates: list[dict],
    used_boxes: list[Tuple[int, int, int, int]],
    img_width: int,
    img_height: int,
    has_stairs: bool = False
) -> List[dict]:
    """
    Procesează toate detecțiile în paralel (template matching simultan).
    
    Args:
        has_stairs: True dacă primul element din used_boxes e scara
    
    Returns:
        List de dicționare cu results pentru fiecare detecție
    """
    def process_one(det_data):
        """Helper pentru procesare paralelă."""
        idx, pred = det_data
        
        x = int(pred.get("x", 0))
        y = int(pred.get("y", 0))
        w = int(pred.get("width", 0))
        h = int(pred.get("height", 0))
        conf = float(pred.get("confidence", 0.0))
        
        x1 = max(0, x - w//2)
        y1 = max(0, y - h//2)
        x2 = min(img_width, x + w//2)
        y2 = min(img_height, y + h//2)
        
        bbox = (x1, y1, x2, y2)
        
        # Check overlap cu scara (mai strict)
        if has_stairs and used_boxes:
            stairs_box = used_boxes[0]
            overlap_ratio = overlap(bbox, stairs_box)
            if overlap_ratio > STAIRS_OVERLAP_THRESHOLD:
                return {
                    "idx": idx,
                    "bbox": bbox,
                    "conf": conf,
                    "best_sim": 0.0,
                    "combined": 0.0,
                    "skip": True,
                    "skip_reason": f"stairs_overlap_{overlap_ratio:.2f}"
                }
        
        # Check overlap cu alte detecții (mai permisiv)
        other_boxes = used_boxes[1:] if has_stairs else used_boxes
        for other_box in other_boxes:
            overlap_ratio = overlap(bbox, other_box)
            if overlap_ratio > DETECTION_OVERLAP_THRESHOLD:
                return {
                    "idx": idx,
                    "bbox": bbox,
                    "conf": conf,
                    "best_sim": 0.0,
                    "combined": 0.0,
                    "skip": True,
                    "skip_reason": f"detection_overlap_{overlap_ratio:.2f}"
                }
        
        # Extract crop
        crop = gray_image[y1:y2, x1:x2]
        if crop.size == 0:
            return {
                "idx": idx,
                "bbox": bbox,
                "conf": conf,
                "best_sim": 0.0,
                "combined": 0.0,
                "skip": True,
                "skip_reason": "empty_crop"
            }
        
        # Template matching (paralel intern)
        best_sim = match_with_rotation(crop, templates)
        combined = (0.6 * conf) + (0.4 * best_sim)
        
        return {
            "idx": idx,
            "bbox": bbox,
            "conf": conf,
            "best_sim": best_sim,
            "combined": combined,
            "skip": False
        }
    
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_DETECTION_WORKERS) as executor:
        indexed_detections = list(enumerate(detections, 1))
        futures = {
            executor.submit(process_one, det): det 
            for det in indexed_detections
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"       [ERR] Processing detection: {e}")
    
    results.sort(key=lambda r: r["idx"])
    return results