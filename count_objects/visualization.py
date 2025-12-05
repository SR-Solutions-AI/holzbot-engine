# new/runner/count_objects/visualization.py
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from .config import COLORS


def draw_results(img: np.ndarray, results: dict, stairs_bbox: dict | None) -> np.ndarray:
    """Desenează rezultatele detaliate pentru Debug (toate culorile)."""
    out_img = img.copy()
    
    if stairs_bbox:
        x1, y1, x2, y2 = stairs_bbox["x1"], stairs_bbox["y1"], stairs_bbox["x2"], stairs_bbox["y2"]
        cv2.rectangle(out_img, (x1, y1), (x2, y2), COLORS["stairs"], 4)
        cv2.putText(out_img, "STAIRS", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["stairs"], 2)
    
    for label in results:
        for (x1, y1, x2, y2) in results[label]["confirm"]:
            cv2.rectangle(out_img, (x1, y1), (x2, y2), COLORS[label]["template"], 3)
        
        for (x1, y1, x2, y2) in results[label]["oblique"]:
            cv2.rectangle(out_img, (x1, y1), (x2, y2), COLORS[label]["gemini"], 2)
        
        for (x1, y1, x2, y2) in results[label]["reject"]:
            cv2.rectangle(out_img, (x1, y1), (x2, y2), COLORS[label]["rejected"], 2)
    
    return out_img


def draw_final_orange(img: np.ndarray, results: dict, stairs_bbox: dict | None) -> np.ndarray:
    """
    Desenează DOAR elementele acceptate cu PORTOCALIU pentru Live Feed.
    Portocaliu BGR: (0, 165, 255)
    """
    out_img = img.copy()
    ORANGE = (0, 165, 255)
    
    # Stairs
    if stairs_bbox:
        x1, y1, x2, y2 = stairs_bbox["x1"], stairs_bbox["y1"], stairs_bbox["x2"], stairs_bbox["y2"]
        cv2.rectangle(out_img, (x1, y1), (x2, y2), ORANGE, 3)
    
    # Objects
    for label in results:
        # Combinăm 'confirm' și 'oblique' (cele validate)
        valid_boxes = results[label]["confirm"] + results[label]["oblique"]
        
        for (x1, y1, x2, y2) in valid_boxes:
            cv2.rectangle(out_img, (x1, y1), (x2, y2), ORANGE, 3)
            # Putem pune un text mic
            cv2.putText(out_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1)
            
    return out_img


def export_to_json(results: dict, stairs_export: dict) -> list[dict]:
    """Construiește lista de export JSON."""
    detections = []
    
    if stairs_export:
        detections.append(stairs_export)
    
    for label in results:
        for (x1, y1, x2, y2) in results[label]["confirm"]:
            detections.append({"type": label, "status": "confirmed", "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        
        for (x1, y1, x2, y2) in results[label]["oblique"]:
            detections.append({"type": label, "status": "gemini", "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        
        for (x1, y1, x2, y2) in results[label]["reject"]:
            detections.append({"type": label, "status": "rejected", "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    
    return detections