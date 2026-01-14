# file: engine/count_objects/visualization.py
from __future__ import annotations

import cv2
import numpy as np
from .config import COLORS

def draw_results(img: np.ndarray, results: dict, stairs_bboxes: list | None) -> np.ndarray:
    out_img = img.copy()
    
    # Stairs (List)
    if stairs_bboxes:
        for bbox in stairs_bboxes:
            # bbox poate fi list [x1, y1, x2, y2] sau dict (dacă vine din legacy)
            if isinstance(bbox, dict):
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            else:
                x1, y1, x2, y2 = bbox
                
            cv2.rectangle(out_img, (int(x1), int(y1)), (int(x2), int(y2)), COLORS["stairs"], 4)
            cv2.putText(out_img, "STAIRS", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["stairs"], 2)
    
    for label in results:
        for (x1, y1, x2, y2) in results[label]["confirm"]:
            cv2.rectangle(out_img, (x1, y1), (x2, y2), COLORS[label]["template"], 3)
        for (x1, y1, x2, y2) in results[label]["oblique"]:
            cv2.rectangle(out_img, (x1, y1), (x2, y2), COLORS[label]["gemini"], 2)
        for (x1, y1, x2, y2) in results[label]["reject"]:
            cv2.rectangle(out_img, (x1, y1), (x2, y2), COLORS[label]["rejected"], 2)
    return out_img


def draw_final_orange(img: np.ndarray, results: dict, stairs_bboxes: list | None) -> np.ndarray:
    out_img = img.copy()
    ORANGE = (0, 165, 255)
    
    if stairs_bboxes:
        for bbox in stairs_bboxes:
            if isinstance(bbox, dict):
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            else:
                x1, y1, x2, y2 = bbox
            cv2.rectangle(out_img, (int(x1), int(y1)), (int(x2), int(y2)), ORANGE, 3)
    
    for label in results:
        valid = results[label]["confirm"] + results[label]["oblique"]
        for (x1, y1, x2, y2) in valid:
            cv2.rectangle(out_img, (x1, y1), (x2, y2), ORANGE, 3)
            
    return out_img


def export_to_json(results: dict, stairs_exports: list) -> list[dict]:
    detections = []
    if stairs_exports:
        detections.extend(stairs_exports)
    
    for label in results:
        # ... (restul rămâne la fel ca înainte)
        pass
    # ... (Codul anterior era corect aici, doar am schimbat numele parametrului pt claritate)
    # Re-scriu complet pentru siguranță:
    for label in results:
        for b in results[label]["confirm"]:
            detections.append({"type": label, "status": "confirmed", "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]})
        for b in results[label]["oblique"]:
            detections.append({"type": label, "status": "gemini", "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]})
        for b in results[label]["reject"]:
            detections.append({"type": label, "status": "rejected", "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]})
            
    return detections

def export_detailed_json(all_results, stairs_exports, meta):
    out = {"meta": meta, "detections": []}
    if stairs_exports:
        for s in stairs_exports:
            out["detections"].append({**s, "validation_source": "specialized_model"})
            
    for r in all_results.values():
        for rec in r.get("records", []):
            out["detections"].append(rec.to_dict())
    return out