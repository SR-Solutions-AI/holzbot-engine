# new/runner/count_objects/visualization.py
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from .config import COLORS


def draw_results(img: np.ndarray, results: dict, stairs_bbox: dict | None) -> np.ndarray:
    """Desenează toate rezultatele pe imagine."""
    out_img = img.copy()
    
    # 1) Desenează scara (VERDE, linii groase)
    if stairs_bbox:
        x1, y1, x2, y2 = stairs_bbox["x1"], stairs_bbox["y1"], stairs_bbox["x2"], stairs_bbox["y2"]
        cv2.rectangle(out_img, (x1, y1), (x2, y2), COLORS["stairs"], 4)
        cv2.putText(
            out_img,
            "STAIRS",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLORS["stairs"],
            2
        )
    
    # 2) Desenează uși/ferestre
    for label in results:
        # Confirmate (template) - linii groase
        for (x1, y1, x2, y2) in results[label]["confirm"]:
            cv2.rectangle(out_img, (x1, y1), (x2, y2), COLORS[label]["template"], 3)
            cv2.putText(
                out_img,
                f"{label[:3].upper()}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLORS[label]["template"],
                2
            )
        
        # Confirmate Gemini - linii medii
        for (x1, y1, x2, y2) in results[label]["oblique"]:
            cv2.rectangle(out_img, (x1, y1), (x2, y2), COLORS[label]["gemini"], 2)
            cv2.putText(
                out_img,
                f"{label[:3].upper()}*",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLORS[label]["gemini"],
                2
            )
        
        # Respinse (ROȘU, linii normale)
        for (x1, y1, x2, y2) in results[label]["reject"]:
            cv2.rectangle(out_img, (x1, y1), (x2, y2), COLORS[label]["rejected"], 2)
            cv2.putText(
                out_img,
                "X",
                (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLORS[label]["rejected"],
                2
            )
    
    return out_img


def export_to_json(results: dict, stairs_export: dict) -> list[dict]:
    """Construiește lista de export JSON."""
    detections = []
    
    # Adaugă scara
    if stairs_export:
        detections.append(stairs_export)
    
    # Adaugă uși/ferestre
    for label in results:
        for (x1, y1, x2, y2) in results[label]["confirm"]:
            detections.append({
                "type": label,
                "status": "confirmed",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })
        
        for (x1, y1, x2, y2) in results[label]["oblique"]:
            detections.append({
                "type": label,
                "status": "gemini",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })
        
        for (x1, y1, x2, y2) in results[label]["reject"]:
            detections.append({
                "type": label,
                "status": "rejected",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })
    
    return detections