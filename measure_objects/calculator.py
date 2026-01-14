# new/runner/measure_objects/calculator.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import statistics


def calculate_widths_from_detections(
    detections_all_json: Path,
    scale_json: Path
) -> dict:
    """
    Calculează lățimile medii pentru fiecare tip de obiect + aria scării.
    
    LOGICA CORECTĂ (plan = vedere de sus):
    - Geamuri/uși = dreptunghiuri alungite
    - Lățime reală = dimensiunea MAI MARE: MAX(x2-x1, y2-y1)
    - Scară: aria = (x2-x1) × (y2-y1) × (meters_per_pixel)²
    
    Args:
        detections_all_json: detections_all.json cu bbox-uri
        scale_json: scale_result.json cu meters_per_pixel
    
    Returns:
        {
          "measurements": {
            "door": {"real_width_meters": 0.89, ...},
            "stairs": {"area_m2": 3.45, ...}
          }
        }
    """
    # Load inputs
    with open(detections_all_json, "r", encoding="utf-8") as f:
        detections = json.load(f)
    
    with open(scale_json, "r", encoding="utf-8") as f:
        scale_data = json.load(f)
    
    meters_per_pixel = float(scale_data.get("meters_per_pixel", 0.0))
    
    if meters_per_pixel <= 0:
        raise ValueError("Scara invalidă în scale_result.json")
    
    print(f"       📐 Calcul lățimi + arii (scala: {meters_per_pixel:.6f} m/px)")
    
    # Grupează pe tipuri
    grouped: Dict[str, List[dict]] = {
        "door": [],
        "double_door": [],
        "window": [],
        "double_window": [],
        "stairs": []  # ← NOU pentru scări
    }
    
    for det in detections:
        obj_type = str(det.get("type", "")).lower()
        status = str(det.get("status", "")).lower()
        
        # Skip obiecte respinse
        if status == "rejected":
            continue
        
        # Extrage bbox
        try:
            x1 = int(det["x1"])
            y1 = int(det["y1"])
            x2 = int(det["x2"])
            y2 = int(det["y2"])
        except (KeyError, ValueError):
            continue
        
        # Calculează dimensiuni în pixeli
        width_px = abs(x2 - x1)
        height_px = abs(y2 - y1)
        
        # ==========================================
        # TRATARE SCĂRI (aria, nu lățimea)
        # ==========================================
        if "stair" in obj_type:
            area_px2 = width_px * height_px
            area_m2 = area_px2 * (meters_per_pixel ** 2)
            
            grouped["stairs"].append({
                "area_m2": area_m2,
                "area_px2": area_px2,
                "bbox_dims_px": (width_px, height_px)
            })
            continue
        
        # ==========================================
        # UȘI/FERESTRE: Lățime = dimensiunea MAI MARE
        # ==========================================
        # Geamurile/ușile sunt dreptunghiuri alungite → partea LUNGĂ = lățimea reală
        actual_width_px = max(width_px, height_px)
        
        # Detectăm orientarea
        if width_px > height_px:
            orientation = "horizontal"  # alungit pe orizontală
        else:
            orientation = "vertical"    # alungit pe verticală
        
        # Convertește în metri
        width_m = actual_width_px * meters_per_pixel
        
        # Grupează pe tip cu date detaliate
        measurement_data = {
            "width_m": width_m,
            "width_px": actual_width_px,
            "bbox_dims_px": (width_px, height_px),
            "orientation": orientation
        }
        
        if "double" in obj_type and "door" in obj_type:
            grouped["double_door"].append(measurement_data)
        elif "double" in obj_type and "window" in obj_type:
            grouped["double_window"].append(measurement_data)
        elif "door" in obj_type:
            grouped["door"].append(measurement_data)
        elif "window" in obj_type:
            grouped["window"].append(measurement_data)
    
    # ==========================================
    # CALCULEAZĂ STATISTICI
    # ==========================================
    result = {
        "scale_meters_per_pixel": meters_per_pixel,
        "measurements": {}
    }
    
    # UȘI/FERESTRE
    for obj_type in ["door", "double_door", "window", "double_window"]:
        measurements_list = grouped[obj_type]
        
        if not measurements_list:
            continue
        
        widths = [m["width_m"] for m in measurements_list]
        
        mean_width = statistics.mean(widths)
        median_width = statistics.median(widths)
        stdev = statistics.stdev(widths) if len(widths) > 1 else 0.0
        
        # Validare: lățimi realiste
        valid_ranges = {
            "door": (0.70, 1.00),
            "double_door": (1.40, 2.00),
            "window": (0.80, 1.50),
            "double_window": (1.60, 3.00)
        }
        
        min_valid, max_valid = valid_ranges[obj_type]
        is_valid = min_valid <= mean_width <= max_valid
        
        # Confidence bazat pe consistență
        if stdev < 0.05:
            confidence = "high"
        elif stdev < 0.10:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Orientarea dominantă
        orientations = [m["orientation"] for m in measurements_list]
        vertical_count = sum(1 for o in orientations if o == "vertical")
        horizontal_count = len(orientations) - vertical_count
        
        if vertical_count > horizontal_count:
            dominant_orientation = "vertical"
        elif horizontal_count > vertical_count:
            dominant_orientation = "horizontal"
        else:
            dominant_orientation = "mixed"
        
        result["measurements"][obj_type] = {
            "real_width_meters": round(mean_width, 3),
            "median_width_meters": round(median_width, 3),
            "std_dev": round(stdev, 3),
            "count_measured": len(widths),
            "min_width": round(min(widths), 3),
            "max_width": round(max(widths), 3),
            "confidence": confidence,
            "validation": "valid" if is_valid else "invalid",
            "expected_range": f"{min_valid}–{max_valid} m",
            "orientation_stats": {
                "dominant": dominant_orientation,
                "vertical_count": vertical_count,
                "horizontal_count": horizontal_count
            },
            "notes": (
                f"Calculat din {len(widths)} detecții. "
                f"Orientare dominantă: {dominant_orientation}. "
                f"Metodă: MAX(bbox_width, bbox_height) × meters_per_pixel"
            )
        }
        
        print(
            f"       ✓ {obj_type}: {mean_width:.3f}m "
            f"(n={len(widths)}, σ={stdev:.3f}, orient={dominant_orientation})"
        )
    
    # SCĂRI
    stairs_list = grouped["stairs"]
    if stairs_list:
        areas = [s["area_m2"] for s in stairs_list]
        
        total_area = sum(areas)
        mean_area = statistics.mean(areas)
        
        result["measurements"]["stairs"] = {
            "total_area_m2": round(total_area, 2),
            "average_area_m2": round(mean_area, 2),
            "count_measured": len(areas),
            "individual_areas_m2": [round(a, 2) for a in areas],
            "notes": f"Calculat din {len(areas)} scări. Metodă: width_px × height_px × (meters_per_pixel)²"
        }
        
        print(f"       ✓ stairs: {total_area:.2f}m² total (n={len(areas)})")
    
    if not result["measurements"]:
        raise ValueError("Nicio măsurare validă găsită în detecții")
    
    return result