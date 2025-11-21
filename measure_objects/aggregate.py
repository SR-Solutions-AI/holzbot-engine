# new/runner/measure_objects/aggregate.py
from __future__ import annotations

import json
from pathlib import Path


def create_openings_all(
    detections_all_json: Path,
    measurements_json: Path,
    exterior_doors_json: Path,
    output_path: Path
) -> int:
    """
    Combină detecții + măsurări + exterior_doors în openings_all.json
    
    Returns:
        Numărul de obiecte în lista finală (fără scări)
    """
    # Load inputs
    with open(detections_all_json, "r", encoding="utf-8") as f:
        detections = json.load(f)
    
    with open(measurements_json, "r", encoding="utf-8") as f:
        meas_data = json.load(f)
    
    measurements = meas_data.get("measurements", {})
    
    # exterior_doors.json e opțional
    if exterior_doors_json.exists():
        with open(exterior_doors_json, "r", encoding="utf-8") as f:
            exterior_data = json.load(f)
        
        door_status_map = {}
        for d in exterior_data:
            bbox = tuple(map(int, d["bbox"]))
            door_status_map[bbox] = d["status"]
    else:
        print("       ℹ️  Lipsă exterior_doors.json — toate ușile vor fi 'unknown'")
        door_status_map = {}
    
    # Helper: extrage lățime pentru un tip
    def get_width_for_type(obj_type: str) -> float | None:
        normalized = obj_type.lower().replace("-", "_")
        meas = measurements.get(normalized)
        if not meas:
            return None
        return float(meas.get("real_width_meters", 0.0))
    
    # Construiește lista finală
    openings = []
    id_counter = 1
    
    for det in detections:
        obj_type = det.get("type", "").lower()
        status_det = det.get("status", "").lower()
        
        # Skip obiecte respinse
        if status_det == "rejected":
            continue
        
        # Skip scările (nu intră în openings_all.json)
        if "stair" in obj_type:
            continue
        
        # Filtrează doar uși/ferestre
        if not any(k in obj_type for k in ["door", "window"]):
            continue
        
        # Mapare tip standard
        if "double" in obj_type and "door" in obj_type:
            standard_type = "double_door"
        elif "double" in obj_type and "window" in obj_type:
            standard_type = "double_window"
        elif "door" in obj_type:
            standard_type = "door"
        elif "window" in obj_type:
            standard_type = "window"
        else:
            continue
        
        # Extrage lățime
        width_m = get_width_for_type(standard_type)
        if width_m is None or width_m <= 0:
            print(f"       ⚠️  Tip fără măsurare validă: {standard_type} (skip)")
            continue
        
        # Determină status (interior/exterior) doar pentru uși
        status = "exterior" if "window" in standard_type else "unknown"
        
        if "door" in standard_type:
            bbox = tuple(map(int, [det["x1"], det["y1"], det["x2"], det["y2"]]))
            
            # Căutăm match în exterior_doors.json (cu toleranță ±15px)
            for door_bbox, door_status in door_status_map.items():
                if all(abs(a - b) < 15 for a, b in zip(door_bbox, bbox)):
                    status = door_status
                    break
        
        openings.append({
            "id": id_counter,
            "type": standard_type,
            "status": status,
            "width_m": round(width_m, 3)
        })
        id_counter += 1
    
    # Salvează
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(openings, f, indent=2, ensure_ascii=False)
    
    return len(openings)