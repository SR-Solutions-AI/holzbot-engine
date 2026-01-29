# new/runner/measure_objects/aggregate.py
from __future__ import annotations
import json
from pathlib import Path

def _normalize_class(cls: str) -> str:
    """Mapează numele clasei brute (ex: sliding-door) la cheia medie (ex: door)."""
    cls = cls.lower()
    if 'double' in cls and 'door' in cls:
        return 'double_door'
    elif 'double' in cls and 'window' in cls:
        return 'double_window'
    elif 'door' in cls:
        return 'door'
    elif 'window' in cls:
        return 'window'
    return 'unknown'

def create_openings_all(
    detections_path: Path, 
    measurements_path: Path, 
    exterior_doors_path: Path,
    output_path: Path
) -> int:
    """
    Agregă informațiile despre deschideri (uși/ferestre) din mai multe surse.
    """
    
    # 1. Încărcăm datele de bază
    objects = []
    if detections_path.exists():
        with open(detections_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                objects = data
            else:
                objects = data.get("predictions", [])

    # 2. Încărcăm măsurătorile (lățimi reale)
    type_widths = {}
    if measurements_path.exists():
        with open(measurements_path, "r", encoding="utf-8") as f:
            meas_data = json.load(f)
            m_root = meas_data.get("measurements", {})
            for k, v in m_root.items():
                if "real_width_meters" in v:
                    type_widths[k] = v["real_width_meters"]

    # 3. Încărcăm statusul Exterior/Interior
    ext_doors_map = {} 
    openings_status_map = {}  # Mapare idx -> status pentru workflow nou
    
    if exterior_doors_path.exists():
        try:
            with open(exterior_doors_path, "r", encoding="utf-8") as f:
                ext_data = json.load(f)
            
            # Format nou: openings_measurements.json din raster_processing
            if "openings" in ext_data:
                for opening in ext_data.get("openings", []):
                    bbox = opening.get("bbox", [])
                    if bbox and len(bbox) == 4:
                        box_key = tuple(map(int, bbox))
                        status = opening.get("status", "unknown")
                        if status in ["exterior", "interior"]:
                            ext_doors_map[box_key] = status
                    # De asemenea, mapăm pe idx pentru matching mai ușor
                    idx = opening.get("idx")
                    if idx is not None:
                        openings_status_map[idx] = opening.get("status", "unknown")
            # Format vechi ELIMINAT - nu mai folosim exterior_doors.json vechi
                    
        except Exception as e:
            print(f"⚠️ Warning loading status source: {e}")

    # 4. Construim lista finală
    final_list = []
    
    for obj in objects:
        # CORECȚIE 1 (Anterior identificată): Citirea clasei din cheia 'type'
        cls = obj.get("type", "unknown")
        
        if cls == "stairs":
            continue
            
        normalized_cls = _normalize_class(cls)
        width = type_widths.get(normalized_cls, 0.0)
        status = "unknown"
        
        # CORECȚIE 2 (CRITICĂ): Asigurăm potrivirea formatului de bounding box (bbox) pentru lookup
        bbox = None
        if "box_2d" in obj:
            bbox = tuple(map(int, obj["box_2d"]))
        # FIX CRITIC: Adăugat suport pentru formatul x1/y1/x2/y2 (cel mai probabil format din detections_all.json)
        elif "x1" in obj and "x2" in obj and "y1" in obj and "y2" in obj:
            bbox = (int(obj["x1"]), int(obj["y1"]), int(obj["x2"]), int(obj["y2"]))
        elif "x" in obj and "y" in obj:
            # Fallback pentru formatul Roboflow center (x/y/w/h)
            x, y, w, h = obj["x"], obj["y"], obj["width"], obj["height"]
            bbox = (int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2))
            
        # Potrivirea statusului de Exterior/Interior
        if bbox and bbox in ext_doors_map:
            status = ext_doors_map[bbox]
        
        # Fallback simplu pe bază de nume clasă dacă nu avem potrivire geometrică
        if status == "unknown":
            if "window" in cls:
                status = "exterior" 
            elif "door" in cls:
                status = "interior" # Default interior dacă nu se potrivește
        
        item = {
            "type": cls,
            "width_m": width,
            "status": status,
            "confidence": obj.get("confidence", 0.0),
            "bbox": list(bbox) if bbox else []
        }
        final_list.append(item)

    # 5. Salvare
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_list, f, indent=2)
        
    return len(final_list)