# new/runner/measure_objects/aggregate.py
from __future__ import annotations
import json
from pathlib import Path

def create_openings_all(
    detections_path: Path, 
    measurements_path: Path, 
    exterior_doors_path: Path,
    output_path: Path
) -> int:
    """
    Agregă informațiile despre deschideri (uși/ferestre) din mai multe surse:
    1. detections_all.json (Tip, Poziție, Scor)
    2. openings_measurements_gemini.json (Lățimi reale calculate)
    3. exterior_doors.json (Status Exterior/Interior)
    """
    
    # 1. Încărcăm datele de bază
    objects = []
    if detections_path.exists():
        with open(detections_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Suportă listă sau dict cu cheia 'predictions'
            if isinstance(data, list):
                objects = data
            else:
                objects = data.get("predictions", [])

    # 2. Încărcăm măsurătorile (lățimi reale)
    # Structura: { "measurements": { "door": { "real_width_meters": 0.9, ... } } }
    type_widths = {}
    if measurements_path.exists():
        with open(measurements_path, "r", encoding="utf-8") as f:
            meas_data = json.load(f)
            m_root = meas_data.get("measurements", {})
            for k, v in m_root.items():
                if "real_width_meters" in v:
                    type_widths[k] = v["real_width_meters"]

    # 3. Încărcăm statusul Exterior/Interior
    # Structura: { "exterior_doors": [...], "interior_doors": [...] }
    ext_doors_map = {} # Cheie: tuple(box) -> "exterior"
    
    if exterior_doors_path.exists():
        try:
            with open(exterior_doors_path, "r", encoding="utf-8") as f:
                ext_data = json.load(f)
                
            # Extragem listele
            ext_list = ext_data.get("exterior_doors", [])
            int_list = ext_data.get("interior_doors", [])
            
            # Mapăm coordonatele la status pentru lookup rapid
            # Folosim coordonatele cutiei ca identificator unic
            for d in ext_list:
                if "box_2d" in d:
                    # Convertim la tuple pentru a fi cheie de dict
                    box_key = tuple(map(int, d["box_2d"]))
                    ext_doors_map[box_key] = "exterior"
            
            for d in int_list:
                if "box_2d" in d:
                    box_key = tuple(map(int, d["box_2d"]))
                    ext_doors_map[box_key] = "interior"
                    
        except Exception as e:
            print(f"⚠️ Warning loading exterior_doors.json: {e}")

    # 4. Construim lista finală
    final_list = []
    
    for obj in objects:
        cls = obj.get("class", "unknown")
        # Ignorăm scările aici (sunt tratate separat la arii)
        if cls == "stairs":
            continue
            
        # Determină lățimea
        width = type_widths.get(cls, 0.0)
        
        # Determină status (doar pentru uși)
        status = "unknown"
        
        # Încercăm să găsim statusul în maparea de uși exterioare
        # Roboflow dă coordonatele în 'x','y','width','height' sau 'box_2d'?
        # Codul de detecție salvează "box_2d": [x1, y1, x2, y2]
        
        # Verificăm cheile disponibile
        bbox = None
        if "box_2d" in obj:
            bbox = tuple(map(int, obj["box_2d"]))
        elif "x" in obj and "y" in obj:
            # Reconstruim box-ul aproximativ dacă lipsește box_2d (fallback)
            x, y, w, h = obj["x"], obj["y"], obj["width"], obj["height"]
            bbox = (int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2))
            
        if bbox and bbox in ext_doors_map:
            status = ext_doors_map[bbox]
        
        # Fallback simplu pe bază de nume clasă dacă nu avem potrivire geometrică
        if status == "unknown":
            if "window" in cls:
                status = "exterior" # Ferestrele sunt implicit exterioare
            elif "door" in cls:
                status = "interior" # Ușile default interior
        
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