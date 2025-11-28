# new/runner/exterior_doors/classify.py
from __future__ import annotations
from pathlib import Path
import json
import cv2
import numpy as np

from .config import (
    COLOR_DARK_BLUE, COLOR_YELLOW, 
    COLOR_RED, COLOR_GREEN,
    MAX_DISTANCE_RATIO
)

def _bbox_diagonal(bbox: tuple[int, int, int, int]) -> float:
    """Calculează lungimea diagonalei bbox."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return np.sqrt(w**2 + h**2)

def _distance_to_outdoor(mask_outdoor: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """
    Calculează distanța minimă de la bbox la zona de EXTERIOR (valoare 255 în mască).
    """
    H, W = mask_outdoor.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    
    # Validare și Clamp
    if x1 >= W or x2 <= 0 or y1 >= H or y2 <= 0: return float('inf')
    x1, x2 = max(0, x1), min(W-1, x2)
    y1, y2 = max(0, y1), min(H-1, y2)
    
    # Verificăm întâi dacă atinge direct (intersecție)
    bbox_region = mask_outdoor[y1:y2, x1:x2]
    if cv2.countNonZero(bbox_region) > 0:
        return 0.0 
        
    # Dacă nu atinge, calculăm distanța
    margin = 100
    x1_exp, x2_exp = max(0, x1 - margin), min(W - 1, x2 + margin)
    y1_exp, y2_exp = max(0, y1 - margin), min(H - 1, y2 + margin)
    
    region = mask_outdoor[y1_exp:y2_exp, x1_exp:x2_exp]
    
    # Inversăm: 0=exterior, 255=interior
    inv_region = cv2.bitwise_not(region)
    dist_transform = cv2.distanceTransform(inv_region, cv2.DIST_L2, 5)
    
    # Coordonate relative
    y_rel_1, y_rel_2 = y1 - y1_exp, y2 - y1_exp
    x_rel_1, x_rel_2 = x1 - x1_exp, x2 - x1_exp
    
    dist_slice = dist_transform[y_rel_1:y_rel_2, x_rel_1:x_rel_2]
    
    if dist_slice.size == 0: return float('inf')
    
    return float(np.min(dist_slice))

def classify_exterior_doors(
    plan_image: Path,
    outdoor_mask_path: Path,
    detections_all_json: Path,
    out_dir: Path
) -> tuple[Path, Path]:
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "exterior_doors.json"
    out_overlay = out_dir / "blue_overlay.jpg"
    
    # 1. Load Resources
    plan = cv2.imread(str(plan_image))
    if plan is None: raise RuntimeError(f"Plan invalid: {plan_image}")
    
    mask_outdoor = cv2.imread(str(outdoor_mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_outdoor is None: raise RuntimeError(f"Mască invalidă: {outdoor_mask_path}")
    
    if mask_outdoor.shape[:2] != plan.shape[:2]:
        print(f"       ⚠️ Resize mask {mask_outdoor.shape} -> {plan.shape}")
        mask_outdoor = cv2.resize(mask_outdoor, (plan.shape[1], plan.shape[0]), interpolation=cv2.INTER_NEAREST)

    # --- FIX CRITIC: Citire Robustă JSON (List vs Dict) ---
    raw_dets = json.loads(detections_all_json.read_text(encoding="utf-8"))
    
    if isinstance(raw_dets, list):
        dets_list = raw_dets
    else:
        dets_list = raw_dets.get("predictions", [])
    
    # Normalizare coordonate (Roboflow center -> Corner)
    normalized_dets = []
    for d in dets_list:
        if "box_2d" not in d:
            if "x" in d and "width" in d:
                # Format Roboflow
                x, y, w, h = d["x"], d["y"], d["width"], d["height"]
                d["box_2d"] = [int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)]
            elif "x1" in d and "y1" in d:
                # Format Count Objects
                d["box_2d"] = [int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])]
        
        normalized_dets.append(d)

    # 2. Visualization Layers
    vis_img = plan.copy()
    overlay_mask = vis_img.copy()
    
    # Overlay Măști
    overlay_mask[mask_outdoor > 127] = COLOR_DARK_BLUE
    overlay_mask[mask_outdoor <= 127] = COLOR_YELLOW
    cv2.addWeighted(overlay_mask, 0.4, vis_img, 0.6, 0, vis_img)
    
    # 3. Clasificare
    results = []
    idx = 0
    
    # Filtrăm doar ușile
    doors = [d for d in normalized_dets if "door" in str(d.get("class", d.get("type", ""))).lower()]
    
    print(f"       Found {len(doors)} doors to classify (Distance Method).")

    for d in doors:
        idx += 1
        box = d.get("box_2d")
        if not box: continue
        
        diagonal = _bbox_diagonal(box)
        dist = _distance_to_outdoor(mask_outdoor, box)
        
        limit = diagonal * MAX_DISTANCE_RATIO
        is_exterior = (dist <= limit)
        
        status = "exterior" if is_exterior else "interior"
        color = COLOR_RED if is_exterior else COLOR_GREEN
        
        res_entry = {
            "id": idx,
            "type": d.get("class", d.get("type", "door")),
            "status": status,
            "bbox": box,
            "box_2d": box,
            "metrics": {
                "distance_px": round(dist, 1),
                "limit_px": round(limit, 1)
            }
        }
        results.append(res_entry)
        
        # Desenare Box SOLID
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 4)
        
        # Etichetă
        label_txt = "EXT" if is_exterior else "INT"
        (w_txt, h_txt), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis_img, (x1, y1 - 25), (x1 + w_txt + 10, y1), color, -1)
        cv2.putText(vis_img, label_txt, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # 4. Salvare
    final_output = {
        "summary": {
            "total": len(results),
            "exterior_count": sum(1 for r in results if r["status"] == "exterior"),
            "interior_count": sum(1 for r in results if r["status"] == "interior")
        },
        "exterior_doors": [r for r in results if r["status"] == "exterior"],
        "interior_doors": [r for r in results if r["status"] == "interior"]
    }
    
    out_json.write_text(json.dumps(final_output, indent=2), encoding="utf-8")
    cv2.imwrite(str(out_overlay), vis_img)
    
    print(f"       ✅ Classified: {final_output['summary']['exterior_count']} Ext, {final_output['summary']['interior_count']} Int")
    
    return out_json, out_overlay