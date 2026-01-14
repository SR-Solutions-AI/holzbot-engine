# file: engine/count_objects/stairs_detection.py
from __future__ import annotations

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Dict

from .roboflow_api import infer_roboflow
from .gemini_verification import verify_candidates_parallel
from .config import (
    ROBOFLOW_STAIRS_PROJECT,
    ROBOFLOW_STAIRS_VERSION,
    ROBOFLOW_STAIRS_WORKSPACE
)

def _is_stairs_outdoor(bbox: list, mask: np.ndarray, threshold: float = 0.75) -> Tuple[bool, float]:
    """Verifică dacă scara e în zona albă (outdoor)."""
    if mask is None:
        return False, 0.0
    
    x1, y1, x2, y2 = bbox
    h, w = mask.shape[:2]
    
    # Clamp
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    
    if x2 <= x1 or y2 <= y1:
        return False, 0.0
        
    crop = mask[y1:y2, x1:x2]
    if crop.size == 0:
        return False, 0.0

    # Alb (255) înseamnă outdoor/gol
    white_pixels = np.sum(crop > 240)
    ratio = white_pixels / crop.size
    
    return ratio > threshold, ratio


def _save_debug_stairs(debug_dir: Path, idx: int, status: str, reason: str, bbox: list, img: np.ndarray, conf: float):
    """Salvează crop-ul și JSON-ul de debug pentru scări."""
    if not debug_dir:
        return
        
    try:
        # Nume folder: 0001_stairs_ACCEPTED_roboflow
        safe_status = "ACCEPTED" if status == "confirmed" else "REJECTED"
        folder_name = f"{idx:04d}_stairs_{safe_status}_{reason.replace(' ', '_')}"
        
        save_path = debug_dir / folder_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # JSON
        info = {
            "idx": idx,
            "class_name": "stairs",
            "status": safe_status,
            "reason": reason,
            "confidence": conf,
            "bbox": bbox
        }
        (save_path / "info.json").write_text(json.dumps(info, indent=2))
        
        # Image
        x1, y1, x2, y2 = map(int, bbox)
        h, w = img.shape[:2]
        pad = 50
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
        
        crop = img[cy1:cy2, cx1:cx2].copy()
        
        # Desenăm box
        color = (0, 0, 255) if safe_status == "REJECTED" else (0, 255, 0)
        rx1, ry1 = x1 - cx1, y1 - cy1
        rx2, ry2 = x2 - cx1, y2 - cy1
        
        cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), color, 3)
        cv2.putText(crop, f"{reason} {conf:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imwrite(str(save_path / "context.jpg"), crop)
        
    except Exception as e:
        print(f"       ⚠️ Error saving stairs debug: {e}")


def process_stairs(
    plan_image: Path, 
    api_key: str, 
    outdoor_mask: np.ndarray, 
    img_gray: np.ndarray,
    img_color: np.ndarray,
    temp_dir: Path,
    debug_dir: Path
) -> Tuple[List[Dict], List[Dict]]:
    """
    Detectează scări (una sau mai multe) și le verifică.
    1. Filtru Outdoor.
    2. Sortare Confidence.
    3. Top 1 = Referință.
    4. Altele (Close Confidence) = Verificare Gemini vs Referință.
    """
    print(f"\n       {'='*30} STAIRS {'='*30}")
    
    # 1. Inferență
    try:
        result = infer_roboflow(
            plan_image,
            api_key,
            ROBOFLOW_STAIRS_WORKSPACE,
            ROBOFLOW_STAIRS_PROJECT,
            ROBOFLOW_STAIRS_VERSION,
            confidence=0.15, # Luăm tot ce pare a fi scară
            overlap=30
        )
    except Exception as e:
        print(f"       ⚠️ Roboflow error: {e}")
        return [], []

    raw_preds = result.get("predictions", [])
    if not raw_preds:
        print("       [STAIRS] None detected.")
        return [], []

    # 2. Convertire și Filtrare
    valid_candidates = []
    
    for i, p in enumerate(raw_preds):
        conf = float(p.get("confidence", 0.0))
        x, y, w, h = p["x"], p["y"], p["width"], p["height"]
        x1, y1 = x - w/2, y - h/2
        x2, y2 = x + w/2, y + h/2
        bbox = [x1, y1, x2, y2]
        
        # Check Outdoor
        is_out, ratio = _is_stairs_outdoor(bbox, outdoor_mask)
        if is_out:
            _save_debug_stairs(debug_dir, 5000+i, "rejected", f"Outdoor_{ratio:.0%}", bbox, img_color, conf)
            print(f"       ❌ Stairs #{i} rejected: Outdoor ({ratio:.0%})")
            continue
            
        valid_candidates.append({
            "idx": 5000 + i, # ID-uri speciale pentru scări
            "bbox": bbox,
            "conf": conf,
            "raw": p
        })

    if not valid_candidates:
        return [], []

    # 3. Sortare
    valid_candidates.sort(key=lambda x: x["conf"], reverse=True)
    
    final_stairs_bboxes = []
    final_stairs_exports = []
    
    # A. PRIMA SCARĂ (Cea mai bună) este acceptată automat ca Referință
    best = valid_candidates[0]
    final_stairs_bboxes.append(best["bbox"])
    final_stairs_exports.append({
        "type": "stairs",
        "status": "confirmed",
        "confidence": round(best["conf"], 3),
        "x1": best["bbox"][0], "y1": best["bbox"][1], 
        "x2": best["bbox"][2], "y2": best["bbox"][3]
    })
    
    _save_debug_stairs(debug_dir, best["idx"], "confirmed", "Primary_Ref", best["bbox"], img_color, best["conf"])
    print(f"       ✅ Primary Stairs accepted (conf: {best['conf']:.2f})")
    
    # Salvăm crop-ul de referință pentru Gemini
    ref_crop = None
    ref_path = temp_dir / "stairs_ref.jpg"
    try:
        rx1, ry1, rx2, ry2 = map(int, best["bbox"])
        ref_crop = img_gray[max(0, ry1):min(img_gray.shape[0], ry2), max(0, rx1):min(img_gray.shape[1], rx2)]
        cv2.imwrite(str(ref_path), ref_crop)
    except:
        pass

    # B. ALTE SCĂRI (Dacă au scor apropiat)
    if len(valid_candidates) > 1 and ref_crop is not None:
        secondary_candidates = []
        
        for cand in valid_candidates[1:]:
            # Dacă scorul e foarte mic față de prima, ignorăm
            if cand["conf"] < (best["conf"] - 0.25): # Ex: Best=0.85, ignorăm sub 0.60
                _save_debug_stairs(debug_dir, cand["idx"], "rejected", "Low_Score_Rel", cand["bbox"], img_color, cand["conf"])
                continue
            
            # Pregătim crop pentru Gemini
            cx1, cy1, cx2, cy2 = map(int, cand["bbox"])
            cand_crop = img_gray[max(0, cy1):min(img_gray.shape[0], cy2), max(0, cx1):min(img_gray.shape[1], cx2)]
            c_path = temp_dir / f"stairs_cand_{cand['idx']}.jpg"
            cv2.imwrite(str(c_path), cand_crop)
            
            secondary_candidates.append({
                "idx": cand["idx"],
                "label": "stairs",
                "tmp_path": c_path,
                "bbox": cand["bbox"],
                "conf": cand["conf"]
            })
            
        # Verificare Gemini
        if secondary_candidates:
            print(f"       🧠 Verifying {len(secondary_candidates)} secondary stairs with Gemini...")
            gemini_results = verify_candidates_parallel(secondary_candidates, ref_path, temp_dir)
            
            for cand in secondary_candidates:
                is_valid = gemini_results.get(cand["idx"], False)
                if is_valid:
                    final_stairs_bboxes.append(cand["bbox"])
                    final_stairs_exports.append({
                        "type": "stairs",
                        "status": "confirmed_gemini",
                        "confidence": round(cand["conf"], 3),
                        "x1": cand["bbox"][0], "y1": cand["bbox"][1], 
                        "x2": cand["bbox"][2], "y2": cand["bbox"][3]
                    })
                    _save_debug_stairs(debug_dir, cand["idx"], "confirmed", "Gemini_Match", cand["bbox"], img_color, cand["conf"])
                    print(f"       ✅ Secondary Stairs confirmed by AI (conf: {cand['conf']:.2f})")
                else:
                    _save_debug_stairs(debug_dir, cand["idx"], "rejected", "Gemini_Reject", cand["bbox"], img_color, cand["conf"])
                    print(f"       ❌ Secondary Stairs rejected by AI (conf: {cand['conf']:.2f})")

    return final_stairs_bboxes, final_stairs_exports