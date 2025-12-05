# engine/count_objects/detector.py
from __future__ import annotations

import json
import time
import shutil
import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Dict, List, Any
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import (
    CONF_THRESHOLD,
    OVERLAP,
    ROBOFLOW_MAIN_PROJECT,
    ROBOFLOW_MAIN_VERSION,
    MAX_TYPE_WORKERS
)
from .roboflow_api import infer_roboflow
from .preprocessing import load_templates
from .template_matching import process_detections_parallel
from .gemini_verification import verify_candidates_parallel
from .stairs_detection import process_stairs
from .visualization import draw_results, draw_final_orange, export_to_json

# ==========================================
# DATA STRUCTURES
# ==========================================

class ValidationSource(Enum):
    ROBOFLOW_DIRECT = "roboflow_direct"
    TEMPLATE_STRONG = "template_strong"
    GEMINI_CONFIRMED = "gemini_confirmed"
    REJECTED_LOW_SCORE = "rejected_low_score"
    REJECTED_OVERLAP = "rejected_overlap"
    REJECTED_STAIRS = "rejected_stairs_overlap"
    REJECTED_GEMINI = "rejected_gemini"
    REJECTED_OUTDOOR = "rejected_outdoor_mask" 
    REJECTED_EMPTY = "rejected_empty_crop"

@dataclass
class DetectionRecord:
    idx: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    template_similarity: float
    combined_score: float
    validation_source: ValidationSource
    reason: str
    details: Dict = None
    
    def to_dict(self):
        d = asdict(self)
        d['validation_source'] = self.validation_source.value
        return d

# ==========================================
# HELPER: OUTDOOR CHECK
# ==========================================

def _is_outdoors(bbox: Tuple[int, int, int, int], mask: np.ndarray, threshold: float = 0.75) -> Tuple[bool, float]:
    """
    Verifică dacă bbox-ul cade pe zona albă (outdoor) a măștii.
    Returnează (True dacă e outdoor, procentaj_alb).
    """
    if mask is None:
        return False, 0.0
    
    x1, y1, x2, y2 = bbox
    # Clamp coordinates
    h, w = mask.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    
    if x2 <= x1 or y2 <= y1:
        return False, 0.0
        
    crop = mask[y1:y2, x1:x2]
    if crop.size == 0:
        return False, 0.0

    # Alb (255) sau foarte deschis înseamnă outdoor/gol
    white_pixels = np.sum(crop > 240)
    ratio = white_pixels / crop.size
    
    return ratio > threshold, ratio

# ==========================================
# DEBUG SAVER
# ==========================================

def save_detection_debug(debug_root: Path, img_color: np.ndarray, record: DetectionRecord):
    """Salvează crop și info pentru debugging."""
    try:
        status = "REJECTED" if record.validation_source.value.startswith("rejected") else "ACCEPTED"
        src_clean = record.validation_source.value.replace("rejected_", "").replace("roboflow_", "")
        
        folder_name = f"{record.idx:04d}_{record.class_name}_{status}_{src_clean}"
        save_path = debug_root / folder_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 1. JSON
        with open(save_path / "info.json", "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)
            
        # 2. IMAGE
        h, w = img_color.shape[:2]
        x1, y1, x2, y2 = map(int, record.bbox)
        
        pad = 60
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
        
        crop = img_color[cy1:cy2, cx1:cx2].copy()
        
        color = (0, 0, 255) if status == "REJECTED" else (0, 255, 0)
        rx1, ry1 = x1 - cx1, y1 - cy1
        rx2, ry2 = x2 - cx1, y2 - cy1
        
        cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), color, 2)
        txt = f"{record.reason} ({record.confidence:.2f})"
        cv2.putText(crop, txt, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imwrite(str(save_path / "context.jpg"), crop)
    except Exception as e:
        print(f"       ⚠️ Failed saving debug for #{record.idx}: {e}")

# ==========================================
# OVERLAP CHECK
# ==========================================

def _check_overlap_all(bbox, used_boxes, has_stairs):
    from .template_matching import overlap
    
    # Verificăm cu TOATE elementele existente (scări + altele)
    for ob in used_boxes:
        # Dacă e scară, toleranță mică (nu vrem geam peste scară) -> dar simplificăm la global 0.60
        # Dacă vrei logica strictă "nu te atinge de scară":
        # Dar `used_boxes` e doar o listă de coordonate.
        
        ov = overlap(bbox, ob)
        if ov > 0.60: # 60% suprapunere = reject
             return True, "overlap", ov
             
    return False, None, 0.0

# ==========================================
# UTILS
# ==========================================

ENHANCED_COLORS = {
    "stairs": (0, 255, 0),
    ValidationSource.ROBOFLOW_DIRECT: (0, 255, 0),
    ValidationSource.TEMPLATE_STRONG: (0, 140, 255),
    ValidationSource.GEMINI_CONFIRMED: (255, 0, 255),
    ValidationSource.REJECTED_LOW_SCORE: (0, 0, 255),
    ValidationSource.REJECTED_OVERLAP: (255, 255, 0),
    ValidationSource.REJECTED_STAIRS: (128, 128, 255),
    ValidationSource.REJECTED_GEMINI: (255, 0, 128),
    ValidationSource.REJECTED_OUTDOOR: (128, 128, 128),
    ValidationSource.REJECTED_EMPTY: (128, 128, 128),
}

def _norm_class(c: str) -> str:
    return (c or "").lower().replace("_", "-").strip()

def create_detailed_legend():
    return np.ones((400, 400, 3), dtype=np.uint8) * 255 # Placeholder

def export_detailed_json(all_results, stairs_exports, meta):
    out = {"meta": meta, "detections": []}
    # Adăugăm toate scările
    if stairs_exports:
        for s in stairs_exports:
            out["detections"].append({**s, "validation_source": "specialized_model"})
            
    for r in all_results.values():
        for rec in r.get("records", []):
            out["detections"].append(rec.to_dict())
    return out

# ==========================================
# PROCESSOR PER TIP
# ==========================================

def _process_object_type(
    label: str, folder: Path, preds_low: list, preds_high: list,
    color_img, gray_img, w, h, used_boxes, temp_dir, debug_dir, 
    has_stairs, outdoor_mask
) -> Dict:
    
    print(f"\n       {'='*30} {label.upper()} {'='*30}")
    templates = load_templates(folder)
    if not templates:
        print(f"       [WARN] No templates for {label}. Low conf will be skipped.")
        templates = {} 

    def is_relevant(p):
        cls = _norm_class(str(p.get("class", "")))
        if label == "door" and ("door" in cls and "double" not in cls): return True
        elif label == "double-door" and ("double" in cls and "door" in cls): return True
        elif label == "window" and ("window" in cls and "double" not in cls): return True
        elif label == "double-window" and ("double" in cls and "window" in cls): return True
        return False

    relevant_low = [p for p in preds_low if is_relevant(p)]
    relevant_high = [p for p in preds_high if is_relevant(p)]
    
    results = {"confirm": [], "oblique": [], "reject": [], "records": []}
    
    # ------------------------------------------------------------------
    # 1. PREGĂTIRE REFERINȚĂ PENTRU GEMINI
    # ------------------------------------------------------------------
    reference_crop_path = None
    if relevant_high:
        # Primul din listă e cel mai bun (sortat anterior)
        best = relevant_high[0]
        bx, by, bw, bh = best["x"], best["y"], best["width"], best["height"]
        bx1, by1 = int(bx - bw/2), int(by - bh/2)
        bx2, by2 = int(bx + bw/2), int(by + bh/2)
        
        # Salvăm crop-ul de referință
        ref_crop = gray_img[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
        if ref_crop.size > 0:
            reference_crop_path = temp_dir / f"ref_{label}.jpg"
            cv2.imwrite(str(reference_crop_path), ref_crop)
            print(f"       ⭐ Reference found (conf: {best['confidence']:.2f})")
    
    if not reference_crop_path and templates:
        # Fallback la un template generic
        generic = list(folder.glob("*.png"))[0]
        reference_crop_path = generic
        print(f"       ⚠️ Using generic template as ref: {generic.name}")

    # ------------------------------------------------------------------
    # 2. LOW CONFIDENCE (TEMPLATE + GEMINI)
    # ------------------------------------------------------------------
    if relevant_low and templates:
        processed_low = process_detections_parallel(relevant_low, gray_img, templates, used_boxes, w, h, has_stairs)
        candidates_gemini = []
        
        for res in processed_low:
            bbox = res["bbox"]
            rec = None
            
            # A. Filtru Mască Outdoor
            is_out, ratio = _is_outdoors(bbox, outdoor_mask)
            if is_out:
                rec = DetectionRecord(res["idx"], label, bbox, res["conf"], 0, 0, 
                                      ValidationSource.REJECTED_OUTDOOR, f"Outdoor {ratio:.0%}", {})
                results["reject"].append(bbox)
                if rec: results["records"].append(rec); save_detection_debug(debug_dir, color_img, rec)
                continue

            # B. Filtru Overlap intern / erori
            if res["skip"]:
                reason = res.get("skip_reason", "err")
                src = ValidationSource.REJECTED_STAIRS if "stairs" in reason else ValidationSource.REJECTED_OVERLAP
                rec = DetectionRecord(res["idx"], label, bbox, res["conf"], 0, 0, src, reason, {})
                results["reject"].append(bbox)
            else:
                sim = res["best_sim"]
                combined = res["combined"]
                conf = res["conf"]
                
                # C. REGULA: > 95% SIMILARITY -> ACCEPT
                if sim > 0.95:
                    rec = DetectionRecord(res["idx"], label, bbox, conf, sim, combined,
                                          ValidationSource.TEMPLATE_STRONG, f"Sim {sim:.2f} > 0.95", {})
                    results["confirm"].append(bbox)
                    used_boxes.append(bbox)
                
                else:
                    # D. Fallback Gemini
                    if combined < 0.20:
                        rec = DetectionRecord(res["idx"], label, bbox, conf, sim, combined,
                                              ValidationSource.REJECTED_LOW_SCORE, f"Score {combined:.2f} too low", {})
                        results["reject"].append(bbox)
                    else:
                        # Queue Gemini
                        x1, y1, x2, y2 = bbox
                        crop = gray_img[y1:y2, x1:x2]
                        p = temp_dir / f"check_{label}_{res['idx']}.jpg"
                        cv2.imwrite(str(p), crop)
                        candidates_gemini.append({**res, "tmp_path": p, "label": label})
                        continue 
            
            if rec:
                results["records"].append(rec)
                save_detection_debug(debug_dir, color_img, rec)

        # E. Procesare Gemini Batch
        if candidates_gemini:
            print(f"       🧠 Gemini checking {len(candidates_gemini)} candidates...")
            
            if reference_crop_path:
                gem_res = verify_candidates_parallel(candidates_gemini, reference_crop_path, temp_dir)
                for cand in candidates_gemini:
                    ok = gem_res.get(cand["idx"], False)
                    src = ValidationSource.GEMINI_CONFIRMED if ok else ValidationSource.REJECTED_GEMINI
                    status_text = "Gemini Confirmed" if ok else "Gemini Rejected"
                    
                    rec = DetectionRecord(cand["idx"], label, cand["bbox"], cand["conf"], cand["best_sim"], 
                                          cand["combined"], src, f"{status_text} (Sim={cand['best_sim']:.2f})", 
                                          {"gemini": ok})
                    
                    if ok:
                        results["oblique"].append(cand["bbox"])
                        used_boxes.append(cand["bbox"])
                    else:
                        results["reject"].append(cand["bbox"])
                    
                    results["records"].append(rec)
                    save_detection_debug(debug_dir, color_img, rec)
            else:
                for cand in candidates_gemini:
                    rec = DetectionRecord(cand["idx"], label, cand["bbox"], cand["conf"], 0, 0, 
                                          ValidationSource.REJECTED_GEMINI, "No Reference for AI", {})
                    results["reject"].append(cand["bbox"])
                    results["records"].append(rec)
                    save_detection_debug(debug_dir, color_img, rec)

    # ------------------------------------------------------------------
    # 3. HIGH CONFIDENCE (DIRECT ACCEPT)
    # ------------------------------------------------------------------
    for idx, p in enumerate(relevant_high, start=3000):
        conf = float(p.get("confidence", 0))
        cx, cy, cw, ch = p["x"], p["y"], p["width"], p["height"]
        x1, y1 = int(cx - cw/2), int(cy - ch/2)
        x2, y2 = int(cx + cw/2), int(cy + ch/2)
        bbox = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
        
        # A. Outdoor Check
        is_out, ratio = _is_outdoors(bbox, outdoor_mask)
        if is_out:
            rec = DetectionRecord(idx, label, bbox, conf, 0, conf, 
                                  ValidationSource.REJECTED_OUTDOOR, f"Outdoor {ratio:.0%}", {})
            results["reject"].append(bbox)
            results["records"].append(rec)
            save_detection_debug(debug_dir, color_img, rec)
            continue

        # B. Overlap Check
        is_ov, o_type, o_val = _check_overlap_all(bbox, used_boxes, has_stairs)
        if is_ov:
            src = ValidationSource.REJECTED_STAIRS if o_type == "stairs" else ValidationSource.REJECTED_OVERLAP
            rec = DetectionRecord(idx, label, bbox, conf, 0, conf, src, f"Overlap {o_val:.2f}", {})
            results["reject"].append(bbox)
        else:
            rec = DetectionRecord(idx, label, bbox, conf, 0, conf, ValidationSource.ROBOFLOW_DIRECT, "High Confidence", {})
            results["confirm"].append(bbox)
            used_boxes.append(bbox)
        
        results["records"].append(rec)
        save_detection_debug(debug_dir, color_img, rec)

    return results


def run_hybrid_detection(
    plan_image: Path,
    exports_dir: Path,
    output_dir: Path,
    roboflow_config: dict,
    total_plans: int = 1,
    external_predictions: list = None,
    outdoor_mask_path: Path = None
) -> Tuple[bool, str]:
    
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp"
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    debug_dir = output_dir / "debug_crops"
    if debug_dir.exists(): shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n 🔍 VALIDATION & DEBUG PIPELINE \n{'='*60}")

    try:
        # Load Images
        img_color = cv2.imread(str(plan_image))
        if img_color is None: return False, "Cannot read plan image"
        img_gray = cv2.equalizeHist(cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY))
        h, w = img_color.shape[:2]

        # Load Outdoor Mask
        outdoor_mask = None
        if outdoor_mask_path and outdoor_mask_path.exists():
            print(f"       🏞️  Loading outdoor mask: {outdoor_mask_path.name}")
            mask_raw = cv2.imread(str(outdoor_mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_raw is not None:
                # Resize la dimensiunea planului curent (pentru siguranță)
                outdoor_mask = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            print("       ⚠️  Outdoor mask not found. Skipping outdoor filter.")

        # =========================================================
        # 1. PROCESS STAIRS (MULTI + AI VERIFICATION)
        # =========================================================
        stairs_bboxes = []
        stairs_exports = []
        
        try:
            # Funcția process_stairs returnează liste (bbox-uri și exporturi)
            # Aceasta include logica de filtrare outdoor și verificare cu Gemini (Ref vs Candidates)
            stairs_bboxes, stairs_exports = process_stairs(
                plan_image, 
                roboflow_config["api_key"], 
                outdoor_mask,
                img_gray, 
                img_color,
                temp_dir,
                debug_dir
            )
        except Exception as e:
            print(f"       ⚠️ Stairs detection failed: {e}")

        # Add stairs to used boxes (all confirmed ones)
        used_boxes = []
        has_stairs = False
        if stairs_bboxes:
            for sb in stairs_bboxes:
                used_boxes.append((sb[0], sb[1], sb[2], sb[3]))
            has_stairs = True

        # =========================================================
        # 2. PROCESS DOORS / WINDOWS
        # =========================================================
        all_detections = external_predictions if external_predictions else []
        preds_filtered = [p for p in all_detections if float(p.get("confidence", 0)) >= 0.10]
        preds_filtered.sort(key=lambda x: float(x.get("confidence", 0)), reverse=True)
        
        preds_low = [p for p in preds_filtered if 0.10 <= float(p.get("confidence", 0)) < 0.50]
        preds_high = [p for p in preds_filtered if float(p.get("confidence", 0)) >= 0.50]

        print(f"       📊 Input: {len(all_detections)} | High: {len(preds_high)} | Low: {len(preds_low)}")

        EXPORTS = {
            "door": exports_dir / "door",
            "double-door": exports_dir / "double_door",
            "window": exports_dir / "window",
            "double-window": exports_dir / "double_window"
        }
        
        all_results = {}
        with ThreadPoolExecutor(max_workers=4) as exc:
            futures = {
                exc.submit(_process_object_type, label, folder, preds_low, preds_high, 
                           img_color, img_gray, w, h, used_boxes, temp_dir, debug_dir, 
                           has_stairs, outdoor_mask): label
                for label, folder in EXPORTS.items()
            }
            for f in as_completed(futures):
                all_results[futures[f]] = f.result()

        # =========================================================
        # 3. EXPORT & VISUALIZATION
        # =========================================================
        
        # Debug Viz (toate culorile)
        viz = draw_results(img_color, all_results, stairs_bboxes)
        cv2.imwrite(str(output_dir / "detections_all_enhanced.jpg"), viz)
        
        # UI Viz (Orange Only)
        viz_orange = draw_final_orange(img_color, all_results, stairs_bboxes)
        cv2.imwrite(str(output_dir / "final_orange.jpg"), viz_orange)
        
        meta = {"image": plan_image.name, "time": time.time()}
        
        # JSON Detailed
        full_json = export_detailed_json(all_results, stairs_exports, meta)
        with open(output_dir / "detections_detailed.json", "w") as f:
            json.dump(full_json, f, indent=2)
            
        # JSON Legacy (pentru următorii pași)
        legacy = []
        if stairs_exports:
            legacy.extend(stairs_exports)
            
        for lbl, res in all_results.items():
            if "confirm" in res:
                for b in res["confirm"]: legacy.append({"type": lbl, "status": "confirmed", "x1":b[0], "y1":b[1], "x2":b[2], "y2":b[3]})
            if "oblique" in res:
                for b in res["oblique"]: legacy.append({"type": lbl, "status": "gemini", "x1":b[0], "y1":b[1], "x2":b[2], "y2":b[3]})
        
        with open(output_dir / "detections_all.json", "w") as f:
            json.dump(legacy, f, indent=2)

        cnt = len([d for d in full_json["detections"] if "rejected" not in d.get("validation_source", "")])
        return True, f"Validated: {cnt}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, str(e)