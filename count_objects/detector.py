# new/runner/count_objects/detector.py
from __future__ import annotations

import json
import time
import shutil
import cv2
from pathlib import Path
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import (
    CONF_THRESHOLD,
    OVERLAP,
    TEMPLATE_SIMILARITY,
    GEMINI_THRESHOLD_MIN,
    GEMINI_THRESHOLD_MAX,
    ROBOFLOW_MAIN_PROJECT,
    ROBOFLOW_MAIN_VERSION,
    MAX_TYPE_WORKERS
)
from .roboflow_api import infer_roboflow
from .preprocessing import load_templates
from .template_matching import process_detections_parallel
from .gemini_verification import verify_candidates_parallel
from .stairs_detection import process_stairs
from .visualization import draw_results, export_to_json


def _norm_class(c: str) -> str:
    """Normalizează numele clasei."""
    return (c or "").lower().replace("_", "-").strip()


def _fetch_roboflow_data(plan_image: Path, roboflow_config: dict):
    """(Legacy) Fetch Roboflow data în paralel: scări + uși/ferestre simultan."""
    def get_stairs():
        return process_stairs(plan_image, roboflow_config["api_key"])
    
    def get_main():
        return infer_roboflow(
            plan_image,
            roboflow_config["api_key"],
            roboflow_config["workspace"],
            ROBOFLOW_MAIN_PROJECT,
            ROBOFLOW_MAIN_VERSION,
            confidence=CONF_THRESHOLD,
            overlap=OVERLAP
        )
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_stairs = executor.submit(get_stairs)
        future_main = executor.submit(get_main)
        return future_stairs.result(), future_main.result()


def _process_object_type(
    label: str,
    folder: Path,
    preds_filtered: list,
    gray_image,
    img_width: int,
    img_height: int,
    used_boxes: list,
    temp_dir: Path,
    has_stairs: bool = False
) -> dict:
    """Procesează un tip de obiect (door/window/etc) complet (Template + Gemini)."""
    print(f"\n       {'='*50}")
    print(f"       {label.upper()}")
    print(f"       {'='*50}")
    
    templates = load_templates(folder)
    if not templates:
        print(f"       [WARN] No templates for {label}")
        return {"confirm": [], "oblique": [], "reject": []}
    
    print(f"       Loaded {len(templates)} template variations")
    
    # Filtrează predicții relevante pentru acest tip (ex: doar 'door')
    relevant = []
    for p in preds_filtered:
        cls = _norm_class(str(p.get("class", "")))
        if label == "door" and ("door" in cls and "double" not in cls):
            relevant.append(p)
        elif label == "double-door" and ("double" in cls and "door" in cls):
            relevant.append(p)
        elif label == "window" and ("window" in cls and "double" not in cls):
            relevant.append(p)
        elif label == "double-window" and ("double" in cls and "window" in cls):
            relevant.append(p)
    
    print(f"       Found {len(relevant)} relevant predictions for {label}")
    
    if not relevant:
        return {"confirm": [], "oblique": [], "reject": []}
    
    # Procesare PARALELĂ a tuturor detecțiilor
    print(f"       🔄 Template matching (parallel)...")
    t0 = time.time()
    
    processed = process_detections_parallel(
        relevant,
        gray_image,
        templates,
        used_boxes,
        img_width,
        img_height,
        has_stairs=has_stairs
    )
    
    print(f"       ✅ Template matching done in {time.time()-t0:.2f}s")
    
    # Clasificare rezultate
    results = {"confirm": [], "oblique": [], "reject": []}
    candidates_for_gemini = []
    
    for res in processed:
        if res["skip"]:
            continue
        
        # Scoruri: Best Similarity (Pixel) + Confidence (Roboflow)
        print(f"       #{res['idx']} conf={res['conf']:.2f}, sim={res['best_sim']:.3f}, combined={res['combined']:.3f}")
        
        # 1. Confirmare puternică prin Template Matching
        if res["combined"] >= GEMINI_THRESHOLD_MAX and res["best_sim"] > TEMPLATE_SIMILARITY:
            results["confirm"].append(res["bbox"])
            used_boxes.append(res["bbox"])
            print(f"       ✅ CONFIRMED (template)")
        
        # 2. Respingere clară
        elif res["combined"] < GEMINI_THRESHOLD_MIN:
            results["reject"].append(res["bbox"])
            print(f"       ❌ REJECTED (low score)")
        
        # 3. Zona gri -> Gemini
        else:
            x1, y1, x2, y2 = res["bbox"]
            crop = gray_image[y1:y2, x1:x2]
            tmp_path = temp_dir / f"maybe_{label}_{res['idx']}.jpg"
            cv2.imwrite(str(tmp_path), crop)
            
            candidates_for_gemini.append({
                "idx": res["idx"],
                "bbox": res["bbox"],
                "tmp_path": tmp_path,
                "label": label
            })
            print(f"       🔍 → Gemini verification")
    
    # Verificare Gemini PARALELIZAT
    if candidates_for_gemini:
        print(f"\n       🧠 Gemini verification ({len(candidates_for_gemini)} candidates)...")
        try:
            sample_template = next(folder.glob("*.png"))
            t0 = time.time()
            gemini_results = verify_candidates_parallel(candidates_for_gemini, sample_template, temp_dir)
            print(f"       ✅ Gemini done in {time.time()-t0:.2f}s")
            
            for cand in candidates_for_gemini:
                if gemini_results.get(cand["idx"], False):
                    results["oblique"].append(cand["bbox"])
                    used_boxes.append(cand["bbox"])
                    print(f"       #{cand['idx']} 🔄 GEMINI CONFIRMED")
                else:
                    results["reject"].append(cand["bbox"])
                    print(f"       #{cand['idx']} ❌ REJECTED (Gemini)")
        except Exception as e:
            print(f"       [WARN] Gemini skip: {e}")
            for cand in candidates_for_gemini:
                results["reject"].append(cand["bbox"])
    
    return results


def run_hybrid_detection(
    plan_image: Path,
    exports_dir: Path,
    output_dir: Path,
    roboflow_config: dict,
    total_plans: int = 1,
    external_predictions: list = None # <--- ARGUMENT NOU
) -> Tuple[bool, str]:
    """
    Rulează pipeline-ul de validare. 
    Integrează Slicing Results (Doors/Windows) cu modelul specializat de Scări.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp"
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"       [CLEANUP] Temp folder ready: {temp_dir}")
    
    try:
        # ==========================================
        # STEP 1: OBȚINERE DETECȚII (EXTERN + SCĂRI)
        # ==========================================
        
        all_detections = []
        stairs_bbox = None
        stairs_export = {}

        # 1.A. DETECTARE SCĂRI (OBLIGATORIU)
        # Rulăm detectia de scări separat pentru că modelul de slicing 
        # (house-plan) poate să nu aibă scări bune.
        print(f"       [STAIRS] Running specialized stairs detection...")
        try:
            # Folosim funcția dedicată din stairs_detection.py
            stairs_bbox, stairs_export = process_stairs(plan_image, roboflow_config["api_key"])
            if stairs_bbox:
                print(f"       [STAIRS] ✅ Found stairs at {stairs_bbox}")
            else:
                print(f"       [STAIRS] No stairs found.")
        except Exception as e:
            print(f"       [STAIRS] ⚠️ Error detecting stairs: {e}")

        # 1.B. DETECȚII PRINCIPALE (Doors/Windows)
        if external_predictions is not None:
            print(f"       🚀 USING EXTERNAL DETECTIONS (Slicing input: {len(external_predictions)} objects)")
            # Folosim ce a venit din slicing
            all_detections = external_predictions
        else:
            # Fallback la metoda veche (Inferență pe toată imaginea)
            print(f"       [FALLBACK] Using standard full-image inference...")
            rf_result = infer_roboflow(
                plan_image,
                roboflow_config["api_key"],
                roboflow_config["workspace"],
                ROBOFLOW_MAIN_PROJECT,
                ROBOFLOW_MAIN_VERSION,
                confidence=CONF_THRESHOLD,
                overlap=OVERLAP
            )
            all_detections = rf_result.get("predictions", [])

        # Filtrare sumară (eliminăm erorile grosolane, sub 10%)
        # Notă: Slicing-ul ne poate da obiecte cu 15-20% pe care vrem să le verificăm
        preds_filtered = [p for p in all_detections if float(p.get("confidence", 0.0)) >= 0.10]
        print(f"       Processing {len(preds_filtered)} potential objects through Pipeline...")
        
        # ==========================================
        # STEP 2: PREPROCESARE IMAGINE
        # ==========================================
        img = cv2.imread(str(plan_image))
        if img is None:
            return False, f"Cannot read image: {plan_image}"
        
        gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        img_height, img_width = img.shape[:2]
        
        EXPORTS = {
            "door": exports_dir / "door",
            "double-door": exports_dir / "double_door",
            "window": exports_dir / "window",
            "double-window": exports_dir / "double_window"
        }
        
        # ==========================================
        # STEP 3: EXCLUDERE ZONA SCĂRILOR
        # ==========================================
        used_boxes = []
        has_stairs = False
        
        if stairs_bbox:
            stairs_box = (
                stairs_bbox["x1"],
                stairs_bbox["y1"],
                stairs_bbox["x2"],
                stairs_bbox["y2"]
            )
            used_boxes.append(stairs_box)
            has_stairs = True
        
        # ==========================================
        # STEP 4: PROCESARE TIPURI OBIECTE - PARALEL
        # ==========================================
        print(f"\n       [STEP] Processing object types (parallel: {len(EXPORTS)} types)...")
        t0 = time.time()
        
        all_results = {}
        
        def process_type(label_folder_pair):
            label, folder = label_folder_pair
            return label, _process_object_type(
                label,
                folder,
                preds_filtered,
                gray,
                img_width,
                img_height,
                used_boxes,
                temp_dir,
                has_stairs=has_stairs
            )
        
        with ThreadPoolExecutor(max_workers=MAX_TYPE_WORKERS) as executor:
            futures = {
                executor.submit(process_type, (label, folder)): label
                for label, folder in EXPORTS.items()
            }
            
            for future in as_completed(futures):
                label, results = future.result()
                all_results[label] = results
        
        print(f"\n       ✅ All types processed in {time.time()-t0:.2f}s")
        
        # ==========================================
        # STEP 5: VIZUALIZARE + EXPORT FINAL
        # ==========================================
        out_img = draw_results(img, all_results, stairs_bbox)
        detections_export = export_to_json(all_results, stairs_export)
        
        out_image_path = output_dir / "plan_detected_all_hybrid.jpg"
        out_json_path = output_dir / "detections_all.json"
        
        cv2.imwrite(str(out_image_path), out_img)
        
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(detections_export, f, indent=2, ensure_ascii=False)
        
        # Rezumat la consolă
        print(f"\n       ✅ Image saved: {out_image_path}")
        if stairs_export:
            conf = stairs_export.get("confidence", 0)
            print(f"       🟢 Stairs: DETECTED (conf: {conf:.2f})")
        else:
            print(f"       ⚪ Stairs: None")

        summary = f"{len(detections_export)} detections"
        return True, summary
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Error: {e}"