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
    """Fetch Roboflow data în paralel: scări + uși/ferestre simultan."""
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
    
    stairs_result = None
    main_result = None
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_stairs = executor.submit(get_stairs)
        future_main = executor.submit(get_main)
        
        stairs_result = future_stairs.result()
        main_result = future_main.result()
    
    return stairs_result, main_result


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
    """Procesează un tip de obiect (door/window/etc) complet."""
    print(f"\n       {'='*50}")
    print(f"       {label.upper()}")
    print(f"       {'='*50}")
    
    templates = load_templates(folder)
    if not templates:
        print(f"       [WARN] No templates for {label}")
        return {"confirm": [], "oblique": [], "reject": []}
    
    print(f"       Loaded {len(templates)} template variations")
    
    # Filtrează predicții relevante
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
    
    print(f"       Found {len(relevant)} relevant predictions")
    
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
            skip_reason = res.get("skip_reason", "unknown")
            print(f"       #{res['idx']} skip → {skip_reason}")
            continue
        
        print(f"       #{res['idx']} conf={res['conf']:.2f}, sim={res['best_sim']:.3f}, combined={res['combined']:.3f}")
        
        if res["combined"] >= GEMINI_THRESHOLD_MAX and res["best_sim"] > TEMPLATE_SIMILARITY:
            results["confirm"].append(res["bbox"])
            used_boxes.append(res["bbox"])
            print(f"       ✅ CONFIRMED (template)")
        
        elif res["combined"] < GEMINI_THRESHOLD_MIN:
            results["reject"].append(res["bbox"])
            print(f"       ❌ REJECTED (low score)")
        
        else:
            # Salvează crop pentru Gemini
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
        except StopIteration:
            print(f"       [WARN] No templates for Gemini verification")
            for cand in candidates_for_gemini:
                results["reject"].append(cand["bbox"])
            return results
        
        t0 = time.time()
        gemini_results = verify_candidates_parallel(
            candidates_for_gemini,
            sample_template,
            temp_dir
        )
        print(f"       ✅ Gemini done in {time.time()-t0:.2f}s")
        
        for cand in candidates_for_gemini:
            is_valid = gemini_results.get(cand["idx"], False)
            if is_valid:
                results["oblique"].append(cand["bbox"])
                used_boxes.append(cand["bbox"])
                print(f"       #{cand['idx']} 🔄 GEMINI CONFIRMED")
            else:
                results["reject"].append(cand["bbox"])
                print(f"       #{cand['idx']} ❌ REJECTED (Gemini)")
    
    return results


def run_hybrid_detection(
    plan_image: Path,
    exports_dir: Path,
    output_dir: Path,
    roboflow_config: dict,
    total_plans: int = 1,
) -> Tuple[bool, str]:
    """
    Rulează detecția hybrid cu PARALELIZARE MAXIMĂ + excludere zone scări.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp"
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"       [CLEANUP] Temp folder ready: {temp_dir}")
    
    try:
        # ==========================================
        # STEP 1: ROBOFLOW (SCĂRI + UȘI/FERESTRE) - PARALEL
        # ==========================================

        # DECIZIE SCĂRI: Skip dacă avem doar 1 plan
        if total_plans == 1:
            print(f"       [STAIRS] Skipping stairs detection (only 1 plan in run)")
            stairs_bbox = None
            stairs_export = {}
            
            # Doar detecția principală
            print(f"       [STEP] Fetching Roboflow main detections...")
            t0 = time.time()
            rf_result = infer_roboflow(
                plan_image,
                roboflow_config["api_key"],
                roboflow_config["workspace"],
                ROBOFLOW_MAIN_PROJECT,
                ROBOFLOW_MAIN_VERSION,
                confidence=CONF_THRESHOLD,
                overlap=OVERLAP
            )
            print(f"       [DONE] Main detections in {time.time()-t0:.2f}s")
        else:

            print(f"       [STEP] Fetching Roboflow data (parallel: stairs + main)...")
            t0 = time.time()
            
            (stairs_bbox, stairs_export), rf_result = _fetch_roboflow_data(plan_image, roboflow_config)
            
            print(f"       [DONE] Roboflow data in {time.time()-t0:.2f}s")
            
        preds = rf_result.get("predictions", []) or []
        preds_filtered = [p for p in preds if float(p.get("confidence", 0.0)) >= CONF_THRESHOLD]
        print(f"       Found {len(preds_filtered)} detections (doors/windows)")
        
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
        # IMPORTANT: Marchează zona scării ca ocupată
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
            print(f"       [STAIRS] Zona scării marcată ca exclusă: {stairs_box}")
        
        # ==========================================
        # STEP 3: PROCESARE TIPURI OBIECTE - PARALEL
        # ==========================================
        print(f"\n       [STEP] Processing object types (parallel: {len(EXPORTS)} types)...")
        t0 = time.time()
        
        all_results = {}
        
        def process_type(label_folder_pair):
            """Helper pentru procesare paralelă."""
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
        
        # Procesează toate tipurile în paralel
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
        # STEP 4: VIZUALIZARE + EXPORT
        # ==========================================
        out_img = draw_results(img, all_results, stairs_bbox)
        detections_export = export_to_json(all_results, stairs_export)
        
        out_image_path = output_dir / "plan_detected_all_hybrid.jpg"
        out_json_path = output_dir / "detections_all.json"
        
        cv2.imwrite(str(out_image_path), out_img)
        
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(detections_export, f, indent=2, ensure_ascii=False)
        
        # Rezumat
        print(f"\n       ✅ Image saved: {out_image_path}")
        print(f"       📄 JSON saved: {out_json_path}")
        print(f"       Total: {len(detections_export)} detections")
        
        if stairs_export:
            conf = stairs_export.get("confidence", 0)
            print(f"       🟢 Stairs: 1 detected (conf: {conf:.2f})")
        
        for cls in all_results:
            conf = len(all_results[cls]["confirm"])
            gem = len(all_results[cls]["oblique"])
            rej = len(all_results[cls]["reject"])
            print(f"       🔹 {cls}: {conf} confirmed | {gem} Gemini | {rej} rejected")
        
        summary = f"{len(detections_export)} detections"
        return True, summary
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Error: {e}"