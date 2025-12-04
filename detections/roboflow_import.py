# file: engine/detections/roboflow_import.py
from __future__ import annotations

import json
import os
import time
import cv2
import shutil
import requests
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Any

# Încercăm să importăm supervision (inference_sdk nu mai e necesar critic, folosim requests)
try:
    import supervision as sv
except ImportError:
    sv = None

def run_roboflow_import(env: Dict[str, str], work_dir: Path) -> Tuple[bool, List[Dict[str, Any]]]:
    print("\n\n" + "="*50)
    print("🕵️  ROBOFLOW IMPORT: REQUESTS MODE (Fix)")
    print("="*50 + "\n\n")
    
    # 1. Verificări preliminare
    if sv is None:
        print("❌ Lipsește biblioteca 'supervision'.")
        return False, []

    API_KEY = env.get("ROBOFLOW_API_KEY", os.getenv("ROBOFLOW_API_KEY", "")).strip()
    PROJECT = env.get("ROBOFLOW_PROJECT", os.getenv("ROBOFLOW_PROJECT", "house-plan-uwkew")).strip()
    VERSION = env.get("ROBOFLOW_VERSION", os.getenv("ROBOFLOW_VERSION", "5")).strip()
    
    # Configurare parametri - FORȚĂM CONFIDENCE MIC PENTRU A ADUCE TOT
    # Filtrarea o vom face noi ulterior, dar vrem ca API-ul să trimită tot ce vede.
    CONFIDENCE_REQUEST = 5 # Trimitem 5 (adică 5%) către API
    OVERLAP_REQUEST = 50   # 50% overlap permis la server

    # --- CONFIGURARE SLICING ---
    SLICING_THRESHOLD_PX = 3000  
    SLICE_SIZE = 2000            
    OVERLAP_PERCENT = 0.20       # 20% Overlap
    
    # Calculăm overlap-ul în pixeli pentru supervision (fix warning)
    overlap_px_val = int(SLICE_SIZE * OVERLAP_PERCENT)
    SLICE_WH = (SLICE_SIZE, SLICE_SIZE)
    OVERLAP_PX = (overlap_px_val, overlap_px_val)

    if not API_KEY:
        print("❌ ROBOFLOW_API_KEY lipsește din environment")
        return False, []

    plan_jpg = work_dir / "plan.jpg"
    if not plan_jpg.exists():
        print(f"❌ Nu găsesc plan.jpg în {work_dir}")
        return False, []

    # --- SETUP DEBUG FOLDERS ---
    debug_dir = work_dir / "debug_roboflow"
    slices_dir = debug_dir / "slices"
    crops_dir = debug_dir / "crops"
    
    if debug_dir.exists():
        try: shutil.rmtree(debug_dir)
        except: pass
    
    slices_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    # ---------------------------

    # Citire imagine
    image = cv2.imread(str(plan_jpg))
    if image is None: return False, []
    h_img, w_img = image.shape[:2]
    
    print(f"  🔍 Input: {plan_jpg.name} ({w_img}x{h_img} px)")
    print(f"     Project: {PROJECT}/{VERSION}")

    # URL API Roboflow
    infer_url = f"https://detect.roboflow.com/{PROJECT}/{VERSION}"

    # Helper function: Inference via Requests
    def infer_via_requests(img_array: np.ndarray) -> sv.Detections:
        # Codificare imagine
        _, img_encoded = cv2.imencode('.jpg', img_array)
        img_bytes = img_encoded.tobytes()
        
        params = {
            "api_key": API_KEY,
            "confidence": CONFIDENCE_REQUEST, 
            "overlap": OVERLAP_REQUEST
        }
        
        files = {
            'file': ('slice.jpg', img_bytes, 'image/jpeg')
        }
        
        try:
            resp = requests.post(infer_url, params=params, files=files)
            if resp.status_code != 200:
                print(f"⚠️ API Error ({resp.status_code}): {resp.text}")
                return sv.Detections.empty()
            
            result = resp.json()
            
            # --- PARSARE MANUALĂ ROBOFLOW JSON ---
            predictions = result.get("predictions", [])
            if not predictions:
                return sv.Detections.empty()

            xyxy = []
            confidences = []
            class_names = []
            
            for pred in predictions:
                x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                
                xyxy.append([x1, y1, x2, y2])
                confidences.append(pred['confidence'])
                class_names.append(pred['class'])

            if not xyxy:
                 return sv.Detections.empty()

            xyxy = np.array(xyxy)
            confidence = np.array(confidences)
            
            # Generăm ID-uri numerice pentru clase
            unique_classes = list(set(class_names))
            class_map = {name: i for i, name in enumerate(unique_classes)}
            class_id = np.array([class_map[name] for name in class_names])

            return sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
                data={"class_name": np.array(class_names)}
            )
            
        except Exception as e:
            print(f"❌ Exception in request: {e}")
            return sv.Detections.empty()

    # --- LOGICA DE DETECȚIE ---
    start = time.time()
    detections = None

    # Debug helpers
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    slice_counter = 0

    # Funcție debug slice (salvează fiecare bucată procesată)
    def debug_save_slice(image_slice: np.ndarray, slice_dets: sv.Detections):
        nonlocal slice_counter
        slice_counter += 1
        
        labels = []
        if slice_dets.class_id is not None and 'class_name' in slice_dets.data:
             for i, _ in enumerate(slice_dets.class_id):
                conf = slice_dets.confidence[i]
                name = slice_dets.data['class_name'][i]
                labels.append(f"{name} {conf:.2f}")

        annotated = box_annotator.annotate(scene=image_slice.copy(), detections=slice_dets)
        annotated = label_annotator.annotate(scene=annotated, detections=slice_dets, labels=labels)
        cv2.imwrite(str(slices_dir / f"slice_{slice_counter}.jpg"), annotated)

    # Decidem dacă facem Slicing sau nu
    use_slicing = (w_img > SLICING_THRESHOLD_PX) or (h_img > SLICING_THRESHOLD_PX)

    if use_slicing:
        print(f"  ✂️  Mode: SLICING ACTIVE (Imagine > {SLICING_THRESHOLD_PX}px)")
        print(f"      Size: {SLICE_SIZE}x{SLICE_SIZE}, Overlap: {int(SLICE_SIZE * OVERLAP_PERCENT)}px")
        
        def callback(image_slice: np.ndarray) -> sv.Detections:
            # Încercăm de 3 ori în caz de eroare de rețea
            for _ in range(3):
                dets = infer_via_requests(image_slice)
                if len(dets) > 0: # Dacă am găsit ceva, salvăm debug
                     debug_save_slice(image_slice, dets)
                return dets # Returnăm rezultatul (chiar și gol)
            return sv.Detections.empty()

        slicer = sv.InferenceSlicer(
            callback=callback,
            slice_wh=SLICE_WH,
            overlap_ratio_wh=None,       # Dezactivăm parametrul vechi
            overlap_wh=OVERLAP_PX,       # Folosim parametrul nou (pixeli)
            iou_threshold=0.5,           # Filtrare NMS locală între slice-uri
            thread_workers=2             # Paralelism
        )
        detections = slicer(image)
        
    else:
        print(f"  📸 Mode: FULL INFERENCE (Imagine < {SLICING_THRESHOLD_PX}px)")
        detections = infer_via_requests(image)

    elapsed = time.time() - start

    # --- POST-PROCESARE ȘI EXPORT ---
    final_predictions = []
    
    # Desenăm rezultatul pe imaginea full pentru debug final
    debug_img = image.copy()
    
    if detections:
        keep_mask = []
        labels_debug = []
        
        # Extragem numele claselor
        class_names_list = []
        if 'class_name' in detections.data:
            class_names_list = detections.data['class_name']
        elif detections.class_id is not None:
             class_names_list = [str(i) for i in detections.class_id]
            
        for i, cls_raw in enumerate(class_names_list):
            cls = str(cls_raw).lower()
            conf = detections.confidence[i]
            
            # Păstrăm TOT ce vine de la API pentru a fi filtrat de detectorul hibrid
            # Singurul filtru minim este pentru a elimina gunoaie absolute (< 5%)
            is_accepted = conf >= 0.05
            
            keep_mask.append(is_accepted)
            
            if is_accepted:
                bbox = detections.xyxy[i]
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2
                cy = y1 + h / 2
                
                final_predictions.append({
                    "x": float(cx), "y": float(cy), 
                    "width": float(w), "height": float(h),
                    "class": cls,
                    "confidence": float(conf)
                })
                
                # Salvează crop debug (opțional)
                try:
                    y1c, y2c = max(0, int(y1)), min(h_img, int(y2))
                    x1c, x2c = max(0, int(x1)), min(w_img, int(x2))
                    if x2c > x1c and y2c > y1c:
                        crop = image[y1c:y2c, x1c:x2c]
                        safe_name = cls.replace(" ", "_").replace("/", "-")
                        cv2.imwrite(str(crops_dir / f"{safe_name}_{i}_{conf:.2f}.jpg"), crop)
                except: pass

            labels_debug.append(f"{cls} {conf:.2f}")

        # Desenare Debug Final
        mask_arr = np.array(keep_mask, dtype=bool)
        if len(mask_arr) > 0 and mask_arr.any():
            detections_filtered = detections[mask_arr]
            labels_filtered = np.array(labels_debug)[mask_arr]
            
            debug_img = box_annotator.annotate(scene=debug_img, detections=detections_filtered)
            debug_img = label_annotator.annotate(scene=debug_img, detections=detections_filtered, labels=labels_filtered)

    cv2.imwrite(str(debug_dir / "full_debug.jpg"), debug_img)

    print(f"  ✅ {len(final_predictions)} detecții în {elapsed:.2f}s")
    
    # Salvare JSON
    final_json = {
        "predictions": final_predictions,
        "image": {"width": w_img, "height": h_img}
    }
    
    detections_dir = work_dir / "export_objects"
    detections_dir.mkdir(parents=True, exist_ok=True)
    (detections_dir / "detections.json").write_text(json.dumps(final_json), encoding="utf-8")

    return True, final_predictions