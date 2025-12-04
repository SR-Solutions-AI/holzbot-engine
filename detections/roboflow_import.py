# file: engine/detections/roboflow_import.py
from __future__ import annotations

import json
import os
import time
import cv2
import shutil
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Any

# Încercăm să importăm bibliotecile necesare
try:
    from inference_sdk import InferenceHTTPClient
    import supervision as sv
except ImportError:
    InferenceHTTPClient = None
    sv = None

def run_roboflow_import(env: Dict[str, str], work_dir: Path) -> Tuple[bool, List[Dict[str, Any]]]:
    print("\n\n" + "="*50)
    print("🕵️  ROBOFLOW IMPORT: SMART MODE (Modified)")
    print("="*50 + "\n\n")
    
    """
    Importă detecții de la Roboflow folosind 'supervision.InferenceSlicer'.
    Returnează lista de predicții pentru a fi procesată ulterior.
    """
    # 1. Verificări preliminare
    if InferenceHTTPClient is None or sv is None:
        print("❌ Lipsesc bibliotecile 'inference-sdk' sau 'supervision'.")
        return False, []

    API_KEY = env.get("ROBOFLOW_API_KEY", os.getenv("ROBOFLOW_API_KEY", "")).strip()
    PROJECT = env.get("ROBOFLOW_PROJECT", os.getenv("ROBOFLOW_PROJECT", "house-plan-uwkew")).strip()
    VERSION = env.get("ROBOFLOW_VERSION", os.getenv("ROBOFLOW_VERSION", "5")).strip()
    
    # Configurare parametri
    raw_conf = env.get("ROBOFLOW_CONFIDENCE", "40")
    CONFIDENCE_GLOBAL = float(raw_conf) / 100.0 if float(raw_conf) > 1.0 else float(raw_conf)
    
    # Prag mic pentru ferestre ca să prindem tot (se filtrează ulterior)
    CONFIDENCE_WINDOW = 0.15  

    # --- CONFIGURARE SLICING ---
    SLICING_THRESHOLD_PX = 3000  # Doar dacă e mai mare de 3000px
    SLICE_SIZE = 2000            # Slice-uri de 2000px
    OVERLAP_PERCENT = 0.15       # 15% Overlap
    
    # Calculăm overlap-ul în pixeli
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

    # DECIDEM STRATEGIA: Doar dacă depășește pragul pe oricare axă
    use_slicing = (w_img > SLICING_THRESHOLD_PX) or (h_img > SLICING_THRESHOLD_PX)
    
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=API_KEY
    )
    
    detections = None
    start = time.time()
    
    # Debug helpers
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    try:
        slice_counter = 0

        # Funcție comună pentru desenat debug pe slices (dacă se folosește slicing)
        def debug_callback(image_slice: np.ndarray, slice_dets: sv.Detections) -> None:
            nonlocal slice_counter
            slice_counter += 1
            labels = []
            if slice_dets.class_id is not None:
                # Obținem numele claselor dacă sunt disponibile
                cls_names = []
                if slice_dets.data and 'class_name' in slice_dets.data:
                    cls_names = slice_dets.data['class_name']
                else:
                    cls_names = [str(i) for i in slice_dets.class_id]

                for i, cls_name in enumerate(cls_names):
                    conf = slice_dets.confidence[i]
                    labels.append(f"{cls_name} {conf:.2f}")

            annotated = box_annotator.annotate(scene=image_slice.copy(), detections=slice_dets)
            annotated = label_annotator.annotate(scene=annotated, detections=slice_dets, labels=labels)
            cv2.imwrite(str(slices_dir / f"slice_{slice_counter}.jpg"), annotated)

        if use_slicing:
            print(f"  ✂️  Mode: SLICING ACTIVE (Imagine > {SLICING_THRESHOLD_PX}px)")
            print(f"      Size: {SLICE_SIZE}x{SLICE_SIZE}, Overlap: {overlap_px_val}px ({int(OVERLAP_PERCENT*100)}%)")
            
            def callback(image_slice: np.ndarray) -> sv.Detections:
                for _ in range(3):
                    try:
                        res = client.infer(image_slice, model_id=f"{PROJECT}/{VERSION}")
                        dets = sv.Detections.from_inference(res)
                        # Salvăm debug slice
                        debug_callback(image_slice, dets)
                        return dets
                    except Exception as e:
                        time.sleep(0.5)
                return sv.Detections.empty()

            slicer = sv.InferenceSlicer(
                callback=callback,
                slice_wh=SLICE_WH,
                overlap_wh=OVERLAP_PX,
                iou_threshold=0.5,
                thread_workers=1
            )
            detections = slicer(image)
            
        else:
            print(f"  📸 Mode: FULL INFERENCE (Imagine < {SLICING_THRESHOLD_PX}px)")
            # Logica Standard
            result = client.infer(image, model_id=f"{PROJECT}/{VERSION}")
            detections = sv.Detections.from_inference(result)

    except Exception as e:
        print(f"❌ Eroare Inferență: {e}")
        return False, []

    elapsed = time.time() - start

    # 6. Conversie rezultat final + Filtrare Flexibilă
    final_predictions = []
    
    # Desenăm rezultatul pe imaginea full pentru debug final
    debug_img = image.copy()
    
    if detections:
        keep_mask = []
        labels_debug = []
        
        # Extragem clasele
        class_names = []
        if detections.data and 'class_name' in detections.data:
            class_names = detections.data['class_name']
        elif detections.class_id is not None:
            class_names = [str(i) for i in detections.class_id]
            
        for i, cls_raw in enumerate(class_names):
            cls = str(cls_raw).lower()
            conf = detections.confidence[i]
            
            # FILTRARE FLEXIBILĂ (conține cuvântul)
            is_window = "window" in cls
            is_door = "door" in cls
            
            is_accepted = False
            if is_window and conf >= CONFIDENCE_WINDOW: is_accepted = True
            elif is_door and conf >= CONFIDENCE_GLOBAL: is_accepted = True
            elif conf >= CONFIDENCE_GLOBAL: is_accepted = True
            
            keep_mask.append(is_accepted)
            
            # Adăugăm în lista finală doar ce e acceptat
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
                
                # Salvează crop debug
                try:
                    # Asigură coordonate valide
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

    # Salvează imaginea de ansamblu cu detecții
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