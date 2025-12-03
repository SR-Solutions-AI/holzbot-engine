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
    print("🕵️  DEBUG MODE ACTIVAT: SLICING & CROPS")
    print("Se vor salva poze în folderul 'debug_roboflow'")
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
    
    # IMPORTANT: Setăm praguri mai mici aici pentru a lăsa 'detector.py' să decidă via Template/Gemini
    # Vrem să "prindem" tot ce ar putea fi fereastră.
    CONFIDENCE_WINDOW = 0.15  
    SLICE_WH = (1280, 1280)
    OVERLAP_PX = (int(SLICE_WH[0] * 0.25), int(SLICE_WH[1] * 0.25))

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
        try:
            shutil.rmtree(debug_dir)
        except Exception:
            pass
    
    slices_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    # ---------------------------

    print(f"  🔍 Roboflow Slicing: {plan_jpg.name}")
    print(f"     Global Conf: {CONFIDENCE_GLOBAL}, Window Conf: {CONFIDENCE_WINDOW}")
    
    start = time.time()
    
    # Debug helpers
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    try:
        # 2. Inițializare Client
        client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=API_KEY
        )

        slice_counter = 0

        # 3. Definire funcție de callback
        def callback(image_slice: np.ndarray) -> sv.Detections:
            nonlocal slice_counter
            slice_counter += 1
            
            max_retries = 3
            last_err = None
            detections = None
            
            # A. Retry Logic
            for attempt in range(max_retries):
                try:
                    result = client.infer(image_slice, model_id=f"{PROJECT}/{VERSION}")
                    detections = sv.Detections.from_inference(result)
                    break
                except Exception as e:
                    last_err = e
                    time.sleep((attempt + 1) * 1.5)
            
            if detections is None:
                print(f"    ⚠️ Slice {slice_counter} failed: {last_err}")
                raise last_err

            # B. Procesare și Salvare Debug
            status_labels = []
            
            # Facem o copie pentru debug
            debug_slice = image_slice.copy()
            h_img, w_img = image_slice.shape[:2]

            if detections.data is not None and 'class_name' in detections.data:
                class_names = detections.data['class_name']
                
                for i, cls in enumerate(class_names):
                    conf = detections.confidence[i]
                    bbox = detections.xyxy[i].astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    # Decidem culoarea (doar vizual, aici păstrăm tot ce vine de la API)
                    # Verde daca e peste prag, Rosu daca e sub (dar tot il pastram pentru downstream)
                    is_strong = conf >= CONFIDENCE_WINDOW
                    color = (0, 255, 0) if is_strong else (0, 0, 255)
                    
                    status_text = "STRONG" if is_strong else "WEAK"
                    status_labels.append(f"{cls} {conf:.2f}")

                    # --- SALVARE CROP (Zoom In) ---
                    # Asigurăm coordonate valide
                    x1_c, y1_c = max(0, x1), max(0, y1)
                    x2_c, y2_c = min(w_img, x2), min(h_img, y2)
                    
                    if x2_c > x1_c and y2_c > y1_c:
                        crop = image_slice[y1_c:y2_c, x1_c:x2_c].copy()
                        
                        # Adăugăm bordură colorată
                        crop = cv2.copyMakeBorder(crop, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=color)
                        
                        # Nume fișier
                        safe_cls = cls.replace(" ", "_")
                        filename = f"{status_text}_{safe_cls}_{conf:.2f}_s{slice_counter}_id{i}.jpg"
                        cv2.imwrite(str(crops_dir / filename), crop)

            # Salvăm Slice-ul desenat
            annotated_slice = box_annotator.annotate(scene=debug_slice, detections=detections)
            annotated_slice = label_annotator.annotate(scene=annotated_slice, detections=detections, labels=status_labels)
            cv2.imwrite(str(slices_dir / f"slice_{slice_counter}.jpg"), annotated_slice)

            # C. Returnăm TOATE detecțiile găsite de Roboflow
            # Nu filtram aici drastic, lăsăm 'detector.py' să facă validarea fină
            return detections

        # 4. Citire imagine
        image = cv2.imread(str(plan_jpg))
        if image is None:
            return False, []

        # 5. Rulare Slicer
        slicer = sv.InferenceSlicer(
            callback=callback,
            slice_wh=SLICE_WH,
            overlap_wh=OVERLAP_PX, 
            overlap_ratio_wh=None,
            iou_threshold=0.5,
            thread_workers=1 # Un singur worker pentru a nu bloca salvarea debug
        )
        
        detections = slicer(image)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, []

    elapsed = time.time() - start

    # 6. Conversie rezultat
    final_predictions = []
    
    for i in range(len(detections)):
        bbox = detections.xyxy[i]
        conf = detections.confidence[i]
        cls_id = detections.class_id[i]
        
        cls_name = str(cls_id)
        if detections.data is not None and 'class_name' in detections.data:
             cls_name = detections.data['class_name'][i]

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        x = x1 + width / 2
        y = y1 + height / 2

        final_predictions.append({
            "x": float(x),
            "y": float(y),
            "width": float(width),
            "height": float(height),
            "class": cls_name,
            "class_id": int(cls_id),
            "confidence": float(conf)
        })

    print(f"  ✅ {len(final_predictions)} detecții brute (Slicing) în {elapsed:.2f}s")
    
    # 7. Salvare JSON (pentru referință)
    final_json = {
        "predictions": final_predictions,
        "image": {"width": image.shape[1], "height": image.shape[0]}
    }
    
    detections_dir = work_dir / "export_objects"
    detections_dir.mkdir(parents=True, exist_ok=True)
    (detections_dir / "detections.json").write_text(json.dumps(final_json), encoding="utf-8")

    # RETURNĂM LISTA, nu doar un string
    return True, final_predictions