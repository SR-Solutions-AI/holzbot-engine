# file: engine/detections/roboflow_import.py
from __future__ import annotations

import json
import os
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

# Încercăm să importăm bibliotecile necesare
try:
    from inference_sdk import InferenceHTTPClient
    import supervision as sv
except ImportError:
    InferenceHTTPClient = None
    sv = None

def run_roboflow_import(env: Dict[str, str], work_dir: Path) -> Tuple[bool, str]:
    """
    Importă detecții de la Roboflow folosind 'supervision.InferenceSlicer'
    pentru a gestiona imagini mari prin tehnica de Slicing (tăiere).
    """
    # 1. Verificări preliminare
    if InferenceHTTPClient is None or sv is None:
        return False, "Lipsesc bibliotecile 'inference-sdk' sau 'supervision'. Verifică requirements.txt"

    API_KEY = env.get("ROBOFLOW_API_KEY", os.getenv("ROBOFLOW_API_KEY", "")).strip()
    PROJECT = env.get("ROBOFLOW_PROJECT", os.getenv("ROBOFLOW_PROJECT", "house-plan-uwkew")).strip()
    VERSION = env.get("ROBOFLOW_VERSION", os.getenv("ROBOFLOW_VERSION", "5")).strip()
    
    # Configurare parametri
    raw_conf = env.get("ROBOFLOW_CONFIDENCE", "40")
    CONFIDENCE_GLOBAL = float(raw_conf) / 100.0 if float(raw_conf) > 1.0 else float(raw_conf)
    
    # 🟢 AICI SETĂM PRAGUL PENTRU FERESTRE (mai mic)
    CONFIDENCE_WINDOW = 0.15  
    
    OVERLAP_RATIO = 0.25
    SLICE_WH = (1280, 1280)

    if not API_KEY:
        return False, "ROBOFLOW_API_KEY lipsește din environment"

    plan_jpg = work_dir / "plan.jpg"
    if not plan_jpg.exists():
        return False, f"Nu găsesc plan.jpg în {work_dir}"

    print(f"  🔍 Roboflow Slicing: {plan_jpg.name} (Global Conf: {CONFIDENCE_GLOBAL}, Window Conf: {CONFIDENCE_WINDOW})")
    start = time.time()

    try:
        # 2. Inițializare Client
        client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=API_KEY
        )

        # 3. Definire funcție de callback cu FILTRARE HIBRIDĂ
        def callback(image_slice: np.ndarray) -> sv.Detections:
            # Facem inferența fără threshold (luăm tot ce vede modelul)
            # sau folosim un threshold foarte mic pentru a filtra noi manual apoi
            result = client.infer(
                image_slice,
                model_id=f"{PROJECT}/{VERSION}"
            )
            
            # Conversie rezultat JSON -> sv.Detections
            detections = sv.Detections.from_inference(result)
            
            # 🟢 LOGICĂ DE FILTRARE PE CLASE
            if detections.data is not None and 'class_name' in detections.data:
                class_names = detections.data['class_name']
                
                # Identificăm care detecții sunt ferestre
                is_window = (class_names == 'window') | (class_names == 'double-window')
                
                # Regula 1: Dacă e fereastră, folosim CONFIDENCE_WINDOW (ex: 0.25)
                keep_windows = is_window & (detections.confidence >= CONFIDENCE_WINDOW)
                
                # Regula 2: Dacă NU e fereastră, folosim CONFIDENCE_GLOBAL (ex: 0.40)
                keep_others = ~is_window & (detections.confidence >= CONFIDENCE_GLOBAL)
                
                # Păstrăm reuniunea celor două reguli
                return detections[keep_windows | keep_others]
            else:
                # Fallback dacă nu avem nume de clase (puțin probabil)
                return detections[detections.confidence >= CONFIDENCE_GLOBAL]

        # 4. Citire imagine cu OpenCV
        image = cv2.imread(str(plan_jpg))
        if image is None:
            return False, "Nu am putut citi imaginea cu OpenCV."

        # 5. Rulare Slicer
        slicer = sv.InferenceSlicer(
            callback=callback,
            slice_wh=SLICE_WH,
            overlap_ratio_wh=(OVERLAP_RATIO, OVERLAP_RATIO),
            iou_threshold=0.5,
            thread_workers=2
        )
        
        detections = slicer(image)

    except Exception as e:
        return False, f"Eroare Supervision Slicer: {e}"

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

    print(f"  ✅ {len(final_predictions)} detecții (Slicing) în {elapsed:.2f}s")

    # 7. Salvare
    final_json = {
        "predictions": final_predictions,
        "image": {
            "width": image.shape[1],
            "height": image.shape[0]
        }
    }

    detections_dir = work_dir / "export_objects"
    detections_dir.mkdir(parents=True, exist_ok=True)
    detections_file = detections_dir / "detections.json"

    detections_file.write_text(
        json.dumps(final_json, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    return True, f"{len(final_predictions)} detecții salvate."