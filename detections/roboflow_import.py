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
    # --- ADAUGĂ ACESTE 3 LINII PENTRU DEBUG ---
    print("\n\n" + "="*50)
    print("🔥🔥🔥 DIAGNOSTIC VERSIUNE FIȘIER 🔥🔥🔥")
    print("Dacă vezi acest mesaj, fișierul a fost actualizat corect.")
    print("="*50 + "\n\n")
    # ------------------------------------------
    """
    Importă detecții de la Roboflow folosind 'supervision.InferenceSlicer'.
    Include Retry Logic pentru conexiuni instabile.
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
    
    # Parametri specifici ferestre/slicing
    CONFIDENCE_WINDOW = 0.25  
    SLICE_WH = (1280, 1280)
    
    # Calculăm Overlap în pixeli (fix pentru warning-ul overlap_ratio_wh deprecated)
    # 0.25 * 1280 = 320 pixeli
    OVERLAP_PX = (int(SLICE_WH[0] * 0.25), int(SLICE_WH[1] * 0.25))

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

        # 3. Definire funcție de callback cu RETRY LOGIC și FILTRARE HIBRIDĂ
        def callback(image_slice: np.ndarray) -> sv.Detections:
            max_retries = 3
            last_err = None
            
            for attempt in range(max_retries):
                try:
                    # A. Inferența (Network Call)
                    result = client.infer(
                        image_slice,
                        model_id=f"{PROJECT}/{VERSION}"
                    )
                    
                    # B. Procesare rezultat
                    detections = sv.Detections.from_inference(result)
                    
                    # 🟢 LOGICĂ DE FILTRARE PE CLASE
                    if detections.data is not None and 'class_name' in detections.data:
                        class_names = detections.data['class_name']
                        
                        # Identificăm care detecții sunt ferestre
                        is_window = (class_names == 'window') | (class_names == 'double-window')
                        
                        # Regula 1: Ferestre -> CONFIDENCE_WINDOW
                        keep_windows = is_window & (detections.confidence >= CONFIDENCE_WINDOW)
                        
                        # Regula 2: Altele -> CONFIDENCE_GLOBAL
                        keep_others = ~is_window & (detections.confidence >= CONFIDENCE_GLOBAL)
                        
                        return detections[keep_windows | keep_others]
                    else:
                        return detections[detections.confidence >= CONFIDENCE_GLOBAL]

                except Exception as e:
                    # Dacă e eroare de rețea, așteptăm și reîncercăm
                    last_err = e
                    wait_time = (attempt + 1) * 2 # 2s, 4s, 6s
                    # Nu printăm pentru orice eroare mică, doar la final dacă eșuează tot
                    time.sleep(wait_time)
            
            # Dacă a eșuat de 3 ori, aruncăm eroarea mai departe
            print(f"    ⚠️ Slice failed after {max_retries} retries: {last_err}")
            raise last_err

        # 4. Citire imagine cu OpenCV
        image = cv2.imread(str(plan_jpg))
        if image is None:
            return False, "Nu am putut citi imaginea cu OpenCV."

        # 5. Rulare Slicer
        # Folosim overlap_wh în loc de overlap_ratio_wh pentru a scăpa de warning
        # Reducem thread_workers la 1 sau 2 pentru a nu satura rețeaua când rulăm 4 planuri deodată
        slicer = sv.InferenceSlicer(
            callback=callback,
            slice_wh=SLICE_WH,
            overlap_wh=OVERLAP_PX, 
            overlap_ratio_wh=None,
            iou_threshold=0.5,
            thread_workers=1  # IMPORTANT: Redus pentru stabilitate rețea (era default mai mare)
        )
        
        detections = slicer(image)

    except Exception as e:
        return False, f"Eroare Supervision Slicer (Critical): {e}"

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