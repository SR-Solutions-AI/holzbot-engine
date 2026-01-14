# file: engine/detections/roboflow_import.py
from __future__ import annotations

import json
import os
import time
import cv2
import numpy as np
import base64
import requests
from pathlib import Path
from typing import Tuple, Dict, List
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from inference_sdk import InferenceHTTPClient
    import supervision as sv
except ImportError:
    InferenceHTTPClient = None
    sv = None

# ==========================================
# MAIN DETECTION PIPELINE
# ==========================================

def run_roboflow_import(env: Dict[str, str], work_dir: Path) -> Tuple[bool, str]:
    """
    Pipeline principal de detecție:
    1. Face Slicing Inference pentru a găsi obiecte mici.
    2. Salvează rezultatele brute în 'export_objects/detections.json'.
    3. Generează imagini de debug.
    """
    if InferenceHTTPClient is None or sv is None:
        return False, "Lipsesc bibliotecile 'inference-sdk' sau 'supervision'."
    
    # Configurare
    API_KEY = env.get("ROBOFLOW_API_KEY", os.getenv("ROBOFLOW_API_KEY", "")).strip()
    PROJECT = env.get("ROBOFLOW_PROJECT", os.getenv("ROBOFLOW_PROJECT", "house-plan-uwkew")).strip()
    VERSION = env.get("ROBOFLOW_VERSION", os.getenv("ROBOFLOW_VERSION", "5")).strip()
    
    # Praguri permisive pentru a colecta toți candidații
    CONFIDENCE_THRESHOLD = 0.10 
    SLICE_WH = (1280, 1280)
    OVERLAP_RATIO = 0.25
    
    if not API_KEY:
        return False, "ROBOFLOW_API_KEY lipsește."
    
    plan_jpg = work_dir / "plan.jpg"
    if not plan_jpg.exists():
        return False, f"Nu găsesc plan.jpg în {work_dir}"
    
    print(f"\n{'='*80}")
    print(f"  🔍 ROBOFLOW DETECTION (SLICING GENERATOR)")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # ==========================================
        # STEP 1: ROBOFLOW INFERENCE (SLICING)
        # ==========================================
        print("📡 STEP 1: Running Roboflow Inference (Slicing)...")
        
        client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com", 
            api_key=API_KEY
        )
        image = cv2.imread(str(plan_jpg))
        h_img, w_img = image.shape[:2]
        
        def callback(image_slice: np.ndarray) -> sv.Detections:
            result = client.infer(image_slice, model_id=f"{PROJECT}/{VERSION}")
            return sv.Detections.from_inference(result)
        
        slicer = sv.InferenceSlicer(
            callback=callback,
            slice_wh=SLICE_WH,
            iou_threshold=0.5,
            thread_workers=2
        )
        
        detections = slicer(image)
        print(f"  ✅ Received {len(detections)} raw detections")
        
        # ==========================================
        # STEP 2: FORMAT DATA
        # ==========================================
        export_predictions = []
        
        for i in range(len(detections)):
            # Extragem datele din supervision
            bbox = detections.xyxy[i]
            conf = float(detections.confidence[i])
            cls_id = detections.class_id[i]
            cls_name = detections.data['class_name'][i] if detections.data is not None else str(cls_id)
            
            # Filtrăm zgomotul extrem
            if conf < CONFIDENCE_THRESHOLD:
                continue
                
            x1, y1, x2, y2 = map(float, bbox)
            
            # Convertim la formatul standard (cx, cy, w, h)
            width = x2 - x1
            height = y2 - y1
            cx = x1 + width / 2
            cy = y1 + height / 2
            
            export_predictions.append({
                "x": cx,
                "y": cy,
                "width": width,
                "height": height,
                "class": cls_name,
                "class_id": int(cls_id),
                "confidence": conf
            })

        # ==========================================
        # STEP 3: SAVE TO DISK (CRITIC PENTRU ORCHESTRATOR)
        # ==========================================
        output_dir = work_dir / "detection_output"
        output_dir.mkdir(exist_ok=True)
        
        export_objects_dir = work_dir / "export_objects"
        export_objects_dir.mkdir(parents=True, exist_ok=True)
        
        final_json = {
            "predictions": export_predictions,
            "image": {"width": w_img, "height": h_img},
            "metadata": {
                "processing_time": time.time() - start_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        json_path = export_objects_dir / "detections.json"
        json_path.write_text(
            json.dumps(final_json, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        
        print(f"  💾 Saved {len(export_predictions)} candidates to {json_path}")
        
        # Desenăm și o imagine de debug rapid
        viz_img = image.copy()
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        labels = [
            f"{detections.data['class_name'][i]} {detections.confidence[i]:.2f}"
            for i in range(len(detections))
        ]
        
        viz_img = box_annotator.annotate(scene=viz_img, detections=detections)
        viz_img = label_annotator.annotate(scene=viz_img, detections=detections, labels=labels)
        
        cv2.imwrite(str(output_dir / "detections_all.jpg"), viz_img)
        
        return True, f"Saved {len(export_predictions)} detections to disk."
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Eroare: {e}"