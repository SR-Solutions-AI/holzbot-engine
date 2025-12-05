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

# Încercăm să importăm bibliotecile necesare
try:
    from inference_sdk import InferenceHTTPClient
    import supervision as sv
except ImportError:
    InferenceHTTPClient = None
    sv = None

# ==========================================
# 1. HELPER: TEMPLATE MATCHING (VERIFICARE GEOMETRICĂ)
# ==========================================

def verify_with_template_matching(crop: np.ndarray, class_name: str, templates_dir: Path) -> bool:
    """
    Verifică dacă crop-ul seamănă cu vreun template cunoscut.
    Dacă nu avem template-uri, folosim o heuristică geometrică simplă.
    """
    # 1. Heuristica Geometrică (Fallback rapid)
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    if "window" in class_name:
        # Ferestrele au multe linii paralele
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=int(min(h,w)*0.5), maxLineGap=10)
        if lines is not None and len(lines) >= 2:
            return True # Are linii lungi, probabil e geam
            
    if "door" in class_name:
        # Ușile au de obicei un arc de cerc sau linii perpendiculare
        # Aici e mai greu geometric, ne bazăm pe AI sau pe densitatea pixelilor
        non_zero = cv2.countNonZero(edges)
        density = non_zero / (h * w)
        if density > 0.05: # Are suficient detaliu
            return True

    # 2. Template Matching Real (Dacă există folderul templates)
    # Se așteaptă ca templates_dir să aibă fișiere gen 'window_1.jpg', 'door_1.jpg'
    if templates_dir.exists():
        best_score = 0.0
        class_templates = list(templates_dir.glob(f"{class_name}*.jpg")) + list(templates_dir.glob(f"{class_name}*.png"))
        
        for tmpl_path in class_templates:
            template = cv2.imread(str(tmpl_path), 0)
            if template is None: continue
            
            # Resize template să fie cam cât crop-ul
            if template.shape[0] > h or template.shape[1] > w:
                template = cv2.resize(template, (w, h))

            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            score = np.max(res)
            if score > best_score:
                best_score = score
        
        if best_score > 0.6: # Prag de similaritate
            return True

    return False # Dacă nici geometria nici template-ul nu confirmă

# ==========================================
# 2. HELPER: AI VERIFICATION (GPT-4o)
# ==========================================

def verify_with_ai(crop: np.ndarray, class_name: str) -> bool:
    """Întreabă GPT-4o dacă în imagine este chiar obiectul respectiv."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: return True # Fail Open: Dacă nu avem AI, păstrăm detecția

    try:
        _, buffer = cv2.imencode('.jpg', crop)
        b64_img = base64.b64encode(buffer).decode('utf-8')

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        
        prompt = f"""Look at this image crop from a floor plan.
        I detected a potential object of type: "{class_name}".
        
        Task: Verify if this looks like a valid {class_name}.
        - Windows: Should look like parallel lines within a wall.
        - Doors: Should look like a swing arc or opening.
        - Noise: Random lines, text, or furniture.
        
        Return ONLY JSON: {{ "valid": boolean }}"""

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "low"}}
                ]}
            ],
            "max_tokens": 20,
            "response_format": { "type": "json_object" }
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=5)
        result = response.json()
        content = json.loads(result['choices'][0]['message']['content'])
        
        is_valid = content.get("valid", False)
        print(f"    🤖 AI Check ({class_name}): {'✅ Valid' if is_valid else '❌ Invalid'}")
        return is_valid

    except Exception as e:
        print(f"    ⚠️ AI Error: {e}")
        return True # Păstrăm în caz de eroare

# ==========================================
# 3. HELPER: CONFLICT RESOLUTION (Ușă vs Geam)
# ==========================================

def calculate_iou(box1, box2):
    x1 = max(box1['x'] - box1['width']/2, box2['x'] - box2['width']/2)
    y1 = max(box1['y'] - box1['height']/2, box2['y'] - box2['height']/2)
    x2 = min(box1['x'] + box1['width']/2, box2['x'] + box2['width']/2)
    y2 = min(box1['y'] + box1['height']/2, box2['y'] + box2['height']/2)
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0: return 0
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    return inter_area / (area1 + area2 - inter_area)

def resolve_conflicts_with_ai(crop, class_a, class_b):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: return None
    try:
        _, buffer = cv2.imencode('.jpg', crop)
        b64_img = base64.b64encode(buffer).decode('utf-8')
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        prompt = f"Is this architectural detail a '{class_a}' or a '{class_b}'? Return ONLY the class name."
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}]}],
            "max_tokens": 10
        }
        res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=8).json()
        dec = res['choices'][0]['message']['content'].strip().lower()
        if class_a in dec: return class_a
        if class_b in dec: return class_b
    except: pass
    return None

# ==========================================
# LOGICA PRINCIPALĂ
# ==========================================

def run_roboflow_import(env: Dict[str, str], work_dir: Path) -> Tuple[bool, str]:
    if InferenceHTTPClient is None or sv is None:
        return False, "Lipsesc bibliotecile 'inference-sdk' sau 'supervision'."

    API_KEY = env.get("ROBOFLOW_API_KEY", os.getenv("ROBOFLOW_API_KEY", "")).strip()
    PROJECT = env.get("ROBOFLOW_PROJECT", os.getenv("ROBOFLOW_PROJECT", "house-plan-uwkew")).strip()
    VERSION = env.get("ROBOFLOW_VERSION", os.getenv("ROBOFLOW_VERSION", "5")).strip()
    
    # 🟢 1. CONFIGURARE CONFIDENCE
    CONFIDENCE_GLOBAL = 0.30  # Pragul pentru acceptare directă (sigur)
    CONFIDENCE_LOW = 0.15     # Pragul de intrare în VERIFICARE
    
    OVERLAP_RATIO = 0.25
    SLICE_WH = (1280, 1280)

    if not API_KEY: return False, "ROBOFLOW_API_KEY lipsește."
    plan_jpg = work_dir / "plan.jpg"
    if not plan_jpg.exists(): return False, f"Nu găsesc plan.jpg în {work_dir}"

    print(f"  🔍 Roboflow Slicing: {plan_jpg.name}")
    print(f"     -> Global Conf: {CONFIDENCE_GLOBAL} (Acceptare automată)")
    print(f"     -> Low Conf: {CONFIDENCE_LOW} (Verificare Template/AI)")

    start = time.time()
    try:
        client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=API_KEY)
        image = cv2.imread(str(plan_jpg))
        
        # Folder pentru template-uri (opțional)
        templates_dir = work_dir.parent.parent / "templates" # Ajustează calea dacă e nevoie

        def callback(image_slice: np.ndarray) -> sv.Detections:
            result = client.infer(image_slice, model_id=f"{PROJECT}/{VERSION}")
            return sv.Detections.from_inference(result)

        slicer = sv.InferenceSlicer(
            callback=callback,
            slice_wh=SLICE_WH,
            overlap_ratio_wh=(OVERLAP_RATIO, OVERLAP_RATIO),
            iou_threshold=0.5,
            thread_workers=2
        )
        detections = slicer(image)

    except Exception as e:
        return False, f"Eroare Slicer: {e}"

    # 🟢 2. POST-PROCESARE: Filtrare și Verificare
    final_predictions = []
    h_img, w_img = image.shape[:2]

    # Lista temporară pentru rezolvare conflicte
    temp_preds = []

    print(f"  📊 Analizez {len(detections)} candidați...")

    for i in range(len(detections)):
        bbox = detections.xyxy[i]
        conf = float(detections.confidence[i])
        cls_id = detections.class_id[i]
        cls_name = detections.data['class_name'][i] if detections.data is not None else str(cls_id)
        
        # Ignorăm tot ce e sub pragul minim absolut (0.15)
        if conf < CONFIDENCE_LOW:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        width, height = x2 - x1, y2 - y1
        cx, cy = x1 + width / 2, y1 + height / 2
        
        obj_data = {
            "x": float(cx), "y": float(cy),
            "width": float(width), "height": float(height),
            "class": cls_name, "class_id": int(cls_id),
            "confidence": conf
        }

        # A. Dacă scorul e mare (> 30%), acceptăm direct
        if conf >= CONFIDENCE_GLOBAL:
            temp_preds.append(obj_data)
            continue
            
        # B. Dacă scorul e mic (15% - 30%), VERIFICĂM
        print(f"  🧐 Verificare Low-Confidence ({conf:.2f}) pentru {cls_name}...")
        
        # Crop pentru analiză
        crop = image[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
        if crop.size == 0: continue

        # Pas 1: Template / Geometrie
        is_geo_valid = verify_with_template_matching(crop, cls_name, templates_dir)
        
        # Pas 2: AI (doar dacă geometria nu e sigură, sau putem cere ambele)
        # Optimizare: Dacă geometria zice DA, nu mai consumăm tokeni AI
        if is_geo_valid:
            print(f"     ✅ Confirmat prin Geometrie/Template.")
            temp_preds.append(obj_data)
        else:
            # Fallback la AI
            is_ai_valid = verify_with_ai(crop, cls_name)
            if is_ai_valid:
                print(f"     ✅ Confirmat prin AI.")
                temp_preds.append(obj_data)
            else:
                print(f"     ❌ Respins (nici Geometrie, nici AI).")

    # 🟢 3. REZOLVARE CONFLICTE (Ușă vs Geam)
    # Acum lucrăm doar cu lista 'temp_preds' care conține elemente verificate
    keep_indices = set(range(len(temp_preds)))
    
    for i in range(len(temp_preds)):
        if i not in keep_indices: continue
        for j in range(i + 1, len(temp_preds)):
            if j not in keep_indices: continue
            
            p1, p2 = temp_preds[i], temp_preds[j]
            
            # Verificăm dacă e conflict Door <-> Window
            classes = {p1['class'], p2['class']}
            if ("door" in p1['class'] or "door" in p2['class']) and \
               ("window" in p1['class'] or "window" in p2['class']):
               
               if calculate_iou(p1, p2) > 0.6:
                   print(f"  ⚔️ Conflict: {p1['class']} vs {p2['class']}")
                   # Crop zona comună
                   x1 = int(min(p1['x'], p2['x']) - max(p1['width'], p2['width'])/2)
                   y1 = int(min(p1['y'], p2['y']) - max(p1['height'], p2['height'])/2)
                   x2 = int(max(p1['x'], p2['x']) + max(p1['width'], p2['width'])/2)
                   y2 = int(max(p1['y'], p2['y']) + max(p1['height'], p2['height'])/2)
                   crop = image[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
                   
                   winner = resolve_conflicts_with_ai(crop, p1['class'], p2['class'])
                   if winner == p1['class']: keep_indices.remove(j)
                   elif winner == p2['class']: keep_indices.remove(i)
                   else: # Fallback scor
                       if p1['confidence'] > p2['confidence']: keep_indices.remove(j)
                       else: keep_indices.remove(i)

    final_predictions = [temp_preds[i] for i in sorted(list(keep_indices))]
    elapsed = time.time() - start
    print(f"  ✅ {len(final_predictions)} detecții finale salvate în {elapsed:.2f}s")

    # 7. Salvare
    final_json = {
        "predictions": final_predictions,
        "image": {"width": w_img, "height": h_img}
    }
    detections_dir = work_dir / "export_objects"
    detections_dir.mkdir(parents=True, exist_ok=True)
    (detections_dir / "detections.json").write_text(json.dumps(final_json, indent=2), encoding="utf-8")

    return True, f"{len(final_predictions)} detecții salvate."