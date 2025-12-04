import os
import cv2
import supervision as sv
import numpy as np
import requests
from dotenv import load_dotenv

# 1. Configurare
load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY", "PUNE_CHEIA_AICI_DACA_NU_E_IN_ENV")
PROJECT = os.getenv("ROBOFLOW_PROJECT", "house-plan-uwkew")
VERSION = os.getenv("ROBOFLOW_VERSION", "5")
IMAGE_PATH = "./debug.png" 

# Parametri relaxați
CONFIDENCE = 0.05       # 5%
SLICE_SIZE = 2000
OVERLAP_RATIO = 0.2     
OVERLAP_PX = int(SLICE_SIZE * OVERLAP_RATIO) 

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Nu găsesc: {IMAGE_PATH}")
        return

    print(f"🚀 Pornesc diagnosticare (V4 - Manual Parsing) pentru: {IMAGE_PATH}")
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("❌ Eroare: Nu pot citi imaginea.")
        return
    
    # URL
    infer_url = f"https://detect.roboflow.com/{PROJECT}/{VERSION}"

    # Callback care folosește REQUESTS + Manual Parsing
    def callback(image_slice: np.ndarray) -> sv.Detections:
        _, img_encoded = cv2.imencode('.jpg', image_slice)
        img_bytes = img_encoded.tobytes()
        
        params = {
            "api_key": API_KEY,
            "confidence": CONFIDENCE * 100, # 5
            "overlap": 50 
        }
        
        files = {
            'file': ('slice.jpg', img_bytes, 'image/jpeg')
        }
        
        try:
            resp = requests.post(infer_url, params=params, files=files)
            if resp.status_code != 200:
                print(f"⚠️ API Error ({resp.status_code}): {resp.text}")
                return sv.Detections.empty()
            
            # --- PARSARE MANUALĂ AICI ---
            result = resp.json()
            predictions = result.get("predictions", [])
            
            if not predictions:
                return sv.Detections.empty()

            xyxy = []
            confidences = []
            class_names = []
            
            for pred in predictions:
                # Roboflow dă x,y (centru) și width, height
                x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                
                xyxy.append([x1, y1, x2, y2])
                confidences.append(pred['confidence'])
                class_names.append(pred['class'])

            # Convertim în numpy arrays pentru Supervision
            xyxy = np.array(xyxy)
            confidence = np.array(confidences)
            
            # Generăm class_ids (mapăm string-urile la int-uri)
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
            print(f"Eroare parsing/requests: {e}")
            return sv.Detections.empty()

    # Slicer
    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(SLICE_SIZE, SLICE_SIZE),
        overlap_ratio_wh=None,
        overlap_wh=(OVERLAP_PX, OVERLAP_PX),
        iou_threshold=0.5,
        thread_workers=2
    )

    print("⏳ Trimit datele...")
    detections = slicer(image)
    print(f"✅ Gata! Roboflow a returnat {len(detections)} obiecte.")

    # Desenare
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5)
    
    labels = []
    # Folosim datele pe care tocmai le-am pus manual
    if detections.class_id is not None and 'class_name' in detections.data:
        for i in range(len(detections.class_id)):
            conf = detections.confidence[i]
            name = detections.data['class_name'][i]
            labels.append(f"{name} {conf:.2f}")

    annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    cv2.imwrite("debug_output_fixed.jpg", annotated)
    print(f"💾 Imagine salvată: debug_output_fixed.jpg")

if __name__ == "__main__":
    main()