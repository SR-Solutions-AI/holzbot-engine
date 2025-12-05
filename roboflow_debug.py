import os
import cv2
import requests
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Configurare
load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY", "PUNE_CHEIA_AICI_DACA_NU_E_IN_ENV")
PROJECT = os.getenv("ROBOFLOW_PROJECT", "house-plan-uwkew")
VERSION = os.getenv("ROBOFLOW_VERSION", "5")
IMAGE_PATH = "./debug3.png" 

# Parametri
TARGET_SIZE = 600        
CONFIDENCE = 0.15       
OVERLAP = 50
SPLIT_THRESHOLD = 1200   

# Clase Canonice
GLOBAL_CLASS_MAPPING = {
    "door": 0, "double_door": 1, "window": 2, "double_window": 3,
    "sliding_door": 4, "sliding_window": 5, "stairs": 6,
}

def normalize_class_name(raw_name: str) -> str | None:
    s = raw_name.lower().strip()
    if "double" in s:
        if "window" in s: return "double_window"
        if "door" in s: return "double_door"
    if "sliding" in s:
        if "window" in s: return "sliding_window"
        if "door" in s: return "sliding_door"
    if "window" in s: return "window"
    if "door" in s: return "door"
    if "stair" in s: return "stairs"
    return None

def resize_to_target(image, target_size):
    """Redimensionează imaginea la target_size păstrând aspect ratio"""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
    img_resized = np.array(pil_resized)
    return cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR), scale

def split_image(image):
    h, w = image.shape[:2]
    crops = []
    
    if w > h:
        mid_x = w // 2
        print(f"  ✂️  Taie vertical la x={mid_x}")
        crop_left = image[:, :mid_x]
        crops.append((crop_left, 0, 0))
        crop_right = image[:, mid_x:]
        crops.append((crop_right, mid_x, 0))
    else:
        mid_y = h // 2
        print(f"  ✂️  Taie orizontal la y={mid_y}")
        crop_top = image[:mid_y, :]
        crops.append((crop_top, 0, 0))
        crop_bottom = image[mid_y:, :]
        crops.append((crop_bottom, 0, mid_y))
    
    return crops

def infer_on_roboflow(image, label="image"):
    _, img_encoded = cv2.imencode('.png', image)
    img_bytes = img_encoded.tobytes()
    
    h, w = image.shape[:2]
    print(f"  📤 Trimit {label}: {w}x{h}px, {len(img_bytes)/1024:.1f}KB")
    
    infer_url = f"https://detect.roboflow.com/{PROJECT}/{VERSION}"
    params = {
        "api_key": API_KEY,
        "confidence": CONFIDENCE * 100,
        "overlap": OVERLAP
    }
    files = {'file': ('image.png', img_bytes, 'image/png')}
    
    try:
        resp = requests.post(infer_url, params=params, files=files, timeout=30)
        if resp.status_code != 200:
            print(f"  ⚠️ API Error: {resp.text}")
            return []
        
        data = resp.json()
        predictions = data.get("predictions", [])
        print(f"  ✅ Primit {len(predictions)} predicții")
        return predictions
        
    except Exception as e:
        print(f"  ❌ Eroare: {e}")
        return []

def main():
    print(f"\n{'='*60}")
    print(f"🚀 SPLIT & COMPRESS MODE - FIXED")
    print(f"{'='*60}\n")
    
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Nu găsesc: {IMAGE_PATH}")
        return
    
    image_full = cv2.imread(IMAGE_PATH)
    if image_full is None:
        print("❌ Eroare la citirea imaginii.")
        return
    
    h_orig, w_orig = image_full.shape[:2]
    print(f"📐 Dimensiune originală: {w_orig}x{h_orig}px")
    
    max_dim = max(h_orig, w_orig)
    all_predictions = []
    
    if max_dim > SPLIT_THRESHOLD:
        print(f"📏 Imaginea e mare ({max_dim}px > {SPLIT_THRESHOLD}px)")
        print(f"✂️  O tai în 2 părți și compresez fiecare la ~{TARGET_SIZE}px\n")
        
        crops = split_image(image_full)
        
        for idx, (crop, offset_x, offset_y) in enumerate(crops, 1):
            h_crop, w_crop = crop.shape[:2]
            print(f"\n{'─'*60}")
            print(f"🔹 CROP {idx}/2: {w_crop}x{h_crop}px (offset: x={offset_x}, y={offset_y})")
            
            crop_resized, scale = resize_to_target(crop, TARGET_SIZE)
            h_resized, w_resized = crop_resized.shape[:2]
            print(f"  🔽 Redimensionat la: {w_resized}x{h_resized}px (scale={scale:.4f})")
            
            cv2.imwrite(f"debug_crop_{idx}_resized.png", crop_resized)
            
            predictions = infer_on_roboflow(crop_resized, f"Crop {idx}")
            
            for pred in predictions:
                # Scalează înapoi
                pred['x'] = (pred['x'] / scale) + offset_x
                pred['y'] = (pred['y'] / scale) + offset_y
                pred['width'] = pred['width'] / scale
                pred['height'] = pred['height'] / scale
                all_predictions.append(pred)
    
    else:
        print(f"📏 Imaginea e suficient de mică ({max_dim}px ≤ {SPLIT_THRESHOLD}px)")
        print(f"🔽 O compresez direct la ~{TARGET_SIZE}px\n")
        
        image_resized, scale = resize_to_target(image_full, TARGET_SIZE)
        h_resized, w_resized = image_resized.shape[:2]
        print(f"  📉 Redimensionat la: {w_resized}x{h_resized}px (scale={scale:.4f})")
        
        cv2.imwrite("debug_full_resized.png", image_resized)
        
        predictions = infer_on_roboflow(image_resized, "Full Image")
        
        for pred in predictions:
            pred['x'] = pred['x'] / scale
            pred['y'] = pred['y'] / scale
            pred['width'] = pred['width'] / scale
            pred['height'] = pred['height'] / scale
            all_predictions.append(pred)
    
    # Procesare și desenare
    print(f"\n{'='*60}")
    print(f"📊 REZULTATE FINALE")
    print(f"{'='*60}")
    print(f"Total predicții: {len(all_predictions)}\n")
    
    debug_img = image_full.copy()
    valid_count = 0
    class_counts = {}
    
    for pred in all_predictions:
        # Extrage valorile ODATĂ și le salvează în variabile separate
        pred_x = float(pred['x'])
        pred_y = float(pred['y'])
        pred_w = float(pred['width'])
        pred_h = float(pred['height'])
        raw_cls = pred['class']
        conf = pred['confidence']
        
        cls = normalize_class_name(raw_cls)
        if not cls:
            continue
        
        # Calculăm bbox - FĂRĂ a folosi min/max care pot cauza probleme
        x1_calc = pred_x - pred_w/2
        y1_calc = pred_y - pred_h/2
        x2_calc = pred_x + pred_w/2
        y2_calc = pred_y + pred_h/2
        
        # Convertim la int și facem clipping MANUAL
        x1 = int(x1_calc)
        y1 = int(y1_calc)
        x2 = int(x2_calc)
        y2 = int(y2_calc)
        
        # Clipping manual
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > w_orig: x2 = w_orig
        if y2 > h_orig: y2 = h_orig
        
        # Validare
        if x2 <= x1 or y2 <= y1:
            print(f"⚠️ Skip {cls}: bbox invalid ({x1},{y1})-({x2},{y2})")
            continue
        
        valid_count += 1
        class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # Desenare
        color = (0, 255, 0)
        if "window" in cls: color = (255, 0, 0)
        if "door" in cls: color = (0, 165, 255)
        
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {conf:.2f}"
        cv2.putText(debug_img, label, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Salvare
    output_path = "debug_split_result.jpg"
    cv2.imwrite(output_path, debug_img)
    
    print(f"✅ Detectat {valid_count} obiecte valide")
    print(f"📊 Distribuție: {class_counts}")
    print(f"💾 Rezultat: {output_path}")
    
    # Salvare crops
    crops_dir = "crops_split"
    os.makedirs(crops_dir, exist_ok=True)
    
    crop_count = 0
    for pred in all_predictions:
        cls = normalize_class_name(pred['class'])
        if not cls: continue
        
        px = float(pred['x'])
        py = float(pred['y'])
        pw = float(pred['width'])
        ph = float(pred['height'])
        
        cx1 = int(px - pw/2)
        cy1 = int(py - ph/2)
        cx2 = int(px + pw/2)
        cy2 = int(py + ph/2)
        
        if cx1 < 0: cx1 = 0
        if cy1 < 0: cy1 = 0
        if cx2 > w_orig: cx2 = w_orig
        if cy2 > h_orig: cy2 = h_orig
        
        if cx2 > cx1 and cy2 > cy1:
            crop = image_full[cy1:cy2, cx1:cx2]
            conf_int = int(pred['confidence'] * 100)
            cv2.imwrite(f"{crops_dir}/{cls}_{crop_count:03d}_conf{conf_int}.jpg", crop)
            crop_count += 1
    
    print(f"📁 Crops: {crops_dir}/ ({crop_count} fișiere)")
    print(f"\n{'='*60}")
    print("🎉 DONE!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()