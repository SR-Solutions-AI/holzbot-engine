# new/runner/cubicasa_detector/detector.py
from __future__ import annotations

import sys
import os
import cv2
import numpy as np
import torch
import json
import math
from pathlib import Path
from PIL import Image

from google import genai
from google.genai.errors import APIError

# ============================================
# CONFIGURARE PATHS
# ============================================

def _get_cubicasa_path():
    """Găsește automat CubiCasa5k relativ la acest fișier."""
    current_dir = Path(__file__).parent
    candidates = [
        current_dir / "CubiCasa5k",
        current_dir.parent / "CubiCasa5k",
        current_dir.parent.parent / "CubiCasa5k",
    ]
    
    for path in candidates:
        if path.exists():
            return str(path)
    
    raise FileNotFoundError(
        "Nu găsesc folderul CubiCasa5k. "
        "Plasează-l în runner/cubicasa_detector/ sau runner/"
    )

CUBICASA_PATH = _get_cubicasa_path()
sys.path.insert(0, CUBICASA_PATH)

try:
    from floortrans.models.hg_furukawa_original import hg_furukawa_original
except ImportError as e:
    raise ImportError(f"Nu pot importa modelul CubiCasa: {e}")


# ============================================
# HELPERS
# ============================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_step(name, img, steps_dir):
    path = Path(steps_dir) / f"{name}.png"
    cv2.imwrite(str(path), img)

def get_strict_1px_outline(mask):
    if np.count_nonzero(mask) == 0:
        return np.zeros_like(mask)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    return cv2.subtract(mask, eroded)

def smooth_walls_mask(mask):
    """Netezire pentru eliminarea jitter-ului."""
    if np.count_nonzero(mask) == 0:
        return mask
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return smoothed

def straighten_mask(mask, epsilon_factor=0.003):
    """Îndreptare vizuală pentru overlay."""
    if np.count_nonzero(mask) == 0:
        return mask
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    for cnt in contours:
        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(clean_mask, [approx], -1, 255, -1)
    return clean_mask

def flood_fill_room(indoor_mask, seed_pt, search_radius=10):
    """Flood fill pentru segmentare camere."""
    h, w = indoor_mask.shape[:2]
    x0, y0 = seed_pt
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))

    if indoor_mask[y0, x0] == 0:
        found = False
        for dy in range(-search_radius, search_radius + 1):
            yy = y0 + dy
            if 0 <= yy < h:
                for dx in range(-search_radius, search_radius + 1):
                    xx = x0 + dx
                    if 0 <= xx < w:
                        if indoor_mask[yy, xx] > 0:
                            x0, y0 = xx, yy
                            found = True
                            break
                if found:
                    break
        if not found:
            return np.zeros_like(indoor_mask), 0, seed_pt

    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    temp = indoor_mask.copy()
    try:
        _, _, mask, _ = cv2.floodFill(
            temp, mask, (x0, y0), 255,
            loDiff=(0,), upDiff=(0,),
            flags=cv2.FLOODFILL_MASK_ONLY | 4
        )
    except:
        return np.zeros_like(indoor_mask), 0, (x0, y0)

    room_mask = (mask[1:-1, 1:-1] != 0).astype(np.uint8) * 255
    area_px = int(np.count_nonzero(room_mask))
    return room_mask, area_px, (x0, y0)


# ============================================
# 3D RENDERER (Adăugat din codul vechi)
# ============================================

def render_obj_to_image(vertices_raw, faces_raw, output_image_path, width=1024, height=1024):
    """Randează o previzualizare 3D simplă a modelului OBJ."""
    if not vertices_raw or not faces_raw: return
    print("   📸 Randez previzualizarea 3D...")
    
    verts = []
    faces = []
    for v_line in vertices_raw: 
        parts = v_line.split()
        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    verts = np.array(verts)
    
    for f_line in faces_raw: 
        parts = f_line.split()
        # OBJ faces are 1-indexed
        faces.append([int(parts[1])-1, int(parts[2])-1, int(parts[3])-1, int(parts[4])-1])

    # Rotație izometrică-ish
    angle_y = np.radians(45)
    angle_x = np.radians(30)
    
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
    
    center = verts.mean(axis=0)
    verts_centered = verts - center
    rotated_verts = verts_centered @ Ry.T @ Rx.T
    
    range_val = np.max(rotated_verts) - np.min(rotated_verts)
    if range_val == 0: range_val = 1
    
    scale = min(width, height) / range_val * 0.7
    
    projected_2d = rotated_verts[:, :2] * scale
    projected_2d[:, 0] += width / 2
    projected_2d[:, 1] += height / 2
    
    # Flip Y pentru coordonate imagine
    projected_2d[:, 1] = height - projected_2d[:, 1]

    # Sortare fețe după adâncime (Painter's algorithm simplificat)
    face_depths = []
    for idx, face in enumerate(faces): 
        face_depths.append((np.mean(rotated_verts[face, 2]), idx))
    face_depths.sort(key=lambda x: x[0])

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    wall_color = (200, 200, 200)
    edge_color = (50, 50, 50)
    
    for _, face_idx in face_depths:
        pts = projected_2d[faces[face_idx]].astype(np.int32)
        cv2.fillPoly(canvas, [pts], wall_color)
        cv2.polylines(canvas, [pts], True, edge_color, 1, cv2.LINE_AA)
        
    cv2.imwrite(str(output_image_path), canvas)


def export_walls_to_obj(walls_mask, output_path, scale_m_px, wall_height_m=2.5, image_output_path=None):
    """Exportă masca pereților într-un fișier .OBJ 3D."""
    print("   🏗️  Generez model 3D (Smoothed)...")
    contours, _ = cv2.findContours(walls_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    vertices_str_list = []
    faces_str_list = []
    vertex_count = 0
    h_img, w_img = walls_mask.shape[:2]

    for cnt in contours:
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 3: continue
        
        base_idx = vertex_count + 1
        num_pts = len(approx)
        
        # Generăm vertecșii (jos și sus)
        for pt in approx:
            px, py = pt[0]
            # Centrare pe (0,0) pentru model
            x = (px - w_img/2) * scale_m_px
            z = (h_img - py - h_img/2) * scale_m_px 
            
            # Vertecs jos (y=0) și sus (y=height)
            vertices_str_list.append(f"v {x:.4f} 0.0000 {-z:.4f}")
            vertices_str_list.append(f"v {x:.4f} {wall_height_m:.4f} {-z:.4f}")
            
        # Generăm fețele (quads)
        for k in range(num_pts):
            curr = k
            next_p = (k + 1) % num_pts
            
            # Indici vertecși (1-based în OBJ)
            v_bl = base_idx + 2*curr
            v_tl = base_idx + 2*curr + 1
            v_br = base_idx + 2*next_p
            v_tr = base_idx + 2*next_p + 1
            
            faces_str_list.append(f"f {v_bl} {v_br} {v_tr} {v_tl}")
            
        vertex_count += 2 * num_pts

    # Scriere fișier OBJ
    with open(output_path, "w") as f:
        f.write(f"# Scale: {scale_m_px} m/px\n")
        f.write(f"# Holzbot Auto-Generated 3D Model\n")
        for v in vertices_str_list: f.write(v + "\n")
        for face in faces_str_list: f.write(face + "\n")
        
    print(f"   ✅ 3D Salvat: {output_path.name}")
    
    # Generare imagine preview
    if image_output_path:
        render_obj_to_image(vertices_str_list, faces_str_list, image_output_path)


# ============================================
# GEMINI API
# ============================================

GEMINI_PROMPT_CROP = """
Ești un expert în analiza planurilor arhitecturale. Din imaginea decupată, extrage:
1. "room_name": Numele camerei.
2. "area_m2": Suprafața numerică (folosește PUNCT pt zecimale).
Dacă nu e clar, returnează JSON gol {}.
Returnează DOAR JSON.
"""

GEMINI_PROMPT_TOTAL_SUM = """
Analizează întregul plan al etajului.
Identifică TOATE numerele care reprezintă suprafețe de camere (de obicei au m² lângă ele).
Adună toate aceste valori pentru a obține Suprafața Totală Utilă (Suma Label-urilor).
Returnează un JSON cu un singur câmp:
{"total_sum_m2": <suma_tuturor_camerelor_float>}
Returnează DOAR JSON.
"""

def call_gemini(image_path, prompt, api_key):
    """Apel Gemini cu handling robust."""
    try:
        client = genai.Client(api_key=api_key)
        img_pil = Image.open(image_path)
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=[prompt, img_pil]
        )
        reply = response.text.strip()
        
        if reply.startswith("```"):
            reply = reply[reply.find('{'):reply.rfind('}') + 1]
        
        if len(prompt) == len(GEMINI_PROMPT_CROP):
            if reply.count('\n') > 3 or len(reply) > 250:
                return None
        
        return json.loads(reply)
    except Exception as e:
        print(f"⚠️  Gemini error: {e}")
        return None


# ============================================
# SCALE DETECTION
# ============================================

def detect_scale_from_room_labels(
    image_path, 
    indoor_mask, 
    walls_mask, 
    steps_dir,
    min_room_area=1.0, 
    max_room_area=300.0, 
    api_key=None
):
    """
    Detectează scala din label-uri de camere.
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None
    
    h, w = img_bgr.shape[:2]

    # A. SEGMENTARE CAMERE
    room_candidates = []
    min_dim = min(h, w)
    kernel_size = max(3, int(min_dim / 100))
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    kernel_dynamic = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    walls_dilated = cv2.dilate(walls_mask, kernel_dynamic, iterations=1)
    fillable_mask = cv2.bitwise_and(cv2.bitwise_not(walls_dilated), indoor_mask)
    unvisited_mask = fillable_mask.copy()
    
    sample_stride = max(5, int(w / 200))
    MIN_PIXEL_AREA = max(100, int((w * h) * 0.0005))
    room_idx = 0
    debug_iteration = img_bgr.copy()

    print(f"   🔍 Segmentare camere (Kernel {kernel_size}px)...")
    
    for y in range(0, h, sample_stride):
        for x in range(0, w, sample_stride):
            if unvisited_mask[y, x] > 0:
                room_mask, area_px, _ = flood_fill_room(unvisited_mask, (x, y))
                
                if area_px < MIN_PIXEL_AREA:
                    unvisited_mask[room_mask > 0] = 0
                    continue
                
                room_idx += 1
                coords = np.where(room_mask > 0)
                
                if not coords[0].size:
                    continue
                
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                padding = max(5, int(min_dim / 100))
                y_s = max(0, y_min - padding)
                y_e = min(h, y_max + 1 + padding)
                x_s = max(0, x_min - padding)
                x_e = min(w, x_max + 1 + padding)
                
                crop_path = Path(steps_dir) / f"crop_{room_idx}.png"
                cv2.imwrite(str(crop_path), img_bgr[y_s:y_e, x_s:x_e])
                
                print(f"      ⏳ Segment {room_idx} ({area_px} px)...")
                res = call_gemini(crop_path, GEMINI_PROMPT_CROP, api_key)
                
                if res and 'area_m2' in res:
                    area_m2 = float(res['area_m2'])
                    if min_room_area <= area_m2 <= max_room_area:
                        room_candidates.append({
                            "index": room_idx,
                            "area_m2_label": area_m2,
                            "area_px": int(area_px),
                            "room_name": res.get('room_name', 'Unknown')
                        })
                        print(f"         ✅ {res.get('room_name')} -> {area_m2} m²")
                        cv2.rectangle(debug_iteration, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    else:
                        cv2.rectangle(debug_iteration, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
                else:
                    cv2.rectangle(debug_iteration, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
                
                unvisited_mask[room_mask > 0] = 0
    
    save_step("debug_segmentation_final", debug_iteration, steps_dir)
    
    if not room_candidates:
        print("      ❌ Nu s-au găsit camere valide")
        return None

    # B. CALCUL SCARĂ LOCALĂ
    area_labels = np.array([r["area_m2_label"] for r in room_candidates])
    area_pixels = np.array([r["area_px"] for r in room_candidates])
    
    num = np.sum(area_pixels * area_labels)
    den = np.sum(area_pixels ** 2)
    
    if den == 0:
        return None
    
    m_px_local = float(np.sqrt(num / den))
    print(f"   📊 Scara Locală: {m_px_local:.9f} m/px")

    # C. CALCUL SCARĂ GLOBALĂ
    print("   🔍 Calcul Scară Globală (Suma Totală)...")
    total_indoor_px = int(np.count_nonzero(indoor_mask))
    
    gemini_total = call_gemini(image_path, GEMINI_PROMPT_TOTAL_SUM, api_key)
    
    if gemini_total and 'total_sum_m2' in gemini_total:
        sum_labels_m2 = float(gemini_total['total_sum_m2'])
        print(f"      ℹ️  Suma Gemini: {sum_labels_m2:.2f} m²")
    else:
        sum_labels_m2 = sum(area_labels)
        print(f"      ℹ️  Suma Segmente: {sum_labels_m2:.2f} m²")

    m_px_target = math.sqrt(sum_labels_m2 / float(total_indoor_px))
    suprafata_cu_camere = total_indoor_px * (m_px_local ** 2)
    
    # D. DECIZIE FINALĂ
    final_m_px = m_px_local
    method_note = "Scară Locală"
    
    if suprafata_cu_camere < sum_labels_m2:
        print("      ⚠️  Aria Calculată < Aria Etichetelor! Recalculez.")
        final_m_px = m_px_target
        method_note = "Forțat pe Suma Totală"
    elif suprafata_cu_camere > sum_labels_m2 * 1.20:
        print("      ℹ️  Aria Calculată > Aria Etichetelor. Păstrez scara locală.")
    else:
        print("      ✅ Valorile sunt consistente.")
    
    print(f"   ✅ Scară Finală: {final_m_px:.9f} m/px")

    # E. CALCUL ERORI PER CAMERĂ
    m_px_sq = final_m_px ** 2
    
    for r in room_candidates:
        est = r["area_px"] * m_px_sq
        err = abs(r["area_m2_label"] - est) / r["area_m2_label"] * 100
        r["error_percent"] = round(err, 4)
        r["area_est"] = round(est, 2)
    
    return {
        "method": "cubicasa_gemini",
        "meters_per_pixel": float(final_m_px),  # ✅ Explicit float
        "rooms_used": len(room_candidates),
        "optimization_info": {
            "m_px_local": float(m_px_local),
            "m_px_target": float(m_px_target),
            "sum_labels": float(sum_labels_m2),
            "calc_area_local": float(suprafata_cu_camere),
            "method_note": method_note
        },
        "per_room": room_candidates
    }


# ============================================
# FUNCȚIA PRINCIPALĂ
# ============================================

def run_cubicasa_detection(
    image_path: str,
    model_weights_path: str,
    output_dir: str,
    gemini_api_key: str,
    device: torch.device,
    save_debug_steps: bool = True
) -> dict:
    """
    Rulează detecția CubiCasa + măsurări + 3D Generation.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    model_weights_path = Path(model_weights_path)
    
    steps_dir = output_dir / "cubicasa_steps"
    ensure_dir(steps_dir)
    
    print(f"   🤖 CubiCasa: Procesez {image_path.name}")
    
    # 1. SETUP MODEL
    print(f"   ⏳ Încarc AI pe {device.type}...")
    model = hg_furukawa_original(n_classes=44)
    
    if not model_weights_path.exists():
        raise FileNotFoundError(f"Lipsesc weights: {model_weights_path}")
    
    checkpoint = torch.load(str(model_weights_path), map_location=device)
    model.load_state_dict(
        checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    )
    model.to(device)
    model.eval()

    # 2. PREPROCESARE & INFERENȚĂ
    img = cv2.imread(str(image_path))
    h_orig, w_orig = img.shape[:2]
    save_step("00_original", img, str(steps_dir))
    
    input_img = cv2.resize(img, (512, 512))
    norm_img = 2 * (input_img.astype(np.float32) / 255.0) - 1
    tensor = torch.from_numpy(norm_img).float().permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        pred_mask = torch.argmax(output[:, 21:33, :, :], dim=1).squeeze().cpu().numpy()
    
    ai_walls_raw = cv2.resize(
        (pred_mask == 2).astype('uint8') * 255,
        (w_orig, h_orig),
        interpolation=cv2.INTER_NEAREST
    )
    save_step("01_ai_walls_raw", ai_walls_raw, str(steps_dir))

    # 3. REPARARE PEREȚI
    print("   📐 Repar pereții...")
    min_dim = min(h_orig, w_orig)
    rep_k = max(3, int(min_dim * 0.005))
    if rep_k % 2 == 0:
        rep_k += 1
    
    kernel_repair = cv2.getStructuringElement(cv2.MORPH_RECT, (rep_k, rep_k))
    ai_walls_closed = cv2.morphologyEx(ai_walls_raw, cv2.MORPH_CLOSE, kernel_repair, iterations=1)
    save_step("01b_ai_walls_closed", ai_walls_closed, str(steps_dir))
    
    # 4. NETEZIRE
    print("   📐 Aplic smoothing...")
    ai_walls_smoothed = smooth_walls_mask(ai_walls_closed)
    save_step("02_ai_walls_smoothed", ai_walls_smoothed, str(steps_dir))
    
    # 5. ÎNDREPTARE
    ai_walls_straight = straighten_mask(ai_walls_closed, 0.003)
    save_step("02_ai_walls_straight", ai_walls_straight, str(steps_dir))

    # 6. ZONE (INDOOR/OUTDOOR)
    print("   🌊 Analizez zonele...")
    walls_thick = cv2.dilate(ai_walls_closed, kernel_repair, iterations=3)
    h_pad, w_pad = h_orig + 2, w_orig + 2
    pad_walls = np.zeros((h_pad, w_pad), dtype=np.uint8)
    pad_walls[1:h_orig+1, 1:w_orig+1] = walls_thick
    
    inv_pad_walls = cv2.bitwise_not(pad_walls)
    flood_mask = np.zeros((h_pad+2, w_pad+2), dtype=np.uint8)
    cv2.floodFill(inv_pad_walls, flood_mask, (0, 0), 128)
    
    outdoor_mask = (inv_pad_walls[1:h_orig+1, 1:w_orig+1] == 128).astype(np.uint8) * 255
    
    kernel_grow = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    free_space = cv2.bitwise_not(ai_walls_closed)
    
    for _ in range(30):
        outdoor_mask = cv2.bitwise_and(cv2.dilate(outdoor_mask, kernel_grow), free_space)
    
    save_step("03_outdoor_mask", outdoor_mask, str(steps_dir))
    
    total_space = np.ones_like(outdoor_mask) * 255
    occupied = cv2.bitwise_or(outdoor_mask, ai_walls_closed)
    indoor_mask = cv2.subtract(total_space, occupied)
    
    # Debug visualizations
    vis_outdoor = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    vis_outdoor[outdoor_mask > 0] = [0, 0, 255]
    save_step("debug_zone_exterior", vis_outdoor, str(steps_dir))
    
    vis_indoor = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    vis_indoor[indoor_mask > 0] = [0, 255, 255]
    save_step("debug_zone_interior", vis_indoor, str(steps_dir))

    # 7. SCALE DETECTION
    print("   🔍 Determin scala...")
    scale_result = detect_scale_from_room_labels(
        str(image_path),
        indoor_mask,
        ai_walls_closed,
        str(steps_dir),
        api_key=gemini_api_key
    )
    
    if not scale_result:
        raise RuntimeError("Nu am putut determina scala!")
    
    m_px = scale_result["meters_per_pixel"]

    # 8. MĂSURĂTORI
    print("   📏 Calculez măsurători...")
    outline = get_strict_1px_outline(ai_walls_closed)
    touch_zone = cv2.dilate(outdoor_mask, kernel_grow, iterations=2)
    
    outline_ext_mask = cv2.bitwise_and(outline, touch_zone)
    outline_int_mask = cv2.subtract(outline, outline_ext_mask)
    
    px_len_ext = int(np.count_nonzero(outline_ext_mask))
    px_len_int = int(np.count_nonzero(outline_int_mask))
    px_area_indoor = int(np.count_nonzero(indoor_mask))
    px_area_total = int(np.count_nonzero(cv2.bitwise_not(outdoor_mask)))

    walls_ext_m = px_len_ext * m_px
    walls_int_m = px_len_int * m_px
    area_indoor_m2 = px_area_indoor * (m_px ** 2)
    area_total_m2 = px_area_total * (m_px ** 2)

    measurements = {
        "pixels": {
            "walls_len_ext": int(px_len_ext),
            "walls_len_int": int(px_len_int),
            "indoor_area": int(px_area_indoor),
            "total_area": int(px_area_total)
        },
        "metrics": {
            "scale_m_per_px": float(m_px),
            "walls_ext_m": float(round(walls_ext_m, 2)),
            "walls_int_m": float(round(walls_int_m, 2)),
            "area_indoor_m2": float(round(area_indoor_m2, 2)),
            "area_total_m2": float(round(area_total_m2, 2))
        }
    }

    print(f"   ✅ Scară: {m_px:.9f} m/px")
    print(f"   🧱 Pereți Ext: {walls_ext_m:.2f} m")
    print(f"   🧱 Pereți Int: {walls_int_m:.2f} m")
    print(f"   🏠 Arie Indoor: {area_indoor_m2:.2f} m²")
    
    # ============================================
    # 9. GENERARE 3D (NOU)
    # ============================================
    print("   🏗️  Pornesc generarea 3D...")
    
    # Generăm modelul OBJ și imaginea de previzualizare
    # Imaginea se salvează ca 'walls_3d_view.png'
    export_walls_to_obj(
        ai_walls_smoothed, 
        output_dir / "walls_3d.obj", 
        m_px, 
        image_output_path=output_dir / "walls_3d_view.png"
    )

    # 10. SALVARE VIZUALIZĂRI 2D
    black_vis = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    black_vis[outline_ext_mask > 0] = [255, 0, 0]  # Roșu = exterior
    black_vis[outline_int_mask > 0] = [0, 255, 0]  # Verde = interior
    cv2.imwrite(str(output_dir / "final_colored_lines.png"), black_vis)

    overlay = img.copy()
    overlay[outline_ext_mask > 0] = [255, 0, 0]
    overlay[outline_int_mask > 0] = [0, 255, 0]
    cv2.imwrite(str(output_dir / "visualization_overlay.png"), overlay)
    
    clean_overlay = img.copy()
    clean_overlay[ai_walls_straight > 0] = [0, 255, 0]
    cv2.imwrite(str(output_dir / "final_viz.png"), clean_overlay)

    # ✅ RETURN rezultate (Path-uri actualizate)
    return {
        "scale_result": scale_result,
        "measurements": measurements,
        "masks": {
            "walls_closed": str(steps_dir / "01b_ai_walls_closed.png"),
            "walls_smoothed": str(steps_dir / "02_ai_walls_smoothed.png"),
            "indoor_mask": str(steps_dir / "debug_zone_interior.png"),
            "outdoor_mask": str(steps_dir / "debug_zone_exterior.png"),
            "segmentation": str(steps_dir / "debug_segmentation_final.png"),
            "visualization": str(output_dir / "visualization_overlay.png"),
            "visualization_3d": str(output_dir / "walls_3d_view.png") # <--- Aici e imaginea 3D
        }
    }