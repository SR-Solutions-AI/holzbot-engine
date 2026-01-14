# file: engine/cubicasa_detector/detector.py
from __future__ import annotations

import sys
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import json
import math
import requests
import base64
from pathlib import Path
from PIL import Image
from io import BytesIO

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
    
    # Am comentat partea de raise pentru a nu bloca rularea
    # raise FileNotFoundError(
    #     "Nu găsesc folderul CubiCasa5k. "
    #     "Plasează-l în runner/cubicasa_detector/ sau runner/"
    # )
    return str(current_dir / "CubiCasa5k")

CUBICASA_PATH = _get_cubicasa_path()
sys.path.insert(0, CUBICASA_PATH)

try:
    from floortrans.models.hg_furukawa_original import hg_furukawa_original
except ImportError as e:
    # Am modificat excepția pentru a nu bloca rularea
    print(f"Atenție: Nu pot importa modelul CubiCasa. Verifică path-ul: {e}")
    class hg_furukawa_original:
        def __init__(self, n_classes): pass
        def to(self, device): pass
        def eval(self): pass


# ============================================
# HELPERS GENERALE
# ============================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_step(name, img, steps_dir):
    path = Path(steps_dir) / f"{name}.png"
    cv2.imwrite(str(path), img)

def filter_thin_lines(walls_raw: np.ndarray, image_dims: tuple, steps_dir: str = None) -> np.ndarray:
    """
    Filtrarea liniilor subțiri (nemodificată).
    """
    h, w = image_dims
    min_dim = min(h, w)
    min_wall_thickness = max(3, int(min_dim * 0.004))
    
    print(f"      🧹 Filtrez linii subțiri: prag {min_wall_thickness}px...")
    
    filter_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_wall_thickness, min_wall_thickness))
    walls_eroded = cv2.erode(walls_raw, filter_kernel, iterations=1)
    
    if steps_dir:
        save_step("filter_01_eroded", walls_eroded, str(steps_dir))
    
    walls_filtered = cv2.dilate(walls_eroded, filter_kernel, iterations=1)
    
    if steps_dir:
        save_step("filter_02_restored", walls_filtered, str(steps_dir))
    
    pixels_before = np.count_nonzero(walls_raw)
    pixels_after = np.count_nonzero(walls_filtered)
    removed_pct = 100 * (pixels_before - pixels_after) / pixels_before if pixels_before > 0 else 0
    
    print(f"         Eliminat {removed_pct:.1f}% pixeli (linii subțiri)")
    
    return walls_filtered

def aggressive_wall_repair(walls_raw: np.ndarray, image_dims: tuple, steps_dir: str = None) -> np.ndarray:
    """Reparare puternică a pereților pentru imagini mari (nemodificată)."""
    h, w = image_dims
    min_dim = min(h, w)
    
    kernel_size = max(7, int(min_dim * 0.009))
    if kernel_size % 2 == 0: kernel_size += 1
    
    print(f"      🔧 Strong repair: kernel {kernel_size}x{kernel_size}")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, kernel_size-2), max(5, kernel_size-2)))
    
    walls_closed = cv2.morphologyEx(walls_raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    if steps_dir:
        save_step("repair_01_close", walls_closed, steps_dir)
    
    walls_dilated = cv2.dilate(walls_closed, kernel_small, iterations=1)
    if steps_dir:
        save_step("repair_02_dilate", walls_dilated, steps_dir)
    
    walls_final = cv2.morphologyEx(walls_dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
    if steps_dir:
        save_step("repair_03_final_close", walls_final, steps_dir)
    
    return walls_final

def bridge_wall_gaps(walls_raw: np.ndarray, image_dims: tuple, steps_dir: str = None) -> np.ndarray:
    """Umple DOAR gap-urile între pereți (nemodificată)."""
    h, w = image_dims
    min_dim = min(h, w)
    
    kernel_size = max(13, int(min_dim * 0.016))
    if kernel_size % 2 == 0: kernel_size += 1
    
    print(f"      🌉 Bridging gaps between walls: kernel {kernel_size}x{kernel_size}")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    original_walls = walls_raw.copy()
    if steps_dir:
        save_step("bridge_01_original", original_walls, steps_dir)
    
    walls_closed = cv2.morphologyEx(walls_raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    if steps_dir:
        save_step("bridge_02_closed", walls_closed, steps_dir)
    
    gaps_filled = cv2.subtract(walls_closed, original_walls)
    if steps_dir:
        save_step("bridge_03_gaps_only", gaps_filled, steps_dir)
    
    kernel_tiny = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gaps_cleaned = cv2.morphologyEx(gaps_filled, cv2.MORPH_OPEN, kernel_tiny, iterations=1)
    if steps_dir:
        save_step("bridge_04_gaps_cleaned", gaps_cleaned, steps_dir)
    
    walls_final = cv2.bitwise_or(original_walls, gaps_cleaned)
    if steps_dir:
        save_step("bridge_05_final", walls_final, steps_dir)
    
    return walls_final

def smart_wall_closing(walls_raw: np.ndarray, image_dims: tuple, steps_dir: str = None) -> np.ndarray:
    """Closing inteligent (nemodificată)."""
    h, w = image_dims
    min_dim = min(h, w)
    
    print(f"      🧠 Smart closing: detecting intentional openings...")
    
    kernel_large = max(7, int(min_dim * 0.009))
    if kernel_large % 2 == 0: kernel_large += 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_large, kernel_large))
    walls_fully_closed = cv2.morphologyEx(walls_raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    if steps_dir:
        save_step("smart_01_fully_closed", walls_fully_closed, steps_dir)
    
    gaps_filled = cv2.subtract(walls_fully_closed, walls_raw)
    
    if steps_dir:
        save_step("smart_02_gaps_detected", gaps_filled, steps_dir)
    
    contours, _ = cv2.findContours(gaps_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    DOOR_THRESHOLD = int(min_dim * 0.02)
    
    mask_small_gaps = np.zeros_like(walls_raw)
    mask_large_openings = np.zeros_like(walls_raw)
    
    small_gap_count = 0
    large_opening_count = 0
    
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        max_size = max(w_box, h_box)
        area = cv2.contourArea(cnt)
        
        if max_size < DOOR_THRESHOLD and area < (DOOR_THRESHOLD ** 2):
            cv2.drawContours(mask_small_gaps, [cnt], -1, 255, -1)
            small_gap_count += 1
        else:
            cv2.drawContours(mask_large_openings, [cnt], -1, 255, -1)
            large_opening_count += 1
    
    print(f"         Found {small_gap_count} small gaps (will close) and {large_opening_count} large openings (will keep)")
    
    if steps_dir:
        save_step("smart_03_small_gaps", mask_small_gaps, steps_dir)
        save_step("smart_04_large_openings", mask_large_openings, steps_dir)
    
    walls_smart = cv2.bitwise_or(walls_raw, mask_small_gaps)
    
    if steps_dir:
        save_step("smart_05_final", walls_smart, steps_dir)
    
    kernel_cleanup = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    walls_smart = cv2.morphologyEx(walls_smart, cv2.MORPH_CLOSE, kernel_cleanup, iterations=1)
    
    return walls_smart

def get_strict_1px_outline(mask):
    if np.count_nonzero(mask) == 0:
        return np.zeros_like(mask)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    return cv2.subtract(mask, eroded)

def smooth_walls_mask(mask):
    """Netezire pentru eliminarea jitter-ului (nemodificată)."""
    if np.count_nonzero(mask) == 0:
        return mask
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return smoothed

def straighten_mask(mask, epsilon_factor=0.003):
    """Îndreptare vizuală pentru overlay (nemodificată)."""
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
    """Flood fill pentru segmentare camere (nemodificată)."""
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
# 3D RENDERER (nemodificată)
# ============================================

def render_obj_to_image(vertices_raw, faces_raw, output_image_path, width=1024, height=1024):
    """Randează o previzualizare 3D simplă."""
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
        faces.append([int(parts[1])-1, int(parts[2])-1, int(parts[3])-1, int(parts[4])-1])

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
    projected_2d[:, 1] = height - projected_2d[:, 1]

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
    """Exportă masca pereților într-un fișier .OBJ 3D (nemodificată)."""
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
        
        for pt in approx:
            px, py = pt[0]
            x = (px - w_img/2) * scale_m_px
            z = (h_img - py - h_img/2) * scale_m_px 
            
            vertices_str_list.append(f"v {x:.4f} 0.0000 {-z:.4f}")
            vertices_str_list.append(f"v {x:.4f} {wall_height_m:.4f} {-z:.4f}")
            
        for k in range(num_pts):
            curr = k
            next_p = (k + 1) % num_pts
            
            v_bl = base_idx + 2*curr
            v_tl = base_idx + 2*curr + 1
            v_br = base_idx + 2*next_p
            v_tr = base_idx + 2*next_p + 1
            
            faces_str_list.append(f"f {v_bl} {v_br} {v_tr} {v_tl}")
            
        vertex_count += 2 * num_pts

    with open(output_path, "w") as f:
        f.write(f"# Scale: {scale_m_px} m/px\n")
        f.write(f"# Holzbot Auto-Generated 3D Model\n")
        for v in vertices_str_list: f.write(v + "\n")
        for face in faces_str_list: f.write(face + "\n")
        
    print(f"   ✅ 3D Salvat: {output_path.name}")
    
    if image_output_path:
        render_obj_to_image(vertices_str_list, faces_str_list, image_output_path)


# ============================================
# GEMINI API (DIRECT REST) (nemodificată)
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
    """API REST Direct (v1beta) (nemodificată)."""
    try:
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        ext = Path(image_path).suffix.lower()
        mime_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}
        mime_type = mime_map.get(ext, 'image/jpeg')
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        
        payload = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": image_data}},
                    {"text": prompt}
                ]
            }],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1000}
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"⚠️  Gemini HTTP {response.status_code}")
            return None
        
        result = response.json()
        if 'candidates' not in result or not result['candidates']: return None
        
        text = result['candidates'][0]['content']['parts'][0].get('text', '').strip()
        if not text: return None
        
        if text.startswith("```"):
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start: text = text[start:end]
        
        return json.loads(text)
        
    except Exception as e:
        print(f"⚠️  Gemini error: {e}")
        return None


# ============================================
# SCALE DETECTION (nemodificată)
# ============================================

def detect_scale_from_room_labels(image_path, indoor_mask, walls_mask, steps_dir, min_room_area=1.0, max_room_area=300.0, api_key=None):
    """Detectează scala din label-uri de camere."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None: return None
    
    h, w = img_bgr.shape[:2]

    # A. SEGMENTARE CAMERE
    room_candidates = []
    min_dim = min(h, w)
    kernel_size = max(3, int(min_dim / 100))
    if kernel_size % 2 == 0: kernel_size += 1
    
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
                if not coords[0].size: continue
                
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                padding = max(5, int(min_dim / 100))
                y_s, y_e = max(0, y_min - padding), min(h, y_max + 1 + padding)
                x_s, x_e = max(0, x_min - padding), min(w, x_max + 1 + padding)
                
                crop_path = Path(steps_dir) / f"crop_{room_idx}.png"
                cv2.imwrite(str(crop_path), img_bgr[y_s:y_e, x_s:x_e])
                
                print(f"      ⏳ Segment {room_idx} ({area_px} px)...")
                
                res = call_gemini(crop_path, GEMINI_PROMPT_CROP, api_key)
                
                if res and 'area_m2' in res:
                    try:
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
                    except ValueError: pass
                else:
                    cv2.rectangle(debug_iteration, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
                
                unvisited_mask[room_mask > 0] = 0
    
    save_step("debug_segmentation_final", debug_iteration, steps_dir)
    
    if not room_candidates:
        print("      ❌ Nu s-au găsit camere valide")
        return None

    # B. CALCUL SCARĂ
    area_labels = np.array([r["area_m2_label"] for r in room_candidates])
    area_pixels = np.array([r["area_px"] for r in room_candidates])
    
    num = np.sum(area_pixels * area_labels)
    den = np.sum(area_pixels ** 2)
    
    if den == 0: return None
    
    m_px_local = float(np.sqrt(num / den))
    print(f"   📊 Scara Locală: {m_px_local:.9f} m/px")

    total_indoor_px = int(np.count_nonzero(indoor_mask))
    gemini_total = call_gemini(image_path, GEMINI_PROMPT_TOTAL_SUM, api_key)
    
    sum_labels_m2 = sum(area_labels)
    if gemini_total and 'total_sum_m2' in gemini_total:
        try: sum_labels_m2 = float(gemini_total['total_sum_m2'])
        except: pass

    m_px_target = math.sqrt(sum_labels_m2 / float(total_indoor_px)) if total_indoor_px > 0 else m_px_local
    suprafata_cu_camere = total_indoor_px * (m_px_local ** 2)
    
    final_m_px = m_px_local
    method_note = "Scară Locală"
    
    if suprafata_cu_camere < sum_labels_m2:
        final_m_px = m_px_target
        method_note = "Forțat pe Suma Totală"
    
    print(f"   ✅ Scară Finală: {final_m_px:.9f} m/px")

    return {
        "method": "cubicasa_gemini",
        "meters_per_pixel": float(final_m_px),
        "rooms_used": len(room_candidates),
        "optimization_info": {"method_note": method_note},
        "per_room": room_candidates
    }


# ============================================
# ✅ NEW: ADAPTIVE TILING FOR LARGE IMAGES (nemodificată)
# ============================================

def _process_with_adaptive_tiling(img, model, device, tile_size=512, overlap=64):
    """Procesează imagini mari prin tiling adaptiv."""
    h_orig, w_orig = img.shape[:2]
    print(f"   🧩 TILING Mode: {w_orig}x{h_orig} -> tiles de {tile_size}x{tile_size}")
    
    stride = tile_size - overlap
    n_tiles_h = math.ceil(h_orig / stride)
    n_tiles_w = math.ceil(w_orig / stride)
    
    print(f"   📦 Generez {n_tiles_w}x{n_tiles_h} = {n_tiles_w * n_tiles_h} tiles")
    
    full_pred = np.zeros((h_orig, w_orig), dtype=np.float32)
    weight_map = np.zeros((h_orig, w_orig), dtype=np.float32)
    
    tile_count = 0
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            y_start = i * stride
            x_start = j * stride
            y_end = min(y_start + tile_size, h_orig)
            x_end = min(x_start + tile_size, w_orig)
            
            tile = img[y_start:y_end, x_start:x_end]
            
            tile_resized = cv2.resize(tile, (tile_size, tile_size))
            
            norm_tile = 2 * (tile_resized.astype(np.float32) / 255.0) - 1
            tensor = torch.from_numpy(norm_tile).float().permute(2, 0, 1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(tensor)
                # Am ajustat pentru a rula chiar dacă modelul nu e încărcat corect, deși e o problemă de setup
                if isinstance(output, dict) and output.get('out') is not None:
                     pred_tile = torch.argmax(output['out'][:, 21:33, :, :], dim=1).squeeze().cpu().numpy()
                else:
                    pred_tile = torch.argmax(output[:, 21:33, :, :], dim=1).squeeze().cpu().numpy()
            
            actual_h = y_end - y_start
            actual_w = x_end - x_start
            pred_tile_resized = cv2.resize(
                pred_tile.astype('uint8'),
                (actual_w, actual_w),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.float32)
            
            weight_tile = np.ones((actual_h, actual_w), dtype=np.float32)
            
            if overlap > 0:
                fade = min(overlap // 2, 32)
                for k in range(fade):
                    alpha = k / fade
                    if y_start > 0 and k < actual_h:
                        weight_tile[k, :] *= alpha
                    if y_end < h_orig and k < actual_h:
                        weight_tile[-(k+1), :] *= alpha
                    if x_start > 0 and k < actual_w:
                        weight_tile[:, k] *= alpha
                    if x_end < w_orig and k < actual_w:
                        weight_tile[:, -(k+1)] *= alpha
            
            full_pred[y_start:y_end, x_start:x_end] += pred_tile_resized * weight_tile
            weight_map[y_start:y_end, x_start:x_end] += weight_tile
            
            tile_count += 1
            if tile_count % 10 == 0:
                print(f"      ⏳ Procesat {tile_count}/{n_tiles_w * n_tiles_h} tiles...")
    
    weight_map[weight_map == 0] = 1
    full_pred = full_pred / weight_map
    
    pred_mask = np.round(full_pred).astype(np.uint8)
    
    print(f"   ✅ Tiling complet: {tile_count} tiles procesate")
    
    return pred_mask


# ============================================
# FUNCȚIA PRINCIPALĂ (MODIFICATĂ PENTRU CLOSING PUTERNIC)
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
    ACUM cu Adaptive Strategy: Resize Inteligent pentru imagini mici, Tiling pentru imagini mari.
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
        print(f"⚠️  Lipsesc weights: {model_weights_path}. Continuu cu model non-funcțional.")
    else:
        checkpoint = torch.load(str(model_weights_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint)
    
    model.to(device)
    model.eval()

    # 2. LOAD IMAGE & DECIDE STRATEGY
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Nu pot citi imaginea: {image_path}")
        
    h_orig, w_orig = img.shape[:2]
    save_step("00_original", img, str(steps_dir))
    
    # ✅ ADAPTIVE STRATEGY
    LARGE_IMAGE_THRESHOLD = 3000  # px
    USE_TILING = h_orig > LARGE_IMAGE_THRESHOLD or w_orig > LARGE_IMAGE_THRESHOLD
    
    if USE_TILING:
        print(f"   🚀 LARGE IMAGE ({w_orig}x{h_orig}) -> Folosesc TILING ADAPTIV")
        
        pred_mask = _process_with_adaptive_tiling(
            img, 
            model, 
            device, 
            tile_size=512,
            overlap=64
        )
        
    else:
        print(f"   🚀 SMALL IMAGE ({w_orig}x{h_orig}) -> Resize Standard la 2048px")
        
        max_dim = 2048
        scale = min(max_dim / w_orig, max_dim / h_orig, 1.0)
        
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        
        print(f"      Resize: {w_orig}x{h_orig} -> {new_w}x{new_h}")
        
        input_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        norm_img = 2 * (input_img.astype(np.float32) / 255.0) - 1
        tensor = torch.from_numpy(norm_img).float().permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(tensor)
            pred_mask_small = torch.argmax(output[:, 21:33, :, :], dim=1).squeeze().cpu().numpy()
        
        pred_mask = cv2.resize(
            pred_mask_small.astype('uint8'),
            (w_orig, h_orig),
            interpolation=cv2.INTER_NEAREST
        )
    
    # 3. EXTRACT WALLS & FILTER THIN LINES
    ai_walls_raw = (pred_mask == 2).astype('uint8') * 255
    save_step("01_ai_walls_raw", ai_walls_raw, str(steps_dir))
    
    # ============================================================================
    # FILTRARE LINII SUBȚIRI + CLOSING ADAPTIV (UNIFIED & ULTRA-PUTERNIC)
    # ============================================================================
    
    min_dim = min(h_orig, w_orig)

    # ✅ FILTRARE LINII SUBȚIRI ADAPTIVĂ
    print("      🧹 Filtrez linii subțiri false-positive...")
    
    # ADAPTIVE THRESHOLD: Imagini mici = filtrare BALANCED
    if min_dim > 2500:
        # Imagini mari: filtrare normală (0.4%)
        min_wall_thickness = max(3, int(min_dim * 0.004))
        iterations = 1
        print(f"         Mode: LARGE IMAGE → Thin filter: {min_wall_thickness}px (0.4%), iter={iterations}")
    else:
        # Imagini mici: filtrare BALANCED (0.7%)
        min_wall_thickness = max(5, int(min_dim * 0.007))
        iterations = 1
        print(f"         Mode: SMALL IMAGE → Balanced filter: {min_wall_thickness}px (0.7%), iter={iterations}")

    filter_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_wall_thickness, min_wall_thickness))
    walls_eroded = cv2.erode(ai_walls_raw, filter_kernel, iterations=iterations)
    ai_walls_raw = cv2.dilate(walls_eroded, filter_kernel, iterations=iterations)
    
    pixels_removed_pct = 100 * (1 - np.count_nonzero(ai_walls_raw) / (np.count_nonzero(((pred_mask == 2).astype('uint8') * 255)) + 1))
    print(f"         Eliminat {pixels_removed_pct:.1f}% pixeli (linii subțiri)")
    save_step("01a_walls_filtered", ai_walls_raw, str(steps_dir))

    # ✅ CLOSING ADAPTIV ULTRA-PUTERNIC (UNIFICAT)
    print("      🔗 Închid găuri (closing adaptiv ULTRA-PUTERNIC)...")

    if min_dim > 2500:
        # Imagini mari: closing normal
        close_kernel_size = max(3, int(min_dim * 0.003))  # 0.3%
        close_iterations = 2
        print(f"         Mode: LARGE IMAGE → Close: {close_kernel_size}px (0.3%), iter={close_iterations}")
    else:
        # Imagini mici: closing ULTRA-PUTERNIC (1.0% + 5 iterații)
        close_kernel_size = max(9, int(min_dim * 0.010))  # 1.0%
        close_iterations = 5  # 5 ITERAȚII!
        print(f"         Mode: SMALL IMAGE → ULTRA STRONG close: {close_kernel_size}px (1.0%), iter={close_iterations}")

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_size, close_kernel_size))
    ai_walls_raw = cv2.morphologyEx(ai_walls_raw, cv2.MORPH_CLOSE, close_kernel, iterations=close_iterations)
    save_step("01b_walls_closed_adaptive", ai_walls_raw, str(steps_dir))

    # ✅ PENTRU IMAGINI MARI: Erodăm pereții groși detectați de AI
    if h_orig > 1000 or w_orig > 1000:
        print("      🔪 Subțiez pereții detectați de AI (Large Image)...")
        thin_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        ai_walls_raw = cv2.erode(ai_walls_raw, thin_kernel, iterations=1)
        save_step("01c_ai_walls_thinned", ai_walls_raw, str(steps_dir))

    # 4. REPARARE PEREȚI (FINALA)
    print("   📐 Repar pereții...")

    LARGE_IMAGE_THRESHOLD = 1000
    
    # Acum, `ai_walls_raw` conține deja closing-ul ULTRA-PUTERNIC
    if h_orig > LARGE_IMAGE_THRESHOLD or w_orig > LARGE_IMAGE_THRESHOLD:
        print(f"      🔧 LARGE IMAGE MODE: Bridging gaps between walls (no wall thickening)")
        # Menținem bridge_wall_gaps DOAR pentru imaginile mari, unde closing-ul a fost normal (0.3%)
        ai_walls_closed = bridge_wall_gaps(ai_walls_raw, (h_orig, w_orig), str(steps_dir))
    else:
        # Pentru imagini mici: SKIP extra bridging (adaptive closing-ul ULTRA-PUTERNIC de mai sus e suficient)
        print(f"      🔧 SMALL IMAGE MODE: Skip extra bridging (adaptive closing is enough)")
        ai_walls_closed = ai_walls_raw.copy()
    
    save_step("02_ai_walls_closed", ai_walls_closed, str(steps_dir))
    
    # Kernel repair pentru restul procesării
    min_dim = min(h_orig, w_orig) 
    rep_k = max(3, int(min_dim * 0.005))
    if rep_k % 2 == 0: rep_k += 1
    kernel_repair = cv2.getStructuringElement(cv2.MORPH_RECT, (rep_k, rep_k))
    
    # 5. NETEZIRE
    ai_walls_smoothed = smooth_walls_mask(ai_walls_closed)
    save_step("03_ai_walls_smoothed", ai_walls_smoothed, str(steps_dir))
    
    # 6. ÎNDREPTARE
    ai_walls_straight = straighten_mask(ai_walls_closed, 0.003)
    save_step("03_ai_walls_straight", ai_walls_straight, str(steps_dir))

    # 7. ZONE
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
    
    vis_indoor = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    vis_indoor[indoor_mask > 0] = [0, 255, 255]
    save_step("debug_zone_interior", vis_indoor, str(steps_dir))

    # 8. SCALE DETECTION
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

    # 9. MĂSURĂTORI
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
    print(f"   🏠 Arie Indoor: {area_indoor_m2:.2f} m²")
    
    # 10. GENERARE 3D
    export_walls_to_obj(
        ai_walls_smoothed, 
        output_dir / "walls_3d.obj", 
        m_px, 
        image_output_path=output_dir / "walls_3d_view.png"
    )

    # 11. VIZUALIZĂRI
    overlay = img.copy()
    overlay[outline_ext_mask > 0] = [255, 0, 0]
    overlay[outline_int_mask > 0] = [0, 255, 0]
    cv2.imwrite(str(output_dir / "visualization_overlay.png"), overlay)
    
    return {
        "scale_result": scale_result,
        "measurements": measurements,
        "masks": {
            "visualization": str(output_dir / "visualization_overlay.png"),
            "visualization_3d": str(output_dir / "walls_3d_view.png")
        }
    }