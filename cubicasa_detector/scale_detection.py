# file: engine/cubicasa_detector/scale_detection.py
"""
Module pentru detectarea scalei din planuri.
Conține funcții pentru apelul Gemini API și detectarea scalei din label-uri de camere.
"""

from __future__ import annotations

import json
import math
import base64
import requests
import cv2
import numpy as np
from pathlib import Path


GEMINI_PROMPT_CROP = """
Analizează această imagine de plan de casă și identifică:
1. Numele camerei/camerelor (ex: "Living Room", "Bedroom", "Kitchen", etc.)
2. Suprafața camerei/camerelor în metri pătrați (m²) - DOAR dacă este explicit indicată în imagine

REGULI STRICTE PENTRU EXTRAGEREA SUPRAFEȚEI:

⚠️ CRITICAL: Extrage DOAR valori care sunt EXPLICIT etichetate ca metri pătrați (m²).
   - Caută texte precum: "15.5 m²", "20 m²", "12.3 m²", "25.7 m²", etc.
   - Caută simbolul m² sau textul "m²" sau "m2" sau "sqm" sau "sq m" sau "m^2"
   - Valorile trebuie să fie asociate explicit cu unitatea de măsură m²

❌ NU extrage:
   - Numere care NU au unitatea m² explicită (ex: "15.5", "20", "12.3" fără m²)
   - Numere din dimensiuni (ex: "3.5 x 4.2" - acestea sunt lungimi, nu suprafețe)
   - Numere din scale (ex: "1:100", "1:50")
   - Numere din coordonate sau adrese
   - Orice alte numere care nu sunt explicit etichetate ca metri pătrați

IMPORTANT: 
- În această imagine pot exista una sau mai multe camere (de exemplu, în cazul unui open space pot exista 2 sau mai multe camere cu suprafețe separate).
- Zonele negre din imagine NU fac parte din cameră - acoperă doar ce nu este relevant.
- Dacă sunt mai multe camere cu suprafețe etichetate, adună DOAR cele care au explicit m² și returnează suma totală.
- Dacă NICI O cameră nu are suprafață explicit etichetată cu m², returnează null pentru area_m2.

Returnează JSON cu formatul:
{
  "room_name": "numele camerei sau 'Multiple rooms' dacă sunt mai multe",
  "area_m2": 15.5
}

Dacă nu găsești NICI O valoare explicit etichetată cu m², returnează null pentru area_m2.
"""

GEMINI_PROMPT_TOTAL_SUM = """
Analizează această imagine de plan de casă și identifică suma totală a suprafețelor tuturor camerelor (în metri pătrați, m²).

Returnează JSON cu formatul:
{
  "total_sum_m2": 120.5
}
Dacă nu găsești suma totală, returnează null.
"""


def call_gemini(image_path, prompt, api_key):
    """API REST Direct (v1beta) pentru Gemini."""
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
        
        # Gemini poate răspunde cu markdown code fences (```json ... ```), sau chiar cu un
        # bloc care nu conține un obiect JSON. Încercăm să extragem robust JSON-ul.
        if text.startswith("```"):
            lines = text.splitlines()
            # eliminăm prima linie (``` sau ```json) și ultima linie (```) dacă există
            if lines and lines[0].lstrip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Dacă avem extra text, extragem primul obiect JSON { ... }
        start = text.find("{")
        end = text.rfind("}") + 1
        candidate = text
        if start != -1 and end > start:
            candidate = text[start:end]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Evităm să spargem pipeline-ul pentru răspunsuri non-JSON (rate-limit / refuz / text liber).
            preview = " ".join(text.split())[:160]
            print(f"⚠️  Gemini returned non-JSON: {preview!r}")
            return None
        
    except Exception as e:
        print(f"⚠️  Gemini error: {e}")
        return None


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


def save_step(name, img, steps_dir):
    """Salvează un step de debug."""
    if steps_dir:
        path = Path(steps_dir) / f"{name}.png"
        cv2.imwrite(str(path), img)


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
            # Verificăm dacă coordonatele sunt în limitele valide
            if y >= unvisited_mask.shape[0] or x >= unvisited_mask.shape[1]:
                continue
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
                
                if res and 'area_m2' in res and res['area_m2'] is not None:
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
