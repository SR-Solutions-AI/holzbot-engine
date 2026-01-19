# file: engine/runner/segmenter/clusters.py
from __future__ import annotations

import math
import os
import io
import base64
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

from .common import STEP_DIRS, save_debug, resize_bgr_max_side, get_output_dir
from .classifier import setup_gemini_client

load_dotenv()


def _is_overlapping(box1: list[int], box2: list[int], threshold: float = 0.60) -> bool:
    """
    Verifică dacă două cutii se suprapun semnificativ.
    Folosește Intersection over Minimum Area pentru a detecta
    dacă una este conținută în cealaltă sau dacă sunt aproape identice.
    """
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2

    # Coordonatele intersecției
    ix1 = max(x1, xx1)
    iy1 = max(y1, yy1)
    ix2 = min(x2, xx2)
    iy2 = min(y2, yy2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return False

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (xx2 - xx1) * (yy2 - yy1)
    
    # Raportul față de cea mai mică arie (detectează containment)
    min_area = min(area1, area2)
    if min_area == 0: return False

    overlap_ratio = inter_area / min_area
    return overlap_ratio > threshold


def _draw_debug_boxes(img: np.ndarray, boxes: list[list[int]], color: tuple[int, int, int], label_prefix: str = "") -> np.ndarray:
    vis = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{label_prefix}{i}"
        cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def _draw_detailed_fallback_debug(img: np.ndarray, all_candidates: list[list[int]], accepted_indices: set[int], ref_index: int, parent_box: list[int]) -> np.ndarray:
    """
    Vizualizare Fallback:
    - GRI: Părintele (ignorat).
    - ALBASTRU (CYAN): REFERINȚA REALĂ (Cel mai mare copil valid).
    - VERDE: Acceptat.
    - ROȘU: Respins.
    """
    vis = img.copy()
    
    # 1. Desenăm Părintele (doar contur, pentru context)
    px1, py1, px2, py2 = parent_box
    cv2.rectangle(vis, (px1, py1), (px2, py2), (100, 100, 100), 1)
    cv2.putText(vis, "PARENT (IGNORED)", (px1+10, py1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    # 2. Desenăm elementele respinse
    for i, box in enumerate(all_candidates):
        if i == ref_index or i in accepted_indices: continue
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1) # Roșu

    # 3. Desenăm elementele acceptate
    for i in accepted_indices:
        if i == ref_index: continue
        x1, y1, x2, y2 = all_candidates[i]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2) # Verde
        cv2.putText(vis, f"OK {i}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 4. Desenăm REFERINȚA
    if ref_index >= 0:
        x1, y1, x2, y2 = all_candidates[ref_index]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 4) # Cyan gros
        label = "REFERINTA (MAX CHILD)"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(vis, (x1, y1 - 25), (x1 + w, y1), (255, 255, 0), -1)
        cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return vis


def split_large_cluster(region: np.ndarray, x1: int, y1: int, idx: int) -> list[list[int]]:
    h, w = region.shape
    area = h * w
    if area < 30000:
        return [[x1, y1, x1 + w, y1 + h]]

    col_sum = np.sum(region > 0, axis=0)
    row_sum = np.sum(region > 0, axis=1)
    col_smooth = cv2.GaussianBlur(col_sum.astype(np.float32), (51, 1), 0)
    row_smooth = cv2.GaussianBlur(row_sum.astype(np.float32), (1, 51), 0)
    col_norm = col_smooth / (np.max(col_smooth) + 1e-5)
    row_norm = row_smooth / (np.max(row_smooth) + 1e-5)

    col_split = np.where(col_norm < 0.10)[0]
    row_split = np.where(row_norm < 0.10)[0]

    boxes: list[list[int]] = []

    if len(col_split) > 0:
        gaps = np.diff(col_split)
        big_gaps = np.where(gaps > 50)[0]
        if len(big_gaps) > 0:
            mid = int(np.median(col_split))
            if 0.3 * w < mid < 0.7 * w:
                save_debug(region, STEP_DIRS["clusters"]["split"], f"split_col_{idx}.jpg")
                for part, offset in [(region[:, :mid], 0), (region[:, mid:], mid)]:
                    num, _, stats, _ = cv2.connectedComponentsWithStats(part, 8)
                    for x, y, ww, hh, a in stats[1:]:
                        if a > 0.02 * area:
                            boxes.append([x1 + offset + x, y1 + y, x1 + offset + x + ww, y1 + y + hh])
                return boxes

    if len(row_split) > 0:
        gaps = np.diff(row_split)
        big_gaps = np.where(gaps > 50)[0]
        if len(big_gaps) > 0:
            mid = int(np.median(row_split))
            if 0.3 * h < mid < 0.7 * h:
                save_debug(region, STEP_DIRS["clusters"]["split"], f"split_row_{idx}.jpg")
                for part, offset in [(region[:mid, :], 0), (region[mid:, :], mid)]:
                    num, _, stats, _ = cv2.connectedComponentsWithStats(part, 8)
                    for x, y, ww, hh, a in stats[1:]:
                        if a > 0.02 * area:
                            boxes.append([x1 + x, y1 + offset + y, x1 + x + ww, y1 + offset + y + hh])
                return boxes

    return [[x1, y1, x1 + w, y1 + h]]


def merge_overlapping_boxes(boxes: list[list[int]], shape: tuple[int, int]) -> list[list[int]]:
    h, w = shape[:2]
    diag = math.hypot(h, w)
    prox = 0.005 * diag

    merged = True
    while merged:
        merged = False
        new_boxes: list[list[int]] = []
        while boxes:
            x1, y1, x2, y2 = boxes.pop(0)
            mbox = [x1, y1, x2, y2]
            keep: list[list[int]] = []

            for (xx1, yy1, xx2, yy2) in boxes:
                inter_x1, inter_y1 = max(x1, xx1), max(y1, yy1)
                inter_x2, inter_y2 = min(x2, xx2), min(y2, yy2)
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

                area1 = (x2 - x1) * (y2 - y1)
                area2 = (xx2 - xx1) * (yy2 - yy1)
                smaller_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0

                dx = max(0, max(x1 - xx2, xx1 - x2))
                dy = max(0, max(y1 - yy2, yy1 - y2))
                dist = math.hypot(dx, dy)

                if inter_area > 0 or (dist <= prox and smaller_ratio < 0.3):
                    mbox = [
                        min(mbox[0], xx1),
                        min(mbox[1], yy1),
                        max(mbox[2], xx2),
                        max(mbox[3], yy2),
                    ]
                    merged = True
                else:
                    keep.append([xx1, yy1, xx2, yy2])

            boxes = keep
            new_boxes.append(mbox)
        boxes = new_boxes

    return boxes


def expand_cluster(mask: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> list[int]:
    h, w = mask.shape
    while True:
        expanded = False
        if y1 > 0 and np.any(mask[y1 - 1, x1:x2] == 255):
            y1 -= 1
            expanded = True
        if y2 < h and np.any(mask[y2 - 1, x1:x2] == 255):
            y2 += 1
            expanded = True
        if x1 > 0 and np.any(mask[y1:y2, x1 - 1] == 255):
            x1 -= 1
            expanded = True
        if x2 < w and np.any(mask[y1:y2, x2 - 1] == 255):
            x2 += 1
            expanded = True
        if not expanded:
            break
    return [x1, y1, x2, y2]


def _bgr_to_base64_png(bgr_img: np.ndarray, max_size: int = 2000) -> str: 
    h, w = bgr_img.shape[:2]
    
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        bgr_img = cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95) 
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _ask_ai_how_many_plans(orig_img: np.ndarray) -> int:
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("  ⚠️  [AI] OPENAI_API_KEY lipsă - returnez 1 (default)")
        return 1
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
    except Exception as e:
        print(f"  ⚠️  [AI] Eroare inițializare client OpenAI: {e}")
        return 1
    
    print("  🤖 [AI] Trimit imaginea pentru numărare planuri...")
    b64_img = _bgr_to_base64_png(orig_img)
    
    prompt = """You are an architectural floor plan analyst.
Look at this image and count how many DISTINCT house floor plans you can see.
IMPORTANT:
- If you see multiple floor plans arranged on the same page, count each one.
- If you see just ONE floor plan (even with borders), return 1.
- If you see a site plan, elevation view, or text-heavy page, return 1.
Return ONLY a number (1, 2, 3...). No text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                        }
                    ]
                }
            ],
            max_tokens=10,
            temperature=0.0
        )
        
        answer = response.choices[0].message.content.strip()
        print(f"  💬 [AI] Răspuns brut: '{answer}'")
        
        import re
        match = re.search(r'\d+', answer)
        if match:
            count = int(match.group())
            print(f"  ✅ [AI] Detectat: {count} planuri.")
            return count
        else:
            print(f"  ⚠️  [AI] Nu am găsit un număr în răspuns. Returnez 1.")
            return 1
            
    except Exception as e:
        print(f"  ⚠️  [AI] Eroare la apelare API: {e}")
        return 1


def _detect_frame_geometric(orig_img: np.ndarray, largest_cluster_box: list[int]) -> bool:
    """
    ✅ FUNCȚIE NOUĂ: Detectare geometrică a frame-ului fix de la marginile imaginii.
    Verifică dacă cluster-ul mare este un frame fix la marginile absolute ale imaginii.
    
    Args:
        orig_img: Imaginea originală (BGR)
        largest_cluster_box: [x1, y1, x2, y2] pentru cel mai mare cluster
    
    Returns:
        True dacă detectăm un frame fix la marginile imaginii, False altfel.
    """
    h, w = orig_img.shape[:2]
    x1, y1, x2, y2 = largest_cluster_box
    
    # Convertim la grayscale pentru analiză
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY) if len(orig_img.shape) == 3 else orig_img
    
    # Threshold pentru a obține o imagine binară
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Marginile absolute ale imaginii (primele/ultimele N pixeli)
    edge_threshold = max(10, min(w, h) // 100)  # 1% din dimensiunea minimă, minim 10 pixeli
    
    # Verificare 1: Cluster-ul este foarte aproape de marginile imaginii?
    margin_tolerance = max(5, min(w, h) // 200)  # 0.5% din dimensiunea minimă, minim 5 pixeli
    
    near_top = y1 <= margin_tolerance
    near_bottom = y2 >= (h - margin_tolerance)
    near_left = x1 <= margin_tolerance
    near_right = x2 >= (w - margin_tolerance)
    
    # Verificare 2: Cluster-ul ocupă aproape toată imaginea?
    width_ratio = (x2 - x1) / w
    height_ratio = (y2 - y1) / h
    
    # Verificare 3: Există linii dense de pixeli la marginile absolute?
    # Verificăm primele/ultimele edge_threshold pixeli de la fiecare margine
    top_edge = binary[0:edge_threshold, :]
    bottom_edge = binary[h-edge_threshold:h, :]
    left_edge = binary[:, 0:edge_threshold]
    right_edge = binary[:, w-edge_threshold:w]
    
    # Calculăm densitatea de pixeli negri (contururi) la fiecare margine
    top_density = np.sum(top_edge == 0) / (edge_threshold * w)
    bottom_density = np.sum(bottom_edge == 0) / (edge_threshold * w)
    left_density = np.sum(left_edge == 0) / (edge_threshold * h)
    right_density = np.sum(right_edge == 0) / (edge_threshold * h)
    
    # Dacă toate marginile au densitate mare (>30%), probabil e un frame
    edge_density_threshold = 0.30
    all_edges_dense = (
        top_density > edge_density_threshold and
        bottom_density > edge_density_threshold and
        left_density > edge_density_threshold and
        right_density > edge_density_threshold
    )
    
    # Verificare 4: Cluster-ul formează un dreptunghi aproape complet la marginile imaginii?
    is_near_all_edges = near_top and near_bottom and near_left and near_right
    is_almost_full_size = width_ratio > 0.95 and height_ratio > 0.95
    
    # Logging detaliat pentru debugging
    print(f"   🔍 [GEOMETRIC] Verificări:")
    print(f"      Cluster box: [{x1}, {y1}, {x2}, {y2}] din [{w}x{h}]")
    print(f"      Margin tolerance: {margin_tolerance}px")
    print(f"      Aproape de margini: top={near_top} (y1={y1}), bottom={near_bottom} (y2={y2}, h={h}), left={near_left} (x1={x1}), right={near_right} (x2={x2}, w={w})")
    print(f"      Dimensiuni: {width_ratio*100:.1f}% x {height_ratio*100:.1f}% din imagine")
    print(f"      Densități margini: top={top_density:.2f}, bottom={bottom_density:.2f}, left={left_density:.2f}, right={right_density:.2f}")
    print(f"      Condiții: is_near_all_edges={is_near_all_edges}, is_almost_full_size={is_almost_full_size}, all_edges_dense={all_edges_dense}")
    
    # Dacă toate condițiile sunt îndeplinite, probabil e un frame
    if is_near_all_edges and is_almost_full_size:
        if all_edges_dense:
            print(f"   ✅ [GEOMETRIC] Detectat frame fix la marginile imaginii!")
            return True
        else:
            print(f"   ⚠️  [GEOMETRIC] Cluster mare dar densitățile marginilor nu confirmă frame clar")
            return False
    
    # Dacă cluster-ul ocupă aproape toată imaginea dar nu este aproape de toate marginile,
    # sau dacă este aproape de toate marginile dar densitățile nu confirmă, totuși poate fi un frame
    if is_almost_full_size and (is_near_all_edges or (near_top and near_bottom) or (near_left and near_right)):
        # Relaxăm condițiile: dacă cel puțin 2 margini opuse au densitate mare, probabil e frame
        opposite_edges_dense = (
            (top_density > edge_density_threshold and bottom_density > edge_density_threshold) or
            (left_density > edge_density_threshold and right_density > edge_density_threshold)
        )
        
        if opposite_edges_dense:
            print(f"   ✅ [GEOMETRIC] Detectat frame (condiții relaxate: margini opuse dense)")
            return True
    
    return False


def _ask_ai_if_has_frame(orig_img: np.ndarray) -> bool:
    """
    ✅ FUNCȚIE NOUĂ: Întreabă AI-ul (Gemini sau ChatGPT) dacă imaginea conține un frame/cadru în jurul planului.
    Încearcă mai întâi Gemini (mai bun), apoi ChatGPT ca fallback.
    Returnează True dacă există frame, False altfel.
    """
    # Prompt simplificat și mai neutru pentru a evita blocarea de safety
    prompt = """Analyze this architectural floor plan image. Check if there is a rectangular border line at the absolute edges of the image (top, bottom, left, right edges).

A border frame is a thin rectangular line that forms a complete outline around the entire image, touching or very close to all four edges.

Answer ONLY: YES or NO"""
    
    # ========== ÎNCERCARE 1: GEMINI (mai bun) ==========
    # Creăm un model nou cu safety settings mai permisive direct aici
    try:
        import google.generativeai as genai
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        
        if api_key:
            genai.configure(api_key=api_key)
            
            # Safety settings foarte permisive pentru a evita blocarea
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
            
            # Creăm modelul direct cu safety settings
            models_to_try = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']
            gemini_model = None
            
            for model_name in models_to_try:
                try:
                    gemini_model = genai.GenerativeModel(model_name, safety_settings=safety_settings)
                    break
                except Exception:
                    continue
            
            if gemini_model:
                try:
                    # Convertim numpy array (BGR) la PIL Image (RGB)
                    rgb_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_img)
                    
                    # Resize dacă e prea mare (pentru a evita erori Gemini)
                    w, h = pil_img.size
                    max_size = 2000
                    if max(w, h) > max_size:
                        scale = max_size / max(w, h)
                        new_w = max(1, int(w * scale))
                        new_h = max(1, int(h * scale))
                        # Folosim LANCZOS (nu LANCZOS4 care nu există)
                        try:
                            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        except AttributeError:
                            # Fallback pentru versiuni mai vechi de PIL
                            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                    
                    print("  🤖 [AI] Trimit imaginea către Gemini pentru verificare frame...")
                    
                    # Apelăm cu safety_settings suplimentare în generate_content pentru siguranță
                    response = gemini_model.generate_content(
                        [prompt, pil_img],
                        generation_config={
                            "temperature": 0.0,
                            "max_output_tokens": 10,
                        },
                        safety_settings=safety_settings  # Re-iterăm safety settings aici
                    )
                    
                    # Verificăm finish_reason pentru a vedea dacă răspunsul a fost blocat
                    finish_reason = None
                    if hasattr(response, 'candidates') and response.candidates:
                        finish_reason = response.candidates[0].finish_reason
                        print(f"  🔍 [AI] Gemini finish reason: {finish_reason}")
                        
                        # finish_reason=2 înseamnă SAFETY (blocat din motive de siguranță)
                        if finish_reason == 2:
                            print(f"  ⚠️  [AI] Gemini a blocat răspunsul (SAFETY). Încerc ChatGPT...")
                        else:
                            # Încercăm să obținem răspunsul de la Gemini
                            answer = None
                            
                            if response.parts:
                                try:
                                    answer = response.text.strip().upper()
                                except Exception as e:
                                    print(f"  ⚠️  [AI] Eroare la accesare response.text: {e}")
                            
                            # Fallback: încercăm să accesăm direct text-ul
                            if not answer:
                                try:
                                    if hasattr(response, 'text') and response.text:
                                        answer = response.text.strip().upper()
                                except Exception as e:
                                    print(f"  ⚠️  [AI] Eroare la fallback text access: {e}")
                            
                            if answer:
                                print(f"  💬 [AI] Gemini răspuns: '{answer}'")
                                
                                has_frame = "YES" in answer or "DA" in answer
                                print(f"  ✅ [AI] Gemini detectat frame: {has_frame}")
                                return has_frame
                    
                    # Dacă ajungem aici, Gemini nu a dat răspuns valid
                    if finish_reason != 2:
                        print(f"  ⚠️  [AI] Gemini nu a returnat răspuns valid. Încerc ChatGPT...")
                        
                except Exception as e:
                    import traceback
                    print(f"  ⚠️  [AI] Eroare la apelare Gemini: {e}")
                    print(f"  ⚠️  [AI] Traceback: {traceback.format_exc()}")
                    print(f"  ⚠️  [AI] Încerc ChatGPT ca fallback...")
            else:
                print("  ⚠️  [AI] Nu s-a putut crea model Gemini. Încerc ChatGPT...")
        else:
            print("  ⚠️  [AI] GEMINI_API_KEY lipsă. Încerc ChatGPT...")
    except Exception as e:
        import traceback
        print(f"  ⚠️  [AI] Eroare inițializare Gemini: {e}")
        print(f"  ⚠️  [AI] Traceback: {traceback.format_exc()}")
        print(f"  ⚠️  [AI] Încerc ChatGPT ca fallback...")
    
    # ========== ÎNCERCARE 2: CHATGPT (fallback) ==========
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("  ⚠️  [AI] OPENAI_API_KEY lipsă - returnez False (nu eliminăm cluster)")
        return False
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
    except Exception as e:
        print(f"  ⚠️  [AI] Eroare inițializare client OpenAI: {e}")
        return False
    
    try:
        print("  🤖 [AI] Trimit imaginea către ChatGPT pentru verificare frame...")
        
        b64_img = _bgr_to_base64_png(orig_img)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                        }
                    ]
                }
            ],
            max_tokens=10,
            temperature=0.0
        )
        
        if response.choices and response.choices[0].message.content:
            answer = response.choices[0].message.content.strip().upper()
            print(f"  💬 [AI] ChatGPT răspuns: '{answer}'")
            
            has_frame = "YES" in answer or "DA" in answer
            print(f"  ✅ [AI] ChatGPT detectat frame: {has_frame}")
            return has_frame
        else:
            print(f"  ⚠️  [AI] ChatGPT nu a returnat răspuns valid. Returnez False (nu eliminăm cluster).")
            return False
            
    except Exception as e:
        import traceback
        print(f"  ⚠️  [AI] Eroare la apelare ChatGPT: {e}")
        print(f"  ⚠️  [AI] Traceback: {traceback.format_exc()}")
        return False


def _ask_ai_how_many_buildings_in_cluster(crop_img: np.ndarray) -> int:
    """
    ✅ FUNCȚIE NOUĂ: Întreabă AI-ul câte clădiri separate există într-un cluster.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("     [AI] OPENAI_API_KEY lipsă - returnez 1 (default)")
        return 1
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
    except Exception as e:
        print(f"     [AI] Eroare inițializare: {e}")
        return 1
    
    b64_img = _bgr_to_base64_png(crop_img)
    
    prompt = """You are an architectural analyst.
Look at this floor plan cluster image.
Count how many SEPARATE, DISTINCT BUILDINGS or HOUSE PLANS you can see.
IMPORTANT:
- If you see multiple buildings/houses side by side, count each one.
- If you see one building with multiple floors stacked vertically, that's still ONE building.
- If you see just one house/building, return 1.
Return ONLY a number (1, 2, 3...). No text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                        }
                    ]
                }
            ],
            max_tokens=10,
            temperature=0.0
        )
        
        answer = response.choices[0].message.content.strip()
        
        import re
        match = re.search(r'\d+', answer)
        if match:
            count = int(match.group())
            print(f"     [AI] Detectat {count} clădiri în cluster.")
            return count
        else:
            return 1
            
    except Exception as e:
        print(f"     [AI] Eroare: {e}")
        return 1


def _split_cluster_by_buildings(crop: np.ndarray, offset_x: int, offset_y: int, expected_count: int) -> list[list[int]]:
    """
    ✅ FUNCȚIE NOUĂ: Împarte un cluster în sub-zone bazat pe analiza geometrică.
    Încearcă să găsească gap-uri verticale sau orizontale pentru a separa clădirile.
    """
    print(f"     🔧 Încerc să împart cluster-ul în {expected_count} părți...")
    
    # Convertim la grayscale și binarizăm
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    h, w = binary.shape
    
    # Analizăm densitatea pe axe
    col_density = np.sum(binary > 0, axis=0)
    row_density = np.sum(binary > 0, axis=1)
    
    # Smoothing
    col_smooth = cv2.GaussianBlur(col_density.astype(np.float32).reshape(-1, 1), (31, 1), 0).flatten()
    row_smooth = cv2.GaussianBlur(row_density.astype(np.float32).reshape(1, -1), (1, 31), 0).flatten()
    
    # Normalizare
    col_norm = col_smooth / (np.max(col_smooth) + 1e-5)
    row_norm = row_smooth / (np.max(row_smooth) + 1e-5)
    
    # Detectăm gap-uri (zone cu densitate < 5%)
    col_gaps = np.where(col_norm < 0.05)[0]
    row_gaps = np.where(row_norm < 0.05)[0]
    
    boxes = []
    
    # Încercăm split vertical (dacă avem clădiri side-by-side)
    if len(col_gaps) > w * 0.05:  # Minim 5% din lățime e gol
        # Grupăm gap-urile continue
        gap_groups = []
        if len(col_gaps) > 0:
            current_group = [col_gaps[0]]
            for i in range(1, len(col_gaps)):
                if col_gaps[i] - col_gaps[i-1] <= 5:  # Gap continuu
                    current_group.append(col_gaps[i])
                else:
                    if len(current_group) > 20:  # Gap suficient de larg
                        gap_groups.append(current_group)
                    current_group = [col_gaps[i]]
            if len(current_group) > 20:
                gap_groups.append(current_group)
        
        if len(gap_groups) >= expected_count - 1:
            print(f"     ✅ Găsit {len(gap_groups)} gap-uri verticale.")
            # Împărțim pe verticală
            split_positions = [int(np.median(g)) for g in gap_groups[:expected_count-1]]
            split_positions = [0] + sorted(split_positions) + [w]
            
            for i in range(len(split_positions) - 1):
                x_start = split_positions[i]
                x_end = split_positions[i + 1]
                
                # Găsim zona non-goală pe această bandă
                band = binary[:, x_start:x_end]
                coords = np.where(band > 0)
                if coords[0].size > 0:
                    y_min, y_max = coords[0].min(), coords[0].max()
                    boxes.append([
                        offset_x + x_start,
                        offset_y + y_min,
                        offset_x + x_end,
                        offset_y + y_max
                    ])
            
            if len(boxes) > 1:
                return boxes
    
    # Încercăm split orizontal (dacă avem etaje stacked)
    if len(row_gaps) > h * 0.05:
        gap_groups = []
        if len(row_gaps) > 0:
            current_group = [row_gaps[0]]
            for i in range(1, len(row_gaps)):
                if row_gaps[i] - row_gaps[i-1] <= 5:
                    current_group.append(row_gaps[i])
                else:
                    if len(current_group) > 20:
                        gap_groups.append(current_group)
                    current_group = [row_gaps[i]]
            if len(current_group) > 20:
                gap_groups.append(current_group)
        
        if len(gap_groups) >= expected_count - 1:
            print(f"     ✅ Găsit {len(gap_groups)} gap-uri orizontale.")
            split_positions = [int(np.median(g)) for g in gap_groups[:expected_count-1]]
            split_positions = [0] + sorted(split_positions) + [h]
            
            for i in range(len(split_positions) - 1):
                y_start = split_positions[i]
                y_end = split_positions[i + 1]
                
                band = binary[y_start:y_end, :]
                coords = np.where(band > 0)
                if coords[1].size > 0:
                    x_min, x_max = coords[1].min(), coords[1].max()
                    boxes.append([
                        offset_x + x_min,
                        offset_y + y_start,
                        offset_x + x_max,
                        offset_y + y_end
                    ])
            
            if len(boxes) > 1:
                return boxes
    
    print(f"     ⚠️  Nu am găsit gap-uri clare. Nu pot împărți.")
    return []


def detect_clusters(mask: np.ndarray, orig: np.ndarray, return_boxes: bool = False) -> list[str] | list[list[int]]:
    print("\n========================================================")
    print("[STEP 7] START Detectare clustere")
    print("========================================================")
    
    # 1. Pre-procesare
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask.copy()
    inv = cv2.bitwise_not(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(cv2.dilate(inv, kernel), cv2.MORPH_OPEN, kernel)
    save_debug(clean, STEP_DIRS["clusters"]["initial"], "mask_clean.jpg")

    # 2. Raw Boxes
    num, _, stats, _ = cv2.connectedComponentsWithStats(clean, 8)
    boxes = [[x, y, x + bw, y + bh] for x, y, bw, bh, a in stats[1:] if a > 200]
    
    print(f"🔸 [GEO] 1. Cutii brute (Raw Boxes): {len(boxes)}")
    debug_raw = _draw_debug_boxes(orig, boxes, (0, 0, 255), "Raw_")
    save_debug(debug_raw, STEP_DIRS["clusters"]["initial"], "debug_1_raw_boxes.jpg")

    # 3. Refined (Split)
    refined: list[list[int]] = []
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        reg = clean[y1:y2, x1:x2]
        if reg.size == 0: continue
        sub_boxes = split_large_cluster(reg, x1, y1, i)
        for sb in sub_boxes:
            refined.append(expand_cluster(clean, *sb))
    
    print(f"🔸 [GEO] 2. Cutii rafinate (Refined): {len(refined)}")
    debug_ref = _draw_debug_boxes(orig, refined, (255, 128, 0), "Ref_")
    save_debug(debug_ref, STEP_DIRS["clusters"]["initial"], "debug_2_refined_boxes.jpg")

    # 4. Merged
    merged = merge_overlapping_boxes(refined, clean.shape)
    print(f"🔸 [GEO] 3. Cutii unite (Merged): {len(merged)}")
    debug_mrg = _draw_debug_boxes(orig, merged, (0, 255, 255), "Mrg_")
    save_debug(debug_mrg, STEP_DIRS["clusters"]["initial"], "debug_3_merged_boxes.jpg")

    
    # 5. LOGICĂ DECIZIONALĂ (AI) - Doar dacă avem 1 singur cluster
    filtered: list[list[int]] = []

    print(f"\n--- [DECISION] Analizez rezultatul Merge ({len(merged)} clustere) ---")

    if len(merged) == 1:
        print(" 🔍 OpenCV a găsit 1 singur cluster. Întreb AI-ul pentru validare...")
        ai_plan_count = _ask_ai_how_many_plans(orig)

        if ai_plan_count > 1:
            print(f" ⚠️  [CONFLICT] AI vede {ai_plan_count} planuri, dar geometric am găsit 1.")
            print(" -> [ACTION] Ignor clusterul Părinte și activez FALLBACK (recuperez copiii).")
            
            parent_box = merged[0]
            parent_area = (parent_box[2]-parent_box[0]) * (parent_box[3]-parent_box[1])
            print(f" -> [DEBUG] Aria Părintelui (de ignorat): {int(parent_area)} px")

            source_candidates = refined if len(refined) > 1 else boxes
            print(f" -> [DEBUG] Sursă fallback: {'Refined' if source_candidates == refined else 'Boxes'} ({len(source_candidates)} elemente)")
            
            candidates_with_area = []
            for i, cand in enumerate(source_candidates):
                c_w = cand[2] - cand[0]
                c_h = cand[3] - cand[1]
                c_area = c_w * c_h
                candidates_with_area.append((i, c_area))
            
            candidates_with_area.sort(key=lambda x: x[1], reverse=True)
            
            if not candidates_with_area:
                print(" ❌ [FAIL] Nu am găsit niciun copil în fallback. Revin la merged.")
                filtered = merged
            else:
                ref_idx = -1
                max_child_area = 0
                
                for idx, area in candidates_with_area:
                    if area > 0.90 * parent_area:
                        print(f"    [SKIP Ref] Candidat {idx} e prea mare ({int(area)} px), probabil e un border.")
                        continue
                    
                    ref_idx = idx
                    max_child_area = area
                    break
                
                if ref_idx == -1:
                    print("    [WARN] Toți candidații erau imenși. Iau cel mai mare disponibil.")
                    ref_idx = candidates_with_area[0][0]
                    max_child_area = candidates_with_area[0][1]

                print(f" -> [LOGIC] Referința Aleasă (Cel mai mare Copil): Index {ref_idx}, Aria: {int(max_child_area)} px")
                
                child_threshold = 0.10 * max_child_area
                print(f" -> [LOGIC] Prag minim (10% din ref): {int(child_threshold)} px")

                accepted_indices = set()
                fallback_accepted_count = 0
                
                for i, area in candidates_with_area:
                    cand = source_candidates[i]
                    
                    if area > 0.90 * parent_area:
                        continue

                    if area < child_threshold:
                        continue

                    is_duplicate = False
                    for existing_idx in accepted_indices:
                        existing_box = source_candidates[existing_idx]
                        if _is_overlapping(cand, existing_box, threshold=0.60):
                            print(f"    [SKIP Overlap] Candidat {i} se suprapune cu {existing_idx}. Ignor.")
                            is_duplicate = True
                            break
                    
                    if is_duplicate:
                        continue

                    filtered.append(cand)
                    accepted_indices.add(i)
                    fallback_accepted_count += 1
                
                if fallback_accepted_count > 0:
                    print(f" ✅ [SUCCESS] Am recuperat {fallback_accepted_count} clustere valide și unice.")
                    
                    debug_fb_detailed = _draw_detailed_fallback_debug(orig, source_candidates, accepted_indices, ref_idx, parent_box)
                    save_debug(debug_fb_detailed, STEP_DIRS["clusters"]["initial"], "debug_4_detailed_fallback_logic.jpg")
                    print(" 📸 [DEBUG] Imagine detaliată salvată: debug_4_detailed_fallback_logic.jpg")

                else:
                    print(" ❌ [FAIL] Toți copiii au picat filtrul. Revin la 'Merged'.")
                    filtered = merged
        else:
            print(" ✅ [OK] AI confirmă: Este un singur plan. Păstrez clusterul 'Merged'.")
            filtered = merged
    else:
        print(f" ✅ [OK] OpenCV a găsit deja {len(merged)} clustere. Nu este nevoie de intervenția AI.")
        filtered = merged


    # 6. FILTRARE FINALĂ DIMENSIUNI
    if filtered:
        print("\n--- [FINAL] Filtrare relativă dimensiuni ---")
        
        areas_map = [] 
        for i, box in enumerate(filtered):
            a = (box[2] - box[0]) * (box[3] - box[1])
            areas_map.append((i, float(a)))
        
        areas_map.sort(key=lambda x: x[1], reverse=True)
        
        if len(areas_map) > 1:
            reference_area = areas_map[1][1]
            print(f" -> [LOGIC] Referință Finală: AL DOILEA cel mai mare: {int(reference_area)} px")
        else:
            reference_area = areas_map[0][1]
            print(f" -> [LOGIC] Referință Finală: Singurul element: {int(reference_area)} px")

        img_area = float(orig.shape[0] * orig.shape[1])

        MIN_REL = 0.10
        MIN_ABS = 0.0005
        min_allowed = max(MIN_REL * reference_area, MIN_ABS * img_area)
        
        print(f" -> Min Allowed Final: {int(min_allowed)} px")
        
        final_list = []
        for idx, area in areas_map:
            if area >= min_allowed:
                final_list.append(filtered[idx])
            else:
                print(f"    [DROP Final] Cluster {idx} area={int(area)} < {int(min_allowed)}")
        
        final_list.sort(key=lambda b: (b[1], b[0]))
        
        # Recalculăm areas_map pentru final_list (după sortare)
        final_areas_map = []
        for i, box in enumerate(final_list):
            a = (box[2] - box[0]) * (box[3] - box[1])
            final_areas_map.append((i, float(a)))
        final_areas_map.sort(key=lambda x: x[1], reverse=True)
        
        filtered = final_list
        
        # ✅ VERIFICARE FRAME: Dacă cel mai mare cluster ocupă >95% din imagine
        if len(final_list) > 0 and len(final_areas_map) > 0:
            largest_cluster_area = final_areas_map[0][1]  # Cel mai mare cluster
            largest_cluster_ratio = largest_cluster_area / img_area
            
            if largest_cluster_ratio > 0.95:
                print(f"\n--- [FRAME CHECK] Cel mai mare cluster ocupă {largest_cluster_ratio*100:.1f}% din imagine ---")
                print(f"   🔍 Verific dacă există frame în jurul planului...")
                
                # Găsim box-ul celui mai mare cluster folosind final_areas_map
                largest_idx = final_areas_map[0][0]  # Index în final_list
                largest_cluster_box = final_list[largest_idx]
                
                # Mai întâi încercăm detectarea geometrică (mai rapidă și mai precisă)
                has_frame_geometric = _detect_frame_geometric(orig, largest_cluster_box)
                
                if has_frame_geometric:
                    print(f"   ✅ [GEOMETRIC] Detectat frame fix la marginile imaginii!")
                    print(f"   ✂️  Fac crop la imagine (exclud frame-ul) și reapelez detectarea clusterelor...")
                    
                    # Facem crop la imagine la conținutul din interiorul cluster-ului (cu padding mic)
                    x1, y1, x2, y2 = largest_cluster_box
                    h, w = orig.shape[:2]
                    
                    # Adăugăm un padding mic (2% din dimensiune) pentru a nu tăia prea mult
                    padding_x = int((x2 - x1) * 0.02)
                    padding_y = int((y2 - y1) * 0.02)
                    
                    # Crop la imagine (excludem marginile cu frame)
                    crop_x1 = max(0, x1 + padding_x)
                    crop_y1 = max(0, y1 + padding_y)
                    crop_x2 = min(w, x2 - padding_x)
                    crop_y2 = min(h, y2 - padding_y)
                    
                    cropped_img = orig[crop_y1:crop_y2, crop_x1:crop_x2]
                    print(f"   📐 Crop: [{crop_x1}, {crop_y1}, {crop_x2}, {crop_y2}] din [{w}x{h}]")
                    print(f"   📐 Dimensiuni după crop: {cropped_img.shape[1]}x{cropped_img.shape[0]}")
                    
                    # Reapelează detectarea clusterelor pe imaginea croppată
                    print(f"   🔄 Reapelez detectarea clusterelor pe imaginea croppată...")
                    cropped_clusters = detect_clusters(
                        cropped_img,  # mask
                        cropped_img,  # orig
                        return_boxes=True  # Returnăm box-uri în loc de path-uri
                    )
                    
                    # Ajustăm coordonatele clusterelor detectate la coordonatele originale
                    adjusted_clusters = []
                    for cluster_box in cropped_clusters:
                        cx1, cy1, cx2, cy2 = cluster_box
                        # Adăugăm offset-ul crop-ului
                        adjusted_box = [
                            cx1 + crop_x1,
                            cy1 + crop_y1,
                            cx2 + crop_x1,
                            cy2 + crop_y1
                        ]
                        adjusted_clusters.append(adjusted_box)
                    
                    print(f"   ✅ Detectat {len(adjusted_clusters)} cluster-uri după eliminare frame")
                    filtered = adjusted_clusters
                else:
                    # Dacă detectarea geometrică nu confirmă, încercăm AI-ul
                    print(f"   🔍 [GEOMETRIC] Nu am detectat frame clar. Verific cu AI...")
                    has_frame_ai = _ask_ai_if_has_frame(orig)
                    
                    if has_frame_ai:
                        print(f"   ✅ AI confirmă: Există frame în jurul planului.")
                        print(f"   ✂️  Fac crop la imagine (exclud frame-ul) și reapelez detectarea clusterelor...")
                        
                        # Facem crop la imagine la conținutul din interiorul cluster-ului (cu padding mic)
                        x1, y1, x2, y2 = largest_cluster_box
                        h, w = orig.shape[:2]
                        
                        # Adăugăm un padding mic (2% din dimensiune) pentru a nu tăia prea mult
                        padding_x = int((x2 - x1) * 0.02)
                        padding_y = int((y2 - y1) * 0.02)
                        
                        # Crop la imagine (excludem marginile cu frame)
                        crop_x1 = max(0, x1 + padding_x)
                        crop_y1 = max(0, y1 + padding_y)
                        crop_x2 = min(w, x2 - padding_x)
                        crop_y2 = min(h, y2 - padding_y)
                        
                        cropped_img = orig[crop_y1:crop_y2, crop_x1:crop_x2]
                        print(f"   📐 Crop: [{crop_x1}, {crop_y1}, {crop_x2}, {crop_y2}] din [{w}x{h}]")
                        print(f"   📐 Dimensiuni după crop: {cropped_img.shape[1]}x{cropped_img.shape[0]}")
                        
                        # Reapelează detectarea clusterelor pe imaginea croppată
                        print(f"   🔄 Reapelez detectarea clusterelor pe imaginea croppată...")
                        cropped_clusters = detect_clusters(
                            cropped_img,  # mask
                            cropped_img,  # orig
                            return_boxes=True  # Returnăm box-uri în loc de path-uri
                        )
                        
                        # Ajustăm coordonatele clusterelor detectate la coordonatele originale
                        adjusted_clusters = []
                        for cluster_box in cropped_clusters:
                            cx1, cy1, cx2, cy2 = cluster_box
                            # Adăugăm offset-ul crop-ului
                            adjusted_box = [
                                cx1 + crop_x1,
                                cy1 + crop_y1,
                                cx2 + crop_x1,
                                cy2 + crop_y1
                            ]
                            adjusted_clusters.append(adjusted_box)
                        
                        print(f"   ✅ Detectat {len(adjusted_clusters)} cluster-uri după eliminare frame")
                        filtered = adjusted_clusters
                    else:
                        print(f"   ✅ AI confirmă: NU există frame. Păstrez toate cluster-urile.")
            else:
                print(f"   ℹ️  Cel mai mare cluster ocupă {largest_cluster_ratio*100:.1f}% (<95%). Nu verific frame.")

    # ✅ 7. VALIDARE FINALĂ AI - Verificăm fiecare cluster pentru clădiri multiple
    print("\n========================================================")
    print("[FINAL VALIDATION] Verificare AI pentru clădiri multiple")
    print("========================================================")
    
    final_validated_boxes = []
    
    for i, (x1, y1, x2, y2) in enumerate(filtered, 1):
        print(f"\n🔍 Cluster {i}/{len(filtered)}...")
        
        crop = orig[y1:y2, x1:x2]
        buildings_count = _ask_ai_how_many_buildings_in_cluster(crop)
        
        if buildings_count > 1:
            print(f"   ⚠️  AI: {buildings_count} clădiri separate!")
            print(f"   🔧 Activez sub-împărțire...")
            
            sub_boxes = _split_cluster_by_buildings(crop, x1, y1, buildings_count)
            
            if len(sub_boxes) > 1:
                print(f"   ✅ Împărțit în {len(sub_boxes)} sub-clustere.")
                final_validated_boxes.extend(sub_boxes)
            else:
                print(f"   ⚠️  Nu pot împărți geometric. Păstrez original.")
                final_validated_boxes.append([x1, y1, x2, y2])
        else:
            print(f"   ✅ AI confirmă: 1 clădire. OK.")
            final_validated_boxes.append([x1, y1, x2, y2])
    
    final_validated_boxes.sort(key=lambda b: (b[1], b[0]))
    
    print(f"\n📊 Rezultat: {len(filtered)} → {len(final_validated_boxes)} clustere finale")

    # 8. Output
    result = orig.copy()
    crop_paths: list[str] = []
    crops_dir = get_output_dir() / STEP_DIRS["clusters"]["crops"]
    crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Salvare {len(final_validated_boxes)} clustere validate...")

    for i, (x1, y1, x2, y2) in enumerate(final_validated_boxes, 1):
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result, str(i), (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        crop = orig[y1:y2, x1:x2]
        crop = resize_bgr_max_side(crop)

        crop_path = crops_dir / f"cluster_{i}.jpg"
        cv2.imwrite(str(crop_path), crop)
        crop_paths.append(str(crop_path))

    save_debug(result, STEP_DIRS["clusters"]["final"], "final_clusters_validated.jpg")
    print(f"✅ [DONE] {len(final_validated_boxes)} clustere validate returnate")

    if return_boxes:
        return final_validated_boxes
    return crop_paths


def detect_wall_zones(orig: np.ndarray, thick_mask: np.ndarray) -> list[str]:
    print("\n[STEP 6] Detectare zone pereți...")
    gray = (thick_mask / 255).astype(np.float32)
    dens = cv2.GaussianBlur(gray, (51, 51), 0)
    norm = cv2.normalize(dens, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, dense_mask = cv2.threshold(norm, 60, 255, cv2.THRESH_BINARY)

    filled = dense_mask.copy()
    flood = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), np.uint8)
    cv2.floodFill(filled, flood, (0, 0), 0)
    walls = cv2.bitwise_not(filled)

    save_debug(walls, STEP_DIRS["walls"], "filled_unified.jpg")
    crop_paths = detect_clusters(walls, orig)
    return crop_paths