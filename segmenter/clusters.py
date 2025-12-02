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


def _bgr_to_base64_png(bgr_img: np.ndarray, max_size: int = 1280) -> str:
    h, w = bgr_img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        bgr_img = cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
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
                            "image_url": {"url": f"data:image/png;base64,{b64_img}"}
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


def detect_clusters(mask: np.ndarray, orig: np.ndarray) -> list[str]:
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
            
            # Părintele pe care îl ignorăm
            parent_box = merged[0]
            parent_area = (parent_box[2]-parent_box[0]) * (parent_box[3]-parent_box[1])
            print(f" -> [DEBUG] Aria Părintelui (de ignorat): {int(parent_area)} px")

            # Sursă fallback: Refined sau Boxes
            source_candidates = refined if len(refined) > 1 else boxes
            print(f" -> [DEBUG] Sursă fallback: {'Refined' if source_candidates == refined else 'Boxes'} ({len(source_candidates)} elemente)")
            
            # --- DETERMINARE REFERINȚĂ CORECTĂ (Cel mai mare copil care NU e părintele) ---
            
            # 1. Mapăm index -> arie
            candidates_with_area = []
            for i, cand in enumerate(source_candidates):
                c_w = cand[2] - cand[0]
                c_h = cand[3] - cand[1]
                c_area = c_w * c_h
                candidates_with_area.append((i, c_area))
            
            # 2. Sortăm descrescător după arie (IMPORTANT: pentru a procesa de la mare la mic)
            candidates_with_area.sort(key=lambda x: x[1], reverse=True)
            
            if not candidates_with_area:
                print(" ❌ [FAIL] Nu am găsit niciun copil în fallback. Revin la merged.")
                filtered = merged
            else:
                # 3. Găsim REFERINȚA REALĂ
                # Ignorăm orice candidat care e aproape cât părintele (> 90%) - e probabil un chenar
                ref_idx = -1
                max_child_area = 0
                
                for idx, area in candidates_with_area:
                    # Dacă e > 90% din părinte, e tot un fel de părinte/border
                    if area > 0.90 * parent_area:
                        print(f"    [SKIP Ref] Candidat {idx} e prea mare ({int(area)} px), probabil e un border.")
                        continue
                    
                    # Am găsit primul care e mai mic => E cel mai mare plan real
                    ref_idx = idx
                    max_child_area = area
                    break
                
                if ref_idx == -1:
                    # Dacă toate erau cât părintele (ex: 5 chenare unul într-altul), luăm ultimul
                    print("    [WARN] Toți candidații erau imenși. Iau cel mai mare disponibil.")
                    ref_idx = candidates_with_area[0][0]
                    max_child_area = candidates_with_area[0][1]

                print(f" -> [LOGIC] Referința Aleasă (Cel mai mare Copil): Index {ref_idx}, Aria: {int(max_child_area)} px")
                
                # 4. Setăm pragul (10% din referință)
                child_threshold = 0.10 * max_child_area
                print(f" -> [LOGIC] Prag minim (10% din ref): {int(child_threshold)} px")

                accepted_indices = set()
                fallback_accepted_count = 0
                
                # 5. Filtrăm candidații (Sortați Descrescător)
                for i, area in candidates_with_area:
                    cand = source_candidates[i]
                    
                    # A. Ignorăm candidatul dacă e chiar cel mare pe care l-am sărit anterior (border-ul)
                    if area > 0.90 * parent_area:
                        continue

                    # B. Check Dimensiune Minimă
                    if area < child_threshold:
                        continue

                    # C. Check Suprapunere (NOU - Elimină duplicatele)
                    
                    is_duplicate = False
                    for existing_idx in accepted_indices:
                        existing_box = source_candidates[existing_idx]
                        if _is_overlapping(cand, existing_box, threshold=0.60):
                            print(f"    [SKIP Overlap] Candidat {i} se suprapune cu {existing_idx}. Ignor.")
                            is_duplicate = True
                            break
                    
                    if is_duplicate:
                        continue

                    # D. Dacă trece filtrele, îl acceptăm
                    filtered.append(cand)
                    accepted_indices.add(i)
                    fallback_accepted_count += 1
                
                if fallback_accepted_count > 0:
                    print(f" ✅ [SUCCESS] Am recuperat {fallback_accepted_count} clustere valide și unice.")
                    
                    # Desenăm debug
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


    # -------------------------------------------------------------
    # 6. FILTRARE FINALĂ DIMENSIUNI (Safety Net pentru cazul Multi-Cluster standard)
    # -------------------------------------------------------------
    if filtered:
        print("\n--- [FINAL] Filtrare relativă dimensiuni ---")
        
        areas_map = [] 
        for i, box in enumerate(filtered):
            a = (box[2] - box[0]) * (box[3] - box[1])
            areas_map.append((i, float(a)))
        
        areas_map.sort(key=lambda x: x[1], reverse=True)
        
        # Determinăm Referința FINALĂ
        if len(areas_map) > 1:
            reference_area = areas_map[1][1]
            print(f" -> [LOGIC] Referință Finală: AL DOILEA cel mai mare: {int(reference_area)} px")
        else:
            reference_area = areas_map[0][1]
            print(f" -> [LOGIC] Referință Finală: Singurul element: {int(reference_area)} px")

        img_area = float(orig.shape[0] * orig.shape[1])

        # Prag final
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
        
        # Sortare geometrică (stânga-dreapta, sus-jos)
        final_list.sort(key=lambda b: (b[1], b[0]))
        filtered = final_list

    # 7. Output
    result = orig.copy()
    crop_paths: list[str] = []
    crops_dir = get_output_dir() / STEP_DIRS["clusters"]["crops"]
    crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Salvare {len(filtered)} clustere finale în {crops_dir}...")

    for i, (x1, y1, x2, y2) in enumerate(filtered, 1):
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result, str(i), (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        crop = orig[y1:y2, x1:x2]
        crop = resize_bgr_max_side(crop)

        crop_path = crops_dir / f"cluster_{i}.jpg"
        cv2.imwrite(str(crop_path), crop)
        crop_paths.append(str(crop_path))

    save_debug(result, STEP_DIRS["clusters"]["final"], "final_clusters.jpg")
    print(f"✅ [DONE] Clustere finale returnate: {len(filtered)}")

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