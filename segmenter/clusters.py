# file: engine/runner/segmenter/clusters.py
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .common import STEP_DIRS, save_debug, resize_bgr_max_side, get_output_dir


def split_large_cluster(region: np.ndarray, x1: int, y1: int, idx: int) -> list[list[int]]:
    print(f"  🔹 Split check cluster #{idx}")
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

                # BUG-ul tău original în calculul area2 l-am „reparat” aici,
                # dar dacă vrei 1:1, poți pune la loc (yy2 - yy2).
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


def detect_clusters(mask: np.ndarray, orig: np.ndarray) -> list[str]:
    """
    Detectează clusterele (planurile) și le salvează ca imagini.
    RETURN: listă de path-uri (str) către toate planurile decupate.
    """
    print("\n[STEP 7] Detectare clustere...")
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask.copy()
    inv = cv2.bitwise_not(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(cv2.dilate(inv, kernel), cv2.MORPH_OPEN, kernel)
    save_debug(clean, STEP_DIRS["clusters"]["initial"], "mask_clean.jpg")

    num, _, stats, _ = cv2.connectedComponentsWithStats(clean, 8)
    boxes = [[x, y, x + bw, y + bh] for x, y, bw, bh, a in stats[1:] if a > 200]
    print(f"🔸 Clustere inițiale: {len(boxes)}")

    refined: list[list[int]] = []
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        reg = clean[y1:y2, x1:x2]
        if reg.size == 0:
            continue
        for sb in split_large_cluster(reg, x1, y1, i):
            refined.append(expand_cluster(clean, *sb))

    merged = merge_overlapping_boxes(refined, clean.shape)
    save_debug(orig, STEP_DIRS["clusters"]["merged"], "after_merge.jpg")

    # eliminăm clustere complet conținute în altele
    filtered: list[list[int]] = []
    for i, (ax1, ay1, ax2, ay2) in enumerate(merged):
        contained = False
        for j, (bx1, by1, bx2, by2) in enumerate(merged):
            if i == j:
                continue
            if bx1 <= ax1 and by1 <= ay1 and bx2 >= ax2 and by2 >= ay2:
                contained = True
                break
        if not contained:
            filtered.append([ax1, ay1, ax2, ay2])

    # filtrare relativă/absolută clustere prea mici
    if filtered:
        areas = np.array([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in filtered], dtype=np.float64)
        max_area = float(areas.max())
        img_area = float(orig.shape[0] * orig.shape[1])

        MIN_REL = 0.10
        MIN_ABS = 0.0005
        min_allowed = max(MIN_REL * max_area, MIN_ABS * img_area)
        keep_idx = [i for i, a in enumerate(areas) if a >= min_allowed]
        filtered = [filtered[i] for i in keep_idx]

    result = orig.copy()
    crop_paths: list[str] = []
    crops_dir = get_output_dir() / STEP_DIRS["clusters"]["crops"]
    crops_dir.mkdir(parents=True, exist_ok=True)

    for i, (x1, y1, x2, y2) in enumerate(filtered, 1):
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result, str(i), (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        crop = orig[y1:y2, x1:x2]
        crop = resize_bgr_max_side(crop)

        crop_path = crops_dir / f"cluster_{i}.jpg"
        cv2.imwrite(str(crop_path), crop)
        crop_paths.append(str(crop_path))

    save_debug(result, STEP_DIRS["clusters"]["final"], "final_clusters.jpg")
    print(f"✅ Clustere finale: {len(filtered)}")

    return crop_paths


def detect_wall_zones(orig: np.ndarray, thick_mask: np.ndarray) -> list[str]:
    """
    Construiește masca de pereți și scoate toate clusterele (planurile).
    RETURN: listă de path-uri către planuri.
    """
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
