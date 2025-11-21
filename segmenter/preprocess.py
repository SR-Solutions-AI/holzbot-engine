# file: engine/runner/segmenter/preprocess.py
from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans

from .common import STEP_DIRS, save_debug


def remove_text_regions(img: np.ndarray) -> np.ndarray:
    print("\n[STEP 0] Eliminare text...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 25, 15
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 10 < w * h < 5000:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    cleaned = img.copy()
    cleaned[mask == 255] = (255, 255, 255)
    save_debug(mask, STEP_DIRS["text"], "mask.jpg")
    save_debug(cleaned, STEP_DIRS["text"], "no_text.jpg")
    return cleaned


def remove_hatched_areas(gray: np.ndarray) -> np.ndarray:
    print("\n[STEP 1] Eliminare hașuri...")
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    inv = cv2.bitwise_not(blur)

    responses = []
    for t in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        kernel = cv2.getGaborKernel((25, 25), 4.0, t, 10.0, 0.5, 0)
        responses.append(cv2.filter2D(inv, cv2.CV_8UC3, kernel))

    mean_map = np.mean(responses, axis=0)
    var_map = np.var(responses, axis=0)

    mean_norm = cv2.normalize(mean_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    var_norm = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, strong = cv2.threshold(mean_norm, 120, 255, cv2.THRESH_BINARY)
    _, lowvar = cv2.threshold(var_norm, 40, 255, cv2.THRESH_BINARY_INV)

    hatch_mask = cv2.bitwise_and(strong, lowvar)
    hatch_mask = cv2.morphologyEx(hatch_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    hatch_mask = cv2.morphologyEx(hatch_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    result = gray.copy()
    result[hatch_mask > 0] = 255

    save_debug(hatch_mask, STEP_DIRS["hatch"], "mask.jpg")
    save_debug(result, STEP_DIRS["hatch"], "cleaned.jpg")
    return result


def detect_outlines(gray: np.ndarray) -> np.ndarray:
    print("\n[STEP 2] Detectare contururi...")
    edges = cv2.Canny(gray, 40, 120)
    save_debug(edges, STEP_DIRS["outline"], "edges.jpg")
    return edges


def filter_thick_lines(mask: np.ndarray) -> np.ndarray:
    print("\n[STEP 3] Filtrare grosimi...")
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    vals = dist[dist > 0].reshape(-1, 1)

    if len(vals) < 50:
        return mask
    if len(vals) > 50000:
        vals = vals[np.random.choice(len(vals), 50000, replace=False)]

    km = KMeans(n_clusters=2, n_init=5, random_state=42)
    km.fit(vals)
    thick = (dist > 0.5 * max(km.cluster_centers_.flatten())).astype(np.uint8) * 255

    save_debug(thick, STEP_DIRS["thick"], "thick_lines.jpg")
    return thick


def solidify_walls(mask: np.ndarray) -> np.ndarray:
    print("\n[STEP 4] Solidificare pereți...")
    h, w = mask.shape
    k = max(3, int(min(h, w) * 0.002))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    dil = cv2.dilate(closed, kernel, iterations=2)
    ero = cv2.erode(dil, kernel)
    save_debug(ero, STEP_DIRS["solid"], "solidified.jpg")
    return ero
