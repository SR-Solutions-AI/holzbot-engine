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
    
    # ✅ ADĂUGAT: Eliminare puncte reziduale
    result = remove_residual_noise(result)
    
    return result


def remove_residual_noise(gray: np.ndarray) -> np.ndarray:
    """
    ✅ FUNCȚIE NOUĂ: Elimină punctele/zgomotul rămas după eliminarea hașurilor.
    
    Strategii aplicate:
    1. Median blur pentru zgomot "salt-and-pepper"
    2. Morfologie opening pentru puncte izolate mici
    3. Eliminare connected components foarte mici
    4. Bilateral filter pentru smoothing păstrând marginile
    """
    print("   └─ Curățare puncte reziduale...")
    
    # Pas 1: Median blur (elimină zgomot "salt and pepper")
    # Kernel 5x5 e suficient pentru puncte mici fără să blureze prea mult liniile
    denoised = cv2.medianBlur(gray, 5)
    save_debug(denoised, STEP_DIRS["hatch"], "1_median_blur.jpg")
    
    # Pas 2: Morfologie Opening (elimină puncte izolate foarte mici)
    # Kernel mic (2x2) pentru a nu afecta liniile subțiri ale planului
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel_small)
    save_debug(opened, STEP_DIRS["hatch"], "2_morphology_open.jpg")
    
    # Pas 3: Eliminare Connected Components foarte mici
    # Găsim toate componentele conectate
    _, binary_inv = cv2.threshold(opened, 250, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_inv, connectivity=8
    )
    
    # Calculăm aria minimă acceptabilă (0.0001% din imaginea totală)
    h, w = gray.shape
    img_area = h * w
    min_area = int(img_area * 0.000001)  # 0.0001% din arie
    
    # Creăm o mască care păstrează doar componentele suficient de mari
    clean_mask = np.zeros_like(binary_inv)
    removed_count = 0
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Păstrăm doar componentele mai mari decât pragul
        if area >= min_area:
            clean_mask[labels == i] = 255
        else:
            removed_count += 1
    
    # Aplicăm masca pe imaginea originală
    cleaned = opened.copy()
    cleaned[binary_inv > 0] = 255  # Albim totul mai întâi
    cleaned[clean_mask > 0] = gray[clean_mask > 0]  # Păstrăm doar componentele mari
    
    save_debug(clean_mask, STEP_DIRS["hatch"], "3_components_mask.jpg")
    print(f"      • Eliminate {removed_count} puncte mici (min_area={min_area}px)")
    
    # Pas 4: Bilateral filter (smoothing păstrând marginile)
    # Acest filtru face smoothing pe zone uniforme dar păstrează edge-urile ascuțite
    # Perfect pentru a elimina zgomotul fin rămas păstrând liniile planului
    final = cv2.bilateralFilter(cleaned, d=5, sigmaColor=50, sigmaSpace=50)
    save_debug(final, STEP_DIRS["hatch"], "4_bilateral_final.jpg")
    
    # Pas 5 (opțional): Un ultim threshold pentru a te asigura că totul e alb-negru curat
    _, final_binary = cv2.threshold(final, 240, 255, cv2.THRESH_BINARY)
    save_debug(final_binary, STEP_DIRS["hatch"], "5_final_clean.jpg")
    
    print("   └─ ✅ Puncte reziduale eliminate!")
    
    return final_binary


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