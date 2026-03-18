# file: engine/cubicasa_detector/detector.py
from __future__ import annotations

import sys
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import json
import math
import re
import requests
import base64
from pathlib import Path
from PIL import Image
from io import BytesIO
from skimage.morphology import skeletonize

# Importăm funcțiile OCR și room filling din modulul dedicat
from .ocr_room_filling import (
    fill_stairs_room,
    fill_room_by_ocr,
    preprocess_image_for_ocr,
    run_ocr_on_zones,
    _reconstruct_word_from_chars,
)

# Importăm funcțiile din modulele refactorizate
from .raster_api import (
    call_raster_api,
    generate_raster_images,
    generate_api_walls_mask,
    get_api_walls_mask_for_alignment,
    brute_force_alignment,
    brute_force_translation_only,
    build_api_walls_mask_from_json,
    build_aligned_api_walls_1px_original,
    run_extra_alignment_methods,
    apply_alignment_and_generate_overlay,
    generate_crop_from_raster,
    _draw_response_overlay,
)
from .wall_repair import (
    repair_house_walls_with_floodfill,
    bridge_wall_gaps,
    smart_wall_closing,
    get_strict_1px_outline,
)
from .config import DEBUG
from .raster_processing import (
    generate_raster_walls_overlay,
    detect_interior_exterior_from_raster,
    calculate_scale_per_room,
    generate_walls_interior_exterior,
    generate_interior_structure_walls,
)
from .interior_exterior import detect_interior_exterior_zones
from .scale_detection import detect_scale_from_room_labels, call_gemini
from .measurements import calculate_measurements

# OCR pentru detectarea textului
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️ pytesseract nu este disponibil. Detectarea textului 'terasa' va fi dezactivată.")

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

# Nume de step-uri care trebuie mereu salvate (folosite de phase 2 / scale / count_objects)
_REQUIRED_STEP_NAMES = {"00_original", "02_ai_walls_closed", "03_outdoor_mask"}
# Setat la intrarea în run_cubicasa_detection; folosit de save_step
_save_debug_steps = True

def save_step(name, img, steps_dir):
    """Salvează imaginea doar dacă e necesară sau _save_debug_steps e True."""
    if name not in _REQUIRED_STEP_NAMES and not _save_debug_steps:
        return
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

def border_constrained_fill(walls_mask: np.ndarray, steps_dir: str = None) -> np.ndarray:
    """
    Închide golurile în peretele exterior folosind Border-Constrained Fill.
    
    Detectează cel mai mare contur (perimetrul casei), generează convex hull,
    și folosește hull-ul ca ghidaj pentru a vedea unde ar trebui să existe un perete
    între două puncte extreme.
    
    Args:
        walls_mask: Masca pereților (255 = perete, 0 = spațiu liber)
        steps_dir: Director pentru salvarea step-urilor de debug (opțional)
    
    Returns:
        Masca pereților cu golurile din peretele exterior închise
    """
    h, w = walls_mask.shape[:2]
    
    print(f"      🏛️ Border-Constrained Fill: închid goluri în peretele exterior...")
    
    result = walls_mask.copy()
    
    # Pas 1: Detectăm contururile exterioare
    contours, _ = cv2.findContours(walls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print(f"         ⚠️ Nu s-au detectat contururi")
        return result
    
    # Pas 2: Găsim cel mai mare contur (perimetrul casei)
    largest_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest_contour) < min(h, w) * 10:  # Contur prea mic
        print(f"         ⚠️ Conturul cel mai mare este prea mic")
        return result
    
    # Pas 3: Generăm convex hull pentru conturul cel mai mare
    hull = cv2.convexHull(largest_contour)
    
    if steps_dir:
        # Salvăm vizualizarea hull-ului
        debug_image = cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_image, [largest_contour], -1, (0, 255, 0), 2)  # Verde pentru conturul original
        cv2.drawContours(debug_image, [hull], -1, (0, 0, 255), 2)  # Roșu pentru hull
        cv2.imwrite(str(Path(steps_dir) / "02c_border_hull_debug.png"), debug_image)
    
    # Pas 4: Comparăm hull-ul cu conturul original pentru a găsi golurile
    # Creăm o mască pentru hull
    hull_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(hull_mask, [hull], 255)
    
    # Creăm o mască pentru conturul original
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(contour_mask, [largest_contour], 255)
    
    # Diferența dintre hull și contur indică zonele unde ar trebui să existe pereți
    # Dar nu vrem să umplem interiorul, doar să conectăm punctele extreme
    hull_points = hull.reshape(-1, 2)
    contour_points = largest_contour.reshape(-1, 2)
    
    # Pas 5: Găsim segmentele din hull care reprezintă goluri reale în peretele exterior
    # Strategie: Verificăm doar segmentele care conectează puncte apropiate de conturul original
    connections_made = 0
    max_gap_distance = max(150, int(min(h, w) * 0.12))  # 12% din dimensiunea minimă
    
    # Găsim punctele din conturul original care sunt aproape de hull
    contour_points = largest_contour.reshape(-1, 2)
    
    for i in range(len(hull_points)):
        p1 = tuple(hull_points[i])
        p2 = tuple(hull_points[(i + 1) % len(hull_points)])
        
        # Calculăm distanța dintre puncte
        segment_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        # Ignorăm segmentele prea scurte sau prea lungi
        if segment_length < 20 or segment_length > max_gap_distance:
            continue
        
        # Găsim cele mai apropiate puncte din conturul original pentru p1 și p2
        min_dist_p1 = float('inf')
        min_dist_p2 = float('inf')
        closest_contour_p1 = None
        closest_contour_p2 = None
        
        for cp in contour_points:
            dist_p1 = np.sqrt((cp[0] - p1[0])**2 + (cp[1] - p1[1])**2)
            dist_p2 = np.sqrt((cp[0] - p2[0])**2 + (cp[1] - p2[1])**2)
            
            if dist_p1 < min_dist_p1:
                min_dist_p1 = dist_p1
                closest_contour_p1 = tuple(cp)
            
            if dist_p2 < min_dist_p2:
                min_dist_p2 = dist_p2
                closest_contour_p2 = tuple(cp)
        
        # Verificăm că ambele capete sunt aproape de conturul original (max 50 pixeli)
        max_endpoint_distance = 50
        if min_dist_p1 > max_endpoint_distance or min_dist_p2 > max_endpoint_distance:
            continue
        
        # Verificăm dacă există deja un perete între cele două puncte din conturul original
        # (nu între punctele din hull, ci între cele mai apropiate puncte din contur)
        if closest_contour_p1 is None or closest_contour_p2 is None:
            continue
        
        # Verificăm dacă cele două puncte din contur sunt consecutive sau apropiate
        # Găsim pozițiile lor în contur
        idx1 = None
        idx2 = None
        for idx, cp in enumerate(contour_points):
            if np.allclose(cp, closest_contour_p1, atol=2):
                idx1 = idx
            if np.allclose(cp, closest_contour_p2, atol=2):
                idx2 = idx
        
        if idx1 is None or idx2 is None:
            continue
        
        # Verificăm dacă punctele sunt apropiate în contur (nu la capete opuse)
        contour_distance = min(abs(idx2 - idx1), len(contour_points) - abs(idx2 - idx1))
        if contour_distance > len(contour_points) * 0.3:  # Prea departe în contur
            continue
        
        # Verificăm dacă există un gol între cele două puncte din contur
        # Eșantionăm puncte de-a lungul liniei dintre punctele din contur
        num_samples = max(30, int(segment_length / 5))
        wall_pixels = 0
        space_pixels = 0
        
        for t in np.linspace(0, 1, num_samples):
            px = int(closest_contour_p1[0] + t * (closest_contour_p2[0] - closest_contour_p1[0]))
            py = int(closest_contour_p1[1] + t * (closest_contour_p2[1] - closest_contour_p1[1]))
            
            if 0 <= px < w and 0 <= py < h:
                if walls_mask[py, px] == 255:
                    wall_pixels += 1
                else:
                    space_pixels += 1
        
        # Dacă mai puțin de 40% din linie este perete, există un gol
        if wall_pixels / num_samples < 0.4:
            # Verificăm că linia nu trece prin interiorul casei
            # Eșantionăm câteva puncte și verificăm că nu sunt în interior
            valid_connection = True
            interior_count = 0
            
            for t in np.linspace(0.2, 0.8, 5):  # Verificăm doar punctele din mijloc
                px = int(closest_contour_p1[0] + t * (closest_contour_p2[0] - closest_contour_p1[0]))
                py = int(closest_contour_p1[1] + t * (closest_contour_p2[1] - closest_contour_p1[1]))
                
                if 0 <= px < w and 0 <= py < h:
                    # Verificăm dacă punctul este în interiorul conturului
                    point_inside = cv2.pointPolygonTest(largest_contour, (px, py), False)
                    if point_inside > 0:  # În interior
                        interior_count += 1
                    
                    # Verificăm că există pereți în jur (nu doar spațiu liber)
                    y_min = max(0, py - 15)
                    y_max = min(h, py + 16)
                    x_min = max(0, px - 15)
                    x_max = min(w, px + 16)
                    neighborhood = walls_mask[y_min:y_max, x_min:x_max]
                    if np.count_nonzero(neighborhood) < 10:  # Prea puțini pereți în jur
                        valid_connection = False
                        break
            
            # Dacă mai mult de 2 puncte sunt în interior, nu desenăm (ar tăia colțurile)
            if interior_count > 2:
                valid_connection = False
            
            if valid_connection:
                # Desenăm linia între punctele din conturul original (nu din hull)
                cv2.line(result, closest_contour_p1, closest_contour_p2, 255, 2)
                connections_made += 1
    
    print(f"         ✅ Făcute {connections_made} conexiuni pentru peretele exterior")
    
    # Salvăm rezultatul
    if steps_dir:
        output_path = Path(steps_dir) / "02c_border_constrained_fill.png"
        cv2.imwrite(str(output_path), result)
        print(f"         💾 Salvat: {output_path.name}")
    
    return result

def interval_merging_axis_projections(walls_mask: np.ndarray, steps_dir: str = None, corridor_width: int = 5, max_vacuum_gap: int = 20) -> np.ndarray:
    """
    Unește segmentele de pereți folosind Binary Conflict Profiling.
    Algoritm robust care verifică doar segmentele adiacente și folosește testele de intruziune și Ghost.
    
    Algoritm:
    1. Vectorizare: Detectăm segmentele folosind LSD (Line Segment Detector)
    2. Grupare pe Axe: Grupăm segmentele pe "șine" (aceeași coordonată Y pentru orizontale, X pentru verticale)
    3. Sortare și Perechi Adiacente: Sortăm segmentele și verificăm doar perechile adiacente
    4. Test de Intruziune: Verificăm dacă există pereți perpendiculari între capete
    5. Test Ghost: Verificăm dacă zona e prea albă în original (cameră)
    
    Args:
        walls_mask: Masca pereților (255 = perete, 0 = spațiu liber) - 02_ai_walls_closed.png
        steps_dir: Director pentru salvarea step-urilor de debug (opțional)
        corridor_width: Lățimea coridorului de scanare (nefolosit în acest algoritm, păstrat pentru compatibilitate)
        max_vacuum_gap: Numărul maxim de pixeli albi consecutivi permisi (default: 20)
    
    Returns:
        Masca pereților cu segmentele unite chirurgical
    """
    h, w = walls_mask.shape[:2]
    
    print(f"      📐 Binary Conflict Profiling: unesc segmente adiacente cu validare robustă...")
    
    result = walls_mask.copy()
    
    # Încărcăm imaginea originală (Ghost Layer) pentru validare
    ghost_img = None
    if steps_dir:
        original_path = Path(steps_dir) / "00_original.png"
        if original_path.exists():
            print(f"         🔍 Încărc imaginea originală pentru validare...")
            original_img = cv2.imread(str(original_path))
            if original_img is not None:
                # Convertim la grayscale
                ghost_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                # Redimensionăm dacă e necesar
                if ghost_img.shape[:2] != (h, w):
                    ghost_img = cv2.resize(ghost_img, (w, h))
                print(f"         ✅ Imagine originală încărcată pentru validare")
    
    # Pas 1: Vectorizare prin LSD (Line Segment Detector)
    print(f"         🔍 Pas 1: Detectez segmente cu LSD...")
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(walls_mask)[0]
    
    if lines is None or len(lines) == 0:
        print(f"         ⚠️ Nu s-au detectat segmente")
        return result
    
    print(f"         ✅ Detectat {len(lines)} segmente")
    
    # Creăm imagini pentru vizualizare
    if steps_dir:
        vis_segments = cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
        vis_grouped = cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
        vis_connections = cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
        vis_result = cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
    
    # Paletă de culori
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
    ]
    
    # Desenăm segmentele detectate
    if steps_dir:
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0].astype(int)
            color = colors[i % len(colors)]
            cv2.line(vis_segments, (x1, y1), (x2, y2), color, 2)
        cv2.imwrite(str(Path(steps_dir) / "02f_01_lsd_segments.png"), vis_segments)
        print(f"         💾 Salvat: 02f_01_lsd_segments.png")
    
    # Pas 2: Grupare pe Axe (șine)
    print(f"         🔍 Pas 2: Grupez segmente pe axe...")
    horizontal_rails = {}  # cheie: y_coord (rotunjit), valoare: lista de (x1, x2, line_idx)
    vertical_rails = {}    # cheie: x_coord (rotunjit), valoare: lista de (y1, y2, line_idx)
    
    tolerance = 3  # Toleranță pentru gruparea pe axe
    
    for line_idx, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*dx + dy*dy)
        
        if length < 10:
            continue
        
        # Verificăm dacă e orizontală
        if abs(dy) < tolerance:
            y_coord = int((y1 + y2) / 2)
            y_key = round(y_coord / tolerance) * tolerance
            if y_key not in horizontal_rails:
                horizontal_rails[y_key] = []
            horizontal_rails[y_key].append((min(x1, x2), max(x1, x2), line_idx))
        
        # Verificăm dacă e verticală
        elif abs(dx) < tolerance:
            x_coord = int((x1 + x2) / 2)
            x_key = round(x_coord / tolerance) * tolerance
            if x_key not in vertical_rails:
                vertical_rails[x_key] = []
            vertical_rails[x_key].append((min(y1, y2), max(y1, y2), line_idx))
    
    print(f"         ✅ Grupate: {len(horizontal_rails)} șine orizontale, {len(vertical_rails)} șine verticale")
    
    if steps_dir:
        for y_key, segments in horizontal_rails.items():
            for x1, x2, line_idx in segments:
                color = colors[line_idx % len(colors)]
                cv2.line(vis_grouped, (int(x1), int(y_key)), (int(x2), int(y_key)), color, 2)
        for x_key, segments in vertical_rails.items():
            for y1, y2, line_idx in segments:
                color = colors[line_idx % len(colors)]
                cv2.line(vis_grouped, (int(x_key), int(y1)), (int(x_key), int(y2)), color, 2)
        cv2.imwrite(str(Path(steps_dir) / "02f_02_grouped_rails.png"), vis_grouped)
        print(f"         💾 Salvat: 02f_02_grouped_rails.png")
    
    # Funcții helper pentru validare
    def test_intrusion(start_pt, end_pt, walls_mask_image):
        """
        Test de Intruziune: Verifică dacă între capete există pereți perpendiculari.
        Implementare optimizată: linie pe mască + intersecție cu pereții (OpenCV), fără bucle Python.
        """
        h, w = walls_mask_image.shape[:2]
        start_pt = (int(start_pt[0]), int(start_pt[1]))
        end_pt = (int(end_pt[0]), int(end_pt[1]))
        # Bandă similară cu vechiul check_radius*2 (10 px)
        line_thickness = 10
        line_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(line_mask, start_pt, end_pt, 255, thickness=line_thickness)
        intersection = cv2.bitwise_and(walls_mask_image, line_mask)
        line_pixels = cv2.countNonZero(line_mask)
        wall_pixels_on_line = cv2.countNonZero(intersection)
        # Acceptăm dacă cel mult 20% din pixeli de pe linie sunt pereți (același prag ca înainte)
        threshold = 0.2 * line_pixels if line_pixels > 0 else 0
        return wall_pixels_on_line <= threshold
    
    def test_ghost_profile(start_pt, end_pt, ghost_image):
        """
        Test Ghost: Verifică profilul de intensitate în imaginea originală.
        Dacă zona e prea albă (cameră), respinge conexiunea.
        """
        if ghost_image is None:
            return True
        
        # Creăm o mască temporară cu linia propusă
        temp_mask = np.zeros(ghost_image.shape[:2], dtype=np.uint8)
        cv2.line(temp_mask, start_pt, end_pt, 255, 3)
        
        # Extragem pixelii de sub linie
        path_pixels = ghost_image[temp_mask > 0]
        
        if len(path_pixels) == 0:
            return False
        
        # Verificăm raportul de alb (vid)
        white_ratio = np.sum(path_pixels > 245) / len(path_pixels)
        
        # Dacă mai mult de 85% e alb, e cameră
        return white_ratio < 0.85
    
    # Pas 3: Sortare și verificare perechi adiacente
    print(f"         🔍 Pas 3: Sortez segmente și verific perechi adiacente...")
    connections_made = 0
    
    # Procesăm șinele orizontale
    for y_coord, segments in horizontal_rails.items():
        if len(segments) < 2:
            continue
        
        # Sortăm segmentele de la stânga la dreapta
        segments_sorted = sorted(segments, key=lambda x: x[0])
        
        # Verificăm doar perechile adiacente
        for i in range(len(segments_sorted) - 1):
            x1_end, x1_start, line_idx1 = segments_sorted[i]
            x2_start, x2_end, line_idx2 = segments_sorted[i + 1]
            
            # Capetele segmentelor
            end_pt = (int(x1_end), int(y_coord))
            start_pt = (int(x2_start), int(y_coord))
            
            # Verificăm dacă există un gap
            if x2_start > x1_end:
                gap_size = x2_start - x1_end
                
                # Ignorăm gap-uri prea mari (probabil nu sunt pe aceeași axă)
                if gap_size > w * 0.3:
                    continue
                
                # TEST DE INTRUZIUNE: Verificăm dacă există pereți perpendiculari
                if not test_intrusion(end_pt, start_pt, walls_mask):
                    if steps_dir:
                        cv2.line(vis_connections, end_pt, start_pt, (0, 0, 255), 2)  # Roșu
                    continue
                
                # TEST GHOST: Verificăm profilul de intensitate
                if not test_ghost_profile(end_pt, start_pt, ghost_img):
                    if steps_dir:
                        cv2.line(vis_connections, end_pt, start_pt, (0, 165, 255), 2)  # Portocaliu
                    continue
                
                # Ambele teste au trecut - unim!
                cv2.line(result, end_pt, start_pt, 255, 3)
                
                if steps_dir:
                    cv2.line(vis_connections, end_pt, start_pt, (0, 255, 0), 2)  # Verde
                    cv2.line(vis_result, end_pt, start_pt, (0, 255, 0), 2)
                
                connections_made += 1
    
    # Procesăm șinele verticale (aceeași logică)
    for x_coord, segments in vertical_rails.items():
        if len(segments) < 2:
            continue
        
        # Sortăm segmentele de sus în jos
        segments_sorted = sorted(segments, key=lambda x: x[0])
        
        # Verificăm doar perechile adiacente
        for i in range(len(segments_sorted) - 1):
            y1_end, y1_start, line_idx1 = segments_sorted[i]
            y2_start, y2_end, line_idx2 = segments_sorted[i + 1]
            
            # Capetele segmentelor
            end_pt = (int(x_coord), int(y1_end))
            start_pt = (int(x_coord), int(y2_start))
            
            # Verificăm dacă există un gap
            if y2_start > y1_end:
                gap_size = y2_start - y1_end
                
                # Ignorăm gap-uri prea mari
                if gap_size > h * 0.3:
                    continue
                
                # TEST DE INTRUZIUNE
                if not test_intrusion(end_pt, start_pt, walls_mask):
                    if steps_dir:
                        cv2.line(vis_connections, end_pt, start_pt, (0, 0, 255), 2)  # Roșu
                    continue
                
                # TEST GHOST
                if not test_ghost_profile(end_pt, start_pt, ghost_img):
                    if steps_dir:
                        cv2.line(vis_connections, end_pt, start_pt, (0, 165, 255), 2)  # Portocaliu
                    continue
                
                # Ambele teste au trecut - unim!
                cv2.line(result, end_pt, start_pt, 255, 3)
                
                if steps_dir:
                    cv2.line(vis_connections, end_pt, start_pt, (0, 255, 0), 2)  # Verde
                    cv2.line(vis_result, end_pt, start_pt, (0, 255, 0), 2)
                
                connections_made += 1
    
    print(f"         ✅ Unite {connections_made} perechi de segmente adiacente")
    
    # Salvăm rezultatele
    if steps_dir:
        cv2.imwrite(str(Path(steps_dir) / "02f_03_connections.png"), vis_connections)
        print(f"         💾 Salvat: 02f_03_connections.png (verde=valide, roșu=intruziune, portocaliu=ghost)")
        
        cv2.imwrite(str(Path(steps_dir) / "02f_04_interval_merging_result.png"), vis_result)
        print(f"         💾 Salvat: 02f_04_interval_merging_result.png")
        
        cv2.imwrite(str(Path(steps_dir) / "02f_05_final_result.png"), result)
        print(f"         💾 Salvat: 02f_05_final_result.png (rezultat final binar)")
    
    return result

# Funcțiile OCR (preprocess_image_for_ocr, _reconstruct_word_from_chars, run_ocr_on_zones) 
# au fost mutate în ocr_room_filling.py și sunt importate de acolo.
# Funcțiile OCR duplicate au fost șterse - sunt importate din ocr_room_filling.py

# Funcțiile mutate în modulele refactorizate:
# - repair_house_walls_with_floodfill, bridge_wall_gaps, smart_wall_closing, get_strict_1px_outline -> wall_repair.py
# - detect_interior_exterior_zones -> interior_exterior.py
# - detect_scale_from_room_labels, call_gemini -> scale_detection.py
# - calculate_measurements -> measurements.py
# - call_raster_api, generate_raster_images, brute_force_alignment, etc. -> raster_api.py
    """
    Încearcă să repare toți pereții de jur-împrejurul casei folosind aceeași
    idee ca la terasă: flood fill în interiorul casei, apoi completarea
    golurilor de pe conturul regiunii umplute.
    
    Args:
        walls_mask: Masca pereților (trebuie să includă deja pereții adăugați pentru garaj/scări)
        steps_dir: Director pentru salvarea imaginilor de debug (opțional)
    
    Returns:
        Masca pereților cu pereții exteriori reparați (dacă a fost posibil)
    """
    if walls_mask is None:
        print(f"      ⚠️ walls_mask este None. Skip repararea pereților casei.")
        return None
    
    try:
        h, w = walls_mask.shape[:2]
    except AttributeError:
        print(f"      ⚠️ walls_mask nu are atributul shape. Skip repararea pereților casei.")
        return None
    
    # Folosim o metodă alternativă dacă OCR nu este disponibil
    use_ocr = TESSERACT_AVAILABLE
    if not use_ocr:
        print(f"      ⚠️ pytesseract nu este disponibil. Skip detectarea {room_name}.")
        return walls_mask.copy()
    
    result = walls_mask.copy()
    
    print(f"      🏡 Detectez și umplu camere ({room_name})...")
    
    # Pas 1: Încărcăm overlay-ul sau original-ul pentru OCR
    overlay_path = None
    original_path = None
    if steps_dir:
        overlay_path = Path(steps_dir) / "02d_walls_closed_overlay.png"
        original_path = Path(steps_dir) / "00_original.png"
    
    # Preferăm imaginea originală pentru detectarea textului (textul este mai clar acolo)
    ocr_image = None
    if original_path and original_path.exists():
        ocr_image = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
        if ocr_image is None:
            ocr_image = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
            if ocr_image is not None:
                ocr_image = cv2.cvtColor(ocr_image, cv2.COLOR_GRAY2BGR)
        print(f"         📸 Folosesc original pentru OCR: {original_path.name}")
    elif overlay_path and overlay_path.exists():
        ocr_image = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        print(f"         📸 Folosesc overlay pentru OCR (fallback): {overlay_path.name}")
    
    if ocr_image is None:
        print(f"         ⚠️ Nu s-a găsit overlay sau original. Skip detectarea {room_name}.")
        return result
    
    # Redimensionăm dacă este necesar
    if ocr_image.shape[:2] != (h, w):
        ocr_image = cv2.resize(ocr_image, (w, h))
    
    # Pas 2: Detectăm textul folosind OCR cu preprocesare și analiză pe zone
    print(f"         🔍 Pas 1: Detectez text ({room_name})...")
    
    text_found = False
    text_boxes = []
    all_detections = []  # Inițializăm all_detections pentru a evita NameError
    
    try:
        if use_ocr:
            # Metoda îmbunătățită: OCR cu preprocesare și analiză pe zone
            print(f"         📝 Folosesc OCR cu preprocesare și analiză pe zone...")
            
            # Salvez imaginea preprocesată pentru debug
            if steps_dir:
                processed_img = preprocess_image_for_ocr(ocr_image)
                cv2.imwrite(str(Path(steps_dir) / f"{debug_prefix}_00_preprocessed.png"), processed_img)
                print(f"         💾 Salvat: {debug_prefix}_00_preprocessed.png (imagine preprocesată)")
            
            # Rulează OCR pe zone cu zoom
            text_boxes, all_detections = run_ocr_on_zones(ocr_image, search_terms, steps_dir, 
                                         grid_rows=3, grid_cols=3, zoom_factor=2.0)
            
            if text_boxes:
                            text_found = True
        else:
            print(f"         ⚠️ Fără OCR nu pot identifica specific cuvântul '{room_name}'.")
            text_found = False
            text_boxes = []
            all_detections = []
        
        # Selectăm rezultatul cu confidence maxim (dacă există)
        accepted_boxes = []  # Detecțiile acceptate și folosite (confidence > 60%)
        best_box_all = None  # Cea mai bună detecție (chiar dacă are confidence < 60%)
        
        if text_boxes:
            # Sortăm după confidence (descrescător)
            text_boxes.sort(key=lambda box: box[5], reverse=True)  # box[5] = confidence
            best_box_all = text_boxes[0]  # Cea mai bună detecție (chiar dacă e < 60%)
            
            # Filtram doar detecțiile cu confidence > 60% pentru procesare
            accepted_boxes = [box for box in text_boxes if box[5] > 60]
            
            if accepted_boxes:
                best_box = accepted_boxes[0]
                print(f"         🎯 Selectat rezultatul cu confidence maxim: '{best_box[4]}' cu {best_box[5]:.1f}%")
                text_boxes = accepted_boxes.copy()  # Actualizăm text_boxes pentru procesare
            else:
                print(f"         ⚠️ Cea mai bună detecție: '{best_box_all[4]}' cu {best_box_all[5]:.1f}% (< 60%, nu acceptată)")
                text_boxes = []  # Nu avem detecții acceptate pentru procesare
        elif use_ocr and all_detections:
            # Dacă nu am găsit detecții care se potrivesc cu termenii, folosim cea mai bună detecție din toate
            all_detections.sort(key=lambda box: box[5], reverse=True)  # box[5] = confidence
            best_box_all = all_detections[0]
            print(f"         ⚠️ Nu s-au găsit detecții care se potrivesc cu termenii, dar am găsit '{best_box_all[4]}' cu {best_box_all[5]:.1f}%")
        
        # Salvăm poza cu cea mai bună detecție (chiar dacă nu e acceptată)
        if steps_dir and best_box_all:
            vis_best_detection = ocr_image.copy() if ocr_image is not None else cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
            best_x, best_y, best_width, best_height, best_text, best_conf = best_box_all
            best_center_x = best_x + best_width // 2
            best_center_y = best_y + best_height // 2
            
            # Culoare: verde dacă e acceptată, portocaliu dacă nu
            if best_conf > 60:
                detection_color = (0, 255, 0)  # Verde
                status_label = f"✅ ACCEPTED ({best_conf:.1f}%)"
            else:
                detection_color = (0, 165, 255)  # Portocaliu
                status_label = f"❌ REJECTED ({best_conf:.1f}% < 60%)"
            
            cv2.rectangle(vis_best_detection, (best_x, best_y), (best_x + best_width, best_y + best_height), detection_color, 3)
            cv2.circle(vis_best_detection, (best_center_x, best_center_y), 8, (0, 0, 255), -1)
            
            label_text = f"{best_text} - {status_label} | No Flood Fill"
            font_scale = max(0.7, best_height / 30.0)
            font_thickness = max(2, int(font_scale * 2))
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            text_y = max(text_height + 5, best_y - 5)
            text_x = best_x
            cv2.rectangle(vis_best_detection, 
                         (text_x, text_y - text_height - baseline), 
                         (text_x + text_width + 10, text_y + baseline), 
                         (255, 255, 255), -1)
            
            cv2.putText(vis_best_detection, label_text, (text_x + 5, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, detection_color, font_thickness)
            
            output_path = Path(steps_dir) / f"{debug_prefix}_01c_best_detection_with_fill.png"
            cv2.imwrite(str(output_path), vis_best_detection)
            print(f"         💾 Salvat: {output_path.name} (best detection: {best_conf:.1f}%, no flood fill)")
        
        if not text_found or not accepted_boxes:
            if not text_found:
                print(f"         ⚠️ Nu s-a detectat text ({room_name}) în plan.")
            else:
                print(f"         ⚠️ Nu s-a detectat text ({room_name}) cu confidence > 60%.")
            if steps_dir:
                vis_ocr = ocr_image.copy() if ocr_image is not None else cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(str(Path(steps_dir) / f"{debug_prefix}_01_ocr_result.png"), vis_ocr)
                print(f"         💾 Salvat: {debug_prefix}_01_ocr_result.png")
            return result
        
        # Pas 3: Vizualizăm textul detectat (toate detecțiile)
        if steps_dir:
            vis_ocr = ocr_image.copy()
            for x, y, width, height, text, conf in text_boxes:
                cv2.rectangle(vis_ocr, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(vis_ocr, f"{text} ({conf:.0f}%)", (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite(str(Path(steps_dir) / f"{debug_prefix}_01_ocr_result.png"), vis_ocr)
            print(f"         💾 Salvat: {debug_prefix}_01_ocr_result.png (text detectat)")
        
        # Pas 3b: Vizualizăm DOAR detecțiile acceptate (cele care sunt luate în calcul)
        if steps_dir and accepted_boxes:
            vis_accepted = ocr_image.copy()
            for x, y, width, height, text, conf in accepted_boxes:
                # Desenăm dreptunghiul cu culoare verde mai intensă
                cv2.rectangle(vis_accepted, (x, y), (x + width, y + height), (0, 255, 0), 3)
                
                # Desenăm centrul textului (punctul de start pentru flood fill)
                center_x = x + width // 2
                center_y = y + height // 2
                cv2.circle(vis_accepted, (center_x, center_y), 8, (0, 0, 255), -1)  # Roșu pentru centru
                
                # Desenăm textul cu fundal pentru lizibilitate
                label = f"{text} ({conf:.0f}%)"
                font_scale = max(0.6, height / 25.0)
                font_thickness = max(2, int(font_scale * 2))
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                
                # Poziționăm textul deasupra dreptunghiului
                text_y = max(text_height + 5, y - 5)
                text_x = x
                
                # Desenăm fundal pentru text
                cv2.rectangle(vis_accepted, 
                             (text_x, text_y - text_height - baseline), 
                             (text_x + text_width, text_y + baseline), 
                             (0, 255, 0), -1)
                
                # Desenăm textul
                cv2.putText(vis_accepted, label, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
            
            cv2.imwrite(str(Path(steps_dir) / f"{debug_prefix}_01b_accepted_detections.png"), vis_accepted)
            print(f"         💾 Salvat: {debug_prefix}_01b_accepted_detections.png ({len(accepted_boxes)} detecție/ii acceptată/e)")
        
        # Pas 4: Pentru fiecare text detectat, găsim zona camerei și facem flood fill
        print(f"         🔍 Pas 2: Găsesc zona camerei și fac flood fill...")
        
        # Încărcăm overlay-ul combinat (pereti + original cu 50% transparency)
        overlay_combined = None
        if steps_dir:
            overlay_path = Path(steps_dir) / "02d_walls_closed_overlay.png"
            if overlay_path.exists():
                overlay_combined = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
                if overlay_combined is not None:
                    # Redimensionăm dacă este necesar
                    if overlay_combined.shape[:2] != (h, w):
                        overlay_combined = cv2.resize(overlay_combined, (w, h))
                    print(f"         📸 Folosesc overlay combinat (pereti + original 50%) pentru flood fill")
                else:
                    print(f"         ⚠️ Nu pot încărca overlay-ul. Folosesc walls_mask simplu.")
        
        # Dacă nu avem overlay, folosim walls_mask simplu
        if overlay_combined is None:
            # Creăm o mască pentru spațiile libere (inversul pereților)
            spaces_mask = cv2.bitwise_not(walls_mask)
        else:
            # Convertim overlay-ul la grayscale
            overlay_gray = cv2.cvtColor(overlay_combined, cv2.COLOR_BGR2GRAY)
            
            # Binarizăm overlay-ul pentru a identifica pereții
            # Pereții în overlay sunt mai închiși (din combinația de 50% original + 50% walls)
            # Folosim un threshold adaptiv pentru a separa pereții de spațiile libere
            _, overlay_binary = cv2.threshold(overlay_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Inversăm pentru a obține spațiile libere (0 = perete, 255 = spațiu liber)
            spaces_mask = cv2.bitwise_not(overlay_binary)
            print(f"         📊 Overlay binarizat: pereți identificați din combinație")
        
        # Procesăm DOAR rezultatul cu confidence maxim (dacă există)
        rooms_filled = 0
        if text_boxes:
            # Procesăm doar primul (și singurul) rezultat - cel cu confidence maxim
            box_idx = 0
            x, y, width, height, text, conf = text_boxes[0]
            # Centrul textului
            center_x = x + width // 2
            center_y = y + height // 2
            
            # Determinăm dacă este garaj/carport (care are doar 3 pereți)
            is_garage = room_name.lower() in ['garage', 'garaj', 'carport']
            
            if not use_ocr:
                print(f"         ⚠️ Fără OCR nu pot identifica specific cuvântul. Skip.")
                if steps_dir:
                    vis_fill_attempt = cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
                    cv2.circle(vis_fill_attempt, (center_x, center_y), 8, (0, 0, 255), -1)
                    cv2.rectangle(vis_fill_attempt, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    status_text = "❌ REJECTED: No OCR available"
                    font_scale = 0.8
                    font_thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    cv2.rectangle(vis_fill_attempt, 
                                 (x, y + height + 5), 
                                 (x + text_width + 10, y + height + text_height + baseline + 10), 
                                 (255, 255, 255), -1)
                    cv2.putText(vis_fill_attempt, status_text, (x + 5, y + height + text_height + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
                    output_path = Path(steps_dir) / f"{debug_prefix}_02c_flood_fill_attempt_{box_idx + 1}.png"
                    cv2.imwrite(str(output_path), vis_fill_attempt)
                    print(f"         💾 Salvat: {output_path.name} (REJECTED: no OCR)")
            else:
                print(f"         🎯 Găsit cuvântul '{text}' (confidence {conf:.1f}%) - fac flood fill în jurul textului...")
                # Facem flood fill DIN JURUL textului, nu din interiorul lui
                
                # Calculăm o zonă buffer în jurul textului pentru a găsi puncte de start
                buffer_size = max(20, int(max(width, height) * 1.5))  # Buffer de ~1.5x dimensiunea textului
                
                # Identificăm puncte de start în jurul textului (sus, jos, stânga, dreapta, colțuri)
                seed_points = []
                
                # Puncte pe laturile dreptunghiului textului (la distanță buffer_size)
                seed_y_top = max(0, y - buffer_size)
                seed_points.append((center_x, seed_y_top))
                seed_y_bottom = min(h - 1, y + height + buffer_size)
                seed_points.append((center_x, seed_y_bottom))
                seed_x_left = max(0, x - buffer_size)
                seed_points.append((seed_x_left, center_y))
                seed_x_right = min(w - 1, x + width + buffer_size)
                seed_points.append((seed_x_right, center_y))
                
                # Colțuri (diagonal)
                seed_points.append((seed_x_left, seed_y_top))
                seed_points.append((seed_x_right, seed_y_top))
                seed_points.append((seed_x_left, seed_y_bottom))
                seed_points.append((seed_x_right, seed_y_bottom))
                
                # Verificăm dacă există cel puțin un seed point valid în jurul textului (nu verificăm centrul!)
                valid_seed_found = False
                for seed_x, seed_y in seed_points:
                    if 0 <= seed_y < h and 0 <= seed_x < w:
                        if spaces_mask[seed_y, seed_x] == 255:  # Spațiu liber
                            valid_seed_found = True
                            break
                
                if not valid_seed_found:
                    print(f"         ⚠️ Nu s-au găsit seed points valide în jurul textului '{text}'. Skip.")
                    if steps_dir:
                        vis_fill_attempt = cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
                        cv2.circle(vis_fill_attempt, (center_x, center_y), 8, (0, 0, 255), -1)
                        cv2.rectangle(vis_fill_attempt, (x, y), (x + width, y + height), (0, 255, 0), 2)
                        # Desenăm seed points-urile
                        for seed_x, seed_y in seed_points:
                            if 0 <= seed_y < h and 0 <= seed_x < w:
                                if spaces_mask[seed_y, seed_x] == 255:
                                    cv2.circle(vis_fill_attempt, (seed_x, seed_y), 5, (0, 255, 0), -1)
                                else:
                                    cv2.circle(vis_fill_attempt, (seed_x, seed_y), 5, (128, 128, 128), -1)
                        status_text = "❌ REJECTED: No valid seed points around text"
                        font_scale = 0.8
                        font_thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        cv2.rectangle(vis_fill_attempt, 
                                     (x, y + height + 5), 
                                     (x + text_width + 10, y + height + text_height + baseline + 10), 
                                     (255, 255, 255), -1)
                        cv2.putText(vis_fill_attempt, status_text, (x + 5, y + height + text_height + 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
                        output_path = Path(steps_dir) / f"{debug_prefix}_02c_flood_fill_attempt_{box_idx + 1}.png"
                        cv2.imwrite(str(output_path), vis_fill_attempt)
                        print(f"         💾 Salvat: {output_path.name} (REJECTED: no valid seed points)")
                else:
                    # Creez o mască care exclude textul (pentru a nu umple interiorul textului)
                    text_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.rectangle(text_mask, (x, y), (x + width, y + height), 255, -1)
                    
                    # Mască combinată: pereți + text (nu vrem să umplem nici pereții, nici textul)
                    exclusion_mask = cv2.bitwise_or(walls_mask, text_mask)
                    
                    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
                    flood_fill_flags = 4  # 4-conectivitate
                    flood_fill_flags |= cv2.FLOODFILL_MASK_ONLY
                    flood_fill_flags |= (255 << 8)  # Fill value
                    
                    # Folosim overlay-ul combinat pentru flood fill
                    if overlay_combined is not None:
                        overlay_for_fill = cv2.cvtColor(overlay_combined, cv2.COLOR_BGR2GRAY)
                        lo_diff = 30
                        up_diff = 30
                        fill_image = overlay_for_fill.copy()
                        print(f"         🎨 Folosesc overlay combinat pentru flood fill")
                    else:
                        fill_image = spaces_mask.copy()
                        lo_diff = 0
                        up_diff = 0
                        print(f"         ⚠️ Folosesc spaces_mask simplu (overlay indisponibil)")
                    
                    # Facem flood fill din toate punctele din jurul textului
                    combined_filled_region = np.zeros((h, w), dtype=np.uint8)
                    valid_seeds = 0
                    
                    print(f"         🔍 Încerc flood fill din {len(seed_points)} puncte în jurul textului...")
                    
                    for seed_idx, seed_point in enumerate(seed_points):
                        seed_x, seed_y = seed_point
                        
                        if not (0 <= seed_y < h and 0 <= seed_x < w):
                            continue
                        
                        if exclusion_mask[seed_y, seed_x] > 0:
                            continue
                        
                        if spaces_mask[seed_y, seed_x] != 255:
                            continue
                        
                        temp_flood_mask = np.zeros((h + 2, w + 2), np.uint8)
                        try:
                            _, _, _, rect = cv2.floodFill(
                                fill_image.copy(),
                                temp_flood_mask, 
                                seed_point, 
                                128,
                                lo_diff, 
                                up_diff, 
                                flood_fill_flags
                            )
                            
                            temp_filled = (temp_flood_mask[1:h+1, 1:w+1] == 255).astype(np.uint8) * 255
                            temp_filled = cv2.bitwise_and(temp_filled, cv2.bitwise_not(exclusion_mask))
                            combined_filled_region = cv2.bitwise_or(combined_filled_region, temp_filled)
                            valid_seeds += 1
                            
                        except Exception as e:
                            print(f"         ⚠️ Eroare la flood fill din punct {seed_idx + 1}: {e}")
                            continue
                    
                    print(f"         ✅ Flood fill din {valid_seeds}/{len(seed_points)} puncte valide")
                    
                    filled_region = combined_filled_region
                    
                    # Verificăm că nu am umplut peste pereți
                    overlap_with_walls = np.sum((filled_region > 0) & (walls_mask > 0))
                    if overlap_with_walls > 0:
                        print(f"         ⚠️ Flood fill a depășit pereții ({overlap_with_walls} pixeli). Corectez...")
                        filled_region = cv2.bitwise_and(filled_region, cv2.bitwise_not(walls_mask))
                    
                    filled_area = np.count_nonzero(filled_region)
                    img_total_area = h * w
                    filled_ratio = filled_area / float(img_total_area)
                    
                    # Verificăm dacă flood fill-ul este prea mare (probabil a trecut prin pereți și iese din plan)
                    # Pentru terasă, dacă umple > 50% din imagine, probabil a trecut prin pereți
                    if not is_garage and filled_ratio > 0.50:
                        print(f"         ⚠️ Flood fill prea mare ({filled_area}px, {filled_ratio*100:.1f}% din imagine). Probabil a trecut prin pereți și iese din plan. Skip.")
                        if steps_dir:
                            # Salvăm imagine de debug pentru tentativa respinsă
                            vis_fill_attempt = cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
                            cv2.circle(vis_fill_attempt, (center_x, center_y), 8, (0, 0, 255), -1)
                            cv2.rectangle(vis_fill_attempt, (x, y), (x + width, y + height), (0, 255, 0), 2)
                            filled_colored = np.zeros_like(vis_fill_attempt)
                            filled_colored[filled_region > 0] = [0, 255, 255]  # Galben
                            vis_fill_attempt = cv2.addWeighted(vis_fill_attempt, 0.7, filled_colored, 0.3, 0)
                            status_text = f"❌ REJECTED: Area too large ({filled_area}px, {filled_ratio*100:.1f}%)"
                            font_scale = 0.8
                            font_thickness = 2
                            (text_width, text_height), baseline = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                            cv2.rectangle(vis_fill_attempt, 
                                         (x, y + height + 5), 
                                         (x + text_width + 10, y + height + text_height + baseline + 10), 
                                         (255, 255, 255), -1)
                            cv2.putText(vis_fill_attempt, status_text, (x + 5, y + height + text_height + 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
                            output_path = Path(steps_dir) / f"{debug_prefix}_02c_flood_fill_attempt_{box_idx + 1}.png"
                            cv2.imwrite(str(output_path), vis_fill_attempt)
                            print(f"         💾 Salvat: {output_path.name} (REJECTED: area too large)")
                        # Actualizăm și poza cu best detection
                        if steps_dir and best_box_all:
                            best_x, best_y, best_width, best_height, best_text, best_conf = best_box_all
                            best_center_x = best_x + best_width // 2
                            best_center_y = best_y + best_height // 2
                            vis_best_detection = ocr_image.copy() if ocr_image is not None else cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
                            detection_color = (0, 165, 255)  # Portocaliu pentru respins
                            status_label = f"❌ REJECTED ({best_conf:.1f}%)"
                            cv2.rectangle(vis_best_detection, (best_x, best_y), (best_x + best_width, best_y + best_height), detection_color, 3)
                            cv2.circle(vis_best_detection, (best_center_x, best_center_y), 8, (0, 0, 255), -1)
                            filled_colored = np.zeros_like(vis_best_detection)
                            filled_colored[filled_region > 0] = [0, 255, 255]  # Galben
                            vis_best_detection = cv2.addWeighted(vis_best_detection, 0.7, filled_colored, 0.3, 0)
                            label_text = f"{best_text} - {status_label} | Flood Fill: {filled_area}px ({filled_ratio*100:.1f}%) - TOO LARGE"
                            font_scale = max(0.7, best_height / 30.0)
                            font_thickness = max(2, int(font_scale * 2))
                            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                            text_y = max(text_height + 5, best_y - 5)
                            text_x = best_x
                            cv2.rectangle(vis_best_detection, 
                                         (text_x, text_y - text_height - baseline), 
                                         (text_x + text_width + 10, text_y + baseline), 
                                         (255, 255, 255), -1)
                            cv2.putText(vis_best_detection, label_text, (text_x + 5, text_y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, detection_color, font_thickness)
                            output_path = Path(steps_dir) / f"{debug_prefix}_01c_best_detection_with_fill.png"
                            cv2.imwrite(str(output_path), vis_best_detection)
                            print(f"         💾 Salvat: {output_path.name} (best detection: {best_conf:.1f}%, fill: {filled_area}px - REJECTED: too large)")
                        return result
                    
                    # Pentru garaj/carport (care are doar 3 pereți), folosim o abordare geometrică
                    if is_garage:
                        print(f"         🚗 Detectat garaj/carport - calculez distanțele până la pereți...")
                        
                        # Încărcăm imaginile pentru căutare (mai întâi doar pereții, apoi overlay cu ghost image)
                        walls_image = None  # Doar pereții detectați (02_ai_walls_closed.png)
                        overlay_image = None  # Pereții + ghost image (02d_walls_closed_overlay.png)
                        
                        if steps_dir:
                            # 1. Încărcăm imaginea cu doar pereții detectați
                            walls_image_path = Path(steps_dir) / "02_ai_walls_closed.png"
                            if walls_image_path.exists():
                                walls_image = cv2.imread(str(walls_image_path), cv2.IMREAD_GRAYSCALE)
                                if walls_image is not None:
                                    if walls_image.shape[:2] != (h, w):
                                        walls_image = cv2.resize(walls_image, (w, h))
                                    print(f"         📸 Am încărcat 02_ai_walls_closed.png (doar pereții detectați)")
                            
                            # 2. Încărcăm overlay-ul cu ghost image (dacă există)
                            overlay_path = Path(steps_dir) / "02d_walls_closed_overlay.png"
                            if overlay_path.exists():
                                overlay_bgr = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
                                if overlay_bgr is not None:
                                    if overlay_bgr.shape[:2] != (h, w):
                                        overlay_bgr = cv2.resize(overlay_bgr, (w, h))
                                    # Convertim la grayscale pentru analiză
                                    overlay_image = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2GRAY)
                                    print(f"         📸 Am încărcat 02d_walls_closed_overlay.png (pereții + ghost image)")
                        
                        # Fallback: folosim walls_mask dacă nu am putut încărca niciuna
                        if walls_image is None:
                            walls_image = walls_mask.copy()
                            print(f"         ⚠️ Folosesc walls_mask ca fallback pentru walls_image")
                        
                        # Centrul textului
                        center_x = x + width // 2
                        center_y = y + height // 2
                        
                        # Calculăm distanțele până la primul perete în toate direcțiile
                        # În walls_image, pereții sunt albi (255) și spațiile libere sunt negre (0)
                        wall_threshold = 128  # Valori >= 128 = perete
                        
                        distances = {}
                        directions = {
                            'top': (0, -1),      # Sus
                            'bottom': (0, 1),    # Jos
                            'left': (-1, 0),     # Stânga
                            'right': (1, 0)      # Dreapta
                        }
                        
                        max_search_distance = min(w, h) // 2  # Căutăm până la jumătate din imagine
                        
                        for dir_name, (dx, dy) in directions.items():
                            distance = 0
                            found_wall = False
                            
                            for step in range(1, max_search_distance):
                                check_x = center_x + dx * step
                                check_y = center_y + dy * step
                                
                                if not (0 <= check_y < h and 0 <= check_x < w):
                                    break
                                
                                # Verificăm dacă am găsit un perete (valoare >= threshold = perete)
                                pixel_value = walls_image[check_y, check_x]
                                if pixel_value >= wall_threshold:
                                    distance = step
                                    found_wall = True
                                    break
                            
                            if found_wall:
                                distances[dir_name] = distance
                                print(f"         📏 Distanță {dir_name}: {distance}px")
                            else:
                                distances[dir_name] = max_search_distance  # Nu am găsit perete
                                print(f"         ⚠️ Nu am găsit perete în direcția {dir_name}")
                        
                        # Analizăm distanțele pentru a identifica pereții paraleli și cel lipsă
                        dist_values = list(distances.values())
                        dist_sorted = sorted(set(dist_values))
                        
                        print(f"         📊 Distanțe unice: {dist_sorted}")
                        
                        # Identificăm distanța care este mult diferită (probabil unde lipsește peretele)
                        if len(dist_sorted) >= 2:
                            # Găsim distanța care este cel mai diferită
                            # Comparăm fiecare distanță cu celelalte
                            max_diff = 0
                            outlier_dir = None
                            outlier_value = None
                            
                            for dir_name, dist in distances.items():
                                # Calculăm diferența medie față de celelalte distanțe
                                other_dists = [d for d_name, d in distances.items() if d_name != dir_name]
                                avg_diff = sum(abs(dist - d) for d in other_dists) / len(other_dists) if other_dists else 0
                                
                                if avg_diff > max_diff:
                                    max_diff = avg_diff
                                    outlier_dir = dir_name
                                    outlier_value = dist
                            
                            # Dacă am găsit o distanță outlier, căutăm artefacte (2 linii mici) pentru al 4-lea perete
                            # Căutăm în direcția opusă celui de-al 3-lea perete (direcția outlier_dir)
                            if outlier_dir and max_diff > 50:  # Prag pentru a considera că e diferită
                                print(f"         🔍 Distanța outlier: {outlier_dir} = {outlier_value}px (diferență medie: {max_diff:.1f}px)")
                                print(f"         🔍 Caut artefacte (2 linii mici) pentru peretele lipsă în direcția {outlier_dir} (opus celui de-al 3-lea perete)...")
                                
                                # Găsim direcția paralelă (al 3-lea perete - top-bottom sau left-right)
                                if outlier_dir in ['top', 'bottom']:
                                    # Peretele lipsă este sus sau jos, al 3-lea perete este cel paralel
                                    parallel_dir = 'bottom' if outlier_dir == 'top' else 'top'
                                    parallel_value = distances[parallel_dir]
                                    
                                    # Pornim de după textul găsit, în direcția opusă celui de-al 3-lea perete (outlier_dir)
                                    if outlier_dir == 'top':
                                        # Căutăm sus, pornind de după text (center_y) în direcția opusă celui de-al 3-lea perete (bottom)
                                        search_start_y = center_y + parallel_value  # După text, în direcția celui de-al 3-lea perete
                                        search_end_y = center_y - outlier_value  # Până la distanța outlier (direcția opusă)
                                    else:  # bottom
                                        # Căutăm jos, pornind de după text (center_y) în direcția opusă celui de-al 3-lea perete (top)
                                        search_start_y = center_y - parallel_value  # După text, în direcția celui de-al 3-lea perete
                                        search_end_y = center_y + outlier_value  # Până la distanța outlier (direcția opusă)
                                    
                                    # Căutăm 2 linii mici verticale (artefacte) între left_x și right_x
                                    left_x = center_x - distances['left']
                                    right_x = center_x + distances['right']
                                    
                                    # Căutăm linii mici verticale (artefacte de perete)
                                    artifact_found = False
                                    artifact_y = None
                                    
                                    # Parcurgem zona între search_start_y și search_end_y
                                    search_range = range(min(search_start_y, search_end_y), max(search_start_y, search_end_y))
                                    
                                    # Căutăm linii verticale mici (2-5 pixeli înălțime) care sunt conectate la pereții paraleli
                                    min_line_length = 2
                                    max_line_length = 5
                                    
                                    # METODA 1: Căutăm în walls_image (doar pereții detectați)
                                    print(f"         🔍 Metoda 1: Caut în 02_ai_walls_closed.png (doar pereții detectați)...")
                                    for check_y in search_range:
                                        if not (0 <= check_y < h):
                                            continue
                                        
                                        # Verificăm dacă există o linie verticală mică la această poziție
                                        line_pixels = []
                                        for check_x in range(left_x, right_x):
                                            if 0 <= check_x < w and walls_image[check_y, check_x] >= wall_threshold:
                                                line_pixels.append(check_x)
                                        
                                        # Verificăm dacă avem o linie continuă de lungime între min_line_length și max_line_length
                                        if len(line_pixels) >= min_line_length and len(line_pixels) <= max_line_length:
                                            # Verificăm dacă linia este conectată la pereții paraleli (left sau right)
                                            connected_left = False
                                            if left_x > 0:
                                                for conn_y in range(max(0, check_y - 2), min(h, check_y + 3)):
                                                    if walls_image[conn_y, left_x - 1] >= wall_threshold:
                                                        connected_left = True
                                                        break
                                            
                                            # Verificăm conexiunea la dreapta
                                            connected_right = False
                                            if right_x < w - 1:
                                                for conn_y in range(max(0, check_y - 2), min(h, check_y + 3)):
                                                    if walls_image[conn_y, right_x + 1] >= wall_threshold:
                                                        connected_right = True
                                                        break
                                            
                                            # Dacă linia este conectată la ambele pereți paraleli, am găsit artefactul
                                            if connected_left and connected_right:
                                                artifact_found = True
                                                artifact_y = check_y
                                                print(f"         ✅ Găsit artefact în walls_image (linie mică) la y={artifact_y}, conectat la ambele pereți paraleli")
                                                break
                                    
                                    # METODA 2: Dacă nu am găsit în walls_image, căutăm în overlay_image (pereții + ghost image)
                                    if not artifact_found and overlay_image is not None:
                                        print(f"         🔍 Metoda 2: Caut în 02d_walls_closed_overlay.png (pereții + ghost image)...")
                                        # Folosim un threshold mai mic pentru overlay (poate avea valori intermediare)
                                        overlay_threshold = 100  # Threshold mai mic pentru overlay
                                        
                                        for check_y in search_range:
                                            if not (0 <= check_y < h):
                                                continue
                                            
                                            # Verificăm dacă există o linie verticală mică la această poziție
                                            line_pixels = []
                                            for check_x in range(left_x, right_x):
                                                if 0 <= check_x < w and overlay_image[check_y, check_x] >= overlay_threshold:
                                                    line_pixels.append(check_x)
                                            
                                            # Verificăm dacă avem o linie continuă de lungime între min_line_length și max_line_length
                                            if len(line_pixels) >= min_line_length and len(line_pixels) <= max_line_length:
                                                # Verificăm dacă linia este conectată la pereții paraleli (left sau right)
                                                connected_left = False
                                                if left_x > 0:
                                                    for conn_y in range(max(0, check_y - 2), min(h, check_y + 3)):
                                                        if overlay_image[conn_y, left_x - 1] >= overlay_threshold:
                                                            connected_left = True
                                                            break
                                                
                                                # Verificăm conexiunea la dreapta
                                                connected_right = False
                                                if right_x < w - 1:
                                                    for conn_y in range(max(0, check_y - 2), min(h, check_y + 3)):
                                                        if overlay_image[conn_y, right_x + 1] >= overlay_threshold:
                                                            connected_right = True
                                                            break
                                                
                                                # Dacă linia este conectată la ambele pereți paraleli, am găsit artefactul
                                                if connected_left and connected_right:
                                                    artifact_found = True
                                                    artifact_y = check_y
                                                    print(f"         ✅ Găsit artefact în overlay_image (linie mică) la y={artifact_y}, conectat la ambele pereți paraleli")
                                                    break
                                    
                                    # Actualizăm distanța sau aplicăm fallback
                                    if artifact_found and artifact_y is not None:
                                        # Actualizăm distanța outlier cu poziția artefactului
                                        if outlier_dir == 'top':
                                            distances[outlier_dir] = center_y - artifact_y
                                        else:  # bottom
                                            distances[outlier_dir] = artifact_y - center_y
                                        print(f"         ✅ Actualizat {outlier_dir} cu distanța către artefact: {distances[outlier_dir]}px")
                                    else:
                                        # FALLBACK: Dacă nu am găsit artefacte, folosim distanța paralelă
                                        replacement_value = parallel_value
                                        distances[outlier_dir] = replacement_value
                                        print(f"         ⚠️ Nu am găsit artefacte în niciuna dintre imagini, folosesc fallback (distanța paralelă): {replacement_value}px")
                                else:
                                    # Peretele lipsă este stânga sau dreapta, al 3-lea perete este cel paralel
                                    parallel_dir = 'right' if outlier_dir == 'left' else 'left'
                                    parallel_value = distances[parallel_dir]
                                    
                                    # Pornim de după textul găsit, în direcția opusă celui de-al 3-lea perete (outlier_dir)
                                    if outlier_dir == 'left':
                                        # Căutăm stânga, pornind de după text (center_x) în direcția opusă celui de-al 3-lea perete (right)
                                        search_start_x = center_x + parallel_value  # După text, în direcția celui de-al 3-lea perete
                                        search_end_x = center_x - outlier_value  # Până la distanța outlier (direcția opusă)
                                    else:  # right
                                        # Căutăm dreapta, pornind de după text (center_x) în direcția opusă celui de-al 3-lea perete (left)
                                        search_start_x = center_x - parallel_value  # După text, în direcția celui de-al 3-lea perete
                                        search_end_x = center_x + outlier_value  # Până la distanța outlier (direcția opusă)
                                    
                                    # Căutăm 2 linii mici orizontale (artefacte) între top_y și bottom_y
                                    top_y = center_y - distances['top']
                                    bottom_y = center_y + distances['bottom']
                                    
                                    # Căutăm linii mici orizontale (artefacte de perete)
                                    artifact_found = False
                                    artifact_x = None
                                    
                                    # Parcurgem zona între search_start_x și search_end_x
                                    search_range = range(min(search_start_x, search_end_x), max(search_start_x, search_end_x))
                                    
                                    # Căutăm linii orizontale mici (2-5 pixeli lungime) care sunt conectate la pereții paraleli
                                    min_line_length = 2
                                    max_line_length = 5
                                    
                                    # METODA 1: Căutăm în walls_image (doar pereții detectați)
                                    print(f"         🔍 Metoda 1: Caut în 02_ai_walls_closed.png (doar pereții detectați)...")
                                    for check_x in search_range:
                                        if not (0 <= check_x < w):
                                            continue
                                        
                                        # Verificăm dacă există o linie orizontală mică la această poziție
                                        line_pixels = []
                                        for check_y in range(top_y, bottom_y):
                                            if 0 <= check_y < h and walls_image[check_y, check_x] >= wall_threshold:
                                                line_pixels.append(check_y)
                                        
                                        # Verificăm dacă avem o linie continuă de lungime între min_line_length și max_line_length
                                        if len(line_pixels) >= min_line_length and len(line_pixels) <= max_line_length:
                                            # Verificăm dacă linia este conectată la pereții paraleli (top sau bottom)
                                            # Verificăm conexiunea la sus
                                            connected_top = False
                                            if top_y > 0:
                                                for conn_x in range(max(0, check_x - 2), min(w, check_x + 3)):
                                                    if walls_image[top_y - 1, conn_x] >= wall_threshold:
                                                        connected_top = True
                                                        break
                                            
                                            # Verificăm conexiunea la jos
                                            connected_bottom = False
                                            if bottom_y < h - 1:
                                                for conn_x in range(max(0, check_x - 2), min(w, check_x + 3)):
                                                    if walls_image[bottom_y + 1, conn_x] >= wall_threshold:
                                                        connected_bottom = True
                                                        break
                                            
                                            # Dacă linia este conectată la ambele pereți paraleli, am găsit artefactul
                                            if connected_top and connected_bottom:
                                                artifact_found = True
                                                artifact_x = check_x
                                                print(f"         ✅ Găsit artefact în walls_image (linie mică) la x={artifact_x}, conectat la ambele pereți paraleli")
                                                break
                                    
                                    # METODA 2: Dacă nu am găsit în walls_image, căutăm în overlay_image (pereții + ghost image)
                                    if not artifact_found and overlay_image is not None:
                                        print(f"         🔍 Metoda 2: Caut în 02d_walls_closed_overlay.png (pereții + ghost image)...")
                                        # Folosim un threshold mai mic pentru overlay (poate avea valori intermediare)
                                        overlay_threshold = 100  # Threshold mai mic pentru overlay
                                        
                                        for check_x in search_range:
                                            if not (0 <= check_x < w):
                                                continue
                                            
                                            # Verificăm dacă există o linie orizontală mică la această poziție
                                            line_pixels = []
                                            for check_y in range(top_y, bottom_y):
                                                if 0 <= check_y < h and overlay_image[check_y, check_x] >= overlay_threshold:
                                                    line_pixels.append(check_y)
                                            
                                            # Verificăm dacă avem o linie continuă de lungime între min_line_length și max_line_length
                                            if len(line_pixels) >= min_line_length and len(line_pixels) <= max_line_length:
                                                # Verificăm dacă linia este conectată la pereții paraleli (top sau bottom)
                                                # Verificăm conexiunea la sus
                                                connected_top = False
                                                if top_y > 0:
                                                    for conn_x in range(max(0, check_x - 2), min(w, check_x + 3)):
                                                        if overlay_image[top_y - 1, conn_x] >= overlay_threshold:
                                                            connected_top = True
                                                            break
                                                
                                                # Verificăm conexiunea la jos
                                                connected_bottom = False
                                                if bottom_y < h - 1:
                                                    for conn_x in range(max(0, check_x - 2), min(w, check_x + 3)):
                                                        if overlay_image[bottom_y + 1, conn_x] >= overlay_threshold:
                                                            connected_bottom = True
                                                            break
                                                
                                                # Dacă linia este conectată la ambele pereți paraleli, am găsit artefactul
                                                if connected_top and connected_bottom:
                                                    artifact_found = True
                                                    artifact_x = check_x
                                                    print(f"         ✅ Găsit artefact în overlay_image (linie mică) la x={artifact_x}, conectat la ambele pereți paraleli")
                                                    break
                                    
                                    # Actualizăm distanța sau aplicăm fallback
                                    if artifact_found and artifact_x is not None:
                                        # Actualizăm distanța outlier cu poziția artefactului
                                        if outlier_dir == 'left':
                                            distances[outlier_dir] = center_x - artifact_x
                                        else:  # right
                                            distances[outlier_dir] = artifact_x - center_x
                                        print(f"         ✅ Actualizat {outlier_dir} cu distanța către artefact: {distances[outlier_dir]}px")
                                    else:
                                        # FALLBACK: Dacă nu am găsit artefacte, folosim distanța paralelă
                                        replacement_value = parallel_value
                                        distances[outlier_dir] = replacement_value
                                        print(f"         ⚠️ Nu am găsit artefacte în niciuna dintre imagini, folosesc fallback (distanța paralelă): {replacement_value}px")
                            
                            # Acum desenăm forma geometrică bazată pe distanțele corectate
                            print(f"         🎨 Desenez forma geometrică pentru garaj...")
                            
                            # Calculăm colțurile dreptunghiului
                            top_y = center_y - distances['top']
                            bottom_y = center_y + distances['bottom']
                            left_x = center_x - distances['left']
                            right_x = center_x + distances['right']
                            
                            # Creăm o mască pentru forma geometrică
                            geometric_mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.rectangle(geometric_mask, (left_x, top_y), (right_x, bottom_y), 255, -1)
                            
                            # Înlocuim zona umplută cu forma geometrică
                            filled_region = geometric_mask
                            filled_area = np.count_nonzero(filled_region)
                            
                            print(f"         ✅ Formă geometrică: ({left_x}, {top_y}) -> ({right_x}, {bottom_y}), aria: {filled_area}px")
                            
                            # Desenăm peretele lipsă aproximat
                            if outlier_dir == 'top':
                                wall_thickness = max(3, int(min(w, h) * 0.003))
                                cv2.line(result, (left_x, top_y), (right_x, top_y), 255, wall_thickness)
                                print(f"         ✅ Aproximat perete de sus: linie la y={top_y}")
                            elif outlier_dir == 'bottom':
                                wall_thickness = max(3, int(min(w, h) * 0.003))
                                cv2.line(result, (left_x, bottom_y), (right_x, bottom_y), 255, wall_thickness)
                                print(f"         ✅ Aproximat perete de jos: linie la y={bottom_y}")
                            elif outlier_dir == 'left':
                                wall_thickness = max(3, int(min(w, h) * 0.003))
                                cv2.line(result, (left_x, top_y), (left_x, bottom_y), 255, wall_thickness)
                                print(f"         ✅ Aproximat perete din stânga: linie la x={left_x}")
                            elif outlier_dir == 'right':
                                wall_thickness = max(3, int(min(w, h) * 0.003))
                                cv2.line(result, (right_x, top_y), (right_x, bottom_y), 255, wall_thickness)
                                print(f"         ✅ Aproximat perete din dreapta: linie la x={right_x}")
                            
                            # Actualizăm walls_mask cu peretele aproximat
                            walls_mask = result.copy()
                        else:
                            print(f"         ⚠️ Nu am găsit suficiente pereți pentru analiză geometrică")
                    
                    # Verificăm că zona umplută este suficient de mare
                    if not is_garage and filled_area < 1000:
                        print(f"         ⚠️ Zona detectată prea mică ({filled_area} pixeli). Skip.")
                        # Continuăm cu următorul text_box
                        if steps_dir:
                            # Salvăm imagine de debug pentru tentativa respinsă
                            vis_fill_attempt = cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
                            cv2.circle(vis_fill_attempt, (center_x, center_y), 8, (0, 0, 255), -1)
                            cv2.rectangle(vis_fill_attempt, (x, y), (x + width, y + height), (0, 255, 0), 2)
                            status_text = f"❌ REJECTED: Area too small ({filled_area}px)"
                            font_scale = 0.8
                            font_thickness = 2
                            (text_width, text_height), baseline = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                            cv2.rectangle(vis_fill_attempt, 
                                         (x, y + height + 5), 
                                         (x + text_width + 10, y + height + text_height + baseline + 10), 
                                         (255, 255, 255), -1)
                            cv2.putText(vis_fill_attempt, status_text, (x + 5, y + height + text_height + 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
                            output_path = Path(steps_dir) / f"{debug_prefix}_02c_flood_fill_attempt_{box_idx + 1}.png"
                            cv2.imwrite(str(output_path), vis_fill_attempt)
                            print(f"         💾 Salvat: {output_path.name} (REJECTED: area too small)")
                        # Nu continuăm - procesăm doar primul rezultat, deci returnăm
                        return result
                    
                    if is_garage and filled_area < 1000:
                        print(f"         ⚠️ Zona geometrică prea mică ({filled_area} pixeli). Skip.")
                        # Nu continuăm - procesăm doar primul rezultat, deci returnăm
                        if steps_dir:
                            # Salvăm imagine de debug pentru tentativa respinsă
                            vis_fill_attempt = cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
                            cv2.circle(vis_fill_attempt, (center_x, center_y), 8, (0, 0, 255), -1)
                            cv2.rectangle(vis_fill_attempt, (x, y), (x + width, y + height), (0, 255, 0), 2)
                            status_text = f"❌ REJECTED: Geometric area too small ({filled_area}px)"
                            font_scale = 0.8
                            font_thickness = 2
                            (text_width, text_height), baseline = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                            cv2.rectangle(vis_fill_attempt, 
                                         (x, y + height + 5), 
                                         (x + text_width + 10, y + height + text_height + baseline + 10), 
                                         (255, 255, 255), -1)
                            cv2.putText(vis_fill_attempt, status_text, (x + 5, y + height + text_height + 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
                            output_path = Path(steps_dir) / f"{debug_prefix}_02c_flood_fill_attempt_{box_idx + 1}.png"
                            cv2.imwrite(str(output_path), vis_fill_attempt)
                            print(f"         💾 Salvat: {output_path.name} (REJECTED: geometric area too small)")
                        # Nu continuăm - procesăm doar primul rezultat, deci returnăm
                        return result
                    
                    # Pentru garaj, verificăm că zona este suficient de mare
                    if not is_garage and filled_area < 1000:
                        print(f"         🚗 Detectat garaj/carport - verific pereții și corectez zona umplută (ar trebui să fie 3, nu 4)...")
                        
                        # Găsim conturul zonei umplute
                        contours_temp, _ = cv2.findContours(filled_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours_temp:
                            largest_contour = max(contours_temp, key=cv2.contourArea)
                            
                            # Obținem bounding box-ul zonei umplute
                            bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(largest_contour)
                            
                            # Extindem zona umplută pentru a căuta pereții în jur
                            expanded_region = cv2.dilate(filled_region, np.ones((30, 30), np.uint8), iterations=1)
                            
                            # Verificăm marginile zonei umplute pentru a vedea unde sunt pereții
                            # Folosim bounding box-ul pentru a verifica pereții în jur
                            margin = 30
                            top_wall_check = walls_mask[max(0, bbox_y - margin):bbox_y + bbox_h//4, bbox_x:bbox_x + bbox_w] if bbox_y >= margin else np.array([])
                            bottom_wall_check = walls_mask[bbox_y + 3*bbox_h//4:min(h, bbox_y + bbox_h + margin), bbox_x:bbox_x + bbox_w] if bbox_y + bbox_h + margin <= h else np.array([])
                            left_wall_check = walls_mask[bbox_y:bbox_y + bbox_h, max(0, bbox_x - margin):bbox_x + bbox_w//4] if bbox_x >= margin else np.array([])
                            right_wall_check = walls_mask[bbox_y:bbox_y + bbox_h, bbox_x + 3*bbox_w//4:min(w, bbox_x + bbox_w + margin)] if bbox_x + bbox_w + margin <= w else np.array([])
                            
                            walls_detected = []
                            if top_wall_check.size > 0 and np.any(top_wall_check > 0):
                                walls_detected.append('top')
                            if bottom_wall_check.size > 0 and np.any(bottom_wall_check > 0):
                                walls_detected.append('bottom')
                            if left_wall_check.size > 0 and np.any(left_wall_check > 0):
                                walls_detected.append('left')
                            if right_wall_check.size > 0 and np.any(right_wall_check > 0):
                                walls_detected.append('right')
                            
                            print(f"         📐 Pereți detectați: {len(walls_detected)} ({', '.join(walls_detected)})")
                            
                            # Dacă avem exact 3 pereți, "tăiem" zona umplută la marginea unde lipsește peretele
                            if len(walls_detected) == 3:
                                print(f"         ✅ Confirmat: garaj/carport cu 3 pereți. Corectez zona umplută...")
                                
                                # Determinăm care perete lipsește
                                all_sides = ['top', 'bottom', 'left', 'right']
                                missing_wall = [side for side in all_sides if side not in walls_detected][0]
                                
                                # Analizăm forma zonei umplute pentru a determina unde să o "tăiem"
                                contour_points = largest_contour.reshape(-1, 2)
                                
                                # Creăm o mască pentru a "tăia" zona dincolo de marginea lipsă
                                cut_mask = np.ones((h, w), dtype=np.uint8) * 255
                                
                                if missing_wall == 'top':
                                    # Peretele de sus lipsește - "tăiem" zona deasupra
                                    # Găsim punctele de pe marginea de sus a zonei (unde se opresc cei 3 pereți)
                                    top_points = contour_points[contour_points[:, 1] <= bbox_y + bbox_h//3]
                                    if len(top_points) > 0:
                                        # Găsim linia care conectează punctele extreme de sus
                                        min_x = int(np.min(top_points[:, 0]))
                                        max_x = int(np.max(top_points[:, 0]))
                                        # Folosim y-ul minim al punctelor de sus ca limită
                                        cut_y = int(np.min(top_points[:, 1]))
                                        # "Tăiem" tot ce este deasupra acestei linii
                                        cut_mask[:cut_y, :] = 0
                                        print(f"         ✂️ Tăiat zona deasupra y={cut_y}")
                                elif missing_wall == 'bottom':
                                    # Peretele de jos lipsește - "tăiem" zona de jos
                                    bottom_points = contour_points[contour_points[:, 1] >= bbox_y + 2*bbox_h//3]
                                    if len(bottom_points) > 0:
                                        cut_y = int(np.max(bottom_points[:, 1]))
                                        cut_mask[cut_y:, :] = 0
                                        print(f"         ✂️ Tăiat zona de jos y={cut_y}")
                                elif missing_wall == 'left':
                                    # Peretele din stânga lipsește - "tăiem" zona din stânga
                                    left_points = contour_points[contour_points[:, 0] <= bbox_x + bbox_w//3]
                                    if len(left_points) > 0:
                                        cut_x = int(np.min(left_points[:, 0]))
                                        cut_mask[:, :cut_x] = 0
                                        print(f"         ✂️ Tăiat zona din stânga x={cut_x}")
                                elif missing_wall == 'right':
                                    # Peretele din dreapta lipsește - "tăiem" zona din dreapta
                                    right_points = contour_points[contour_points[:, 0] >= bbox_x + 2*bbox_w//3]
                                    if len(right_points) > 0:
                                        cut_x = int(np.max(right_points[:, 0]))
                                        cut_mask[:, cut_x:] = 0
                                        print(f"         ✂️ Tăiat zona din dreapta x={cut_x}")
                                
                                # Aplicăm masca de tăiere pe zona umplută
                                filled_region = cv2.bitwise_and(filled_region, cut_mask)
                                filled_area = np.count_nonzero(filled_region)
                                print(f"         ✅ Zona corectată: {filled_area} pixeli (după tăiere)")
                                
                                # Acum aproximăm peretele lipsă bazându-ne pe conturul corectat
                                contours_corrected, _ = cv2.findContours(filled_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if contours_corrected:
                                    largest_contour_corrected = max(contours_corrected, key=cv2.contourArea)
                                    contour_points_corrected = largest_contour_corrected.reshape(-1, 2)
                                    
                                    if missing_wall == 'top':
                                        top_points = contour_points_corrected[contour_points_corrected[:, 1] <= bbox_y + bbox_h//3]
                                        if len(top_points) > 0:
                                            min_x = int(np.min(top_points[:, 0]))
                                            max_x = int(np.max(top_points[:, 0]))
                                            avg_y = int(np.mean(top_points[:, 1]))
                                            wall_thickness = max(3, int(min(w, h) * 0.003))
                                            cv2.line(result, (min_x, avg_y), (max_x, avg_y), 255, wall_thickness)
                                            print(f"         ✅ Aproximat perete de sus: linie la y={avg_y} între x={min_x}-{max_x}")
                                    elif missing_wall == 'bottom':
                                        bottom_points = contour_points_corrected[contour_points_corrected[:, 1] >= bbox_y + 2*bbox_h//3]
                                        if len(bottom_points) > 0:
                                            min_x = int(np.min(bottom_points[:, 0]))
                                            max_x = int(np.max(bottom_points[:, 0]))
                                            avg_y = int(np.mean(bottom_points[:, 1]))
                                            wall_thickness = max(3, int(min(w, h) * 0.003))
                                            cv2.line(result, (min_x, avg_y), (max_x, avg_y), 255, wall_thickness)
                                            print(f"         ✅ Aproximat perete de jos: linie la y={avg_y} între x={min_x}-{max_x}")
                                    elif missing_wall == 'left':
                                        left_points = contour_points_corrected[contour_points_corrected[:, 0] <= bbox_x + bbox_w//3]
                                        if len(left_points) > 0:
                                            min_y = int(np.min(left_points[:, 1]))
                                            max_y = int(np.max(left_points[:, 1]))
                                            avg_x = int(np.mean(left_points[:, 0]))
                                            wall_thickness = max(3, int(min(w, h) * 0.003))
                                            cv2.line(result, (avg_x, min_y), (avg_x, max_y), 255, wall_thickness)
                                            print(f"         ✅ Aproximat perete din stânga: linie la x={avg_x} între y={min_y}-{max_y}")
                                    elif missing_wall == 'right':
                                        right_points = contour_points_corrected[contour_points_corrected[:, 0] >= bbox_x + 2*bbox_w//3]
                                        if len(right_points) > 0:
                                            min_y = int(np.min(right_points[:, 1]))
                                            max_y = int(np.max(right_points[:, 1]))
                                            avg_x = int(np.mean(right_points[:, 0]))
                                            wall_thickness = max(3, int(min(w, h) * 0.003))
                                            cv2.line(result, (avg_x, min_y), (avg_x, max_y), 255, wall_thickness)
                                            print(f"         ✅ Aproximat perete din dreapta: linie la x={avg_x} între y={min_y}-{max_y}")
                                
                                # Actualizăm walls_mask cu peretele aproximat pentru a fi folosit în completarea golurilor
                                walls_mask = result.copy()
                    
                    # Salvăm imagine suplimentară: detecția cu cel mai mare procent + flood fill (chiar dacă e respins)
                    if steps_dir and best_box_all:
                        best_x, best_y, best_width, best_height, best_text, best_conf = best_box_all
                        best_center_x = best_x + best_width // 2
                        best_center_y = best_y + best_height // 2
                        
                        vis_best_detection = ocr_image.copy() if ocr_image is not None else cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
                        
                        # Desenăm detecția cu cel mai mare procent (verde pentru acceptat, portocaliu pentru respins)
                        filled_ratio_best = filled_area / float(h * w) if filled_area > 0 else 0
                        if filled_area > 1000 and (is_garage or filled_ratio_best <= 0.50):
                            detection_color = (0, 255, 0)  # Verde pentru acceptat
                            status_label = f"✅ ACCEPTED ({best_conf:.1f}%)"
                        else:
                            detection_color = (0, 165, 255)  # Portocaliu pentru respins
                            if filled_area > 0 and filled_ratio_best > 0.50:
                                status_label = f"❌ REJECTED ({best_conf:.1f}%) - Too large"
                            else:
                                status_label = f"❌ REJECTED ({best_conf:.1f}%)"
                        
                        # Desenăm dreptunghiul detecției
                        cv2.rectangle(vis_best_detection, (best_x, best_y), (best_x + best_width, best_y + best_height), detection_color, 3)
                        
                        # Desenăm centrul detecției
                        cv2.circle(vis_best_detection, (best_center_x, best_center_y), 8, (0, 0, 255), -1)
                        
                        # Desenăm flood fill-ul dacă există (galben)
                        if filled_area > 0:
                            filled_colored = np.zeros_like(vis_best_detection)
                            filled_colored[filled_region > 0] = [0, 255, 255]  # Galben
                            vis_best_detection = cv2.addWeighted(vis_best_detection, 0.7, filled_colored, 0.3, 0)
                        
                        # Adăugăm text cu informații
                        label_text = f"{best_text} - {status_label}"
                        if filled_area > 0:
                            label_text += f" | Flood Fill: {filled_area}px"
                        
                        font_scale = max(0.7, best_height / 30.0)
                        font_thickness = max(2, int(font_scale * 2))
                        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        
                        # Fundal pentru text
                        text_y = max(text_height + 5, best_y - 5)
                        text_x = best_x
                        cv2.rectangle(vis_best_detection, 
                                     (text_x, text_y - text_height - baseline), 
                                     (text_x + text_width + 10, text_y + baseline), 
                                     (255, 255, 255), -1)
                        
                        cv2.putText(vis_best_detection, label_text, (text_x + 5, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, detection_color, font_thickness)
                        
                        output_path = Path(steps_dir) / f"{debug_prefix}_01c_best_detection_with_fill.png"
                        cv2.imwrite(str(output_path), vis_best_detection)
                        print(f"         💾 Salvat: {output_path.name} (best detection: {best_conf:.1f}%, fill: {filled_area}px)")
                    
                    # Salvăm imagine de debug pentru TOATE tentativele de flood fill
                    if steps_dir:
                            vis_fill_attempt = cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
                            filled_colored = np.zeros_like(vis_fill_attempt)
                            filled_colored[filled_region > 0] = [0, 255, 255]  # Galben
                            vis_fill_attempt = cv2.addWeighted(vis_fill_attempt, 0.7, filled_colored, 0.3, 0)
                            
                            for seed_point in seed_points:
                                seed_x, seed_y = seed_point
                                if 0 <= seed_y < h and 0 <= seed_x < w:
                                    if exclusion_mask[seed_y, seed_x] == 0 and spaces_mask[seed_y, seed_x] == 255:
                                        cv2.circle(vis_fill_attempt, (seed_x, seed_y), 5, (255, 0, 0), -1)
                                    else:
                                        cv2.circle(vis_fill_attempt, (seed_x, seed_y), 5, (128, 128, 128), -1)
                            
                            cv2.circle(vis_fill_attempt, (center_x, center_y), 8, (0, 0, 255), -1)
                            cv2.rectangle(vis_fill_attempt, (x, y), (x + width, y + height), (0, 255, 0), 2)
                            
                            exclusion_colored = np.zeros_like(vis_fill_attempt)
                            exclusion_colored[exclusion_mask > 0] = [255, 0, 255]  # Magenta
                            vis_fill_attempt = cv2.addWeighted(vis_fill_attempt, 0.8, exclusion_colored, 0.2, 0)
                            
                            filled_ratio_debug = filled_area / float(h * w)
                            status_text = f"Area: {filled_area}px ({filled_ratio_debug*100:.1f}%)"
                            if filled_area > 1000 and (is_garage or filled_ratio_debug <= 0.50):
                                status_text += " ✅ ACCEPTED"
                                status_color = (0, 255, 0)
                            elif filled_area <= 1000:
                                status_text += f" ❌ REJECTED (< 1000px)"
                                status_color = (0, 0, 255)
                            else:
                                status_text += f" ❌ REJECTED (> 50% of image)"
                                status_color = (0, 0, 255)
                            
                            font_scale = 0.8
                            font_thickness = 2
                            (text_width, text_height), baseline = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                            cv2.rectangle(vis_fill_attempt, 
                                         (x, y + height + 5), 
                                         (x + text_width + 10, y + height + text_height + baseline + 10), 
                                         (255, 255, 255), -1)
                            cv2.putText(vis_fill_attempt, status_text, (x + 5, y + height + text_height + 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, font_thickness)
                            
                            if overlap_with_walls > 0:
                                overlap_text = f"Overlap: {overlap_with_walls}px"
                                cv2.putText(vis_fill_attempt, overlap_text, (x + 5, y + height + text_height + baseline + 20), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                            
                            output_path = Path(steps_dir) / f"{debug_prefix}_02c_flood_fill_attempt_{box_idx + 1}.png"
                            cv2.imwrite(str(output_path), vis_fill_attempt)
                            print(f"         💾 Salvat: {output_path.name} (area={filled_area}px, {'ACCEPTED' if filled_area > 1000 else 'REJECTED'})")
                    
                    # Verificăm că zona umplută este suficient de mare dar nu prea mare (pentru terasă)
                    if filled_area > 1000 and (is_garage or filled_ratio <= 0.50):
                        print(f"         🔍 Extrag conturul zonei detectate pentru a completa golurile...")
                        
                        gaps = None
                        wall_border = None
                        contours = None
                        
                        contours, _ = cv2.findContours(filled_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            contour_mask = np.zeros((h, w), dtype=np.uint8)
                            wall_thickness = max(3, int(min(w, h) * 0.003))
                            cv2.drawContours(contour_mask, [largest_contour], -1, 255, wall_thickness)
                            gaps = cv2.bitwise_and(contour_mask, cv2.bitwise_not(walls_mask))
                            walls_to_add = gaps
                            result = cv2.bitwise_or(result, walls_to_add)
                            gaps_area = np.count_nonzero(gaps)
                            print(f"         ✅ Completat {gaps_area} pixeli de goluri în pereți conform conturului")
                        else:
                            print(f"         ⚠️ Nu s-au găsit contururi în zona umplută")
                            kernel_size = max(3, int(min(w, h) * 0.005))
                            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                            filled_dilated = cv2.dilate(filled_region, kernel, iterations=1)
                            wall_border = cv2.subtract(filled_dilated, filled_region)
                            result = cv2.bitwise_or(result, wall_border)
                        
                        rooms_filled += 1
                        print(f"         ✅ Umplut camera '{text}': {filled_area} pixeli")
                        
                        # Vizualizăm zona umplută și golurile completate
                        if steps_dir:
                            vis_fill = cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
                            filled_colored = np.zeros_like(vis_fill)
                            filled_colored[filled_region > 0] = [0, 255, 255]  # Galben
                            vis_fill = cv2.addWeighted(vis_fill, 0.7, filled_colored, 0.3, 0)
                            
                            if contours and len(contours) > 0:
                                largest_contour = max(contours, key=cv2.contourArea)
                                cv2.drawContours(vis_fill, [largest_contour], -1, (255, 0, 0), 2)
                            
                            if gaps is not None:
                                gaps_colored = np.zeros_like(vis_fill)
                                gaps_colored[gaps > 0] = [0, 255, 0]
                                vis_fill = cv2.addWeighted(vis_fill, 0.5, gaps_colored, 0.5, 0)
                            elif wall_border is not None:
                                wall_border_colored = np.zeros_like(vis_fill)
                                wall_border_colored[wall_border > 0] = [0, 255, 0]
                                vis_fill = cv2.addWeighted(vis_fill, 0.5, wall_border_colored, 0.5, 0)
                            
                            cv2.circle(vis_fill, (center_x, center_y), 5, (0, 0, 255), -1)
                            cv2.rectangle(vis_fill, (x, y), (x + width, y + height), (0, 255, 0), 2)
                            
                            output_path = Path(steps_dir) / f"{debug_prefix}_02_{room_name}_fill_{box_idx + 1}.png"
                            cv2.imwrite(str(output_path), vis_fill)
                            print(f"         💾 Salvat: {output_path.name}")
                        
                        print(f"         ✅ Gata! Am umplut camera {room_name}.")
                    else:
                        if filled_area > 0 and filled_ratio > 0.50:
                            print(f"         ⚠️ Zona detectată prea mare ({filled_area} pixeli, {filled_ratio*100:.1f}% din imagine). Probabil a trecut prin pereți. Skip.")
                        else:
                            print(f"         ⚠️ Zona detectată prea mică ({filled_area} pixeli). Skip.")
                        
                        # Salvăm și imaginea cu detecția cu cel mai mare procent (chiar dacă nu s-a făcut flood fill)
                        if best_box_all:
                            best_x, best_y, best_width, best_height, best_text, best_conf = best_box_all
                            best_center_x = best_x + best_width // 2
                            best_center_y = best_y + best_height // 2
                            
                            vis_best_detection = ocr_image.copy() if ocr_image is not None else cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
                            
                            # Desenăm detecția cu cel mai mare procent (portocaliu pentru respins - centru pe perete)
                            detection_color = (0, 165, 255)  # Portocaliu
                            status_label = f"❌ REJECTED: Center on wall ({best_conf:.1f}%)"
                            
                            cv2.rectangle(vis_best_detection, (best_x, best_y), (best_x + best_width, best_y + best_height), detection_color, 3)
                            cv2.circle(vis_best_detection, (best_center_x, best_center_y), 8, (0, 0, 255), -1)
                            
                            label_text = f"{best_text} - {status_label} | No Flood Fill"
                            font_scale = max(0.7, best_height / 30.0)
                            font_thickness = max(2, int(font_scale * 2))
                            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                            
                            text_y = max(text_height + 5, best_y - 5)
                            text_x = best_x
                            cv2.rectangle(vis_best_detection, 
                                         (text_x, text_y - text_height - baseline), 
                                         (text_x + text_width + 10, text_y + baseline), 
                                         (255, 255, 255), -1)
                            
                            cv2.putText(vis_best_detection, label_text, (text_x + 5, text_y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, detection_color, font_thickness)
                            
                            output_path = Path(steps_dir) / f"{debug_prefix}_01c_best_detection_with_fill.png"
                            cv2.imwrite(str(output_path), vis_best_detection)
                            print(f"         💾 Salvat: {output_path.name} (best detection: {best_conf:.1f}%, no flood fill)")
        
        if rooms_filled > 0:
            print(f"         ✅ Umplut {rooms_filled} camere de tip '{room_name}'")
        else:
            print(f"         ⚠️ Nu s-au umplut camere (zone prea mici sau pe pereți)")
        
        # Pas 5: Salvăm rezultatul final
        if steps_dir:
            vis_final = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            diff = cv2.subtract(result, walls_mask)
            diff_colored = np.zeros_like(vis_final)
            diff_colored[diff > 0] = [0, 255, 0]  # Verde pentru pereții noi
            vis_final = cv2.addWeighted(vis_final, 0.7, diff_colored, 0.3, 0)
            cv2.imwrite(str(Path(steps_dir) / f"{debug_prefix}_03_final_result.png"), vis_final)
            print(f"         💾 Salvat: {debug_prefix}_03_final_result.png (verde=pereți noi)")
    
    except Exception as e:
        print(f"         ❌ Eroare la detectarea/umplerea {room_name}: {e}")
        import traceback
        traceback.print_exc()
        return result
    
    return result




# Funcțiile mutate în wall_repair.py:
# - repair_house_walls_with_floodfill
# - bridge_wall_gaps
# - smart_wall_closing
# - get_strict_1px_outline

# Funcția flood_fill_room este folosită în scale_detection.py


# ============================================
# 3D RENDERER (nemodificată)
# ============================================

def render_obj_to_image(vertices_raw, faces_raw, output_image_path, width=1024, height=1024):
    """Randează o previzualizare 3D simplă (PNG, statică)."""
    if not vertices_raw or not faces_raw:
        return
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
    
    angle_y = np.radians(45.0)
    angle_x = np.radians(30.0)
    
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
    
    center = verts.mean(axis=0)
    verts_centered = verts - center
    rotated_verts = verts_centered @ Ry.T @ Rx.T
    
    range_val = np.max(rotated_verts) - np.min(rotated_verts)
    if range_val == 0:
        range_val = 1
    
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
    wall_color = (215, 215, 215)
    edge_color = (60, 60, 60)
    
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
# ✅ NEW: ADAPTIVE TILING FOR LARGE IMAGES (nemodificată)
# ============================================

def _process_with_adaptive_tiling(img, model, device, tile_size=1536, overlap=64):
    """Procesează imagini mari prin tiling adaptiv, cu batch inference (4 pe CUDA, 2 altfel)."""
    h_orig, w_orig = img.shape[:2]
    stride = tile_size - overlap
    n_tiles_h = math.ceil(h_orig / stride)
    n_tiles_w = math.ceil(w_orig / stride)
    total_tiles = n_tiles_w * n_tiles_h
    batch_size = 4 if device.type == "cuda" else 2
    if DEBUG:
        print(f"   🧩 TILING {w_orig}x{h_orig} -> {n_tiles_w}x{n_tiles_h} tiles (batch={batch_size})", flush=True)
    
    full_pred = np.zeros((h_orig, w_orig), dtype=np.float32)
    weight_map = np.zeros((h_orig, w_orig), dtype=np.float32)
    tile_indices = [(i, j) for i in range(n_tiles_h) for j in range(n_tiles_w)]
    tile_count = 0
    
    for batch_start in range(0, len(tile_indices), batch_size):
        batch_inds = tile_indices[batch_start : batch_start + batch_size]
        tensors_list = []
        tile_infos = []
        
        for i, j in batch_inds:
            y_start = i * stride
            x_start = j * stride
            y_end = min(y_start + tile_size, h_orig)
            x_end = min(x_start + tile_size, w_orig)
            tile = img[y_start:y_end, x_start:x_end]
            tile_resized = cv2.resize(tile, (tile_size, tile_size))
            norm_tile = 2 * (tile_resized.astype(np.float32) / 255.0) - 1
            tensor = torch.from_numpy(norm_tile).float().permute(2, 0, 1).unsqueeze(0).to(device)
            tensors_list.append(tensor)
            actual_h, actual_w = y_end - y_start, x_end - x_start
            tile_infos.append((y_start, x_start, y_end, x_end, actual_h, actual_w))
        
        batch_tensor = torch.cat(tensors_list, dim=0)
        with torch.inference_mode():
            output = model(batch_tensor)
            if isinstance(output, dict) and output.get('out') is not None:
                out_slice = output['out'][:, 21:33, :, :]
            else:
                out_slice = output[:, 21:33, :, :]
            pred_batch = torch.argmax(out_slice, dim=1).cpu().numpy()
        
        for idx, (y_start, x_start, y_end, x_end, actual_h, actual_w) in enumerate(tile_infos):
            pred_tile = pred_batch[idx]
            pred_tile_resized = cv2.resize(
                pred_tile.astype('uint8'),
                (actual_w, actual_h),
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
        
        if DEBUG and (tile_count % 4 == 0 or tile_count == total_tiles):
            print(f"      ⏳ Procesat {tile_count}/{total_tiles} tiles...", flush=True)
    
    weight_map[weight_map == 0] = 1
    full_pred = full_pred / weight_map
    pred_mask = np.round(full_pred).astype(np.uint8)
    if DEBUG:
        print(f"   ✅ Tiling complet: {tile_count} tiles", flush=True)
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
    save_debug_steps: bool = True,
    run_phase: int = 0,
    reused_model=None,
    reused_device=None,
    raster_timings: list | None = None,
    brute_force_no_cache: bool = False,
    use_translation_only_raster: bool = True,
    progress_callback: callable | None = None,
) -> dict:
    """
    Rulează detecția CubiCasa + măsurări + 3D Generation.
    ACUM cu Adaptive Strategy: Resize Inteligent pentru imagini mici, Tiling pentru imagini mari.

    run_phase: 0 = tot pipeline-ul; 1 = doar faza 1 (Raster API + AI walls → 02_ai_walls_closed);
              2 = doar faza 2 (brute force + crop + walls from coords); 3 = doar brute force + overlay
              (fără crop, walls_from_coords, garaj etc.). Pentru tandem: rulezi faza 1, apoi faza 2 sau 3.
    reused_model / reused_device: când run_phase=1, poți pasa model/device reîntors de un plan
                                  anterior ca să nu reîncarci modelul.
    """
    global _save_debug_steps
    _save_debug_steps = save_debug_steps
    output_dir = Path(output_dir)
    steps_dir = output_dir / "cubicasa_steps"
    walls_result_from_coords = None
    raster_aligned_walls_1px = None  # când translation-only: mască 1px aliniată la original

    if run_phase in (2, 3):
        # Faza 2 sau 3: nu încărcăm model sau imagine; folosim doar ce e deja pe disc în steps_dir
        # run_phase=3 = doar Raster API + brute force (fără crop, walls_from_coords, garaj, etc.)
        ensure_dir(steps_dir)
        raster_dir = Path(steps_dir) / "raster"
        ac_path = steps_dir / "02_ai_walls_closed.png"
        ai_walls_closed = cv2.imread(str(ac_path), cv2.IMREAD_GRAYSCALE)
        if ai_walls_closed is None:
            raise RuntimeError(f"Lipsă 02_ai_walls_closed.png în {steps_dir} (rulează mai întâi faza 1)")
        # Pentru scale detection etc. folosim 00_original.png din steps_dir
        image_path = steps_dir / "00_original.png"
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Nu pot citi imaginea: {image_path}")
        h_orig, w_orig = img.shape[:2]
        # Continuăm direct la secțiunea brute force (mai jos)
    else:
        # Faza 0 (all) sau 1 (doar Raster API + AI walls)
        image_path = Path(image_path)
        model_weights_path = Path(model_weights_path)
        ensure_dir(steps_dir)
        print(f"   🤖 CubiCasa: Procesez {image_path.name}")
        # 1. SETUP MODEL (sau folosim cel reîntors)
        if reused_model is not None and reused_device is not None:
            model = reused_model
            device = reused_device
            print(f"   ⏳ Folosesc modelul deja încărcat pe {device.type}...")
        else:
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
        if run_phase != 2:
            # 2a. RASTER TO VECTOR API CALL
            if steps_dir:
                try:
                    if progress_callback is not None:
                        progress_callback(0)  # phase1 start (înainte de Raster API)
                    _rt_api = time.time()
                    if DEBUG:
                        print(f"   🔄 Apel RasterScan API pentru vectorizare...")
            
                    # Creăm folderul raster
                    raster_dir = Path(steps_dir) / "raster"
                    raster_dir.mkdir(parents=True, exist_ok=True)
            
                    # Dimensiuni request (scale-down max 1000px) – pentru imagine fără filtru (editor UI)
                    MAX_RASTER_SIDE = 1000
                    h_orig, w_orig = img.shape[:2]
                    scale_factor = 1.0
                    if max(h_orig, w_orig) > MAX_RASTER_SIDE:
                        scale_factor = MAX_RASTER_SIDE / max(h_orig, w_orig)
                    new_w_api = max(1, int(w_orig * scale_factor))
                    new_h_api = max(1, int(h_orig * scale_factor))
                    img_resized_no_filter = cv2.resize(img, (new_w_api, new_h_api), interpolation=cv2.INTER_AREA)
                    try:
                        cv2.imwrite(str(raster_dir / "input_resized_no_filter.png"), img_resized_no_filter)
                    except OSError:
                        pass
            
                    # ✅ PREPROCESARE: Ștergem liniile foarte subțiri înainte de trimitere la RasterScan
                    if DEBUG:
                        print(f"      🧹 Preprocesare imagine: eliminare linii subțiri...")
                    api_img = img.copy()
            
                    # Convertim la grayscale pentru procesare
                    gray = cv2.cvtColor(api_img, cv2.COLOR_BGR2GRAY)
            
                    # Detectăm liniile subțiri folosind morphological operations
                    # Folosim un kernel mic pentru a identifica liniile subțiri
                    kernel_thin = np.ones((3, 3), np.uint8)
                    thinned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_thin, iterations=1)
            
                    # Detectăm contururi și eliminăm cele foarte mici (linii subțiri)
                    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
                    # Creăm o mască pentru liniile subțiri (contururi cu aria mică)
                    thin_lines_mask = np.zeros_like(gray)
                    min_line_area = (gray.shape[0] * gray.shape[1]) * 0.0001  # 0.01% din imagine
            
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area < min_line_area:
                            # Este o linie subțire - o eliminăm
                            cv2.drawContours(thin_lines_mask, [contour], -1, 255, -1)
            
                    # Eliminăm liniile subțiri din imagine
                    api_img = cv2.inpaint(api_img, thin_lines_mask, 3, cv2.INPAINT_TELEA)
                    
                    # Salvăm copia preprocesată doar în mod debug (nu e necesară pentru phase 2)
                    if _save_debug_steps:
                        preprocessed_path = raster_dir / "00_original_preprocessed.png"
                        cv2.imwrite(str(preprocessed_path), api_img)
                        if DEBUG:
                            print(f"      💾 Salvat: {preprocessed_path.name} (preprocesat)")
            
                    # Scale-down la aceleași dimensiuni ca input_resized_no_filter
                    h_api, w_api = api_img.shape[:2]
                    if (w_api, h_api) != (new_w_api, new_h_api):
                        api_img = cv2.resize(api_img, (new_w_api, new_h_api), interpolation=cv2.INTER_AREA)
                        if DEBUG:
                            print(f"      📐 Scale-down pentru Raster (max {MAX_RASTER_SIDE}px): {w_api}x{h_api} -> {new_w_api}x{new_h_api}")

                    # Salvăm dimensiunile request vs original ca să putem converti coordonatele din JSON (request space) în original
                    request_info_path = raster_dir / "raster_request_info.json"
                    with open(request_info_path, 'w') as f:
                        json.dump({
                            "request_w": int(new_w_api), "request_h": int(new_h_api),
                            "original_w": int(w_api), "original_h": int(h_api),
                            "scale_factor": float(scale_factor)
                        }, f, indent=2)

                    # Salvăm imaginea care se trimite la API (raster_request doar în debug)
                    request_png_path = raster_dir / "raster_request.png"
                    if _save_debug_steps:
                        cv2.imwrite(str(request_png_path), api_img)
                    api_img_path = raster_dir / "input_resized.jpg"
                    cv2.imwrite(str(api_img_path), api_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if DEBUG:
                        print(f"      📄 Salvat (trimis la Raster): {request_png_path.name}")

                    # Convertim în base64 (folosim JPEG comprimat)
                    _, buffer = cv2.imencode('.jpg', api_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    image_base64 = base64.b64encode(buffer).decode('utf-8')
                    if DEBUG:
                        print(f"      📦 Dimensiune payload: {len(image_base64) / 1024 / 1024:.2f} MB")
                    if progress_callback is not None:
                        progress_callback(1)  # preprocess done (înainte de apel API)

                    # Apelăm API-ul RasterScan
                    raster_api_key = os.environ.get('RASTER_API_KEY', '')
                    if raster_api_key:
                        url = "https://backend.rasterscan.com/raster-to-vector-base64"
                        payload = {"image": image_base64}
                        headers = {
                            "x-api-key": raster_api_key,
                            "Content-Type": "application/json"
                        }
                        max_attempts = 6
                        raster_valid = False
                        for attempt in range(max_attempts):
                            if attempt > 0:
                                time.sleep(2 * attempt)  # backoff: 2s, 4s
                                print(f"      🔄 Reîncerc RasterScan API ({attempt + 1}/{max_attempts})...")
                            response = requests.post(url, json=payload, headers=headers, timeout=120)
                            if response.status_code != 200:
                                is_retryable = response.status_code >= 500 or response.status_code == 429
                                print(f"      ⚠️ RasterScan API eroare: {response.status_code} - {response.text[:200]}")
                                if is_retryable and attempt < max_attempts - 1:
                                    print(f"      🔄 Eroare retry-abilă (5xx/429), reîncerc...")
                                    continue
                                break
                            result = response.json()
                            print(f"      ✅ RasterScan API răspuns primit")
                        
                            # Salvăm răspunsul JSON
                            json_path = raster_dir / "response.json"
                            with open(json_path, 'w') as f:
                                json.dump(result, f, indent=2)
                            print(f"      📄 Salvat: {json_path.name}")
                        
                            # Dacă răspunsul conține SVG sau alte fișiere, le salvăm
                            if isinstance(result, dict):
                                # Salvăm fiecare câmp relevant
                                for key, value in result.items():
                                    if key == 'svg' and isinstance(value, str):
                                        svg_path = raster_dir / "output.svg"
                                        with open(svg_path, 'w') as f:
                                            f.write(value)
                                        print(f"      📄 Salvat: {svg_path.name}")
                                    elif key == 'dxf' and isinstance(value, str):
                                        # DXF poate fi base64 encoded
                                        dxf_path = raster_dir / "output.dxf"
                                        try:
                                            dxf_data = base64.b64decode(value)
                                            with open(dxf_path, 'wb') as f:
                                                f.write(dxf_data)
                                        except:
                                            with open(dxf_path, 'w') as f:
                                                f.write(value)
                                        print(f"      📄 Salvat: {dxf_path.name}")
                                    elif key == 'image' and isinstance(value, str):
                                        # Imagine procesată (probabil base64)
                                        try:
                                            # Eliminăm prefixul data:image/... dacă există
                                            img_str = value
                                            if ',' in img_str:
                                                img_str = img_str.split(',')[1]
                                            img_data = base64.b64decode(img_str)
                                            img_path = raster_dir / "processed_image.jpg"
                                            with open(img_path, 'wb') as f:
                                                f.write(img_data)
                                            print(f"      📄 Salvat: {img_path.name}")
                                            nparr = np.frombuffer(img_data, np.uint8)
                                            img_decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                            if img_decoded is not None:
                                                raster_out_path = raster_dir / "raster_out.png"
                                                cv2.imwrite(str(raster_out_path), img_decoded)
                                                response_png_path = raster_dir / "raster_response.png"
                                                cv2.imwrite(str(response_png_path), img_decoded)
                                                print(f"      📄 Salvat (răspuns de la Raster): {response_png_path.name}")
                                        except Exception as e:
                                            print(f"      ⚠️ Eroare salvare imagine: {e}")
                            
                                # Fallback: imaginea poate fi în result['data']['image'] (API o returnează acolo)
                                response_png_path = raster_dir / "raster_response.png"
                                if not response_png_path.exists():
                                    for maybe_img in (result.get('processed_image'), result.get('output_image'),
                                                      (result.get('data') or {}).get('image') if isinstance(result.get('data'), dict) else None):
                                        if isinstance(maybe_img, str):
                                            try:
                                                img_str = maybe_img
                                                if ',' in img_str:
                                                    img_str = img_str.split(',')[1]
                                                img_data = base64.b64decode(img_str)
                                                nparr = np.frombuffer(img_data, np.uint8)
                                                img_decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                                if img_decoded is not None:
                                                    cv2.imwrite(str(response_png_path), img_decoded)
                                                    print(f"      📄 Salvat (răspuns de la Raster): {response_png_path.name}")
                                                    break
                                            except Exception:
                                                pass
                            
                                # Generăm imagini din datele vectoriale
                                data = result.get('data', result)
                            
                                # Calculăm factorul de scalare invers pentru overlay pe original
                                # Coordonatele din API sunt pentru imaginea redimensionată
                                def scale_coord(x, y, for_original=False):
                                    """Scalează coordonatele înapoi la original"""
                                    if for_original:
                                        # Scalăm înapoi la dimensiunile originale
                                        orig_x = int(x / scale_factor)
                                        orig_y = int(y / scale_factor)
                                        return orig_x, orig_y
                                    return int(x), int(y)
                            
                                # Dimensiunile pentru imaginile pe coordonate API
                                raster_h, raster_w = new_h_api, new_w_api
                            
                                # 1. Imagine cu pereții (generați din contururile camerelor)
                                # API-ul nu returnează walls separat, dar le putem genera din contururile rooms
                                if 'rooms' in data and data['rooms']:
                                    walls_img = np.zeros((raster_h, raster_w, 3), dtype=np.uint8)
                                    walls_img.fill(255)  # Fundal alb
                                
                                    wall_count = 0
                                    for room in data['rooms']:
                                        points = []
                                        for point in room:
                                            if 'x' in point and 'y' in point:
                                                points.append([int(point['x']), int(point['y'])])
                                    
                                        if len(points) >= 3:
                                            pts = np.array(points, np.int32)
                                            # Desenăm conturul camerei ca pereți
                                            cv2.polylines(walls_img, [pts], True, (0, 0, 0), 3)
                                            wall_count += len(points)
                                
                                    walls_path = raster_dir / "walls.png"
                                    cv2.imwrite(str(walls_path), walls_img)
                                    print(f"      📄 Salvat: {walls_path.name} ({wall_count} segmente perete din {len(data['rooms'])} camere)")
                            
                                # ⚠️ NU mai generăm rooms.png aici - va fi generat DUPĂ validarea pereților în raster_processing
                            
                                # 3. Imagine cu deschiderile (uși/ferestre - API nu face distincție)
                                if 'doors' in data and data['doors']:
                                    doors_img = np.zeros((raster_h, raster_w, 3), dtype=np.uint8)
                                    doors_img.fill(255)  # Fundal alb
                                
                                    for idx, door in enumerate(data['doors']):
                                        if 'bbox' in door and len(door['bbox']) == 4:
                                            x1, y1, x2, y2 = map(int, door['bbox'])
                                            width = x2 - x1
                                            height = y2 - y1
                                        
                                            # Încercăm să determinăm tipul bazat pe dimensiuni
                                            # Ferestrele tind să fie mai late și mai puțin înalte
                                            aspect = width / max(1, height)
                                            if aspect > 2.5 or (width > 60 and height < 30):
                                                label = "Window"
                                                color_fill = (200, 220, 255)  # Albastru deschis pentru ferestre
                                                color_border = (150, 180, 220)
                                            else:
                                                label = "Door"
                                                color_fill = (0, 150, 255)  # Portocaliu pentru uși
                                                color_border = (0, 100, 200)
                                        
                                            cv2.rectangle(doors_img, (x1, y1), (x2, y2), color_fill, -1)
                                            cv2.rectangle(doors_img, (x1, y1), (x2, y2), color_border, 2)
                                        
                                            # Adăugăm etichetă
                                            font = cv2.FONT_HERSHEY_SIMPLEX
                                            font_scale = 0.35
                                            thickness = 1
                                            cv2.putText(doors_img, label, (x1, y1 - 5 if y1 > 20 else y2 + 12),
                                                       font, font_scale, (0, 0, 150), thickness)
                                
                                    doors_path = raster_dir / "doors.png"
                                    cv2.imwrite(str(doors_path), doors_img)
                                    print(f"      📄 Salvat: {doors_path.name} ({len(data['doors'])} deschideri uși/ferestre)")
                            
                                # 4. Imagine combinată (pereți + camere + uși)
                                combined_img = np.zeros((raster_h, raster_w, 3), dtype=np.uint8)
                                combined_img.fill(255)
                            
                                # Desenăm camerele mai întâi (fundal)
                                if 'rooms' in data and data['rooms']:
                                    for idx, room in enumerate(data['rooms']):
                                        color = room_colors[idx % len(room_colors)] if 'room_colors' in dir() else (220, 220, 220)
                                        points = []
                                        for point in room:
                                            if 'x' in point and 'y' in point:
                                                points.append([int(point['x']), int(point['y'])])
                                        if len(points) >= 3:
                                            pts = np.array(points, np.int32)
                                            cv2.fillPoly(combined_img, [pts], color)
                            
                                # Desenăm pereții (din contururile camerelor)
                                if 'rooms' in data and data['rooms']:
                                    for room in data['rooms']:
                                        points = []
                                        for point in room:
                                            if 'x' in point and 'y' in point:
                                                points.append([int(point['x']), int(point['y'])])
                                        if len(points) >= 3:
                                            pts = np.array(points, np.int32)
                                            cv2.polylines(combined_img, [pts], True, (0, 0, 0), 3)
                            
                                # Desenăm ușile
                                if 'doors' in data and data['doors']:
                                    for door in data['doors']:
                                        if 'bbox' in door and len(door['bbox']) == 4:
                                            x1, y1, x2, y2 = map(int, door['bbox'])
                                            cv2.rectangle(combined_img, (x1, y1), (x2, y2), (0, 150, 255), -1)
                                            cv2.rectangle(combined_img, (x1, y1), (x2, y2), (0, 100, 200), 2)
                            
                                combined_path = raster_dir / "combined.png"
                                cv2.imwrite(str(combined_path), combined_img)
                                print(f"      📄 Salvat: {combined_path.name}")
                            
                                # 5. Overlay pe imaginea ORIGINALĂ (cu coordonate scalate corect)
                                # Folosim imaginea originală, nu cea redimensionată
                                overlay_img = img.copy()
                            
                                # Overlay camere cu transparență
                                if 'rooms' in data and data['rooms']:
                                    rooms_overlay = np.zeros_like(overlay_img)
                                    for idx, room in enumerate(data['rooms']):
                                        color = room_colors[idx % len(room_colors)] if 'room_colors' in dir() else (220, 220, 220)
                                        points = []
                                        for point in room:
                                            if 'x' in point and 'y' in point:
                                                ox, oy = scale_coord(point['x'], point['y'], for_original=True)
                                                # Clamp la dimensiunile imaginii
                                                ox = max(0, min(ox, h_orig - 1))
                                                oy = max(0, min(oy, w_orig - 1))
                                                points.append([ox, oy])
                                        if len(points) >= 3:
                                            pts = np.array(points, np.int32)
                                            cv2.fillPoly(rooms_overlay, [pts], color)
                                
                                    # Blend cu original
                                    mask = (rooms_overlay.sum(axis=2) > 0).astype(np.uint8)
                                    mask = np.stack([mask, mask, mask], axis=2)
                                    overlay_img = np.where(mask, cv2.addWeighted(overlay_img, 0.6, rooms_overlay, 0.4, 0), overlay_img)
                            
                                # Desenăm pereții din contururile camerelor (scalați la original)
                                if 'rooms' in data and data['rooms']:
                                    for room in data['rooms']:
                                        points = []
                                        for point in room:
                                            if 'x' in point and 'y' in point:
                                                ox, oy = scale_coord(point['x'], point['y'], for_original=True)
                                                ox = max(0, min(ox, w_orig - 1))
                                                oy = max(0, min(oy, h_orig - 1))
                                                points.append([ox, oy])
                                        if len(points) >= 3:
                                            pts = np.array(points, np.int32)
                                            cv2.polylines(overlay_img, [pts], True, (0, 0, 255), 2)
                            
                                # Desenăm deschiderile (uși/ferestre) scalate la original, cu etichete
                                if 'doors' in data and data['doors']:
                                    for door in data['doors']:
                                        if 'bbox' in door and len(door['bbox']) == 4:
                                            x1, y1, x2, y2 = door['bbox']
                                            ox1, oy1 = scale_coord(x1, y1, for_original=True)
                                            ox2, oy2 = scale_coord(x2, y2, for_original=True)
                                        
                                            # Determinăm tipul bazat pe dimensiuni
                                            width = abs(ox2 - ox1)
                                            height = abs(oy2 - oy1)
                                            aspect = width / max(1, height)
                                            if aspect > 2.5 or (width > 60 and height < 30):
                                                label = "Win"
                                                color = (220, 180, 0)  # Cyan pentru ferestre
                                            else:
                                                label = "Door"
                                                color = (255, 100, 0)  # Portocaliu pentru uși
                                        
                                            cv2.rectangle(overlay_img, (ox1, oy1), (ox2, oy2), color, 2)
                                        
                                            # Etichetă
                                            font = cv2.FONT_HERSHEY_SIMPLEX
                                            cv2.putText(overlay_img, label, (ox1, oy1 - 5 if oy1 > 20 else oy2 + 15),
                                                       font, 0.4, color, 1)
                            
                                overlay_path = raster_dir / "overlay.png"
                                cv2.imwrite(str(overlay_path), overlay_img)
                                print(f"      📄 Salvat: {overlay_path.name}")
                            
                                # 6. RENDER 3D IZOMETRIC (îmbunătățit)
                                print(f"      🎨 Generez render 3D izometric...")
                            
                                # Parametri pentru proiecția izometrică
                                wall_height = 60  # Înălțimea pereților în pixeli
                            
                                # Calculăm bounding box-ul datelor pentru centrare
                                all_points = []
                                if 'walls' in data and data['walls']:
                                    for wall in data['walls']:
                                        if 'position' in wall:
                                            for pt in wall['position']:
                                                all_points.append(pt)
                                if 'rooms' in data and data['rooms']:
                                    for room in data['rooms']:
                                        for point in room:
                                            if 'x' in point and 'y' in point:
                                                all_points.append([point['x'], point['y']])
                            
                                if all_points:
                                    all_points = np.array(all_points)
                                    min_x, min_y = all_points.min(axis=0)
                                    max_x, max_y = all_points.max(axis=0)
                                
                                    # Scalăm pentru a încăpea în canvas
                                    data_w = max_x - min_x
                                    data_h = max_y - min_y
                                
                                    # Canvas size
                                    canvas_w = int(data_w * 1.5 + data_h * 0.5 + 200)
                                    canvas_h = int(data_h * 0.7 + wall_height + 150)
                                
                                    # Offset pentru centrare
                                    offset_x = 50
                                    offset_y = wall_height + 30
                                
                                    # Funcție pentru transformare izometrică (proiecție oblică)
                                    def to_iso_3d(x, y, z=0):
                                        # Normalizăm coordonatele
                                        nx = x - min_x
                                        ny = y - min_y
                                        # Proiecție izometrică simplificată
                                        iso_x = int(offset_x + nx + ny * 0.4)
                                        iso_y = int(offset_y + ny * 0.6 - z)
                                        return (iso_x, iso_y)
                                
                                    # Canvas cu fundal gradient (cer)
                                    iso_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                                    for row in range(canvas_h):
                                        ratio = row / canvas_h
                                        b = int(250 - ratio * 20)
                                        g = int(248 - ratio * 15)
                                        r = int(245 - ratio * 10)
                                        iso_img[row, :] = [b, g, r]
                                
                                    # Culori pentru podele camerelor
                                    iso_room_colors = [
                                        (210, 225, 210),  # Verde deschis
                                        (210, 210, 225),  # Albastru deschis
                                        (225, 210, 210),  # Roșu deschis
                                        (225, 225, 210),  # Galben deschis
                                        (210, 225, 225),  # Cyan deschis
                                        (225, 210, 225),  # Magenta deschis
                                        (220, 220, 220),  # Gri deschis
                                    ]
                                
                                    # Desenăm podelele camerelor (sortate de la spate la față)
                                    if 'rooms' in data and data['rooms']:
                                        sorted_rooms = []
                                        for idx, room in enumerate(data['rooms']):
                                            points = []
                                            for point in room:
                                                if 'x' in point and 'y' in point:
                                                    points.append([int(point['x']), int(point['y'])])
                                            if len(points) >= 3:
                                                # Folosim Y minim pentru sortare (spate -> față)
                                                min_room_y = min(p[1] for p in points)
                                                sorted_rooms.append((min_room_y, idx, points))
                                    
                                        sorted_rooms.sort(key=lambda x: x[0])
                                    
                                        for min_room_y, idx, points in sorted_rooms:
                                            color = iso_room_colors[idx % len(iso_room_colors)]
                                            floor_pts = np.array([to_iso_3d(p[0], p[1], 0) for p in points], np.int32)
                                            cv2.fillPoly(iso_img, [floor_pts], color)
                                            cv2.polylines(iso_img, [floor_pts], True, (180, 180, 180), 1)
                                
                                    # Desenăm pereții 3D (sortați de la spate la față)
                                    if 'walls' in data and data['walls']:
                                        sorted_walls = []
                                        for wall in data['walls']:
                                            if 'position' in wall and len(wall['position']) >= 2:
                                                pt1 = wall['position'][0]
                                                pt2 = wall['position'][1]
                                                # Sortăm după Y minim
                                                min_wall_y = min(pt1[1], pt2[1])
                                                sorted_walls.append((min_wall_y, pt1, pt2))
                                    
                                        sorted_walls.sort(key=lambda x: x[0])
                                    
                                        for min_wall_y, pt1, pt2 in sorted_walls:
                                            x1, y1 = int(pt1[0]), int(pt1[1])
                                            x2, y2 = int(pt2[0]), int(pt2[1])
                                        
                                            # Punctele peretelui 3D
                                            bl = to_iso_3d(x1, y1, 0)  # bottom-left
                                            br = to_iso_3d(x2, y2, 0)  # bottom-right
                                            tl = to_iso_3d(x1, y1, wall_height)  # top-left
                                            tr = to_iso_3d(x2, y2, wall_height)  # top-right
                                        
                                            # Determinăm culoarea bazat pe orientare
                                            dx = abs(x2 - x1)
                                            dy = abs(y2 - y1)
                                        
                                            if dy < dx:  # Perete orizontal
                                                wall_color = (230, 230, 230)  # Mai luminos
                                            else:  # Perete vertical
                                                wall_color = (200, 200, 200)  # Mai întunecat
                                        
                                            # Desenăm fața frontală a peretelui
                                            wall_pts = np.array([bl, br, tr, tl], np.int32)
                                            cv2.fillPoly(iso_img, [wall_pts], wall_color)
                                            cv2.polylines(iso_img, [wall_pts], True, (120, 120, 120), 1)
                                        
                                            # Partea de sus a peretelui (opțional, pentru grosime)
                                            thickness_offset = 6
                                            if dy < dx:  # Perete orizontal - adăugăm grosime în Y
                                                tl2 = to_iso_3d(x1, y1 + thickness_offset, wall_height)
                                                tr2 = to_iso_3d(x2, y2 + thickness_offset, wall_height)
                                                top_pts = np.array([tl, tr, tr2, tl2], np.int32)
                                                cv2.fillPoly(iso_img, [top_pts], (240, 240, 240))
                                                cv2.polylines(iso_img, [top_pts], True, (150, 150, 150), 1)
                                            else:  # Perete vertical - adăugăm grosime în X
                                                tl2 = to_iso_3d(x1 + thickness_offset, y1, wall_height)
                                                tr2 = to_iso_3d(x2 + thickness_offset, y2, wall_height)
                                                top_pts = np.array([tl, tr, tr2, tl2], np.int32)
                                                cv2.fillPoly(iso_img, [top_pts], (240, 240, 240))
                                                cv2.polylines(iso_img, [top_pts], True, (150, 150, 150), 1)
                                
                                    iso_path = raster_dir / "render_3d.png"
                                    cv2.imwrite(str(iso_path), iso_img)
                                    print(f"      📄 Salvat: {iso_path.name}")
                            
                                # Afișăm statistici
                                if 'area' in data:
                                    print(f"      📊 Arie totală: {data['area']}")
                                if 'perimeter' in data:
                                    print(f"      📊 Perimetru: {data['perimeter']:.2f}")
                            
                                # Notă: Brute force pentru api_walls_mask va fi executat mai târziu,
                                # după generarea 02_ai_walls_closed.png (vezi linia ~2801)
                                # Aici doar generăm api_walls_mask.png
                                api_walls_mask = None
                                try:
                                    # 1. Generăm api_walls_mask.png din imaginea procesată de API
                                    if 'image' in result.get('data', result):
                                        img_str = result['data']['image']
                                        if ',' in img_str:
                                            img_str = img_str.split(',')[1]
                                        img_data = base64.b64decode(img_str)
                                        nparr = np.frombuffer(img_data, np.uint8)
                                        api_processed_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                    
                                        # Size-up imaginea de la API la original cu același factor pe ambele axe (păstrăm aspect ratio)
                                        if scale_factor < 1.0 and api_processed_img is not None:
                                            scale_up = 1.0 / scale_factor
                                            target_w = max(1, int(round(api_processed_img.shape[1] * scale_up)))
                                            target_h = max(1, int(round(api_processed_img.shape[0] * scale_up)))
                                            api_processed_img = cv2.resize(api_processed_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                                            print(f"      📐 Size-up imagine API la original (aspect ratio păstrat): {api_processed_img.shape[1]}x{api_processed_img.shape[0]}")
                                    
                                        # Detectăm pereții din imaginea API (gri, nu colorați)
                                        api_gray = cv2.cvtColor(api_processed_img, cv2.COLOR_BGR2GRAY)
                                        api_hsv = cv2.cvtColor(api_processed_img, cv2.COLOR_BGR2HSV)
                                        saturation = api_hsv[:, :, 1]
                                    
                                        # Pixelii cu saturație mică și gri mediu sunt pereți
                                        api_walls_mask = ((api_gray > 100) & (api_gray < 180) & (saturation < 30)).astype(np.uint8) * 255
                                    
                                        api_walls_path = raster_dir / "api_walls_mask.png"
                                        cv2.imwrite(str(api_walls_path), api_walls_mask)
                                        print(f"      📄 Salvat: {api_walls_path.name}")
                                        # Actualizăm raster_request_info cu dimensiunile reale ale mastii (pentru conversie request→original corectă)
                                        try:
                                            request_info_path = raster_dir / "raster_request_info.json"
                                            if request_info_path.exists():
                                                with open(request_info_path, 'r') as f:
                                                    req_info = json.load(f)
                                                req_info["mask_w"] = int(api_walls_mask.shape[1])
                                                req_info["mask_h"] = int(api_walls_mask.shape[0])
                                                with open(request_info_path, 'w') as f:
                                                    json.dump(req_info, f, indent=2)
                                        except Exception:
                                            pass
                                        # Notă: Brute force pentru api_walls_mask va fi executat mai târziu,
                                        # după generarea 02_ai_walls_closed.png (vezi linia ~2814)
                                        
                                except Exception as e:
                                    import traceback
                                    print(f"      ⚠️ Eroare brute force: {e}")
                                    traceback.print_exc()
                            
                                # Mască acceptată (fără validare room ratio)
                                if api_walls_mask is not None:
                                    break
                            else:
                                print(f"      ⚠️ RasterScan API eroare: {response.status_code} - {response.text[:200]}")
                    else:
                        print(f"      ⚠️ RASTER_API_KEY nu este setat în environment")
                
                except requests.exceptions.Timeout:
                    print(f"      ⚠️ RasterScan API timeout (120s)")
                except Exception as e:
                    print(f"      ⚠️ RasterScan API eroare: {e}")
                finally:
                    if raster_timings is not None:
                        raster_timings.append(("Raster P1: API + imagini", time.time() - _rt_api))
                    if progress_callback is not None:
                        progress_callback(2)  # raster API response primit
    
        # ✅ ADAPTIVE STRATEGY (tot în run_phase != 2)
        if raster_timings is not None:
            _rt_ai = time.time()
        LARGE_IMAGE_THRESHOLD = 4000  # px – peste acest prag folosim tiling (mai mare = mai puține planuri cu tiling)
        USE_TILING = h_orig > LARGE_IMAGE_THRESHOLD or w_orig > LARGE_IMAGE_THRESHOLD
        
        if USE_TILING:
            if DEBUG:
                print(f"   🚀 LARGE IMAGE ({w_orig}x{h_orig}) -> Folosesc TILING ADAPTIV")
            pred_mask = _process_with_adaptive_tiling(
                img,
                model,
                device,
                tile_size=1536,
                overlap=64
            )
        else:
            if DEBUG:
                print(f"   🚀 SMALL IMAGE ({w_orig}x{h_orig}) -> Resize Standard la 1536px")
            max_dim = 1536
            scale = min(max_dim / w_orig, max_dim / h_orig, 1.0)
            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)
            if DEBUG:
                print(f"      Resize: {w_orig}x{h_orig} -> {new_w}x{new_h}")
            input_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            norm_img = 2 * (input_img.astype(np.float32) / 255.0) - 1
            tensor = torch.from_numpy(norm_img).float().permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.inference_mode():
                output = model(tensor)
                if isinstance(output, dict) and output.get('out') is not None:
                    out_slice = output['out'][:, 21:33, :, :]
                else:
                    out_slice = output[:, 21:33, :, :]
                pred_mask_small = torch.argmax(out_slice, dim=1).squeeze().cpu().numpy()
            pred_mask = cv2.resize(
                pred_mask_small.astype('uint8'),
                (w_orig, h_orig),
                interpolation=cv2.INTER_NEAREST
            )
    
    if run_phase not in (2, 3):
        # 3. EXTRACT WALLS & FILTER THIN LINES (doar phase 0/1; phase 2/3 au deja ai_walls_closed pe disc)
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
    
        # ✅ CLOSING ADAPTIV ULTRA-PUTERNIC (UNIFICAT) - ÎMBUNĂTĂȚIT PENTRU VPS
        print("      🔗 Închid găuri (closing adaptiv)...")
    
        if min_dim > 2500:
            # Imagini mari: closing redus pentru a nu uni pereți paraleli
            close_kernel_size = max(5, int(min_dim * 0.003))  # Redus înapoi la 0.3%
            close_iterations = 2  # Redus înapoi la 2 iterații
            print(f"         Mode: LARGE IMAGE → Close: {close_kernel_size}px (0.3%), iter={close_iterations}")
        else:
            # Imagini mici: closing redus pentru a nu uni pereți paraleli
            close_kernel_size = max(12, int(min_dim * 0.010))  # Redus înapoi la 1.0%
            close_iterations = 5  # Redus înapoi la 5 iterații
            print(f"         Mode: SMALL IMAGE → Close: {close_kernel_size}px (1.0%), iter={close_iterations}")
    
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
        
        # Acum, `ai_walls_raw` conține deja closing-ul adaptiv
        # Redus agresivitatea: skip bridge_wall_gaps pentru a nu uni pereți paraleli
        print(f"      🔧 Skip extra bridging (adaptive closing is enough, bridge_wall_gaps e prea agresiv)")
        ai_walls_closed = ai_walls_raw.copy()
        
        save_step("02_ai_walls_closed", ai_walls_closed, str(steps_dir))
        if raster_timings is not None:
            raster_timings.append(("Raster P1: AI walls", time.time() - _rt_ai))
        if progress_callback is not None:
            progress_callback(3)  # phase1 end (AI walls done)
        if run_phase == 1:
            return {"model": model, "device": device}
    
    # ============================================================
    # 🔥 BRUTE FORCE ALGORITM PENTRU TRANSFORMARE COORDONATE
    # (După generarea 02_ai_walls_closed.png)
    # Folosim cache pentru a evita rularea de mai multe ori
    # ============================================================
    best_config = None
    if steps_dir:
        raster_dir = Path(steps_dir) / "raster"
        if raster_dir.exists():
            # Verificăm dacă există deja configurația salvată (cache), dacă nu e dezactivat
            if progress_callback is not None and run_phase == 2:
                progress_callback(0)  # phase2 start
            config_path = raster_dir / "brute_force_best_config.json"
            _rt_bf = time.time() if raster_timings is not None else None
            if not brute_force_no_cache and config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        best_config = json.load(f)
                    print(f"\n      ✅ Folosesc configurația brute force din cache")
                    # Rulează totuși cei 3 algoritmi (Log-Polar, ECC, Pyramid) și salvează rezultatele
                    api_walls_path = raster_dir / "api_walls_mask.png"
                    orig_walls_path = Path(steps_dir) / "02_ai_walls_closed.png"
                    if api_walls_path.exists() and orig_walls_path.exists():
                        _api = cv2.imread(str(api_walls_path), cv2.IMREAD_GRAYSCALE)
                        _orig = cv2.imread(str(orig_walls_path), cv2.IMREAD_GRAYSCALE)
                        if _api is not None and _orig is not None:
                            _, binary_api = cv2.threshold(_api, 127, 255, cv2.THRESH_BINARY)
                            _, binary_orig = cv2.threshold(_orig, 127, 255, cv2.THRESH_BINARY)
                            run_extra_alignment_methods(raster_dir, binary_orig, binary_api)
                except Exception as e:
                    print(f"      ⚠️ Eroare la citirea cache-ului: {e}")
                    best_config = None
            
            # Dacă nu există cache (sau e dezactivat), rulăm brute force
            translation_only_config = None
            if best_config is None and use_translation_only_raster:
                try:
                    request_info_path = raster_dir / "raster_request_info.json"
                    orig_walls_path = Path(steps_dir) / "02_ai_walls_closed.png"
                    if request_info_path.exists() and orig_walls_path.exists():
                        with open(request_info_path, "r") as f:
                            ri = json.load(f)
                        req_w = int(ri.get("request_w", 0))
                        req_h = int(ri.get("request_h", 0))
                        orig_walls = cv2.imread(str(orig_walls_path), cv2.IMREAD_GRAYSCALE)
                        if orig_walls is not None and req_w > 0 and req_h > 0:
                            ref_request = cv2.resize(orig_walls, (req_w, req_h), interpolation=cv2.INTER_NEAREST)
                            _, ref_bin = cv2.threshold(ref_request, 127, 255, cv2.THRESH_BINARY)
                            api_request = build_api_walls_mask_from_json(raster_dir, req_w, req_h)
                            if api_request is not None:
                                _, api_bin = cv2.threshold(api_request, 127, 255, cv2.THRESH_BINARY)
                                if ref_bin.shape == api_bin.shape:
                                    translation_only_config = brute_force_translation_only(ref_bin, api_bin)
                                    if translation_only_config is not None:
                                        brute_steps_dir = raster_dir / "brute_steps"
                                        brute_steps_dir.mkdir(parents=True, exist_ok=True)
                                        with open(brute_steps_dir / "translation_only_config.json", "w") as f:
                                            json.dump(translation_only_config, f, indent=2)
                                        tx, ty = translation_only_config["position"]
                                        overlay_req = np.zeros((*ref_bin.shape[:2], 3), dtype=np.uint8)
                                        overlay_req[:, :, 2] = ref_bin
                                        x_dst, y_dst = max(0, tx), max(0, ty)
                                        x_src, y_src = max(0, -tx), max(0, -ty)
                                        w_c = min(ref_bin.shape[1] - x_dst, api_bin.shape[1] - x_src)
                                        h_c = min(ref_bin.shape[0] - y_dst, api_bin.shape[0] - y_src)
                                        if w_c > 0 and h_c > 0:
                                            overlay_req[y_dst:y_dst+h_c, x_dst:x_dst+w_c, 0] = api_bin[y_src:y_src+h_c, x_src:x_src+w_c]
                                            overlay_req[y_dst:y_dst+h_c, x_dst:x_dst+w_c, 1] = api_bin[y_src:y_src+h_c, x_src:x_src+w_c]
                                        cv2.imwrite(str(brute_steps_dir / "translation_only_overlay.png"), overlay_req)
                                        # Poza originală (request space) cu skeletul de pereți (masca roșie) suprapus
                                        base_req = cv2.imread(str(raster_dir / "input_resized_no_filter.png"))
                                        if base_req is None:
                                            base_req = cv2.imread(str(raster_dir / "input_resized.jpg"))
                                        if base_req is None:
                                            base_req = cv2.imread(str(raster_dir / "raster_request.png"))
                                        if base_req is not None and base_req.shape[:2] == ref_bin.shape[:2]:
                                            base_with_skeleton = base_req.copy()
                                            base_with_skeleton[:, :, 2] = np.where(ref_bin > 0, 255, base_with_skeleton[:, :, 2])
                                            base_with_skeleton[:, :, 0] = np.where(ref_bin > 0, 0, base_with_skeleton[:, :, 0])
                                            base_with_skeleton[:, :, 1] = np.where(ref_bin > 0, 0, base_with_skeleton[:, :, 1])
                                            cv2.imwrite(str(brute_steps_dir / "base_with_walls_skeleton.png"), base_with_skeleton)
                                            # Plan + masca Raster în albastru la EXACT aceeași poziție ca în translation_only_overlay (tx, ty)
                                            base_with_blue = base_req.copy()
                                            if w_c > 0 and h_c > 0:
                                                api_region = api_bin[y_src:y_src + h_c, x_src:x_src + w_c]
                                                blue_mask = api_region > 0
                                                base_with_blue[y_dst:y_dst + h_c, x_dst:x_dst + w_c][blue_mask] = [255, 0, 0]  # BGR blue
                                            cv2.imwrite(str(brute_steps_dir / "base_with_blue_mask.png"), base_with_blue)
                                            # Aceeași mască Raster albastră (la (tx,ty)) + poligoanele camerelor și uși/geamuri la aceeași translație
                                            _draw_response_overlay(
                                                base_with_blue,
                                                raster_dir,
                                                brute_steps_dir / "base_with_blue_and_rooms.png",
                                                draw_rooms=True,
                                                draw_doors=True,
                                                draw_walls=False,
                                                offset_xy=(tx, ty),
                                            )
                                        print(f"      📐 Brute force doar translații: (tx, ty) = ({tx}, {ty}), score = {translation_only_config['score']:.2%}")
                except Exception as e:
                    import traceback
                    print(f"      ⚠️ Eroare brute force translation-only: {e}")
                    traceback.print_exc()
            if best_config is None and translation_only_config is None:
                try:
                    orig_walls_path = Path(steps_dir) / "02_ai_walls_closed.png"
                    if orig_walls_path.exists():
                        orig_walls = cv2.imread(str(orig_walls_path), cv2.IMREAD_GRAYSCALE)
                        if orig_walls is not None:
                            oh, ow = orig_walls.shape[:2]
                            api_walls_mask = get_api_walls_mask_for_alignment(raster_dir, oh, ow)
                            if api_walls_mask is not None:
                                best_config = brute_force_alignment(
                                    api_walls_mask,
                                    orig_walls,
                                    raster_dir,
                                    str(steps_dir)
                                )
                except Exception as e:
                    import traceback
                    print(f"      ⚠️ Eroare brute force pentru api_walls_mask: {e}")
                    traceback.print_exc()
            if _rt_bf is not None and raster_timings is not None:
                raster_timings.append(("Raster P2: Brute force", time.time() - _rt_bf))
            if progress_callback is not None and run_phase == 2:
                progress_callback(1)  # brute force done
            
            # Cale translation-only: mască 1px aliniată → garage + interior/exterior, fără construire segmente
            if translation_only_config is not None:
                try:
                    original_path = Path(steps_dir) / "00_original.png"
                    if original_path.exists():
                        original_img = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
                        if original_img is not None:
                            oh, ow = original_img.shape[:2]
                            tx, ty = translation_only_config["position"]
                            aligned_1px = build_aligned_api_walls_1px_original(raster_dir, tx, ty, ow, oh)
                            if aligned_1px is not None:
                                cv2.imwrite(str(raster_dir / "api_walls_mask.png"), aligned_1px)
                                raster_aligned_walls_1px = aligned_1px
                                from .raster_processing import generate_walls_from_room_coordinates
                                _rt_wfc = time.time() if raster_timings is not None else None
                                walls_result = generate_walls_from_room_coordinates(
                                    original_img,
                                    None,
                                    raster_dir,
                                    str(steps_dir),
                                    gemini_api_key,
                                    initial_walls_mask_1px=aligned_1px,
                                    progress_callback=(lambda: progress_callback(2)) if (progress_callback is not None and run_phase == 2) else None,
                                )
                                if _rt_wfc is not None and raster_timings is not None:
                                    raster_timings.append(("Raster P2: Garage + interior/exterior (1px)", time.time() - _rt_wfc))
                                if walls_result:
                                    print(f"      ✅ Garaj + interior/exterior din mască 1px aliniată")
                                    walls_result_from_coords = walls_result
                                if run_phase == 3:
                                    return {}
                except Exception as e:
                    import traceback
                    print(f"      ⚠️ Eroare cale translation-only: {e}")
                    traceback.print_exc()
            
            # Cale clasică: aplicăm transformarea și generăm crop + walls from coords
            if best_config is not None:
                try:
                    # Aplicăm transformarea și generăm overlay-ul pe original
                    original_path = Path(steps_dir) / "00_original.png"
                    response_json_path = raster_dir / "response.json"
                    
                    if original_path.exists() and response_json_path.exists():
                        original_img = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
                        if original_img is not None:
                            # Folosim funcția existentă pentru overlay și crop
                            api_result = {'raster_dir': raster_dir}
                            api_walls_mask = cv2.imread(str(raster_dir / "api_walls_mask.png"), cv2.IMREAD_GRAYSCALE)
                            
                            if api_walls_mask is not None:
                                if raster_timings is not None:
                                    _rt_ov = time.time()
                                # Aplicăm transformarea și generăm overlay-ul
                                apply_alignment_and_generate_overlay(
                                    best_config,
                                    api_result,
                                    original_img,
                                    str(steps_dir)
                                )
                                if run_phase == 3:
                                    # Doar Raster + brute force: stop aici (fără crop, walls_from_coords, garaj etc.)
                                    if raster_timings is not None:
                                        raster_timings.append(("Raster P2: Overlay", time.time() - _rt_ov))
                                    return {}
                                # Generăm crop-ul
                                generate_crop_from_raster(
                                    best_config,
                                    api_walls_mask,
                                    original_img,
                                    api_result
                                )
                                if raster_timings is not None:
                                    raster_timings.append(("Raster P2: Overlay + crop", time.time() - _rt_ov))
                                
                                # ============================================================
                                # GENEREZ PEREȚI DIN COORDONATELE CAMERELOR
                                # (folosind coordonatele din overlay_on_original.png)
                                # ============================================================
                                from .raster_processing import generate_walls_from_room_coordinates
                                
                                # ✅ Wrap în try-except pentru a permite workflow-ul să continue chiar dacă generarea pereților eșuează
                                walls_result = None
                                try:
                                    _rt_wfc = time.time() if raster_timings is not None else None
                                    walls_result = generate_walls_from_room_coordinates(
                                        original_img,
                                        best_config,
                                        raster_dir,
                                        str(steps_dir),
                                        gemini_api_key,
                                        progress_callback=(lambda: progress_callback(2)) if (progress_callback is not None and run_phase == 2) else None,
                                    )
                                    if _rt_wfc is not None and raster_timings is not None:
                                        raster_timings.append(("Raster P2: Walls from room coords", time.time() - _rt_wfc))
                                except Exception as walls_error:
                                    import traceback
                                    print(f"      ⚠️ Eroare la generarea pereților din coordonate: {walls_error}")
                                    traceback.print_exc()
                                    if raster_timings is not None and _rt_wfc is not None:
                                        raster_timings.append(("Raster P2: Walls from room coords", time.time() - _rt_wfc))
                                    # ✅ Verificăm dacă room_scales.json a fost salvat în ciuda erorii
                                    room_scales_path = Path(steps_dir) / "raster_processing" / "walls_from_coords" / "room_scales.json"
                                    if room_scales_path.exists():
                                        print(f"      ✅ room_scales.json a fost salvat în ciuda erorii, workflow-ul poate continua")
                                    else:
                                        print(f"      ⚠️ room_scales.json nu există, workflow-ul va folosi fallback-ul")
                                
                                if walls_result:
                                    print(f"      ✅ Generat pereți din coordonatele camerelor")
                                    # Stocăm walls_result pentru a-l folosi la calculul măsurătorilor
                                    walls_result_from_coords = walls_result
                except Exception as e:
                    import traceback
                    print(f"      ⚠️ Eroare aplicare transformare: {e}")
                    traceback.print_exc()

        if run_phase == 3:
            # Doar Raster + brute force: nu rulăm restul pipeline-ului (scări, garaj, walls_from_coords etc.)
            return {}
    
    # 4b. WORKFLOW REORGANIZAT:
    # 1. Detectare scări + completare pereți scări
    # 2. Umplere găuri mici dintre pereți
    
    print("   🏡 Pas 1: Detectare scări + completare pereți...")
    # Detectează scările folosind detecțiile Roboflow și completează pereții
    # Încercăm să citim detecțiile pentru scări din export_objects/detections.json
    stairs_bboxes = []
    try:
        # Căutăm directorul de output pentru a găsi detecțiile
        detections_json_path = Path(output_dir) / "export_objects" / "detections.json"
        if not detections_json_path.exists():
            # Încercăm și varianta cu count_objects
            detections_json_path = Path(output_dir).parent / "count_objects" / "export_objects" / "detections.json"
        
        if detections_json_path.exists():
            with open(detections_json_path, 'r', encoding='utf-8') as f:
                detections_data = json.load(f)
            
            # Extragem detecțiile pentru scări
            predictions = detections_data.get("predictions", [])
            for pred in predictions:
                if pred.get("class", "").lower() == "stairs" or pred.get("class_name", "").lower() == "stairs":
                    x = pred.get("x", 0)
                    y = pred.get("y", 0)
                    width = pred.get("width", 0)
                    height = pred.get("height", 0)
                    # Convertim din format (cx, cy, w, h) la (x1, y1, x2, y2)
                    x1 = x - width / 2
                    y1 = y - height / 2
                    x2 = x + width / 2
                    y2 = y + height / 2
                    stairs_bboxes.append((x1, y1, x2, y2))
            
            if stairs_bboxes:
                print(f"         📍 Găsit {len(stairs_bboxes)} detecție/ii pentru scări în {detections_json_path.name}")
            else:
                print(f"         ⚠️ Nu s-au găsit detecții pentru scări în {detections_json_path.name}")
        else:
            print(f"         ⚠️ Nu s-a găsit fișierul de detecții: {detections_json_path}")
    except Exception as e:
        print(f"         ⚠️ Eroare la citirea detecțiilor pentru scări: {e}")
    
    # Detectează scările și completează pereții
    ai_walls_stairs = fill_stairs_room(ai_walls_closed, stairs_bboxes, steps_dir=str(steps_dir))
    if ai_walls_stairs is None:
        print(f"      ⚠️ fill_stairs_room a returnat None. Folosesc ai_walls_closed.")
        ai_walls_stairs = ai_walls_closed.copy()
    
    # Pas 1b: Repar pereții exteriori folosind envelope-ul casei (doar completare pereți, nu final)
    # Acest pas completează golurile din pereții exteriori înainte de pașii normali de procesare
    # ai_walls_stairs include deja pereții adăugați pentru terasă/garaj/scări, deci envelope-ul
    # va include automat aceste zone în calcul
    ai_walls_repaired_house = repair_house_walls_with_floodfill(ai_walls_stairs, steps_dir=str(steps_dir))
    if ai_walls_repaired_house is None:
        print("      ⚠️ repair_house_walls_with_floodfill a eșuat. Folosesc ai_walls_stairs.")
        ai_walls_repaired_house = ai_walls_stairs.copy()
    
    # Folosim direct rezultatul de la reparare
    ai_walls_final = ai_walls_repaired_house.copy()
    
    # Verificăm dacă există crop-ul generat de RasterScan
    raster_dir = Path(steps_dir) / "raster" if steps_dir else None
    crop_path = raster_dir / "00_original_crop.png" if raster_dir and raster_dir.exists() else None
    crop_info_path = raster_dir / "crop_info.json" if raster_dir and raster_dir.exists() else None
    
    use_crop = False
    crop_img = None
    crop_info = None
    api_walls_mask_crop = None
    
    # Translation-only: folosim masca 1px aliniată pe planul full (fără crop)
    if raster_aligned_walls_1px is not None:
        use_crop = True
        crop_img = img.copy()
        api_walls_mask_crop = raster_aligned_walls_1px.copy()
        ai_walls_final = api_walls_mask_crop.copy()
        h_orig, w_orig = crop_img.shape[:2]
        print(f"   ✅ Folosesc mască Raster 1px aliniată (translation-only, {w_orig}x{h_orig}px)")
    elif crop_path and crop_path.exists() and crop_info_path and crop_info_path.exists():
        try:
            crop_img = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
            if crop_img is not None:
                with open(crop_info_path, 'r') as f:
                    crop_info = json.load(f)
                
                # Încărcăm și api_walls_mask pentru a-l folosi în loc de ai_walls_final
                api_walls_path = raster_dir / "api_walls_mask.png"
                if api_walls_path.exists():
                    api_walls_mask_crop = cv2.imread(str(api_walls_path), cv2.IMREAD_GRAYSCALE)
                    if api_walls_mask_crop is not None:
                        use_crop = True
                        print(f"   ✅ Folosesc crop-ul generat de RasterScan ({crop_info['width']}x{crop_info['height']}px)")
                        # Folosim api_walls_mask_crop în loc de ai_walls_final pentru calcule
                        ai_walls_final = api_walls_mask_crop.copy()
                        h_orig, w_orig = crop_img.shape[:2]
                        # Actualizăm și img pentru a folosi crop-ul
                        img = crop_img.copy()
                    else:
                        print(f"   ⚠️ Nu am putut încărca api_walls_mask.png, folosesc workflow-ul normal")
                else:
                    print(f"   ⚠️ api_walls_mask.png nu există, folosesc workflow-ul normal")
            else:
                print(f"   ⚠️ Nu am putut încărca crop-ul, folosesc workflow-ul normal")
        except Exception as e:
            print(f"   ⚠️ Eroare la încărcarea crop-ului: {e}, folosesc workflow-ul normal")
    
    if use_crop:
        # WORKFLOW CU RASTERSCAN CROP
        print(f"   🔄 Workflow cu RasterScan crop...")
        
        # 1. Generez imagine cu masca RasterScan peste crop
        if crop_img is not None and api_walls_mask_crop is not None:
            # Salvăm în folderul raster (unde sunt și celelalte imagini RasterScan)
            raster_output_path = raster_dir / "walls_overlay_on_crop.png"
            generate_raster_walls_overlay(
                crop_img,
                api_walls_mask_crop,
                raster_output_path
            )
            print(f"      ✅ Generat overlay: {raster_output_path.name}")
        
        # ⚠️ NU mai generăm rooms_overlay_on_crop.png aici - rooms.png nu mai este generat la pasul 1
        # rooms.png va fi generat DUPĂ validarea pereților în raster_processing
        
        # 2. Detectare interior/exterior folosind masca RasterScan
        if raster_timings is not None:
            _rt_ie = time.time()
        indoor_mask, outdoor_mask = detect_interior_exterior_from_raster(
            api_walls_mask_crop,
            steps_dir=str(steps_dir)
        )
        if raster_timings is not None:
            raster_timings.append(("Raster P2: Interior/exterior", time.time() - _rt_ie))
        if progress_callback is not None and run_phase == 2:
            progress_callback(3)  # phase2 end
        
        # 3. Citim scala din room_scales.json generat de RasterScan (NU calculăm din nou!)
        if crop_img is not None:
            # ✅ Scala a fost deja calculată de RasterScan în generate_walls_from_room_coordinates
            # Citim direct din room_scales.json
            room_scales_path = Path(steps_dir) / "raster_processing" / "walls_from_coords" / "room_scales.json"
            
            if room_scales_path.exists():
                try:
                    with open(room_scales_path, 'r', encoding='utf-8') as f:
                        room_scales_data = json.load(f)
                    
                    # Încercăm să citim m_px direct
                    m_px = room_scales_data.get('m_px')
                    
                    # Dacă nu există m_px, calculăm din total_area_m2 și total_area_px
                    if m_px is None or m_px <= 0:
                        total_area_m2 = room_scales_data.get('total_area_m2', 0)
                        total_area_px = room_scales_data.get('total_area_px', 0)
                        if total_area_px > 0 and total_area_m2 > 0:
                            m_px = np.sqrt(total_area_m2 / total_area_px)
                            print(f"   ✅ Calculat m_px din total_area: {m_px:.9f} m/px")
                        else:
                            raise RuntimeError("Nu am putut determina scala din room_scales.json (lipsesc total_area_m2 sau total_area_px)")
                    else:
                        print(f"   ✅ Scala din RasterScan (room_scales.json): {m_px:.9f} m/px")
                    
                    # Verificăm că avem m_px valid
                    if m_px is None or m_px <= 0:
                        raise RuntimeError("m_px invalid din room_scales.json")
                    
                    # Numărăm camerele folosite pentru calcularea scalei
                    room_scales = room_scales_data.get('room_scales', {})
                    rooms_used = len(room_scales) if isinstance(room_scales, dict) else 0
                    
                    # ✅ Construim scale_result pentru return (compatibil cu workflow-ul normal)
                    scale_result = {
                        "meters_per_pixel": float(m_px),
                        "method": "raster_scan",
                        "confidence": "high" if rooms_used >= 3 else "medium",
                        "source": "room_scales.json",
                        "rooms_used": rooms_used,  # ✅ Necesar pentru scale/jobs.py
                        "optimization_info": {
                            "method": "raster_scan_direct",
                            "rooms_count": rooms_used
                        },
                        "per_room": []  # Listă goală pentru compatibilitate
                    }
                    
                except Exception as e:
                    import traceback
                    print(f"   ⚠️ Eroare la citirea room_scales.json: {e}")
                    traceback.print_exc()
                    # ✅ Încercăm să creăm un fișier minimal pentru a permite workflow-ul să continue
                    try:
                        output_dir = Path(steps_dir) / "raster_processing" / "walls_from_coords"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        minimal_data = {
                            'rooms': {},
                            'total_area_m2': 0.0,
                            'total_area_px': 0,
                            'm_px': None,
                            'weighted_average_m_px': None,
                            'room_scales': {},
                            'error': f"Eroare la citire: {str(e)}"
                        }
                        with open(room_scales_path, 'w', encoding='utf-8') as f:
                            json.dump(minimal_data, f, indent=2, ensure_ascii=False)
                        print(f"   ⚠️ Creat room_scales.json minimal pentru compatibilitate")
                    except Exception as e2:
                        print(f"   ❌ Nu am putut crea room_scales.json minimal: {e2}")
                    raise RuntimeError(f"Nu am putut citi scala din RasterScan: {e}")
            else:
                # ✅ În loc să aruncăm eroare imediat, încercăm să creăm un fișier minimal
                print(f"   ⚠️ room_scales.json nu există la {room_scales_path}")
                try:
                    output_dir = Path(steps_dir) / "raster_processing" / "walls_from_coords"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    minimal_data = {
                        'rooms': {},
                        'total_area_m2': 0.0,
                        'total_area_px': 0,
                        'm_px': None,
                        'weighted_average_m_px': None,
                        'room_scales': {},
                        'error': 'Fișierul nu a fost generat de RasterScan'
                    }
                    with open(room_scales_path, 'w', encoding='utf-8') as f:
                        json.dump(minimal_data, f, indent=2, ensure_ascii=False)
                    print(f"   ⚠️ Creat room_scales.json minimal pentru compatibilitate")
                    print(f"   ⚠️ Workflow-ul va continua, dar scale-ul va trebui calculat altfel")
                except Exception as e2:
                    print(f"   ❌ Nu am putut crea room_scales.json minimal: {e2}")
                raise RuntimeError(f"room_scales.json nu există la {room_scales_path}. RasterScan trebuie să genereze acest fișier înainte.")
        else:
            raise RuntimeError("crop_img este None!")
        
        # 4. Generez pereți interiori și exteriori cu 1px
        walls_int_1px, walls_ext_1px = generate_walls_interior_exterior(
            api_walls_mask_crop,
            indoor_mask,
            outdoor_mask,
            steps_dir=str(steps_dir)
        )
        
        # 5. Generez structură pereți interiori
        walls_int_structure = generate_interior_structure_walls(
            api_walls_mask_crop,
            walls_int_1px,
            steps_dir=str(steps_dir)
        )
        
        # Actualizăm ai_walls_final pentru măsurători
        ai_walls_final = api_walls_mask_crop.copy()
        
        # Variabile pentru măsurători (folosim pereții cu 1px)
        walls_int_1px_for_measurements = walls_int_1px
        walls_ext_1px_for_measurements = walls_ext_1px
        walls_int_structure_for_measurements = walls_int_structure
        
    else:
        # WORKFLOW NORMAL (FĂRĂ CROP RASTERSCAN)
        print(f"   ℹ️ Folosesc workflow-ul normal (fără crop RasterScan)")
        
        # run_phase==2 nu setează h_orig/w_orig; folosim dimensiunile din ai_walls_final
        h_orig, w_orig = ai_walls_final.shape[:2]
        
        # Kernel repair pentru restul procesării
        min_dim = min(h_orig, w_orig) 
        rep_k = max(3, int(min_dim * 0.005))
        if rep_k % 2 == 0: rep_k += 1
        kernel_repair = cv2.getStructuringElement(cv2.MORPH_RECT, (rep_k, rep_k))

        # 7. ZONE
        print("   🌊 Analizez zonele...")
        
        walls_thick = cv2.dilate(ai_walls_final, kernel_repair, iterations=3)
        
        h_pad, w_pad = h_orig + 2, w_orig + 2
        pad_walls = np.zeros((h_pad, w_pad), dtype=np.uint8)
        pad_walls[1:h_orig+1, 1:w_orig+1] = walls_thick
        
        inv_pad_walls = cv2.bitwise_not(pad_walls)
        flood_mask = np.zeros((h_pad+2, w_pad+2), dtype=np.uint8)
        cv2.floodFill(inv_pad_walls, flood_mask, (0, 0), 128)
        
        outdoor_mask = (inv_pad_walls[1:h_orig+1, 1:w_orig+1] == 128).astype(np.uint8) * 255
        
        kernel_grow = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        free_space = cv2.bitwise_not(ai_walls_final)
        
        for _ in range(30):
            outdoor_mask = cv2.bitwise_and(cv2.dilate(outdoor_mask, kernel_grow), free_space)
        
        save_step("03_outdoor_mask", outdoor_mask, str(steps_dir))
        
        total_space = np.ones_like(outdoor_mask) * 255
        occupied = cv2.bitwise_or(outdoor_mask, ai_walls_final)
        indoor_mask = cv2.subtract(total_space, occupied)
        
        vis_indoor = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
        vis_indoor[indoor_mask > 0] = [0, 255, 255]
        save_step("debug_zone_interior", vis_indoor, str(steps_dir))

        # 8. SCALE DETECTION
        print("   🔍 Determin scala...")
        image_for_scale = str(image_path)
        scale_result = detect_scale_from_room_labels(
            image_for_scale,
            indoor_mask,
            ai_walls_final,
            str(steps_dir),
            api_key=gemini_api_key
        )
        
        if not scale_result:
            raise RuntimeError("Nu am putut determina scala!")
        
        m_px = scale_result["meters_per_pixel"]
        
        # Variabile pentru măsurători (workflow normal)
        walls_int_1px_for_measurements = None
        walls_ext_1px_for_measurements = None
        walls_int_structure_for_measurements = None

    # 9. MĂSURĂTORI
    print("   📏 Calculez măsurători...")
    
    # Inițializăm variabilele pentru măsurători
    px_len_ext = 0
    px_len_int = 0
    px_len_skeleton_ext = 0
    px_len_skeleton_structure_int = 0
    
    # Verificăm dacă avem rezultate din raster_processing (pereți din coordonatele camerelor)
    use_raster_measurements = False
    if walls_result_from_coords and walls_result_from_coords.get('m_px'):
        use_raster_measurements = True
        print(f"      ✅ Folosesc măsurătorile din raster_processing (pereți din coordonatele camerelor)")
        
        # Încărcăm imaginile generate
        raster_processing_dir = Path(steps_dir) / "raster_processing" / "walls_from_coords"
        
        # Lungimi structură (din imagini 11 și 12)
        interior_structure_img_path = raster_processing_dir / "11_interior_structure.png"
        exterior_structure_img_path = raster_processing_dir / "12_exterior_structure.png"
        
        # Suprafete finisaje (din imagini 07 și 08)
        walls_interior_img_path = raster_processing_dir / "07_walls_interior.png"
        walls_exterior_img_path = raster_processing_dir / "08_walls_exterior.png"
        
        # Metri per pixel din room_scales.json
        m_px_raster = walls_result_from_coords.get('m_px')
        
        if m_px_raster and interior_structure_img_path.exists() and exterior_structure_img_path.exists():
            # Încărcăm imaginile
            interior_structure_img = cv2.imread(str(interior_structure_img_path), cv2.IMREAD_GRAYSCALE)
            exterior_structure_img = cv2.imread(str(exterior_structure_img_path), cv2.IMREAD_GRAYSCALE)
            
            if interior_structure_img is not None and exterior_structure_img is not None:
                # Calculăm lungimile structurii (număr de pixeli * m_px)
                px_len_skeleton_structure_int = int(np.count_nonzero(interior_structure_img > 0))
                px_len_skeleton_ext = int(np.count_nonzero(exterior_structure_img > 0))
                
                # Folosim m_px din room_scales.json
                m_px = m_px_raster
                
                # Calculăm lungimile structurii în metri
                walls_skeleton_structure_int_m = px_len_skeleton_structure_int * m_px
                walls_skeleton_ext_m = px_len_skeleton_ext * m_px
                
                print(f"         📐 Structură interior: {px_len_skeleton_structure_int} px = {walls_skeleton_structure_int_m:.2f} m")
                print(f"         📐 Structură exterior: {px_len_skeleton_ext} px = {walls_skeleton_ext_m:.2f} m")
        
        # Suprafete finisaje (din imagini 07 și 08)
        if walls_interior_img_path.exists() and walls_exterior_img_path.exists():
            walls_interior_img = cv2.imread(str(walls_interior_img_path), cv2.IMREAD_GRAYSCALE)
            walls_exterior_img = cv2.imread(str(walls_exterior_img_path), cv2.IMREAD_GRAYSCALE)
            
            if walls_interior_img is not None and walls_exterior_img is not None:
                # Calculăm lungimile pentru finisaje (număr de pixeli * m_px)
                px_len_int = int(np.count_nonzero(walls_interior_img > 0))
                px_len_ext = int(np.count_nonzero(walls_exterior_img > 0))
                
                # Calculăm lungimile în metri
                walls_int_m = px_len_int * m_px
                walls_ext_m = px_len_ext * m_px
                
                print(f"         🎨 Finisaje interior: {px_len_int} px = {walls_int_m:.2f} m")
                print(f"         🎨 Finisaje exterior: {px_len_ext} px = {walls_ext_m:.2f} m")
        
        # Suprafete camere din room_scales.json
        room_scales = walls_result_from_coords.get('room_scales', {})
        if room_scales:
            total_area_m2_raster = walls_result_from_coords.get('total_area_m2', 0.0)
            total_area_px_raster = walls_result_from_coords.get('total_area_px', 0)
            
            print(f"         🏠 Suprafete camere: {total_area_m2_raster:.2f} m² ({len(room_scales)} camere)")
    
    if not use_raster_measurements:
        if use_crop:
            # Folosim pereții cu 1px generați anterior
            px_len_ext = int(np.count_nonzero(walls_ext_1px_for_measurements))
            px_len_int = int(np.count_nonzero(walls_int_1px_for_measurements))
            px_len_skeleton_ext = px_len_ext  # Pentru consistență
            px_len_skeleton_structure_int = int(np.count_nonzero(walls_int_structure_for_measurements))
        else:
            # Workflow normal - calculăm outline
            outline = get_strict_1px_outline(ai_walls_final)
            touch_zone = cv2.dilate(outdoor_mask, kernel_grow, iterations=2)
            
            outline_ext_mask = cv2.bitwise_and(outline, touch_zone)
            outline_int_mask = cv2.subtract(outline, outline_ext_mask)
            
            # Calculăm lungimea pereților exteriori (din outline)
            px_len_skeleton_ext = int(np.count_nonzero(outline_ext_mask))
            
            # Lungimea structurii pereților interiori (folosim outline interior)
            px_len_skeleton_structure_int = int(np.count_nonzero(outline_int_mask))
            
            # Lungimi din outline (pentru finisaje)
            px_len_ext = int(np.count_nonzero(outline_ext_mask))
            px_len_int = int(np.count_nonzero(outline_int_mask))
    
    # Arii
    px_area_indoor = int(np.count_nonzero(indoor_mask))
    px_area_total = int(np.count_nonzero(cv2.bitwise_not(outdoor_mask)))

    # Conversii în metri
    walls_ext_m = px_len_ext * m_px  # Pentru finisaje
    walls_int_m = px_len_int * m_px  # Pentru finisaje
    
    # Lungimi din skeleton (pentru structură)
    walls_skeleton_ext_m = px_len_skeleton_ext * m_px
    walls_skeleton_structure_int_m = px_len_skeleton_structure_int * m_px  # Structură pereți interiori (din outline)
    
    area_indoor_m2 = px_area_indoor * (m_px ** 2)
    area_total_m2 = px_area_total * (m_px ** 2)

    measurements = {
        "pixels": {
            "walls_len_ext": int(px_len_ext),
            "walls_len_int": int(px_len_int),
            "walls_skeleton_ext": int(px_len_skeleton_ext),
            "walls_skeleton_structure_int": int(px_len_skeleton_structure_int),
            "indoor_area": int(px_area_indoor),
            "total_area": int(px_area_total)
        },
        "metrics": {
            "scale_m_per_px": float(m_px),
            "walls_ext_m": float(round(walls_ext_m, 2)),  # Pentru finisaje
            "walls_int_m": float(round(walls_int_m, 2)),  # Pentru finisaje
            "walls_skeleton_ext_m": float(round(walls_skeleton_ext_m, 2)),
            "walls_skeleton_structure_int_m": float(round(walls_skeleton_structure_int_m, 2)),  # Pentru structură pereți interiori
            "area_indoor_m2": float(round(area_indoor_m2, 2)),
            "area_total_m2": float(round(area_total_m2, 2))
        }
    }

    print(f"   ✅ Scară: {m_px:.9f} m/px")
    print(f"   🏠 Arie Indoor: {area_indoor_m2:.2f} m²")
    print(f"   📏 Lungimi pereți:")
    print(f"      - Exterior (outline): {walls_ext_m:.2f} m (pentru finisaje)")
    print(f"      - Interior (outline): {walls_int_m:.2f} m (pentru finisaje)")
    print(f"      - Skeleton exterior (din outline): {walls_skeleton_ext_m:.2f} m")
    print(f"      - Structură interior (din outline): {walls_skeleton_structure_int_m:.2f} m")
    
    # ✅ Salvează walls_measurements într-un fișier separat pentru pricing (FĂRĂ dependență de CubiCasa)
    if steps_dir:
        raster_processing_dir = Path(steps_dir) / "raster_processing" / "walls_from_coords"
        raster_processing_dir.mkdir(parents=True, exist_ok=True)
        walls_measurements_file = raster_processing_dir / "walls_measurements.json"
        walls_measurements_data = {
            "estimations": {
                "average_result": {
                    "interior_meters": float(round(walls_int_m, 2)),
                    "exterior_meters": float(round(walls_ext_m, 2)),
                    "interior_meters_structure": float(round(walls_skeleton_structure_int_m, 2))
                }
            }
        }
        with open(walls_measurements_file, "w", encoding="utf-8") as f:
            json.dump(walls_measurements_data, f, indent=2, ensure_ascii=False)
        print(f"   💾 Salvat: walls_measurements.json (pentru pricing)")
        # wall_lengths.json: lungimi explicite din pixeli (07, 08, 11) × m_px pentru documentație și PDF
        wall_lengths_file = raster_processing_dir / "wall_lengths.json"
        wall_lengths_data = {
            "scale_m_per_px": float(m_px),
            "interior_finish_length_px": int(px_len_int),
            "interior_finish_length_m": float(round(walls_int_m, 4)),
            "exterior_length_px": int(px_len_ext),
            "exterior_length_m": float(round(walls_ext_m, 4)),
            "interior_structure_length_px": int(px_len_skeleton_structure_int),
            "interior_structure_length_m": float(round(walls_skeleton_structure_int_m, 4)),
            "exterior_structure_length_px": int(px_len_skeleton_ext),
            "exterior_structure_length_m": float(round(walls_skeleton_ext_m, 4)),
        }
        with open(wall_lengths_file, "w", encoding="utf-8") as f:
            json.dump(wall_lengths_data, f, indent=2, ensure_ascii=False)
        print(f"   💾 Salvat: wall_lengths.json (lungimi pereți pentru PDF)")
    
    # 10. FILTRARE ZGOMOT CU MASCĂ ZONĂ INTERIOARĂ (înainte de generarea 3D)
    print("   🎨 Filtrez zgomotul de fundal cu masca zonei interioare...")
    
    # Încărcăm masca din debug_zone_interior.png
    debug_mask_path = Path(steps_dir) / "debug_zone_interior.png"
    interior_zone_mask = None
    
    if debug_mask_path.exists():
        debug_mask_img = cv2.imread(str(debug_mask_path), cv2.IMREAD_COLOR)
        if debug_mask_img is not None:
            # debug_zone_interior are culoarea galbenă (cyan în BGR: [0, 255, 255]) pentru zona interioară
            b, g, r = cv2.split(debug_mask_img)
            # Extragem masca: zona interioară este unde componenta verde este mare
            interior_zone_mask = (g > 128).astype(np.uint8) * 255
            
            # Redimensionăm dacă e necesar
            if interior_zone_mask.shape[:2] != ai_walls_final.shape[:2]:
                interior_zone_mask = cv2.resize(interior_zone_mask, (ai_walls_final.shape[1], ai_walls_final.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Calculăm grosimea medie a pereților pentru marjă de eroare
            dist_from_edge = cv2.distanceTransform(ai_walls_final, cv2.DIST_L2, 5)
            wall_thicknesses = dist_from_edge[ai_walls_final > 0] * 2
            if len(wall_thicknesses) > 0:
                avg_thickness = np.median(wall_thicknesses)
                margin_size = max(3, int(round(avg_thickness)))
                
                print(f"      📏 Grosime medie pereți: {avg_thickness:.2f}px, marjă de eroare: {margin_size}px")
                
                # Dilatăm masca zonei interioare cu marja de eroare (grosimea pereților)
                kernel_margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin_size * 2 + 1, margin_size * 2 + 1))
                interior_zone_mask_dilated = cv2.dilate(interior_zone_mask, kernel_margin, iterations=1)
                
                # Aplicăm masca pe pereți: păstrăm doar pereții din zona interioară (cu marjă)
                ai_walls_final_filtered = cv2.bitwise_and(ai_walls_final, interior_zone_mask_dilated)
                
                # Folosim pereții filtrați pentru generarea 3D
                ai_walls_for_3d = ai_walls_final_filtered
                
                removed_pixels = np.count_nonzero(ai_walls_final) - np.count_nonzero(ai_walls_final_filtered)
                print(f"      ✅ Eliminat {removed_pixels:,} pixeli de zgomot din afara zonei interioare")
            else:
                ai_walls_for_3d = ai_walls_final
                print(f"      ⚠️ Nu s-a putut calcula grosimea, folosesc pereții fără filtrare")
        else:
            ai_walls_for_3d = ai_walls_final
            print(f"      ⚠️ Nu s-a putut încărca masca, folosesc pereții fără filtrare")
    else:
        ai_walls_for_3d = ai_walls_final
        print(f"      ⚠️ Mască lipsă, folosesc pereții fără filtrare")
    
    # 10. GENERARE 3D (cu pereții filtrați - zgomotul a fost eliminat înainte)
    export_walls_to_obj(
        ai_walls_for_3d, 
        output_dir / "walls_3d.obj", 
        m_px, 
        image_output_path=output_dir / "walls_3d_view.png"
    )

    # 11. VIZUALIZĂRI
    overlay = img.copy()
    # Verificăm dacă outline_ext_mask și outline_int_mask sunt definite
    try:
        if 'outline_ext_mask' in locals() and outline_ext_mask is not None:
            overlay[outline_ext_mask > 0] = [255, 0, 0]
        if 'outline_int_mask' in locals() and outline_int_mask is not None:
            overlay[outline_int_mask > 0] = [0, 255, 0]
        cv2.imwrite(str(output_dir / "visualization_overlay.png"), overlay)
    except (NameError, UnboundLocalError):
        # Dacă nu sunt definite (workflow cu RasterScan crop), folosim pereții direct dacă există
        try:
            if 'walls_ext_1px_for_measurements' in locals() and walls_ext_1px_for_measurements is not None:
                overlay[walls_ext_1px_for_measurements > 0] = [255, 0, 0]
            if 'walls_int_1px_for_measurements' in locals() and walls_int_1px_for_measurements is not None:
                overlay[walls_int_1px_for_measurements > 0] = [0, 255, 0]
            cv2.imwrite(str(output_dir / "visualization_overlay.png"), overlay)
        except (NameError, UnboundLocalError):
            # Dacă niciunul nu este disponibil, salvăm overlay-ul fără modificări
            cv2.imwrite(str(output_dir / "visualization_overlay.png"), overlay)
    
    return {
        "scale_result": scale_result,
        "measurements": measurements,
        "masks": {
            "visualization": str(output_dir / "visualization_overlay.png"),
            "visualization_3d": str(output_dir / "walls_3d_view.png"),
            "interior_walls_skeleton": str(steps_dir / "04_interior_walls_skeleton.png")
        }
    }