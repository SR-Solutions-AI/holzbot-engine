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
import re
import requests
import base64
from pathlib import Path
from PIL import Image
from io import BytesIO
from skimage.morphology import skeletonize

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
        """
        # Eșantionăm puncte de-a lungul liniei
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        dist = np.sqrt(dx*dx + dy*dy)
        num_samples = max(10, int(dist / 5))
        perpendicular_walls = 0
        
        for t in np.linspace(0.1, 0.9, num_samples):  # Evităm capetele
            px = int(start_pt[0] + t * dx)
            py = int(start_pt[1] + t * dy)
            
            if 0 <= px < w and 0 <= py < h:
                # Verificăm pe o bandă perpendiculară pe linie
                check_radius = 5
                if abs(dx) > abs(dy):  # Linie orizontală
                    for offset in range(-check_radius, check_radius + 1):
                        check_y = py + offset
                        if 0 <= check_y < h and abs(offset) > 2:
                            if walls_mask_image[check_y, px] == 255:
                                perpendicular_walls += 1
                                break
                else:  # Linie verticală
                    for offset in range(-check_radius, check_radius + 1):
                        check_x = px + offset
                        if 0 <= check_x < w and abs(offset) > 2:
                            if walls_mask_image[py, check_x] == 255:
                                perpendicular_walls += 1
                                break
        
        # Dacă mai mult de 20% din eșantioane au pereți perpendiculari, respingem
        return perpendicular_walls <= num_samples * 0.2
    
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

def preprocess_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocesează imaginea pentru a îmbunătăți detecția OCR.
    
    Aplică:
    - Conversie la grayscale dacă este color
    - Contrast enhancement (CLAHE)
    - Sharpening
    - Thresholding adaptiv pentru text clar
    
    Args:
        image: Imaginea de preprocesat (BGR sau grayscale)
    
    Returns:
        Imaginea preprocesată (grayscale)
    """
    # Convertim la grayscale dacă este color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 1. Contrast enhancement cu CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 2. Sharpening pentru a face textul mai clar
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
    
    # 3. Denoising (reducere zgomot)
    denoised = cv2.fastNlMeansDenoising(sharpened, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # 4. Thresholding adaptiv pentru text clar (opțional, dar poate ajuta)
    # Nu aplicăm thresholding direct, ci păstrăm imaginea grayscale pentru OCR
    # OCR-ul funcționează mai bine pe imagini grayscale cu contrast bun
    
    return denoised

def _reconstruct_word_from_chars(chars_list, search_terms, max_horizontal_distance=50, max_vertical_distance=10):
    """
    Reconstituie cuvântul complet din caracterele detectate individual.
    
    Args:
        chars_list: Lista de caractere detectate: [(x, y, width, height, text, conf), ...]
        search_terms: Lista de termeni de căutat
        max_horizontal_distance: Distanța maximă orizontală între caractere pentru a fi considerate parte din același cuvânt
        max_vertical_distance: Distanța maximă verticală între caractere pentru a fi considerate pe aceeași linie
    
    Returns:
        Lista de cuvinte reconstituite: [(x, y, width, height, text, conf), ...]
    """
    if not chars_list:
        return []
    
    # Sortăm caracterele de la stânga la dreapta, apoi de sus în jos
    sorted_chars = sorted(chars_list, key=lambda c: (c[1], c[0]))  # Sort by y, then x
    
    reconstructed_words = []
    
    # Pentru fiecare termen căutat, încercăm să reconstituim cuvântul
    for search_term in search_terms:
        search_term_lower = search_term.lower()
        term_length = len(search_term_lower)
        
        # Grupăm caracterele care sunt pe aceeași linie (aproximativ)
        lines = []
        current_line = [sorted_chars[0]]
        
        for char in sorted_chars[1:]:
            prev_char = current_line[-1]
            # Verificăm dacă caracterul este pe aceeași linie (similar y)
            y_diff = abs(char[1] - prev_char[1])
            if y_diff <= max_vertical_distance:
                current_line.append(char)
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [char]
        
        if current_line:
            lines.append(current_line)
        
        # Pentru fiecare linie, încercăm să găsim secvența care formează cuvântul
        for line in lines:
            # Sortăm caracterele din linie de la stânga la dreapta
            line_sorted = sorted(line, key=lambda c: c[0])
            
            # Căutăm secvențe de caractere care pot forma cuvântul
            for start_idx in range(len(line_sorted)):
                # Verificăm dacă primul caracter face parte din cuvântul căutat
                first_char_text = line_sorted[start_idx][4].lower()  # text
                
                # Verificăm dacă primul caracter se potrivește cu prima literă din termen
                # OCR poate detecta un singur caracter sau un grup de caractere
                if search_term_lower[0] not in first_char_text:
                    continue
                
                # Construim cuvântul de la acest caracter
                used_indices = [start_idx]
                x_start = line_sorted[start_idx][0]
                y_start = line_sorted[start_idx][1]
                x_end = x_start + line_sorted[start_idx][2]
                y_end = y_start + line_sorted[start_idx][3]
                conf_sum = line_sorted[start_idx][5]
                conf_count = 1
                
                # Căutăm caracterele următoare
                # Construim cuvântul caracter cu caracter, verificând dacă fiecare caracter detectat
                # se potrivește cu următoarea literă din termenul căutat
                reconstructed_chars = list(first_char_text)  # Lista de caractere reconstituite
                
                for term_idx in range(1, term_length):
                    target_char = search_term_lower[term_idx]
                    
                    # Verificăm dacă avem deja caracterul în reconstructed
                    if term_idx < len(reconstructed_chars):
                        # Caracterul a fost deja adăugat (poate fi parte dintr-un grup de caractere detectat)
                        continue
                    
                    # Căutăm următorul caracter care se potrivește
                    best_match = None
                    best_distance = float('inf')
                    
                    for char_idx, char in enumerate(line_sorted):
                        if char_idx in used_indices:
                            continue
                        
                        char_text = char[4].lower()
                        char_x = char[0]
                        
                        # Verificăm dacă caracterul conține litera căutată
                        # Poate fi un singur caracter sau un grup de caractere
                        if target_char in char_text:
                            # Verificăm distanța orizontală față de ultimul caracter folosit
                            last_char = line_sorted[used_indices[-1]]
                            last_x_end = last_char[0] + last_char[2]
                            distance = char_x - last_x_end
                            
                            # Caracterul trebuie să fie la dreapta ultimului și la o distanță rezonabilă
                            if distance >= 0 and distance <= max_horizontal_distance:
                                if distance < best_distance:
                                    best_match = char_idx
                                    best_distance = distance
                    
                    if best_match is not None:
                        char = line_sorted[best_match]
                        char_text_lower = char[4].lower()
                        
                        # Adăugăm doar caracterul care se potrivește (nu întregul grup)
                        # Dacă grupul conține mai multe caractere, adăugăm doar primul care se potrivește
                        char_added = False
                        for c in char_text_lower:
                            if c == target_char and len(reconstructed_chars) < term_length:
                                reconstructed_chars.append(c)
                                char_added = True
                                break
                        
                        # Dacă nu am găsit caracterul exact, încercăm să adăugăm primul caracter din grup
                        # (poate OCR a detectat greșit sau caracterul este similar)
                        if not char_added and len(reconstructed_chars) < term_length:
                            # Adăugăm primul caracter din grup dacă este similar cu cel căutat
                            if len(char_text_lower) > 0:
                                reconstructed_chars.append(char_text_lower[0])
                                char_added = True
                        
                        if char_added:
                            used_indices.append(best_match)
                            x_end = char[0] + char[2]
                            y_end = max(y_end, char[1] + char[3])
                            conf_sum += char[5]
                            conf_count += 1
                        else:
                            # Nu am putut adăuga caracterul, încercăm să continuăm
                            break
                    else:
                        # Nu am găsit următorul caracter, încercăm să continuăm cu ce avem
                        break
                
                # Reconstruim string-ul din caracterele colectate
                reconstructed = ''.join(reconstructed_chars[:term_length])
                
                # Verificăm dacă cuvântul reconstituit se potrivește cu termenul căutat
                if reconstructed == search_term_lower:
                    # Calculăm bounding box-ul pentru întregul cuvânt
                    width = x_end - x_start
                    height = y_end - y_start
                    avg_conf = conf_sum / conf_count if conf_count > 0 else 0
                    
                    if avg_conf > 60:
                        reconstructed_words.append((x_start, y_start, width, height, search_term, avg_conf))
                        print(f"         🔤 Reconstituit cuvântul '{search_term}' din {conf_count} caractere (confidență medie: {avg_conf:.1f}%)")
                        break  # Nu mai căutăm în această linie dacă am găsit deja cuvântul
    
    return reconstructed_words


def run_ocr_on_zones(image: np.ndarray, search_terms: list, steps_dir: str = None, 
                     grid_rows: int = 3, grid_cols: int = 3, zoom_factor: float = 2.0) -> list:
    """
    Rulează OCR pe zone diferite ale imaginii cu zoom pentru a detecta mai bine textul mic.
    
    Împarte imaginea în grid și rulează OCR pe fiecare zonă, eventual cu zoom.
    Dacă detectează caractere individuale, încearcă să reconstituie cuvântul complet.
    
    Args:
        image: Imaginea de analizat (grayscale sau BGR)
        search_terms: Lista de termeni de căutat
        steps_dir: Director pentru debug (opțional)
        grid_rows: Număr de rânduri în grid
        grid_cols: Număr de coloane în grid
        zoom_factor: Factor de zoom pentru fiecare zonă (1.0 = fără zoom)
    
    Returns:
        Lista de text_boxes detectate: [(x, y, width, height, text, conf), ...]
    """
    if not TESSERACT_AVAILABLE:
        return []
    
    h, w = image.shape[:2]
    text_boxes = []
    all_chars = []  # Colectăm toate caracterele detectate pentru reconstituire
    all_detections = []  # Colectăm TOATE detecțiile OCR pentru debug (nu doar cele care se potrivesc)
    
    # Preprocesăm întreaga imagine
    processed_full = preprocess_image_for_ocr(image)
    
    # 1. OCR pe întreaga imagine preprocesată
    print(f"         📝 OCR pe întreaga imagine (preprocesată)...")
    ocr_data_full = pytesseract.image_to_data(processed_full, output_type=pytesseract.Output.DICT, lang='deu+eng')
    
    for i, text in enumerate(ocr_data_full['text']):
        if text.strip():
            text_clean = text.strip()
            text_lower = text_clean.lower()
            
            x = ocr_data_full['left'][i]
            y = ocr_data_full['top'][i]
            width = ocr_data_full['width'][i]
            height = ocr_data_full['height'][i]
            conf = ocr_data_full['conf'][i]
            
            # Colectăm TOATE detecțiile pentru debug
            all_detections.append((x, y, width, height, text_clean, conf))
            
            found_term = None
            for term in search_terms:
                term_lower = term.lower()
                # Verificăm doar match exact (case-insensitive)
                # Nu acceptăm fragmente - doar cuvântul complet "terasa" sau variantele sale exacte
                if term_lower == text_lower:
                    found_term = term
                    break
            
            if found_term:
                if conf > 60:
                    text_boxes.append((x, y, width, height, text_clean, conf))
                    print(f"         ✅ Detectat (full): '{text_clean}' la ({x}, {y}) cu confidență {conf:.1f}%")
            else:
                # Colectăm și caracterele individuale pentru reconstituire
                # Verificăm dacă caracterul face parte din unul dintre termenii căutați
                for term in search_terms:
                    term_lower = term.lower()
                    if any(char in text_lower for char in term_lower):
                        all_chars.append((x, y, width, height, text_clean, conf))
                        break
    
    # 2. Dacă nu am găsit nimic sau am găsit doar rezultate cu confidence scăzut, încercăm pe zone
    if not text_boxes or (text_boxes and max(box[5] for box in text_boxes) < 60):
        print(f"         🔍 Împărțim imaginea în {grid_rows}x{grid_cols} zone pentru OCR detaliat...")
        
        overlap = 50  # pixeli de overlap între zone
        
        # Calculăm dimensiunile de bază ale unei zone (folosind diviziune reală pentru acoperire completă)
        zone_height_base = h / grid_rows
        zone_width_base = w / grid_cols
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                # Calculăm coordonatele zonei astfel încât să acopere complet imaginea
                # Prima zonă începe de la 0
                if row == 0:
                    y_start = 0
                else:
                    # Zonele intermediare au overlap cu zona anterioară
                    y_start = max(0, int(row * zone_height_base) - overlap)
                
                # Ultima zonă merge până la marginea imaginii (h)
                if row == grid_rows - 1:
                    y_end = h
                else:
                    # Zonele intermediare se termină cu overlap pentru următoarea zonă
                    y_end = min(h, int((row + 1) * zone_height_base) + overlap)
                
                # Același lucru pentru coloane
                if col == 0:
                    x_start = 0
                else:
                    x_start = max(0, int(col * zone_width_base) - overlap)
                
                if col == grid_cols - 1:
                    x_end = w
                else:
                    x_end = min(w, int((col + 1) * zone_width_base) + overlap)
                
                # Verificăm că zona este validă
                if x_start >= x_end or y_start >= y_end:
                    print(f"         ⚠️ Zona ({row+1},{col+1}) invalidă: x=[{x_start},{x_end}), y=[{y_start},{y_end})")
                    continue
                
                # Extragem zona
                zone = image[y_start:y_end, x_start:x_end]
                
                if zone.size == 0:
                    continue
                
                # Dimensiunile zonei originale (înainte de zoom)
                zone_orig_h, zone_orig_w = zone.shape[:2]
                
                # Preprocesăm zona
                zone_processed = preprocess_image_for_ocr(zone)
                
                # Aplicăm zoom dacă este necesar
                if zoom_factor > 1.0:
                    zone_h_scaled = int(zone_processed.shape[0] * zoom_factor)
                    zone_w_scaled = int(zone_processed.shape[1] * zoom_factor)
                    zone_zoomed = cv2.resize(zone_processed, (zone_w_scaled, zone_h_scaled), 
                                            interpolation=cv2.INTER_CUBIC)
                else:
                    zone_zoomed = zone_processed
                
                # Salvăm zonele procesate pentru debug (toate zonele)
                if steps_dir:
                    debug_path = Path(steps_dir) / f"02g_zone_{row+1}_{col+1}_processed.png"
                    cv2.imwrite(str(debug_path), zone_zoomed)
                    # Salvăm și zona originală pentru comparație
                    debug_path_orig = Path(steps_dir) / f"02g_zone_{row+1}_{col+1}_original.png"
                    cv2.imwrite(str(debug_path_orig), zone)
                
                # OCR pe zonă
                try:
                    ocr_data_zone = pytesseract.image_to_data(zone_zoomed, output_type=pytesseract.Output.DICT, lang='deu+eng')
                    
                    for i, text in enumerate(ocr_data_zone['text']):
                        if text.strip():
                            text_clean = text.strip()
                            text_lower = text_clean.lower()
                            
                            # Coordonatele în zona zoomed
                            rel_x_zoomed = ocr_data_zone['left'][i]
                            rel_y_zoomed = ocr_data_zone['top'][i]
                            rel_width_zoomed = ocr_data_zone['width'][i]
                            rel_height_zoomed = ocr_data_zone['height'][i]
                            
                            # Convertim coordonatele din zona zoomed la zona originală (fără zoom)
                            if zoom_factor > 1.0:
                                # Coordonatele relative în zona originală (fără zoom)
                                rel_x = rel_x_zoomed / zoom_factor
                                rel_y = rel_y_zoomed / zoom_factor
                                rel_width = rel_width_zoomed / zoom_factor
                                rel_height = rel_height_zoomed / zoom_factor
                            else:
                                rel_x = rel_x_zoomed
                                rel_y = rel_y_zoomed
                                rel_width = rel_width_zoomed
                                rel_height = rel_height_zoomed
                            
                            # Convertim la coordonatele absolute în imaginea completă
                            orig_x = int(rel_x) + x_start
                            orig_y = int(rel_y) + y_start
                            orig_width = int(rel_width)
                            orig_height = int(rel_height)
                            
                            # Verificăm că coordonatele sunt în limitele imaginii
                            orig_x = max(0, min(orig_x, w - 1))
                            orig_y = max(0, min(orig_y, h - 1))
                            orig_width = min(orig_width, w - orig_x)
                            orig_height = min(orig_height, h - orig_y)
                            
                            conf = ocr_data_zone['conf'][i]
                            
                            # Colectăm TOATE detecțiile pentru debug
                            all_detections.append((orig_x, orig_y, orig_width, orig_height, text_clean, conf))
                            
                            found_term = None
                            for term in search_terms:
                                term_lower = term.lower()
                                # Verificăm doar match exact (case-insensitive)
                                # Nu acceptăm fragmente - doar cuvântul complet "terasa" sau variantele sale exacte
                                if term_lower == text_lower:
                                    found_term = term
                                    break
                            
                            if found_term:
                                if conf > 60:
                                    text_boxes.append((orig_x, orig_y, orig_width, orig_height, text_clean, conf))
                                    print(f"         ✅ Detectat (zona {row+1},{col+1}): '{text_clean}' la ({orig_x}, {orig_y}) cu confidență {conf:.1f}%")
                            else:
                                # Colectăm și caracterele individuale pentru reconstituire
                                # Verificăm dacă caracterul face parte din unul dintre termenii căutați
                                for term in search_terms:
                                    term_lower = term.lower()
                                    if any(char in text_lower for char in term_lower):
                                        all_chars.append((orig_x, orig_y, orig_width, orig_height, text_clean, conf))
                                        break
                except Exception as e:
                    print(f"         ⚠️ Eroare OCR pe zona {row+1},{col+1}: {e}")
                    continue
    
    # 3. Dacă nu am găsit cuvinte complete, încercăm să reconstituim din caractere
    if not text_boxes and all_chars:
        print(f"         🔤 Am detectat {len(all_chars)} caractere individuale. Încerc să reconstitui cuvântul...")
        reconstructed = _reconstruct_word_from_chars(all_chars, search_terms)
        text_boxes.extend(reconstructed)
    
    # 4. Generăm imagine de debug cu TOATE detecțiile OCR
    if steps_dir and all_detections:
        # Convertim imaginea la BGR dacă este grayscale
        if len(image.shape) == 2:
            debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            debug_img = image.copy()
        else:
            debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        print(f"         📊 Generând imagine de debug cu {len(all_detections)} detecții OCR...")
        
        # Desenăm toate detecțiile
        for x, y, width, height, text, conf in all_detections:
            # Verificăm dacă detecția se potrivește cu unul dintre termenii căutați
            text_lower = text.lower()
            is_match = False
            for term in search_terms:
                if term.lower() == text_lower:
                    is_match = True
                    break
            
            # Culoare: verde pentru match-uri, albastru pentru restul
            color = (0, 255, 0) if is_match else (255, 0, 0)
            thickness = 3 if is_match else 2
            
            # Desenăm dreptunghiul
            cv2.rectangle(debug_img, (x, y), (x + width, y + height), color, thickness)
            
            # Desenăm textul cu procentajul
            label = f"{text} ({conf:.0f}%)"
            
            # Calculăm dimensiunea fontului în funcție de înălțimea detecției
            font_scale = max(0.4, height / 30.0)
            font_thickness = max(1, int(font_scale * 2))
            
            # Calculăm dimensiunea textului pentru a-l poziționa corect
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            # Poziționăm textul deasupra dreptunghiului (sau dedesubt dacă nu încape)
            text_y = y - 5 if y - 5 > text_height else y + height + text_height + 5
            
            # Desenăm fundal pentru text (pentru lizibilitate)
            cv2.rectangle(debug_img, 
                         (x, text_y - text_height - baseline), 
                         (x + text_width, text_y + baseline), 
                         color, -1)
            
            # Desenăm textul
            cv2.putText(debug_img, label, (x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        # Salvăm imaginea de debug
        debug_path = Path(steps_dir) / "02g_02_all_ocr_detections.png"
        cv2.imwrite(str(debug_path), debug_img)
        print(f"         💾 Salvat: 02g_02_all_ocr_detections.png ({len(all_detections)} detecții)")
    
    return text_boxes

def fill_terrace_room(walls_mask: np.ndarray, steps_dir: str = None) -> np.ndarray:
    """
    Detectează cuvântul "terasa" (sau variante în germană) în plan și umple camera respectivă cu flood fill.
    
    Folosește overlay-ul de 50% (ghost) pentru a detecta textul, apoi face flood fill
    în zona respectivă pentru a închide camera.
    
    Args:
        walls_mask: Masca pereților (255 = perete, 0 = spațiu liber)
        steps_dir: Director pentru salvarea step-urilor de debug (opțional)
    
    Returns:
        Masca pereților cu camera terasei umplută (dacă a fost detectată)
    """
    # Folosim o metodă alternativă dacă OCR nu este disponibil
    use_ocr = TESSERACT_AVAILABLE
    if not use_ocr:
        print(f"      ⚠️ pytesseract nu este disponibil. Folosesc metoda alternativă de detectare text.")
    
    h, w = walls_mask.shape[:2]
    result = walls_mask.copy()
    
    print(f"      🏡 Detectez și umplu camere (terasa/etc)...")
    
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
        print(f"         ⚠️ Nu s-a găsit overlay sau original. Skip detectarea terasei.")
        return result
    
    # Redimensionăm dacă este necesar
    if ocr_image.shape[:2] != (h, w):
        ocr_image = cv2.resize(ocr_image, (w, h))
    
    # Pas 2: Detectăm textul folosind OCR cu preprocesare și analiză pe zone
    print(f"         🔍 Pas 1: Detectez text (terasa/etc)...")
    
    # Variante ale cuvântului "terasa" (fără "erdgeschoss" care înseamnă parter)
    search_terms = [
        "terrasse", "Terrasse", "TERRASSE", "terasa", "Terasa", "TERASA",
        "terrace", "Terrace", "TERRACE",  # engleză
        "terrasa", "Terrasa", "TERRASA",  # variante
        "terras", "Terras", "TERRAS"  # variante scurte
    ]
    
    text_found = False
    text_boxes = []
    
    try:
        if use_ocr:
            # Metoda îmbunătățită: OCR cu preprocesare și analiză pe zone
            print(f"         📝 Folosesc OCR cu preprocesare și analiză pe zone...")
            
            # Salvez imaginea preprocesată pentru debug
            if steps_dir:
                processed_img = preprocess_image_for_ocr(ocr_image)
                cv2.imwrite(str(Path(steps_dir) / "02g_00_preprocessed.png"), processed_img)
                print(f"         💾 Salvat: 02g_00_preprocessed.png (imagine preprocesată)")
            
            # Rulează OCR pe zone cu zoom
            text_boxes = run_ocr_on_zones(ocr_image, search_terms, steps_dir, 
                                         grid_rows=3, grid_cols=3, zoom_factor=2.0)
            
            if text_boxes:
                text_found = True
        else:
            # Metoda 2: FĂRĂ OCR nu putem identifica specific cuvântul "terasa"
            # Deci nu mai detectăm zone de text generic, ci doar returnăm fără să facem nimic
            print(f"         ⚠️ Fără OCR nu pot identifica specific cuvântul 'terasa'.")
            print(f"         ⚠️ Metoda alternativă este dezactivată - necesită OCR pentru identificare precisă.")
            text_found = False
            text_boxes = []
        
        # Selectăm rezultatul cu confidence maxim (dacă există)
        if text_boxes:
            # Sortăm după confidence (descrescător) și luăm primul
            text_boxes.sort(key=lambda box: box[5], reverse=True)  # box[5] = confidence
            best_box = text_boxes[0]
            text_boxes = [best_box]  # Păstrăm doar cel mai bun rezultat
            print(f"         🎯 Selectat rezultatul cu confidence maxim: '{best_box[4]}' cu {best_box[5]:.1f}%")
        
        if not text_found:
            print(f"         ⚠️ Nu s-a detectat text (terasa/etc) în plan sau toate au confidence < 60%.")
            if steps_dir:
                vis_ocr = ocr_image.copy()
                cv2.imwrite(str(Path(steps_dir) / "02g_01_ocr_result.png"), vis_ocr)
                print(f"         💾 Salvat: 02g_01_ocr_result.png")
            return result
        
        # Pas 3: Vizualizăm textul detectat
        if steps_dir:
            vis_ocr = ocr_image.copy()
            for x, y, width, height, text, conf in text_boxes:
                cv2.rectangle(vis_ocr, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(vis_ocr, f"{text} ({conf:.0f}%)", (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite(str(Path(steps_dir) / "02g_01_ocr_result.png"), vis_ocr)
            print(f"         💾 Salvat: 02g_01_ocr_result.png (text detectat)")
        
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
        
        # Încărcăm outdoor_mask pentru a identifica zonele exterioare (terase)
        outdoor_mask = None
        if steps_dir:
            outdoor_mask_path = Path(steps_dir) / "03_outdoor_mask.png"
            if outdoor_mask_path.exists():
                outdoor_mask = cv2.imread(str(outdoor_mask_path), cv2.IMREAD_GRAYSCALE)
                if outdoor_mask is not None and outdoor_mask.shape[:2] != (h, w):
                    outdoor_mask = cv2.resize(outdoor_mask, (w, h))
                print(f"         📸 Folosesc outdoor_mask pentru filtrare")
        
        # Procesăm DOAR rezultatul cu confidence maxim (dacă există)
        rooms_filled = 0
        if text_boxes:
            # Procesăm doar primul (și singurul) rezultat - cel cu confidence maxim
            box_idx = 0
            x, y, width, height, text, conf = text_boxes[0]
            # Centrul textului
            center_x = x + width // 2
            center_y = y + height // 2
            
            # Verificăm dacă centrul este într-un spațiu liber (nu pe perete)
            if 0 <= center_y < h and 0 <= center_x < w:
                if spaces_mask[center_y, center_x] == 255:  # Spațiu liber
                    # Dacă OCR a detectat textul, facem flood fill direct (fără filtrare suplimentară)
                    # pentru că OCR deja a identificat specific cuvântul "terasa"
                    if not use_ocr:
                        # Dacă nu avem OCR, nu mai facem nimic (metoda alternativă este dezactivată)
                        print(f"         ⚠️ Fără OCR nu pot identifica specific cuvântul. Skip.")
                    else:
                        print(f"         🎯 Găsit cuvântul '{text}' (confidence {conf:.1f}%) - fac flood fill pentru această cameră...")
                        # Facem flood fill din centrul textului pe overlay-ul combinat
                        # Flood fill-ul se va opri automat când întâlnește pereți (linii închise în overlay)
                        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
                        flood_fill_flags = 4  # 4-conectivitate
                        flood_fill_flags |= cv2.FLOODFILL_MASK_ONLY
                        flood_fill_flags |= (255 << 8)  # Fill value
                        
                        seed_point = (center_x, center_y)
                        
                        # Folosim overlay-ul combinat pentru flood fill
                        if overlay_combined is not None:
                            # Convertim overlay-ul la grayscale pentru flood fill
                            overlay_for_fill = cv2.cvtColor(overlay_combined, cv2.COLOR_BGR2GRAY)
                            # Toleranță pentru a se opri la pereți (pereții sunt mai închiși în overlay)
                            lo_diff = 30  # Toleranță pentru diferențe de culoare
                            up_diff = 30
                            fill_image = overlay_for_fill.copy()
                            print(f"         🎨 Folosesc overlay combinat pentru flood fill")
                        else:
                            # Fallback la spaces_mask dacă nu avem overlay
                            fill_image = spaces_mask.copy()
                            lo_diff = 0  # Nu acceptă diferențe - se oprește exact la pereți
                            up_diff = 0
                            print(f"         ⚠️ Folosesc spaces_mask simplu (overlay indisponibil)")
                        
                        # Facem flood fill pe imaginea combinată
                        # Flood fill-ul se va opri automat când întâlnește pereți (valori diferite)
                        _, _, _, rect = cv2.floodFill(
                            fill_image, 
                            flood_mask, 
                            seed_point, 
                            128,  # Valoare de fill (nu este folosită cu FLOODFILL_MASK_ONLY)
                            lo_diff, 
                            up_diff, 
                            flood_fill_flags
                        )
                        
                        # Extragem zona umplută din mask
                        filled_region = (flood_mask[1:h+1, 1:w+1] == 255).astype(np.uint8) * 255
                        
                        # Verificăm că nu am umplut peste pereți (safety check)
                        # Zona umplută nu trebuie să conțină pixeli de perete
                        overlap_with_walls = np.sum((filled_region > 0) & (walls_mask > 0))
                        if overlap_with_walls > 0:
                            print(f"         ⚠️ Flood fill a depășit pereții ({overlap_with_walls} pixeli). Corectez...")
                            # Eliminăm pixeli care sunt pe pereți
                            filled_region = cv2.bitwise_and(filled_region, cv2.bitwise_not(walls_mask))
                        
                        # Verificăm dacă am umplut o zonă suficient de mare (minim 1000 pixeli)
                        filled_area = np.count_nonzero(filled_region)
                        if filled_area > 1000:
                            print(f"         🔍 Extrag conturul zonei detectate pentru a completa golurile...")
                            
                            # Inițializăm variabilele pentru vizualizare
                            gaps = None
                            wall_border = None
                            contours = None
                            
                            # Extragem conturul zonei umplute
                            contours, _ = cv2.findContours(filled_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if contours:
                                # Găsim cel mai mare contur (zona principală)
                                largest_contour = max(contours, key=cv2.contourArea)
                                
                                # Creăm o mască pentru conturul complet
                                contour_mask = np.zeros((h, w), dtype=np.uint8)
                                # Desenăm conturul cu grosime adaptivă (grosimea peretelui)
                                wall_thickness = max(3, int(min(w, h) * 0.003))  # Grosime adaptivă
                                cv2.drawContours(contour_mask, [largest_contour], -1, 255, wall_thickness)
                                
                                # Identificăm golurile: unde conturul există dar pereții nu există
                                # Golurile sunt în contur_mask dar nu în walls_mask
                                gaps = cv2.bitwise_and(contour_mask, cv2.bitwise_not(walls_mask))
                                
                                # Completăm doar golurile (nu desenăm peste pereții existenți)
                                walls_to_add = gaps
                                
                                # Adăugăm pereții noi (doar golurile) la masca finală
                                result = cv2.bitwise_or(result, walls_to_add)
                                
                                gaps_area = np.count_nonzero(gaps)
                                print(f"         ✅ Completat {gaps_area} pixeli de goluri în pereți conform conturului")
                            else:
                                print(f"         ⚠️ Nu s-au găsit contururi în zona umplută")
                                # Fallback la metoda veche (dilatare)
                                kernel_size = max(3, int(min(w, h) * 0.005))  # Adaptiv
                                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                                filled_dilated = cv2.dilate(filled_region, kernel, iterations=1)
                                wall_border = cv2.subtract(filled_dilated, filled_region)
                                result = cv2.bitwise_or(result, wall_border)
                            
                            rooms_filled += 1
                            
                            print(f"         ✅ Umplut camera '{text}': {filled_area} pixeli")
                            
                            # Vizualizăm zona umplută și golurile completate
                            if steps_dir:
                                vis_fill = cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
                                # Desenăm zona umplută cu transparență (galben)
                                filled_colored = np.zeros_like(vis_fill)
                                filled_colored[filled_region > 0] = [0, 255, 255]  # Galben
                                vis_fill = cv2.addWeighted(vis_fill, 0.7, filled_colored, 0.3, 0)
                                
                                # Desenăm conturul complet (albastru)
                                if contours and len(contours) > 0:
                                    largest_contour = max(contours, key=cv2.contourArea)
                                    cv2.drawContours(vis_fill, [largest_contour], -1, (255, 0, 0), 2)  # Albastru pentru contur
                                
                                # Desenăm golurile completate (verde)
                                if gaps is not None:
                                    gaps_colored = np.zeros_like(vis_fill)
                                    gaps_colored[gaps > 0] = [0, 255, 0]  # Verde pentru goluri completate
                                    vis_fill = cv2.addWeighted(vis_fill, 0.5, gaps_colored, 0.5, 0)
                                elif wall_border is not None:
                                    # Fallback: desenăm pereții noi (verde)
                                    wall_border_colored = np.zeros_like(vis_fill)
                                    wall_border_colored[wall_border > 0] = [0, 255, 0]  # Verde
                                    vis_fill = cv2.addWeighted(vis_fill, 0.5, wall_border_colored, 0.5, 0)
                                
                                # Desenăm centrul textului (roșu)
                                cv2.circle(vis_fill, (center_x, center_y), 5, (0, 0, 255), -1)
                                cv2.rectangle(vis_fill, (x, y), (x + width, y + height), (0, 255, 0), 2)
                                
                                output_path = Path(steps_dir) / f"02g_02_terrace_fill_{box_idx + 1}.png"
                                cv2.imwrite(str(output_path), vis_fill)
                                print(f"         💾 Salvat: {output_path.name}")
                            
                            # Am umplut camera terasei
                            print(f"         ✅ Gata! Am umplut camera terasei.")
                        else:
                            print(f"         ⚠️ Zona detectată prea mică ({filled_area} pixeli). Skip.")
                else:
                    print(f"         ⚠️ Centrul textului '{text}' este pe un perete. Skip.")
        
        if rooms_filled > 0:
            print(f"         ✅ Umplut {rooms_filled} camere de tip 'terasa'")
        else:
            print(f"         ⚠️ Nu s-au umplut camere (zone prea mici sau pe pereți)")
        
        # Pas 5: Salvăm rezultatul final
        if steps_dir:
            vis_final = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            # Comparăm cu originalul
            diff = cv2.subtract(result, walls_mask)
            diff_colored = np.zeros_like(vis_final)
            diff_colored[diff > 0] = [0, 255, 0]  # Verde pentru pereții noi
            vis_final = cv2.addWeighted(vis_final, 0.7, diff_colored, 0.3, 0)
            
            cv2.imwrite(str(Path(steps_dir) / "02g_03_final_result.png"), vis_final)
            print(f"         💾 Salvat: 02g_03_final_result.png (verde=pereți noi)")
    
    except Exception as e:
        print(f"         ❌ Eroare la detectarea/umplerea terasei: {e}")
        import traceback
        traceback.print_exc()
        return result
    
    return result

def detect_and_visualize_wall_closures(mask: np.ndarray, steps_dir: str = None) -> None:
    """
    Detectează camerele (spațiile libere închise de pereți) folosind Watershed cu Distance Transform.
    
    Această metodă detectează atât camerele complet închise cât și cele cu goluri în pereți,
    fără a distorsiona structura prin operații morfologice agresive.
    
    Args:
        mask: Masca pereților (255 = perete, 0 = spațiu liber)
        steps_dir: Director pentru salvarea step-urilor de debug (opțional)
    """
    h, w = mask.shape[:2]
    
    print(f"      🎨 Detectez camere folosind Watershed + Distance Transform...")
    
    # Creăm o imagine colorată (BGR) pentru vizualizare
    vis_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Masca: negru (0) = spațiu liber, alb (255) = perete
    spaces_mask = cv2.bitwise_not(mask)
    
    # Paletă de culori (BGR format pentru OpenCV)
    colors = [
        (255, 0, 0),      # Albastru
        (0, 255, 0),      # Verde
        (0, 0, 255),      # Roșu
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Galben
        (128, 0, 128),    # Mov
        (255, 165, 0),    # Portocaliu
        (0, 128, 255),    # Albastru deschis
        (128, 255, 0),    # Verde deschis
        (255, 192, 203),  # Roz
        (0, 255, 127),    # Verde primăvară
        (255, 20, 147),   # Roz adânc
        (0, 191, 255),    # Albastru cer
        (255, 140, 0),    # Portocaliu închis
    ]
    
    # Pas 1: Calculează Distance Transform
    # Distanța de la fiecare pixel la cel mai apropiat perete
    dist_transform = cv2.distanceTransform(spaces_mask, cv2.DIST_L2, 5)
    
    # Normalizăm pentru vizualizare (opțional, pentru debug)
    if steps_dir:
        dist_vis = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(str(Path(steps_dir) / "02c_distance_transform.png"), dist_vis)
    
    # Pas 2: Găsește maximale locale (centrele camerelor)
    # Folosim o metodă mai robustă: găsim maximale locale în distance transform
    # Threshold adaptiv bazat pe distribuția distanțelor
    max_dist = np.max(dist_transform)
    # Pentru camere cu goluri mari, folosim un threshold mai mic (30% din maxim)
    min_distance_threshold = max(5, max_dist * 0.3)
    
    # Aplicăm threshold pentru a găsi zonele cu distanță mare (centrele camerelor)
    _, sure_fg = cv2.threshold(dist_transform, min_distance_threshold, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    
    # Găsim maximale locale folosind peak detection
    # Folosim morphological operations pentru a găsi vârfurile locale
    kernel_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    local_maxima = cv2.morphologyEx(dist_transform, cv2.MORPH_TOPHAT, kernel_peak)
    _, local_maxima = cv2.threshold(local_maxima, max_dist * 0.2, 255, cv2.THRESH_BINARY)
    local_maxima = local_maxima.astype(np.uint8)
    
    # Combinăm threshold-ul simplu cu maximalele locale
    sure_fg = cv2.bitwise_or(sure_fg, local_maxima)
    
    # Pas 3: Găsește maximale locale folosind connected components
    num_markers, markers = cv2.connectedComponents(sure_fg)
    
    # Pas 4: Aplică Watershed
    # Marcăm pereții ca fiind sigur fundal
    sure_bg = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Adăugăm pereții la markeri (label 0 = fundal/pereți)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Aplică Watershed pe o copie a imaginii (watershed modifică imaginea)
    watershed_image = vis_image.copy()
    cv2.watershed(watershed_image, markers)
    
    # Pas 5: Desenează fiecare cameră detectată
    room_count = 0
    detected_rooms = []
    # Mască pentru zonele deja desenate (pentru a evita suprapunerea)
    drawn_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Colectăm toate camerele cu ariile lor pentru a le sorta și procesa corect
    rooms_data = []
    for label_id in range(2, num_markers + 1):  # Începem de la 2 (1 este pereții)
        room_mask = (markers == label_id).astype(np.uint8) * 255
        area = np.count_nonzero(room_mask)
        
        if area < 500:  # Ignorăm camere foarte mici
            continue
        
        rooms_data.append((label_id, area, room_mask))
    
    # Sortăm camerele după arie (descrescător) pentru a procesa mai întâi camerele mari
    rooms_data.sort(key=lambda x: x[1], reverse=True)
    
    # Procesăm fiecare cameră
    for label_id, area, room_mask in rooms_data:
        # Verificăm suprapunerea cu zonele deja desenate
        overlap = cv2.bitwise_and(room_mask, drawn_mask)
        overlap_area = np.count_nonzero(overlap)
        overlap_ratio = overlap_area / area if area > 0 else 0
        
        # Dacă mai mult de 15% din cameră este deja desenată, o ignorăm
        if overlap_ratio > 0.15:
            continue
        
        # Detectăm conturul camerei
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
        
        # Luăm cel mai mare contur
        contour = max(contours, key=cv2.contourArea)
        
        if len(contour) < 3:
            continue
        
        # Calculăm forma geometrică
        x, y, w_box, h_box = cv2.boundingRect(contour)
        bbox_area = w_box * h_box
        area_ratio = area / bbox_area if bbox_area > 0 else 0
        
        # Determinăm forma de desenat
        if area_ratio >= 0.6:  # Cameră bine definită
            if len(contour) > 3:
                shape_to_draw = cv2.convexHull(contour)
            else:
                shape_to_draw = contour
        else:  # Cameră parțial închisă
            if len(contour) > 3:
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0 and area / hull_area >= 0.6:
                    shape_to_draw = hull
                else:
                    shape_to_draw = np.array([[x, y], [x + w_box, y], 
                                             [x + w_box, y + h_box], [x, y + h_box]], dtype=np.int32)
            else:
                shape_to_draw = np.array([[x, y], [x + w_box, y], 
                                         [x + w_box, y + h_box], [x, y + h_box]], dtype=np.int32)
        
        # Selectăm culoarea
        color = colors[room_count % len(colors)]
        
        # Marcăm zona ca desenată ÎNAINTE de a desena (pentru a evita suprapunerea)
        cv2.fillPoly(drawn_mask, [shape_to_draw], 255)
        
        # Desenăm camera
        overlay = vis_image.copy()
        cv2.fillPoly(overlay, [shape_to_draw], color)
        cv2.addWeighted(overlay, 0.4, vis_image, 0.6, 0, vis_image)
        cv2.polylines(vis_image, [shape_to_draw], True, color, 3)
        
        # Marcăm camera ca detectată
        detected_rooms.append((label_id, area, room_mask))
        room_count += 1
    
    # Salvăm imaginea colorată
    if steps_dir:
        output_path = Path(steps_dir) / "02c_wall_closures_visualized.png"
        cv2.imwrite(str(output_path), vis_image)
        print(f"      ✅ Detectat {room_count} camere, salvat în {output_path.name}")
    else:
        print(f"      ✅ Detectat {room_count} camere")

def bridge_wall_gaps(walls_raw: np.ndarray, image_dims: tuple, steps_dir: str = None) -> np.ndarray:
    """Umple DOAR gap-urile între pereți - îmbunătățit pentru VPS."""
    h, w = image_dims
    min_dim = min(h, w)
    
    # Mărit kernel size de la 1.6% la 2.0% pentru VPS
    kernel_size = max(15, int(min_dim * 0.020))  # Mărit de la 0.016 la 0.020
    if kernel_size % 2 == 0: kernel_size += 1
    
    print(f"      🌉 Bridging gaps between walls: kernel {kernel_size}x{kernel_size} (îmbunătățit pentru VPS)")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    original_walls = walls_raw.copy()
    if steps_dir:
        save_step("bridge_01_original", original_walls, steps_dir)
    
    # Mărit iterațiile de la 2 la 3 pentru VPS
    walls_closed = cv2.morphologyEx(walls_raw, cv2.MORPH_CLOSE, kernel, iterations=3)
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
    """Closing inteligent - îmbunătățit pentru VPS."""
    h, w = image_dims
    min_dim = min(h, w)
    
    print(f"      🧠 Smart closing: detecting intentional openings (îmbunătățit pentru VPS)...")
    
    # Mărit kernel size de la 0.9% la 1.2% pentru VPS
    kernel_large = max(9, int(min_dim * 0.012))  # Mărit de la 0.009 la 0.012
    if kernel_large % 2 == 0: kernel_large += 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_large, kernel_large))
    # Mărit iterațiile de la 2 la 3 pentru VPS
    walls_fully_closed = cv2.morphologyEx(walls_raw, cv2.MORPH_CLOSE, kernel, iterations=3)
    
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

def skeletonize_and_straighten_walls(walls_mask: np.ndarray, steps_dir: str = None) -> np.ndarray:
    """
    Skeletonizare, îndreptare și uniformizare grosime pereți.
    
    Pași:
    1. Calculează media grosimii pereților
    2. Aplică skeletonizarea pentru a obține linii de 1 pixel
    3. Îndrepte liniile care nu sunt foarte drepte
    4. Aplică aceeași grosime (media calculată) la toate liniile
    
    Args:
        walls_mask: Masca pereților (255 = perete, 0 = spațiu liber)
        steps_dir: Director pentru salvarea step-urilor de debug (opțional)
    
    Returns:
        Masca pereților cu skeletonizare, îndreptare și grosime uniformă
    """
    if np.count_nonzero(walls_mask) == 0:
        return walls_mask
    
    h, w = walls_mask.shape[:2]
    print(f"      🦴 Skeletonizare și îndreptare pereți...")
    
    # Pas 1: Calculăm media grosimii pereților
    print(f"         🔍 Pas 1: Calculez media grosimii pereților...")
    
    # Distance transform pe masca de pereți (distanța de la centru la margine)
    # Aceasta ne dă distanța de la fiecare pixel de perete la marginea cea mai apropiată
    dist_from_edge = cv2.distanceTransform(walls_mask, cv2.DIST_L2, 5)
    
    # Grosimea într-un punct = 2 * distanța (pentru că avem două margini)
    wall_thicknesses = dist_from_edge[walls_mask > 0] * 2
    
    if len(wall_thicknesses) == 0:
        print(f"         ⚠️ Nu s-au găsit pereți pentru calcul grosime")
        return walls_mask
    
    # Calculăm media grosimii (folosim median pentru a fi mai robust la outliers)
    avg_thickness = np.median(wall_thicknesses)
    avg_thickness_int = max(3, int(round(avg_thickness)))
    
    print(f"         ✅ Media grosimii pereților: {avg_thickness:.2f}px (folosit: {avg_thickness_int}px)")
    
    if steps_dir:
        # Vizualizăm distance transform pentru debug
        dist_vis = cv2.normalize(dist_from_edge, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(str(Path(steps_dir) / "02h_01_distance_transform.png"), dist_vis)
        print(f"         💾 Salvat: 02h_01_distance_transform.png")
    
    # Pas 2: Skeletonizare
    print(f"         🔍 Pas 2: Aplic skeletonizare...")
    
    # Convertim în format binar (0 și 1) pentru skeletonize
    binary_norm = (walls_mask / 255).astype(bool)
    
    # Aplicăm Skeletonization
    skeleton = skeletonize(binary_norm)
    
    # Convertim înapoi în format OpenCV (0-255)
    skeleton_img = (skeleton.astype(np.uint8) * 255)
    
    # Conectăm liniile discontinue folosind morphological closing pe distanțe mici
    # Folosim un kernel mic pentru a conecta doar liniile foarte apropiate
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skeleton_img = cv2.morphologyEx(skeleton_img, cv2.MORPH_CLOSE, connect_kernel, iterations=1)
    
    skeleton_pixels = np.count_nonzero(skeleton_img)
    print(f"         ✅ Schelet generat: {skeleton_pixels:,} pixeli (1px grosime)")
    
    if steps_dir:
        cv2.imwrite(str(Path(steps_dir) / "02h_02_skeleton.png"), skeleton_img)
        print(f"         💾 Salvat: 02h_02_skeleton.png")
    
    # Pas 3: Îndreptare liniilor
    print(f"         🔍 Pas 3: Îndrept liniile (NU șterg pixeli, doar înlocuiesc liniile tremurate)...")
    
    # Pornim cu skeleton-ul original - NU ștergem nimic
    skeleton_straight = skeleton_img.copy()
    
    # Folosim HoughLinesP pentru a detecta linii drepte
    lines = cv2.HoughLinesP(
        skeleton_img,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=max(15, int(min(h, w) * 0.015)),
        maxLineGap=max(10, int(min(h, w) * 0.008))
    )
    
    if lines is None or len(lines) == 0:
        print(f"         ⚠️ Nu s-au detectat linii cu HoughLinesP, păstrez skeleton-ul original")
        # Păstrăm skeleton-ul original fără modificări
    else:
        print(f"         ✅ Detectat {len(lines)} linii cu HoughLinesP - înlocuiesc segmentele tremurate")
        
        # Pentru fiecare linie detectată, găsim pixeli din skeleton care sunt aproape
        # și îi înlocuim cu linia dreaptă
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculăm distanța de la fiecare pixel din skeleton la linia dreaptă
            # Dacă distanța este mică, înlocuim pixelul cu linia dreaptă
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if line_length < 5:  # Ignorăm linii prea scurte
                continue
            
            # Găsim toți pixelii din skeleton care sunt aproape de această linie
            # Folosim o bandă de toleranță de ~3 pixeli
            tolerance = 3
            
            # Creăm o mască temporară pentru linia dreaptă cu o bandă de toleranță
            line_mask = np.zeros_like(skeleton_img)
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, tolerance * 2 + 1)
            
            # Găsim intersecția dintre skeleton și banda de toleranță
            pixels_near_line = cv2.bitwise_and(skeleton_img, line_mask)
            
            # Dacă există pixeli aproape de linie, îi înlocuim cu linia dreaptă
            if np.count_nonzero(pixels_near_line) > 0:
                # Ștergem pixeli din skeleton care sunt în banda de toleranță
                skeleton_straight = cv2.bitwise_and(skeleton_straight, cv2.bitwise_not(line_mask))
                # Adăugăm linia dreaptă
                cv2.line(skeleton_straight, (x1, y1), (x2, y2), 255, 1)
    
    # Aplicăm straighten_mask pentru a netezi colțurile (dar păstrăm toți pixelii)
    skeleton_straight = straighten_mask(skeleton_straight, epsilon_factor=0.002)
    
    if steps_dir:
        vis_straight = cv2.cvtColor(skeleton_straight, cv2.COLOR_GRAY2BGR)
        if lines is not None and len(lines) > 0:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(vis_straight, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imwrite(str(Path(steps_dir) / "02h_03_straightened.png"), vis_straight)
        print(f"         💾 Salvat: 02h_03_straightened.png")
    
    # Pas 4: Aplicăm aceeași grosime la toate liniile (grosimea calculată ÎNAINTE de skeletonizare)
    print(f"         🔍 Pas 4: Aplic grosime uniformă ({avg_thickness_int}px - grosimea originală)...")
    
    # Creăm un kernel pătrat pentru a da grosime uniformă și pereți pătrați
    # Pentru a obține grosimea avg_thickness_int, folosim un kernel mai mic (pentru a compensa)
    # Când dilatezi cu un kernel pătrat de dimensiune N, adaugi (N-1)/2 pixeli pe fiecare parte
    # Folosim kernel_size mai mic pentru a obține grosimea exactă
    kernel_size = max(3, avg_thickness_int - 2)
    # Asigurăm că e impar (pentru kernel-uri morfologice)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Folosim MORPH_RECT pentru pereți pătrați (nu MORPH_ELLIPSE care face pereți rotunjiți)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Dilatăm skeleton-ul îndreptat cu grosimea calculată inițial
    # Folosim o singură iterație pentru a obține grosimea exactă
    walls_final = cv2.dilate(skeleton_straight, kernel, iterations=1)
    
    # Verificăm grosimea rezultată pentru debug
    if steps_dir:
        # Calculăm grosimea efectivă a pereților rezultați
        walls_final_inv = cv2.bitwise_not(walls_final)
        dist_final = cv2.distanceTransform(walls_final_inv, cv2.DIST_L2, 5)
        final_thicknesses = dist_final[walls_final > 0] * 2
        if len(final_thicknesses) > 0:
            final_avg_thickness = np.median(final_thicknesses)
            print(f"         📊 Grosime efectivă rezultată: {final_avg_thickness:.2f}px (țintă: {avg_thickness_int}px, kernel: {kernel_size}px)")
    
    final_pixels = np.count_nonzero(walls_final)
    print(f"         ✅ Pereți finalizați: {final_pixels:,} pixeli (grosime uniformă {avg_thickness_int}px)")
    
    if steps_dir:
        # Vizualizare comparativă
        vis_comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
        
        # Original
        vis_comparison[:, :w] = cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(vis_comparison, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Skeleton
        vis_comparison[:, w:2*w] = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2BGR)
        cv2.putText(vis_comparison, "Skeleton", (w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Final
        vis_comparison[:, 2*w:] = cv2.cvtColor(walls_final, cv2.COLOR_GRAY2BGR)
        cv2.putText(vis_comparison, "Final", (2*w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite(str(Path(steps_dir) / "02h_04_comparison.png"), vis_comparison)
        print(f"         💾 Salvat: 02h_04_comparison.png")
        
        # Rezultat final
        cv2.imwrite(str(Path(steps_dir) / "02h_05_final_walls.png"), walls_final)
        print(f"         💾 Salvat: 02h_05_final_walls.png")
    
    return walls_final

def generate_interior_walls_skeleton(walls_mask: np.ndarray, outdoor_mask: np.ndarray) -> np.ndarray:
    """
    Generează scheletul (skeleton) pereților interiori cu grosime de 1 pixel.
    
    Args:
        walls_mask: Masca cu pereții (alb = pereți, negru = spațiu liber)
        outdoor_mask: Masca exterioară (alb = exterior, negru = interior)
    
    Returns:
        Imagine binară cu scheletul pereților interiori (1 pixel grosime)
    """
    if np.count_nonzero(walls_mask) == 0:
        return np.zeros_like(walls_mask)
    
    print("   🦴 Generez scheletul pereților interiori...")
    
    # 1. Pre-procesare: Curățăm imaginea de zgomot mic
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    walls_cleaned = cv2.morphologyEx(walls_mask, cv2.MORPH_OPEN, kernel_clean, iterations=1)
    walls_cleaned = cv2.morphologyEx(walls_cleaned, cv2.MORPH_CLOSE, kernel_clean, iterations=1)
    
    # 2. Convertim în format binar (0 și 1) pentru skeletonize
    binary_norm = (walls_cleaned / 255).astype(bool)
    
    # 3. Aplicăm Skeletonization
    skeleton = skeletonize(binary_norm)
    
    # 4. Convertim înapoi în format OpenCV (0-255)
    skeleton_img = (skeleton.astype(np.uint8) * 255)
    
    # 5. Eliminăm pereții exteriori
    # Mărim puțin masca exterioară (Dilation) ca să fim siguri că acoperim tot perimetrul
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    exterior_dilated = cv2.dilate(outdoor_mask, kernel_dilate, iterations=1)
    
    # Eliminăm perimetrul exterior din schelet
    interior_skeleton = cv2.bitwise_and(skeleton_img, cv2.bitwise_not(exterior_dilated))
    
    # 6. Opțional: Eliminăm "crenguțe" mici (spur pixels) - pruning simplu
    # Eliminăm componente conectate foarte mici (mai mici de 10 pixeli)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(interior_skeleton, connectivity=8)
    cleaned_skeleton = np.zeros_like(interior_skeleton)
    
    min_component_size = 10  # Păstrăm doar componentele cu cel puțin 10 pixeli
    for label_id in range(1, num_labels):  # Skip background (label 0)
        component_size = stats[label_id, cv2.CC_STAT_AREA]
        if component_size >= min_component_size:
            cleaned_skeleton[labels == label_id] = 255
    
    skeleton_pixels = np.count_nonzero(cleaned_skeleton)
    print(f"      ✅ Schelet generat: {skeleton_pixels:,} pixeli (1px grosime)")
    
    return cleaned_skeleton

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

    # ✅ CLOSING ADAPTIV ULTRA-PUTERNIC (UNIFICAT) - ÎMBUNĂTĂȚIT PENTRU VPS
    print("      🔗 Închid găuri (closing adaptiv ULTRA-PUTERNIC - îmbunătățit pentru VPS)...")

    if min_dim > 2500:
        # Imagini mari: closing mai puternic pentru VPS (mărit de la 0.3% la 0.5%)
        close_kernel_size = max(5, int(min_dim * 0.005))  # 0.5% (mărit de la 0.3%)
        close_iterations = 3  # Mărit de la 2 la 3 iterații
        print(f"         Mode: LARGE IMAGE → Close: {close_kernel_size}px (0.5%), iter={close_iterations}")
    else:
        # Imagini mici: closing ULTRA-PUTERNIC (mărit de la 1.0% la 1.5% + mai multe iterații)
        close_kernel_size = max(12, int(min_dim * 0.015))  # 1.5% (mărit de la 1.0%)
        close_iterations = 7  # Mărit de la 5 la 7 iterații
        print(f"         Mode: SMALL IMAGE → ULTRA STRONG close: {close_kernel_size}px (1.5%), iter={close_iterations}")

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
    
    # Acum, `ai_walls_raw` conține deja closing-ul ULTRA-PUTERNIC (îmbunătățit pentru VPS)
    if h_orig > LARGE_IMAGE_THRESHOLD or w_orig > LARGE_IMAGE_THRESHOLD:
        print(f"      🔧 LARGE IMAGE MODE: Bridging gaps between walls (no wall thickening)")
        # Menținem bridge_wall_gaps DOAR pentru imaginile mari, unde closing-ul a fost îmbunătățit (0.5%)
        ai_walls_closed = bridge_wall_gaps(ai_walls_raw, (h_orig, w_orig), str(steps_dir))
    else:
        # Pentru imagini mici: SKIP extra bridging (adaptive closing-ul ULTRA-PUTERNIC de mai sus e suficient)
        print(f"      🔧 SMALL IMAGE MODE: Skip extra bridging (adaptive closing is enough)")
        ai_walls_closed = ai_walls_raw.copy()
    
    save_step("02_ai_walls_closed", ai_walls_closed, str(steps_dir))
    
    # 4a. GENEREZ IMAGINE SUPrapusă: 02_ai_walls_closed + 00_original (cu transparență)
    if steps_dir:
        try:
            original_path = Path(steps_dir) / "00_original.png"
            if original_path.exists():
                original_img = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
                if original_img is not None:
                    # Redimensionăm dacă este necesar
                    if original_img.shape[:2] != ai_walls_closed.shape[:2]:
                        original_img = cv2.resize(original_img, (ai_walls_closed.shape[1], ai_walls_closed.shape[0]))
                    
                    # Facem invert la original (negru devine alb, alb devine negru)
                    original_inverted = cv2.bitwise_not(original_img)
                    
                    # Convertim la BGR pentru suprapunere
                    original_bgr = cv2.cvtColor(original_inverted, cv2.COLOR_GRAY2BGR)
                    walls_bgr = cv2.cvtColor(ai_walls_closed, cv2.COLOR_GRAY2BGR)
                    
                    # Suprapunem cu transparență 50% (original) + 50% (walls_closed)
                    overlay = cv2.addWeighted(original_bgr, 0.5, walls_bgr, 0.5, 0)
                    
                    # Salvăm rezultatul
                    output_path = Path(steps_dir) / "02d_walls_closed_overlay.png"
                    cv2.imwrite(str(output_path), overlay)
                    print(f"      📸 Generat overlay: {output_path.name}")
                    
                    # 4a.2. OVERLAY GENERAT - Nu mai facem reparări cu Hough Lines sau skeleton
                    # Overlay-ul este generat doar pentru a fi folosit de fill_terrace_room
        except Exception as e:
            print(f"      ⚠️ Nu s-a putut genera overlay: {e}")
    
    # 4b. WORKFLOW REORGANIZAT:
    # 1. Detectare terasa + flood fill + completare pereți terasa
    # 2. Umplere găuri mici dintre pereți
    
    print("   🏡 Pas 1: Detectare terasa și completare pereți...")
    # Detectează terasa, face flood fill și completează pereții terasei
    ai_walls_terrace = fill_terrace_room(ai_walls_closed, steps_dir=str(steps_dir))
    
    # Pas nou: Skeletonizare, îndreptare și uniformizare grosime pereți
    print("   🦴 Pas 2: Skeletonizare, îndreptare și uniformizare grosime pereți...")
    ai_walls_skeletonized = skeletonize_and_straighten_walls(ai_walls_terrace, steps_dir=str(steps_dir))
    
    # Folosim rezultatul de la skeletonizare
    ai_walls_final = ai_walls_skeletonized.copy()
    
    # 4d. DETECTARE ȘI VIZUALIZARE ÎNCHIDERI PEREȚI
    detect_and_visualize_wall_closures(ai_walls_final, steps_dir=str(steps_dir))
    
    # Kernel repair pentru restul procesării
    min_dim = min(h_orig, w_orig) 
    rep_k = max(3, int(min_dim * 0.005))
    if rep_k % 2 == 0: rep_k += 1
    kernel_repair = cv2.getStructuringElement(cv2.MORPH_RECT, (rep_k, rep_k))
    
    # 5. NETEZIRE
    ai_walls_smoothed = smooth_walls_mask(ai_walls_final)
    save_step("03_ai_walls_smoothed", ai_walls_smoothed, str(steps_dir))
    
    # 6. ÎNDREPTARE
    ai_walls_straight = straighten_mask(ai_walls_final, 0.003)
    save_step("03_ai_walls_straight", ai_walls_straight, str(steps_dir))

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
    scale_result = detect_scale_from_room_labels(
        str(image_path),
        indoor_mask,
        ai_walls_final,
        str(steps_dir),
        api_key=gemini_api_key
    )
    
    if not scale_result:
        raise RuntimeError("Nu am putut determina scala!")
    
    m_px = scale_result["meters_per_pixel"]

    # 9. MĂSURĂTORI
    print("   📏 Calculez măsurători...")
    outline = get_strict_1px_outline(ai_walls_final)
    touch_zone = cv2.dilate(outdoor_mask, kernel_grow, iterations=2)
    
    outline_ext_mask = cv2.bitwise_and(outline, touch_zone)
    outline_int_mask = cv2.subtract(outline, outline_ext_mask)
    
    # 9a. GENERARE SCHELET PEREȚI INTERIORI
    print("   🦴 Generez scheletul pereților interiori...")
    interior_walls_skeleton = generate_interior_walls_skeleton(ai_walls_final, outdoor_mask)
    save_step("04_interior_walls_skeleton", interior_walls_skeleton, str(steps_dir))
    
    # Calculăm lungimea skeleton-ului pereților interiori (din imaginea generată)
    px_len_skeleton_int = int(np.count_nonzero(interior_walls_skeleton))
    
    # Calculăm lungimea pereților exteriori (din outline)
    px_len_skeleton_ext = int(np.count_nonzero(outline_ext_mask))
    
    # Lungimea structurii pereților interiori = skeleton interior - exterior
    # (scădem exteriorul pentru că skeleton-ul interior poate conține și pereții exteriori)
    px_len_skeleton_structure_int = max(0, px_len_skeleton_int - px_len_skeleton_ext)
    
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
    walls_skeleton_int_m = px_len_skeleton_int * m_px  # Skeleton interior (din imagine)
    walls_skeleton_structure_int_m = px_len_skeleton_structure_int * m_px  # Structură pereți interiori (skeleton - exterior)
    
    area_indoor_m2 = px_area_indoor * (m_px ** 2)
    area_total_m2 = px_area_total * (m_px ** 2)

    measurements = {
        "pixels": {
            "walls_len_ext": int(px_len_ext),
            "walls_len_int": int(px_len_int),
            "walls_skeleton_ext": int(px_len_skeleton_ext),
            "walls_skeleton_int": int(px_len_skeleton_int),
            "walls_skeleton_structure_int": int(px_len_skeleton_structure_int),
            "indoor_area": int(px_area_indoor),
            "total_area": int(px_area_total)
        },
        "metrics": {
            "scale_m_per_px": float(m_px),
            "walls_ext_m": float(round(walls_ext_m, 2)),  # Pentru finisaje
            "walls_int_m": float(round(walls_int_m, 2)),  # Pentru finisaje
            "walls_skeleton_ext_m": float(round(walls_skeleton_ext_m, 2)),
            "walls_skeleton_int_m": float(round(walls_skeleton_int_m, 2)),
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
    print(f"      - Skeleton interior (din imagine): {walls_skeleton_int_m:.2f} m")
    print(f"      - Skeleton exterior (din outline): {walls_skeleton_ext_m:.2f} m")
    print(f"      - Structură interior (skeleton - exterior): {walls_skeleton_structure_int_m:.2f} m")
    
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
            if interior_zone_mask.shape[:2] != ai_walls_smoothed.shape[:2]:
                interior_zone_mask = cv2.resize(interior_zone_mask, (ai_walls_smoothed.shape[1], ai_walls_smoothed.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Calculăm grosimea medie a pereților pentru marjă de eroare
            dist_from_edge = cv2.distanceTransform(ai_walls_smoothed, cv2.DIST_L2, 5)
            wall_thicknesses = dist_from_edge[ai_walls_smoothed > 0] * 2
            if len(wall_thicknesses) > 0:
                avg_thickness = np.median(wall_thicknesses)
                margin_size = max(3, int(round(avg_thickness)))
                
                print(f"      📏 Grosime medie pereți: {avg_thickness:.2f}px, marjă de eroare: {margin_size}px")
                
                # Dilatăm masca zonei interioare cu marja de eroare (grosimea pereților)
                kernel_margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin_size * 2 + 1, margin_size * 2 + 1))
                interior_zone_mask_dilated = cv2.dilate(interior_zone_mask, kernel_margin, iterations=1)
                
                # Aplicăm masca pe pereți: păstrăm doar pereții din zona interioară (cu marjă)
                ai_walls_smoothed_filtered = cv2.bitwise_and(ai_walls_smoothed, interior_zone_mask_dilated)
                
                # Folosim pereții filtrați pentru generarea 3D
                ai_walls_for_3d = ai_walls_smoothed_filtered
                
                removed_pixels = np.count_nonzero(ai_walls_smoothed) - np.count_nonzero(ai_walls_smoothed_filtered)
                print(f"      ✅ Eliminat {removed_pixels:,} pixeli de zgomot din afara zonei interioare")
            else:
                ai_walls_for_3d = ai_walls_smoothed
                print(f"      ⚠️ Nu s-a putut calcula grosimea, folosesc pereții fără filtrare")
        else:
            ai_walls_for_3d = ai_walls_smoothed
            print(f"      ⚠️ Nu s-a putut încărca masca, folosesc pereții fără filtrare")
    else:
        ai_walls_for_3d = ai_walls_smoothed
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
    overlay[outline_ext_mask > 0] = [255, 0, 0]
    overlay[outline_int_mask > 0] = [0, 255, 0]
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