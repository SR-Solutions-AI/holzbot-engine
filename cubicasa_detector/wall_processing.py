# file: engine/cubicasa_detector/wall_processing.py
"""
Funcții pentru procesarea pereților: filtrare, reparare, unire gap-uri, etc.
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def save_step(name, img, steps_dir):
    """Helper pentru salvarea step-urilor de debug."""
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


def bridge_wall_gaps(walls_raw: np.ndarray, image_dims: tuple, steps_dir: str = None) -> np.ndarray:
    """Umple DOAR gap-urile între pereți - îmbunătățit pentru VPS."""
    h, w = image_dims
    min_dim = min(h, w)
    
    # Kernel size redus pentru a nu uni pereți paraleli
    kernel_size = max(15, int(min_dim * 0.016))  # Redus înapoi la 0.016
    if kernel_size % 2 == 0: kernel_size += 1
    
    print(f"      🌉 Bridging gaps between walls: kernel {kernel_size}x{kernel_size}")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    original_walls = walls_raw.copy()
    if steps_dir:
        save_step("bridge_01_original", original_walls, steps_dir)
    
    # Redus iterațiile înapoi la 2 pentru a nu uni pereți paraleli
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
    """Closing inteligent - îmbunătățit pentru VPS."""
    h, w = image_dims
    min_dim = min(h, w)
    
    print(f"      🧠 Smart closing: detecting intentional openings...")
    
    # Kernel size redus pentru a nu uni pereți paraleli
    kernel_large = max(9, int(min_dim * 0.009))  # Redus înapoi la 0.009
    if kernel_large % 2 == 0: kernel_large += 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_large, kernel_large))
    # Redus iterațiile înapoi la 2 pentru a nu uni pereți paraleli
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
    """Generează outline strict de 1 pixel grosime."""
    if np.count_nonzero(mask) == 0:
        return np.zeros_like(mask)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    return cv2.subtract(mask, eroded)
