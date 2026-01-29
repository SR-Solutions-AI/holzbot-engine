# file: engine/cubicasa_detector/wall_repair.py
"""
Module pentru repararea pereților.
Conține funcții pentru repararea pereților exteriori, închiderea gap-urilor și closing inteligent.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def save_step(name, img, steps_dir):
    """Salvează un step de debug."""
    if steps_dir:
        path = Path(steps_dir) / f"{name}.png"
        cv2.imwrite(str(path), img)


def repair_house_walls_with_floodfill(walls_mask: np.ndarray, steps_dir: str = None) -> np.ndarray:
    """
    Încearcă să repare toți pereții de jur-împrejurul casei folosind aceeași
    idee ca la terasă: flood fill în interiorul casei, apoi completarea
    golurilor de pe conturul regiunii umplute.
    
    walls_mask trebuie să includă deja pereții adăugați pentru terasă/garaj/scări,
    astfel încât envelope-ul să includă automat aceste zone în calcul.
    
    Args:
        walls_mask: Masca pereților (255 = perete, 0 = spațiu liber)
                    Trebuie să includă deja pereții adăugați pentru terasă/garaj/scări
        steps_dir: Director pentru salvarea imaginilor de debug (opțional)
    
    Returns:
        Masca pereților cu pereții exteriori reparați (dacă a fost posibil)
    """
    if walls_mask is None:
        print("      ⚠️ repair_house_walls_with_floodfill: walls_mask este None. Skip.")
        return None
    
    try:
        h, w = walls_mask.shape[:2]
    except AttributeError:
        print("      ⚠️ repair_house_walls_with_floodfill: walls_mask nu are shape. Skip.")
        return None
    
    print("      🏠 Repar pereții exteriori folosind ENVELOPE-ul casei (din pereți).")
    print("         ℹ️ Folosesc walls_mask care include deja pereții adăugați pentru terasă/garaj.")
    
    # Creăm folder pentru output-uri structurate
    wall_repair_output_dir = None
    if steps_dir:
        wall_repair_output_dir = Path(steps_dir) / "wall_repair"
        wall_repair_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Identificăm componentele CONECTATE de pereți
    #    - dacă între două zone de pereți nu există conectivitate, sunt clădiri separate
    #    - walls_mask include deja pereții adăugați pentru terasă/garaj, deci envelope-ul
    #      va include automat aceste zone
    walls_binary = (walls_mask > 0).astype(np.uint8) * 255
    num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(walls_binary, connectivity=8)
    
    if num_components < 2:  # 0 = background, 1+ = componente
        print("         ⚠️ Nu am găsit componente conectate de pereți. Skip reparare pereți exteriori.")
        return walls_mask
    
    print(f"         🔍 Am găsit {num_components - 1} componente conectate de pereți.")
    
    # 2. Pentru fiecare componentă conectată, construim un ENVELOPE separat
    #    și reparăm pereții separat (nu unim clădiri separate)
    result = walls_mask.copy()
    spaces_mask = cv2.bitwise_not(walls_mask)
    img_area = h * w
    
    # Pentru debug: colectăm toate envelope-urile și interior-urile
    all_envelope_masks = []
    all_interior_masks = []
    all_exterior_masks = []
    all_gaps = []
    
    for comp_id in range(1, num_components):  # Skip background (0)
        comp_area = stats[comp_id, cv2.CC_STAT_AREA]
        if comp_area < 100:  # Skip componente foarte mici
            continue
        
        # Extragem masca componentei curente
        # walls_mask include deja pereții adăugați pentru terasă/garaj, deci comp_mask
        # va include automat aceste zone dacă sunt conectate la această componentă
        comp_mask = (labels == comp_id).astype(np.uint8) * 255
        
        # Găsim contururile acestei componente
        # Acestea includ deja pereții adăugați pentru terasă/garaj dacă sunt conectate
        comp_contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not comp_contours:
            continue
        
        # Construim ENVELOPE pentru această componentă (convex hull)
        # Envelope-ul va include automat zonele de terasă/garaj dacă sunt conectate
        comp_points = np.vstack(comp_contours)
        comp_hull = cv2.convexHull(comp_points)
        
        comp_envelope = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(comp_envelope, [comp_hull], 255)
        
        comp_envelope_area = int(np.count_nonzero(comp_envelope))
        comp_envelope_ratio = comp_envelope_area / float(img_area) if img_area > 0 else 0.0
        
        if comp_envelope_area < 1000:
            print(f"         ⚠️ Componenta {comp_id}: envelope prea mic ({comp_envelope_area}px). Skip.")
            continue
        
        print(f"         📐 Componenta {comp_id}: envelope_area={comp_envelope_area}px ({comp_envelope_ratio*100:.1f}%)")
        
        # INTERIOR pentru această componentă = spațiu liber din interiorul envelope-ului
        comp_interior_raw = cv2.bitwise_and(spaces_mask, comp_envelope)
        comp_interior_area_before = int(np.count_nonzero(comp_interior_raw))
        
        # Eliminăm artefactele mici (componente conectate prea mici) din masca cu interiorul
        comp_interior_binary = (comp_interior_raw > 0).astype(np.uint8) * 255
        num_interior_components, interior_labels, interior_stats, _ = cv2.connectedComponentsWithStats(comp_interior_binary, connectivity=8)
        
        # Păstrăm doar componentele mari (mai mari de 0.5% din imagine sau minim 500px)
        min_interior_area = max(500, int(img_area * 0.005))
        comp_interior_cleaned = np.zeros_like(comp_interior_raw)
        
        for interior_comp_id in range(1, num_interior_components):
            interior_comp_area = interior_stats[interior_comp_id, cv2.CC_STAT_AREA]
            if interior_comp_area >= min_interior_area:
                comp_interior_cleaned[interior_labels == interior_comp_id] = 255
        
        comp_interior = comp_interior_cleaned
        comp_interior_area = int(np.count_nonzero(comp_interior))
        comp_interior_ratio = comp_interior_area / float(img_area) if img_area > 0 else 0.0
        removed_artifacts = comp_interior_area_before - comp_interior_area
        
        if comp_interior_area < 1000:
            print(f"         ⚠️ Componenta {comp_id}: interior prea mic ({comp_interior_area}px) după eliminarea artefactelor. Skip.")
            continue
        
        print(f"         📐 Componenta {comp_id}: interior_area={comp_interior_area}px ({comp_interior_ratio*100:.1f}%), artefacte eliminate: {removed_artifacts}px")
        
        # EXTERIOR pentru această componentă (doar pentru debug)
        comp_exterior = cv2.bitwise_and(spaces_mask, cv2.bitwise_not(comp_envelope))
        
        # Găsim conturul INTERIORULUI și completăm golurile din pereți
        comp_interior_contours, _ = cv2.findContours(comp_interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if comp_interior_contours:
            comp_largest_contour = max(comp_interior_contours, key=cv2.contourArea)
            comp_contour_mask = np.zeros((h, w), dtype=np.uint8)
            wall_thickness = max(3, int(min(w, h) * 0.003))
            cv2.drawContours(comp_contour_mask, [comp_largest_contour], -1, 255, wall_thickness)
            
            # Găuri în peretele exterior: pixeli pe conturul interiorului care nu sunt încă pereți
            comp_gaps = cv2.bitwise_and(comp_contour_mask, cv2.bitwise_not(walls_mask))
            comp_gaps_area = int(np.count_nonzero(comp_gaps))
            
            if comp_gaps_area > 0:
                result = cv2.bitwise_or(result, comp_gaps)
                print(f"         ✅ Componenta {comp_id}: completat {comp_gaps_area} pixeli de goluri în pereții exteriori")
            
            # Colectăm pentru debug
            all_envelope_masks.append(comp_envelope)
            all_interior_masks.append(comp_interior)
            all_exterior_masks.append(comp_exterior)
            all_gaps.append(comp_gaps)
        else:
            print(f"         ⚠️ Componenta {comp_id}: nu s-au găsit contururi pentru zona interioară. Skip.")
    
    # Imagini de debug
    if steps_dir:
        # 0. Imagine cu envelope / interior / exterior colorate diferit pentru TOATE componentele
        vis_ie = np.zeros((h, w, 3), dtype=np.uint8)
        # Pereți în gri
        vis_ie[walls_mask > 0] = [180, 180, 180]
        
        # Pentru fiecare componentă, colorăm diferit
        colors = [
            (255, 0, 0),    # Albastru
            (0, 255, 0),    # Verde
            (0, 0, 255),    # Roșu
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Galben
        ]
        
        for idx, (comp_envelope, comp_interior, comp_exterior) in enumerate(zip(all_envelope_masks, all_interior_masks, all_exterior_masks)):
            color = colors[idx % len(colors)]
            # Exterior (spațiu liber în afara envelope-ului acestei componente) în roșu închis
            vis_ie[(comp_exterior > 0) & (walls_mask == 0)] = [0, 0, 150]
            # Interior (spațiu liber în interiorul envelope-ului) în verde
            vis_ie[(comp_interior > 0) & (walls_mask == 0)] = [0, 150, 0]
            # Envelope (contur) cu culoare specifică componentei
            envelope_contours, _ = cv2.findContours(comp_envelope, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if envelope_contours:
                cv2.drawContours(vis_ie, envelope_contours, -1, color, 2)
        
        if wall_repair_output_dir:
            cv2.imwrite(str(wall_repair_output_dir / "00_interior_exterior_debug.png"), vis_ie)
            
            # 1. Imagine cu interiorul (maskă) peste pereți pentru toate componentele
            vis_fill = cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
            for comp_interior in all_interior_masks:
                filled_colored = np.zeros_like(vis_fill)
                filled_colored[comp_interior > 0] = [0, 255, 255]  # Galben
                vis_fill = cv2.addWeighted(vis_fill, 0.7, filled_colored, 0.3, 0)
            cv2.imwrite(str(wall_repair_output_dir / "01_house_interior_fill.png"), vis_fill)
            
            # 2. Imagine cu pereții exteriori reparați
            vis_repair = cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
            added_mask = cv2.bitwise_and(result, cv2.bitwise_not(walls_mask))
            added_colored = np.zeros_like(vis_repair)
            added_colored[added_mask > 0] = [0, 255, 0]  # Verde pentru pereți noi
            vis_repair = cv2.addWeighted(vis_repair, 0.7, added_colored, 0.3, 0)
            cv2.imwrite(str(wall_repair_output_dir / "02_house_wall_repair.png"), vis_repair)
            print(f"         💾 Salvat imagini în {wall_repair_output_dir.name}/")
    
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
    """Generează un contur strict de 1px din mască."""
    if np.count_nonzero(mask) == 0:
        return np.zeros_like(mask)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    return cv2.subtract(mask, eroded)
