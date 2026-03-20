# file: engine/cubicasa_detector/raster_processing.py
"""
Module pentru procesarea bazată pe RasterScan.
Conține funcții pentru generarea măștilor, calcul metri per pixel per cameră, etc.
"""

from __future__ import annotations

import json
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from .scale_detection import call_gemini, GEMINI_PROMPT_CROP, call_gemini_zone_labels, call_gemini_blacklist, call_gemini_rooms_per_crop, call_gemini_doors_batch, is_informational_total_result
from .ocr_room_filling import (
    run_ocr_on_zones,
    run_ocr_scan_regions,
    run_ocr_all_zones_fallback,
    GARAGE_SCAN_REGIONS,
    preprocess_image_for_ocr,
    _deduplicate_detections,
)
from .wall_repair import get_strict_1px_outline
from .config import DEBUG

def _log(*args, **kwargs):
    """Afișează doar când HOLZBOT_DEBUG=1 (reduce I/O în producție)."""
    if DEBUG:
        print(*args, **kwargs)

from .wall_gap_detector import (
    find_missing_wall,
    find_missing_wall_raycast,
    find_missing_wall_raycast_3walls,
    draw_plan_closed,
    draw_plan_closed_raycast,
    save_raycast_debug,
    get_red_dot_mask,
    check_garage_flood_touches_edge,
    save_check_garage_image,
    draw_segment_on_mask,
    remove_walls_adjacent_to_region,
    compute_zone_boundary_length_and_area,
)
from .config import MODULE_DIR

# Import notify_ui pentru UI notifications
try:
    from orchestrator import notify_ui
except ImportError:
    # Fallback dacă nu poate importa (pentru testare standalone)
    def notify_ui(stage_tag: str, image_path: Path | str | None = None):
        msg = f">>> UI:STAGE:{stage_tag}"
        if image_path:
            abs_path = Path(image_path).resolve()
            if abs_path.exists():
                msg += f"|IMG:{str(abs_path)}"
        print(msg, flush=True)
        sys.stdout.flush()


# Euristică aspect pentru window vs door (orizontal + vertical)
_ASPECT_WINDOW_MIN = 2.5   # aspect > 2.5 → horizontal window
_ASPECT_BAND_WIDTH, _ASPECT_BAND_HEIGHT = 60, 30  # band dimensions → window
_ASPECT_WINDOW_MAX_VERTICAL = 0.35  # aspect < 0.35 → vertical window (height >> width)


def _is_window_by_aspect(width: float, height: float) -> bool:
    """Unified heuristic: horizontal or vertical window band → window; else door."""
    aspect = width / max(1, height)
    is_horizontal_window = aspect > _ASPECT_WINDOW_MIN or (width > _ASPECT_BAND_WIDTH and height < _ASPECT_BAND_HEIGHT)
    is_vertical_window = aspect < _ASPECT_WINDOW_MAX_VERTICAL or (height > _ASPECT_BAND_WIDTH and width < _ASPECT_BAND_HEIGHT)
    return is_horizontal_window or is_vertical_window


def generate_raster_walls_overlay(
    crop_img: np.ndarray,
    api_walls_mask: np.ndarray,
    output_path: Path
) -> None:
    """
    Generează imagine cu masca de pereți RasterScan peste planul original cropped.
    
    Args:
        crop_img: Imaginea crop-ului din original (BGR)
        api_walls_mask: Masca pereților de la RasterScan (grayscale, 255 = perete)
        output_path: Path unde se salvează imaginea
    """
    # Verificăm și aliniem dimensiunile
    h_crop, w_crop = crop_img.shape[:2]
    h_mask, w_mask = api_walls_mask.shape[:2]
    
    if (h_crop, w_crop) != (h_mask, w_mask):
        _log(f"      ⚠️ Dimensiuni diferite: crop={h_crop}x{w_crop}, mask={h_mask}x{w_mask}. Redimensionăm masca...")
        api_walls_mask = cv2.resize(api_walls_mask, (w_crop, h_crop), interpolation=cv2.INTER_NEAREST)
    
    # Convertim masca la BGR pentru overlay
    walls_bgr = cv2.cvtColor(api_walls_mask, cv2.COLOR_GRAY2BGR)
    
    # Suprapunem cu transparență 70% original + 30% pereți (roșu)
    walls_colored = walls_bgr.copy()
    walls_colored[api_walls_mask > 0] = [0, 0, 255]  # Roșu pentru pereți
    
    overlay = cv2.addWeighted(crop_img, 0.7, walls_colored, 0.3, 0)
    
    cv2.imwrite(str(output_path), overlay)
    _log(f"      📸 Salvat: {output_path.name}")


def generate_raster_rooms_overlay(
    crop_img: np.ndarray,
    api_rooms_img: np.ndarray,
    output_path: Path
) -> None:
    """
    Generează imagine cu camerele RasterScan peste planul original cropped.
    
    Args:
        crop_img: Imaginea crop-ului din original (BGR)
        api_rooms_img: Imaginea cu camerele de la RasterScan (BGR, colorată)
        output_path: Path unde se salvează imaginea
    """
    # Verificăm și aliniem dimensiunile
    h_crop, w_crop = crop_img.shape[:2]
    h_rooms, w_rooms = api_rooms_img.shape[:2]
    
    if (h_crop, w_crop) != (h_rooms, w_rooms):
        _log(f"      ⚠️ Dimensiuni diferite: crop={h_crop}x{w_crop}, rooms={h_rooms}x{w_rooms}. Redimensionăm rooms...")
        api_rooms_img = cv2.resize(api_rooms_img, (w_crop, h_crop), interpolation=cv2.INTER_NEAREST)
    
    # Creăm o mască pentru camere (excludem fundalul alb)
    # Fundalul alb este (255, 255, 255)
    rooms_mask = np.all(api_rooms_img != [255, 255, 255], axis=2).astype(np.uint8) * 255
    
    # Suprapunem cu transparență 70% original + 30% camere
    overlay = cv2.addWeighted(crop_img, 0.7, api_rooms_img, 0.3, 0)
    
    cv2.imwrite(str(output_path), overlay)
    _log(f"      📸 Salvat: {output_path.name}")


def detect_interior_exterior_from_raster(
    api_walls_mask: np.ndarray,
    steps_dir: str = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detectează zonele interior și exterior folosind masca RasterScan.
    
    Args:
        api_walls_mask: Masca pereților de la RasterScan (grayscale, 255 = perete)
        steps_dir: Director pentru salvarea imaginilor de debug
    
    Returns:
        Tuple: (indoor_mask, outdoor_mask) - ambele ca măști binare (255 = zonă, 0 = rest)
    """
    print("   🌊 Analizez zonele folosind masca RasterScan...")
    
    h, w = api_walls_mask.shape[:2]
    
    # Kernel repair
    min_dim = min(h, w)
    rep_k = max(3, int(min_dim * 0.005))
    if rep_k % 2 == 0:
        rep_k += 1
    kernel_repair = cv2.getStructuringElement(cv2.MORPH_RECT, (rep_k, rep_k))
    
    # Dilatăm pereții
    walls_thick = cv2.dilate(api_walls_mask, kernel_repair, iterations=3)
    
    # Flood fill pentru exterior (din colțuri)
    h_pad, w_pad = h + 2, w + 2
    pad_walls = np.zeros((h_pad, w_pad), dtype=np.uint8)
    pad_walls[1:h+1, 1:w+1] = walls_thick
    
    inv_pad_walls = cv2.bitwise_not(pad_walls)
    flood_mask = np.zeros((h_pad+2, w_pad+2), dtype=np.uint8)
    cv2.floodFill(inv_pad_walls, flood_mask, (0, 0), 128, flags=4)  # 4-conectivitate: nu trece prin diagonală
    
    outdoor_mask = (inv_pad_walls[1:h+1, 1:w+1] == 128).astype(np.uint8) * 255

    # Contur exterior al casei + umplere: exterior = 255, interior contur = 0 (fără 30x dilate)
    contours, _ = cv2.findContours(api_walls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outdoor_mask = np.ones_like(api_walls_mask) * 255
    if contours:
        cv2.drawContours(outdoor_mask, contours, -1, 0, thickness=-1)
    
    # Interior = total - exterior - pereți
    total_space = np.ones_like(outdoor_mask) * 255
    occupied = cv2.bitwise_or(outdoor_mask, api_walls_mask)
    indoor_mask = cv2.subtract(total_space, occupied)
    
    # Salvăm măștile
    if steps_dir:
        output_dir = Path(steps_dir) / "raster_processing"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_dir / "outdoor_mask.png"), outdoor_mask)
        cv2.imwrite(str(output_dir / "indoor_mask.png"), indoor_mask)
        
        # Vizualizare
        vis_indoor = np.zeros((h, w, 3), dtype=np.uint8)
        vis_indoor[indoor_mask > 0] = [0, 255, 255]  # Galben pentru interior
        cv2.imwrite(str(output_dir / "indoor_visualization.png"), vis_indoor)
        
        vis_outdoor = np.zeros((h, w, 3), dtype=np.uint8)
        vis_outdoor[outdoor_mask > 0] = [255, 0, 0]  # Roșu pentru exterior
        cv2.imwrite(str(output_dir / "outdoor_visualization.png"), vis_outdoor)
        
        _log(f"      💾 Salvat măști în {output_dir.name}/")
    
    return indoor_mask, outdoor_mask


def calculate_scale_per_room(
    crop_img: np.ndarray,
    indoor_mask: np.ndarray,
    api_walls_mask: np.ndarray,
    steps_dir: str,
    gemini_api_key: str
) -> dict:
    """
    Calculează metri per pixel pentru fiecare cameră în parte.
    
    Pentru fiecare cameră detectată:
    1. Extrage numele camerei prin OCR
    2. Trimite la Gemini pentru a obține suprafața
    3. Calculează metri per pixel bazat pe suprafață și pixeli
    
    Args:
        crop_img: Imaginea crop-ului (BGR)
        indoor_mask: Masca zonei interioare
        api_walls_mask: Masca pereților RasterScan
        steps_dir: Director pentru salvarea rezultatelor
        gemini_api_key: API key pentru Gemini
    
    Returns:
        Dict cu scale-uri per cameră: {room_id: {"room_name": str, "area_m2": float, "m_px": float}}
    """
    print("   🔍 Calculez scala pentru fiecare cameră...")
    
    # Detectăm camerele separate prin connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        indoor_mask, connectivity=8
    )
    
    room_scales = {}
    output_dir = Path(steps_dir) / "raster_processing" / "rooms"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Preprocesăm imaginea pentru OCR
    processed_img = preprocess_image_for_ocr(crop_img)
    
    # Funcție helper pentru procesarea unei camere (pentru paralelizare)
    def process_single_room(room_id):
        area_px = stats[room_id, cv2.CC_STAT_AREA]
        
        # Skip camere prea mici
        if area_px < 1000:
            return None
        
        # Extragem bounding box-ul camerei
        x = stats[room_id, cv2.CC_STAT_LEFT]
        y = stats[room_id, cv2.CC_STAT_TOP]
        w = stats[room_id, cv2.CC_STAT_WIDTH]
        h = stats[room_id, cv2.CC_STAT_HEIGHT]
        
        # Validăm dimensiunile crop-ului
        if w <= 0 or h <= 0:
            return None
        
        # Asigurăm că coordonatele sunt în limitele imaginii
        h_img, w_img = crop_img.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        if w <= 0 or h <= 0:
            return None
        
        # Crop din imagine pentru OCR
        room_crop = crop_img[y:y+h, x:x+w]
        
        # Verificăm că crop-ul nu este gol
        if room_crop.size == 0 or room_crop.shape[0] == 0 or room_crop.shape[1] == 0:
            return None
        
        room_mask_crop = (labels[y:y+h, x:x+w] == room_id).astype(np.uint8) * 255
        
        # Salvăm crop-ul camerei
        room_crop_path = output_dir / f"room_{room_id}_crop.png"
        cv2.imwrite(str(room_crop_path), room_crop)
        
        # Detectăm textul în cameră prin OCR
        text_boxes, all_detections = run_ocr_on_zones(
            processed_img[y:y+h, x:x+w], 
            search_terms=[],
            steps_dir=None,
            grid_rows=1,
            grid_cols=1
        )
        
        detected_text = []
        for detection in all_detections:
            if len(detection) >= 5 and detection[4]:
                detected_text.append(detection[4])
        
        # Trimitem la Gemini pentru a extrage numele și suprafața
        if room_crop_path.exists():
            gemini_result = call_gemini(str(room_crop_path), GEMINI_PROMPT_CROP, gemini_api_key)
            
            if gemini_result:
                room_name = gemini_result.get('room_name', 'Unknown')
                area_m2 = gemini_result.get('area_m2')
                
                if area_m2 and area_m2 > 0:
                    m_px = np.sqrt(area_m2 / area_px)
                    return {
                        "room_id": room_id,
                        "room_name": room_name,
                        "area_m2": float(area_m2),
                        "area_px": int(area_px),
                        "m_px": float(m_px),
                        "detected_text": detected_text
                    }
        return None
    
    # Procesăm camerele în paralel
    room_ids = list(range(1, num_labels))
    max_workers = max(1, min(4, len(room_ids)))  # Max 4 thread-uri pentru Gemini API, minim 1
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_room, room_id): room_id for room_id in room_ids}
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                room_id = result["room_id"]
                room_scales[room_id] = {
                    "room_name": result["room_name"],
                    "area_m2": result["area_m2"],
                    "area_px": result["area_px"],
                    "m_px": result["m_px"],
                    "detected_text": result["detected_text"]
                }
                _log(f"      ✅ Camera {room_id} ({result['room_name']}): {result['area_m2']:.2f} m², {result['m_px']:.9f} m/px")
    
    # Calculează media ponderată a scale-urilor (ponderată după arie)
    if room_scales:
        total_area_m2 = sum(r['area_m2'] for r in room_scales.values())
        weighted_m_px = sum(r['m_px'] * r['area_m2'] for r in room_scales.values()) / total_area_m2
        
        # Salvăm rezultatele
        results_path = output_dir / "room_scales.json"
        with open(results_path, 'w') as f:
            json.dump({
                "room_scales": room_scales,
                "weighted_average_m_px": float(weighted_m_px),
                "total_rooms": len(room_scales)
            }, f, indent=2)
        
        _log(f"      📊 Media ponderată: {weighted_m_px:.9f} m/px ({len(room_scales)} camere)")
        
        return {
            "room_scales": room_scales,
            "weighted_average_m_px": weighted_m_px
        }
    else:
        _log(f"      ⚠️ Nu s-au detectat camere cu suprafață validă")
        return {"room_scales": {}, "weighted_average_m_px": None}


def generate_walls_interior_exterior(
    api_walls_mask: np.ndarray,
    indoor_mask: np.ndarray,
    outdoor_mask: np.ndarray,
    steps_dir: str = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generează pereți interiori și exteriori cu grosime de 1px.
    
    Args:
        api_walls_mask: Masca pereților RasterScan
        indoor_mask: Masca zonei interioare
        outdoor_mask: Masca zonei exterioare
        steps_dir: Director pentru salvarea imaginilor
    
    Returns:
        Tuple: (walls_int_1px, walls_ext_1px) - pereți interiori și exteriori cu 1px grosime
    """
    print("   📐 Generez pereți interiori și exteriori (1px)...")
    
    # Extragem outline-ul strict de 1px
    outline = get_strict_1px_outline(api_walls_mask)
    
    # Dilatăm outdoor_mask pentru a detecta pereții care ating exterior
    kernel_grow = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    touch_zone = cv2.dilate(outdoor_mask, kernel_grow, iterations=2)
    
    # Pereți exteriori = outline care atinge exterior
    walls_ext_1px = cv2.bitwise_and(outline, touch_zone)
    
    # Pereți interiori = outline care nu atinge exterior
    walls_int_1px = cv2.subtract(outline, walls_ext_1px)
    
    # Salvăm imaginile
    if steps_dir:
        output_dir = Path(steps_dir) / "raster_processing" / "walls"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pereți exteriori (roșu)
        vis_ext = np.zeros((*api_walls_mask.shape[:2], 3), dtype=np.uint8)
        vis_ext[walls_ext_1px > 0] = [0, 0, 255]  # Roșu
        
        # Pereți interiori (verde)
        vis_int = np.zeros((*api_walls_mask.shape[:2], 3), dtype=np.uint8)
        vis_int[walls_int_1px > 0] = [0, 255, 0]  # Verde
        
        # Combinat
        vis_combined = vis_ext.copy()
        vis_combined[walls_int_1px > 0] = [0, 255, 0]  # Verde pentru interiori
        
        cv2.imwrite(str(output_dir / "walls_exterior_1px.png"), walls_ext_1px)
        cv2.imwrite(str(output_dir / "walls_interior_1px.png"), walls_int_1px)
        cv2.imwrite(str(output_dir / "walls_exterior_visualization.png"), vis_ext)
        cv2.imwrite(str(output_dir / "walls_interior_visualization.png"), vis_int)
        cv2.imwrite(str(output_dir / "walls_combined_visualization.png"), vis_combined)
        
        _log(f"      💾 Salvat pereți în {output_dir.name}/")
    
    return walls_int_1px, walls_ext_1px


def generate_interior_structure_walls(
    api_walls_mask: np.ndarray,
    walls_int_1px: np.ndarray,
    steps_dir: str = None
) -> np.ndarray:
    """
    Generează structura pereților interiori (fără cei exteriori).
    
    Args:
        api_walls_mask: Masca pereților RasterScan
        walls_int_1px: Pereții interiori cu 1px grosime
        steps_dir: Director pentru salvarea imaginii
    
    Returns:
        Mască cu structura pereților interiori
    """
    print("   🏗️ Generez structura pereților interiori...")
    
    # Structura = pereții interiori (deja calculați)
    structure = walls_int_1px.copy()
    
    if steps_dir:
        output_dir = Path(steps_dir) / "raster_processing" / "walls"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Vizualizare
        vis_structure = np.zeros((*api_walls_mask.shape[:2], 3), dtype=np.uint8)
        vis_structure[structure > 0] = [0, 255, 0]  # Verde
        
        cv2.imwrite(str(output_dir / "walls_interior_structure.png"), structure)
        cv2.imwrite(str(output_dir / "walls_interior_structure_visualization.png"), vis_structure)
        
        _log(f"      💾 Salvat structură în {output_dir.name}/")
    
    return structure


def clean_room_points(points):
    """
    Curăță punctele unei camere pentru a forma un poligon valid.
    Elimină duplicatele și detectează auto-intersecții (loop-uri).
    Aplică algoritmul iterativ până când nu mai există duplicate.
    
    Args:
        points: Lista de liste cu [x, y] (ex: [[100, 200], ...])
    
    Returns:
        Lista de puncte curățate
    """
    if len(points) == 0:
        return []
    
    current = points
    max_iterations = 10  # Prevenim loop-uri infinite
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # PASUL 1: Elimină duplicatele consecutive
        no_dup_consec = [current[0]]
        for i in range(1, len(current)):
            prev = no_dup_consec[-1]
            if current[i][0] != prev[0] or current[i][1] != prev[1]:
                no_dup_consec.append(current[i])
        
        # PASUL 2: Verifică dacă mai există duplicate non-consecutive
        counts = {}
        for p in no_dup_consec:
            key = (p[0], p[1])
            counts[key] = counts.get(key, 0) + 1
        
        # Verifică dacă există vreun punct care apare de mai multe ori
        has_duplicates = any(count > 1 for count in counts.values())
        
        # PASUL 3: Dacă DA, detectăm primul duplicat și tăiem array-ul acolo
        if has_duplicates:
            # Găsim prima repetiție (primul punct care apare a doua oară)
            seen = {}
            found_loop = False
            for i, p in enumerate(no_dup_consec):
                key = (p[0], p[1])
                if key in seen:
                    # Găsit duplicat! Tăiem array-ul aici
                    first_occurrence = seen[key]
                    # Returnăm punctele de la 0 la prima apariție a duplicatului
                    result = no_dup_consec[:first_occurrence + 1]
                    # Adăugăm și punctele de după duplicat până la final
                    result.extend(no_dup_consec[i + 1:])
                    _log(f"         🔧 Detectat loop: punct {key} la poziția {first_occurrence} și {i}, eliminat segment {first_occurrence+1}..{i}")
                    current = result
                    found_loop = True
                    break
                seen[key] = i
            
            if not found_loop:
                # Nu am găsit loop, dar există duplicate - eliminăm toate duplicatele
                seen = set()
                result = []
                for p in no_dup_consec:
                    key = (p[0], p[1])
                    if key not in seen:
                        seen.add(key)
                        result.append(p)
                current = result
        else:
            # Nu mai există duplicate, returnăm rezultatul
            return no_dup_consec
    
    # Dacă am ajuns aici, am depășit numărul maxim de iterații
    _log(f"         ⚠️ Avertisment: algoritm oprit după {max_iterations} iterații")
    return current


def generate_walls_from_room_coordinates(
    original_img: np.ndarray,
    best_config: Dict[str, Any],
    raster_dir: Path,
    steps_dir: str,
    gemini_api_key: str = None,
    initial_walls_mask_1px: np.ndarray = None,
    progress_callback: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """
    Generează pereții pentru camere folosind coordonatele din overlay_on_original.png,
    face flood fill pentru fiecare cameră, calculează metri per pixel și extrage pereții interiori/exteriori.

    Dacă initial_walls_mask_1px este dat (mască 1px la dimensiunea originalului), se sare peste
    construirea segmentelor din JSON; se folosește direct această mască pentru garaj + interior/exterior.

    Args:
        original_img: Imaginea originală (BGR)
        best_config: Configurația brute force (poate fi None când initial_walls_mask_1px e dat)
        raster_dir: Directorul raster
        steps_dir: Directorul pentru steps
        gemini_api_key: Cheia API pentru Gemini (opțional)
        initial_walls_mask_1px: Mască pereți 1px deja aliniată la original (skip construire segmente)

    Returns:
        Dict cu rezultatele: walls_mask, room_scales, indoor_mask, outdoor_mask, walls_int, walls_ext
    """
    def _tick():
        if progress_callback:
            progress_callback()

    _log("\n   🏗️ Generez pereți din coordonatele camerelor...")
    
    output_dir = Path(steps_dir) / "raster_processing" / "walls_from_coords"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    h_orig, w_orig = original_img.shape[:2]
    
    # Cale scurtă: mască 1px furnizată (translation-only), fără construire segmente din JSON
    rooms_from_response = False  # True când camerele vin din response.json (ex.: după edit în UI)
    if initial_walls_mask_1px is not None:
        if initial_walls_mask_1px.shape[:2] != (h_orig, w_orig):
            initial_walls_mask_1px = cv2.resize(
                initial_walls_mask_1px, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST
            )
        _, initial_walls_mask_1px = cv2.threshold(initial_walls_mask_1px, 127, 255, cv2.THRESH_BINARY)
        accepted_wall_segments_mask = initial_walls_mask_1px.copy()
        walls_mask_validated = initial_walls_mask_1px.copy()
        walls_overlay_mask = initial_walls_mask_1px
        mask_for_coverage = initial_walls_mask_1px.copy()
        use_crop_for_coverage = False
        crop_img = None
        crop_info = None
        h_plan, w_plan = h_orig, w_orig
        crop_x, crop_y = 0, 0
        rooms_polygons_original = []
        data = {}
        if (raster_dir / "response.json").exists():
            try:
                with open(raster_dir / "response.json", "r") as f:
                    data = json.load(f).get("data", {})
            except Exception:
                data = {}
        # Pentru procesarea doors (openings) avem nevoie de api_to_original_coords: request space → original (scale + offset translation-only)
        req_w, req_h = w_orig, h_orig
        tx, ty = 0, 0
        request_info_path = raster_dir / "raster_request_info.json"
        if request_info_path.exists():
            try:
                with open(request_info_path, "r", encoding="utf-8") as f:
                    ri = json.load(f)
                req_w = int(ri.get("request_w", w_orig))
                req_h = int(ri.get("request_h", h_orig))
            except Exception:
                pass
        brute_steps_dir = raster_dir / "brute_steps"
        trans_config_path = brute_steps_dir / "translation_only_config.json"
        if trans_config_path.exists():
            try:
                with open(trans_config_path, "r", encoding="utf-8") as f:
                    tc = json.load(f)
                pos = tc.get("position", [0, 0])
                tx, ty = int(pos[0]), int(pos[1])
            except Exception:
                pass
        scale_x = w_orig / max(1, req_w)
        scale_y = h_orig / max(1, req_h)

        def api_to_original_coords(x, y):
            # Request space: aplicăm offset (tx, ty) apoi scale la original (ca la build_aligned_api_walls_1px_original)
            x_m = (x + tx) * scale_x
            y_m = (y + ty) * scale_y
            return int(x_m), int(y_m)

        def _touches_border_skip(mask: np.ndarray) -> bool:
            h_f, w_f = mask.shape[:2]
            if np.any(mask[0, :] > 0) or np.any(mask[h_f - 1, :] > 0):
                return True
            if np.any(mask[:, 0] > 0) or np.any(mask[:, w_f - 1] > 0):
                return True
            return False

        # Încarcă response.json în skip path (pentru a detecta camere editate în UI → refacem pereții din ele)
        response_json_path = raster_dir / "response.json"
        if response_json_path.exists():
            try:
                with open(response_json_path, "r", encoding="utf-8") as f:
                    result_data = json.load(f)
                data = result_data.get("data", result_data)
            except Exception:
                data = {}
        else:
            data = {}

        rooms_polygons = []
        # După edit în UI: preferăm camerele din response.json ca să refacem 01_walls_from_coords din noile forme
        if data.get("rooms"):
            for i, room in enumerate(data["rooms"]):
                pts_raw = []
                for point in room:
                    if isinstance(point, dict) and "x" in point and "y" in point:
                        ox, oy = api_to_original_coords(point["x"], point["y"])
                        ox = max(0, min(w_orig - 1, ox))
                        oy = max(0, min(h_orig - 1, oy))
                        pts_raw.append([ox, oy])
                if len(pts_raw) >= 3:
                    rooms_polygons.append(np.array(pts_raw, dtype=np.int32))
            if rooms_polygons:
                rooms_from_response = True
                _log(f"      📐 [skip path] {len(rooms_polygons)} camere din response.json (după edit) → refacem pereți + 09_interior")
        if not rooms_polygons:
            # Prima dată (fără edit): camere din pereții raster (spațiu liber)
            free_space = (255 - (initial_walls_mask_1px > 0).astype(np.uint8) * 255).astype(np.uint8)
            num_labels, labels, _, _ = cv2.connectedComponentsWithStats(free_space, connectivity=4)
            total_px = h_orig * w_orig
            for label_id in range(1, num_labels):
                comp_mask = (labels == label_id).astype(np.uint8) * 255
                if _touches_border_skip(comp_mask):
                    continue
                area = np.count_nonzero(comp_mask)
                if area < 100:
                    continue
                if total_px > 0 and area >= 0.95 * total_px:
                    continue
                contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    if len(largest) >= 3:
                        rooms_polygons.append(largest)
            if rooms_polygons:
                _log(f"      📐 [skip path] {len(rooms_polygons)} camere din pereții raster (spațiu liber) → room_scales / Gemini")
            else:
                _log(f"      ⚠️ [skip path] Nici camere din raster, nici din response → lista camere goale")

        # Sărim direct la crearea folderului garage (același cod ca mai jos)
        _skip_to_garage = True
        _input_is_1px = True  # Folosim mască 1px direct, fără dilatare grosă
        walls_overlay_path = output_dir / "walls_overlay_on_crop.png"
        # ✅ [skip path] Salvăm rooms.png ca orchestratorul (_check_raster_complete) să găsească fișierul
        _room_colors_skip = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 0), (128, 0, 128)]
        rooms_img_skip = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
        rooms_img_skip.fill(255)
        for idx, rp in enumerate(rooms_polygons):
            col = _room_colors_skip[idx % len(_room_colors_skip)]
            cv2.fillPoly(rooms_img_skip, [rp], col)
            cv2.polylines(rooms_img_skip, [rp], True, (0, 0, 0), 2)
        cv2.imwrite(str(raster_dir / "rooms.png"), rooms_img_skip)
        _log(f"      💾 Salvat: rooms.png ({len(rooms_polygons)} camere, skip path)")
        # 09_interior.png din aceleași poligoane (spațiu liber raster) ca ordinea să coincidă cu Gemini și editorul
        if rooms_polygons:
            interior_img_skip = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
            for rp in rooms_polygons:
                cv2.fillPoly(interior_img_skip, [rp], [0, 165, 255])
            cv2.imwrite(str(output_dir / "09_interior.png"), interior_img_skip)
            _log(f"      💾 Salvat: 09_interior.png (skip path, aceleași camere ca pentru Gemini)")
        # Poligoanele sunt cele de la rooms (boss); după modificări le folosim pentru walls_from_coords
        if rooms_from_response and len(rooms_polygons) > 0:
            # Aceleași poligoane ca în rooms.png (sursă: rooms boss)
            walls_from_rooms = np.zeros((h_orig, w_orig), dtype=np.uint8)
            for rp in rooms_polygons:
                cv2.polylines(walls_from_rooms, [rp], isClosed=True, color=255, thickness=2)
            _, walls_from_rooms = cv2.threshold(walls_from_rooms, 127, 255, cv2.THRESH_BINARY)
            # Unim pereții la distanță ≤ 1% (din dim. imagine): dilatare 1% apoi skeleton → un singur perete 1px
            k = max(3, int(min(h_orig, w_orig) * 0.01))
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            walls_from_rooms = cv2.dilate(walls_from_rooms, kernel)
            from skimage.morphology import skeletonize
            walls_from_rooms = (skeletonize((walls_from_rooms > 0).astype(bool)).astype(np.uint8)) * 255
            accepted_wall_segments_mask = walls_from_rooms.copy()
            walls_mask_validated = walls_from_rooms.copy()
            walls_overlay_mask = walls_from_rooms.copy()
            mask_for_coverage = walls_from_rooms.copy()
            _log(f"      🔄 Mască pereți (01_walls_from_coords) din rooms boss, pereți uniți (1px)")
            segments_path_skip = output_dir / "01_walls_from_coords.png"
            cv2.imwrite(str(segments_path_skip), walls_from_rooms)
            _log(f"      💾 Salvat: 01_walls_from_coords.png (skip path, din camere editate)")
    else:
        _skip_to_garage = False
        _input_is_1px = False
        walls_overlay_path = output_dir / "walls_overlay_on_crop.png"

    if not _skip_to_garage:
        rooms_from_response = False

    if not _skip_to_garage and not walls_overlay_path.exists():
        api_walls_mask_path = raster_dir / "api_walls_mask.png"
        if api_walls_mask_path.exists() and original_img is not None:
            api_walls_mask = cv2.imread(str(api_walls_mask_path), cv2.IMREAD_GRAYSCALE)
            if api_walls_mask is not None:
                h_orig, w_orig = original_img.shape[:2]
                h_mask, w_mask = api_walls_mask.shape[:2]
                if (h_mask, w_mask) != (h_orig, w_orig):
                    api_walls_mask = cv2.resize(api_walls_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                generate_raster_walls_overlay(original_img, api_walls_mask, walls_overlay_path)
                _log(f"      ✅ Generat walls_overlay_on_crop.png pentru validare")
    
    if not _skip_to_garage:
        # 1. Încărcăm response.json pentru coordonatele camerelor
        response_json_path = raster_dir / "response.json"
        if not response_json_path.exists():
            _log(f"      ⚠️ response.json nu există")
            return None
        
        with open(response_json_path, 'r') as f:
            result_data = json.load(f)
        
        data = result_data.get('data', result_data)
    
        # Factor request → mask: coordonatele din JSON pot fi în REQUEST space SAU în RESPONSE space.
        # Când API returnează imagine la aceleași dimensiuni ca request-ul, JSON e în request space.
        # Când API returnează alt sizing (ex. alt aspect ratio), JSON e în response space (dimensiunile imaginii returnate).
        request_info_path = raster_dir / "raster_request_info.json"
        req_w = req_h = mask_w = mask_h = scale_factor = None
        if request_info_path.exists():
            try:
                with open(request_info_path, 'r') as f:
                    ri = json.load(f)
                req_w = ri.get('request_w')
                req_h = ri.get('request_h')
                mask_w = ri.get('mask_w')
                mask_h = ri.get('mask_h')
                scale_factor = ri.get('scale_factor', 1.0)
            except Exception:
                pass
        if mask_w is None or mask_h is None:
            mask_w = best_config.get('mask_w')
            mask_h = best_config.get('mask_h')
        if (mask_w is None or mask_h is None) and (raster_dir / "api_walls_mask.png").exists():
            try:
                m = cv2.imread(str(raster_dir / "api_walls_mask.png"), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    mask_h, mask_w = m.shape[:2]
            except Exception:
                pass
        if not req_w or not req_h:
            req_w, req_h = mask_w, mask_h
        scale_up = (1.0 / scale_factor) if (scale_factor and scale_factor > 0 and scale_factor < 1.0) else 1.0
        # Detectăm dacă masca este request * scale_up (API a returnat aceleași dimensiuni ca request-ul)
        expected_mask_w = req_w * scale_up if req_w else 0
        expected_mask_h = req_h * scale_up if req_h else 0
        tol = 0.08
        request_matches_response = (
            mask_w and mask_h and expected_mask_w and expected_mask_h
            and abs(mask_w - expected_mask_w) <= max(2, expected_mask_w * tol)
            and abs(mask_h - expected_mask_h) <= max(2, expected_mask_h * tol)
        )
        if request_matches_response:
            r2m_x = (mask_w / req_w) if (req_w and req_w > 0) else 1.0
            r2m_y = (mask_h / req_h) if (req_h and req_h > 0) else 1.0
            _log(f"      📐 Coordonate JSON: request space (request {req_w}x{req_h} → mask {mask_w}x{mask_h})")
        else:
            # API a returnat alt sizing: JSON e în response space (coord. în imaginea returnată, înainte de size-up)
            # response -> mask: x_m = x_json * scale_up
            r2m_x = scale_up
            r2m_y = scale_up
            _log(f"      📐 Coordonate JSON: response space (mask {mask_w}x{mask_h}, scale_up={scale_up:.3f})")
    
        # Funcție de transformare coordonate API (request space) → Original.
        # Pas 1: request → mask; Pas 2: mask → original (brute force: position + scale).
        def api_to_original_coords(x, y):
            x_m = x * r2m_x
            y_m = y * r2m_y
            if best_config['direction'] == 'api_to_orig':
                orig_x = x_m * best_config['scale'] + best_config['position'][0]
                orig_y = y_m * best_config['scale'] + best_config['position'][1]
                return int(orig_x), int(orig_y)
            else:
                orig_x = (x_m - best_config['position'][0]) / best_config['scale']
                orig_y = (y_m - best_config['position'][1]) / best_config['scale']
                return int(orig_x), int(orig_y)
    
        h_orig, w_orig = original_img.shape[:2]
    
        # ✅ Pentru coverage: folosim crop + mască-on-crop (exact ca walls_overlay_on_crop) când există crop.
        # Altfel folosim planul full (original) + mască resize la plan.
        use_crop_for_coverage = False
        crop_img = None
        crop_info = None
        h_plan = h_orig
        w_plan = w_orig
        crop_x = 0
        crop_y = 0
        crop_path = raster_dir / "00_original_crop.png"
        crop_info_path = raster_dir / "crop_info.json"
        if crop_path.exists() and crop_info_path.exists():
            crop_img = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
            try:
                with open(crop_info_path, "r", encoding="utf-8") as f:
                    crop_info = json.load(f)
            except Exception:
                crop_info = None
            if crop_img is not None and crop_info and "x" in crop_info and "y" in crop_info and "width" in crop_info and "height" in crop_info:
                use_crop_for_coverage = True
                h_plan = int(crop_info["height"])
                w_plan = int(crop_info["width"])
                crop_x = int(crop_info["x"])
                crop_y = int(crop_info["y"])
                _log(f"      📐 COVERAGE folosește CROP ({w_plan}x{h_plan}, offset {crop_x},{crop_y}), mască ca walls_overlay_on_crop")
    
        # ✅ Colectăm coordonatele camerelor din JSON (doar pentru a găsi centrele camerelor în regenerare)
        # NU generăm rooms_mask sau rooms_polygons aici - vor fi generate DUPĂ validarea pereților
        rooms_polygons_original = []  # Păstrăm doar pentru a găsi centrele camerelor
    
        # ✅ Colectăm toate bounding boxes-urile pentru uși, geamuri etc.
        bbox_rects: list[list[int]] = []
        for element_type in ['doors', 'windows', 'openings']:
            if element_type in data and data[element_type]:
                for element in data[element_type]:
                    if 'bbox' in element and len(element['bbox']) == 4:
                        bbox = element['bbox']
                        # Transformăm coordonatele bbox la coordonatele originale
                        x1, y1 = api_to_original_coords(bbox[0], bbox[1])
                        x2, y2 = api_to_original_coords(bbox[2], bbox[3])
                        # Validăm coordonatele
                        x1 = max(0, min(w_orig - 1, x1))
                        y1 = max(0, min(h_orig - 1, y1))
                        x2 = max(0, min(w_orig - 1, x2))
                        y2 = max(0, min(h_orig - 1, y2))
                        # Normalizăm astfel încât x1 < x2, y1 < y2
                        if x2 < x1:
                            x1, x2 = x2, x1
                        if y2 < y1:
                            y1, y2 = y2, y1
                        bbox_rects.append([x1, y1, x2, y2])
    
        if 'rooms' in data and data['rooms']:
            _log(f"      📐 Colectez coordonatele pentru {len(data['rooms'])} camere (pentru regenerare ulterioară)...")
            if bbox_rects:
                _log(f"      🚪 Am găsit {len(bbox_rects)} bounding boxes pentru uși/geamuri")
        
            for i, room in enumerate(data['rooms']):
                # Extragem punctele camerei din JSON (doar pentru a găsi centrul camerei)
                pts_raw = []
                for point in room:
                    if 'x' in point and 'y' in point:
                        ox, oy = api_to_original_coords(point['x'], point['y'])
                        # Validăm că coordonatele sunt în limitele imaginii
                        ox = max(0, min(w_orig - 1, ox))
                        oy = max(0, min(h_orig - 1, oy))
                        pts_raw.append([ox, oy])
            
                if len(pts_raw) < 3:
                    _log(f"         ⚠️ Camera {i}: prea puține puncte ({len(pts_raw)})")
                    continue
            
                # Convertim la numpy array (doar pentru a calcula centrul)
                pts_np = np.array(pts_raw, dtype=np.int32)
                rooms_polygons_original.append(pts_np)
                _log(f"         ✅ Camera {i}: {len(pts_raw)} puncte (coordonate colectate)")
    
        # ✅ PASUL 2: Generăm walls.png din rooms_mask (rooms.png va fi generat DUPĂ validarea pereților)
        # Culori pentru camere (BGR format) - folosite mai târziu pentru rooms.png
        room_colors = [
            (200, 230, 200),  # Verde deschis
            (200, 200, 230),  # Albastru deschis
            (230, 200, 200),  # Roșu deschis
            (230, 230, 200),  # Galben deschis
            (200, 230, 230),  # Cyan deschis
            (230, 200, 230),  # Magenta deschis
            (220, 220, 220),  # Gri deschis
            (210, 230, 210),  # Verde mentă
        ]
    
        # ⚠️ rooms.png va fi generat DUPĂ validarea pereților și regenerarea camerelor (folosind pereții validați)
    
        # ✅ PASUL 2: Încărcăm api_walls_mask.png (dacă e deja la original, o folosim direct; altfel o transformăm cu brute force)
        api_walls_mask_path = raster_dir / "api_walls_mask.png"
        walls_overlay_mask = None
        if api_walls_mask_path.exists():
            api_walls_mask = cv2.imread(str(api_walls_mask_path), cv2.IMREAD_GRAYSCALE)
            if api_walls_mask is not None:
                h_api, w_api = api_walls_mask.shape[:2]
                if (h_api, w_api) == (h_orig, w_orig):
                    # Masca e deja la dimensiunile originale (size-up făcut înainte de construirea mastii)
                    walls_overlay_mask = api_walls_mask.copy()
                    _, walls_overlay_mask = cv2.threshold(walls_overlay_mask, 127, 255, cv2.THRESH_BINARY)
                    _log(f"      ✅ Încărcat api_walls_mask.png (deja la original {w_orig}x{h_orig}, fără transformare)")
                elif best_config:
                    # Masca e la dimensiunea request → transformăm cu brute force
                    scale = best_config['scale']
                    x_pos, y_pos = best_config['position']
                    direction = best_config['direction']
                    if direction == 'api_to_orig':
                        M = np.array([[scale, 0, x_pos], [0, scale, y_pos]], dtype=np.float32)
                    else:
                        M = np.array([[scale, 0, x_pos], [0, scale, y_pos]], dtype=np.float32)
                    walls_overlay_mask = cv2.warpAffine(
                        api_walls_mask, M, (w_orig, h_orig),
                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
                    )
                    _, walls_overlay_mask = cv2.threshold(walls_overlay_mask, 127, 255, cv2.THRESH_BINARY)
                    _log(f"      ✅ Încărcat api_walls_mask.png și transformat la coordonatele originale folosind transformarea brută ({w_api}x{h_api} → {w_orig}x{h_orig}, scale={scale:.3f}x, pos=({x_pos}, {y_pos}))")
                else:
                    _log(f"      ⚠️ api_walls_mask.png are {w_api}x{h_api}, best_config lipsește – nu pot transforma")
            else:
                _log(f"      ⚠️ Nu am putut încărca api_walls_mask.png")
        else:
            _log(f"      ⚠️ api_walls_mask.png nu există în {raster_dir}")
    
        # ✅ Masca pentru coverage: api_walls_mask (returnată de Raster), redimensionată la plan (sau la CROP când există).
        #    Când există crop, aplicăm masca exact ca walls_overlay_on_crop (crop + mask-on-crop).
        #    Overlap = câți pixeli din linia galbenă sunt pe mască.
        mask_for_coverage = None
        mask_source_name = ""
        api_walls_mask_path = raster_dir / "api_walls_mask.png"
        if api_walls_mask_path.exists():
            api_mask = cv2.imread(str(api_walls_mask_path), cv2.IMREAD_GRAYSCALE)
            if api_mask is not None:
                h_m, w_m = api_mask.shape[:2]
                if (h_m, w_m) != (h_plan, w_plan):
                    mask_for_coverage = cv2.resize(api_mask, (w_plan, h_plan), interpolation=cv2.INTER_NEAREST)
                else:
                    mask_for_coverage = api_mask.copy()
                _, mask_for_coverage = cv2.threshold(mask_for_coverage, 127, 255, cv2.THRESH_BINARY)
                if use_crop_for_coverage:
                    mask_source_name = "api_walls_mask.png (resize la crop, ca walls_overlay_on_crop)"
                else:
                    mask_source_name = "api_walls_mask.png (resize la plan)"
        if mask_for_coverage is None:
            _log(f"      ❌ Nu am mască (api_walls_mask) pentru coverage. Nu pot continua.")
            return None

        # ✅ PASUL 3: Construim walls_mask validată trăgând liniile din JSON peste mască
        # și validând cu >= 40% coverage între linia segmentului (1px) și mască. Aceasta va fi masca finală
        # folosită pentru generarea camerelor și pentru toate fișierele derivate (walls_thick, flood fill etc.).
        _log(f"      🧱 Construiesc walls_mask validată: linia între cele 2 puncte vs mască, valid >= 40% coverage...")
    
        # ⚠️ Conform cerinței: pentru pașii următori folosim DOAR segmentele acceptate.
        # Deci masca de bază pornește GOALĂ, iar noi desenăm doar segmentele validate.
        walls_mask_validated = np.zeros((h_orig, w_orig), dtype=np.uint8)

        # ✅ IMPORTANT: 01_walls_from_coords.png NU trebuie să fie o mască completă de pereți.
        # Conform cerinței: aici salvăm DOAR segmentele de pereți ACCEPTATE (după validarea >= 40% pe segment).
        accepted_wall_segments_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    
        # Director pentru imagini de debug (doar când DEBUG)
        if DEBUG:
            debug_walls_dir = output_dir / "wall_segments_debug"
            debug_walls_dir.mkdir(exist_ok=True)
        else:
            debug_walls_dir = None
    
        valid_segments = 0
        invalid_segments = 0
    
        # ✅ Adăugăm segmentele de pereți din JSON-ul RasterScan
        # Verificăm fiecare segment: câți pixeli din linia galbenă (segmente paralele) se află pe mască. Valid dacă >= 40%.
        plan_label = "crop" if use_crop_for_coverage else "plan"
        _log(f"      📐 COVERAGE: mască = {mask_source_name}, overlap = pixeli linie galbenă pe mască, dim. {plan_label} {w_plan}x{h_plan}")

        if 'walls' in data and data['walls']:
            _log(f"      🧱 Verific {len(data['walls'])} segmente de pereți din JSON...")
        
            for idx, wall in enumerate(data['walls']):
                pos = wall.get('position')
                if not pos or len(pos) != 2:
                    continue
                try:
                    x1_api, y1_api = pos[0]
                    x2_api, y2_api = pos[1]
                except Exception:
                    continue

                # Transformăm coordonatele pereților din sistemul Raster în coordonatele originalului
                x1, y1 = api_to_original_coords(x1_api, y1_api)
                x2, y2 = api_to_original_coords(x2_api, y2_api)

                # Clamp în interiorul imaginii (original)
                x1 = max(0, min(w_orig - 1, x1))
                y1 = max(0, min(h_orig - 1, y1))
                x2 = max(0, min(w_orig - 1, x2))
                y2 = max(0, min(h_orig - 1, y2))

                # Ignorăm segmentele degenerate în original
                if x1 == x2 and y1 == y2:
                    continue

                # ✅ Pentru coverage: folosim crop (ca walls_overlay_on_crop) când există; altfel plan full.
                if use_crop_for_coverage:
                    x1u = x1 - crop_x
                    y1u = y1 - crop_y
                    x2u = x2 - crop_x
                    y2u = y2 - crop_y
                    x1u = max(0, min(w_plan - 1, x1u))
                    y1u = max(0, min(h_plan - 1, y1u))
                    x2u = max(0, min(w_plan - 1, x2u))
                    y2u = max(0, min(h_plan - 1, y2u))
                    if x1u == x2u and y1u == y2u:
                        continue
                else:
                    x1u, y1u, x2u, y2u = x1, y1, x2, y2
            
                # ✅ Calculăm grosimea liniei: 2.5% din lățimea planului folosit pentru coverage (crop sau plan)
                line_thickness = max(1, int(w_plan * 0.025))
            
                # ✅ Coverage: câți pixeli din linia galbenă (paralele 1px) se află pe mască. overlap = linie ∩ mască, coverage = overlap/total. Valid dacă best >= 40%.
                should_draw = False
                coverage_percent = 0.0
                best_coverage = 0.0
                best_overlap = 0
                best_total = 0
                dx = x2u - x1u
                dy = y2u - y1u
                line_length = np.sqrt(dx * dx + dy * dy)
                # Număr de linii galbene paralele = grosimea segmentului în pixeli (2.5% din plan): exact line_thickness linii
                half_count = max(0, (line_thickness - 1) // 2)
                parallel_offsets = list(range(-half_count, half_count + 1))
                off_axis = 'y' if abs(dx) >= abs(dy) else 'x'

                segment_mask = np.zeros((h_plan, w_plan), dtype=np.uint8)
                if line_length > 0:
                    best_coverage = 0.0
                    for k in parallel_offsets:
                        if off_axis == 'y':
                            x1_k, y1_k = x1u, y1u + k
                            x2_k, y2_k = x2u, y2u + k
                        else:
                            x1_k, y1_k = x1u + k, y1u
                            x2_k, y2_k = x2u + k, y2u
                        x1_k = max(0, min(w_plan - 1, x1_k))
                        y1_k = max(0, min(h_plan - 1, y1_k))
                        x2_k = max(0, min(w_plan - 1, x2_k))
                        y2_k = max(0, min(h_plan - 1, y2_k))
                        seg_mask = np.zeros((h_plan, w_plan), dtype=np.uint8)
                        cv2.line(seg_mask, (x1_k, y1_k), (x2_k, y2_k), 255, 1)
                        segment_mask = np.maximum(segment_mask, seg_mask)
                        total_px = np.sum(seg_mask > 0)
                        if total_px > 0:
                            overlap = int(np.sum((mask_for_coverage > 0) & (seg_mask > 0)))
                            if not (0 <= overlap <= total_px):
                                _log(f"         ⚠️ Segment {idx} k={k}: overlap={overlap} total={total_px} (skip)")
                                continue
                            cov = (overlap / total_px) * 100.0
                            if cov > best_coverage:
                                best_coverage = cov
                                best_overlap = int(overlap)
                                best_total = int(total_px)
                    coverage_percent = best_coverage
                    should_draw = best_coverage >= 40.0
            
                # Generez imagine de debug: crop (ca walls_overlay_on_crop) când folosim crop, altfel original
                debug_base = crop_img if use_crop_for_coverage else original_img
                debug_img = debug_base.copy()
                hu, wu = debug_base.shape[:2]
                color = (0, 255, 0) if should_draw else (0, 0, 255)
                cv2.line(debug_img, (x1u, y1u), (x2u, y2u), color, line_thickness)
                cv2.circle(debug_img, (x1u, y1u), 5, (255, 0, 0), -1)
                cv2.circle(debug_img, (x2u, y2u), 5, (255, 0, 0), -1)
            
                # ✅ Highlight: linii galbene paralele pe segment (în spațiul folosit pentru coverage)
                if line_length > 0:
                    for k in parallel_offsets:
                        if off_axis == 'y':
                            x1_k, y1_k = x1u, y1u + k
                            x2_k, y2_k = x2u, y2u + k
                        else:
                            x1_k, y1_k = x1u + k, y1u
                            x2_k, y2_k = x2u + k, y2u
                        x1_k = max(0, min(wu - 1, x1_k))
                        y1_k = max(0, min(hu - 1, y1_k))
                        x2_k = max(0, min(wu - 1, x2_k))
                        y2_k = max(0, min(hu - 1, y2_k))
                        cv2.line(debug_img, (x1_k, y1_k), (x2_k, y2_k), (0, 255, 255), 1)
            
                walls_colored_debug = cv2.cvtColor(mask_for_coverage, cv2.COLOR_GRAY2BGR)
                walls_colored_debug[mask_for_coverage > 0] = [0, 0, 255]
                debug_img = cv2.addWeighted(debug_img, 0.7, walls_colored_debug, 0.3, 0)
            
                status = "✅ VALID" if should_draw else "❌ INVALID"
                n_parallel = len(parallel_offsets)
                text = f"Segment {idx}: best of {n_parallel} parallel lines={coverage_percent:.1f}% (>=40%) - {status}"
                cv2.putText(debug_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if best_total > 0:
                    cv2.putText(debug_img, f"overlap {best_overlap}/{best_total} px (linie pe mască)", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(debug_img, f"mask: {mask_source_name}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                if debug_walls_dir is not None:
                    cv2.imwrite(str(debug_walls_dir / f"wall_segment_{idx:03d}.png"), debug_img)

                # ✅ Imagine suplimentară: mască (mov), segment (portocaliu), overlap (galben), fundal negru – în spațiul coverage (crop sau plan)
                if debug_walls_dir is not None and line_length > 0:
                    overlay_img = np.zeros((h_plan, w_plan, 3), dtype=np.uint8)
                    overlay_img[mask_for_coverage > 0] = [128, 0, 128]
                    overlay_img[segment_mask > 0] = [0, 165, 255]
                    overlap_px = (segment_mask > 0) & (mask_for_coverage > 0)
                    overlay_img[overlap_px] = [0, 255, 255]
                    cv2.imwrite(str(debug_walls_dir / f"wall_segment_{idx:03d}_mask_overlay.png"), overlay_img)

                if best_total > 0:
                    _log(f"         Segment {idx}: overlap {best_overlap}/{best_total} = {coverage_percent:.1f}% -> {status}")
            
                # ✅ Trasez linia peste walls_mask_validated DOAR dacă trece validarea (>= 40% coverage pe segment)
                if should_draw:
                    cv2.line(walls_mask_validated, (x1, y1), (x2, y2), 255, line_thickness)
                    # ✅ Desenăm cu grosime 2px ca la colțuri să fie suprapunere (fără gol); apoi skeleton → 1px
                    cv2.line(accepted_wall_segments_mask, (x1, y1), (x2, y2), 255, 2)
                    valid_segments += 1
                else:
                    invalid_segments += 1
        
            _log(f"      ✅ Segmente valide: {valid_segments} / {len(data['walls'])}")
            _log(f"      ❌ Segmente invalide (coverage < 40% pe segment): {invalid_segments}")
            if DEBUG and debug_walls_dir is not None:
                _log(f"      📸 Imagini de debug salvate în: {debug_walls_dir.name}/")

        # ✅ Reducere la 1px: segmentele sunt desenate cu grosime 2px (colțurile se suprapun, fără gol).
        # O singură skeletonizare dă linii 1px continue, inclusiv la colțuri (fără dilatare prealabilă care crea găuri).
        if valid_segments > 0 and np.any(accepted_wall_segments_mask > 0):
            from skimage.morphology import skeletonize
            binary = (accepted_wall_segments_mask > 0).astype(np.uint8)
            skel = skeletonize(binary.astype(bool))
            accepted_wall_segments_mask = (skel.astype(np.uint8)) * 255
            _log(f"      🔗 Skeleton 1px (din 2px): linii fără goluri la colțuri sau diagonale")
    
        # ✅ De aici încolo, "walls_overlay_mask" va însemna DOAR segmentele acceptate (thin 1px),
        # nu masca completă de la API.
        walls_overlay_mask = walls_mask_validated.copy()
    
        # Generez walls.png - imagine cu segmentele validate (vizualizare)
        walls_img = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
        walls_img.fill(255)  # Fundal alb
        walls_colored = cv2.cvtColor(walls_overlay_mask, cv2.COLOR_GRAY2BGR)
        walls_colored[walls_overlay_mask > 0] = [0, 0, 0]  # Negru pentru pereți
        walls_img = cv2.addWeighted(walls_img, 0.0, walls_colored, 1.0, 0)
    
        # Salvăm walls.png
        walls_output_path = raster_dir / "walls.png"
        cv2.imwrite(str(walls_output_path), walls_img)
        _log(f"      💾 Salvat: walls.png (pereți validați cu >= 40% coverage pe segment)")
    
        # ⚠️ NU salvăm 01_walls_from_coords.png aici - va fi salvat DUPĂ ce walls_overlay_mask este disponibilă
        # pentru a folosi exact aceiași pereți perfecți ca în room_x_debug.png
    
        # ✅ REGENEREZ camerele pe baza pereților validați (>= 40% coverage pe segment)
        # Folosim EXACT aceeași mască validată (walls_overlay_mask) pentru a genera camerele
        _log(f"      🔄 Regenerez camerele pe baza pereților validați (mască validată >= 40% coverage pe segment)...")
        rooms_mask_validated = np.zeros((h_orig, w_orig), dtype=np.uint8)
        rooms_polygons_validated = []
    
        # Inițializăm rooms_polygons și rooms_mask ca liste goale (vor fi populate după regenerare)
        # Acestea vor fi folosite mai târziu în cod, deci trebuie să fie definite
        rooms_polygons = []
        rooms_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    
        # ✅ Pentru flood fill și separări avem nevoie de o "barieră" (pereți cu grosime).
        # Conform cerinței, baza rămâne DOAR segmentele acceptate; grosimea este derivată din ele.
        if walls_overlay_mask is None:
            _log(f"      ❌ walls_overlay_mask nu este disponibilă. Nu pot continua regenerarea camerelor.")
            return None

        # Grosime pereți: 0.005% din lățime (min 5px) - derivat din segmente
        wall_thickness = max(5, int(w_orig * 0.00005))
        _log(f"      📏 Grosime pereți (pentru barrier): {wall_thickness}px (0.005% din {w_orig}px)")

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wall_thickness, wall_thickness))
        walls_barrier = cv2.dilate(walls_overlay_mask, kernel, iterations=1)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        walls_barrier = cv2.morphologyEx(walls_barrier, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # Creăm o mască inversă pentru flood fill (spațiile libere sunt 255, pereții sunt 0)
        flood_fill_base = (255 - walls_barrier).astype(np.uint8)
    
        if 'rooms' in data and data['rooms'] and rooms_polygons_original:
            for i, room in enumerate(data['rooms']):
                # Găsim un punct din interiorul camerei originale (din JSON)
                if i >= len(rooms_polygons_original):
                    continue
                room_poly_original = rooms_polygons_original[i]
            
                # Calculăm centrul camerei originale
                M = cv2.moments(room_poly_original)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    # Fallback: primul punct
                    cx, cy = int(room_poly_original[0][0]), int(room_poly_original[0][1])
            
                # Clamp în interiorul imaginii
                cx = max(1, min(w_orig - 2, cx))
                cy = max(1, min(h_orig - 2, cy))
            
                # Verificăm dacă punctul este într-un spațiu liber (nu pe perete)
                # Folosim bariera (pereți îngroșați) derivată din segmentele acceptate
                if walls_barrier[cy, cx] > 0:
                    # Căutăm un punct aproape care este spațiu liber
                    found = False
                    for radius in range(1, 50):
                        for angle in range(0, 360, 30):
                            test_x = int(cx + radius * np.cos(np.radians(angle)))
                            test_y = int(cy + radius * np.sin(np.radians(angle)))
                            if 0 <= test_x < w_orig and 0 <= test_y < h_orig:
                                if walls_barrier[test_y, test_x] == 0:
                                    cx, cy = test_x, test_y
                                    found = True
                                    break
                        if found:
                            break
                    if not found:
                        continue
            
                # Facem flood fill din acest punct, limitat de pereții validați
                flood_mask = np.zeros((h_orig + 2, w_orig + 2), dtype=np.uint8)
                seed_value = 128 + i  # Valoare unică pentru fiecare cameră
                cv2.floodFill(flood_fill_base, flood_mask, (cx, cy), seed_value, 
                             loDiff=(0,), upDiff=(0,), flags=4)
            
                # Extragem zona umplută pentru această cameră
                room_mask_validated = (flood_fill_base == seed_value).astype(np.uint8) * 255
            
                room_area = np.count_nonzero(room_mask_validated)
                total_image_area = h_orig * w_orig
            
                if room_area < 100:  # Prea mică
                    continue
            
                # ✅ Ignorăm camerele care acoperă toată imaginea (sau aproape toată)
                room_coverage_ratio = room_area / total_image_area
                if room_coverage_ratio >= 0.95:  # Acoperă >= 95% din imagine
                    _log(f"      ⚠️ Ignorat camera {i}: acoperă {room_coverage_ratio:.1%} din imagine (prea mare)")
                    continue
            
                # Găsim conturul camerei validate
                contours_room, _ = cv2.findContours(room_mask_validated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours_room:
                    # Folosim cel mai mare contur
                    largest_contour = max(contours_room, key=cv2.contourArea)
                    if len(largest_contour) >= 3:
                        rooms_polygons_validated.append(largest_contour)
                        # Adăugăm la masca de camere validate
                        cv2.fillPoly(rooms_mask_validated, [largest_contour], 255)
    
        # ✅ Dacă nu există camere în JSON, generăm automat camere din zonele închise din 02_walls_thick.png
        if len(rooms_polygons_validated) == 0:
            _log(f"      🔍 Nu există camere în JSON. Generez automat camere din zonele închise din pereți...")
        
            # Folosim walls_barrier pentru a identifica zonele închise (spații libere delimitate de pereți)
            # Creăm o mască inversă: spațiile libere sunt 255, pereții sunt 0
            free_space_mask = (255 - walls_barrier).astype(np.uint8)
        
            # Identificăm toate componentele conectate de spații libere (4-conectivitate: camere doar diagonală = separate)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(free_space_mask, connectivity=4)
        
            # Colțurile imaginii pentru verificare
            corners = [
                (0, 0),  # Top-left
                (w_orig - 1, 0),  # Top-right
                (0, h_orig - 1),  # Bottom-left
                (w_orig - 1, h_orig - 1)  # Bottom-right
            ]
        
            # Mască pentru a evita suprapunerea camerelor
            used_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        
            # Procesăm fiecare componentă conectată (skip label 0 = fundal/pereți)
            for label_id in range(1, num_labels):
                # Obținem masca pentru această componentă
                component_mask = (labels == label_id).astype(np.uint8) * 255
            
                # Verificăm dacă componenta atinge colțurile
                touches_corner = False
                for cx, cy in corners:
                    if component_mask[cy, cx] > 0:
                        touches_corner = True
                        break
            
                if touches_corner:
                    continue  # Skip zonele care ating colțurile
            
                # Verificăm dacă componenta se suprapune cu o cameră deja procesată
                overlap = cv2.bitwise_and(component_mask, used_mask)
                if np.count_nonzero(overlap) > 0:
                    continue  # Skip zonele care se suprapun
            
                # Calculăm aria componentei
                component_area = np.count_nonzero(component_mask)
                total_image_area = h_orig * w_orig
            
                # Skip componente prea mici sau prea mari
                if component_area < 100:  # Prea mică
                    continue
            
                room_coverage_ratio = component_area / total_image_area
                if room_coverage_ratio >= 0.95:  # Acoperă >= 95% din imagine
                    continue
            
                # Găsim conturul componentei
                contours_component, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours_component:
                    # Folosim cel mai mare contur
                    largest_contour = max(contours_component, key=cv2.contourArea)
                    if len(largest_contour) >= 3:
                        rooms_polygons_validated.append(largest_contour)
                        # Adăugăm la masca de camere validate
                        cv2.fillPoly(rooms_mask_validated, [largest_contour], 255)
                        # Marcăm zona ca folosită
                        used_mask = cv2.bitwise_or(used_mask, component_mask)
                        _log(f"         ✅ Camera auto-generată {len(rooms_polygons_validated) - 1}: aria={component_area}px ({room_coverage_ratio:.1%})")
        
            _log(f"      ✅ Generat automat {len(rooms_polygons_validated)} camere din zonele închise")
    
        # Actualizăm rooms_polygons cu versiunile validate
        rooms_polygons = rooms_polygons_validated
        rooms_mask = rooms_mask_validated
    
        # ✅ Generez imaginile de debug ale camerelor (doar când DEBUG)
        if DEBUG:
            _log(f"      🔄 Generez imagini de debug camere cu pereții validați...")
            for i, room_poly in enumerate(rooms_polygons):
                # room_{i}_debug.png - cu pereții validați
                debug_img_room = original_img.copy()
                if walls_barrier is not None:
                    walls_colored = cv2.cvtColor(walls_barrier, cv2.COLOR_GRAY2BGR)
                    walls_colored[walls_barrier > 0] = [0, 255, 0]
                    debug_img_room = cv2.addWeighted(debug_img_room, 0.7, walls_colored, 0.3, 0)
                cv2.polylines(debug_img_room, [room_poly], True, (0, 255, 0), 2)
                first_pt = room_poly[0][0] if len(room_poly[0].shape) > 1 else room_poly[0]
                cv2.putText(debug_img_room, f"Room {i} (validated walls)", 
                            (int(first_pt[0]), int(first_pt[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                for idx, pt in enumerate(room_poly):
                    pt_coords = pt[0] if len(pt.shape) > 1 else pt
                    cv2.circle(debug_img_room, (int(pt_coords[0]), int(pt_coords[1])), 3, (255, 0, 0), -1)
                    cv2.putText(debug_img_room, str(idx), (int(pt_coords[0]) + 5, int(pt_coords[1]) + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                cv2.imwrite(str(output_dir / f"room_{i}_debug.png"), debug_img_room)
    
        # ✅ Regenerez rooms.png cu camerele validate (pe baza pereților validați)
        _log(f"      🔄 Regenerez rooms.png cu camerele validate...")
        rooms_img_validated = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
        rooms_img_validated.fill(255)  # Fundal alb
    
        for idx, room_poly in enumerate(rooms_polygons):
            color = room_colors[idx % len(room_colors)]
            # Umplem camera cu culoare
            cv2.fillPoly(rooms_img_validated, [room_poly], color)
            # Desenăm conturul camerei (negru)
            cv2.polylines(rooms_img_validated, [room_poly], True, (0, 0, 0), 2)
    
        # Salvăm rooms.png cu camerele validate
        rooms_output_path = raster_dir / "rooms.png"
        cv2.imwrite(str(rooms_output_path), rooms_img_validated)
        _log(f"      💾 Salvat: rooms.png ({len(rooms_polygons)} camere validate)")
    
        # ✅ IMPORTANT: Pentru pașii următori folosim DOAR segmentele acceptate.
        if walls_overlay_mask is None:
            _log(f"      ❌ walls_overlay_mask nu este disponibilă. Nu pot continua generarea fișierelor de pereți.")
            return None

    # ✅ Folder garage: plan_raw (fără curățare), plan_garage (OCR garage/carport pe plan normal), plan_detected (plan_raw + punct roșu la garaj)
    garage_dir = output_dir / "garage"
    garage_dir.mkdir(parents=True, exist_ok=True)
    plan_raw_img = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    plan_raw_img[accepted_wall_segments_mask > 0] = [255, 255, 255]
    cv2.imwrite(str(garage_dir / "plan_raw.png"), plan_raw_img)
    _log(f"      💾 Salvat: garage/plan_raw.png (segmente pereți fără curățarea flood fill)")

    # Detectare zone (garaj, terasă, balcon, intrare acoperită) cu Gemini; fallback OCR doar la EROARE (nu când răspunsul e gol)
    gemini_zone_detections = {}
    gemini_zone_labels_ok = False
    if gemini_api_key:
        try:
            plan_for_gemini_path = output_dir / "plan_for_gemini_zones.png"
            # Redimensionare la max 1024px pentru apel Gemini mai rapid (payload mai mic)
            MAX_ZONE_IMAGE_PX = 1024
            h_z, w_z = original_img.shape[:2]
            if max(h_z, w_z) > MAX_ZONE_IMAGE_PX:
                scale_z = MAX_ZONE_IMAGE_PX / max(h_z, w_z)
                w_resized = max(1, int(round(w_z * scale_z)))
                h_resized = max(1, int(round(h_z * scale_z)))
                plan_resized = cv2.resize(original_img, (w_resized, h_resized), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(plan_for_gemini_path), plan_resized)
                gemini_zone_detections = call_gemini_zone_labels(
                    plan_for_gemini_path, gemini_api_key, w_resized, h_resized
                )
                # Scalare centre din imagine redimensionată la coordonate original
                if gemini_zone_detections and (w_resized != w_orig or h_resized != h_orig):
                    scale_x = w_orig / max(1, w_resized)
                    scale_y = h_orig / max(1, h_resized)
                    scaled = {}
                    for label, centers in gemini_zone_detections.items():
                        scaled[label] = [(int(cx * scale_x), int(cy * scale_y)) for (cx, cy) in centers]
                    gemini_zone_detections = scaled
            else:
                cv2.imwrite(str(plan_for_gemini_path), original_img)
                gemini_zone_detections = call_gemini_zone_labels(
                    plan_for_gemini_path, gemini_api_key, w_orig, h_orig
                )
            gemini_zone_labels_ok = True
            if gemini_zone_detections:
                _log(f"      [GEMINI] Zone detectate: {list(gemini_zone_detections.keys())}")
        except Exception as e:
            import traceback
            _log(f"      [GEMINI] Eroare detectare zone: {e}")
            traceback.print_exc()

    # Normalizăm la listă: Gemini returnează listă de centre per label (pot fi mai multe garaje, terase etc.)
    def _as_centers_list(v):
        if v is None:
            return []
        if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (list, tuple)):
            return list(v)
        if isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], (int, float)):
            return [tuple(v)]
        return []

    use_ocr_fallback = False  # Dezactivat: OCR durează mult; folosim doar Gemini pentru zone
    ocr_all = {}
    if use_ocr_fallback:
        _log(f"      [OCR] Un singur pas OCR pentru toate zonele (garaj, terasă, balcon, intrare, wintergarden)")
        try:
            ocr_all = run_ocr_all_zones_fallback(original_img, lang="deu+eng", min_conf=25)
        except Exception as e:
            import traceback
            _log(f"      [OCR] Eroare OCR combinat: {e}")
            traceback.print_exc()

    garage_centers_list = gemini_zone_detections.get("garage") or []
    garage_centers_list = _as_centers_list(garage_centers_list) if garage_centers_list else []
    if not garage_centers_list and use_ocr_fallback:
        garage_centers_list = ocr_all.get("garage") or []
    if not garage_centers_list and use_ocr_fallback:
        # Fallback suplimentar doar pentru garaj (dezactivat împreună cu use_ocr_fallback)
        GARAGE_OCR_SEARCH_TERMS = [
            "garage", "garaj", "carport", "parking", "stellplatz",
            "garaz", "garáž", "garaž", "garaż", "autohaus", "gara", "überdacht",
        ]
        _log(f"      [GARAGE] Fallback OCR garaj (grid 2x2): căutare termeni")
        try:
            text_boxes_garage = run_ocr_scan_regions(
                original_img, GARAGE_OCR_SEARCH_TERMS,
                scan_regions=GARAGE_SCAN_REGIONS, lang="deu+eng", min_conf=25
            )
            if not text_boxes_garage:
                text_boxes_garage, _ = run_ocr_on_zones(
                    original_img, GARAGE_OCR_SEARCH_TERMS,
                    steps_dir=None, grid_rows=2, grid_cols=2, zoom_factor=2.0
                )
            if text_boxes_garage:
                text_boxes_garage.sort(key=lambda b: b[5], reverse=True)
                x, y, w, h, text, conf = text_boxes_garage[0]
                garage_centers_list = [(x + w // 2, y + h // 2)]
                _log(f"      [GARAGE] OCR: '{text}' la ({garage_centers_list[0][0]},{garage_centers_list[0][1]})")
        except Exception as e:
            import traceback
            _log(f"      [GARAGE] OCR EROARE: {e}")
            traceback.print_exc()
    elif garage_centers_list:
        _log(f"      [GARAGE] Gemini: {len(garage_centers_list)} centru/e garaj")
    elif gemini_zone_labels_ok:
        _log(f"      [GARAGE] Gemini: garaj negăsit pe plan.")
    plan_garage_img = original_img.copy()
    for cx, cy in garage_centers_list:
        cv2.circle(plan_garage_img, (cx, cy), 8, (0, 0, 255), -1)
    cv2.imwrite(str(garage_dir / "plan_garage.png"), plan_garage_img)

    plan_detected_img = plan_raw_img.copy()
    for idx_garage, (cx, cy) in enumerate(garage_centers_list):
        radius = max(8, min(25, w_orig // 80))
        cv2.circle(plan_detected_img, (cx, cy), radius, (0, 0, 255), -1)
    if garage_centers_list:
        _log(f"      💾 Salvat: garage/plan_detected.png (plan_raw + {len(garage_centers_list)} punct/e garaj)")
    else:
        _log(f"      💾 Salvat: garage/plan_detected.png (plan_raw, fără punct garaj)")
    cv2.imwrite(str(garage_dir / "plan_detected.png"), plan_detected_img)

    # ✅ check_garage: pentru fiecare garaj, flood fill; dacă atinge marginea → pereți insuficienți, aplicăm algoritmul
    for idx_garage, (cx, cy) in enumerate(garage_centers_list):
        check_garage_dir = output_dir / ("check_garage" if len(garage_centers_list) == 1 else f"check_garage_{idx_garage}")
        check_garage_dir.mkdir(parents=True, exist_ok=True)
        plan_for_check = plan_detected_img.copy()
        mask_red = get_red_dot_mask(plan_for_check)
        try:
            touches_edge, visited = check_garage_flood_touches_edge(
                plan_for_check, cx, cy, wall_mask=accepted_wall_segments_mask
            )
            if DEBUG:
                save_check_garage_image(
                    plan_for_check, visited, mask_red,
                    check_garage_dir / "flood_fill.png",
                )
            if touches_edge:
                _log(f"      [GARAGE] Garaj {idx_garage + 1}/{len(garage_centers_list)}: flood atinge marginea → aplic algoritm perete lipsă")
                detection_steps_dir = (garage_dir / ("detection_steps" if len(garage_centers_list) == 1 else f"detection_steps_{idx_garage}")) if DEBUG else None
                if detection_steps_dir is not None:
                    detection_steps_dir.mkdir(parents=True, exist_ok=True)
                segments = find_missing_wall_raycast(
                    plan_for_check, start_x=cx, start_y=cy,
                    wall_mask=accepted_wall_segments_mask,
                    min_gap_px=3,
                )
                if not segments:
                    segments = find_missing_wall_raycast_3walls(
                        plan_for_check, start_x=cx, start_y=cy,
                        wall_mask=accepted_wall_segments_mask,
                    )
                if segments:
                    for seg in segments:
                        draw_segment_on_mask(accepted_wall_segments_mask, seg, 255)
                        draw_segment_on_mask(walls_mask_validated, seg, 255)
                    plan_closed_img = draw_plan_closed_raycast(
                        plan_for_check, segments, mask_red=mask_red,
                        color=(255, 255, 255), thickness=1,
                    )
                    name_closed = f"plan_closed_{idx_garage}.png" if len(garage_centers_list) > 1 else "plan_closed.png"
                    cv2.imwrite(str(garage_dir / name_closed), plan_closed_img)
                    if DEBUG and detection_steps_dir is not None:
                        save_raycast_debug(
                            plan_for_check, cx, cy, segments,
                            detection_steps_dir / "raycast_debug.png",
                            wall_mask=accepted_wall_segments_mask,
                        )
                    _log(f"      [GARAGE] Garaj {idx_garage + 1}: perete lipsă închis (ray cast, {len(segments)} segmente) → {name_closed}")
                else:
                    result = find_missing_wall(
                        plan_for_check,
                        start_x=cx,
                        start_y=cy,
                        steps_dir=detection_steps_dir,
                        step_interval=5000,
                        max_iterations=30000,
                        wall_mask=accepted_wall_segments_mask,
                    )
                    if result is not None:
                        draw_segment_on_mask(accepted_wall_segments_mask, result, 255)
                        draw_segment_on_mask(walls_mask_validated, result, 255)
                        plan_closed_img = draw_plan_closed(
                            plan_for_check,
                            result,
                            mask_red=mask_red,
                            color=(255, 255, 255),
                            thickness=1,
                        )
                        name_closed = f"plan_closed_{idx_garage}.png" if len(garage_centers_list) > 1 else "plan_closed.png"
                        cv2.imwrite(str(garage_dir / name_closed), plan_closed_img)
                        _log(f"      [GARAGE] Garaj {idx_garage + 1}: perete lipsă închis (BFS) → {name_closed}")
                    else:
                        plan_closed_img = plan_for_check.copy()
                        plan_closed_img[mask_red > 0] = (0, 0, 0)
                        name_closed = f"plan_closed_{idx_garage}.png" if len(garage_centers_list) > 1 else "plan_closed.png"
                        cv2.imwrite(str(garage_dir / name_closed), plan_closed_img)
                        _log(f"      [GARAGE] Garaj {idx_garage + 1}: algoritm negăsit segment")
            else:
                _log(f"      [GARAGE] Garaj {idx_garage + 1}/{len(garage_centers_list)}: flood nu atinge marginea → garaj închis")
        except Exception as e:
            import traceback
            _log(f"      [GARAGE] check_garage / wall gap EROARE: {e}")
            traceback.print_exc()

    def run_zone_reconstruction(zone_name: str, search_terms: list, center_from_gemini: tuple = None, use_ocr_fallback: bool = False, zone_index: int = None):
        """Aceeași logică ca la garaj: centru (Gemini sau OCR) → flood fill → dacă atinge marginea → find_missing_wall → desen segment.
        Returnează (center_xy, touches_edge) sau (None, None). zone_index: folosit la mai multe zone (ex. terasa_0)."""
        zone_dir = output_dir / zone_name
        zone_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{zone_index}" if zone_index is not None else ""
        center_xy = center_from_gemini
        if center_xy is None and use_ocr_fallback:
            _log(f"      [{zone_name.upper()}] Fallback OCR (Gemini indisponibil): căutare termeni")
            try:
                boxes = run_ocr_scan_regions(
                    original_img, search_terms,
                    scan_regions=GARAGE_SCAN_REGIONS, lang="deu+eng", min_conf=25
                )
                if not boxes:
                    boxes, _ = run_ocr_on_zones(
                        original_img, search_terms,
                        steps_dir=None, grid_rows=2, grid_cols=2, zoom_factor=2.0
                    )
                if boxes:
                    boxes.sort(key=lambda b: b[5], reverse=True)
                    x, y, w, h, text, conf = boxes[0]
                    center_xy = (x + w // 2, y + h // 2)
                    _log(f"      [{zone_name.upper()}] OCR: '{text}' la ({center_xy[0]},{center_xy[1]})")
                    vis = original_img.copy()
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(vis, center_xy, 8, (0, 0, 255), -1)
                    cv2.imwrite(str(zone_dir / f"plan_detected{suffix}.png"), vis)
                else:
                    _log(f"      [{zone_name.upper()}] TEXT NEGĂSIT.")
            except Exception as e:
                import traceback
                _log(f"      [{zone_name.upper()}] OCR EROARE: {e}")
                traceback.print_exc()
        elif center_xy is None:
            _log(f"      [{zone_name.upper()}] Gemini: zonă negăsită pe plan.")
        else:
            _log(f"      [{zone_name.upper()}] Gemini: centru la ({center_xy[0]},{center_xy[1]})")
            vis = original_img.copy()
            cv2.circle(vis, center_xy, 8, (0, 0, 255), -1)
            cv2.imwrite(str(zone_dir / f"plan_detected{suffix}.png"), vis)
        if center_xy is None:
            return None, None
        cx, cy = center_xy
        # Plan curent (pereți actuali) + bulină roșie
        plan_img = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
        plan_img[accepted_wall_segments_mask > 0] = [255, 255, 255]
        radius = max(8, min(25, w_orig // 80))
        cv2.circle(plan_img, (cx, cy), radius, (0, 0, 255), -1)
        check_dir = output_dir / f"check_{zone_name}{suffix}"
        check_dir.mkdir(parents=True, exist_ok=True)
        mask_red = get_red_dot_mask(plan_img)
        touches_edge = True
        try:
            touches_edge, visited = check_garage_flood_touches_edge(
                plan_img, cx, cy, wall_mask=accepted_wall_segments_mask
            )
            if DEBUG:
                save_check_garage_image(plan_img, visited, mask_red, check_dir / "flood_fill.png")
            if touches_edge:
                _log(f"      [{zone_name.upper()}] Flood atinge marginea → aplic algoritm perete lipsă")
                steps_dir_zone = (zone_dir / "detection_steps") if DEBUG else None
                if steps_dir_zone is not None:
                    steps_dir_zone.mkdir(parents=True, exist_ok=True)
                # Întâi încercăm ray cast (Alg 1: 4 pereți + gap-uri)
                segments = find_missing_wall_raycast(
                    plan_img, start_x=cx, start_y=cy,
                    wall_mask=accepted_wall_segments_mask,
                    min_gap_px=3,
                )
                if not segments:
                    # Fallback: ray cast 3 pereți (Alg 2: o latură complet lipsă)
                    segments = find_missing_wall_raycast_3walls(
                        plan_img, start_x=cx, start_y=cy,
                        wall_mask=accepted_wall_segments_mask,
                    )
                if segments:
                    for seg in segments:
                        draw_segment_on_mask(accepted_wall_segments_mask, seg, 255)
                        draw_segment_on_mask(walls_mask_validated, seg, 255)
                    plan_closed = draw_plan_closed_raycast(
                        plan_img, segments, mask_red=mask_red, color=(255, 255, 255), thickness=1
                    )
                    cv2.imwrite(str(zone_dir / f"plan_closed{suffix}.png"), plan_closed)
                    if steps_dir_zone is not None:
                        save_raycast_debug(
                            plan_img, cx, cy, segments,
                            steps_dir_zone / "raycast_debug.png",
                            wall_mask=accepted_wall_segments_mask,
                        )
                    _log(f"      [{zone_name.upper()}] Perete lipsă închis (ray cast, {len(segments)} segmente) → {zone_name}/plan_closed{suffix}.png")
                else:
                    # Fallback: BFS flood-fill cu parametri adaptați la dimensiunea planului
                    area_px = h_orig * w_orig
                    stable_min_px = max(80, min(2500, int((area_px ** 0.5) / 40)))
                    min_area_before_stable = max(400, int(0.002 * area_px))
                    result = find_missing_wall(
                        plan_img, start_x=cx, start_y=cy,
                        steps_dir=steps_dir_zone, step_interval=5000, max_iterations=30000,
                        wall_mask=accepted_wall_segments_mask,
                        stable_min_px=stable_min_px,
                        min_area_before_stable=min_area_before_stable,
                    )
                    if result is not None:
                        draw_segment_on_mask(accepted_wall_segments_mask, result, 255)
                        draw_segment_on_mask(walls_mask_validated, result, 255)
                        plan_closed = draw_plan_closed(plan_img, result, mask_red=mask_red, color=(255, 255, 255), thickness=1)
                        cv2.imwrite(str(zone_dir / f"plan_closed{suffix}.png"), plan_closed)
                        _log(f"      [{zone_name.upper()}] Perete lipsă închis (BFS): {result.get('type', '?')} → {zone_name}/plan_closed{suffix}.png")
                    else:
                        plan_closed = plan_img.copy()
                        plan_closed[mask_red > 0] = (0, 0, 0)
                        cv2.imwrite(str(zone_dir / f"plan_closed{suffix}.png"), plan_closed)
                        if zone_name in ("terasa", "balcon", "wintergarden"):
                            _log(f"      [{zone_name.upper()}] Nu s-a găsit segment de închidere → zonă rămâne deschisă")
            else:
                _log(f"      [{zone_name.upper()}] Flood nu atinge marginea → zonă închisă, nu aplic algoritm")
        except Exception as e:
            import traceback
            _log(f"      [{zone_name.upper()}] check / find_missing_wall EROARE: {e}")
            traceback.print_exc()
            touches_edge = True
        return center_xy, touches_edge

    # ✅ Fallback OCR doar când Gemini a dat eroare; dacă răspunsul e valid dar gol pentru o zonă, nu căutăm cu OCR
    # Când am făcut deja un OCR combinat (ocr_all), nu mai rulăm OCR per zonă
    use_ocr_fallback_for_zones = use_ocr_fallback and not ocr_all

    terasa_list = gemini_zone_detections.get("terasa") or []
    terasa_list = _as_centers_list(terasa_list) if terasa_list else []
    if use_ocr_fallback and ocr_all:
        terasa_list = ocr_all.get("terasa") or terasa_list
    balcon_list = gemini_zone_detections.get("balcon") or []
    balcon_list = _as_centers_list(balcon_list) if balcon_list else []
    if use_ocr_fallback and ocr_all:
        balcon_list = ocr_all.get("balcon") or balcon_list
    wintergarden_list = gemini_zone_detections.get("wintergarden") or []
    wintergarden_list = _as_centers_list(wintergarden_list) if wintergarden_list else []
    if use_ocr_fallback and ocr_all:
        wintergarden_list = ocr_all.get("wintergarden") or wintergarden_list
    intrare_list = gemini_zone_detections.get("intrare_acoperita") or []
    intrare_list = _as_centers_list(intrare_list) if intrare_list else []
    if use_ocr_fallback and ocr_all:
        intrare_list = ocr_all.get("intrare_acoperita") or intrare_list

    # ✅ Terasă: poate fi mai multe; fiecare centru → reconstructie
    TERASA_OCR_SEARCH_TERMS = [
        "terasa", "terasă", "terrace", "terrasse", "tarrace", "patio", "garden",
    ]
    terasa_results = []  # (center, touches_edge)
    for i, c in enumerate(terasa_list):
        center_xy, touches = run_zone_reconstruction(
            "terasa", TERASA_OCR_SEARCH_TERMS,
            center_from_gemini=c,
            use_ocr_fallback=use_ocr_fallback_for_zones and i == 0,  # OCR doar pentru primul dacă lista e goală
            zone_index=i if len(terasa_list) > 1 else None,
        )
        if center_xy is not None:
            terasa_results.append((center_xy, touches))
    if not terasa_list and use_ocr_fallback_for_zones:
        center_xy, touches = run_zone_reconstruction(
            "terasa", TERASA_OCR_SEARCH_TERMS,
            center_from_gemini=None,
            use_ocr_fallback=True,
            zone_index=None,
        )
        if center_xy is not None:
            terasa_results.append((center_xy, touches))

    # ✅ Balcon: poate fi mai multe; după fiecare zonă închisă calculăm boundary_length_px și area_px
    BALCON_OCR_SEARCH_TERMS = [
        "balcon", "balcony", "balkon", "balkón", "loggia",
    ]
    balcon_results = []  # (center_xy, touches_edge, boundary_length_px, area_px)
    for i, c in enumerate(balcon_list):
        center_xy, touches = run_zone_reconstruction(
            "balcon", BALCON_OCR_SEARCH_TERMS,
            center_from_gemini=c,
            use_ocr_fallback=use_ocr_fallback_for_zones and i == 0,
            zone_index=i if len(balcon_list) > 1 else None,
        )
        if center_xy is not None:
            boundary_px, area_px = 0, 0
            if not touches:
                cx, cy = center_xy
                boundary_px, area_px = compute_zone_boundary_length_and_area(
                    accepted_wall_segments_mask, cx, cy, h_orig, w_orig
                )
            balcon_results.append((center_xy, touches, boundary_px, area_px))
    if not balcon_list and use_ocr_fallback_for_zones:
        center_xy, touches = run_zone_reconstruction(
            "balcon", BALCON_OCR_SEARCH_TERMS,
            center_from_gemini=None,
            use_ocr_fallback=True,
            zone_index=None,
        )
        if center_xy is not None:
            boundary_px, area_px = 0, 0
            if not touches:
                cx, cy = center_xy
                boundary_px, area_px = compute_zone_boundary_length_and_area(
                    accepted_wall_segments_mask, cx, cy, h_orig, w_orig
                )
            balcon_results.append((center_xy, touches, boundary_px, area_px))

    # ✅ Wintergarden: poate fi mai multe; calculăm boundary_length_px și area_px
    WINTERGARDEN_OCR_SEARCH_TERMS = [
        "wintergarten", "wintergarden", "winter garden", "glasanbau",
    ]
    wintergarden_results = []  # (center_xy, touches_edge, boundary_length_px, area_px)
    for i, c in enumerate(wintergarden_list):
        center_xy, touches = run_zone_reconstruction(
            "wintergarden", WINTERGARDEN_OCR_SEARCH_TERMS,
            center_from_gemini=c,
            use_ocr_fallback=use_ocr_fallback_for_zones and i == 0,
            zone_index=i if len(wintergarden_list) > 1 else None,
        )
        if center_xy is not None:
            boundary_px, area_px = 0, 0
            if not touches:
                cx, cy = center_xy
                boundary_px, area_px = compute_zone_boundary_length_and_area(
                    accepted_wall_segments_mask, cx, cy, h_orig, w_orig
                )
            wintergarden_results.append((center_xy, touches, boundary_px, area_px))
    if not wintergarden_list and use_ocr_fallback_for_zones:
        center_xy, touches = run_zone_reconstruction(
            "wintergarden", WINTERGARDEN_OCR_SEARCH_TERMS,
            center_from_gemini=None,
            use_ocr_fallback=True,
            zone_index=None,
        )
        if center_xy is not None:
            boundary_px, area_px = 0, 0
            if not touches:
                cx, cy = center_xy
                boundary_px, area_px = compute_zone_boundary_length_and_area(
                    accepted_wall_segments_mask, cx, cy, h_orig, w_orig
                )
            wintergarden_results.append((center_xy, touches, boundary_px, area_px))

    # ✅ Intrare acoperită (doar din Gemini; poate fi mai multe); colectăm rezultate pentru strip
    INTREE_ACOPERITA_OCR_TERMS = []
    intrare_results = []
    for i, c in enumerate(intrare_list):
        center_xy, touches = run_zone_reconstruction(
            "intrare_acoperita",
            INTREE_ACOPERITA_OCR_TERMS,
            center_from_gemini=c,
            use_ocr_fallback=False,
            zone_index=i if len(intrare_list) > 1 else None,
        )
        if center_xy is not None:
            intrare_results.append((center_xy, touches))

    # ✅ CALCULUL LATURI BALCON: ÎNAINTE de strip (pereții de contact încă există!)
    balcon_center = None
    for center_xy, touches_edge, _bp, _ap in balcon_results:
        if not touches_edge:
            balcon_center = center_xy
            break
    if balcon_center is None and balcon_results:
        balcon_center = balcon_results[0][0]

    accepted_wall_segments_mask_before_strip = accepted_wall_segments_mask.copy()

    # === strip terasa/balcon (codul existent) ===
    # ✅ Strip doar terasă + balcon.
    strip_dir = output_dir / "terasa_balcon_strip"
    strip_dir.mkdir(parents=True, exist_ok=True)
    flood_terasa_balcon_combined = np.zeros((h_orig, w_orig), dtype=np.uint8)
    plan_ff = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    plan_ff[accepted_wall_segments_mask > 0] = [255, 255, 255]
    for center_xy, touches_edge in terasa_results:
        cx, cy = center_xy
        touches, visited = check_garage_flood_touches_edge(
            plan_ff, cx, cy, wall_mask=accepted_wall_segments_mask
        )
        if touches:
            _log(f"      [TERASA] Terasă la ({cx},{cy}): flood atinge marginea → invalidă, nu o scot din pereți")
            continue
        flood_terasa_balcon_combined = np.maximum(flood_terasa_balcon_combined, visited.astype(np.uint8))
    for center_xy, touches_edge, _bound_px, _area_px in balcon_results:
        cx, cy = center_xy
        touches, visited = check_garage_flood_touches_edge(
            plan_ff, cx, cy, wall_mask=accepted_wall_segments_mask
        )
        if touches:
            _log(f"      [BALCON] Balcon la ({cx},{cy}): flood atinge marginea → invalid, nu îl scot din pereți")
            continue
        flood_terasa_balcon_combined = np.maximum(flood_terasa_balcon_combined, visited.astype(np.uint8))
    for center_xy, touches_edge, _bound_px, _area_px in wintergarden_results:
        cx, cy = center_xy
        touches, visited = check_garage_flood_touches_edge(
            plan_ff, cx, cy, wall_mask=accepted_wall_segments_mask
        )
        if touches:
            _log(f"      [WINTERGARDEN] Wintergarden la ({cx},{cy}): flood atinge marginea → invalid, nu îl scot din pereți")
            continue
        flood_terasa_balcon_combined = np.maximum(flood_terasa_balcon_combined, visited.astype(np.uint8))
    # Nu includem intrare_acoperita în strip: 01_walls_from_coords păstrează garajele ȘI intrările acoperite; scoatem doar terasa, balconul și wintergarden.
    # Salvare măsurători balcon/wintergarden pentru pricing (lungime perimetru + suprafață)
    balcon_wintergarden_measurements = []
    for idx, (center_xy, _t, boundary_length_px, area_px) in enumerate(balcon_results):
        balcon_wintergarden_measurements.append({
            "type": "balcon",
            "index": idx,
            "boundary_length_px": boundary_length_px,
            "area_px": area_px,
        })
    for idx, (center_xy, _t, boundary_length_px, area_px) in enumerate(wintergarden_results):
        balcon_wintergarden_measurements.append({
            "type": "wintergarden",
            "index": idx,
            "boundary_length_px": boundary_length_px,
            "area_px": area_px,
        })
    measurements_path = strip_dir / "balcon_wintergarden_measurements.json"
    try:
        with open(measurements_path, "w", encoding="utf-8") as f:
            json.dump(balcon_wintergarden_measurements, f, indent=2, ensure_ascii=False)
        if balcon_wintergarden_measurements:
            _log(f"      💾 Măsurători balcon/wintergarden: {measurements_path.name}")
    except Exception as e:
        _log(f"      ⚠️ Nu s-a putut salva {measurements_path.name}: {e}")

    removed_terasa_balcon = 0
    if np.any(flood_terasa_balcon_combined > 0):
        # Interior casă: flood în spațiul liber (non-perete), excluzând terasa/balcon/wintergarden,
        # ca să păstrăm pereții „lipiți de casă" (frontiera casă–terasă/balcon).
        wall_2d = (accepted_wall_segments_mask > 0).astype(np.uint8)
        free_space = (1 - wall_2d) * 255
        house_interior = np.zeros((h_orig, w_orig), dtype=np.uint8)
        seed_candidates = [
            (w_orig // 2, h_orig // 2),
            (w_orig // 4, h_orig // 2),
            (3 * w_orig // 4, h_orig // 2),
            (w_orig // 2, h_orig // 4),
            (w_orig // 2, 3 * h_orig // 4),
        ]
        step = max(10, min(w_orig, h_orig) // 15)
        for py in range(step, h_orig - step, step):
            for px in range(step, w_orig - step, step):
                seed_candidates.append((px, py))
        for sx, sy in seed_candidates:
            if sy >= h_orig or sx >= w_orig:
                continue
            if free_space[sy, sx] == 0 or flood_terasa_balcon_combined[sy, sx] > 0:
                continue
            mask_ff = np.zeros((h_orig + 2, w_orig + 2), dtype=np.uint8)
            cv2.floodFill(free_space.copy(), mask_ff, (sx, sy), 128, None, None, cv2.FLOODFILL_MASK_ONLY | 4)
            house_interior = np.maximum(house_interior, (mask_ff[1:-1, 1:-1] > 0).astype(np.uint8) * 255)
            # Nu facem break: reunim toate componentele de spațiu liber care nu sunt terasă/balcon,
            # ca pereții „lipiți de casă" (frontiera casă–terasă) să aibă vecin în house_interior.
        if np.any(house_interior > 0):
            cv2.imwrite(str(strip_dir / "flood_interior_casa.png"), house_interior)

        to_remove = np.zeros((h_orig, w_orig), dtype=np.uint8)
        is_wall_tb = (accepted_wall_segments_mask > 0).copy()
        # Vectorizat: pereți care au vecin terasă/balcon dar nu vecin house_interior (dilatare + mască)
        kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        neighbor_terasa = cv2.dilate((flood_terasa_balcon_combined > 0).astype(np.uint8), kernel_3)
        neighbor_house = cv2.dilate((house_interior > 0).astype(np.uint8), kernel_3)
        to_remove = np.where(
            (is_wall_tb > 0) & (neighbor_terasa > 0) & (neighbor_house == 0),
            255, 0
        ).astype(np.uint8)
        removed_terasa_balcon = int(np.sum(to_remove > 0))
        accepted_wall_segments_mask[to_remove > 0] = 0
        walls_mask_validated[to_remove > 0] = 0
        _log(f"      [STRIP] Eliminat {removed_terasa_balcon} pixeli pereți (doar terasă + balcon + wintergarden; pereții lipiți de casă păstrați)")
    cv2.imwrite(str(strip_dir / "flood_interior_terasa_balcon.png"), (flood_terasa_balcon_combined > 0).astype(np.uint8) * 255)
    without_tb = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    without_tb[accepted_wall_segments_mask > 0] = [255, 255, 255]
    cv2.imwrite(str(strip_dir / "walls_without_terasa_balcon.png"), without_tb)
    _log(f"      💾 Salvat: terasa_balcon_strip/ (01_walls_from_coords include garaj + intrare acoperită; fără pereți doar terasă/balcon/wintergarden)")

    # ✅ Reconstruim plan_raw din masca curentă (după strip) ca blacklist și pașii următori să folosească pereții actualizați
    plan_raw_img = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    plan_raw_img[accepted_wall_segments_mask > 0] = [255, 255, 255]

    # ✅ Balcon: mască casă fără balcon, mască doar balcon, regulă pentru includere la acoperiș (folosim primul balcon valid)
    def _flood_fill_remove_supplementary_lines(wall_mask_2d: np.ndarray, h: int, w: int) -> np.ndarray:
        """Elimină liniile suplimentare: pixeli de perete cu ≥2 vecini flood în părți opuse (N-S sau E-W) și <3 vecini pereți (8-conectivitate). Variantă vectorizată."""
        work = (wall_mask_2d > 0).astype(np.uint8) * 255
        flood_base = np.where(work > 0, 0, 255).astype(np.uint8)
        seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
        step = max(1, min(50, w // 10, h // 10))
        for px in range(0, w, step):
            seeds.append((px, 0))
            seeds.append((px, h - 1))
        for py in range(0, h, step):
            seeds.append((0, py))
            seeds.append((w - 1, py))
        kernel_neighbors = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        for _ in range(50):  # max 50 runde (în loc de 5000), fiecare vectorizată
            flood_any = np.zeros((h, w), dtype=np.uint8)
            for cx, cy in seeds:
                if cy >= h or cx >= w or flood_base[cy, cx] != 255:
                    continue
                region_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
                img_f = flood_base.copy()
                cv2.floodFill(img_f, region_mask, (cx, cy), 128, None, None, cv2.FLOODFILL_MASK_ONLY | 4)
                flood_any[region_mask[1:-1, 1:-1] > 0] = 255
            flood_mask = (flood_any > 0)
            wall_bin = (work > 0).astype(np.uint8)
            flood_N = np.roll(flood_mask, 1, axis=0)
            flood_N[0, :] = False
            flood_S = np.roll(flood_mask, -1, axis=0)
            flood_S[-1, :] = False
            flood_E = np.roll(flood_mask, -1, axis=1)
            flood_E[:, -1] = False
            flood_W = np.roll(flood_mask, 1, axis=1)
            flood_W[:, 0] = False
            opp_NS = flood_N & flood_S
            opp_EW = flood_E & flood_W
            has_opposite_flood = opp_NS | opp_EW
            wall_neighbors = cv2.filter2D(wall_bin, -1, kernel_neighbors)
            to_remove = (wall_bin.astype(bool) & has_opposite_flood & (wall_neighbors < 3))
            if np.sum(to_remove) == 0:
                break
            work[to_remove] = 0
            flood_base = np.where(work > 0, 0, 255).astype(np.uint8)
        return work

    def _estimate_wall_thickness(walls_mask: np.ndarray) -> int:
        """Estimează grosimea medie a pereților din mască (în pixeli)."""
        wm = np.max(walls_mask, axis=-1) if walls_mask.ndim == 3 else walls_mask
        wall_bin = (wm > 0).astype(np.uint8)
        h, w = wall_bin.shape
        thick_samples: list[int] = []
        for x in range(0, w, 20):
            col = wall_bin[:, x]
            wl = np.where(col > 0)[0]
            if len(wl) < 2:
                continue
            segs: list[list[int]] = [[wl[0]]]
            for i in range(1, len(wl)):
                if wl[i] - wl[i - 1] <= 2:
                    segs[-1].append(wl[i])
                else:
                    segs.append([wl[i]])
            for s in segs:
                if 2 <= len(s) <= 20:
                    thick_samples.append(len(s))
        if not thick_samples:
            return 3
        return max(2, int(np.median(thick_samples)))

    def _count_balcony_contact_segments(walls_all_mask: np.ndarray, cx_b: int, cy_b: int, h: int, w: int, wall_thickness: int = 3) -> tuple[int, np.ndarray]:
        """
        Numără câte laturi distincte ale casei sunt lipite de balcon.
        Funcționează corect și cu pereți groși (3px+).

        Strategie:
        1. Flood fill din centrul balconului → interior balcon
        2. Flood fill din colțuri → exterior general
        3. Dilată balconul și exteriorul cu kernel bazat pe wall_thickness
        4. Contact = pixeli de perete în zona balcon dilatat, dar NU în zona exterior dilatat
        5. Clasifică pixelii de contact ca H sau V după vecini, grupează în segmente
        """
        wall_bin = (np.max(walls_all_mask, axis=-1) if walls_all_mask.ndim == 3 else walls_all_mask) > 0
        walls_all_mask = (wall_bin.astype(np.uint8)) * 255
        # 1. Flood din centrul balconului
        fb = np.where(walls_all_mask > 0, 0, 255).astype(np.uint8)
        fm = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(fb, fm, (cx_b, cy_b), 128, None, None, cv2.FLOODFILL_MASK_ONLY | 4)
        bal_int = fm[1:-1, 1:-1]

        # 2. Flood exterior (din toate marginile)
        fb_ext = np.where(walls_all_mask > 0, 0, 255).astype(np.uint8)
        ext = np.zeros((h, w), dtype=np.uint8)
        step = max(1, min(50, w // 10, h // 10))
        seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
        for px in range(0, w, step):
            seeds.append((px, 0))
            seeds.append((px, h - 1))
        for py in range(0, h, step):
            seeds.append((0, py))
            seeds.append((w - 1, py))
        for sx, sy in seeds:
            if sy >= h or sx >= w or fb_ext[sy, sx] != 255:
                continue
            rm = np.zeros((h + 2, w + 2), dtype=np.uint8)
            ife = fb_ext.copy()
            cv2.floodFill(ife, rm, (sx, sy), 128, None, None, cv2.FLOODFILL_MASK_ONLY | 4)
            ext[rm[1:-1, 1:-1] > 0] = 255

        # 3. Dilată balcon și exterior cu kernel proporcional cu grosimea pereților
        k = wall_thickness * 2 + 1
        kernel = np.ones((k, k), np.uint8)
        bal_dil = cv2.dilate((bal_int > 0).astype(np.uint8) * 255, kernel)
        ext_dil = cv2.dilate(ext, kernel)

        # 4. Contact: perete adiacent cu balcon dilatat, dar NU cu exterior dilatat
        contact = np.zeros((h, w), dtype=np.uint8)
        contact[(wall_bin > 0) & (bal_dil > 0) & (ext_dil == 0)] = 255

        if np.sum(contact > 0) < 2:
            return 0, contact

        pts = np.column_stack(np.where(contact > 0))
        pts_set = set(map(tuple, pts.tolist()))

        # 5. Clasifică H vs V după vecini de contact
        h_pixels: set[tuple[int, int]] = set()
        v_pixels: set[tuple[int, int]] = set()
        for y, x in pts:
            n_horiz = sum(1 for dx in [-1, 1] if (y, x + dx) in pts_set)
            n_vert = sum(1 for dy in [-1, 1] if (y + dy, x) in pts_set)
            if n_horiz >= n_vert:
                h_pixels.add((y, x))
            else:
                v_pixels.add((y, x))

        tol = wall_thickness + 2
        min_len = wall_thickness * 2

        def cluster_1d(coords: list[int], gap: int = tol) -> list[list[int]]:
            coords = sorted(set(coords))
            if not coords:
                return []
            groups = [[coords[0]]]
            for v in coords[1:]:
                if v - groups[-1][-1] <= gap:
                    groups[-1].append(v)
                else:
                    groups.append([v])
            return groups

        segments: list[dict] = []

        if h_pixels:
            for yg in cluster_1d([y for y, x in h_pixels]):
                in_band = [(y, x) for y, x in h_pixels if yg[0] <= y <= yg[-1]]
                if not in_band:
                    continue
                xs = [x for _, x in in_band]
                if max(xs) - min(xs) >= min_len:
                    segments.append({"type": "h", "y": int(np.mean(yg)), "length": max(xs) - min(xs)})

        if v_pixels:
            for xg in cluster_1d([x for y, x in v_pixels]):
                in_band = [(y, x) for y, x in v_pixels if xg[0] <= x <= xg[-1]]
                if not in_band:
                    continue
                ys = [y for y, _ in in_band]
                if max(ys) - min(ys) >= min_len:
                    segments.append({"type": "v", "x": int(np.mean(xg)), "length": max(ys) - min(ys)})

        return len(segments), contact

    # ✅ Scheletonizare înainte de blacklist: imaginea și masca folosite la blacklist au pereți 1px, astfel eliminarea pereților zonei blacklist șterge tot
    if np.any(accepted_wall_segments_mask > 0):
        try:
            from skimage.morphology import skeletonize
            wall_2d = accepted_wall_segments_mask
            if wall_2d.ndim == 3:
                wall_2d = np.max(wall_2d, axis=-1).astype(np.uint8)
            _, wall_2d = cv2.threshold(wall_2d, 0, 255, cv2.THRESH_BINARY)
            binary_1px = (wall_2d > 0).astype(np.uint8)
            skel = skeletonize(binary_1px.astype(bool))
            accepted_wall_segments_mask_1px = (skel.astype(np.uint8)) * 255
            if accepted_wall_segments_mask.ndim == 3:
                accepted_wall_segments_mask[:, :, 0] = accepted_wall_segments_mask_1px
                accepted_wall_segments_mask[:, :, 1] = accepted_wall_segments_mask_1px
                accepted_wall_segments_mask[:, :, 2] = accepted_wall_segments_mask_1px
            else:
                accepted_wall_segments_mask[:, :] = accepted_wall_segments_mask_1px
            if walls_mask_validated is not None and walls_mask_validated.size > 0:
                wv = np.max(walls_mask_validated, axis=-1).astype(np.uint8) if walls_mask_validated.ndim == 3 else walls_mask_validated
                _, wv = cv2.threshold(wv, 0, 255, cv2.THRESH_BINARY)
                skel_wv = skeletonize((wv > 0).astype(bool))
                wv_1px = (skel_wv.astype(np.uint8)) * 255
                if walls_mask_validated.ndim == 3:
                    walls_mask_validated[:, :, 0] = wv_1px
                    walls_mask_validated[:, :, 1] = wv_1px
                    walls_mask_validated[:, :, 2] = wv_1px
                else:
                    walls_mask_validated[:, :] = wv_1px
            plan_raw_img = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
            plan_raw_img[accepted_wall_segments_mask_1px > 0] = [255, 255, 255]
            _log(f"      🔗 Skeleton 1px aplicat înainte de blacklist (imaginea și masca pentru blacklist au pereți 1px)")
        except Exception as e:
            _log(f"      ⚠️ Skeleton înainte de blacklist eșuat: {e}")

    # ✅ Blacklist: cuvinte (ex. pool) – detectare prin Gemini (fără OCR); flood fill din fiecare detecție; dacă nu atinge marginea, ștergem pereții din regiune
    blacklist_path = MODULE_DIR / "blacklist_words.json"
    if blacklist_path.exists():
        try:
            with open(blacklist_path, "r", encoding="utf-8") as f:
                blacklist_cfg = json.load(f)
            blacklist_terms = blacklist_cfg.get("blacklist_terms", [])
        except (json.JSONDecodeError, OSError) as e:
            blacklist_terms = []
            _log(f"      [BLACKLIST] Eroare citire JSON: {e}")
    else:
        blacklist_terms = []
    blacklist_detections: list[tuple[int, int, str]] = []
    if blacklist_terms and gemini_api_key:
        try:
            plan_for_blacklist = output_dir / "plan_for_gemini_zones.png"
            if not plan_for_blacklist.exists():
                cv2.imwrite(str(plan_for_blacklist), original_img)
            blacklist_detections = call_gemini_blacklist(
                plan_for_blacklist, gemini_api_key, w_orig, h_orig, blacklist_terms
            )
            if blacklist_detections:
                _log(f"      [BLACKLIST] Gemini: {len(blacklist_detections)} detecții")
        except Exception as e:
            import traceback
            _log(f"      [BLACKLIST] Gemini EROARE: {e}")
            traceback.print_exc()
    if blacklist_terms or blacklist_detections:
        blacklist_dir = output_dir / "blacklist"
        blacklist_dir.mkdir(parents=True, exist_ok=True)
        if blacklist_detections:
            mask_red_empty = np.zeros((h_orig, w_orig), dtype=np.uint8)
            for idx, (cx, cy, label) in enumerate(blacklist_detections):
                touches_edge, visited = check_garage_flood_touches_edge(
                    plan_raw_img, cx, cy, wall_mask=accepted_wall_segments_mask
                )
                safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)[:25]
                save_check_garage_image(
                    plan_raw_img, visited, mask_red_empty,
                    blacklist_dir / f"detection_{idx}_{safe_label}_flood.png",
                )
                # Pereții blacklist se șterg doar dacă flood fill-ul NU atinge marginea imaginii (zonă închisă, ex. piscină).
                # 01_walls_from_coords nu include pereții cuvintelor din blacklist în acest caz.
                # Eliminare iterativă: pereții pot fi groși (2+ px), deci îi scoatem strat cu strat până nu mai e niciunul adiacent de zonă.
                if not touches_edge:
                    wall_2d = accepted_wall_segments_mask
                    if wall_2d.ndim == 3:
                        wall_2d = np.max(wall_2d, axis=-1).astype(np.uint8)
                    wall_2d = (wall_2d > 0).astype(np.uint8) * 255
                    visited_work = visited.copy()
                    total_removed = 0
                    while True:
                        prev = wall_2d.copy()
                        n1 = remove_walls_adjacent_to_region(wall_2d, visited_work)
                        if n1 == 0:
                            break
                        total_removed += n1
                        removed = (prev > 0) & (wall_2d == 0)
                        visited_work = np.maximum(visited_work, removed.astype(np.uint8))
                    if accepted_wall_segments_mask.ndim == 3:
                        accepted_wall_segments_mask[:, :, 0] = wall_2d
                        accepted_wall_segments_mask[:, :, 1] = wall_2d
                        accepted_wall_segments_mask[:, :, 2] = wall_2d
                    else:
                        accepted_wall_segments_mask[:, :] = wall_2d
                    walls_mask_validated_2d = (np.max(walls_mask_validated, axis=-1) if walls_mask_validated.ndim == 3 else walls_mask_validated.copy()).astype(np.uint8)
                    walls_mask_validated_2d = (walls_mask_validated_2d > 0).astype(np.uint8) * 255
                    visited_work = visited.copy()
                    while True:
                        prev = walls_mask_validated_2d.copy()
                        n2 = remove_walls_adjacent_to_region(walls_mask_validated_2d, visited_work)
                        if n2 == 0:
                            break
                        removed = (prev > 0) & (walls_mask_validated_2d == 0)
                        visited_work = np.maximum(visited_work, removed.astype(np.uint8))
                    if walls_mask_validated.ndim == 3:
                        walls_mask_validated[:, :, 0] = walls_mask_validated_2d
                        walls_mask_validated[:, :, 1] = walls_mask_validated_2d
                        walls_mask_validated[:, :, 2] = walls_mask_validated_2d
                    else:
                        walls_mask_validated[:, :] = walls_mask_validated_2d
                    _log(f"      [BLACKLIST] '{label}' (idx {idx}): zonă închisă (flood nu atinge marginea) → eliminat {total_removed} pixeli pereți (exclus din 01_walls_from_coords)")
                else:
                    _log(f"      [BLACKLIST] '{label}' (idx {idx}): flood atinge marginea → zonă deschisă, pereții rămân în 01_walls_from_coords")
                # Plan cu zona atinsă de flood fill ștearsă (folosit mai departe în proiect)
                plan_without_zone = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
                walls_for_plan = np.max(accepted_wall_segments_mask, axis=-1) if accepted_wall_segments_mask.ndim == 3 else accepted_wall_segments_mask
                plan_without_zone[walls_for_plan > 0] = [255, 255, 255]
                plan_without_zone[visited > 0] = [0, 0, 0]  # zona blacklist ștearsă
                plan_without_zone_path = blacklist_dir / f"detection_{idx}_{safe_label}_plan_without_zone.png"
                cv2.imwrite(str(plan_without_zone_path), plan_without_zone)
            # După blacklist: plan_raw_img reflectă masca actuală (fără pereții zonelor blacklist șterse), folosit mai departe
            plan_raw_img = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
            plan_raw_img[accepted_wall_segments_mask > 0] = [255, 255, 255]
            _log(f"      💾 [BLACKLIST] plan_raw_img actualizat (fără pereții zonelor blacklist șterse)")
        elif blacklist_terms and not gemini_api_key:
            _log(f"      [BLACKLIST] Termeni încărcați dar GEMINI_API_KEY lipsește – fără detectare.")

    # ✅ Flood fill din colțuri + margini. Eliminăm pixelii de perete care au ≥2 vecini flood în părți OPUSE
    # (N-S sau E-W); nu eliminăm joncțiunile (≥2 vecini pereți în 8-conectivitate).
    # Verificăm FIECARE pixel care e diferit de negru (non-black = perete).
    if np.any(accepted_wall_segments_mask > 0):
        try:
            from skimage.morphology import skeletonize
            # Normalizare la 2D: orice pixel non-negru = perete (pentru logică clară)
            wall_mask_2d = accepted_wall_segments_mask
            if wall_mask_2d.ndim == 3:
                wall_mask_2d = np.max(wall_mask_2d, axis=-1).astype(np.uint8)
            _, wall_mask_2d = cv2.threshold(wall_mask_2d, 0, 255, cv2.THRESH_BINARY)
            binary_1px = (wall_mask_2d > 0).astype(np.uint8)
            skel = skeletonize(binary_1px.astype(bool))
            accepted_wall_segments_mask = (skel.astype(np.uint8)) * 255
            walls_mask_validated = accepted_wall_segments_mask.copy()
            walls_overlay_mask = accepted_wall_segments_mask.copy()
            _log(f"      🔗 Skeleton 1px aplicat înainte de flood fill (detectare vecini opuși)")
        except Exception as e:
            _log(f"      ⚠️ Skeleton 1px înainte de flood fill eșuat: {e}")

    # Asigurăm mască 2D pentru pasul următor: orice pixel diferit de negru = perete
    wall_mask_2d = accepted_wall_segments_mask
    if wall_mask_2d.ndim == 3:
        wall_mask_2d = np.max(wall_mask_2d, axis=-1).astype(np.uint8)
    _, wall_mask_2d = cv2.threshold(wall_mask_2d, 0, 255, cv2.THRESH_BINARY)
    accepted_wall_segments_mask = wall_mask_2d
    # is_wall = fiecare pixel care e diferit de negru (verificăm toți acești pixeli)
    is_wall = (accepted_wall_segments_mask != 0)
    total_non_black = int(np.sum(is_wall))
    _log(f"      🌊 Flood fill din colțuri + margini; verific {total_non_black} pixeli non-negri (≥2 vecini flood opuse N-S sau E-W)...")

    flood_base = np.where(is_wall, 0, 255).astype(np.uint8)

    # Seed-uri: colțuri + margini (pas ~50px) ca să acoperim tot exteriorul; 8-conectivitate
    seeds = [(0, 0), (w_orig - 1, 0), (0, h_orig - 1), (w_orig - 1, h_orig - 1)]
    step = max(1, min(50, w_orig // 10, h_orig // 10))
    for px in range(0, w_orig, step):
        seeds.append((px, 0))
        seeds.append((px, h_orig - 1))
    for py in range(0, h_orig, step):
        seeds.append((0, py))
        seeds.append((w_orig - 1, py))
    flood_any = np.zeros((h_orig, w_orig), dtype=np.uint8)
    for cx, cy in seeds:
        if cy >= h_orig or cx >= w_orig:
            continue
        if flood_base[cy, cx] != 255:
            continue
        region_mask = np.zeros((h_orig + 2, w_orig + 2), dtype=np.uint8)
        img_copy = flood_base.copy()
        # 4-conectivitate: flood fill nu trece prin pixeli diagonali
        cv2.floodFill(img_copy, region_mask, (cx, cy), 128, None, None, cv2.FLOODFILL_MASK_ONLY | 4)
        r = region_mask[1:-1, 1:-1] > 0
        flood_any[r] = 255

    # Vizualizare flood fill înainte de generarea 01_walls_from_coords: exterior (flood din colțuri) + pereți
    flood_cleanup_viz = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    flood_cleanup_viz[flood_any > 0] = [0, 180, 0]  # Verde = exterior (flood din colțuri/margini)
    flood_cleanup_viz[accepted_wall_segments_mask > 0] = [255, 255, 255]  # Alb = pereți (înainte de ștergerea liniilor suplimentare)
    flood_viz_path = output_dir / "00_flood_fill_cleanup_viz.png"
    cv2.imwrite(str(flood_viz_path), flood_cleanup_viz)
    _log(f"      💾 Salvat: {flood_viz_path.name} (flood din colțuri = verde, pereți = alb, înainte de ștergerea liniilor cu ≥2 vecini flood opuși)")

    # Eliminăm pixelii de perete care au ≥2 vecini flood în părți OPUSE (N-S sau E-W).
    # Un pixel este șters doar dacă are 0, 1 sau 2 vecini pereți (8-conectivitate).
    # Variantă vectorizată (NumPy/OpenCV) în loc de buclă Python pe milioane de pixeli.
    flood_mask = (flood_any > 0)
    wall_bin = (accepted_wall_segments_mask > 0).astype(np.uint8)

    # 1. Vecinii flood N/S/E/W prin shiftare matriceală (fără wrap la margini)
    flood_N = np.roll(flood_mask, 1, axis=0).astype(bool)
    flood_N[0, :] = False
    flood_S = np.roll(flood_mask, -1, axis=0).astype(bool)
    flood_S[-1, :] = False
    flood_E = np.roll(flood_mask, -1, axis=1).astype(bool)
    flood_E[:, -1] = False
    flood_W = np.roll(flood_mask, 1, axis=1).astype(bool)
    flood_W[:, 0] = False

    opp_NS = flood_N & flood_S
    opp_EW = flood_E & flood_W
    has_opposite_flood = opp_NS | opp_EW

    # 2. Număr de vecini pereți (8-conectivitate) cu convoluție
    kernel_neighbors = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    wall_neighbors = cv2.filter2D(wall_bin, -1, kernel_neighbors)

    # 3. Pixelii de șters: perete + flood opus + < 3 vecini pereți
    to_remove = wall_bin.astype(bool) & has_opposite_flood & (wall_neighbors < 3)
    removed_total = int(np.sum(to_remove))
    accepted_wall_segments_mask[to_remove] = 0

    # Vizualizare prima rundă (roșu = eliminate, alb = rămase)
    if removed_total > 0:
        lines_removed_viz = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
        lines_removed_viz[flood_any > 0] = [0, 180, 0]
        lines_removed_viz[accepted_wall_segments_mask > 0] = [255, 255, 255]
        lines_removed_viz[to_remove] = [0, 0, 255]
        n_red = removed_total
        n_white = int(np.sum(accepted_wall_segments_mask > 0))
        label = f"rosii: {n_red}  albi: {n_white}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(1.2, h_orig / 800))
        thickness = max(1, int(round(2 * font_scale)))
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        pad = 8
        x2, y2 = tw + 2 * pad, th + 2 * pad
        roi = lines_removed_viz[0:y2, 0:x2]
        is_red = (roi[:, :, 2] == 255) & (roi[:, :, 0] == 0)
        roi_bg = np.full_like(roi, (40, 40, 40))
        roi_border = roi_bg.copy()
        cv2.rectangle(roi_border, (0, 0), (x2 - 1, y2 - 1), (200, 200, 200), 1)
        cv2.putText(roi_border, label, (pad, th + pad), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        np.copyto(roi, roi_border, where=np.broadcast_to(~is_red[:, :, None], roi.shape))
        lines_removed_path = output_dir / "00_flood_fill_lines_removed.png"
        cv2.imwrite(str(lines_removed_path), lines_removed_viz)
        _log(f"      💾 Salvat: {lines_removed_path.name} (roșu = linii suplimentare eliminate, alb = pereți care rămân)")
        white_only = (accepted_wall_segments_mask > 0)
        flood_test = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
        flood_test[white_only] = [255, 255, 255]
        flood_test_path = output_dir / "00_flood_test.png"
        cv2.imwrite(str(flood_test_path), flood_test)
        _log(f"      💾 Salvat: {flood_test_path.name} (doar albi = pereți care rămân)")
    walls_mask_validated = accepted_wall_segments_mask.copy()
    walls_overlay_mask = walls_mask_validated.copy()
    if removed_total > 0:
        _log(f"      ✅ Eliminat {removed_total} pixeli de perete (≥2 vecini flood în părți opuse)")
    else:
        _log(f"      ℹ️ Nu s-au găsit pixeli de perete de eliminat (din {total_non_black} pixeli non-negri verificați)")
    
    # ✅ Asigurăm grosime 1px înainte de salvare (skeletonizare dacă masca are linii groase)
    if np.any(accepted_wall_segments_mask > 0):
        try:
            from skimage.morphology import skeletonize
            binary_1px = (accepted_wall_segments_mask > 0).astype(np.uint8)
            skel = skeletonize(binary_1px.astype(bool))
            accepted_wall_segments_mask = (skel.astype(np.uint8)) * 255
            walls_mask_validated = accepted_wall_segments_mask.copy()
            walls_overlay_mask = accepted_wall_segments_mask.copy()
            _log(f"      🔗 Skeleton 1px aplicat pentru 01_walls_from_coords.png")
        except Exception as e:
            _log(f"      ⚠️ Skeleton 1px pentru 01_walls_from_coords eșuat: {e}")

    # ✅ Regulă balcon acoperiș: număr laturi lipite de casă pe masca curată; walls_mask_for_roof construit aici
    # Folosim masca dinainte de strip (accepted_wall_segments_mask_before_strip) ca pereții de contact să existe
    if balcon_center is not None:
        cx_b, cy_b = balcon_center
        walls_2d = (np.max(accepted_wall_segments_mask_before_strip, axis=-1) if accepted_wall_segments_mask_before_strip.ndim == 3 else accepted_wall_segments_mask_before_strip).astype(np.uint8)
        if np.any(walls_2d > 0):
            walls_for_count = (walls_2d > 0).astype(np.uint8) * 255
            wt = _estimate_wall_thickness(accepted_wall_segments_mask_before_strip)
            n_sides, contact_mask = _count_balcony_contact_segments(
                walls_for_count, cx_b, cy_b, h_orig, w_orig, wall_thickness=wt
            )
        else:
            n_sides, contact_mask = 0, np.zeros((h_orig, w_orig), dtype=np.uint8)
        _log(f"      [BALCON ROOF] laturi lipite de casă: {n_sides}")

        if n_sides >= 2:
            # Masca CU balcon (dinainte de strip), ca dreptunghiurile acoperiș să includă balconul
            walls_mask_for_roof = accepted_wall_segments_mask_before_strip.copy()
            include_balcon_in_roof = True
            _log(f"      [BALCON ROOF] {n_sides} laturi → balcon INCLUS la acoperiș")
        else:
            fb_b = np.where(accepted_wall_segments_mask > 0, 0, 255).astype(np.uint8)
            fm_b = np.zeros((h_orig + 2, w_orig + 2), dtype=np.uint8)
            cv2.floodFill(fb_b, fm_b, (cx_b, cy_b), 128, None, None, cv2.FLOODFILL_MASK_ONLY | 4)
            bal_int_b = fm_b[1:-1, 1:-1]

            walls_mask_for_roof = accepted_wall_segments_mask.copy()
            # Vectorizat: eliminăm pereții care au vecin în balcon (4-conectivitate)
            kernel_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
            neighbor_balcon = cv2.dilate((bal_int_b > 0).astype(np.uint8), kernel_4)
            walls_mask_for_roof = np.where(neighbor_balcon > 0, 0, walls_mask_for_roof).astype(np.uint8)
            include_balcon_in_roof = False
            _log(f"      [BALCON ROOF] 1 latură → balcon EXCLUS din acoperiș")

        balcon_dir = output_dir / "balcon_roof_rule"
        balcon_dir.mkdir(parents=True, exist_ok=True)
        contact_viz = original_img.copy()
        if contact_viz.ndim == 2:
            contact_viz = cv2.cvtColor(contact_viz, cv2.COLOR_GRAY2BGR)
        contact_viz[contact_mask > 0] = [0, 255, 255]
        cv2.putText(contact_viz, f"Laturi lipite: {n_sides}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imwrite(str(balcon_dir / "laturi_lipite_viz.png"), contact_viz)
    else:
        walls_mask_for_roof = accepted_wall_segments_mask.copy()
        include_balcon_in_roof = False

    # ✅ Salvăm 01_walls_from_coords.png: același conținut ca 00_flood_test (albi fără roșu), skeletonizat 1px.
    # Margine neagră 20px SUPLIMENTARĂ: mărim poza (canvas +40px pe lățime/înălțime), conținutul rămâne în centru.
    segments_path = output_dir / "01_walls_from_coords.png"
    segments_img = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    segments_img[accepted_wall_segments_mask > 0] = [255, 255, 255]
    border_px = 20
    H, W = h_orig + 2 * border_px, w_orig + 2 * border_px
    segments_img_with_border = np.zeros((H, W, 3), dtype=np.uint8)
    segments_img_with_border[border_px : border_px + h_orig, border_px : border_px + w_orig] = segments_img
    _log(f"      🔲 Margine neagră {border_px}px suplimentară (poza mărită la {W}x{H})")
    _log(f"      💾 Salvat: {segments_path.name} (același conținut ca 00_flood_test, 1px)")
    cv2.imwrite(str(segments_path), segments_img_with_border)

    # ✅ Recalculăm walls_barrier din segmentele acceptate DUPĂ eliminare; dacă inputul e 1px (api_walls_from_json_1px), păstrăm 1px
    if _input_is_1px:
        wall_thickness_barrier = 1
    else:
        wall_thickness_barrier = max(5, int(w_orig * 0.00005))
    kernel_barrier = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wall_thickness_barrier, wall_thickness_barrier))
    walls_barrier = cv2.dilate(accepted_wall_segments_mask, kernel_barrier, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    walls_barrier = cv2.morphologyEx(walls_barrier, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # 2. Aplicăm grosimea pereților -> 02_walls_thick.png (derivat din segmente acceptate DUPĂ eliminare)
    walls_thick = walls_barrier.copy()
    
    cv2.imwrite(str(output_dir / "02_walls_thick.png"), walls_thick)
    _log(f"      ✅ Salvat: 02_walls_thick.png (derivat din segmente acceptate)")
    
    # 2b. Generează outline-ul pereților (fără interior) -> 02b_walls_outline.png
    _log(f"      🔲 Generez outline pereți (fără interior)...")
    walls_outline_only = np.zeros((h_orig, w_orig), dtype=np.uint8)
    
    # Găsim toți pixelii care sunt lângă un pixel alb (pereți) dar nu sunt pereți (outline = dilatare apoi exclude interior)
    kernel_outline = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_walls = cv2.dilate(walls_thick, kernel_outline)
    walls_outline_only = np.where((dilated_walls > 0) & (walls_thick == 0), 255, 0).astype(np.uint8)
    _log(f"      💾 Salvat: 02b_walls_outline.png")
    
    # 3. Suprapunem pereții peste planul original, colorați cu mov -> 03_walls_overlay.png
    _log(f"      🎨 Generez overlay pereți peste plan...")
    overlay = original_img.copy()
    # Mov în BGR: [128, 0, 128]
    overlay[walls_thick > 0] = [128, 0, 128]
    cv2.imwrite(str(output_dir / "03_walls_overlay.png"), overlay)
    _log(f"      💾 Salvat: 03_walls_overlay.png")
    
    # 4. Randare 3D va fi generată după pasul 8 (după generarea pereților interiori/exteriori)
    # (mutată mai jos pentru a putea folosi walls_exterior și walls_interior)
    
    # 4. Aplicăm outline roșu pe ambele părți ale pereților -> 05_walls_outline.png
    _log(f"      🔲 Aplic outline roșu pe ambele părți...")
    walls_outline = np.where((dilated_walls > 0) & (walls_thick == 0), 255, 0).astype(np.uint8)

    # Creăm imaginea cu outline roșu, păstrând pereții albi
    outline_img = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    # Pereții albi
    outline_img[walls_thick > 0] = [255, 255, 255]  # BGR: alb
    # Outline roșu peste pereți
    outline_img[walls_outline > 0] = [0, 0, 255]  # BGR: roșu
    
    cv2.imwrite(str(output_dir / "05_walls_outline.png"), outline_img)
    _log(f"      💾 Salvat: 05_walls_outline.png")
    
    # 6. Flood fill albastru deschis din colțuri, outline-urile care ating devin albastru închis, cele care nu ating devin galben -> 06_walls_separated.png
    _log(f"      🌊 Fac flood fill și separare outline-uri...")
    
    # Creăm o imagine pentru flood fill (spațiile libere sunt 255, pereții sunt 0)
    # Astfel, flood fill-ul se va opri automat la pereți
    flood_fill_image = (255 - walls_thick).astype(np.uint8)
    
    # Creăm o mască pentru flood fill (dimensiuni +2 pe fiecare parte)
    flood_fill_mask = np.zeros((h_orig + 2, w_orig + 2), dtype=np.uint8)
    
    # Flood fill albastru deschis din colțuri, folosind pereții ca și barieră
    corners = [(0, 0), (w_orig-1, 0), (0, h_orig-1), (w_orig-1, h_orig-1)]
    for x, y in corners:
        if walls_thick[y, x] == 0:  # Nu este perete
            # Flood fill se va opri automat la pereți (unde valoarea este 0)
            cv2.floodFill(flood_fill_image, flood_fill_mask, (x, y), 128, 
                         loDiff=(0,), upDiff=(0,), flags=4)
    
    # Rezultatul flood fill: pixelii cu valoarea 128 sunt cei atinși de flood fill
    flood_fill_result = (flood_fill_image == 128).astype(np.uint8) * 255
    
    # Creăm imaginea separată
    separated_img = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    
    # Flood fill albastru deschis
    separated_img[flood_fill_result > 0] = [255, 200, 100]  # BGR: albastru deschis
    
    # Outline-urile care ating flood fill devin albastru închis (vectorizat)
    dilated_flood = cv2.dilate(flood_fill_result, kernel_outline)
    outline_touches_flood = np.where((walls_outline > 0) & (dilated_flood > 0), 255, 0).astype(np.uint8)
    separated_img[walls_outline > 0] = [0, 255, 255]  # galben default
    separated_img[outline_touches_flood > 0] = [255, 100, 0]  # albastru închis unde atinge flood
    
    cv2.imwrite(str(output_dir / "06_walls_separated.png"), separated_img)
    _log(f"      💾 Salvat: 06_walls_separated.png")
    
    # 7. Generăm pereții interiori -> 07_walls_interior.png
    _log(f"      🏗️ Generez pereții interiori...")
    walls_interior = np.zeros((h_orig, w_orig), dtype=np.uint8)
    
    # Pereții interiori = pereții care sunt lângă outline-urile care NU ating flood fill (galben) — vectorizat
    outline_interior = (walls_outline > 0) & (outline_touches_flood == 0)
    dilated_outline_int = cv2.dilate(outline_interior.astype(np.uint8), kernel_outline)
    walls_interior = np.where((walls_thick > 0) & (dilated_outline_int > 0), 255, 0).astype(np.uint8)
    _log(f"      💾 Salvat: 07_walls_interior.png")
    
    # 8. Generăm pereții exteriori -> 08_walls_exterior.png
    _log(f"      🏗️ Generez pereții exteriori...")
    # Pereții exteriori = pereții care sunt lângă outline-urile care ating flood fill — vectorizat
    outline_exterior = (walls_outline > 0) & (outline_touches_flood > 0)
    dilated_outline_ext = cv2.dilate(outline_exterior.astype(np.uint8), kernel_outline)
    walls_exterior = np.where((walls_thick > 0) & (dilated_outline_ext > 0), 255, 0).astype(np.uint8)
    
    cv2.imwrite(str(output_dir / "07_walls_interior.png"), walls_interior)
    _log(f"      💾 Salvat: 07_walls_interior.png")
    
    cv2.imwrite(str(output_dir / "08_walls_exterior.png"), walls_exterior)
    _log(f"      💾 Salvat: 08_walls_exterior.png")
    
    # 4b. Randare 3D izometrică reală cu extruziune și iluminare -> 04_walls_3d.png
    _log(f"      🎨 Generez randare 3D izometrică (extruziune reală cu iluminare)...")
    rendering_success = False
    try:
        # ✅ Folosim walls_thick direct pentru extrudare
        # walls_thick conține TOȚI pereții validați (exteriori + interiori) cu grosime aplicată
        walls_all = walls_thick.copy()
        
        wall_pixels = np.where(walls_all > 0)
        if len(wall_pixels[0]) == 0:
            _log(f"         ⚠️ Nu s-au găsit pereți pentru randare 3D")
        else:
            # Înălțime pereți în pixeli (proporțională, dar mai mică pentru a nu ieși din cadru)
            wall_height_px = max(24, int(min(w_orig, h_orig) * 0.10))  # ~10% din dimensiunea minimă
            
            # Calculăm dimensiunile canvas-ului pentru proiecție izometrică
            # Proiecția izometrică: x' = (x - y) * cos(30°), y' = (x + y) * sin(30°) + z
            # Factorii: cos(30°) ≈ 0.866, sin(30°) = 0.5
            iso_scale_x = 0.8
            iso_scale_y = 0.45
            
            # Margină pentru a include toată extruziunea
            margin = int(wall_height_px * 3)
            canvas_w = int((w_orig + h_orig) * iso_scale_x * 2 + margin * 2)
            canvas_h = int((w_orig + h_orig) * iso_scale_y + wall_height_px + margin * 2)
            
            # Canvas pentru randare (BGR)
            render_3d = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            
            # Z-buffer pentru depth sorting (folosit pentru iluminare)
            z_buffer = np.full((canvas_h, canvas_w), -np.inf, dtype=np.float32)
            
            # Funcție de transformare izometrică
            def to_iso(x, y, z=0):
                # Centru canvas
                center_x = canvas_w // 2
                center_y = canvas_h // 2
                
                # Transformare izometrică
                iso_x = int((x - y) * iso_scale_x + center_x)
                iso_y = int((x + y) * iso_scale_y - z + center_y)
                return iso_x, iso_y
            
            # Vector de iluminare (direcție lumină din stânga-sus)
            light_dir = np.array([0.5, 0.5, 0.7])
            light_dir = light_dir / np.linalg.norm(light_dir)
            
            # Normalizăm pereții pentru a obține outline-uri clare
            walls_binary = (walls_all > 0).astype(np.uint8) * 255
            
            # Obținem outline-urile pereților
            kernel = np.ones((3, 3), np.uint8)
            walls_dilated = cv2.dilate(walls_binary, kernel, iterations=1)
            outline = walls_dilated - walls_binary
            
            # Desenăm fețele 3D ale pereților
            #
            # 1. Fața superioară (top face) - cea mai luminată
            _log(f"         📐 Desenez fața superioară...")
            top_face_pixels = np.where(walls_binary > 0)
            for y, x in zip(top_face_pixels[0], top_face_pixels[1]):
                iso_x, iso_y = to_iso(x, y, wall_height_px)
                if 0 <= iso_x < canvas_w and 0 <= iso_y < canvas_h:
                    # Normală pentru fața superioară (0, 0, 1)
                    normal = np.array([0, 0, 1])
                    # Calculăm iluminarea (foarte deschisă, ca un capac alb)
                    light_intensity = max(0.4, np.dot(normal, light_dir))
                    color = int(220 + 35 * light_intensity)
                    render_3d[iso_y, iso_x] = [color, color, color]
                    z_buffer[iso_y, iso_x] = wall_height_px
            
            # 2. Fețele laterale (side faces) - cu gradient de iluminare, mai deschise
            _log(f"         📐 Desenez fețele laterale...")
            
            # Fețe laterale: pentru fiecare pixel de perete, desenăm linia verticală (cu sampling pe înălțime)
            wall_coords = list(zip(wall_pixels[0], wall_pixels[1]))
            
            # Procesăm în batch-uri pentru performanță
            batch_size = 1500
            for batch_start in range(0, len(wall_coords), batch_size):
                batch_end = min(batch_start + batch_size, len(wall_coords))
                batch = wall_coords[batch_start:batch_end]
                
                for y, x in batch:
                    # Desenăm linia verticală pentru acest pixel de perete (sampling pentru viteză)
                    for z in range(0, wall_height_px + 1, 2):
                        iso_x, iso_y = to_iso(x, y, z)
                        if 0 <= iso_x < canvas_w and 0 <= iso_y < canvas_h:
                            # Verificăm dacă acest pixel este mai aproape de camera (z-buffer)
                            if z > z_buffer[iso_y, iso_x]:
                                # Calculăm normala pentru fața laterală
                                # Pentru fețele laterale, normala depinde de direcția peretelui
                                # Folosim un vector aproximativ perpendicular pe plan
                                normal = np.array([0.5, 0.5, 0.7])
                                normal = normal / np.linalg.norm(normal)
                                
                                # Iluminare bazată pe înălțime (mai jos = mai întunecat)
                                height_factor = 1.0 - (z / max(1, wall_height_px)) * 0.3
                                light_intensity = max(0.35, np.dot(normal, light_dir) * height_factor)
                                
                                # Culoare cu gradient vertical (mai deschis)
                                base_color = int(150 + 80 * light_intensity)
                                render_3d[iso_y, iso_x] = [base_color, base_color, base_color]
                                z_buffer[iso_y, iso_x] = z
            
            # 3. Outline-uri pentru claritate (margini ușor întunecate)
            _log(f"         📐 Desenez outline-urile...")
            outline_pixels = np.where(outline > 0)
            for y, x in zip(outline_pixels[0], outline_pixels[1]):
                # Desenăm outline-ul de la bază până la vârf
                for z in range(0, wall_height_px + 1, 2):
                    iso_x, iso_y = to_iso(x, y, z)
                    if 0 <= iso_x < canvas_w and 0 <= iso_y < canvas_h:
                        if z > z_buffer[iso_y, iso_x] - 0.5:  # Toleranță pentru outline
                            # Outline moderat pentru contrast
                            outline_intensity = max(0.2, 0.6 - z / max(1, wall_height_px) * 0.25)
                            outline_color = int(70 * outline_intensity)
                            render_3d[iso_y, iso_x] = [outline_color, outline_color, outline_color]
                            z_buffer[iso_y, iso_x] = z
            
            # 4. Aplicăm anti-aliasing și smoothing pentru un aspect mai profesional
            _log(f"         🎨 Aplic smoothing ușor...")
            render_3d = cv2.GaussianBlur(render_3d, (3, 3), 0)
            
            # 5. Crop la zona non-zero
            non_zero = np.where(np.any(render_3d > 10, axis=2))  # Threshold pentru a exclude negrul complet
            if len(non_zero[0]) > 0:
                y_min, y_max = max(0, non_zero[0].min() - 10), min(canvas_h, non_zero[0].max() + 11)
                x_min, x_max = max(0, non_zero[1].min() - 10), min(canvas_w, non_zero[1].max() + 11)
                render_3d = render_3d[y_min:y_max, x_min:x_max]
            
            # 6. Ajustăm contrastul pentru un aspect mai clar
            render_3d = cv2.convertScaleAbs(render_3d, alpha=1.2, beta=10)
            
            output_path = output_dir / "04_walls_3d.png"
            cv2.imwrite(str(output_path), render_3d)
            _log(f"      💾 Salvat: 04_walls_3d.png ({len(wall_pixels[0])} pixeli pereți extruși, înălțime {wall_height_px}px)")
            
            # ✅ Notificare UI imediat după generarea fișierului 3D
            if output_path.exists():
                notify_ui("scale", output_path)
                _log(f"      📢 Notificat UI pentru 04_walls_3d.png")
            rendering_success = True
    except Exception as e:
        import traceback
        _log(f"         ⚠️ Eroare la randarea 3D: {e}")
        _log(f"         ⚠️ Continuăm cu workflow-ul (randarea 3D nu este critică)...")
        traceback.print_exc()
        rendering_success = False
    
    # 9. Generăm interiorul casei (pixelii negri din flood fill devin portocaliu) -> 09_interior.png
    _log(f"      🏠 Generez interiorul casei...")
    interior_img = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    
    # Pixelii negri (nu au fost atinși de flood fill) devin portocaliu
    interior_mask = (flood_fill_result == 0) & (walls_thick == 0)
    interior_img[interior_mask] = [0, 165, 255]  # BGR: portocaliu
    
    cv2.imwrite(str(output_dir / "09_interior.png"), interior_img)
    _log(f"      💾 Salvat: 09_interior.png")
    _tick()

    # 10. Flood fill pe structura inițială (01_walls_from_coords) -> 10_flood_structure.png
    _log(f"      🌊 Aplic flood fill pe structura inițială...")
    
    # ✅ Pentru 11_interior_structure.png avem nevoie de grosime 1px, deci folosim accepted_wall_segments_mask
    # Aceasta este masca cu segmentele acceptate cu grosime 1px (folosită și pentru 01_walls_from_coords.png)
    # Dacă accepted_wall_segments_mask nu este disponibilă, folosim walls_overlay_mask ca fallback
    if 'accepted_wall_segments_mask' in locals() and accepted_wall_segments_mask is not None:
        walls_mask_for_flood = accepted_wall_segments_mask.copy()
    else:
        walls_mask_for_flood = walls_overlay_mask if walls_overlay_mask is not None else walls_mask
    
    flood_structure_image = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    flood_structure_mask = np.zeros((h_orig + 2, w_orig + 2), dtype=np.uint8)
    
    # Creăm o imagine pentru flood fill (spațiile libere sunt 255, pereții sunt 0)
    flood_structure_base = (255 - walls_mask_for_flood).astype(np.uint8)
    
    # Flood fill din colțuri cu culoarea albastră
    corners = [(0, 0), (w_orig-1, 0), (0, h_orig-1), (w_orig-1, h_orig-1)]
    for x, y in corners:
        if walls_mask_for_flood[y, x] == 0:  # Nu este perete
            cv2.floodFill(flood_structure_base, flood_structure_mask, (x, y), 128, 
                         loDiff=(0,), upDiff=(0,), flags=4)
    
    # Rezultatul flood fill: pixelii cu valoarea 128 sunt cei atinși de flood fill
    flood_structure_result = (flood_structure_base == 128).astype(np.uint8) * 255
    
    # Găsim pixelii albi (pereți) care au vecini cu flood fill și îi marcăm ca roșii
    walls_touching_flood = np.zeros((h_orig, w_orig), dtype=np.uint8)
    for y in range(h_orig):
        for x in range(w_orig):
            if walls_mask_for_flood[y, x] > 0:  # Este perete (alb)
                # Verificăm vecinii (8-conectivitate)
                touches_flood = False
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h_orig and 0 <= nx < w_orig:
                            if flood_structure_result[ny, nx] > 0:
                                touches_flood = True
                                break
                    if touches_flood:
                        break
                if touches_flood:
                    walls_touching_flood[y, x] = 255
    
    # Creăm imaginea cu flood fill albastru și pereții roșii
    # Zonele atinse de flood fill devin albastre
    flood_structure_image[flood_structure_result > 0] = [255, 200, 100]  # BGR: albastru deschis
    # Pereții care ating flood fill devin roșii
    flood_structure_image[walls_touching_flood > 0] = [0, 0, 255]  # BGR: roșu
    # Pereții care nu ating flood fill rămân albi
    # ✅ Folosim walls_mask_for_flood (walls_overlay_mask) pentru consistență
    walls_not_touching = (walls_mask_for_flood > 0) & (walls_touching_flood == 0)
    flood_structure_image[walls_not_touching] = [255, 255, 255]  # BGR: alb
    
    flood_structure_path = output_dir / "10_flood_structure.png"
    cv2.imwrite(str(flood_structure_path), flood_structure_image)
    _log(f"      💾 Salvat: 10_flood_structure.png")
    _tick()
    
    # ✅ Notificarea UI pentru scale_flood se face în batch din orchestrator după ce toate planurile sunt gata.
    
    # 11. Structura pereților interiori (pixelii albi care nu au vecini cu flood fill) -> 11_interior_structure.png
    _log(f"      🏗️ Generez structura pereților interiori...")
    interior_structure = np.zeros((h_orig, w_orig), dtype=np.uint8)
    
    # Pixelii albi (pereți) care NU au vecini cu flood fill
    interior_structure[walls_not_touching] = 255
    
    cv2.imwrite(str(output_dir / "11_interior_structure.png"), interior_structure)
    _log(f"      💾 Salvat: 11_interior_structure.png")
    
    # 12. Structura pereților exteriori (pixelii care au devenit roșii) -> 12_exterior_structure.png
    _log(f"      🏗️ Generez structura pereților exteriori...")
    exterior_structure = np.zeros((h_orig, w_orig), dtype=np.uint8)
    
    # Pixelii care au devenit roșii (dar albi în poza finală)
    exterior_structure[walls_touching_flood > 0] = 255
    
    cv2.imwrite(str(output_dir / "12_exterior_structure.png"), exterior_structure)
    _log(f"      💾 Salvat: 12_exterior_structure.png")
    
    # 13. Procesăm doors (openings) -> openings/
    _log(f"      🚪 Procesez deschideri (doors) din RasterScan...")
    openings_dir = output_dir.parent / "openings"
    openings_dir.mkdir(parents=True, exist_ok=True)

    # (Filtru flood-fill dezactivat: păstrăm toate ușile/ferestrele din RasterScan.)

    # Prompt pentru clasificarea doors – ferestre (inclusiv verticale), scări, garaj
    DOOR_CLASSIFICATION_PROMPT = """Analyze this floor plan crop. Classify the opening as exactly ONE type. Do NOT default to "door".

- "window" = Any glass/fenestration: parallel lines, grid, or strip (horizontal OR vertical). Narrow vertical strip = window. Horizontal strip = window. Use "window" whenever you see typical window depiction (Fenster, geam).
- "door" = Passage for people (single/double leaf, swing arc). Use "door" for normal doors and for very wide openings (garage) – user sets garage in the app.
- "stairs" = Staircase (Treppe): visible step pattern (parallel lines up/down). If you see steps, answer "stairs".

Rule: Lines/grid or strip shape → "window". Steps visible → "stairs". Otherwise → "door". Do NOT use garage_door.
Respond ONLY with JSON: {"type": "window"} or {"type": "door"} or {"type": "stairs"}"""

    # Verificare Gemini pentru deschideri rămase "door" (evită scări/geamuri trecute ca uși). Fără garage – îl pune userul.
    DOOR_VERIFY_PROMPT = """This opening was classified as a door. Look again: could it be a window (glass/strip/lines) or stairs (step pattern)? Reply ONLY with JSON: {"type": "door"} or {"type": "window"} or {"type": "stairs"}."""

    # Stocăm toate openings-urile pentru generarea imaginilor
    openings_list = []

    def _bbox_iou(a, b):
        """IoU între două bbox (x_min, y_min, x_max, y_max)."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    if 'doors' in data and data['doors']:
        _log(f"      📐 Procesez {len(data['doors'])} deschideri...")
        door_items = []  # list of (idx, temp_path, room_name, door_crop, x_min, y_min, x_max, y_max, door_crop_small)
        kept_bboxes = []  # fără suprapuneri: excludem deschideri care se suprapun peste altele
        for idx, door in enumerate(data['doors']):
            if 'bbox' not in door or len(door['bbox']) != 4:
                continue
            bbox_api = door['bbox']
            x1_api, y1_api, x2_api, y2_api = bbox_api
            x1_orig, y1_orig = api_to_original_coords(x1_api, y1_api)
            x2_orig, y2_orig = api_to_original_coords(x2_api, y2_api)
            x_min = max(0, min(x1_orig, x2_orig))
            y_min = max(0, min(y1_orig, y2_orig))
            x_max = min(w_orig, max(x1_orig, x2_orig))
            y_max = min(h_orig, max(y1_orig, y2_orig))
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            padding = min(20, max(5, int(min(bbox_width, bbox_height) * 0.1)))
            x_min_crop = max(0, x_min - padding)
            y_min_crop = max(0, y_min - padding)
            x_max_crop = min(w_orig, x_max + padding)
            y_max_crop = min(h_orig, y_max + padding)
            if x_max_crop <= x_min_crop or y_max_crop <= y_min_crop:
                continue
            door_crop = original_img[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
            if door_crop.size == 0:
                continue
            room_name = "Unknown"
            if 'rooms' in data and data['rooms']:
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                for room_idx, room_poly in enumerate(rooms_polygons[:len(data['rooms'])]):
                    if cv2.pointPolygonTest(room_poly, (center_x, center_y), False) >= 0:
                        if room_idx < len(data['rooms']):
                            room_data = data['rooms'][room_idx]
                            if isinstance(room_data, dict):
                                room_name = room_data.get('name', room_data.get('label', 'Unknown'))
                            else:
                                room_name = 'Unknown'
                        break
            # Fără suprapuneri: excludem deschideri care se suprapun peste altele (IoU > 15%)
            bbox = (x_min, y_min, x_max, y_max)
            if any(_bbox_iou(bbox, k) > 0.15 for k in kept_bboxes):
                _log(f"         ⏭️ Deschidere {idx} ignorată (suprapunere cu altă deschidere)")
                continue
            kept_bboxes.append(bbox)

            temp_crop_path = openings_dir / f"door_{idx}_temp.png"
            cv2.imwrite(str(temp_crop_path), door_crop)
            door_crop_small = original_img[y_min:y_max, x_min:x_max]
            if door_crop_small.size == 0:
                door_crop_small = door_crop
            door_items.append((idx, str(temp_crop_path), room_name, door_crop, x_min, y_min, x_max, y_max, door_crop_small))

        batch_types = None
        if door_items and gemini_api_key:
            paths = [item[1] for item in door_items]
            batch_types = call_gemini_doors_batch(paths, gemini_api_key)

        for pos, (idx, temp_crop_path, room_name, door_crop, x_min, y_min, x_max, y_max, door_crop_small) in enumerate(door_items):
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            door_type = "door"
            if batch_types and pos < len(batch_types):
                door_type = batch_types[pos]
                if door_type == "garage_door":
                    door_type = "door"  # nu detectăm garaj – îl pune userul în editor
            elif gemini_api_key:
                try:
                    from .scale_detection import call_gemini as gemini_classify
                    context_prompt = DOOR_CLASSIFICATION_PROMPT
                    if room_name != "Unknown":
                        context_prompt += f"\n\nContext: This opening is located in the '{room_name}' room."
                    result = gemini_classify(temp_crop_path, context_prompt, gemini_api_key)
                    if result and isinstance(result, dict) and 'type' in result:
                        door_type = result['type'].lower().strip()
                    if door_type == "garage_door":
                        door_type = "door"  # nu detectăm garaj – îl pune userul
                    if door_type not in ['door', 'window', 'garage_door', 'stairs']:
                        door_type = 'door'
                except Exception as e:
                    _log(f"         ⚠️ Eroare clasificare door {idx}: {e}")
            if door_type not in ['door', 'window', 'garage_door', 'stairs']:
                door_type = 'door'
            if door_type == "garage_door":
                door_type = "door"  # garaj doar din editor
            # Euristică: bandă orizontală sau verticală (ferestre) → window
            if door_type == 'door' and bbox_w > 0 and bbox_h > 0:
                if _is_window_by_aspect(float(bbox_w), float(bbox_h)):
                    door_type = 'window'

            # Verificare Gemini: deschideri rămase "door" pot fi de fapt geamuri/scări/garaj
            if door_type == 'door' and gemini_api_key and Path(temp_crop_path).exists():
                try:
                    from .scale_detection import call_gemini as gemini_verify
                    result = gemini_verify(temp_crop_path, DOOR_VERIFY_PROMPT, gemini_api_key)
                    if result and isinstance(result, dict) and result.get('type'):
                        t = str(result['type']).lower().strip()
                        if t in ('door', 'window', 'garage_door', 'stairs'):
                            door_type = t
                        if door_type == "garage_door":
                            door_type = "door"  # nu detectăm garaj – îl pune userul
                except Exception as e:
                    _log(f"         ⚠️ Verificare Gemini door {idx}: {e}")

            if Path(temp_crop_path).exists():
                Path(temp_crop_path).unlink()

            door_crop_with_text = door_crop_small.copy()
            text = f"{door_type.upper()}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(door_crop_with_text,
                          (5, 5),
                          (text_width + 15, text_height + baseline + 15),
                          (0, 0, 0), -1)
            cv2.putText(door_crop_with_text, text,
                        (10, text_height + 10),
                        font, font_scale, (255, 255, 255), thickness)
            door_path = openings_dir / f"door_{idx}_{door_type}.png"
            cv2.imwrite(str(door_path), door_crop_with_text)
            _log(f"         💾 Salvat: door_{idx}_{door_type}.png")
            openings_list.append({
                'idx': idx,
                'type': door_type,
                'bbox': (x_min, y_min, x_max, y_max),
                'center': ((x_min + x_max) // 2, (y_min + y_max) // 2)
            })
    
    # Salvare tipuri uși/geamuri pentru detections_review – index = poziția în data['doors'] (obligatoriu 1:1)
    num_doors = len(data.get("doors") or [])
    if num_doors > 0:
        doors_types_path = raster_dir / "doors_types.json"
        try:
            types_list = [{"type": "door"} for _ in range(num_doors)]
            for op in openings_list:
                idx = op.get("idx", -1)
                if 0 <= idx < num_doors:
                    types_list[idx] = {"type": op["type"]}
            with open(doors_types_path, "w", encoding="utf-8") as f:
                json.dump(types_list, f, indent=2)
            _log(f"      💾 Salvat: {doors_types_path.name} ({num_doors} deschideri, pentru review UI)")
            # Regenerăm doors.png și overlay_on_original.png cu tipurile Gemini
            try:
                from .raster_api import regenerate_doors_and_overlay_from_doors_types
                if regenerate_doors_and_overlay_from_doors_types(raster_dir, original_img):
                    _log(f"      📄 Regenerat doors.png și overlay_on_original.png (tipuri Gemini)")
            except Exception as e:
                _log(f"      ⚠️ Regenerare doors/overlay: {e}")
        except Exception as e:
            _log(f"      ⚠️ Nu s-a putut scrie {doors_types_path.name}: {e}")
    
    # Generăm 01_openings.png - planul cu toate openings-urile colorate
    if openings_list:
        _log(f"      🎨 Generez 01_openings.png...")
        openings_img = original_img.copy()
        
        for opening in openings_list:
            x_min, y_min, x_max, y_max = opening['bbox']
            door_type = opening['type']
            
            # Culori pentru fiecare tip
            if door_type == 'window':
                color = [255, 0, 0]  # BGR: albastru
            elif door_type == 'door':
                color = [0, 255, 0]  # BGR: verde
            elif door_type == 'stairs':
                color = [203, 192, 255]  # BGR: roz
            elif door_type == 'garage_door':
                color = [128, 0, 128]  # BGR: mov
            else:
                color = [255, 255, 255]  # BGR: alb (default)
            
            # Desenăm dreptunghiul
            cv2.rectangle(openings_img, (x_min, y_min), (x_max, y_max), color, 3)
        
        openings_img_path = openings_dir / "01_openings.png"
        cv2.imwrite(str(openings_img_path), openings_img)
        _log(f"         💾 Salvat: 01_openings.png")
        
        # Notificarea UI pentru detections se face în batch din orchestrator după ce toate planurile sunt gata.
        
        # Generăm 02_exterior_doors.png - ușile interioare (verde) și exterioare (roșu)
        _log(f"      🎨 Generez 02_exterior_doors.png...")
        exterior_doors_img = original_img.copy()
        
        # Verificăm pentru fiecare opening dacă este exterior (atinge flood fill)
        for opening in openings_list:
            if opening['type'] != 'door' and opening['type'] != 'garage_door':
                continue  # Doar ușile
            
            x_min, y_min, x_max, y_max = opening['bbox']
            center_x, center_y = opening['center']
            
            # Verificăm dacă centrul sau bbox-ul atinge flood fill (exterior)
            is_exterior = False
            # Verificăm centrul și colțurile
            check_points = [
                (center_x, center_y),
                (x_min, y_min),
                (x_max, y_min),
                (x_min, y_max),
                (x_max, y_max)
            ]
            
            for px, py in check_points:
                if 0 <= py < h_orig and 0 <= px < w_orig:
                    if flood_fill_result[py, px] > 0:
                        is_exterior = True
                        break
                    # Verificăm și vecinii
                    for dy in [-2, -1, 0, 1, 2]:
                        for dx in [-2, -1, 0, 1, 2]:
                            ny, nx = py + dy, px + dx
                            if 0 <= ny < h_orig and 0 <= nx < w_orig:
                                if flood_fill_result[ny, nx] > 0:
                                    is_exterior = True
                                    break
                        if is_exterior:
                            break
                if is_exterior:
                    break
            
            # Colorăm: verde pentru interior, roșu pentru exterior
            if is_exterior:
                color = [0, 0, 255]  # BGR: roșu (exterior)
            else:
                color = [0, 255, 0]  # BGR: verde (interior)
            
            # Desenăm dreptunghiul
            cv2.rectangle(exterior_doors_img, (x_min, y_min), (x_max, y_max), color, 3)
            
            # Adăugăm statusul în opening pentru a fi salvat în openings_measurements.json
            opening['status'] = 'exterior' if is_exterior else 'interior'
        
        exterior_doors_img_path = openings_dir / "02_exterior_doors.png"
        cv2.imwrite(str(exterior_doors_img_path), exterior_doors_img)
        _log(f"         💾 Salvat: 02_exterior_doors.png")
        
        # Notificarea UI pentru exterior_doors se face în batch din orchestrator după ce toate planurile sunt gata.
    
    # 14. Suprapunem rooms cu casa și calculăm metri per pixel (păstrăm doar pozele legate de room)
    _log(f"      📏 Calculez metri per pixel pentru fiecare cameră...")
    
    room_scales = {}
    total_area_m2 = 0.0
    total_area_px = 0
    
    # Funcție care construiește doar crop-ul (pentru batch Gemini)
    def build_room_crop_for_batch(i, room, room_poly, walls_mask_validated):
        """Returnează (i, room_area_px, room_mask_crop, room_crop, crop_path) sau None."""
        if i >= len(rooms_polygons):
            return None
        room_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        cv2.fillPoly(room_mask, [room_poly], 255)
        room_area_px = np.count_nonzero(room_mask)
        if room_area_px < 100:
            return None
        x, y, w, h = cv2.boundingRect(room_poly)
        x = max(0, x - 10)
        y = max(0, y - 10)
        w = min(w_orig - x, w + 20)
        h = min(h_orig - y, h + 20)
        room_crop = original_img[y:y+h, x:x+w].copy()
        if room_crop.size == 0:
            return None
        room_mask_crop = np.zeros((h, w), dtype=np.uint8)
        if len(room_poly.shape) == 3 and room_poly.shape[1] == 1:
            room_poly_normalized = room_poly.reshape(-1, 2)
        else:
            room_poly_normalized = room_poly.copy()
        room_poly_crop = room_poly_normalized.copy()
        room_poly_crop[:, 0] -= x
        room_poly_crop[:, 1] -= y
        cv2.fillPoly(room_mask_crop, [room_poly_crop], 255)
        room_crop[room_mask_crop == 0] = [0, 0, 0]
        walls_mask_crop = walls_mask_validated[y:y+h, x:x+w].copy()
        walls_colored_crop = cv2.cvtColor(walls_mask_crop, cv2.COLOR_GRAY2BGR)
        walls_colored_crop[walls_mask_crop > 0] = [0, 0, 255]
        room_crop = cv2.addWeighted(room_crop, 0.8, walls_colored_crop, 0.2, 0)
        room_location_img = original_img.copy()
        if walls_mask_validated is not None:
            walls_colored_loc = cv2.cvtColor(walls_mask_validated, cv2.COLOR_GRAY2BGR)
            walls_colored_loc[walls_mask_validated > 0] = [0, 0, 255]
            room_location_img = cv2.addWeighted(room_location_img, 0.7, walls_colored_loc, 0.3, 0)
        room_colored_mask = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
        room_colored_mask[room_mask > 0] = [0, 255, 255]
        room_location_img = cv2.addWeighted(room_location_img, 0.5, room_colored_mask, 0.5, 0)
        cv2.imwrite(str(output_dir / f"room_{i}_location.png"), room_location_img)
        crop_path = output_dir / f"room_{i}_temp_for_gemini.png"
        cv2.imwrite(str(crop_path), room_crop)
        return (i, int(room_area_px), room_mask[y:y+h, x:x+w].copy(), room_crop, str(crop_path))

    def process_room_for_scale(i, room, room_poly, walls_mask_validated):
        """Apel per-cameră Gemini (folosit la fallback când batch eșuează)."""
        built = build_room_crop_for_batch(i, room, room_poly, walls_mask_validated)
        if built is None:
            return None
        i, room_area_px, room_mask_crop, room_crop, crop_path = built
        result_data = None
        if gemini_api_key:
            try:
                result = call_gemini(crop_path, GEMINI_PROMPT_CROP, gemini_api_key)
                if result and is_informational_total_result(result):
                    result = None
                if result and result.get('area_m2') is not None:
                    area_m2 = float(result['area_m2'])
                    if area_m2 > 0:
                        result_data = {
                            'idx': i,
                            'area_m2': float(area_m2),
                            'area_px': int(room_area_px),
                            'room_name': result.get('room_name', f'Room_{i}'),
                            'room_crop': room_crop,
                            'room_mask': room_mask_crop
                        }
            except Exception as e:
                _log(f"         Camera {i}: Eroare Gemini: {e}")
        if result_data:
            cv2.imwrite(str(output_dir / f"room_{i}_crop.png"), result_data['room_crop'])
            cv2.imwrite(str(output_dir / f"room_{i}_mask.png"), result_data['room_mask'])
        return result_data
    
    # ✅ Regiuni pentru crop + Gemini: folosim 09_interior.png – fiecare zonă portocalie = o cameră.
    # Mapare etichete: rooms_polygons[i] = camera i → trimitem room_i_crop la Gemini → răspunsul batch[pos]
    # corespunde aceluiași i (valid_entries conține (idx, (i, ...)) în ordine). Eticheta Gemini se aplică
    # la room_scales[i]['room_type']. Ordinea trebuie identică: 09_interior, rooms_polygons, batch, room_scales.
    # Încărcăm fișierul 09_interior.png și extragem toate componentele portocalii (exact ce vede utilizatorul).
    def _touches_image_border(comp_mask: np.ndarray, h_f: int, w_f: int) -> bool:
        if np.any(comp_mask[0, :] > 0) or np.any(comp_mask[h_f - 1, :] > 0):
            return True
        if np.any(comp_mask[:, 0] > 0) or np.any(comp_mask[:, w_f - 1] > 0):
            return True
        return False

    room_region_polygons = []
    interior_png_path = output_dir / "09_interior.png"
    # Când camerele vin din response.json (ex. după edit în UI), nu suprascriem cu 09_interior.png vechi
    if not rooms_from_response and interior_png_path.exists():
        try:
            img_09 = cv2.imread(str(interior_png_path), cv2.IMREAD_COLOR)
            if img_09 is not None and img_09.shape[:2] == (h_orig, w_orig):
                # Portocaliu în 09_interior: BGR [0, 165, 255] – toleranță pentru salvare/compresie
                low = np.array([0, 140, 240], dtype=np.uint8)
                high = np.array([40, 200, 255], dtype=np.uint8)
                orange_mask = cv2.inRange(img_09, low, high)
                # 4-conectivitate: zone care se ating doar pe diagonală rămân camere separate
                num_labels, labels, _, _ = cv2.connectedComponentsWithStats(orange_mask, connectivity=4)
                total_px = h_orig * w_orig
                for label_id in range(1, num_labels):
                    comp_mask = (labels == label_id).astype(np.uint8) * 255
                    if _touches_image_border(comp_mask, h_orig, w_orig):
                        continue
                    area = np.count_nonzero(comp_mask)
                    if area < 100:
                        continue
                    if total_px > 0 and area >= 0.95 * total_px:
                        continue
                    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        if len(largest) >= 3:
                            room_region_polygons.append(largest)
                if room_region_polygons:
                    rooms_polygons = room_region_polygons
                    _log(f"      📐 [09_interior.png] {len(rooms_polygons)} camere (zone portocalii din fișier) → crop + Gemini")
            _tick()
        except Exception as e:
            _log(f"      ⚠️ [09_interior.png] Eroare la citire/componente: {e}, încerc fallback interior_mask")
    
    # Fallback: masca interior din memorie (interior_mask) – aceeași logică ca înainte (nu când avem camere din response)
    if not rooms_from_response and not room_region_polygons:
        try:
            im = interior_mask
            if im is not None and im.ndim >= 2:
                im_2d = np.max(im, axis=-1).astype(np.uint8) if im.ndim == 3 else np.asarray(im).astype(np.uint8)
                if im_2d.shape[:2] == (h_orig, w_orig):
                    interior_uint8 = (im_2d > 0).astype(np.uint8) * 255
                    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(interior_uint8, connectivity=4)
                    total_px = h_orig * w_orig
                    for label_id in range(1, num_labels):
                        comp_mask = (labels == label_id).astype(np.uint8) * 255
                        if _touches_image_border(comp_mask, h_orig, w_orig):
                            continue
                        area = np.count_nonzero(comp_mask)
                        if area < 100:
                            continue
                        if total_px > 0 and area >= 0.95 * total_px:
                            continue
                        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            largest = max(contours, key=cv2.contourArea)
                            if len(largest) >= 3:
                                room_region_polygons.append(largest)
                    if room_region_polygons:
                        rooms_polygons = room_region_polygons
                        _log(f"      📐 [09_interior fallback] {len(rooms_polygons)} camere (interior_mask) → crop + Gemini")
        except NameError:
            pass
        except Exception as e:
            _log(f"      ⚠️ [09_interior fallback] Eroare: {e}")

    if not rooms_from_response and not room_region_polygons and accepted_wall_segments_mask is not None:
        wall_for_free = accepted_wall_segments_mask
        if wall_for_free.ndim == 3:
            wall_for_free = cv2.cvtColor(wall_for_free, cv2.COLOR_BGR2GRAY)
        free_space_mask = (255 - (wall_for_free > 0).astype(np.uint8) * 255).astype(np.uint8)
        # 4-conectivitate: zone care se ating doar pe diagonală rămân separate
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(free_space_mask, connectivity=4)
        h_f, w_f = free_space_mask.shape[:2]
        total_px = h_f * w_f
        for label_id in range(1, num_labels):
            comp_mask = (labels == label_id).astype(np.uint8) * 255
            if _touches_image_border(comp_mask, h_f, w_f):
                continue
            area = np.count_nonzero(comp_mask)
            if area < 100:
                continue
            if total_px > 0 and area >= 0.95 * total_px:
                continue
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if len(largest) >= 3:
                    room_region_polygons.append(largest)
        if room_region_polygons:
            rooms_polygons = room_region_polygons
            _log(f"      📐 [flood-fill regiuni] {len(rooms_polygons)} zone închise (nu ating marginea) → crop + Gemini")

    # Numerotare camere pe o COPIE 09_interior_annotated.png (09_interior rămâne curat pentru extragere poligoane)
    if len(rooms_polygons) > 0 and interior_png_path.exists():
        try:
            img_09_clean = cv2.imread(str(interior_png_path), cv2.IMREAD_COLOR)
            if img_09_clean is not None and img_09_clean.shape[:2] == (h_orig, w_orig):
                img_09_annotated = img_09_clean.copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                for idx, poly in enumerate(rooms_polygons):
                    M = cv2.moments(np.asarray(poly, dtype=np.float32))
                    if M["m00"] and M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        label = str(idx)
                        scale = max(1.2, min(3.0, h_orig / 400))
                        thick = max(2, int(scale * 2))
                        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
                        x1 = max(0, cx - tw // 2 - 8)
                        y1 = max(0, cy - th // 2 - 8)
                        x2 = min(w_orig, cx + tw // 2 + 8)
                        y2 = min(h_orig, cy + th // 2 + 8)
                        cv2.rectangle(img_09_annotated, (x1, y1), (x2, y2), (0, 0, 0), -1)
                        cv2.rectangle(img_09_annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        cv2.putText(img_09_annotated, label, (x1 + 4, y2 - 4), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
                interior_annotated_path = output_dir / "09_interior_annotated.png"
                cv2.imwrite(str(interior_annotated_path), img_09_annotated)
                _log(f"      🔢 Numerotat 09_interior_annotated.png: {len(rooms_polygons)} camere (0..{len(rooms_polygons)-1}); 09_interior.png rămâne curat")
        except Exception as e:
            _log(f"      ⚠️ Numerotare 09_interior_annotated: {e}")

    # Procesăm camerele în paralel (mereu din rooms_polygons = regiuni flood-fill când avem mască)
    room_tasks = []
    if len(rooms_polygons) > 0:
        _log(f"      🔍 Procesez {len(rooms_polygons)} regiuni pentru calcularea scale-ului (crop + Gemini)...")
        room_tasks = [
            (i, {'type': 'flood_fill_region', 'id': i}, rooms_polygons[i])
            for i in range(len(rooms_polygons))
        ]
    
    # Camere care acoperă (aproape) tot planul = flood fill eșuat → nu le folosim la total_area / scală
    MAX_ROOM_COVERAGE_RATIO = 0.95
    img_total_px = h_orig * w_orig

    if room_tasks:
        
        # ✅ Verificăm suprapunerea între camere și eliminăm duplicatele (>70% suprapunere)
        # ✅ Excludem camerele prea mari (flood fill eșuat: o cameră = tot ecranul)
        _log(f"      🔍 Verific suprapunerea între camere și exclud camerele prea mari (flood fill eșuat)...")
        room_masks = {}
        rooms_to_process = []
        overlap_skipped_indices = set()
        flood_fill_skipped_indices = set()

        for i, room, room_poly in room_tasks:
            # Creăm masca pentru această cameră
            room_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
            cv2.fillPoly(room_mask, [room_poly], 255)

            room_area_px = np.count_nonzero(room_mask)
            if img_total_px > 0 and room_area_px / img_total_px > MAX_ROOM_COVERAGE_RATIO:
                _log(f"         ⚠️ Camera {i}: acoperă {100 * room_area_px / img_total_px:.1f}% din plan (flood fill eșuat) → exclud din calcul")
                flood_fill_skipped_indices.add(i)
                continue

            room_masks[i] = room_mask
            
            # Verificăm suprapunerea cu camerele deja procesate
            should_process = True
            for j, other_mask in room_masks.items():
                if j == i:
                    continue
                
                # Calculăm suprapunerea (Intersection over Union - IoU)
                intersection = np.logical_and(room_mask, other_mask)
                union = np.logical_or(room_mask, other_mask)
                
                intersection_area = np.count_nonzero(intersection)
                union_area = np.count_nonzero(union)
                
                if union_area > 0:
                    iou = intersection_area / union_area
                    # Dacă suprapunerea este > 70%, skip această cameră complet (nu o folosim deloc)
                    if iou > 0.70:
                        _log(f"         ⚠️ Camera {i} are suprapunere {iou*100:.1f}% cu Camera {j} -> skip complet (suprafață 0)")
                        should_process = False
                        overlap_skipped_indices.add(i)
                        break
            
            if should_process:
                rooms_to_process.append((i, room, room_poly))
        
        _log(f"      ✅ {len(rooms_to_process)} camere unice pentru procesare (din {len(room_tasks)} total, {len(room_tasks) - len(rooms_to_process)} skip-uite)")
        
        walls_mask_for_rooms = walls_overlay_mask if walls_overlay_mask is not None else walls_mask
        max_workers = max(1, min(4, len(rooms_to_process)))
        ordered_crop_results = [None] * len(rooms_to_process)  # idx -> (i, room_area_px, room_mask_crop, room_crop, crop_path)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(build_room_crop_for_batch, i, room, room_poly, walls_mask_for_rooms): idx
                for idx, (i, room, room_poly) in enumerate(rooms_to_process)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    ordered_crop_results[idx] = future.result()
                except Exception:
                    ordered_crop_results[idx] = None

        valid_entries = [(idx, t) for idx, t in enumerate(ordered_crop_results) if t is not None]
        paths = [t[4] for _, t in valid_entries]
        # Copii cu area_px desenat în colț (pentru asociere label după răspuns Gemini), apoi redimensionate (max 512px) pentru batch
        paths_for_batch = []
        MAX_BATCH_CROP_PX = 512
        for (idx, t) in valid_entries:
            i, room_area_px, room_mask_crop, room_crop, crop_path = t
            img = cv2.imread(str(crop_path))
            if img is not None:
                # area_px + room_no (același număr ca pe 09_interior) – Gemini returnează room_number pentru asociere OCR
                label1 = f"area_px:{int(room_area_px)}"
                label2 = f"room_no:{i}"
                h_c, w_c = img.shape[:2]
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = max(0.4, min(1.2, w_c / 400))
                thick = max(1, int(scale * 2))
                (tw1, th1), _ = cv2.getTextSize(label1, font, scale, thick)
                (tw2, th2), _ = cv2.getTextSize(label2, font, scale, thick)
                x0, y0 = 8, 20 + th1
                cv2.rectangle(img, (x0 - 2, y0 - th1 - 2), (x0 + max(tw1, tw2) + 2, y0 + 2), (0, 0, 0), -1)
                cv2.putText(img, label1, (x0, y0), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
                y0 += th1 + 4
                cv2.rectangle(img, (x0 - 2, y0 - th2 - 2), (x0 + tw2 + 2, y0 + 2), (0, 0, 0), -1)
                cv2.putText(img, label2, (x0, y0), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
                if max(h_c, w_c) > MAX_BATCH_CROP_PX:
                    scale_b = MAX_BATCH_CROP_PX / max(h_c, w_c)
                    w_b = max(1, int(round(w_c * scale_b)))
                    h_b = max(1, int(round(h_c * scale_b)))
                    img = cv2.resize(img, (w_b, h_b), interpolation=cv2.INTER_AREA)
            batch_path = output_dir / f"room_{i}_batch.png"
            if img is not None:
                cv2.imwrite(str(batch_path), img)
                paths_for_batch.append(batch_path)
            else:
                paths_for_batch.append(Path(crop_path) if not isinstance(crop_path, Path) else crop_path)
        # Camere care au intrat în batch (trimise la Gemini)
        sent_room_indices = [t[0] for _, t in valid_entries]
        # Camere care nu au intrat în batch (crop build a returnat None)
        not_sent_room_indices = [i for idx, (i, _, _) in enumerate(rooms_to_process) if ordered_crop_results[idx] is None]
        if not_sent_room_indices:
            _log(f"         ⚠️ Camere neincluse în batch (crop eșuat / prea mici): {not_sent_room_indices}")
        if sent_room_indices:
            _log(f"         📤 Trimise la Gemini per crop: {len(sent_room_indices)} camere (indici: {sent_room_indices})")
        _tick()
        batch_results = None
        if gemini_api_key and paths_for_batch and len(paths_for_batch) == len(valid_entries):
            batch_results = call_gemini_rooms_per_crop([str(p) for p in paths_for_batch], gemini_api_key)
        _tick()
        if batch_results is None or len(batch_results) != len(paths_for_batch):
            batch_results = None

        if batch_results is not None:
            for (pos, (idx, (i, room_area_px, room_mask_crop, room_crop, crop_path))) in enumerate(valid_entries):
                r = batch_results[pos] if pos < len(batch_results) else None
                room_name = (r.get('room_name') or f'Room_{i}').strip() if r else f'Room_{i}'
                room_type = (r.get('room_type') or 'Raum').strip() if r else 'Raum'
                area_m2 = None
                if r and r.get('area_m2') is not None:
                    try:
                        area_m2 = float(r['area_m2'])
                    except (TypeError, ValueError):
                        area_m2 = 0.0
                if area_m2 is not None and area_m2 > 0 and not is_informational_total_result(dict(room_name=room_name, area_m2=area_m2)):
                    area_px_stored = int(room_area_px)
                    cv2.imwrite(str(output_dir / f"room_{i}_crop.png"), room_crop)
                    cv2.imwrite(str(output_dir / f"room_{i}_mask.png"), room_mask_crop)
                    m_px_room = float(np.sqrt(area_m2 / room_area_px)) if room_area_px > 0 else None
                    room_scales[i] = {
                        'room_number': i,
                        'room_name': room_name,
                        'room_type': room_type,
                        'area_m2': area_m2,
                        'area_px': area_px_stored,
                        'm_px': m_px_room,
                        'crop_image': f'room_{i}_crop.png',
                    }
                    total_area_m2 += area_m2
                    total_area_px += room_area_px
                    _log(f"         Camera {i} ({room_name}): {area_m2:.2f} m², {room_area_px} px → {m_px_room:.9f} m/px")
                else:
                    area_px_est = int(room_area_px) if room_area_px else (int(np.count_nonzero(room_masks.get(i))) if i in room_masks else 0)
                    room_scales[i] = {
                        'room_number': i,
                        'room_name': (room_name if (r and room_name and room_name != f'Room_{i}') else 'Raum'),
                        'room_type': (room_type if (r and room_type and room_type != 'Raum') else 'Raum'),
                        'area_m2': 0.0,
                        'area_px': area_px_est,
                        'm_px': None,
                        'crop_image': f'room_{i}_temp_for_gemini.png',
                    }
                    _log(f"         ⚠️ Camera {i}: fără area_m2 valid -> etichetă 'Raum'")
            # Fără retry Gemini pentru camere eșuate (cerere utilizator)
        else:
            # Fallback: apel Gemini per cameră (crop-urile sunt deja pe disk)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_room_for_scale, i, room, room_poly, walls_mask_for_rooms): i
                    for i, room, room_poly in rooms_to_process
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        i = result['idx']
                        ar_m2, ar_px = result['area_m2'], result['area_px']
                        m_px_room = float(np.sqrt(ar_m2 / ar_px)) if ar_px > 0 else None
                        room_scales[i] = {
                            'room_number': int(result.get('room_number', i)),
                            'room_name': result['room_name'],
                            'room_type': result.get('room_type') or 'Raum',
                            'area_m2': ar_m2,
                            'area_px': int(ar_px),
                            'm_px': m_px_room,
                            'crop_image': f'room_{i}_crop.png',
                        }
                        total_area_m2 += ar_m2
                        total_area_px += ar_px
                        _log(f"         Camera {i} ({result['room_name']}): {ar_m2:.2f} m², {ar_px} px → {m_px_room:.9f} m/px")

        if batch_results is not None:
            processed_ok_indices = {t[0] for _, t in valid_entries if (room_scales.get(t[0]) or {}).get('area_m2', 0) > 0}
        else:
            processed_ok_indices = {i for i, r in room_scales.items() if r.get('area_m2', 0) > 0}
        attempted_indices = {i for i, _, _ in rooms_to_process}
        gemini_failed_indices = attempted_indices - processed_ok_indices

        # Mark rooms that had no valid crop (not in valid_entries) -> NU au fost trimise la Gemini
        for idx, (i, _, _) in enumerate(rooms_to_process):
            if ordered_crop_results[idx] is None and i not in room_scales:
                area_px_est = int(np.count_nonzero(room_masks.get(i))) if i in room_masks else 0
                room_scales[i] = {
                    'room_number': i,
                    'room_name': 'Raum',
                    'room_type': 'Raum',
                    'area_m2': 0.0,
                    'area_px': area_px_est,
                    'm_px': None,
                    'crop_image': None,
                }
                gemini_failed_indices.add(i)
                _log(f"         ⚠️ Camera {i}: crop eșuat (prea mică sau invalidă) -> nu trimis la Gemini, etichetă 'Raum'")

        for i in sorted(overlap_skipped_indices):
            room_scales[i] = {
                'room_number': i,
                'room_name': 'Raum',
                'room_type': 'Raum',
                'area_m2': 0.0,
                'area_px': 0,
                'm_px': None,
                'crop_image': None,
            }

        for i in sorted(flood_fill_skipped_indices):
            room_scales[i] = {
                'room_number': i,
                'room_name': 'Raum',
                'room_type': 'Raum',
                'area_m2': 0.0,
                'area_px': 0,
                'm_px': None,
                'crop_image': None,
            }

        for i in sorted(gemini_failed_indices):
            # Nu suprascriem camerele deja marcate ca crop_failed (nu au fost trimise)
            if room_scales.get(i, {}).get('crop_image') is None:
                continue
            area_px_est = int(np.count_nonzero(room_masks.get(i))) if i in room_masks else 0
            room_scales[i] = {
                'room_number': i,
                'room_name': 'Raum',
                'room_type': 'Raum',
                'area_m2': 0.0,
                'area_px': area_px_est,
                'm_px': None,
                'crop_image': f'room_{i}_temp_for_gemini.png',
            }
            _log(f"         ⚠️ Camera {i}: Gemini invalid -> etichetă 'Raum', nu intră la scală")
    
    # Calculăm metri per pixel global doar din camere pentru care avem area_m2 de la Gemini (nu includem arii de pixeli fără m_px)
    m_px = None
    if total_area_px > 0 and total_area_m2 > 0:
        m_px = np.sqrt(total_area_m2 / total_area_px)
        _log(f"      📏 Metri per pixel global: {m_px:.9f} m/px (total: {total_area_m2:.2f} m², {total_area_px} px, doar camere cu m_px valid)")
    
    # Calculăm măsurătorile openings (după calculul m_px)
    if m_px is not None and openings_list:
        _log(f"      📐 Calculez măsurătorile openings...")
        for opening in openings_list:
            x_min, y_min, x_max, y_max = opening['bbox']
            door_type = opening['type']
            
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            
            # Pentru uși: folosim dimensiunea mai mică (lățimea ușii)
            # Pentru geamuri: folosim dimensiunea mai mare (lungimea geamului)
            if door_type in ['door', 'garage_door']:
                size_px = min(bbox_width, bbox_height)  # Dimensiunea mai mică pentru uși
            else:  # window, stairs
                size_px = max(bbox_width, bbox_height)  # Dimensiunea mai mare pentru geamuri
            size_m = size_px * m_px
            
            opening['width_px'] = bbox_width
            opening['height_px'] = bbox_height
            opening['size_px'] = size_px
            opening['size_m'] = size_m
            
            if door_type in ['door', 'garage_door']:
                _log(f"         🚪 {door_type} {opening['idx']}: lățime = {size_m:.3f} m ({size_px} px) [min: {min(bbox_width, bbox_height)}px]")
            elif door_type == 'window':
                _log(f"         🪟 window {opening['idx']}: lungime = {size_m:.3f} m ({size_px} px) [max: {max(bbox_width, bbox_height)}px]")
    
    # Salvăm scale-urile
    # ✅ IMPORTANT: Salvăm room_scales.json chiar dacă room_scales este gol sau dacă au fost erori
    # Acest fișier este necesar pentru workflow-ul ulterior
    # Când nu avem nici o cameră (0 rooms), folosim o scară estimată ca fallback ca pipeline-ul să nu crape
    estimated_scale = False
    if total_area_m2 <= 0 or total_area_px <= 0:
        img_area_px = h_orig * w_orig
        if img_area_px > 0:
            DEFAULT_PLAN_AREA_M2 = 70.0
            total_area_m2 = float(DEFAULT_PLAN_AREA_M2)
            total_area_px = int(img_area_px)
            m_px = np.sqrt(total_area_m2 / total_area_px)
            estimated_scale = True
            _log(f"      ⚠️ 0 camere detectate: folosesc scară estimată {m_px:.6f} m/px (presupunere {DEFAULT_PLAN_AREA_M2} m² pentru întreg planul)")
    try:
        scale_data = {
            'rooms': room_scales if room_scales else {},
            'total_area_m2': float(total_area_m2) if total_area_m2 > 0 else 0.0,
            'total_area_px': int(total_area_px) if total_area_px > 0 else 0,
            'm_px': float(m_px) if m_px is not None and m_px > 0 else None,
            'weighted_average_m_px': float(m_px) if m_px is not None and m_px > 0 else None,
            'room_scales': room_scales if room_scales else {},
            'estimated_scale': estimated_scale,
        }
        with open(output_dir / "room_scales.json", 'w', encoding='utf-8') as f:
            json.dump(scale_data, f, indent=2, ensure_ascii=False)
        _log(f"      💾 Salvat: room_scales.json ({len(room_scales)} camere)")
        _tick()
    except Exception as e:
        import traceback
        _log(f"      ⚠️ Eroare la salvarea room_scales.json: {e}")
        traceback.print_exc()
        # ✅ Încercăm să salvăm un fișier minim pentru a permite workflow-ul să continue
        try:
            minimal_data = {
                'rooms': {},
                'total_area_m2': 0.0,
                'total_area_px': 0,
                'm_px': None,
                'weighted_average_m_px': None,
                'room_scales': {},
                'error': str(e)
            }
            with open(output_dir / "room_scales.json", 'w', encoding='utf-8') as f:
                json.dump(minimal_data, f, indent=2, ensure_ascii=False)
            _log(f"      ⚠️ Salvat room_scales.json minimal (pentru compatibilitate workflow)")
        except Exception as e2:
            _log(f"      ❌ Eroare critică la salvarea room_scales.json: {e2}")
    
    # Salvăm măsurătorile openings într-un format compatibil cu workflow-ul
    if openings_list and m_px is not None:
        openings_measurements = {
            'scale_meters_per_pixel': float(m_px),
            'openings': []
        }
        
        for opening in openings_list:
            if opening.get('size_m') is not None:
                opening_data = {
                    'idx': opening['idx'],
                    'type': opening['type'],
                    'bbox': opening['bbox'],
                    'width_m': opening.get('size_m'),  # Lățime pentru uși, lungime pentru geamuri
                    'width_px': opening.get('size_px'),
                    'bbox_width_px': opening.get('width_px'),
                    'bbox_height_px': opening.get('height_px'),
                    'status': opening.get('status', 'unknown')  # exterior/interior pentru uși
                }
                openings_measurements['openings'].append(opening_data)
        
        if openings_measurements['openings']:
            with open(output_dir / "openings_measurements.json", 'w') as f:
                json.dump(openings_measurements, f, indent=2)
            _log(f"      💾 Salvat: openings_measurements.json ({len(openings_measurements['openings'])} openings)")
    
    # ✅ Notificare UI pentru randarea 3D (verificare finală - dacă nu a fost notificat deja)
    # Notificarea este trimisă imediat după generarea fișierului, dar verificăm și aici pentru siguranță
    output_path_3d = output_dir / "04_walls_3d.png"
    if output_path_3d.exists():
        # Verificăm dacă notificarea a fost deja trimisă (pentru a evita duplicate)
        # Notificarea principală este trimisă imediat după generarea fișierului în ambele căi (matplotlib și fallback)
        pass  # Notificarea este deja trimisă imediat după generare
    
    _tick()
    return {
        'walls_mask': walls_thick,  # ✅ walls_thick este generat din walls_overlay_mask (fără pereții terasei/balconului)
        'walls_mask_perfect': walls_overlay_mask,  # ✅ Masca perfectă folosită în room_x_debug.png (fără terasă/balcon)
        'walls_mask_for_roof': walls_mask_for_roof,  # ✅ Masca cu terasă + balcon, pentru generarea dreptunghiurilor acoperiș
        'walls_interior': walls_interior,
        'walls_exterior': walls_exterior,
        'room_scales': room_scales,
        'indoor_mask': interior_mask.astype(np.uint8) * 255,
        'outdoor_mask': flood_fill_result,
        'interior_structure': interior_structure,
        'exterior_structure': exterior_structure,
        'm_px': m_px,
        'total_area_m2': total_area_m2,
        'total_area_px': total_area_px,
        'openings': openings_list,  # Lista cu openings-uri și măsurătorile lor
        'rooms_polygons': rooms_polygons,  # Poligoanele camerelor pentru detectare garaj
        'api_to_original_coords': api_to_original_coords  # Funcție pentru transformare coordonate
    }
