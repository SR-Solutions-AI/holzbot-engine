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
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from .scale_detection import call_gemini, GEMINI_PROMPT_CROP
from .ocr_room_filling import run_ocr_on_zones, preprocess_image_for_ocr
from .wall_repair import get_strict_1px_outline

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
        print(f"      ⚠️ Dimensiuni diferite: crop={h_crop}x{w_crop}, mask={h_mask}x{w_mask}. Redimensionăm masca...")
        api_walls_mask = cv2.resize(api_walls_mask, (w_crop, h_crop), interpolation=cv2.INTER_NEAREST)
    
    # Convertim masca la BGR pentru overlay
    walls_bgr = cv2.cvtColor(api_walls_mask, cv2.COLOR_GRAY2BGR)
    
    # Suprapunem cu transparență 70% original + 30% pereți (roșu)
    walls_colored = walls_bgr.copy()
    walls_colored[api_walls_mask > 0] = [0, 0, 255]  # Roșu pentru pereți
    
    overlay = cv2.addWeighted(crop_img, 0.7, walls_colored, 0.3, 0)
    
    cv2.imwrite(str(output_path), overlay)
    print(f"      📸 Salvat: {output_path.name}")


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
        print(f"      ⚠️ Dimensiuni diferite: crop={h_crop}x{w_crop}, rooms={h_rooms}x{w_rooms}. Redimensionăm rooms...")
        api_rooms_img = cv2.resize(api_rooms_img, (w_crop, h_crop), interpolation=cv2.INTER_NEAREST)
    
    # Creăm o mască pentru camere (excludem fundalul alb)
    # Fundalul alb este (255, 255, 255)
    rooms_mask = np.all(api_rooms_img != [255, 255, 255], axis=2).astype(np.uint8) * 255
    
    # Suprapunem cu transparență 70% original + 30% camere
    overlay = cv2.addWeighted(crop_img, 0.7, api_rooms_img, 0.3, 0)
    
    cv2.imwrite(str(output_path), overlay)
    print(f"      📸 Salvat: {output_path.name}")


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
    cv2.floodFill(inv_pad_walls, flood_mask, (0, 0), 128)
    
    outdoor_mask = (inv_pad_walls[1:h+1, 1:w+1] == 128).astype(np.uint8) * 255
    
    # Dilatăm exterior-ul pentru a acoperi zonele libere
    kernel_grow = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    free_space = cv2.bitwise_not(api_walls_mask)
    
    for _ in range(30):
        outdoor_mask = cv2.bitwise_and(cv2.dilate(outdoor_mask, kernel_grow), free_space)
    
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
        
        print(f"      💾 Salvat măști în {output_dir.name}/")
    
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
                print(f"      ✅ Camera {room_id} ({result['room_name']}): {result['area_m2']:.2f} m², {result['m_px']:.9f} m/px")
    
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
        
        print(f"      📊 Media ponderată: {weighted_m_px:.9f} m/px ({len(room_scales)} camere)")
        
        return {
            "room_scales": room_scales,
            "weighted_average_m_px": weighted_m_px
        }
    else:
        print(f"      ⚠️ Nu s-au detectat camere cu suprafață validă")
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
        
        print(f"      💾 Salvat pereți în {output_dir.name}/")
    
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
        
        print(f"      💾 Salvat structură în {output_dir.name}/")
    
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
                    print(f"         🔧 Detectat loop: punct {key} la poziția {first_occurrence} și {i}, eliminat segment {first_occurrence+1}..{i}")
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
    print(f"         ⚠️ Avertisment: algoritm oprit după {max_iterations} iterații")
    return current


def generate_walls_from_room_coordinates(
    original_img: np.ndarray,
    best_config: Dict[str, Any],
    raster_dir: Path,
    steps_dir: str,
    gemini_api_key: str = None
) -> Dict[str, Any]:
    """
    Generează pereții pentru camere folosind coordonatele din overlay_on_original.png,
    face flood fill pentru fiecare cameră, calculează metri per pixel și extrage pereții interiori/exteriori.
    
    Args:
        original_img: Imaginea originală (BGR)
        best_config: Configurația brute force pentru transformare coordonate
        raster_dir: Directorul raster
        steps_dir: Directorul pentru steps
        gemini_api_key: Cheia API pentru Gemini (opțional)
    
    Returns:
        Dict cu rezultatele: walls_mask, room_scales, indoor_mask, outdoor_mask, walls_int, walls_ext
    """
    print("\n   🏗️ Generez pereți din coordonatele camerelor...")
    
    output_dir = Path(steps_dir) / "raster_processing" / "walls_from_coords"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ✅ Generez walls_overlay_on_crop.png local în walls_from_coords (necesar pentru validarea segmentelor)
    walls_overlay_path = output_dir / "walls_overlay_on_crop.png"
    if not walls_overlay_path.exists():
        api_walls_mask_path = raster_dir / "api_walls_mask.png"
        if api_walls_mask_path.exists() and original_img is not None:
            api_walls_mask = cv2.imread(str(api_walls_mask_path), cv2.IMREAD_GRAYSCALE)
            if api_walls_mask is not None:
                h_orig, w_orig = original_img.shape[:2]
                h_mask, w_mask = api_walls_mask.shape[:2]
                if (h_mask, w_mask) != (h_orig, w_orig):
                    api_walls_mask = cv2.resize(api_walls_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                generate_raster_walls_overlay(original_img, api_walls_mask, walls_overlay_path)
                print(f"      ✅ Generat walls_overlay_on_crop.png pentru validare")
    
    # 1. Încărcăm response.json pentru coordonatele camerelor
    response_json_path = raster_dir / "response.json"
    if not response_json_path.exists():
        print(f"      ⚠️ response.json nu există")
        return None
    
    with open(response_json_path, 'r') as f:
        result_data = json.load(f)
    
    data = result_data.get('data', result_data)
    
    # Funcție de transformare coordonate API -> Original
    def api_to_original_coords(x, y):
        if best_config['direction'] == 'api_to_orig':
            x_scaled = x * best_config['scale']
            y_scaled = y * best_config['scale']
            orig_x = x_scaled + best_config['position'][0]
            orig_y = y_scaled + best_config['position'][1]
            return int(orig_x), int(orig_y)
        else:
            x_in_template = x - best_config['position'][0]
            y_in_template = y - best_config['position'][1]
            orig_x = x_in_template / best_config['scale']
            orig_y = y_in_template / best_config['scale']
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
            print(f"      📐 COVERAGE folosește CROP ({w_plan}x{h_plan}, offset {crop_x},{crop_y}), mască ca walls_overlay_on_crop")
    
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
        print(f"      📐 Colectez coordonatele pentru {len(data['rooms'])} camere (pentru regenerare ulterioară)...")
        if bbox_rects:
            print(f"      🚪 Am găsit {len(bbox_rects)} bounding boxes pentru uși/geamuri")
        
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
                print(f"         ⚠️ Camera {i}: prea puține puncte ({len(pts_raw)})")
                continue
            
            # Convertim la numpy array (doar pentru a calcula centrul)
            pts_np = np.array(pts_raw, dtype=np.int32)
            rooms_polygons_original.append(pts_np)
            print(f"         ✅ Camera {i}: {len(pts_raw)} puncte (coordonate colectate)")
    
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
    
    # ✅ PASUL 2: Încărcăm api_walls_mask.png și o transformăm la coordonatele originale folosind transformarea brută
    # Această mască este aliniată cu planul original folosind transformarea brută (scale + position)
    # Aceasta va fi masca de bază peste care vom trasa și valida liniile din JSON
    api_walls_mask_path = raster_dir / "api_walls_mask.png"
    walls_overlay_mask = None
    if api_walls_mask_path.exists() and best_config:
        api_walls_mask = cv2.imread(str(api_walls_mask_path), cv2.IMREAD_GRAYSCALE)
        if api_walls_mask is not None:
            h_api, w_api = api_walls_mask.shape[:2]
            
            # ✅ Folosim transformarea brută pentru a transforma api_walls_mask la coordonatele originale
            # Transformarea brută folosește scale + position (fără rotație)
            scale = best_config['scale']
            x_pos, y_pos = best_config['position']
            direction = best_config['direction']
            
            # Pentru a transforma masca API la coordonatele originale, folosim întotdeauna transformarea directă api_to_orig
            # Dacă direction == 'orig_to_api', înseamnă că transformarea brută a găsit potrivirea inversă,
            # dar pentru a transforma masca API la original, trebuie să inversăm transformarea
            if direction == 'api_to_orig':
                # Transformare directă: API -> Original
                # x_orig = x_api * scale + x_pos
                # y_orig = y_api * scale + y_pos
                M = np.array([
                    [scale, 0, x_pos],
                    [0, scale, y_pos]
                ], dtype=np.float32)
            else:
                # direction == 'orig_to_api' - inversăm transformarea pentru a obține API -> Original
                # Transformarea brută a găsit: x_api = (x_orig - x_pos) / scale
                # Pentru a transforma API -> Original: x_orig = x_api * scale + x_pos (aceeași formulă!)
                # Deci folosim aceeași transformare
                M = np.array([
                    [scale, 0, x_pos],
                    [0, scale, y_pos]
                ], dtype=np.float32)
            
            # Aplicăm transformarea afine
            walls_overlay_mask = cv2.warpAffine(
                api_walls_mask, 
                M, 
                (w_orig, h_orig),
                flags=cv2.INTER_NEAREST,  # INTER_NEAREST pentru mască binară
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # Binarizăm masca (în cazul în care warpAffine produce valori intermediare)
            _, walls_overlay_mask = cv2.threshold(walls_overlay_mask, 127, 255, cv2.THRESH_BINARY)
            
            print(f"      ✅ Încărcat api_walls_mask.png și transformat la coordonatele originale folosind transformarea brută ({w_api}x{h_api} → {w_orig}x{h_orig}, scale={scale:.3f}x, pos=({x_pos}, {y_pos}))")
        else:
            print(f"      ⚠️ Nu am putut încărca api_walls_mask.png")
    else:
        if not api_walls_mask_path.exists():
            print(f"      ⚠️ api_walls_mask.png nu există în {raster_dir}")
        if not best_config:
            print(f"      ⚠️ best_config nu este disponibil pentru transformarea măștii")
    
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
        print(f"      ❌ Nu am mască (api_walls_mask) pentru coverage. Nu pot continua.")
        return None

    # ✅ PASUL 3: Construim walls_mask validată trăgând liniile din JSON peste mască
    # și validând cu >= 40% coverage între linia segmentului (1px) și mască. Aceasta va fi masca finală
    # folosită pentru generarea camerelor și pentru toate fișierele derivate (walls_thick, flood fill etc.).
    print(f"      🧱 Construiesc walls_mask validată: linia între cele 2 puncte vs mască, valid >= 40% coverage...")
    
    # ⚠️ Conform cerinței: pentru pașii următori folosim DOAR segmentele acceptate.
    # Deci masca de bază pornește GOALĂ, iar noi desenăm doar segmentele validate.
    walls_mask_validated = np.zeros((h_orig, w_orig), dtype=np.uint8)

    # ✅ IMPORTANT: 01_walls_from_coords.png NU trebuie să fie o mască completă de pereți.
    # Conform cerinței: aici salvăm DOAR segmentele de pereți ACCEPTATE (după validarea >= 40% pe segment).
    accepted_wall_segments_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    
    # Director pentru imagini de debug
    debug_walls_dir = output_dir / "wall_segments_debug"
    debug_walls_dir.mkdir(exist_ok=True)
    
    valid_segments = 0
    invalid_segments = 0
    
    # ✅ Adăugăm segmentele de pereți din JSON-ul RasterScan
    # Verificăm fiecare segment: câți pixeli din linia galbenă (segmente paralele) se află pe mască. Valid dacă >= 40%.
    plan_label = "crop" if use_crop_for_coverage else "plan"
    print(f"      📐 COVERAGE: mască = {mask_source_name}, overlap = pixeli linie galbenă pe mască, dim. {plan_label} {w_plan}x{h_plan}")

    if 'walls' in data and data['walls']:
        print(f"      🧱 Verific {len(data['walls'])} segmente de pereți din JSON...")
        
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
                            print(f"         ⚠️ Segment {idx} k={k}: overlap={overlap} total={total_px} (skip)")
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
            cv2.imwrite(str(debug_walls_dir / f"wall_segment_{idx:03d}.png"), debug_img)

            # ✅ Imagine suplimentară: mască (mov), segment (portocaliu), overlap (galben), fundal negru – în spațiul coverage (crop sau plan)
            if line_length > 0:
                overlay_img = np.zeros((h_plan, w_plan, 3), dtype=np.uint8)
                overlay_img[mask_for_coverage > 0] = [128, 0, 128]
                overlay_img[segment_mask > 0] = [0, 165, 255]
                overlap_px = (segment_mask > 0) & (mask_for_coverage > 0)
                overlay_img[overlap_px] = [0, 255, 255]
                cv2.imwrite(str(debug_walls_dir / f"wall_segment_{idx:03d}_mask_overlay.png"), overlay_img)

            if best_total > 0:
                print(f"         Segment {idx}: overlap {best_overlap}/{best_total} = {coverage_percent:.1f}% -> {status}")
            
            # ✅ Trasez linia peste walls_mask_validated DOAR dacă trece validarea (>= 40% coverage pe segment)
            if should_draw:
                cv2.line(walls_mask_validated, (x1, y1), (x2, y2), 255, line_thickness)
                # ✅ Salvăm separat DOAR segmentele acceptate (pentru 01_walls_from_coords.png) - GROSIME 1px
                cv2.line(accepted_wall_segments_mask, (x1, y1), (x2, y2), 255, 1)
                valid_segments += 1
            else:
                invalid_segments += 1
        
        print(f"      ✅ Segmente valide: {valid_segments} / {len(data['walls'])}")
        print(f"      ❌ Segmente invalide (coverage < 40% pe segment): {invalid_segments}")
        print(f"      📸 Imagini de debug salvate în: {debug_walls_dir.name}/")
    
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
    print(f"      💾 Salvat: walls.png (pereți validați cu >= 40% coverage pe segment)")
    
    # ⚠️ NU salvăm 01_walls_from_coords.png aici - va fi salvat DUPĂ ce walls_overlay_mask este disponibilă
    # pentru a folosi exact aceiași pereți perfecți ca în room_x_debug.png
    
    # ✅ REGENEREZ camerele pe baza pereților validați (>= 40% coverage pe segment)
    # Folosim EXACT aceeași mască validată (walls_overlay_mask) pentru a genera camerele
    print(f"      🔄 Regenerez camerele pe baza pereților validați (mască validată >= 40% coverage pe segment)...")
    rooms_mask_validated = np.zeros((h_orig, w_orig), dtype=np.uint8)
    rooms_polygons_validated = []
    
    # Inițializăm rooms_polygons și rooms_mask ca liste goale (vor fi populate după regenerare)
    # Acestea vor fi folosite mai târziu în cod, deci trebuie să fie definite
    rooms_polygons = []
    rooms_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    
    # ✅ Pentru flood fill și separări avem nevoie de o "barieră" (pereți cu grosime).
    # Conform cerinței, baza rămâne DOAR segmentele acceptate; grosimea este derivată din ele.
    if walls_overlay_mask is None:
        print(f"      ❌ walls_overlay_mask nu este disponibilă. Nu pot continua regenerarea camerelor.")
        return None

    # Grosime pereți: 0.005% din lățime (min 5px) - derivat din segmente
    wall_thickness = max(5, int(w_orig * 0.00005))
    print(f"      📏 Grosime pereți (pentru barrier): {wall_thickness}px (0.005% din {w_orig}px)")

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
                print(f"      ⚠️ Ignorat camera {i}: acoperă {room_coverage_ratio:.1%} din imagine (prea mare)")
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
        print(f"      🔍 Nu există camere în JSON. Generez automat camere din zonele închise din pereți...")
        
        # Folosim walls_barrier pentru a identifica zonele închise (spații libere delimitate de pereți)
        # Creăm o mască inversă: spațiile libere sunt 255, pereții sunt 0
        free_space_mask = (255 - walls_barrier).astype(np.uint8)
        
        # Identificăm toate componentele conectate de spații libere
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(free_space_mask, connectivity=8)
        
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
                    print(f"         ✅ Camera auto-generată {len(rooms_polygons_validated) - 1}: aria={component_area}px ({room_coverage_ratio:.1%})")
        
        print(f"      ✅ Generat automat {len(rooms_polygons_validated)} camere din zonele închise")
    
    # Actualizăm rooms_polygons cu versiunile validate
    rooms_polygons = rooms_polygons_validated
    rooms_mask = rooms_mask_validated
    
    # ✅ Generez imaginile de debug ale camerelor (DUPĂ regenerarea camerelor)
    # Conform cerinței, folosim doar segmentele acceptate (cu o barieră derivată pentru vizibilitate).
    print(f"      🔄 Generez imagini de debug camere cu pereții validați...")
    for i, room_poly in enumerate(rooms_polygons):
        # room_{i}_debug.png - cu pereții validați
        debug_img_room = original_img.copy()
        # Desenăm pereții (bariera) derivați din segmentele acceptate
        if walls_barrier is not None:
            walls_colored = cv2.cvtColor(walls_barrier, cv2.COLOR_GRAY2BGR)
            walls_colored[walls_barrier > 0] = [0, 255, 0]  # Verde pentru pereți
            debug_img_room = cv2.addWeighted(debug_img_room, 0.7, walls_colored, 0.3, 0)
        # Desenăm conturul camerei
        cv2.polylines(debug_img_room, [room_poly], True, (0, 255, 0), 2)
        # Debug info
        # room_poly din cv2.findContours are shape (N, 1, 2), deci trebuie să accesăm corect
        first_pt = room_poly[0][0] if len(room_poly[0].shape) > 1 else room_poly[0]
        cv2.putText(debug_img_room, f"Room {i} (validated walls)", 
                    (int(first_pt[0]), int(first_pt[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Desenăm punctele camerei
        for idx, pt in enumerate(room_poly):
            # pt poate fi (1, 2) sau (2,), trebuie să normalizăm
            pt_coords = pt[0] if len(pt.shape) > 1 else pt
            cv2.circle(debug_img_room, (int(pt_coords[0]), int(pt_coords[1])), 3, (255, 0, 0), -1)
            cv2.putText(debug_img_room, str(idx), (int(pt_coords[0]) + 5, int(pt_coords[1]) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        cv2.imwrite(str(output_dir / f"room_{i}_debug.png"), debug_img_room)
    
    # ✅ Regenerez rooms.png cu camerele validate (pe baza pereților validați)
    print(f"      🔄 Regenerez rooms.png cu camerele validate...")
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
    print(f"      💾 Salvat: rooms.png ({len(rooms_polygons)} camere validate)")
    
    # ✅ IMPORTANT: Pentru pașii următori folosim DOAR segmentele acceptate.
    if walls_overlay_mask is None:
        print(f"      ❌ walls_overlay_mask nu este disponibilă. Nu pot continua generarea fișierelor de pereți.")
        return None

    # ✅ Salvăm 01_walls_from_coords.png ca DOAR segmentele acceptate pe fundal negru (fără overlay)
    segments_path = output_dir / "01_walls_from_coords.png"
    segments_img = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)  # Fundal negru
    segments_img[accepted_wall_segments_mask > 0] = [255, 255, 255]  # Pereți albi
    cv2.imwrite(str(segments_path), segments_img)
    print(f"      💾 Salvat: {segments_path.name} (doar segmente pereți pe fundal negru)")
    
    # ✅ Flood fill din cele 4 colțuri (fără dilatare). Eliminăm pixel de perete dacă ≥2 din cei 4 vecini (N,S,E,W) sunt flood.
    print(f"      🌊 Flood fill din 4 colțuri, elimin pereți cu ≥2 vecini flood...")
    
    flood_base = (255 - accepted_wall_segments_mask).astype(np.uint8)
    corners = [(0, 0), (w_orig - 1, 0), (0, h_orig - 1), (w_orig - 1, h_orig - 1)]
    flood_any = np.zeros((h_orig, w_orig), dtype=np.uint8)
    
    for corner_idx, (cx, cy) in enumerate(corners):
        if flood_base[cy, cx] == 255:
            region_mask = np.zeros((h_orig + 2, w_orig + 2), dtype=np.uint8)
            img_copy = flood_base.copy()
            cv2.floodFill(img_copy, region_mask, (cx, cy), 128 + corner_idx, None, None, cv2.FLOODFILL_MASK_ONLY | 4)
            r = region_mask[1:-1, 1:-1] > 0
            flood_any[r] = 255
    
    walls_to_remove = np.zeros((h_orig, w_orig), dtype=np.uint8)
    for y in range(h_orig):
        for x in range(w_orig):
            if accepted_wall_segments_mask[y, x] == 0:
                continue
            n = 0
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h_orig and 0 <= nx < w_orig and flood_any[ny, nx] > 0:
                    n += 1
            if n >= 2:
                walls_to_remove[y, x] = 255
    
    accepted_wall_segments_mask[walls_to_remove > 0] = 0
    removed_count = int(np.sum(walls_to_remove > 0))
    if removed_count > 0:
        print(f"      ✅ Eliminat {removed_count} pixeli de perete (≥2 vecini flood)")
        walls_mask_validated[walls_to_remove > 0] = 0
        walls_overlay_mask = walls_mask_validated.copy()
    else:
        print(f"      ℹ️ Nu s-au găsit pixeli de perete de eliminat")
    
    # ✅ Recalculăm walls_barrier din segmentele acceptate (1px) DUPĂ eliminare; workflow-ul continuă pe baza pixelilor eliminați
    wall_thickness_barrier = max(5, int(w_orig * 0.00005))
    kernel_barrier = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wall_thickness_barrier, wall_thickness_barrier))
    walls_barrier = cv2.dilate(accepted_wall_segments_mask, kernel_barrier, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    walls_barrier = cv2.morphologyEx(walls_barrier, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # 2. Aplicăm grosimea pereților -> 02_walls_thick.png (derivat din segmente acceptate DUPĂ eliminare)
    walls_thick = walls_barrier.copy()
    
    cv2.imwrite(str(output_dir / "02_walls_thick.png"), walls_thick)
    print(f"      ✅ Salvat: 02_walls_thick.png (derivat din segmente acceptate)")
    
    # 2b. Generează outline-ul pereților (fără interior) -> 02b_walls_outline.png
    print(f"      🔲 Generez outline pereți (fără interior)...")
    walls_outline_only = np.zeros((h_orig, w_orig), dtype=np.uint8)
    
    # Găsim toți pixelii care sunt lângă un pixel alb (pereți) dar nu sunt pereți
    for y in range(h_orig):
        for x in range(w_orig):
            if walls_thick[y, x] == 0:  # Nu este perete
                # Verificăm vecinii (8-conectivitate)
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h_orig and 0 <= nx < w_orig:
                            if walls_thick[ny, nx] > 0:  # Vecinul este perete
                                walls_outline_only[y, x] = 255
                                break
                    if walls_outline_only[y, x] > 0:
                        break
    
    cv2.imwrite(str(output_dir / "02b_walls_outline.png"), walls_outline_only)
    print(f"      💾 Salvat: 02b_walls_outline.png")
    
    # 3. Suprapunem pereții peste planul original, colorați cu mov -> 03_walls_overlay.png
    print(f"      🎨 Generez overlay pereți peste plan...")
    overlay = original_img.copy()
    # Mov în BGR: [128, 0, 128]
    overlay[walls_thick > 0] = [128, 0, 128]
    cv2.imwrite(str(output_dir / "03_walls_overlay.png"), overlay)
    print(f"      💾 Salvat: 03_walls_overlay.png")
    
    # 4. Randare 3D va fi generată după pasul 8 (după generarea pereților interiori/exteriori)
    # (mutată mai jos pentru a putea folosi walls_exterior și walls_interior)
    
    # 4. Aplicăm outline roșu pe ambele părți ale pereților -> 05_walls_outline.png
    print(f"      🔲 Aplic outline roșu pe ambele părți...")
    # 4. Aplicăm outline roșu pe ambele părți ale pereților -> 05_walls_outline.png
    print(f"      🔲 Aplic outline roșu pe ambele părți...")
    walls_outline = np.zeros((h_orig, w_orig), dtype=np.uint8)
    
    # Găsim toți pixelii care sunt lângă un pixel alb (pereți)
    for y in range(h_orig):
        for x in range(w_orig):
            if walls_thick[y, x] == 0:  # Nu este perete
                # Verificăm vecinii (8-conectivitate)
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h_orig and 0 <= nx < w_orig:
                            if walls_thick[ny, nx] > 0:  # Vecinul este perete
                                walls_outline[y, x] = 255
                                break
                    if walls_outline[y, x] > 0:
                        break
    
    # Creăm imaginea cu outline roșu, păstrând pereții albi
    outline_img = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    # Pereții albi
    outline_img[walls_thick > 0] = [255, 255, 255]  # BGR: alb
    # Outline roșu peste pereți
    outline_img[walls_outline > 0] = [0, 0, 255]  # BGR: roșu
    
    cv2.imwrite(str(output_dir / "05_walls_outline.png"), outline_img)
    print(f"      💾 Salvat: 05_walls_outline.png")
    
    # 6. Flood fill albastru deschis din colțuri, outline-urile care ating devin albastru închis, cele care nu ating devin galben -> 06_walls_separated.png
    print(f"      🌊 Fac flood fill și separare outline-uri...")
    
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
    
    # Outline-urile care ating flood fill devin albastru închis
    outline_touches_flood = np.zeros((h_orig, w_orig), dtype=np.uint8)
    for y in range(h_orig):
        for x in range(w_orig):
            if walls_outline[y, x] > 0:
                # Verificăm vecinii
                touches = False
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h_orig and 0 <= nx < w_orig:
                            if flood_fill_result[ny, nx] > 0:
                                touches = True
                                break
                    if touches:
                        break
                if touches:
                    outline_touches_flood[y, x] = 255
                    separated_img[y, x] = [255, 100, 0]  # BGR: albastru închis
                else:
                    separated_img[y, x] = [0, 255, 255]  # BGR: galben
    
    cv2.imwrite(str(output_dir / "06_walls_separated.png"), separated_img)
    print(f"      💾 Salvat: 06_walls_separated.png")
    
    # 7. Generăm pereții interiori -> 07_walls_interior.png
    print(f"      🏗️ Generez pereții interiori...")
    walls_interior = np.zeros((h_orig, w_orig), dtype=np.uint8)
    
    # Pereții interiori = pereții care sunt lângă outline-urile care NU ating flood fill (galben)
    outline_interior = (walls_outline > 0) & (outline_touches_flood == 0)
    
    # Găsim pereții care sunt lângă outline-urile interioare
    for y in range(h_orig):
        for x in range(w_orig):
            if outline_interior[y, x]:
                # Verificăm vecinii pentru a găsi pereții
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h_orig and 0 <= nx < w_orig:
                            if walls_thick[ny, nx] > 0:
                                walls_interior[ny, nx] = 255
    
    cv2.imwrite(str(output_dir / "07_walls_interior.png"), walls_interior)
    print(f"      💾 Salvat: 07_walls_interior.png")
    
    # 8. Generăm pereții exteriori -> 08_walls_exterior.png
    print(f"      🏗️ Generez pereții exteriori...")
    walls_exterior = np.zeros((h_orig, w_orig), dtype=np.uint8)
    
    # Pereții exteriori = pereții care sunt lângă outline-urile care ating flood fill (albastru închis)
    outline_exterior = (walls_outline > 0) & (outline_touches_flood > 0)
    
    # Găsim pereții care sunt lângă outline-urile exterioare
    for y in range(h_orig):
        for x in range(w_orig):
            if outline_exterior[y, x]:
                # Verificăm vecinii pentru a găsi pereții
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h_orig and 0 <= nx < w_orig:
                            if walls_thick[ny, nx] > 0:
                                walls_exterior[ny, nx] = 255
    
    cv2.imwrite(str(output_dir / "08_walls_exterior.png"), walls_exterior)
    print(f"      💾 Salvat: 08_walls_exterior.png")
    
    # 4b. Randare 3D izometrică reală cu extruziune și iluminare -> 04_walls_3d.png
    print(f"      🎨 Generez randare 3D izometrică (extruziune reală cu iluminare)...")
    rendering_success = False
    try:
        # ✅ Folosim walls_thick direct pentru extrudare
        # walls_thick conține TOȚI pereții validați (exteriori + interiori) cu grosime aplicată
        walls_all = walls_thick.copy()
        
        wall_pixels = np.where(walls_all > 0)
        if len(wall_pixels[0]) == 0:
            print(f"         ⚠️ Nu s-au găsit pereți pentru randare 3D")
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
            print(f"         📐 Desenez fața superioară...")
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
            print(f"         📐 Desenez fețele laterale...")
            
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
            print(f"         📐 Desenez outline-urile...")
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
            print(f"         🎨 Aplic smoothing ușor...")
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
            print(f"      💾 Salvat: 04_walls_3d.png ({len(wall_pixels[0])} pixeli pereți extruși, înălțime {wall_height_px}px)")
            
            # ✅ Notificare UI imediat după generarea fișierului 3D
            if output_path.exists():
                notify_ui("scale", output_path)
                print(f"      📢 Notificat UI pentru 04_walls_3d.png")
            rendering_success = True
    except Exception as e:
        import traceback
        print(f"         ⚠️ Eroare la randarea 3D: {e}")
        print(f"         ⚠️ Continuăm cu workflow-ul (randarea 3D nu este critică)...")
        traceback.print_exc()
        rendering_success = False
    
    # 9. Generăm interiorul casei (pixelii negri din flood fill devin portocaliu) -> 09_interior.png
    print(f"      🏠 Generez interiorul casei...")
    interior_img = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    
    # Pixelii negri (nu au fost atinși de flood fill) devin portocaliu
    interior_mask = (flood_fill_result == 0) & (walls_thick == 0)
    interior_img[interior_mask] = [0, 165, 255]  # BGR: portocaliu
    
    cv2.imwrite(str(output_dir / "09_interior.png"), interior_img)
    print(f"      💾 Salvat: 09_interior.png")
    
    # 10. Flood fill pe structura inițială (01_walls_from_coords) -> 10_flood_structure.png
    print(f"      🌊 Aplic flood fill pe structura inițială...")
    
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
    print(f"      💾 Salvat: 10_flood_structure.png")
    
    # ✅ Notificăm UI cu imaginea de flood fill într-un stage SEPARAT (fără alte imagini),
    # dar încadrat în aceeași etapă logică de scale.
    if flood_structure_path.exists():
        notify_ui("scale_flood", flood_structure_path)
        print(f"      📢 Notificat UI pentru 10_flood_structure.png")
    
    # 11. Structura pereților interiori (pixelii albi care nu au vecini cu flood fill) -> 11_interior_structure.png
    print(f"      🏗️ Generez structura pereților interiori...")
    interior_structure = np.zeros((h_orig, w_orig), dtype=np.uint8)
    
    # Pixelii albi (pereți) care NU au vecini cu flood fill
    interior_structure[walls_not_touching] = 255
    
    cv2.imwrite(str(output_dir / "11_interior_structure.png"), interior_structure)
    print(f"      💾 Salvat: 11_interior_structure.png")
    
    # 12. Structura pereților exteriori (pixelii care au devenit roșii) -> 12_exterior_structure.png
    print(f"      🏗️ Generez structura pereților exteriori...")
    exterior_structure = np.zeros((h_orig, w_orig), dtype=np.uint8)
    
    # Pixelii care au devenit roșii (dar albi în poza finală)
    exterior_structure[walls_touching_flood > 0] = 255
    
    cv2.imwrite(str(output_dir / "12_exterior_structure.png"), exterior_structure)
    print(f"      💾 Salvat: 12_exterior_structure.png")
    
    # 13. Procesăm doors (openings) -> openings/
    print(f"      🚪 Procesez deschideri (doors) din RasterScan...")
    openings_dir = output_dir.parent / "openings"
    openings_dir.mkdir(parents=True, exist_ok=True)
    
    # Prompt pentru clasificarea doors - îmbunătățit cu exemple clare
    DOOR_CLASSIFICATION_PROMPT = """Analyze this architectural floor plan image showing a door or window opening.

CRITICAL DISTINCTIONS:
- "window" - Has glass panes (usually shown as parallel lines or grid pattern), typically narrower than doors, often near exterior walls. May have a sill or frame detail.
- "door" - Single or double opening for passage, usually wider than windows, may show door swing arc or simple opening. Interior doors connect rooms, exterior doors lead outside.
- "garage_door" - Very wide opening (typically 2-3x wider than regular doors), usually at ground level, often shows multiple panels or tracks. Located near driveway or exterior.
- "stairs" - Shows steps pattern (parallel lines going up/down), usually in a rectangular or L-shaped opening, connects floors.

VISUAL CLUES:
- Windows: Thin opening with lines/grid inside, often rectangular, smaller than doors
- Doors: Wider opening, may show arc (door swing), connects spaces
- Garage doors: Very wide (often 3-4m), multiple horizontal panels visible
- Stairs: Distinct step pattern visible, usually in corner or central area

Respond ONLY with JSON: {"type": "window"} or {"type": "door"} or {"type": "garage_door"} or {"type": "stairs"}"""

    # Stocăm toate openings-urile pentru generarea imaginilor
    openings_list = []
    
    if 'doors' in data and data['doors']:
        print(f"      📐 Procesez {len(data['doors'])} deschideri...")
        
        for idx, door in enumerate(data['doors']):
            if 'bbox' not in door or len(door['bbox']) != 4:
                continue
            
            # Transformăm coordonatele API -> Original
            bbox_api = door['bbox']
            x1_api, y1_api, x2_api, y2_api = bbox_api
            
            # Transformăm coordonatele
            x1_orig, y1_orig = api_to_original_coords(x1_api, y1_api)
            x2_orig, y2_orig = api_to_original_coords(x2_api, y2_api)
            
            # Normalizăm coordonatele
            x_min = max(0, min(x1_orig, x2_orig))
            y_min = max(0, min(y1_orig, y2_orig))
            x_max = min(w_orig, max(x1_orig, x2_orig))
            y_max = min(h_orig, max(y1_orig, y2_orig))
            
            # Adăugăm padding redus pentru a evita să includem elemente adiacente
            # Calculăm padding proporțional cu dimensiunea opening-ului
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            # Padding maxim 20px sau 10% din dimensiunea opening-ului
            padding = min(20, max(5, int(min(bbox_width, bbox_height) * 0.1)))
            x_min_crop = max(0, x_min - padding)
            y_min_crop = max(0, y_min - padding)
            x_max_crop = min(w_orig, x_max + padding)
            y_max_crop = min(h_orig, y_max + padding)
            
            if x_max_crop <= x_min_crop or y_max_crop <= y_min_crop:
                continue
            
            # Facem crop cu mai mult context
            door_crop = original_img[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
            
            if door_crop.size == 0:
                continue
            
            # Găsim camera în care se află opening-ul
            room_name = "Unknown"
            if 'rooms' in data and data['rooms']:
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                for room_idx, room_poly in enumerate(rooms_polygons[:len(data['rooms'])]):
                    if cv2.pointPolygonTest(room_poly, (center_x, center_y), False) >= 0:
                        # Găsim numele camerei dacă există
                        if room_idx < len(data['rooms']):
                            room_data = data['rooms'][room_idx]
                            if 'name' in room_data:
                                room_name = room_data['name']
                            elif 'label' in room_data:
                                room_name = room_data['label']
                        break
            
            # Clasificăm folosind doar Gemini (fără template matching)
            door_type = "door"  # Default
            
            # Folosim Gemini pentru clasificare
            if gemini_api_key:
                try:
                    # Import local pentru a evita probleme de scope
                    from .scale_detection import call_gemini as gemini_classify
                    
                    # Salvăm crop-ul temporar
                    temp_crop_path = openings_dir / f"door_{idx}_temp.png"
                    cv2.imwrite(str(temp_crop_path), door_crop)
                    
                    # Creăm prompt cu context
                    context_prompt = DOOR_CLASSIFICATION_PROMPT
                    if room_name != "Unknown":
                        context_prompt += f"\n\nContext: This opening is located in the '{room_name}' room."
                    
                    # Apelăm Gemini
                    result = gemini_classify(str(temp_crop_path), context_prompt, gemini_api_key)
                    
                    if result and isinstance(result, dict) and 'type' in result:
                        door_type = result['type'].lower().strip()
                        # Normalizăm tipul
                        if door_type not in ['door', 'window', 'garage_door', 'stairs']:
                            door_type = 'door'
                    else:
                        # Dacă nu e sigur, folosim default
                        print(f"         ⚠️ Gemini nu a returnat un tip valid pentru door {idx}, folosesc default: door")
                        door_type = 'door'
                    
                    # Ștergem fișierul temporar
                    if temp_crop_path.exists():
                        temp_crop_path.unlink()
                except Exception as e:
                    print(f"         ⚠️ Eroare clasificare door {idx}: {e}")
            
            # Salvăm crop-ul cu text peste imagine (folosim crop-ul original mai mic pentru salvare)
            door_crop_small = original_img[y_min:y_max, x_min:x_max]
            if door_crop_small.size == 0:
                door_crop_small = door_crop
            door_crop_with_text = door_crop_small.copy()
            
            # Adăugăm text
            text = f"{door_type.upper()}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # Calculăm dimensiunea textului
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Desenăm background pentru text
            cv2.rectangle(door_crop_with_text, 
                         (5, 5), 
                         (text_width + 15, text_height + baseline + 15),
                         (0, 0, 0), -1)
            
            # Desenăm text
            cv2.putText(door_crop_with_text, text, 
                       (10, text_height + 10),
                       font, font_scale, (255, 255, 255), thickness)
            
            # Salvăm
            door_path = openings_dir / f"door_{idx}_{door_type}.png"
            cv2.imwrite(str(door_path), door_crop_with_text)
            print(f"         💾 Salvat: door_{idx}_{door_type}.png")
            
            # Stocăm pentru generarea imaginilor (măsurătorile vor fi calculate mai târziu după calculul m_px)
            openings_list.append({
                'idx': idx,
                'type': door_type,
                'bbox': (x_min, y_min, x_max, y_max),
                'center': ((x_min + x_max) // 2, (y_min + y_max) // 2)
            })
    
    # Generăm 01_openings.png - planul cu toate openings-urile colorate
    if openings_list:
        print(f"      🎨 Generez 01_openings.png...")
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
        print(f"         💾 Salvat: 01_openings.png")
        
        # Notificare UI pentru detections
        if openings_img_path.exists():
            notify_ui("detections", openings_img_path)
        
        # Generăm 02_exterior_doors.png - ușile interioare (verde) și exterioare (roșu)
        print(f"      🎨 Generez 02_exterior_doors.png...")
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
        print(f"         💾 Salvat: 02_exterior_doors.png")
        
        # Notificare UI pentru exterior doors
        if exterior_doors_img_path.exists():
            notify_ui("exterior_doors", exterior_doors_img_path)
    
    # 14. Suprapunem rooms cu casa și calculăm metri per pixel (păstrăm doar pozele legate de room)
    print(f"      📏 Calculez metri per pixel pentru fiecare cameră...")
    
    room_scales = {}
    total_area_m2 = 0.0
    total_area_px = 0
    
    # Funcție helper pentru procesarea unei camere (pentru paralelizare)
    # ✅ Această funcție va fi apelată DUPĂ corecția pereților, deci walls_overlay_mask este disponibil
    # walls_mask_validated este walls_overlay_mask (masca validată cu 70% coverage)
    def process_room_for_scale(i, room, room_poly, walls_mask_validated):
        if i >= len(rooms_polygons):
            return None
        
        # Creăm o mască pentru această cameră
        room_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        cv2.fillPoly(room_mask, [room_poly], 255)
        
        # Calculăm aria în pixeli
        room_area_px = np.count_nonzero(room_mask)
        
        if room_area_px < 100:  # Prea mică
            return None
        
        # Salvăm crop-ul camerei pentru Gemini
        x, y, w, h = cv2.boundingRect(room_poly)
        x = max(0, x - 10)
        y = max(0, y - 10)
        w = min(w_orig - x, w + 20)
        h = min(h_orig - y, h + 20)
        
        room_crop = original_img[y:y+h, x:x+w].copy()
        
        if room_crop.size == 0:
            return None
        
        # Acoperim cu negru ce nu face parte din cameră
        room_mask_crop = np.zeros((h, w), dtype=np.uint8)
        # ✅ Normalizăm room_poly la shape (N, 2) pentru a evita erorile de indexing
        # room_poly vine din cv2.findContours cu shape (N, 1, 2), trebuie să normalizăm
        if len(room_poly.shape) == 3 and room_poly.shape[1] == 1:
            # Shape (N, 1, 2) -> reshape la (N, 2)
            room_poly_normalized = room_poly.reshape(-1, 2)
        else:
            room_poly_normalized = room_poly.copy()
        room_poly_crop = room_poly_normalized.copy()
        room_poly_crop[:, 0] -= x
        room_poly_crop[:, 1] -= y
        cv2.fillPoly(room_mask_crop, [room_poly_crop], 255)
        room_crop[room_mask_crop == 0] = [0, 0, 0]
        
        # ✅ Adăugăm pereții validați în crop-ul pentru Gemini
        # walls_mask_validated este walls_overlay_mask (masca validată cu 70% coverage)
        walls_mask_crop = walls_mask_validated[y:y+h, x:x+w].copy()
        # Desenăm pereții validați peste crop (roșu pentru vizibilitate)
        walls_colored_crop = cv2.cvtColor(walls_mask_crop, cv2.COLOR_GRAY2BGR)
        walls_colored_crop[walls_mask_crop > 0] = [0, 0, 255]  # Roșu pentru pereții validați
        room_crop = cv2.addWeighted(room_crop, 0.8, walls_colored_crop, 0.2, 0)
        
        # ✅ Generăm room_x_location.png cu pereții validați
        # Folosim EXACT walls_overlay_mask (masca validată cu 70%) pentru a desena pereții
        room_location_img = original_img.copy()
        # Desenăm pereții validați (roșu) - folosim walls_overlay_mask dacă este disponibilă
        if walls_mask_validated is not None:
            # walls_mask_validated este walls_overlay_mask (masca validată cu 70%)
            walls_colored_loc = cv2.cvtColor(walls_mask_validated, cv2.COLOR_GRAY2BGR)
            walls_colored_loc[walls_mask_validated > 0] = [0, 0, 255]  # Roșu pentru pereții validați
            room_location_img = cv2.addWeighted(room_location_img, 0.7, walls_colored_loc, 0.3, 0)
        # Desenăm camera (galben)
        room_colored_mask = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
        room_colored_mask[room_mask > 0] = [0, 255, 255]  # Galben pentru cameră
        room_location_img = cv2.addWeighted(room_location_img, 0.5, room_colored_mask, 0.5, 0)
        cv2.imwrite(str(output_dir / f"room_{i}_location.png"), room_location_img)
        
        # Apelăm Gemini pentru suprafață
        result_data = None
        if gemini_api_key:
            try:
                crop_path = output_dir / f"room_{i}_temp_for_gemini.png"
                cv2.imwrite(str(crop_path), room_crop)
                from .scale_detection import call_gemini, GEMINI_PROMPT_CROP, is_informational_total_result
                result = call_gemini(str(crop_path), GEMINI_PROMPT_CROP, gemini_api_key)
                # Ignorăm rezultate care sunt texte informativ (NNF, Nutzfläche etc.), nu camere concrete
                if result and is_informational_total_result(result):
                    result = None
                if result and 'area_m2' in result:
                    area_m2 = float(result['area_m2'])
                    if area_m2 > 0:
                        result_data = {
                            'idx': i,
                            'area_m2': float(area_m2),
                            'area_px': int(room_area_px),
                            'room_name': result.get('room_name', f'Room_{i}'),
                            'room_crop': room_crop,
                            'room_mask': room_mask[y:y+h, x:x+w]
                        }
            except Exception as e:
                print(f"         Camera {i}: Eroare Gemini: {e}")
        
        # Salvăm crop-ul camerei
        if result_data:
            cv2.imwrite(str(output_dir / f"room_{i}_crop.png"), result_data['room_crop'])
            cv2.imwrite(str(output_dir / f"room_{i}_mask.png"), result_data['room_mask'])
        
        return result_data
    
    # Procesăm camerele în paralel
    # ✅ Include atât camerele din JSON cât și camerele auto-generate
    room_tasks = []
    if 'rooms' in data and data['rooms']:
        # Camere din JSON
        room_tasks = [
            (i, room, rooms_polygons[i]) 
            for i, room in enumerate(data['rooms']) 
            if i < len(rooms_polygons)
        ]
    elif len(rooms_polygons) > 0:
        # ✅ Camere auto-generate: creăm room_tasks din rooms_polygons
        print(f"      🔍 Procesez {len(rooms_polygons)} camere auto-generate pentru calcularea scale-ului...")
        room_tasks = [
            (i, {'type': 'auto_generated', 'id': i}, rooms_polygons[i])
            for i in range(len(rooms_polygons))
        ]
    
    # Camere care acoperă (aproape) tot planul = flood fill eșuat → nu le folosim la total_area / scală
    MAX_ROOM_COVERAGE_RATIO = 0.95
    img_total_px = h_orig * w_orig

    if room_tasks:
        
        # ✅ Verificăm suprapunerea între camere și eliminăm duplicatele (>70% suprapunere)
        # ✅ Excludem camerele prea mari (flood fill eșuat: o cameră = tot ecranul)
        print(f"      🔍 Verific suprapunerea între camere și exclud camerele prea mari (flood fill eșuat)...")
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
                print(f"         ⚠️ Camera {i}: acoperă {100 * room_area_px / img_total_px:.1f}% din plan (flood fill eșuat) → exclud din calcul")
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
                        print(f"         ⚠️ Camera {i} are suprapunere {iou*100:.1f}% cu Camera {j} -> skip complet (suprafață 0)")
                        should_process = False
                        overlap_skipped_indices.add(i)
                        break
            
            if should_process:
                rooms_to_process.append((i, room, room_poly))
        
        print(f"      ✅ {len(rooms_to_process)} camere unice pentru procesare (din {len(room_tasks)} total, {len(room_tasks) - len(rooms_to_process)} skip-uite)")
        
        max_workers = max(1, min(4, len(rooms_to_process)))  # Max 4 thread-uri pentru Gemini API, minim 1
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # ✅ Trimitem walls_overlay_mask (masca validată cu 70% coverage) către funcție
            # Aceasta este masca aliniată cu planul original folosind transformarea brută
            walls_mask_for_rooms = walls_overlay_mask if walls_overlay_mask is not None else walls_mask
            futures = {
                executor.submit(process_room_for_scale, i, room, room_poly, walls_mask_for_rooms): i 
                for i, room, room_poly in rooms_to_process
            }
            future_results = {}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    i = result['idx']
                    future_results[i] = result
                    room_scales[i] = {
                        'area_m2': result['area_m2'],
                        'area_px': result['area_px'],
                        'room_name': result['room_name']
                    }
                    total_area_m2 += result['area_m2']
                    total_area_px += result['area_px']
                    print(f"         Camera {i}: {result['area_m2']:.2f} m², {result['area_px']} px")
                else:
                    # Notăm explicit camerele procesate dar fără rezultat (Gemini a eșuat / nu a returnat area_m2)
                    i = futures.get(future)
                    if i is not None:
                        future_results[i] = None
            
            # ✅ Separăm clar:
            # - camere skip-uite din cauza overlap (>70%)
            # - camere la care Gemini a eșuat (nu trebuie etichetate ca overlap)
            processed_ok_indices = {i for i, res in future_results.items() if res is not None}
            attempted_indices = {i for i, _, _ in rooms_to_process}
            gemini_failed_indices = attempted_indices - processed_ok_indices

            for i in sorted(overlap_skipped_indices):
                room_scales[i] = {
                    'area_m2': 0.0,
                    'area_px': 0,
                    'room_name': f'Room_{i} (skipped - overlap >70%)'
                }

            for i in sorted(flood_fill_skipped_indices):
                room_scales[i] = {
                    'area_m2': 0.0,
                    'area_px': 0,
                    'room_name': f'Room_{i} (skipped - flood fill failed, room too large)'
                }

            for i in sorted(gemini_failed_indices):
                # Pentru transparență, păstrăm aria în pixeli (nu intră în total_area_*).
                area_px_est = int(np.count_nonzero(room_masks.get(i))) if i in room_masks else 0
                room_scales[i] = {
                    'area_m2': 0.0,
                    'area_px': area_px_est,
                    'room_name': f'Room_{i} (gemini_failed)'
                }
                print(f"         ⚠️ Camera {i}: Gemini a eșuat -> nu este folosită la scală (area_px={area_px_est})")
    
    # Calculăm metri per pixel global
    m_px = None
    if total_area_px > 0 and total_area_m2 > 0:
        m_px = np.sqrt(total_area_m2 / total_area_px)
        print(f"      📏 Metri per pixel global: {m_px:.9f} m/px (total: {total_area_m2:.2f} m², {total_area_px} px)")
    
    # Calculăm măsurătorile openings (după calculul m_px)
    if m_px is not None and openings_list:
        print(f"      📐 Calculez măsurătorile openings...")
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
                print(f"         🚪 {door_type} {opening['idx']}: lățime = {size_m:.3f} m ({size_px} px) [min: {min(bbox_width, bbox_height)}px]")
            elif door_type == 'window':
                print(f"         🪟 window {opening['idx']}: lungime = {size_m:.3f} m ({size_px} px) [max: {max(bbox_width, bbox_height)}px]")
    
    # Salvăm scale-urile
    # ✅ IMPORTANT: Salvăm room_scales.json chiar dacă room_scales este gol sau dacă au fost erori
    # Acest fișier este necesar pentru workflow-ul ulterior
    try:
        scale_data = {
            'rooms': room_scales if room_scales else {},
            'total_area_m2': float(total_area_m2) if total_area_m2 > 0 else 0.0,
            'total_area_px': int(total_area_px) if total_area_px > 0 else 0,
            'm_px': float(m_px) if m_px is not None and m_px > 0 else None,
            'weighted_average_m_px': float(m_px) if m_px is not None and m_px > 0 else None,
            'room_scales': room_scales if room_scales else {}  # Format compatibil cu scale/jobs.py
        }
        with open(output_dir / "room_scales.json", 'w', encoding='utf-8') as f:
            json.dump(scale_data, f, indent=2, ensure_ascii=False)
        print(f"      💾 Salvat: room_scales.json ({len(room_scales)} camere)")
    except Exception as e:
        import traceback
        print(f"      ⚠️ Eroare la salvarea room_scales.json: {e}")
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
            print(f"      ⚠️ Salvat room_scales.json minimal (pentru compatibilitate workflow)")
        except Exception as e2:
            print(f"      ❌ Eroare critică la salvarea room_scales.json: {e2}")
    
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
            print(f"      💾 Salvat: openings_measurements.json ({len(openings_measurements['openings'])} openings)")
    
    # ✅ Notificare UI pentru randarea 3D (verificare finală - dacă nu a fost notificat deja)
    # Notificarea este trimisă imediat după generarea fișierului, dar verificăm și aici pentru siguranță
    output_path_3d = output_dir / "04_walls_3d.png"
    if output_path_3d.exists():
        # Verificăm dacă notificarea a fost deja trimisă (pentru a evita duplicate)
        # Notificarea principală este trimisă imediat după generarea fișierului în ambele căi (matplotlib și fallback)
        pass  # Notificarea este deja trimisă imediat după generare
    
    return {
        'walls_mask': walls_thick,  # ✅ walls_thick este generat din walls_overlay_mask (masca perfectă)
        'walls_mask_perfect': walls_overlay_mask,  # ✅ Masca perfectă folosită în room_x_debug.png (validată cu 70% coverage)
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
