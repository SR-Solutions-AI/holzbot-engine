# file: engine/cubicasa_detector/ocr_room_filling.py
"""
Module pentru detectarea textului prin OCR și umplerea camerelor (terasă, garaj, scări).
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

# OCR pentru detectarea textului
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️ pytesseract nu este disponibil. Detectarea textului va fi dezactivată.")


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
                     grid_rows: int = 3, grid_cols: int = 3, zoom_factor: float = 2.0) -> tuple:
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
        Tuple: (lista de text_boxes detectate, lista cu toate detecțiile OCR)
    """
    if not TESSERACT_AVAILABLE:
        return [], []
    
    h, w = image.shape[:2]
    text_boxes = []
    all_chars = []  # Colectăm toate caracterele detectate pentru reconstituire
    all_detections = []  # Colectăm TOATE detecțiile OCR pentru debug (nu doar cele care se potrivesc)
    all_words = []  # Colectăm TOATE cuvintele detectate pentru comparație ulterioară
    
    # Preprocesăm întreaga imagine
    processed_full = preprocess_image_for_ocr(image)
    
    # 1. OCR pe întreaga imagine preprocesată
    # Folosim PSM 11 (Sparse text) pentru a detecta cât mai mult text posibil în planuri
    print(f"         📝 OCR pe întreaga imagine (preprocesată)...")
    ocr_data_full = pytesseract.image_to_data(processed_full, output_type=pytesseract.Output.DICT, lang='deu+eng', config='--psm 11')
    
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
            
            # Verificăm direct dacă textul detectat se potrivește cu termenii căutați (EXACT CA LA TERASA)
            found_term = None
            for term in search_terms:
                term_lower = term.lower()
                text_lower = text_clean.lower()
                
                # Match exact (case-insensitive)
                if term_lower == text_lower:
                    found_term = term
                    break
                
                # Match dacă termenul este conținut în text (ex: "carport" în "Carport 22.40 m²")
                if term_lower in text_lower:
                    found_term = term
                    break
            
            if found_term:
                # Adăugăm detecția chiar dacă are confidence < 60% (pentru poza de debug)
                # Dar o marchem ca neacceptată dacă confidence < 60%
                text_boxes.append((x, y, width, height, text_clean, conf))
                if conf > 60:
                    print(f"         ✅ Detectat (full): '{text_clean}' la ({x}, {y}) cu confidență {conf:.1f}%")
                else:
                    print(f"         ⚠️ Detectat (full, LOW CONF): '{text_clean}' la ({x}, {y}) cu confidență {conf:.1f}% (< 60%)")
            else:
                # Colectăm și caracterele individuale pentru reconstituire
                # Verificăm dacă caracterul face parte din unul dintre termenii căutați
                for term in search_terms:
                    term_lower = term.lower()
                    if any(char in text_lower for char in term_lower):
                        all_chars.append((x, y, width, height, text_clean, conf))
                        break
    
    # 2. Întotdeauna încercăm și pe zone pentru a găsi textul mai mic sau mai clar
    # (chiar dacă am găsit ceva pe întreaga imagine, zonele pot găsi mai multe detecții)
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
                # Folosim PSM 11 (Sparse text) pentru a detecta cât mai mult text posibil
                ocr_data_zone = pytesseract.image_to_data(zone_zoomed, output_type=pytesseract.Output.DICT, lang='deu+eng', config='--psm 11')
                
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
                        
                        # Verificăm direct dacă textul detectat se potrivește cu termenii căutați (EXACT CA LA TERASA)
                        found_term = None
                        for term in search_terms:
                            term_lower = term.lower()
                            text_lower = text_clean.lower()
                            
                            # Match exact (case-insensitive)
                            if term_lower == text_lower:
                                found_term = term
                                break
                            
                            # Match dacă termenul este conținut în text (ex: "carport" în "Carport 22.40 m²")
                            if term_lower in text_lower:
                                found_term = term
                                break
                        
                        if found_term:
                            if conf > 60:
                                text_boxes.append((orig_x, orig_y, orig_width, orig_height, text_clean, conf))
                                print(f"         ✅ Detectat (zona {row+1},{col+1}): '{text_clean}' la ({orig_x}, {orig_y}) cu confidență {conf:.1f}%")
                            else:
                                # Adăugăm și detecțiile cu confidence < 60% pentru poza de debug
                                text_boxes.append((orig_x, orig_y, orig_width, orig_height, text_clean, conf))
                                print(f"         ⚠️ Detectat (zona {row+1},{col+1}, LOW CONF): '{text_clean}' la ({orig_x}, {orig_y}) cu confidență {conf:.1f}% (< 60%)")
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
    
    return text_boxes, all_detections


# NOTE: Funcția fill_room_by_ocr este foarte mare (peste 1200 linii) și este definită în detector.py.
# Pentru moment, o importăm din detector.py folosind TYPE_CHECKING pentru a evita importuri circulare.
# În viitor, va fi mutată complet aici.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Pentru type checking, importăm tipurile
    pass

def fill_room_by_ocr(walls_mask: np.ndarray, search_terms: list, room_name: str, 
                     steps_dir: str = None, debug_prefix: str = "02g") -> np.ndarray:
    """
    Funcție generică pentru detectarea și umplerea camerelor prin OCR.
    
    Detectează cuvintele din search_terms în plan și umple camera respectivă cu flood fill.
    
    Args:
        walls_mask: Masca pereților (255 = perete, 0 = spațiu liber)
        search_terms: Lista de termeni de căutat (ex: ["terasa", "Terasa", "TERASA"])
        room_name: Numele camerei pentru mesaje (ex: "terasa", "garage")
        steps_dir: Director pentru salvarea step-urilor de debug (opțional)
        debug_prefix: Prefix pentru fișierele de debug (ex: "02g" pentru terasa, "02h" pentru garaje)
    
    Returns:
        Masca pereților cu camera umplută (dacă a fost detectată)
    """
    if walls_mask is None:
        print(f"      ⚠️ walls_mask este None. Skip detectarea {room_name}.")
        return None
    
    try:
        h, w = walls_mask.shape[:2]
    except AttributeError:
        print(f"      ⚠️ walls_mask nu are atributul shape. Skip detectarea {room_name}.")
        return None
    
    # Folosim o metodă alternativă dacă OCR nu este disponibil
    use_ocr = TESSERACT_AVAILABLE
    if not use_ocr:
        print(f"      ⚠️ pytesseract nu este disponibil. Skip detectarea {room_name}.")
        return walls_mask.copy()
    
    result = walls_mask.copy()
    
    print(f"      🏡 Detectez și umplu camere ({room_name})...")
    
    # Creăm folder pentru output-uri structurate
    room_output_dir = None
    if steps_dir:
        room_output_dir = Path(steps_dir) / "ocr_room_filling" / room_name
        room_output_dir.mkdir(parents=True, exist_ok=True)
    
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
    all_detections = []
    
    try:
        if use_ocr:
            # Metoda îmbunătățită: OCR cu preprocesare și analiză pe zone
            print(f"         📝 Folosesc OCR cu preprocesare și analiză pe zone...")
            
            # Salvez imaginea preprocesată pentru debug
            if steps_dir:
                processed_img = preprocess_image_for_ocr(ocr_image)
                if room_output_dir:
                    cv2.imwrite(str(room_output_dir / "00_preprocessed.png"), processed_img)
                    print(f"         💾 Salvat: {room_output_dir.name}/00_preprocessed.png (imagine preprocesată)")
            
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
            
            if room_output_dir:
                output_path = room_output_dir / "01c_best_detection_with_fill.png"
                cv2.imwrite(str(output_path), vis_best_detection)
                print(f"         💾 Salvat: {room_output_dir.name}/01c_best_detection_with_fill.png (best detection: {best_conf:.1f}%, no flood fill)")
        
        if not text_found or not accepted_boxes:
            if not text_found:
                print(f"         ⚠️ Nu s-a detectat text ({room_name}) în plan.")
            else:
                print(f"         ⚠️ Nu s-a detectat text ({room_name}) cu confidence > 60%.")
            if steps_dir:
                vis_ocr = ocr_image.copy() if ocr_image is not None else cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
                if room_output_dir:
                    cv2.imwrite(str(room_output_dir / "01_ocr_result.png"), vis_ocr)
                    print(f"         💾 Salvat: {room_output_dir.name}/01_ocr_result.png")
            return result
        
        # Pas 3: Vizualizăm textul detectat (toate detecțiile)
        if steps_dir:
            vis_ocr = ocr_image.copy()
            for x, y, width, height, text, conf in text_boxes:
                cv2.rectangle(vis_ocr, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(vis_ocr, f"{text} ({conf:.0f}%)", (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if room_output_dir:
                cv2.imwrite(str(room_output_dir / "01_ocr_result.png"), vis_ocr)
                print(f"         💾 Salvat: {room_output_dir.name}/01_ocr_result.png (text detectat)")
        
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
            
            if room_output_dir:
                cv2.imwrite(str(room_output_dir / "01b_accepted_detections.png"), vis_accepted)
                print(f"         💾 Salvat: {room_output_dir.name}/01b_accepted_detections.png ({len(accepted_boxes)} detecție/ii acceptată/e)")
        
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
                return result
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
                    return result
                
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
                    return result
                
                # Verificăm că zona umplută este suficient de mare
                if not is_garage and filled_area < 1000:
                    print(f"         ⚠️ Zona detectată prea mică ({filled_area} pixeli). Skip.")
                    return result
                
                # Pentru garaj/carport, folosim o abordare geometrică simplificată
                if is_garage:
                    print(f"         🚗 Detectat garaj/carport - folosesc abordare geometrică simplificată...")
                    # Pentru garaj, folosim direct zona umplută de flood fill
                    # (logica geometrică complexă este în blocul orfan, dar nu o mutăm)
                    if filled_area < 1000:
                        print(f"         ⚠️ Zona geometrică prea mică ({filled_area} pixeli). Skip.")
                        return result
                
                # Verificăm că zona umplută este suficient de mare dar nu prea mare (pentru terasă)
                if filled_area > 1000 and (is_garage or filled_ratio <= 0.50):
                    print(f"         🔍 Extrag conturul zonei detectate pentru a completa golurile...")
                    
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
                        
                        gaps = cv2.bitwise_and(contour_mask, cv2.bitwise_not(walls_mask)) if contours else None
                        if gaps is not None:
                            gaps_colored = np.zeros_like(vis_fill)
                            gaps_colored[gaps > 0] = [0, 255, 0]
                            vis_fill = cv2.addWeighted(vis_fill, 0.5, gaps_colored, 0.5, 0)
                        
                        cv2.circle(vis_fill, (center_x, center_y), 5, (0, 0, 255), -1)
                        cv2.rectangle(vis_fill, (x, y), (x + width, y + height), (0, 255, 0), 2)
                        
                        if room_output_dir:
                            output_path = room_output_dir / f"02_fill_attempt_{box_idx + 1}.png"
                            cv2.imwrite(str(output_path), vis_fill)
                            print(f"         💾 Salvat: {room_output_dir.name}/{output_path.name}")
                    
                    print(f"         ✅ Gata! Am umplut camera {room_name}.")
                else:
                    if filled_area > 0 and filled_ratio > 0.50:
                        print(f"         ⚠️ Zona detectată prea mare ({filled_area} pixeli, {filled_ratio*100:.1f}% din imagine). Probabil a trecut prin pereți. Skip.")
                    else:
                        print(f"         ⚠️ Zona detectată prea mică ({filled_area} pixeli). Skip.")
        
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
            if room_output_dir:
                cv2.imwrite(str(room_output_dir / "03_final_result.png"), vis_final)
                print(f"         💾 Salvat: {room_output_dir.name}/03_final_result.png (verde=pereți noi)")
    
    except Exception as e:
        print(f"         ❌ Eroare la detectarea/umplerea {room_name}: {e}")
        import traceback
        traceback.print_exc()
        return result
    
    return result


def fill_stairs_room(walls_mask: np.ndarray, stairs_bboxes: list, steps_dir: str = None) -> np.ndarray:
    """
    Detectează scările folosind detecțiile Roboflow și reconstruiește pereții din jurul lor.
    NU face flood fill, doar reconstruiește pereții care lipsesc în jurul scărilor.
    
    Args:
        walls_mask: Masca pereților (255 = perete, 0 = spațiu liber)
        stairs_bboxes: Lista de bounding boxes pentru scări detectate de Roboflow
                      Format: [(x1, y1, x2, y2), ...]
        steps_dir: Director pentru salvarea step-urilor de debug (opțional)
    
    Returns:
        Masca pereților cu pereții reconstruiți în jurul scărilor (dacă au fost detectate)
    """
    # Verifică dacă walls_mask este None sau invalid
    if walls_mask is None:
        print(f"      ⚠️ walls_mask este None. Skip reconstruirea pereților pentru scări.")
        return None
    
    try:
        h, w = walls_mask.shape[:2]
    except AttributeError:
        print(f"      ⚠️ walls_mask nu are atributul shape. Skip reconstruirea pereților pentru scări.")
        return None
    
    result = walls_mask.copy()
    
    # Creăm folder pentru output-uri structurate
    stairs_output_dir = None
    if steps_dir:
        stairs_output_dir = Path(steps_dir) / "ocr_room_filling" / "stairs"
        stairs_output_dir.mkdir(parents=True, exist_ok=True)
    
    if not stairs_bboxes:
        print(f"      🏠 Nu s-au detectat scări. Skip reconstruirea pereților pentru scări.")
        return result
    
    print(f"      🏠 Reconstruiesc pereții în jurul scărilor ({len(stairs_bboxes)} detectate)...")
    
    stairs_processed = 0
    
    for stair_idx, bbox in enumerate(stairs_bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        
        # Asigurăm că coordonatele sunt în limitele imaginii
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            print(f"         ⚠️ Scara #{stair_idx + 1}: bbox invalid. Skip.")
            continue
        
        print(f"         🎯 Scara #{stair_idx + 1}: verific și reconstruiesc pereții în jurul scării...")
        
        # Calculăm o zonă buffer în jurul scării pentru a căuta pereți
        width_bbox = x2 - x1
        height_bbox = y2 - y1
        # Buffer mai mic - doar pentru a detecta pereții imediat în jurul scării
        buffer_size = max(20, int(max(width_bbox, height_bbox) * 0.3))  # 30% din dimensiunea scării
        
        # Creăm o zonă extinsă în jurul scării pentru a căuta pereți
        search_x1 = max(0, x1 - buffer_size)
        search_y1 = max(0, y1 - buffer_size)
        search_x2 = min(w, x2 + buffer_size)
        search_y2 = min(h, y2 + buffer_size)
        
        # Extragem zona de căutare din masca pereților
        search_region = result[search_y1:search_y2, search_x1:search_x2]
        
        # Creăm o mască pentru scara (excludem scara din procesare)
        stairs_mask_region = np.zeros((search_y2 - search_y1, search_x2 - search_x1), dtype=np.uint8)
        local_x1 = x1 - search_x1
        local_y1 = y1 - search_y1
        local_x2 = x2 - search_x1
        local_y2 = y2 - search_y1
        cv2.rectangle(stairs_mask_region, (local_x1, local_y1), (local_x2, local_y2), 255, -1)
        
        # Excludem scara din zona de căutare
        search_region_no_stairs = cv2.bitwise_and(search_region, cv2.bitwise_not(stairs_mask_region))
        
        # Detectăm pereții existenți în jurul scării
        existing_walls = search_region_no_stairs > 0
        
        # Creăm o zonă buffer extinsă pentru a desena pereții
        # Folosim conturul scării ca bază pentru reconstruirea pereților
        stairs_contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(stairs_contour_mask, (x1, y1), (x2, y2), 255, -1)
        
        # Dilatăm conturul scării pentru a crea o zonă buffer
        # Grosimea peretelui va fi adaptivă
        wall_thickness = max(3, int(min(w, h) * 0.003))
        kernel_size = max(3, int(min(w, h) * 0.005))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Dilatăm scara pentru a obține zona în care ar trebui să fie pereții
        stairs_dilated = cv2.dilate(stairs_contour_mask, kernel, iterations=1)
        
        # Extragem conturul zonei dilatate (perimetrul scării + buffer)
        contours, _ = cv2.findContours(stairs_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Găsim cel mai mare contur (scara principală)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Creăm o mască pentru conturul perimetrului (unde ar trebui să fie pereții)
            contour_perimeter_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(contour_perimeter_mask, [largest_contour], -1, 255, wall_thickness)
            
            # Identificăm golurile: unde conturul există dar pereții nu există
            gaps = cv2.bitwise_and(contour_perimeter_mask, cv2.bitwise_not(result))
            
            gaps_area = np.count_nonzero(gaps)
            
            if gaps_area > 0:
                # Adăugăm pereții noi (doar golurile) la masca finală
                result = cv2.bitwise_or(result, gaps)
                print(f"         ✅ Reconstruit {gaps_area} pixeli de pereți în jurul scării #{stair_idx + 1}")
                stairs_processed += 1
                
                # Vizualizăm reconstruirea
                if stairs_output_dir:
                    vis_reconstruction = cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
                    
                    # Desenăm scara (magenta)
                    cv2.rectangle(vis_reconstruction, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    
                    # Desenăm conturul perimetrului (albastru)
                    cv2.drawContours(vis_reconstruction, [largest_contour], -1, (255, 0, 0), 2)
                    
                    # Desenăm pereții reconstruiți (verde)
                    gaps_colored = np.zeros_like(vis_reconstruction)
                    gaps_colored[gaps > 0] = [0, 255, 0]
                    vis_reconstruction = cv2.addWeighted(vis_reconstruction, 0.7, gaps_colored, 0.3, 0)
                    
                    # Adăugăm text cu informații
                    status_text = f"Gaps filled: {gaps_area}px ✅"
                    font_scale = 0.8
                    font_thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    cv2.rectangle(vis_reconstruction, 
                                 (x1, y2 + 5), 
                                 (x1 + text_width + 10, y2 + text_height + baseline + 10), 
                                 (255, 255, 255), -1)
                    cv2.putText(vis_reconstruction, status_text, (x1 + 5, y2 + text_height + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
                    
                    output_path = stairs_output_dir / f"reconstruction_{stair_idx + 1}.png"
                    cv2.imwrite(str(output_path), vis_reconstruction)
                    print(f"         💾 Salvat: {stairs_output_dir.name}/reconstruction_{stair_idx + 1}.png")
            else:
                print(f"         ℹ️ Nu s-au găsit goluri în pereți în jurul scării #{stair_idx + 1}")
                if stairs_output_dir:
                    vis_no_gaps = cv2.cvtColor(walls_mask, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(vis_no_gaps, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    status_text = "No gaps found"
                    font_scale = 0.8
                    font_thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    cv2.rectangle(vis_no_gaps, 
                                 (x1, y2 + 5), 
                                 (x1 + text_width + 10, y2 + text_height + baseline + 10), 
                                 (255, 255, 255), -1)
                    cv2.putText(vis_no_gaps, status_text, (x1 + 5, y2 + text_height + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (128, 128, 128), font_thickness)
                    output_path = stairs_output_dir / f"reconstruction_{stair_idx + 1}.png"
                    cv2.imwrite(str(output_path), vis_no_gaps)
                    print(f"         💾 Salvat: {stairs_output_dir.name}/reconstruction_{stair_idx + 1}.png")
        else:
            print(f"         ⚠️ Nu s-au găsit contururi pentru scara #{stair_idx + 1}")
    
    if stairs_processed > 0:
        print(f"         ✅ Reconstruit pereți pentru {stairs_processed} scări")
    else:
        print(f"         ℹ️ Nu s-au reconstruit pereți (nu s-au găsit goluri)")
    
    # Salvăm rezultatul final
    if stairs_output_dir:
        vis_final = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        diff = cv2.subtract(result, walls_mask)
        diff_colored = np.zeros_like(vis_final)
        diff_colored[diff > 0] = [0, 255, 0]  # Verde pentru pereții noi
        vis_final = cv2.addWeighted(vis_final, 0.7, diff_colored, 0.3, 0)
        cv2.imwrite(str(stairs_output_dir / "final_result.png"), vis_final)
        print(f"         💾 Salvat: {stairs_output_dir.name}/final_result.png (verde=pereți noi)")
    
    return result
