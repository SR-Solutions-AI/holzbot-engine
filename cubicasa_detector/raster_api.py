# file: engine/cubicasa_detector/raster_api.py
"""
Module pentru integrarea cu RasterScan API.
Conține funcții pentru apelul API, generarea imaginilor, alinierea brute-force și generarea crop-ului.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import json
import base64
import requests
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List


def call_raster_api(img: np.ndarray, steps_dir: str) -> Optional[Dict[str, Any]]:
    """
    Apelează RasterScan API pentru vectorizarea imaginii.
    
    Args:
        img: Imaginea de procesat (BGR)
        steps_dir: Director pentru salvarea rezultatelor
    
    Returns:
        Dict cu răspunsul API sau None dacă a eșuat
    """
    try:
        print(f"   🔄 Apel RasterScan API pentru vectorizare...")
        
        # Creăm folderul raster
        raster_dir = Path(steps_dir) / "raster"
        raster_dir.mkdir(parents=True, exist_ok=True)
        
        # ✅ PREPROCESARE: Ștergem liniile foarte subțiri înainte de trimitere la RasterScan
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
        
        # Salvăm copia preprocesată în folderul raster
        preprocessed_path = raster_dir / "00_original_preprocessed.png"
        cv2.imwrite(str(preprocessed_path), api_img)
        print(f"      💾 Salvat: {preprocessed_path.name} (preprocesat - linii subțiri eliminate)")
        
        # Redimensionăm imaginea dacă e prea mare (API limit ~4MB)
        MAX_API_DIM = 2048
        h_api, w_api = api_img.shape[:2]
        scale_factor = 1.0
        
        if max(h_api, w_api) > MAX_API_DIM:
            scale_factor = MAX_API_DIM / max(h_api, w_api)
            new_w_api = int(w_api * scale_factor)
            new_h_api = int(h_api * scale_factor)
            api_img = cv2.resize(api_img, (new_w_api, new_h_api), interpolation=cv2.INTER_AREA)
            print(f"      📐 Redimensionat pentru API: {w_api}x{h_api} -> {new_w_api}x{new_h_api}")
        else:
            new_w_api, new_h_api = w_api, h_api
        
        # Salvăm imaginea pentru API
        api_img_path = raster_dir / "input_resized.jpg"
        cv2.imwrite(str(api_img_path), api_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Convertim în base64 (folosim JPEG comprimat)
        _, buffer = cv2.imencode('.jpg', api_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        print(f"      📦 Dimensiune payload: {len(image_base64) / 1024 / 1024:.2f} MB")
        
        # Apelăm API-ul RasterScan
        raster_api_key = os.environ.get('RASTER_API_KEY', '')
        if not raster_api_key:
            print(f"      ⚠️ RASTER_API_KEY nu este setat în environment")
            return None
        
        url = "https://backend.rasterscan.com/raster-to-vector-base64"
        payload = {"image": image_base64}
        headers = {
            "x-api-key": raster_api_key,
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print(f"      ✅ RasterScan API răspuns primit")
            
            # Salvăm răspunsul JSON
            json_path = raster_dir / "response.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"      📄 Salvat: {json_path.name}")
            
            # Salvăm SVG, DXF, și imaginea procesată dacă există
            if isinstance(result, dict):
                for key, value in result.items():
                    if key == 'svg' and isinstance(value, str):
                        svg_path = raster_dir / "output.svg"
                        with open(svg_path, 'w') as f:
                            f.write(value)
                        print(f"      📄 Salvat: {svg_path.name}")
                    elif key == 'dxf' and isinstance(value, str):
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
                        try:
                            img_str = value
                            if ',' in img_str:
                                img_str = img_str.split(',')[1]
                            img_data = base64.b64decode(img_str)
                            img_path = raster_dir / "processed_image.jpg"
                            with open(img_path, 'wb') as f:
                                f.write(img_data)
                            print(f"      📄 Salvat: {img_path.name}")
                        except Exception as e:
                            print(f"      ⚠️ Eroare salvare imagine: {e}")
            
            # Returnăm rezultatul cu scale_factor pentru transformări ulterioare
            return {
                'result': result,
                'scale_factor': scale_factor,
                'api_dimensions': (new_w_api, new_h_api),
                'raster_dir': raster_dir
            }
        else:
            print(f"      ⚠️ RasterScan API eroare: {response.status_code} - {response.text[:200]}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"      ⚠️ RasterScan API timeout (120s)")
        return None
    except Exception as e:
        print(f"      ⚠️ RasterScan API eroare: {e}")
        return None


def generate_raster_images(api_result: Dict[str, Any], original_img: np.ndarray, h_orig: int, w_orig: int) -> None:
    """
    Generează imagini din datele RasterScan API (walls, rooms, doors, combined, overlay, 3D render).
    
    Args:
        api_result: Rezultatul de la call_raster_api
        original_img: Imaginea originală (BGR)
        h_orig: Înălțimea imaginii originale
        w_orig: Lățimea imaginii originale
    """
    result = api_result['result']
    scale_factor = api_result['scale_factor']
    raster_dir = api_result['raster_dir']
    new_w_api, new_h_api = api_result['api_dimensions']
    
    data = result.get('data', result)
    
    # Funcție pentru scalare coordonate
    def scale_coord(x, y, for_original=False):
        """Scalează coordonatele înapoi la original"""
        if for_original:
            orig_x = int(x / scale_factor)
            orig_y = int(y / scale_factor)
            return orig_x, orig_y
        return int(x), int(y)
    
    raster_h, raster_w = new_h_api, new_w_api
    
    # Culori pentru camere
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
    
    # 1. Imagine cu pereții (generați din contururile camerelor)
    if 'rooms' in data and data['rooms']:
        walls_img = np.zeros((raster_h, raster_w, 3), dtype=np.uint8)
        walls_img.fill(255)
        
        wall_count = 0
        for room in data['rooms']:
            points = []
            for point in room:
                if 'x' in point and 'y' in point:
                    points.append([int(point['x']), int(point['y'])])
            
            if len(points) >= 3:
                pts = np.array(points, np.int32)
                cv2.polylines(walls_img, [pts], True, (0, 0, 0), 3)
                wall_count += len(points)
        
        walls_path = raster_dir / "walls.png"
        cv2.imwrite(str(walls_path), walls_img)
        print(f"      📄 Salvat: {walls_path.name} ({wall_count} segmente perete din {len(data['rooms'])} camere)")
    
    # 2. Imagine cu camerele (poligoane colorate)
    if 'rooms' in data and data['rooms']:
        rooms_img = np.zeros((raster_h, raster_w, 3), dtype=np.uint8)
        rooms_img.fill(255)
        
        for idx, room in enumerate(data['rooms']):
            color = room_colors[idx % len(room_colors)]
            points = []
            for point in room:
                if 'x' in point and 'y' in point:
                    points.append([int(point['x']), int(point['y'])])
            
            if len(points) >= 3:
                pts = np.array(points, np.int32)
                cv2.fillPoly(rooms_img, [pts], color)
                cv2.polylines(rooms_img, [pts], True, (0, 0, 0), 2)
        
        rooms_path = raster_dir / "rooms.png"
        cv2.imwrite(str(rooms_path), rooms_img)
        print(f"      📄 Salvat: {rooms_path.name} ({len(data['rooms'])} camere)")
    
    # 3. Imagine cu deschiderile (uși/ferestre)
    if 'doors' in data and data['doors']:
        doors_img = np.zeros((raster_h, raster_w, 3), dtype=np.uint8)
        doors_img.fill(255)
        
        for idx, door in enumerate(data['doors']):
            if 'bbox' in door and len(door['bbox']) == 4:
                x1, y1, x2, y2 = map(int, door['bbox'])
                width = x2 - x1
                height = y2 - y1
                
                aspect = width / max(1, height)
                if aspect > 2.5 or (width > 60 and height < 30):
                    label = "Window"
                    color_fill = (200, 220, 255)
                    color_border = (150, 180, 220)
                else:
                    label = "Door"
                    color_fill = (0, 150, 255)
                    color_border = (0, 100, 200)
                
                cv2.rectangle(doors_img, (x1, y1), (x2, y2), color_fill, -1)
                cv2.rectangle(doors_img, (x1, y1), (x2, y2), color_border, 2)
                
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
    
    if 'rooms' in data and data['rooms']:
        for idx, room in enumerate(data['rooms']):
            color = room_colors[idx % len(room_colors)]
            points = []
            for point in room:
                if 'x' in point and 'y' in point:
                    points.append([int(point['x']), int(point['y'])])
            if len(points) >= 3:
                pts = np.array(points, np.int32)
                cv2.fillPoly(combined_img, [pts], color)
        
        for room in data['rooms']:
            points = []
            for point in room:
                if 'x' in point and 'y' in point:
                    points.append([int(point['x']), int(point['y'])])
            if len(points) >= 3:
                pts = np.array(points, np.int32)
                cv2.polylines(combined_img, [pts], True, (0, 0, 0), 3)
    
    if 'doors' in data and data['doors']:
        for door in data['doors']:
            if 'bbox' in door and len(door['bbox']) == 4:
                x1, y1, x2, y2 = map(int, door['bbox'])
                cv2.rectangle(combined_img, (x1, y1), (x2, y2), (0, 150, 255), -1)
                cv2.rectangle(combined_img, (x1, y1), (x2, y2), (0, 100, 200), 2)
    
    combined_path = raster_dir / "combined.png"
    cv2.imwrite(str(combined_path), combined_img)
    print(f"      📄 Salvat: {combined_path.name}")
    
    # 5. Overlay pe imaginea originală
    overlay_img = original_img.copy()
    
    if 'rooms' in data and data['rooms']:
        rooms_overlay = np.zeros_like(overlay_img)
        for idx, room in enumerate(data['rooms']):
            color = room_colors[idx % len(room_colors)]
            points = []
            for point in room:
                if 'x' in point and 'y' in point:
                    ox, oy = scale_coord(point['x'], point['y'], for_original=True)
                    ox = max(0, min(ox, w_orig - 1))
                    oy = max(0, min(oy, h_orig - 1))
                    points.append([ox, oy])
            if len(points) >= 3:
                pts = np.array(points, np.int32)
                cv2.fillPoly(rooms_overlay, [pts], color)
        
        mask = (rooms_overlay.sum(axis=2) > 0).astype(np.uint8)
        mask = np.stack([mask, mask, mask], axis=2)
        overlay_img = np.where(mask, cv2.addWeighted(overlay_img, 0.6, rooms_overlay, 0.4, 0), overlay_img)
    
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
    
    if 'doors' in data and data['doors']:
        for door in data['doors']:
            if 'bbox' in door and len(door['bbox']) == 4:
                x1, y1, x2, y2 = door['bbox']
                ox1, oy1 = scale_coord(x1, y1, for_original=True)
                ox2, oy2 = scale_coord(x2, y2, for_original=True)
                
                width = abs(ox2 - ox1)
                height = abs(oy2 - oy1)
                aspect = width / max(1, height)
                if aspect > 2.5 or (width > 60 and height < 30):
                    label = "Win"
                    color = (220, 180, 0)
                else:
                    label = "Door"
                    color = (255, 100, 0)
                
                cv2.rectangle(overlay_img, (ox1, oy1), (ox2, oy2), color, 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(overlay_img, label, (ox1, oy1 - 5 if oy1 > 20 else oy2 + 15),
                           font, 0.4, color, 1)
    
    # 5. (opțional) Overlay / randare 3D
    #   Notă: pentru a reduce clutter-ul în folderul Raster, nu mai generăm
    #   fișierele overlay.png / render_3d.png aici. Vizualizările sunt produse
    #   ulterior în pașii dedicați (raster_processing / walls_from_coords).
    
    # Afișăm statistici
    if 'area' in data:
        print(f"      📊 Arie totală: {data['area']}")
    if 'perimeter' in data:
        print(f"      📊 Perimetru: {data['perimeter']:.2f}")


def _generate_3d_render(data: Dict[str, Any], raster_dir: Path, room_colors: list) -> None:
    """Generează render 3D izometric."""
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
    
    if not all_points:
        return
    
    all_points = np.array(all_points)
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)
    
    data_w = max_x - min_x
    data_h = max_y - min_y
    
    wall_height = 60
    canvas_w = int(data_w * 1.5 + data_h * 0.5 + 200)
    canvas_h = int(data_h * 0.7 + wall_height + 150)
    
    offset_x = 50
    offset_y = wall_height + 30
    
    def to_iso_3d(x, y, z=0):
        nx = x - min_x
        ny = y - min_y
        iso_x = int(offset_x + nx + ny * 0.4)
        iso_y = int(offset_y + ny * 0.6 - z)
        return (iso_x, iso_y)
    
    iso_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    for row in range(canvas_h):
        ratio = row / canvas_h
        b = int(250 - ratio * 20)
        g = int(248 - ratio * 15)
        r = int(245 - ratio * 10)
        iso_img[row, :] = [b, g, r]
    
    iso_room_colors = [
        (210, 225, 210), (210, 210, 225), (225, 210, 210),
        (225, 225, 210), (210, 225, 225), (225, 210, 225), (220, 220, 220),
    ]
    
    if 'rooms' in data and data['rooms']:
        sorted_rooms = []
        for idx, room in enumerate(data['rooms']):
            points = []
            for point in room:
                if 'x' in point and 'y' in point:
                    points.append([int(point['x']), int(point['y'])])
            if len(points) >= 3:
                min_room_y = min(p[1] for p in points)
                sorted_rooms.append((min_room_y, idx, points))
        
        sorted_rooms.sort(key=lambda x: x[0])
        
        for min_room_y, idx, points in sorted_rooms:
            color = iso_room_colors[idx % len(iso_room_colors)]
            floor_pts = np.array([to_iso_3d(p[0], p[1], 0) for p in points], np.int32)
            cv2.fillPoly(iso_img, [floor_pts], color)
            cv2.polylines(iso_img, [floor_pts], True, (180, 180, 180), 1)
    
    if 'walls' in data and data['walls']:
        sorted_walls = []
        for wall in data['walls']:
            if 'position' in wall and len(wall['position']) >= 2:
                pt1 = wall['position'][0]
                pt2 = wall['position'][1]
                min_wall_y = min(pt1[1], pt2[1])
                sorted_walls.append((min_wall_y, pt1, pt2))
        
        sorted_walls.sort(key=lambda x: x[0])
        
        for min_wall_y, pt1, pt2 in sorted_walls:
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]), int(pt2[1])
            
            bl = to_iso_3d(x1, y1, 0)
            br = to_iso_3d(x2, y2, 0)
            tl = to_iso_3d(x1, y1, wall_height)
            tr = to_iso_3d(x2, y2, wall_height)
            
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
            if dy < dx:
                wall_color = (230, 230, 230)
            else:
                wall_color = (200, 200, 200)
            
            wall_pts = np.array([bl, br, tr, tl], np.int32)
            cv2.fillPoly(iso_img, [wall_pts], wall_color)
            cv2.polylines(iso_img, [wall_pts], True, (120, 120, 120), 1)
            
            thickness_offset = 6
            if dy < dx:
                tl2 = to_iso_3d(x1, y1 + thickness_offset, wall_height)
                tr2 = to_iso_3d(x2, y2 + thickness_offset, wall_height)
                top_pts = np.array([tl, tr, tr2, tl2], np.int32)
                cv2.fillPoly(iso_img, [top_pts], (240, 240, 240))
                cv2.polylines(iso_img, [top_pts], True, (150, 150, 150), 1)
            else:
                tl2 = to_iso_3d(x1 + thickness_offset, y1, wall_height)
                tr2 = to_iso_3d(x2 + thickness_offset, y2, wall_height)
                top_pts = np.array([tl, tr, tr2, tl2], np.int32)
                cv2.fillPoly(iso_img, [top_pts], (240, 240, 240))
                cv2.polylines(iso_img, [top_pts], True, (150, 150, 150), 1)
    
    iso_path = raster_dir / "render_3d.png"
    cv2.imwrite(str(iso_path), iso_img)
    print(f"      📄 Salvat: {iso_path.name}")


def generate_api_walls_mask(api_result: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Generează masca de pereți din imaginea procesată de API.
    
    Args:
        api_result: Rezultatul de la call_raster_api
    
    Returns:
        Masca de pereți (grayscale) sau None dacă a eșuat
    """
    result = api_result['result']
    raster_dir = api_result['raster_dir']
    
    try:
        if 'image' not in result.get('data', result):
            return None
        
        img_str = result['data']['image']
        if ',' in img_str:
            img_str = img_str.split(',')[1]
        img_data = base64.b64decode(img_str)
        nparr = np.frombuffer(img_data, np.uint8)
        api_processed_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if api_processed_img is None:
            return None
        
        # Detectăm pereții din imaginea API (gri, nu colorați)
        api_gray = cv2.cvtColor(api_processed_img, cv2.COLOR_BGR2GRAY)
        api_hsv = cv2.cvtColor(api_processed_img, cv2.COLOR_BGR2HSV)
        saturation = api_hsv[:, :, 1]
        
        # Pixelii cu saturație mică și gri mediu sunt pereți
        api_walls_mask = ((api_gray > 100) & (api_gray < 180) & (saturation < 30)).astype(np.uint8) * 255
        
        api_walls_path = raster_dir / "api_walls_mask.png"
        cv2.imwrite(str(api_walls_path), api_walls_mask)
        print(f"      📄 Salvat: {api_walls_path.name}")
        
        return api_walls_mask
        
    except Exception as e:
        print(f"      ⚠️ Eroare generare api_walls_mask: {e}")
        return None


def validate_api_walls_mask(
    api_walls_mask: np.ndarray,
    rooms: List,
    min_interior_area: int = 5000,
    max_wall_ratio_in_room: float = 0.30,
) -> Tuple[bool, str]:
    """
    Verifică dacă masca de pereți nu are camere "inundate" (interior plin de pixeli perete).
    Folosește poligoanele camerelor din răspunsul API ca referință pentru interior.

    Returns:
        (is_valid, details): False dacă măcar o cameră are ratio pereți/interior > max_wall_ratio_in_room.
    """
    h, w = api_walls_mask.shape[:2]
    if not rooms:
        return True, "no rooms to validate"

    for idx, room in enumerate(rooms):
        points = []
        for pt in room:
            if isinstance(pt, dict) and "x" in pt and "y" in pt:
                x_val = int(pt["x"])
                y_val = int(pt["y"])
                x_val = max(0, min(x_val, w - 1))
                y_val = max(0, min(y_val, h - 1))
                points.append([x_val, y_val])
        if len(points) < 3:
            continue
        pts = np.array(points, dtype=np.int32)
        room_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(room_mask, [pts], 255)
        # Erodăm marginea ca să luăm doar interiorul (excludem pereții)
        kernel = np.ones((5, 5), np.uint8)
        interior = cv2.erode(room_mask, kernel)
        interior_area = int(np.count_nonzero(interior))
        if interior_area < min_interior_area:
            continue
        wall_inside = int(np.count_nonzero((api_walls_mask > 0) & (interior > 0)))
        ratio = wall_inside / interior_area
        if ratio > max_wall_ratio_in_room:
            return False, f"room {idx} ratio {ratio:.2f} (>{max_wall_ratio_in_room})"
    return True, "ok"


def brute_force_alignment(
    api_walls_mask: np.ndarray,
    orig_walls: np.ndarray,
    raster_dir: Path,
    steps_dir: str
) -> Optional[Dict[str, Any]]:
    """
    Algoritm brute-force pentru alinierea măștilor de pereți API și original.
    
    Args:
        api_walls_mask: Masca de pereți de la API (grayscale)
        orig_walls: Masca de pereți originală (grayscale)
        raster_dir: Directorul raster
        steps_dir: Directorul pentru steps
    
    Returns:
        Dict cu configurația cea mai bună sau None dacă a eșuat
    """
    try:
        print(f"\n      🔥 BRUTE FORCE: Căutare transformare între API walls și original walls...")
        
        print(f"      📊 API walls: {api_walls_mask.shape[1]} x {api_walls_mask.shape[0]}")
        print(f"      📊 Original walls: {orig_walls.shape[1]} x {orig_walls.shape[0]}")
        
        # Binarizare
        _, binary_api = cv2.threshold(api_walls_mask, 127, 255, cv2.THRESH_BINARY_INV)
        _, binary_orig = cv2.threshold(orig_walls, 127, 255, cv2.THRESH_BINARY_INV)
        
        # PARAMETRI BRUTE FORCE (FĂRĂ ROTAȚII)
        scales = np.arange(0.3, 3.5, 0.05)
        
        print(f"      📊 Scale-uri: {len(scales)} ({scales[0]:.2f} - {scales[-1]:.2f})")
        print(f"      📊 Rotații: 1 (0° - fără rotație)")
        print(f"      📊 Total teste: {len(scales) * 2} (API→Orig + Orig→API)")
        
        top_results = []
        
        def add_to_top_results(config, max_results=10):
            top_results.append(config)
            top_results.sort(key=lambda x: x['score'], reverse=True)
            if len(top_results) > max_results:
                top_results.pop()
        
        # Testare API -> Original
        print(f"      🚀 Testare API walls → Original walls...")
        total = len(scales)
        tested = 0
        
        for idx, scale in enumerate(scales):
            tested += 1
            
            if idx % 10 == 0 or tested == 1:
                print(f"         ⏳ Test {tested}/{total}: scale={scale:.2f}x...")
            
            api_rot = binary_api.copy()
            
            # Scalăm
            new_w = int(api_rot.shape[1] * scale)
            new_h = int(api_rot.shape[0] * scale)
            
            if new_w > binary_orig.shape[1] or new_h > binary_orig.shape[0]:
                if idx % 10 == 0:
                    print(f"            ⚠️ Skip (prea mare: {new_w}x{new_h})")
                continue
            if new_w < 30 or new_h < 30:
                if idx % 10 == 0:
                    print(f"            ⚠️ Skip (prea mic: {new_w}x{new_h})")
                continue
            
            api_scaled = cv2.resize(api_rot, (new_w, new_h))
            
            # Template matching
            result_match = cv2.matchTemplate(binary_orig, api_scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result_match)
            
            config = {
                'direction': 'api_to_orig',
                'scale': float(scale),
                'rotation': 0,
                'position': (int(max_loc[0]), int(max_loc[1])),
                'score': float(max_val),
                'template_size': (int(new_w), int(new_h))
            }
            add_to_top_results(config)
            
            if idx % 10 == 0:
                print(f"            ✅ Score: {max_val:.4f} (Best so far: {top_results[0]['score']:.4f})")
        
        if top_results:
            print(f"      ✅ Finalizat API→Orig: {tested}/{total} teste, best score: {top_results[0]['score']:.4f}")
        else:
            print(f"      ⚠️ Nu s-au găsit rezultate valide pentru API→Orig")
        
        # Testare Original -> API
        print(f"      🚀 Testare Original walls → API walls...")
        tested = 0
        
        for idx, scale in enumerate(scales):
            tested += 1
            
            if idx % 10 == 0 or tested == 1:
                print(f"         ⏳ Test {tested}/{total}: scale={scale:.2f}x...")
            
            orig_rot = binary_orig.copy()
            
            # Scalăm
            new_w = int(orig_rot.shape[1] * scale)
            new_h = int(orig_rot.shape[0] * scale)
            
            if new_w > binary_api.shape[1] or new_h > binary_api.shape[0]:
                if idx % 10 == 0:
                    print(f"            ⚠️ Skip (prea mare: {new_w}x{new_h})")
                continue
            if new_w < 30 or new_h < 30:
                if idx % 10 == 0:
                    print(f"            ⚠️ Skip (prea mic: {new_w}x{new_h})")
                continue
            
            orig_scaled = cv2.resize(orig_rot, (new_w, new_h))
            
            # Template matching
            result_match = cv2.matchTemplate(binary_api, orig_scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result_match)
            
            config = {
                'direction': 'orig_to_api',
                'scale': float(scale),
                'rotation': 0,
                'position': (int(max_loc[0]), int(max_loc[1])),
                'score': float(max_val),
                'template_size': (int(new_w), int(new_h))
            }
            add_to_top_results(config)
            
            if idx % 10 == 0:
                print(f"            ✅ Score: {max_val:.4f} (Best so far: {top_results[0]['score']:.4f})")
        
        if not top_results:
            print(f"      ⚠️ Nu s-au găsit rezultate valide pentru brute force")
            return None
        
        print(f"      ✅ Finalizat Orig→API: {tested}/{total} teste, best score: {top_results[0]['score']:.4f}")
        
        # Găsim cel mai bun rezultat
        best = top_results[0]
        print(f"\n      🏆 CEL MAI BUN REZULTAT:")
        print(f"         Score: {best['score']:.4f}")
        print(f"         Direcție: {best['direction']}")
        print(f"         Scale: {best['scale']:.3f}x")
        print(f"         Poziție: {best['position']}")
        print(f"         Template size: {best['template_size']}")
        
        # Salvăm configurația
        config_path = raster_dir / "brute_force_best_config.json"
        with open(config_path, 'w') as f:
            json.dump(best, f, indent=2)
        print(f"      📄 Salvat: {config_path.name}")
        
        # Generăm vizualizare pentru cel mai bun rezultat
        if best['direction'] == 'api_to_orig':
            base_img = orig_walls
            base_binary = binary_orig
            template_img = api_walls_mask
            template_binary = binary_api
        else:
            base_img = api_walls_mask
            base_binary = binary_api
            template_img = orig_walls
            template_binary = binary_orig
        
        # Aplicăm transformarea (fără rotație)
        tw, th = best['template_size']
        template_scaled = cv2.resize(template_binary, (tw, th))
        
        x_pos, y_pos = best['position']
        
        # Overlay binar
        overlay_binary = np.zeros((base_binary.shape[0], base_binary.shape[1], 3), dtype=np.uint8)
        overlay_binary[:, :, 2] = base_binary  # Red
        overlay_binary[y_pos:y_pos+th, x_pos:x_pos+tw, 1] = template_scaled  # Green
        overlay_binary[y_pos:y_pos+th, x_pos:x_pos+tw, 0] = template_scaled  # Blue
        
        best_overlay_path = raster_dir / "brute_force_best_overlay.png"
        cv2.imwrite(str(best_overlay_path), overlay_binary)
        print(f"      📄 Salvat: {best_overlay_path.name}")
        
        return best
        
    except Exception as e:
        import traceback
        print(f"      ⚠️ Eroare brute force: {e}")
        traceback.print_exc()
        return None


def apply_alignment_and_generate_overlay(
    best_config: Dict[str, Any],
    api_result: Dict[str, Any],
    original_img: np.ndarray,
    steps_dir: str
) -> Optional[Dict[str, Any]]:
    """
    Aplică transformarea găsită de brute-force și generează overlay-ul pe original.
    
    Args:
        best_config: Configurația cea mai bună de la brute_force_alignment
        api_result: Rezultatul de la call_raster_api
        original_img: Imaginea originală (BGR)
        steps_dir: Directorul pentru steps
    
    Returns:
        Dict cu funcția de transformare coordonate sau None dacă a eșuat
    """
    try:
        raster_dir = api_result['raster_dir']
        response_json_path = raster_dir / "response.json"
        
        if not response_json_path.exists():
            print(f"      ⚠️ response.json nu există")
            return None
        
        print(f"\n      🎯 Transformare coordonate și generare overlay pe original...")
        
        # Încărcăm response.json pentru a obține data
        with open(response_json_path, 'r') as f:
            result_data = json.load(f)
        
        data = result_data.get('data', result_data)
        
        # Funcție de transformare coordonate API -> Original (FĂRĂ ROTAȚII)
        def api_to_original_coords(x, y):
            """Transformă coordonate din sistemul API la original"""
            if best_config['direction'] == 'api_to_orig':
                # Transformare directă: API -> Original
                x_scaled = x * best_config['scale']
                y_scaled = y * best_config['scale']
                orig_x = x_scaled + best_config['position'][0]
                orig_y = y_scaled + best_config['position'][1]
                return int(orig_x), int(orig_y)
            else:
                # orig_to_api - inversăm transformarea
                x_in_template = x - best_config['position'][0]
                y_in_template = y - best_config['position'][1]
                orig_x = x_in_template / best_config['scale']
                orig_y = y_in_template / best_config['scale']
                return int(orig_x), int(orig_y)
        
        # Desenăm rooms și doors pe original
        overlay_orig = original_img.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        # Rooms
        if 'rooms' in data and data['rooms']:
            for i, room in enumerate(data['rooms']):
                pts = []
                for point in room:
                    if 'x' in point and 'y' in point:
                        ox, oy = api_to_original_coords(point['x'], point['y'])
                        pts.append((ox, oy))
                
                if len(pts) >= 3:
                    pts_np = np.array(pts, dtype=np.int32)
                    color = colors[i % len(colors)]
                    cv2.polylines(overlay_orig, [pts_np], True, color, 4)
                    
                    # Label
                    if pts:
                        cx = sum(p[0] for p in pts) // len(pts)
                        cy = sum(p[1] for p in pts) // len(pts)
                        cv2.putText(overlay_orig, f'Room {i}', (cx-50, cy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Doors
        if 'doors' in data and data['doors']:
            for door in data['doors']:
                if 'bbox' in door and len(door['bbox']) == 4:
                    bbox = door['bbox']
                    x1, y1 = api_to_original_coords(bbox[0], bbox[1])
                    x2, y2 = api_to_original_coords(bbox[2], bbox[3])
                    cv2.rectangle(overlay_orig, (x1, y1), (x2, y2), (0, 165, 255), 3)
                    cv2.putText(overlay_orig, 'Door', (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        # Salvăm overlay-ul pe original
        overlay_orig_path = raster_dir / "overlay_on_original.png"
        cv2.imwrite(str(overlay_orig_path), overlay_orig)
        print(f"      📄 Salvat: {overlay_orig_path.name}")
        
        print(f"      ✅ Transformare coordonate completă!")
        
        return {
            'api_to_original_coords': api_to_original_coords,
            'best_config': best_config
        }
        
    except Exception as e:
        import traceback
        print(f"      ⚠️ Eroare aplicare transformare: {e}")
        traceback.print_exc()
        return None


def brute_force_alignment_for_walls_image(
    walls_img: np.ndarray,
    orig_walls: np.ndarray,
    raster_dir: Path,
    steps_dir: str
) -> Optional[Dict[str, Any]]:
    """
    Algoritm brute-force pentru alinierea imaginii walls.png (colorată) cu pereții originali.
    Similar cu brute_force_alignment dar pentru imaginea colorată walls.png.
    
    Args:
        walls_img: Imaginea walls.png de la API (BGR, colorată)
        orig_walls: Masca de pereți originală (grayscale)
        raster_dir: Directorul raster
        steps_dir: Directorul pentru steps
    
    Returns:
        Dict cu configurația cea mai bună sau None dacă a eșuat
    """
    try:
        print(f"\n      🔥 BRUTE FORCE: Căutare transformare între walls.png și original walls...")
        
        # Convertim walls.png la grayscale pentru matching
        walls_gray = cv2.cvtColor(walls_img, cv2.COLOR_BGR2GRAY)
        
        # Detectăm pereții din walls.png (similar cu api_walls_mask)
        # Pereții sunt de obicei gri sau colorați, dar nu alb
        # Folosim o metodă similară cu cea din call_raster_api
        api_hsv = cv2.cvtColor(walls_img, cv2.COLOR_BGR2HSV)
        saturation = api_hsv[:, :, 1]
        
        # Pixelii cu saturație mică și gri mediu sunt pereți
        walls_mask = ((walls_gray > 100) & (walls_gray < 180) & (saturation < 30)).astype(np.uint8) * 255
        
        # Folosim funcția existentă brute_force_alignment
        return brute_force_alignment(walls_mask, orig_walls, raster_dir, steps_dir)
        
    except Exception as e:
        import traceback
        print(f"      ⚠️ Eroare brute force pentru walls.png: {e}")
        traceback.print_exc()
        return None


def generate_walls_overlay_on_original(
    best_config: Dict[str, Any],
    walls_img: np.ndarray,
    original_img: np.ndarray,
    raster_dir: Path
) -> bool:
    """
    Generează overlay-ul walls.png peste imaginea originală folosind transformarea găsită.
    
    Args:
        best_config: Configurația cea mai bună de la brute_force_alignment
        walls_img: Imaginea walls.png de la API (BGR, colorată)
        original_img: Imaginea originală (BGR)
        raster_dir: Directorul raster
    
    Returns:
        True dacă a reușit, False altfel
    """
    try:
        print(f"\n      🎯 Generez overlay walls.png peste original...")
        
        # Funcție de transformare coordonate
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
        
        # Scalăm walls_img conform transformării
        scale = best_config['scale']
        new_w = int(walls_img.shape[1] * scale)
        new_h = int(walls_img.shape[0] * scale)
        
        walls_scaled = cv2.resize(walls_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Creăm overlay-ul
        overlay = original_img.copy()
        x_pos, y_pos = best_config['position']
        
        # Verificăm limitele
        h_orig, w_orig = original_img.shape[:2]
        y_end = min(y_pos + new_h, h_orig)
        x_end = min(x_pos + new_w, w_orig)
        y_start = max(0, y_pos)
        x_start = max(0, x_pos)
        
        # Ajustăm și walls_scaled dacă e necesar
        if y_start > y_pos or x_start > x_pos:
            y_offset = y_start - y_pos
            x_offset = x_start - x_pos
            walls_scaled = walls_scaled[y_offset:, x_offset:]
        
        if y_end < y_pos + new_h or x_end < x_pos + new_w:
            walls_scaled = walls_scaled[:y_end-y_start, :x_end-x_start]
        
        # Suprapunem cu transparență
        if walls_scaled.shape[0] > 0 and walls_scaled.shape[1] > 0:
            # Creăm o mască pentru a exclude fundalul alb
            walls_mask = np.all(walls_scaled != [255, 255, 255], axis=2).astype(np.uint8)
            
            # Suprapunem doar unde nu e fundal alb
            overlay[y_start:y_end, x_start:x_end] = cv2.addWeighted(
                overlay[y_start:y_end, x_start:x_end], 0.7,
                walls_scaled, 0.3, 0
            )
        
        # Salvăm overlay-ul
        overlay_path = raster_dir / "walls_overlay_on_original.png"
        cv2.imwrite(str(overlay_path), overlay)
        print(f"      📄 Salvat: {overlay_path.name}")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"      ⚠️ Eroare generare overlay walls.png: {e}")
        traceback.print_exc()
        return False


def generate_crop_from_raster(
    best_config: Dict[str, Any],
    api_walls_mask: np.ndarray,
    original_img: np.ndarray,
    api_result: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Generează crop din 00_original.png bazat pe masca RasterScan.
    
    Args:
        best_config: Configurația cea mai bună de la brute_force_alignment
        api_walls_mask: Masca de pereți de la API
        original_img: Imaginea originală (BGR)
        api_result: Rezultatul de la call_raster_api
    
    Returns:
        Dict cu informații despre crop sau None dacă a eșuat
    """
    try:
        raster_dir = api_result['raster_dir']
        
        print(f"\n      🎯 Generez crop din 00_original.png bazat pe masca RasterScan...")
        
        # Funcție de transformare coordonate
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
        
        # Calculăm bounding box-ul din api_walls_mask în coordonatele originale
        api_h, api_w = api_walls_mask.shape[:2]
        
        if best_config['direction'] == 'api_to_orig':
            x1_api, y1_api = 0, 0
            x2_api, y2_api = api_w, api_h
            x1_orig, y1_orig = api_to_original_coords(x1_api, y1_api)
            x2_orig, y2_orig = api_to_original_coords(x2_api, y2_api)
            crop_x1 = min(x1_orig, x2_orig)
            crop_y1 = min(y1_orig, y2_orig)
            crop_x2 = max(x1_orig, x2_orig)
            crop_y2 = max(y1_orig, y2_orig)
        else:
            x_pos, y_pos = best_config['position']
            tw, th = best_config['template_size']
            x1_api, y1_api = x_pos, y_pos
            x2_api, y2_api = x_pos + tw, y_pos + th
            x1_orig, y1_orig = api_to_original_coords(x1_api, y1_api)
            x2_orig, y2_orig = api_to_original_coords(x2_api, y2_api)
            crop_x1 = min(x1_orig, x2_orig)
            crop_y1 = min(y1_orig, y2_orig)
            crop_x2 = max(x1_orig, x2_orig)
            crop_y2 = max(y1_orig, y2_orig)
        
        # Asigurăm că crop-ul este în limitele imaginii originale
        orig_h, orig_w = original_img.shape[:2]
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(orig_w, crop_x2)
        crop_y2 = min(orig_h, crop_y2)
        
        # Generez crop-ul
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1
        
        if crop_width > 0 and crop_height > 0:
            original_crop = original_img[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Salvăm crop-ul
            crop_path = raster_dir / "00_original_crop.png"
            cv2.imwrite(str(crop_path), original_crop)
            print(f"      📄 Salvat crop: {crop_path.name} ({crop_width}x{crop_height}px, offset: {crop_x1},{crop_y1})")
            
            # Salvăm și informațiile despre crop
            crop_info = {
                "x": int(crop_x1),
                "y": int(crop_y1),
                "width": int(crop_width),
                "height": int(crop_height),
                "original_width": int(orig_w),
                "original_height": int(orig_h)
            }
            crop_info_path = raster_dir / "crop_info.json"
            with open(crop_info_path, 'w') as f:
                json.dump(crop_info, f, indent=2)
            print(f"      📄 Salvat crop info: {crop_info_path.name}")
            
            return crop_info
        else:
            print(f"      ⚠️ Crop invalid: {crop_width}x{crop_height}px")
            return None
            
    except Exception as e:
        import traceback
        print(f"      ⚠️ Eroare generare crop: {e}")
        traceback.print_exc()
        return None
