# file: engine/cubicasa_detector/raster_api.py
"""
Module pentru integrarea cu RasterScan API.
Conține funcții pentru apelul API, generarea imaginilor, alinierea brute-force și generarea crop-ului.
"""

from __future__ import annotations

import os
import time
import threading
import cv2
import numpy as np
import json
import base64
import requests
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from .alignment_methods import (
    align_brute_force_pyramid,
    build_config_from_pyramid,
)

# Dimensiune maximă pentru overlay-uri la algoritmii extra (folosit și în brute_force_alignment)
MAX_OVERLAY_OUTPUT_SIDE_EXTRA = 1200


def run_extra_alignment_methods(
    raster_dir: Path,
    binary_orig: np.ndarray,
    binary_api: np.ndarray,
) -> None:
    """
    Rulează cei 3 algoritmi (Log-Polar FFT, Affine ECC, Coarse-to-Fine Pyramid) și salvează
    rezultatele în raster_dir/brute_steps/. Poate fi apelat și când se folosește cache-ul
    brute force, astfel încât outputul celor 3 metode să fie mereu disponibil.
    """
    brute_steps_dir = raster_dir / "brute_steps"
    brute_steps_dir.mkdir(parents=True, exist_ok=True)

    def _resize(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) <= MAX_OVERLAY_OUTPUT_SIDE_EXTRA:
            return img
        scale = MAX_OVERLAY_OUTPUT_SIDE_EXTRA / max(h, w)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    def _save_overlay(base_binary: np.ndarray, template_binary: np.ndarray, config: Dict[str, Any], path: Path) -> None:
        tw, th = config["template_size"]
        template_scaled = cv2.resize(template_binary, (tw, th), interpolation=cv2.INTER_NEAREST)
        x_pos, y_pos = config["position"]
        h, w = base_binary.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[:, :, 2] = base_binary
        y_end = min(y_pos + th, h)
        x_end = min(x_pos + tw, w)
        overlay[y_pos:y_end, x_pos:x_end, 0] = template_scaled[: y_end - y_pos, : x_end - x_pos]
        overlay[y_pos:y_end, x_pos:x_end, 1] = template_scaled[: y_end - y_pos, : x_end - x_pos]
        cv2.imwrite(str(path), _resize(overlay))

    alignment_results: Dict[str, Any] = {}

    # Doar Coarse-to-Fine Pyramid (Log-Polar și ECC dezactivate)
    print(f"      📐 Coarse-to-Fine Pyramid: rulează (scale + poziție, poate dura 1–2 min)...")
    try:
        scale_py, tx_py, ty_py, iou_py = align_brute_force_pyramid(binary_orig, binary_api)
        cfg_py = build_config_from_pyramid(
            scale_py, tx_py, ty_py, iou_py, binary_api, direction="api_to_orig"
        )
        alignment_results["coarse_to_fine_pyramid"] = {
            "scale": scale_py,
            "position": (tx_py, ty_py),
            "template_size": cfg_py["template_size"],
            "iou": iou_py,
            "score": iou_py,
        }
        _save_overlay(
            binary_orig,
            binary_api,
            cfg_py,
            brute_steps_dir / f"align_pyramid_scale_{scale_py:.3f}_iou_{iou_py:.3f}.png",
        )
        print(f"      📐 Coarse-to-Fine Pyramid: scară {scale_py:.4f}, pos ({tx_py}, {ty_py}), IoU {iou_py:.2%}")
    except Exception as e:
        alignment_results["coarse_to_fine_pyramid"] = None
        print(f"      📐 Coarse-to-Fine Pyramid: excepție – {e}")

    try:
        with open(brute_steps_dir / "alignment_results.json", "w", encoding="utf-8") as f:
            json.dump(alignment_results, f, indent=2)
        print(f"      📄 Rezultate aliniere (piramidă): {brute_steps_dir.name}/alignment_results.json")
    except OSError as e:
        if e.errno != 28:
            raise


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
        try:
            if not cv2.imwrite(str(preprocessed_path), api_img):
                print(f"      ⚠️ Nu s-a putut salva {preprocessed_path.name}")
        except OSError as e:
            if e.errno == 28:
                print(f"      ⚠️ Disc plin: nu s-a salvat {preprocessed_path.name}")
            else:
                raise
        else:
            print(f"      💾 Salvat: {preprocessed_path.name} (preprocesat - linii subțiri eliminate)")
        
        # Scale-down doar pentru trimiterea la Raster: max 1000px pe latura lungă (restul pipeline-ului nu se atinge)
        MAX_RASTER_SIDE = 1000
        h_api, w_api = api_img.shape[:2]
        scale_factor = 1.0
        if max(h_api, w_api) > MAX_RASTER_SIDE:
            scale_factor = MAX_RASTER_SIDE / max(h_api, w_api)
            new_w_api = max(1, int(w_api * scale_factor))
            new_h_api = max(1, int(h_api * scale_factor))
            api_img = cv2.resize(api_img, (new_w_api, new_h_api), interpolation=cv2.INTER_AREA)
            print(f"      📐 Scale-down pentru Raster (max {MAX_RASTER_SIDE}px): {w_api}x{h_api} -> {new_w_api}x{new_h_api}")
        else:
            new_w_api, new_h_api = w_api, h_api

        # Salvăm imaginea care se trimite la API (scale/raster) – nume explicit ca să fie ușor de găsit
        try:
            api_img_path = raster_dir / "input_resized.jpg"
            cv2.imwrite(str(api_img_path), api_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            request_png_path = raster_dir / "raster_request.png"
            if not cv2.imwrite(str(request_png_path), api_img):
                print(f"      ⚠️ Nu s-a putut salva {request_png_path.name}")
            else:
                print(f"      📄 Salvat (trimis la Raster): {request_png_path.name}")
        except OSError as e:
            if e.errno == 28:
                print(f"      ⚠️ Disc plin: nu s-a salvat request image")
            else:
                raise

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
        
        max_attempts = 6
        response = None
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    time.sleep(2 * attempt)
                    print(f"      🔄 Reîncerc RasterScan API ({attempt + 1}/{max_attempts})...")
                response = requests.post(url, json=payload, headers=headers, timeout=120)
            except requests.exceptions.Timeout:
                print(f"      ⚠️ RasterScan API timeout (120s)")
                if attempt < max_attempts - 1:
                    continue
                return None
            if response.status_code == 200:
                break
            is_retryable = response.status_code >= 500 or response.status_code == 429
            print(f"      ⚠️ RasterScan API eroare: {response.status_code} - {response.text[:200]}")
            if not is_retryable or attempt >= max_attempts - 1:
                return None
        
        if response is not None and response.status_code == 200:
            try:
                result = response.json()
            except Exception as e:
                print(f"      ⚠️ RasterScan API: eroare parsare JSON: {e}")
                return None
            print(f"      ✅ RasterScan API răspuns primit")
            
            # Salvăm răspunsul JSON
            json_path = raster_dir / "response.json"
            try:
                with open(json_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"      📄 Salvat: {json_path.name}")
            except OSError as e:
                if e.errno == 28:
                    print(f"      ⚠️ Disc plin: nu s-a salvat {json_path.name}")
                else:
                    raise
            
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

            # Fallback: imaginea poate fi în result['data']['image'] sau result['processed_image']
            if isinstance(result, dict):
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
                                    print(f"      📄 Salvat (răspuns de la Raster, fallback): {response_png_path.name}")
                                    break
                            except Exception:
                                pass

            # Salvăm dimensiunile request vs original ca să putem converti coordonatele din JSON (request space) în original
            orig_w = max(1, int(round(new_w_api / scale_factor)))
            orig_h = max(1, int(round(new_h_api / scale_factor)))
            request_info = {
                "request_w": int(new_w_api), "request_h": int(new_h_api),
                "original_w": orig_w, "original_h": orig_h,
                "scale_factor": float(scale_factor)
            }
            # Detectăm dacă Raster a returnat o imagine cu dimensiuni diferite (crop intern) – API-ul nu expune bbox crop
            response_png_path = raster_dir / "raster_response.png"
            if response_png_path.exists():
                resp_img = cv2.imread(str(response_png_path))
                if resp_img is not None:
                    resp_h, resp_w = resp_img.shape[:2]
                    if (resp_w != new_w_api) or (resp_h != new_h_api):
                        request_info["response_image_w"] = int(resp_w)
                        request_info["response_image_h"] = int(resp_h)
                        request_info["raster_may_crop"] = True
                        print(f"      ⚠️ Raster a returnat imagine {resp_w}x{resp_h} (request: {new_w_api}x{new_h_api}) – posibil crop intern; alinierea pe original poate fi incorectă.")
            request_info_path = raster_dir / "raster_request_info.json"
            try:
                with open(request_info_path, 'w') as f:
                    json.dump(request_info, f, indent=2)
            except OSError as e:
                if e.errno == 28:
                    print(f"      ⚠️ Disc plin: nu s-a salvat {request_info_path.name}")
                else:
                    raise

            # Overlay pereți/camere/uși pe imaginea cu scale-down trimisă la Raster (coordonate 1:1, fără scalare)
            if not save_overlay_on_request_image(raster_dir):
                print(f"      ⚠️ overlay_on_request.png nu s-a putut genera")
            # Mască pereți din JSON în spațiul request (pentru aliniere consistentă când Raster face crop)
            if build_api_walls_mask_from_json(raster_dir, new_w_api, new_h_api) is not None:
                pass  # salvat ca api_walls_from_json.png

            # Returnăm rezultatul cu scale_factor pentru transformări ulterioare
            return {
                'result': result,
                'scale_factor': scale_factor,
                'api_dimensions': (new_w_api, new_h_api),
                'raster_dir': raster_dir
            }
        return None
            
    except requests.exceptions.Timeout:
        print(f"      ⚠️ RasterScan API timeout (120s) - toate încercările eșuate")
        return None
    except Exception as e:
        print(f"      ⚠️ RasterScan API eroare: {e}")
        return None


def save_overlay_on_request_image(raster_dir: Path) -> bool:
    """
    Desenează pereți, camere și uși din response.json pe imaginea cu scale-down care se trimite la Raster.
    Coordonatele din JSON sunt exact în spațiul acestei imagini (request space) – fără nicio scalare.

    Salvează: overlay_on_request.png (pereții/camerele/ușile pe raster_request.png).

    Returns:
        True dacă overlay-ul a fost salvat, False altfel.
    """
    try:
        request_img = cv2.imread(str(raster_dir / "raster_request.png"))
        if request_img is None:
            request_img = cv2.imread(str(raster_dir / "input_resized.jpg"))
        if request_img is None:
            return False
        response_path = raster_dir / "response.json"
        if not response_path.exists():
            return False
        with open(response_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        data = result.get("data", result)
        h_req, w_req = request_img.shape[:2]
        overlay = request_img.copy()

        def pt(x: float, y: float):
            return (int(round(x)), int(round(y)))

        colors_rooms = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

        # Pereți (segmente) – desenați primii, sub camere
        if "walls" in data and data["walls"]:
            for wall in data["walls"]:
                pos = wall.get("position")
                if pos and len(pos) >= 2:
                    p1 = pos[0]
                    p2 = pos[1]
                    x1, y1 = (p1["x"], p1["y"]) if isinstance(p1, dict) else (p1[0], p1[1])
                    x2, y2 = (p2["x"], p2["y"]) if isinstance(p2, dict) else (p2[0], p2[1])
                    cv2.line(overlay, pt(x1, y1), pt(x2, y2), (0, 255, 0), 3)

        # Camere (poligoane)
        if "rooms" in data and data["rooms"]:
            for i, room in enumerate(data["rooms"]):
                pts = []
                for point in room:
                    if "x" in point and "y" in point:
                        pts.append(pt(point["x"], point["y"]))
                if len(pts) >= 3:
                    pts_np = np.array(pts, dtype=np.int32)
                    color = colors_rooms[i % len(colors_rooms)]
                    cv2.polylines(overlay, [pts_np], True, color, 2)
                    if pts:
                        cx = sum(p[0] for p in pts) // len(pts)
                        cy = sum(p[1] for p in pts) // len(pts)
                        cv2.putText(overlay, f"R{i}", (cx - 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Uși (bbox)
        if "doors" in data and data["doors"]:
            for door in data["doors"]:
                if "bbox" in door and len(door["bbox"]) == 4:
                    x1, y1, x2, y2 = map(int, door["bbox"])
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(overlay, "D", (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)

        out_path = raster_dir / "overlay_on_request.png"
        if not cv2.imwrite(str(out_path), overlay):
            return False
        return True
    except Exception as e:
        print(f"      ⚠️ overlay_on_request: {e}")
        return False


def generate_ref_walls_on_request_image(
    raster_dir: Path,
    orig_walls: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Generează masca de pereți „ref” (ca _ref_walls.png) pe imaginea comprimată trimisă la Raster.
    Redimensionează 02_ai_walls_closed (orig_walls) la dimensiunile request și binarizează.
    Salvează: brute_steps/_ref_walls_request.png

    Returns:
        Mască binară (request_h x request_w), 255=perete, sau None la eroare.
    """
    try:
        request_info_path = raster_dir / "raster_request_info.json"
        if not request_info_path.exists():
            return None
        with open(request_info_path, "r", encoding="utf-8") as f:
            ri = json.load(f)
        request_w = ri.get("request_w")
        request_h = ri.get("request_h")
        if not request_w or not request_h:
            return None
        ref_resized = cv2.resize(
            orig_walls, (request_w, request_h), interpolation=cv2.INTER_NEAREST
        )
        _, ref_binary = cv2.threshold(ref_resized, 127, 255, cv2.THRESH_BINARY)
        brute_steps_dir = raster_dir / "brute_steps"
        brute_steps_dir.mkdir(parents=True, exist_ok=True)
        out_path = brute_steps_dir / "_ref_walls_request.png"
        cv2.imwrite(str(out_path), ref_binary)
        return ref_binary
    except Exception:
        return None


def brute_force_translation_only(
    ref_binary: np.ndarray,
    api_binary: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Brute force doar cu translații (fără scale) între două măști de aceeași dimensiune.
    Faza 1: căutare cu pas 5 px pe intervalul complet. Faza 2: rafinare cu pas 1 px
    în jurul zonei bune (±5 px pe fiecare axă).

    Returns:
        Dict cu position: (tx, ty), score: float, direction: "translation_only", template_size: (w, h)
        sau None dacă măștile au dimensiuni diferite.
    """
    if ref_binary.shape != api_binary.shape:
        return None
    h, w = ref_binary.shape[:2]
    tx_min, tx_max = -w + 1, w
    ty_min, ty_max = -h + 1, h
    best_score = 0.0
    best_tx = best_ty = 0
    canvas = np.zeros_like(ref_binary)
    step_coarse = 5
    refine_radius = 5  # rafinare ±5 px (acoperă gap-ul dintre pașii de 5)

    def overlap_at(tx: int, ty: int) -> float:
        x_src_start = max(0, -tx)
        y_src_start = max(0, -ty)
        x_dst_start = max(0, tx)
        y_dst_start = max(0, ty)
        w_copy = min(w - x_src_start, w - x_dst_start)
        h_copy = min(h - y_src_start, h - y_dst_start)
        if w_copy <= 0 or h_copy <= 0:
            return 0.0
        canvas.fill(0)
        canvas[y_dst_start : y_dst_start + h_copy, x_dst_start : x_dst_start + w_copy] = (
            api_binary[y_src_start : y_src_start + h_copy, x_src_start : x_src_start + w_copy]
        )
        inter = int(((canvas == 255) & (ref_binary == 255)).sum())
        count_api = int((canvas == 255).sum())
        count_ref = int((ref_binary == 255).sum())
        if count_api == 0 or count_ref == 0:
            return 0.0
        pct_api_on_ref = inter / count_api
        pct_ref_covered = inter / count_ref
        return float(min(pct_api_on_ref, pct_ref_covered))

    # Faza 1: căutare grosieră, pas 5 px
    for tx in range(tx_min, tx_max, step_coarse):
        for ty in range(ty_min, ty_max, step_coarse):
            s = overlap_at(tx, ty)
            if s > best_score:
                best_score = s
                best_tx, best_ty = tx, ty

    # Faza 2: rafinare cu pas 1 în zona bună (±refine_radius)
    for tx in range(best_tx - refine_radius, best_tx + refine_radius + 1):
        for ty in range(best_ty - refine_radius, best_ty + refine_radius + 1):
            if tx < tx_min or tx >= tx_max or ty < ty_min or ty >= ty_max:
                continue
            s = overlap_at(tx, ty)
            if s > best_score:
                best_score = s
                best_tx, best_ty = tx, ty

    return {
        "position": (int(best_tx), int(best_ty)),
        "score": float(best_score),
        "direction": "translation_only",
        "template_size": (int(w), int(h)),
    }


def build_api_walls_mask_from_json(raster_dir: Path, request_w: int, request_h: int) -> Optional[np.ndarray]:
    """
    Construiește masca de pereți (binară) din response.json în spațiul request (request_w x request_h).
    Utilă când Raster face crop intern: masca din imaginea returnată nu coincide cu request;
    această mască este mereu în același spațiu ca raster_request.png, deci transformarea la original
    e doar scale_factor, fără offset de crop.

    Returns:
        Mască numpy (request_h x request_w), 255=perete, 0=rest, sau None la eroare.
    """
    try:
        response_path = raster_dir / "response.json"
        if not response_path.exists():
            return None
        with open(response_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        data = result.get("data", result)
        if "walls" not in data or not data["walls"]:
            return None
        mask = np.zeros((request_h, request_w), dtype=np.uint8)

        def pt(x: float, y: float):
            return (int(round(x)), int(round(y)))

        for wall in data["walls"]:
            pos = wall.get("position")
            if pos and len(pos) >= 2:
                p1, p2 = pos[0], pos[1]
                x1, y1 = (p1["x"], p1["y"]) if isinstance(p1, dict) else (p1[0], p1[1])
                x2, y2 = (p2["x"], p2["y"]) if isinstance(p2, dict) else (p2[0], p2[1])
                cv2.line(mask, pt(x1, y1), pt(x2, y2), 255, 2)
        out_path = raster_dir / "api_walls_from_json.png"
        cv2.imwrite(str(out_path), mask)
        return mask
    except Exception:
        return None


def build_api_walls_mask_from_json_1px(
    raster_dir: Path, request_w: int, request_h: int
) -> Optional[np.ndarray]:
    """
    Construiește masca de pereți din response.json cu grosime 1 pixel, fără găuri în colțuri/diagonale.
    Desenează linii cu thickness=1 și aplică un close 2x2 pentru a umple eventuale goluri la joncțiuni.

    Returns:
        Mască (request_h x request_w), 255=perete, 0=rest, sau None.
    """
    try:
        response_path = raster_dir / "response.json"
        if not response_path.exists():
            return None
        with open(response_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        data = result.get("data", result)
        if "walls" not in data or not data["walls"]:
            return None
        mask = np.zeros((request_h, request_w), dtype=np.uint8)

        def pt(x: float, y: float):
            return (int(round(x)), int(round(y)))

        for wall in data["walls"]:
            pos = wall.get("position")
            if pos and len(pos) >= 2:
                p1, p2 = pos[0], pos[1]
                x1, y1 = (p1["x"], p1["y"]) if isinstance(p1, dict) else (p1[0], p1[1])
                x2, y2 = (p2["x"], p2["y"]) if isinstance(p2, dict) else (p2[0], p2[1])
                cv2.line(mask, pt(x1, y1), pt(x2, y2), 255, 1)
        # Închidem găuri de 1 pixel la colțuri/diagonale
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        out_path = raster_dir / "api_walls_from_json_1px.png"
        cv2.imwrite(str(out_path), mask)
        return mask
    except Exception:
        return None


def build_aligned_api_walls_1px_original(
    raster_dir: Path,
    tx: int,
    ty: int,
    orig_w: int,
    orig_h: int,
) -> Optional[np.ndarray]:
    """
    Construiește masca API pereți 1px, aplică translația (tx, ty) în spațiul request,
    apoi scalează la dimensiunile originalului. Util după brute_force_translation_only.

    Returns:
        Mască (orig_h x orig_w), 255=perete, 0=rest, sau None.
    """
    try:
        request_info_path = raster_dir / "raster_request_info.json"
        if not request_info_path.exists():
            req_w, req_h = orig_w, orig_h
        else:
            with open(request_info_path, "r", encoding="utf-8") as f:
                ri = json.load(f)
            req_w = int(ri.get("request_w", orig_w))
            req_h = int(ri.get("request_h", orig_h))
        api_1px = build_api_walls_mask_from_json_1px(raster_dir, req_w, req_h)
        if api_1px is None:
            return None
        h, w = api_1px.shape[:2]
        canvas = np.zeros((h, w), dtype=np.uint8)
        x_src = max(0, -tx)
        y_src = max(0, -ty)
        x_dst = max(0, tx)
        y_dst = max(0, ty)
        w_c = min(w - x_src, w - x_dst)
        h_c = min(h - y_src, h - y_dst)
        if w_c <= 0 or h_c <= 0:
            return None
        canvas[y_dst : y_dst + h_c, x_dst : x_dst + w_c] = api_1px[
            y_src : y_src + h_c, x_src : x_src + w_c
        ]
        aligned_original = cv2.resize(
            canvas, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        _, aligned_original = cv2.threshold(aligned_original, 127, 255, cv2.THRESH_BINARY)
        return aligned_original
    except Exception:
        return None


def save_overlay_on_original_from_response(
    raster_dir: Path,
    original_img: np.ndarray,
) -> bool:
    """
    Desenează camere și uși din response.json pe imaginea originală (full size) și salvează overlay_on_original.png.
    Coordonatele din JSON sunt în spațiul imaginii trimise la API; le scalăm la dimensiunea originală
    folosind raster_request_info.json (fără brute force / aliniere).

    Returns:
        True dacă overlay-ul a fost salvat, False altfel.
    """
    try:
        response_path = raster_dir / "response.json"
        request_info_path = raster_dir / "raster_request_info.json"
        if not response_path.exists() or not request_info_path.exists():
            return False
        with open(response_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        with open(request_info_path, "r", encoding="utf-8") as f:
            ri = json.load(f)
        data = result.get("data", result)
        req_w = ri.get("request_w") or 1
        req_h = ri.get("request_h") or 1
        orig_w = ri.get("original_w") or original_img.shape[1]
        orig_h = ri.get("original_h") or original_img.shape[0]
        scale_x = orig_w / max(1, req_w)
        scale_y = orig_h / max(1, req_h)

        def to_orig(x: float, y: float):
            return (
                int(round(x * scale_x)),
                int(round(y * scale_y)),
            )

        overlay = original_img.copy()
        h_orig, w_orig = overlay.shape[:2]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

        if "rooms" in data and data["rooms"]:
            for i, room in enumerate(data["rooms"]):
                pts = []
                for point in room:
                    if "x" in point and "y" in point:
                        ox, oy = to_orig(point["x"], point["y"])
                        ox = max(0, min(ox, w_orig - 1))
                        oy = max(0, min(oy, h_orig - 1))
                        pts.append((ox, oy))
                if len(pts) >= 3:
                    pts_np = np.array(pts, dtype=np.int32)
                    color = colors[i % len(colors)]
                    cv2.polylines(overlay, [pts_np], True, color, 4)
                    if pts:
                        cx = sum(p[0] for p in pts) // len(pts)
                        cy = sum(p[1] for p in pts) // len(pts)
                        cv2.putText(
                            overlay, f"Room {i}", (cx - 50, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3,
                        )

        if "doors" in data and data["doors"]:
            for door in data["doors"]:
                if "bbox" in door and len(door["bbox"]) == 4:
                    x1, y1, x2, y2 = door["bbox"]
                    ox1, oy1 = to_orig(x1, y1)
                    ox2, oy2 = to_orig(x2, y2)
                    ox1, ox2 = max(0, min(ox1, w_orig)), max(0, min(ox2, w_orig))
                    oy1, oy2 = max(0, min(oy1, h_orig)), max(0, min(oy2, h_orig))
                    cv2.rectangle(overlay, (ox1, oy1), (ox2, oy2), (0, 165, 255), 3)
                    cv2.putText(
                        overlay, "Door", (ox1, oy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2,
                    )

        out_path = raster_dir / "overlay_on_original.png"
        cv2.imwrite(str(out_path), overlay)
        return True
    except Exception:
        return False


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


def get_api_walls_mask_for_alignment(raster_dir: Path, orig_h: int, orig_w: int) -> Optional[np.ndarray]:
    """
    Returnează masca de pereți API la dimensiunea originală, potrivită pentru brute-force alignment.
    Când Raster face crop (raster_may_crop), folosește api_walls_from_json.png (în spațiul request),
    scalat la original, astfel încât alinierea să nu depindă de offset-ul de crop necunoscut.
    """
    request_info_path = raster_dir / "raster_request_info.json"
    if not request_info_path.exists():
        return _load_and_resize_api_walls_mask(raster_dir, orig_h, orig_w)
    try:
        with open(request_info_path, "r", encoding="utf-8") as f:
            ri = json.load(f)
    except Exception:
        return _load_and_resize_api_walls_mask(raster_dir, orig_h, orig_w)
    req_w = ri.get("request_w")
    req_h = ri.get("request_h")
    json_mask_path = raster_dir / "api_walls_from_json.png"
    if ri.get("raster_may_crop") and json_mask_path.exists() and req_w and req_h:
        mask = cv2.imread(str(json_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape[1] == req_w and mask.shape[0] == req_h:
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            print(f"      📐 Folosesc api_walls_from_json.png (spațiu request) scalat la original – Raster a făcut crop")
            return mask
    return _load_and_resize_api_walls_mask(raster_dir, orig_h, orig_w)


def _load_and_resize_api_walls_mask(raster_dir: Path, orig_h: int, orig_w: int) -> Optional[np.ndarray]:
    api_walls_path = raster_dir / "api_walls_mask.png"
    if not api_walls_path.exists():
        return None
    mask = cv2.imread(str(api_walls_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape[0] != orig_h or mask.shape[1] != orig_w:
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask


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
        
        # Size-up imaginea de la API la original cu același factor pe ambele axe (păstrăm aspect ratio)
        scale_factor = api_result.get('scale_factor', 1.0)
        if scale_factor < 1.0:
            scale_up = 1.0 / scale_factor
            api_w, api_h = api_processed_img.shape[1], api_processed_img.shape[0]
            target_w = max(1, int(round(api_w * scale_up)))
            target_h = max(1, int(round(api_h * scale_up)))
            api_processed_img = cv2.resize(api_processed_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            print(f"      📐 Size-up imagine API la original (aspect ratio păstrat): {target_w}x{target_h}")
        
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
    steps_dir: str,
    gemini_api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Algoritm brute-force pentru alinierea măștilor de pereți API și original.
    """
    try:
        print(f"\n      🔥 BRUTE FORCE: Căutare transformare între API walls și original walls...")
        
        # Masca de la Raster trebuie readusă la marimea originală (cea înainte de trimitere la API) pentru brute force
        h_orig, w_orig = orig_walls.shape[:2]
        if api_walls_mask.shape[0] != h_orig or api_walls_mask.shape[1] != w_orig:
            api_walls_mask = cv2.resize(
                api_walls_mask, (w_orig, h_orig),
                interpolation=cv2.INTER_NEAREST
            )
            print(f"      📐 Mască API readusă la marimea originală: {w_orig}x{h_orig}")
        
        print(f"      📊 API walls: {api_walls_mask.shape[1]} x {api_walls_mask.shape[0]}")
        print(f"      📊 Original walls: {orig_walls.shape[1]} x {orig_walls.shape[0]}")
        
        # Binarizare: 255 = perete, 0 = fundal (convenția pipeline: pereții sunt albi/deschiși)
        _, binary_api = cv2.threshold(api_walls_mask, 127, 255, cv2.THRESH_BINARY)
        _, binary_orig = cv2.threshold(orig_walls, 127, 255, cv2.THRESH_BINARY)
        
        def _overlap_scores(
            base_binary: np.ndarray,
            template_binary: np.ndarray,
            position: Tuple[int, int],
            template_size: Tuple[int, int],
        ) -> Tuple[float, float, float]:
            """
            Plasează template pe base la position (aceeași dimensiune canvas ca base).
            Pereți = 255, fundal = 0.
            Returnează:
              - pct_template_on_base: % din pereții TEMPLATE care cad peste pereți BASE (cât din albastru e peste roșu)
              - pct_base_covered: % din pereții BASE acoperiți de pereți TEMPLATE (cât din roșu e acoperit de albastru)
              - combined: min(pct_template_on_base, pct_base_covered) – ambele măști trebuie să se suprapună bine
            """
            h_base, w_base = base_binary.shape[:2]
            x, y = position
            tw, th = template_size
            tw_crop = min(tw, max(0, w_base - x))
            th_crop = min(th, max(0, h_base - y))
            if tw_crop <= 0 or th_crop <= 0 or x < 0 or y < 0:
                return 0.0, 0.0, 0.0
            template_scaled = cv2.resize(template_binary, (tw, th), interpolation=cv2.INTER_NEAREST)
            placed = np.zeros((h_base, w_base), dtype=np.uint8)
            placed[y : y + th_crop, x : x + tw_crop] = template_scaled[:th_crop, :tw_crop]
            overlap_mask = (placed == 255) & (base_binary == 255)
            inter = int(overlap_mask.sum())
            count_placed = (placed == 255).sum()
            count_base = (base_binary == 255).sum()
            pct_template_on_base = (inter / count_placed) if count_placed else 0.0
            pct_base_covered = (inter / count_base) if count_base else 0.0
            # Cea mai bună suprapunere = cat mai mulți pereți ai fiecărei măști suprapuși, cat mai puțini nesuprapuși
            combined = min(pct_template_on_base, pct_base_covered)
            return float(pct_template_on_base), float(pct_base_covered), float(combined)
        
        # Dimensiune maximă pentru toate overlay-urile salvate (același sizing între planuri)
        MAX_OVERLAY_OUTPUT_SIDE = 1200
        
        def _maybe_resize_to_standard(img: np.ndarray) -> np.ndarray:
            """Redimensionează imaginea dacă depășește MAX_OVERLAY_OUTPUT_SIDE, păstrând aspect ratio."""
            h, w = img.shape[:2]
            if max(h, w) <= MAX_OVERLAY_OUTPUT_SIDE:
                return img
            scale = MAX_OVERLAY_OUTPUT_SIDE / max(h, w)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Folder pentru pașii brute force (vizualizare) – golit înainte de repopulare
        brute_steps_dir = raster_dir / "brute_steps"
        brute_steps_dir.mkdir(parents=True, exist_ok=True)
        for f in brute_steps_dir.iterdir():
            if f.is_file():
                try:
                    f.unlink()
                except OSError:
                    pass

        # Brute force DOAR translații (fără scale): ref pe imaginea comprimată vs api_walls din JSON
        trans_only_cfg = None
        ref_request = generate_ref_walls_on_request_image(raster_dir, orig_walls)
        api_request_path = raster_dir / "api_walls_from_json.png"
        if ref_request is not None and api_request_path.exists():
            api_request = cv2.imread(str(api_request_path), cv2.IMREAD_GRAYSCALE)
            if api_request is not None and api_request.shape == ref_request.shape:
                _, api_req_bin = cv2.threshold(api_request, 127, 255, cv2.THRESH_BINARY)
                trans_only_cfg = brute_force_translation_only(ref_request, api_req_bin)
                if trans_only_cfg is not None:
                    trans_path = brute_steps_dir / "translation_only_config.json"
                    try:
                        with open(trans_path, "w") as f:
                            json.dump(
                                {
                                    "position": list(trans_only_cfg["position"]),
                                    "score": trans_only_cfg["score"],
                                    "direction": "translation_only",
                                    "template_size": list(trans_only_cfg["template_size"]),
                                },
                                f,
                                indent=2,
                            )
                    except OSError:
                        pass
                    print(f"      📐 Brute force doar translații (request space): offset {trans_only_cfg['position']}, score {trans_only_cfg['score']:.2%}")
                    # Overlay vizual: ref = roșu, api deplasat = verde
                    tx, ty = trans_only_cfg["position"]
                    h_r, w_r = ref_request.shape[:2]
                    overlay_req = np.zeros((h_r, w_r, 3), dtype=np.uint8)
                    overlay_req[:, :, 2] = ref_request
                    x_dst = max(0, tx)
                    y_dst = max(0, ty)
                    x_src = max(0, -tx)
                    y_src = max(0, -ty)
                    w_c = min(w_r - x_dst, api_req_bin.shape[1] - x_src)
                    h_c = min(h_r - y_dst, api_req_bin.shape[0] - y_src)
                    if w_c > 0 and h_c > 0:
                        overlay_req[y_dst : y_dst + h_c, x_dst : x_dst + w_c, 0] = api_req_bin[y_src : y_src + h_c, x_src : x_src + w_c]
                        overlay_req[y_dst : y_dst + h_c, x_dst : x_dst + w_c, 1] = api_req_bin[y_src : y_src + h_c, x_src : x_src + w_c]
                    try:
                        cv2.imwrite(str(brute_steps_dir / "translation_only_overlay.png"), overlay_req)
                    except OSError:
                        pass

        # Dacă avem rezultat doar translații, îl folosim ca rezultat final (fără scale, fără Gemini)
        use_translation_only = trans_only_cfg is not None
        if use_translation_only:
            tx, ty = trans_only_cfg["position"]
            req_h, req_w = ref_request.shape[0], ref_request.shape[1]
            scale_factor = 1.0
            request_info_path = raster_dir / "raster_request_info.json"
            if request_info_path.exists():
                try:
                    with open(request_info_path, "r") as f:
                        ri = json.load(f)
                    scale_factor = float(ri.get("scale_factor", 1.0))
                except Exception:
                    pass
            if scale_factor <= 0 or scale_factor > 1:
                scale_factor = req_w / binary_orig.shape[1] if binary_orig.shape[1] else 1.0
            scale_up = 1.0 / scale_factor if 0 < scale_factor < 1 else 1.0
            tx_orig = int(round(tx * scale_up))
            ty_orig = int(round(ty * scale_up))
            tw_orig = int(binary_api.shape[1])
            th_orig = int(binary_api.shape[0])
            best_trans = {
                "direction": "api_to_orig",
                "scale": 1.0,
                "rotation": 0,
                "position": (tx_orig, ty_orig),
                "template_size": (tw_orig, th_orig),
                "score": float(trans_only_cfg["score"]),
                "score_api2orig": float(trans_only_cfg["score"]),
                "score_orig2api": float(trans_only_cfg["score"]),
            }
            top_results = [best_trans]
            print(f"      ✅ Folosim doar translații (fără scale): offset original ({tx_orig}, {ty_orig}), score {trans_only_cfg['score']:.2%}")

        if not use_translation_only:
            # Gemini în paralel (ca în segmenter: client + procente 0–1000) – spune concret cum să suprapui cele două imagini
            gemini_hint = [None]  # [0] setat de thread
            _ref_path = brute_steps_dir / "_ref_walls.png"
            _api_path = brute_steps_dir / "_api_walls.png"
            cv2.imwrite(str(_ref_path), binary_orig)
            cv2.imwrite(str(_api_path), binary_api)

            # _plan_walls.png: masca de pereți API desenată pe planul trimis la Raster (raster_request.png)
            _plan_walls_path = brute_steps_dir / "_plan_walls.png"
            plan_request = cv2.imread(str(raster_dir / "raster_request.png"))
            if plan_request is None:
                plan_request = cv2.imread(str(raster_dir / "input_resized.jpg"))
            if plan_request is not None and binary_api is not None:
                h_req, w_req = plan_request.shape[:2]
                mask_at_request_size = cv2.resize(
                    binary_api, (w_req, h_req), interpolation=cv2.INTER_NEAREST
                )
                plan_walls_img = plan_request.copy()
                plan_walls_img[mask_at_request_size > 0] = [0, 255, 0]  # verde = pereți
                try:
                    cv2.imwrite(str(_plan_walls_path), plan_walls_img)
                except OSError:
                    pass

            def _run_gemini_hint():
                try:
                    from segmenter.gemini_crop import get_gemini_wall_alignment
                    out = get_gemini_wall_alignment(_ref_path, _api_path)
                    if out is not None:
                        gemini_hint[0] = out  # orice răspuns (salvare/overlay); candidat doar dacă confidence >= 0.3
                except Exception as e:
                    gemini_hint[0] = {"_error": str(e)}  # ca să știm că a fost eroare

            gemini_thread = None
            if os.environ.get("GEMINI_API_KEY"):
                gemini_thread = threading.Thread(target=_run_gemini_hint, daemon=True)
                gemini_thread.start()
                print(f"      🤖 Gemini: aliniere în paralel (procente suprapunere)...")
            
            def _save_step_overlay(base_binary, template_binary, config, path: Path):
                """Salvează overlay: roșu = base, verde/albastru = template scalat la position. Dimensiune normalizată."""
                try:
                    tw, th = config['template_size']
                    template_scaled = cv2.resize(template_binary, (tw, th))
                    x_pos, y_pos = config['position']
                    h, w = base_binary.shape[:2]
                    overlay = np.zeros((h, w, 3), dtype=np.uint8)
                    overlay[:, :, 2] = base_binary
                    y_end = min(y_pos + th, h)
                    x_end = min(x_pos + tw, w)
                    dy, dx = y_end - y_pos, x_end - x_pos
                    overlay[y_pos:y_end, x_pos:x_end, 0] = template_scaled[:dy, :dx]
                    overlay[y_pos:y_end, x_pos:x_end, 1] = template_scaled[:dy, :dx]
                    overlay = _maybe_resize_to_standard(overlay)
                    if not cv2.imwrite(str(path), overlay):
                        print(f"      ⚠️ Nu s-a putut salva {path.name} (posibil disc plin)")
                except OSError as e:
                    if e.errno == 28:  # No space left on device
                        print(f"      ⚠️ Disc plin: nu s-a salvat {path.name}")
                    else:
                        raise
            
            # Brute force inteligent: mai întâi interval 0.8–1.2 (scale aproape 1:1); doar dacă nu găsim acuratețe mare, trecem la 0.1–10.0
            FOCUS_SCALE_MIN, FOCUS_SCALE_MAX, FOCUS_STEP = 0.8, 1.2, 0.005
            FULL_SCALE_MIN, FULL_SCALE_MAX, FULL_STEP = 0.1, 10.0, 0.01
            ACCURACY_THRESHOLD = 0.48  # peste acest score în [0.8, 1.2] rămânem pe interval focus
            
            top_results = []
            all_scale_candidates = []  # (scale, config) pentru fiecare scale testat – folosit pentru best per interval
    
            def add_to_top_results(config, max_results=10):
                top_results.append(config)
                top_results.sort(key=lambda x: x['score'], reverse=True)
                if len(top_results) > max_results:
                    top_results.pop()
    
            def _append_scale_candidate(scale_val, cfg):
                c = {k: v for k, v in cfg.items()}
                all_scale_candidates.append((float(scale_val), c))
    
            # Refinare poziție: rază și pas ca fracțiuni din dimensiunea imaginii (independent de rezoluție)
            def _search_position_grid(
                base_binary: np.ndarray,
                template_scaled: np.ndarray,
                initial_pos: Tuple[int, int],
                size: Tuple[int, int],
                radius_frac: float = 0.025,
                step_frac: float = 0.004,
            ) -> Tuple[Tuple[int, int], float, float, float]:
                """Caută în grid în jurul initial_pos. radius_frac/step_frac = fracțiuni din min(lățime, înălțime)."""
                h, w = base_binary.shape[:2]
                tw, th = size
                small = min(w, h)
                radius = max(5, int(radius_frac * small))
                step = max(2, int(step_frac * small))
                pct1, pct2, combined = _overlap_scores(base_binary, template_scaled, initial_pos, size)
                best_pos = initial_pos
                best_combined = combined
                best_pct1, best_pct2 = pct1, pct2
                for dy in range(-radius, radius + 1, step):
                    for dx in range(-radius, radius + 1, step):
                        if dx == 0 and dy == 0:
                            continue
                        x = initial_pos[0] + dx
                        y = initial_pos[1] + dy
                        if x + tw <= 0 or y + th <= 0 or x >= w or y >= h:
                            continue
                        pct1, pct2, combined = _overlap_scores(base_binary, template_scaled, (x, y), size)
                        if combined > best_combined:
                            best_combined = combined
                            best_pos = (x, y)
                            best_pct1, best_pct2 = pct1, pct2
                return best_pos, best_pct1, best_pct2, best_combined
    
            def _best_overlap_near(
                base_binary: np.ndarray,
                template_scaled: np.ndarray,
                initial_pos: Tuple[int, int],
                size: Tuple[int, int],
                grid_step_frac: float = 0.015,
            ) -> Tuple[Tuple[int, int], float, float, float]:
                """Scor maxim într-un mic grid 3x3 în jurul poziției inițiale – evită să ratăm vârful."""
                h, w = base_binary.shape[:2]
                tw, th = size
                small = min(w, h)
                step = max(2, int(grid_step_frac * small))
                best_pos = initial_pos
                pct1, pct2, combined = _overlap_scores(base_binary, template_scaled, initial_pos, size)
                best_pct1, best_pct2, best_combined = pct1, pct2, combined
                for dy in (-step, 0, step):
                    for dx in (-step, 0, step):
                        if dx == 0 and dy == 0:
                            continue
                        x = initial_pos[0] + dx
                        y = initial_pos[1] + dy
                        if x < 0 or y < 0 or x + tw > w or y + th > h:
                            continue
                        pct1, pct2, combined = _overlap_scores(base_binary, template_scaled, (x, y), size)
                        if combined > best_combined:
                            best_combined = combined
                            best_pos = (x, y)
                            best_pct1, best_pct2 = pct1, pct2
                return best_pos, best_pct1, best_pct2, best_combined
    
            def _best_position_and_score(
                base_binary: np.ndarray,
                template_scaled: np.ndarray,
                initial_pos: Tuple[int, int],
                size: Tuple[int, int],
            ) -> Tuple[Tuple[int, int], float, float, float]:
                """Refinare în două etape: grosier apoi fin; toate razele/pasii sunt fracțiuni din imagine."""
                # Etapa 1: ~3.2% din imagine, pas ~0.5% – găsește zona bună (rază mărită pentru planuri cu offset mare)
                pos_coarse, _, _, _ = _search_position_grid(
                    base_binary, template_scaled, initial_pos, size,
                    radius_frac=0.032, step_frac=0.005
                )
                # Etapa 2: ~0.8% rază, pas ~0.15% – poziționare precisă
                return _search_position_grid(
                    base_binary, template_scaled, pos_coarse, size,
                    radius_frac=0.008, step_frac=0.0015
                )
    
            def _global_coarse_position_search(
                base_binary: np.ndarray,
                template_scaled: np.ndarray,
                size: Tuple[int, int],
                step_frac: float = 0.028,
                max_evals: int = 1800,
                return_top_k: int = 0,
            ):
                """
                Caută poziția optimă pe întreaga regiune validă.
                Dacă return_top_k>0, returnează și lista top return_top_k (pos, pct1, pct2, combined) pentru multi-start refinare.
                """
                h, w = base_binary.shape[:2]
                tw, th = size
                valid_w = max(0, w - tw)
                valid_h = max(0, h - th)
                if valid_w <= 0 or valid_h <= 0:
                    pct1, pct2, combined = _overlap_scores(base_binary, template_scaled, (0, 0), size)
                    if return_top_k > 0:
                        return (0, 0), pct1, pct2, combined, [((0, 0), pct1, pct2, combined)]
                    return (0, 0), pct1, pct2, combined, None
                small = min(w, h)
                step = max(8, int(step_frac * small))
                nx = max(1, valid_w // step + 1)
                ny = max(1, valid_h // step + 1)
                while nx * ny > max_evals and step < min(valid_w, valid_h):
                    step = step + max(4, int(0.01 * small))
                    nx = max(1, valid_w // step + 1)
                    ny = max(1, valid_h // step + 1)
                best_pos = (0, 0)
                best_combined = -1.0
                best_pct1, best_pct2 = 0.0, 0.0
                candidates = [] if return_top_k > 0 else None
                for iy in range(ny):
                    y = min(iy * step, valid_h)
                    for ix in range(nx):
                        x = min(ix * step, valid_w)
                        pct1, pct2, combined = _overlap_scores(base_binary, template_scaled, (x, y), size)
                        if return_top_k > 0:
                            candidates.append(((x, y), pct1, pct2, combined))
                        if combined > best_combined:
                            best_combined = combined
                            best_pos = (x, y)
                            best_pct1, best_pct2 = pct1, pct2
                if return_top_k > 0 and candidates:
                    candidates.sort(key=lambda t: t[3], reverse=True)
                    top_list = candidates[:return_top_k]
                    return best_pos, best_pct1, best_pct2, best_combined, top_list
                return best_pos, best_pct1, best_pct2, best_combined, None
    
            def run_scale_search(scales_arr, save_step_indices=None, refine_position=False):
                """refine_position=False: un singur scor la poziția matchTemplate (rapid). True: grid poziții (precis)."""
                total = len(scales_arr)
                log_every = max(1, total // 10)
                for idx, scale in enumerate(scales_arr):
                    cfg_a2o, cfg_o2a = None, None
                    new_w = int(binary_api.shape[1] * scale)
                    new_h = int(binary_api.shape[0] * scale)
                    if new_w <= binary_orig.shape[1] and new_h <= binary_orig.shape[0] and new_w >= 30 and new_h >= 30:
                        api_scaled = cv2.resize(binary_api, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                        result_match = cv2.matchTemplate(binary_orig, api_scaled, cv2.TM_CCOEFF_NORMED)
                        _, _, _, max_loc = cv2.minMaxLoc(result_match)
                        pos_init = (int(max_loc[0]), int(max_loc[1]))
                        if refine_position:
                            pos_a2o, score_a2o_pct, score_o2a_pct, combined = _best_position_and_score(
                                binary_orig, api_scaled, pos_init, (new_w, new_h)
                            )
                        else:
                            pos_a2o, score_a2o_pct, score_o2a_pct, combined = _best_overlap_near(
                                binary_orig, api_scaled, pos_init, (new_w, new_h)
                            )
                        cfg_a2o = {
                            'direction': 'api_to_orig', 'scale': float(scale), 'rotation': 0,
                            'position': pos_a2o, 'template_size': (new_w, new_h),
                            'score': combined, 'score_api2orig': score_a2o_pct, 'score_orig2api': score_o2a_pct,
                        }
                    new_w_o = int(binary_orig.shape[1] * scale)
                    new_h_o = int(binary_orig.shape[0] * scale)
                    if new_w_o <= binary_api.shape[1] and new_h_o <= binary_api.shape[0] and new_w_o >= 30 and new_h_o >= 30:
                        orig_scaled = cv2.resize(binary_orig, (new_w_o, new_h_o), interpolation=cv2.INTER_NEAREST)
                        result_match = cv2.matchTemplate(binary_api, orig_scaled, cv2.TM_CCOEFF_NORMED)
                        _, _, _, max_loc = cv2.minMaxLoc(result_match)
                        pos_init = (int(max_loc[0]), int(max_loc[1]))
                        if refine_position:
                            pos_o2a, score_o2a_pct, score_a2o_pct, combined = _best_position_and_score(
                                binary_api, orig_scaled, pos_init, (new_w_o, new_h_o)
                            )
                        else:
                            pos_o2a, score_o2a_pct, score_a2o_pct, combined = _best_overlap_near(
                                binary_api, orig_scaled, pos_init, (new_w_o, new_h_o)
                            )
                        cfg_o2a = {
                            'direction': 'orig_to_api', 'scale': float(scale), 'rotation': 0,
                            'position': pos_o2a, 'template_size': (new_w_o, new_h_o),
                            'score': combined, 'score_api2orig': score_a2o_pct, 'score_orig2api': score_o2a_pct,
                        }
                    if cfg_a2o and cfg_o2a:
                        best_cfg = cfg_a2o if cfg_a2o['score'] >= cfg_o2a['score'] else cfg_o2a
                    else:
                        best_cfg = cfg_a2o or cfg_o2a
                    if best_cfg:
                        add_to_top_results(best_cfg)
                        _append_scale_candidate(scale, best_cfg)
                    if save_step_indices is not None and idx in save_step_indices:
                        if cfg_a2o:
                            _save_step_overlay(binary_orig, binary_api, {**cfg_a2o, 'template_size': cfg_a2o['template_size']}, brute_steps_dir / f"step_api2orig_scale_{scale:.2f}_score_{cfg_a2o['score']:.3f}.png")
                        if cfg_o2a:
                            _save_step_overlay(binary_api, binary_orig, {**cfg_o2a, 'template_size': cfg_o2a['template_size']}, brute_steps_dir / f"step_orig2api_scale_{scale:.2f}_score_{cfg_o2a['score']:.3f}.png")
                    if idx % log_every == 0 and top_results:
                        c = top_results[0]
                        s2 = f" (api→orig: {c.get('score_api2orig', 0):.2%}, orig→api: {c.get('score_orig2api', 0):.2%})" if c.get('score_orig2api') is not None else ""
                        print(f"         ⏳ Test {idx+1}/{total}: scale={scale:.2f}x... Best: {top_results[0]['score']:.2%}{s2}")
            
            # Coarse-to-fine: pași suficienți ca să nu sărim peste scale-ul optim
            FOCUS_COARSE_STEP = 0.02   # ~21 scale-uri [0.8, 1.2]
            FOCUS_REFINE_WINDOW = 0.04
            FOCUS_REFINE_STEP = 0.005
            FULL_COARSE_STEP = 0.05    # ~199 scale-uri [0.1, 10]
            FULL_REFINE_WINDOW = 0.06
            FULL_REFINE_STEP = 0.01
    
            # Faza 1: focus 0.8–1.2, coarse apoi fine (fără grid poziție în căutare)
            scales_focus_coarse = np.arange(FOCUS_SCALE_MIN, FOCUS_SCALE_MAX + FOCUS_COARSE_STEP / 2, FOCUS_COARSE_STEP)
            print(f"      📊 Faza 1 – focus 80%–120%: coarse {len(scales_focus_coarse)} scale-uri (pas {FOCUS_COARSE_STEP})")
            step_indices = list(np.linspace(0, max(0, len(scales_focus_coarse) - 1), min(10, len(scales_focus_coarse)), dtype=int))
            run_scale_search(scales_focus_coarse, save_step_indices=step_indices, refine_position=False)
            best_scale_focus = top_results[0]['scale'] if top_results else 1.0
            scales_focus_fine = np.unique(np.clip(
                np.arange(best_scale_focus - FOCUS_REFINE_WINDOW, best_scale_focus + FOCUS_REFINE_WINDOW + FOCUS_REFINE_STEP / 2, FOCUS_REFINE_STEP),
                FOCUS_SCALE_MIN, FOCUS_SCALE_MAX
            ))
            if len(scales_focus_fine) > 0:
                print(f"      📊 Faza 1 – refinare: {len(scales_focus_fine)} scale-uri în [{scales_focus_fine[0]:.2f}, {scales_focus_fine[-1]:.2f}]")
                run_scale_search(scales_focus_fine, refine_position=False)
            
            use_full_range = True
            if top_results and top_results[0]['score'] >= ACCURACY_THRESHOLD:
                print(f"      ✅ Score bun în focus: {top_results[0]['score']:.2%} (>= {ACCURACY_THRESHOLD})")
                use_full_range = False
            
            if use_full_range:
                top_results.clear()
                scales_full_coarse = np.arange(FULL_SCALE_MIN, FULL_SCALE_MAX + FULL_COARSE_STEP / 2, FULL_COARSE_STEP)
                print(f"      📊 Faza 2 – interval 10%–1000%: coarse {len(scales_full_coarse)} scale-uri (pas {FULL_COARSE_STEP})")
                step_indices_full = list(np.linspace(0, max(0, len(scales_full_coarse) - 1), min(10, len(scales_full_coarse)), dtype=int))
                run_scale_search(scales_full_coarse, save_step_indices=step_indices_full, refine_position=False)
                best_scale_full = top_results[0]['scale'] if top_results else 0.5
                scales_full_fine = np.unique(np.clip(
                    np.arange(best_scale_full - FULL_REFINE_WINDOW, best_scale_full + FULL_REFINE_WINDOW + FULL_REFINE_STEP / 2, FULL_REFINE_STEP),
                    FULL_SCALE_MIN, FULL_SCALE_MAX
                ))
                if len(scales_full_fine) > 0:
                    print(f"      📊 Faza 2 – refinare: {len(scales_full_fine)} scale-uri în [{scales_full_fine[0]:.2f}, {scales_full_fine[-1]:.2f}]")
                    run_scale_search(scales_full_fine, refine_position=False)
            
            if not top_results:
                print(f"      ⚠️ Nu s-au găsit rezultate valide pentru brute force")
                return None
    
            if gemini_thread:
                gemini_thread.join(timeout=3)
                if gemini_hint[0] is None:
                    print(f"      🤖 Gemini: nu a returnat răspuns (timeout sau fără răspuns).")
    
            # Refinare poziție: grid global + o refinare locală (fără multi-start)
            def refine_position_for_best(cfg, step_frac: float = 0.028, max_evals: int = 1000, top_k: int = 0):
                scale = cfg['scale']
                direction = cfg['direction']
                def do_search(base_binary, template_binary, new_w, new_h):
                    size = (new_w, new_h)
                    best_pos, best_pct1, best_pct2, best_combined, _ = _global_coarse_position_search(
                        base_binary, template_binary, size,
                        step_frac=step_frac, max_evals=max_evals, return_top_k=top_k
                    )
                    # O singură refinare locală în jurul best_pos
                    pos, best_pct1, best_pct2, best_combined = _search_position_grid(
                        base_binary, template_binary, best_pos, size,
                        radius_frac=0.018, step_frac=0.0012
                    )
                    return pos, best_pct1, best_pct2, best_combined
                if direction == 'api_to_orig':
                    new_w = int(binary_api.shape[1] * scale)
                    new_h = int(binary_api.shape[0] * scale)
                    if new_w > binary_orig.shape[1] or new_h > binary_orig.shape[0] or new_w < 30 or new_h < 30:
                        return cfg
                    api_scaled = cv2.resize(binary_api, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    pos, score_a2o, score_o2a, combined = do_search(binary_orig, api_scaled, new_w, new_h)
                    return {
                        'direction': 'api_to_orig', 'scale': float(scale), 'rotation': 0,
                        'position': pos, 'template_size': (new_w, new_h),
                        'score': float(combined), 'score_api2orig': score_a2o, 'score_orig2api': score_o2a,
                    }
                else:
                    new_w = int(binary_orig.shape[1] * scale)
                    new_h = int(binary_orig.shape[0] * scale)
                    if new_w > binary_api.shape[1] or new_h > binary_api.shape[0] or new_w < 30 or new_h < 30:
                        return cfg
                    orig_scaled = cv2.resize(binary_orig, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    pos, score_o2a, score_a2o, combined = do_search(binary_api, orig_scaled, new_w, new_h)
                    return {
                        'direction': 'orig_to_api', 'scale': float(scale), 'rotation': 0,
                        'position': pos, 'template_size': (new_w, new_h),
                        'score': float(combined), 'score_api2orig': score_a2o, 'score_orig2api': score_o2a,
                    }
            # Variantă Gemini: aliniere directă din procente (fără refinare) – o a doua variantă de calcul
            gemini_candidate = None
            if gemini_hint[0]:
                hint = gemini_hint[0]
                err = hint.get("_error")
                if err:
                    print(f"      🤖 Gemini: eroare – {err}")
                    try:
                        with open(brute_steps_dir / "gemini_response.json", "w", encoding="utf-8") as f:
                            json.dump({"error": err}, f, indent=2)
                    except Exception:
                        pass
                else:
                    # Salvare răspuns raw Gemini în brute_steps
                    try:
                        with open(brute_steps_dir / "gemini_response.json", "w", encoding="utf-8") as f:
                            json.dump({k: hint.get(k) for k in ("scale", "offset_x_pct", "offset_y_pct", "direction", "confidence")}, f, indent=2)
                    except Exception:
                        pass
                if not err:
                    scale = hint.get("scale")
                    direction = hint.get("direction", "api_to_orig")
                    ox = hint.get("offset_x_pct", 0)
                    oy = hint.get("offset_y_pct", 0)
                    conf = hint.get("confidence", 0)
                    scale_clamp = max(0.2, min(2.0, float(scale))) if scale is not None else 1.0
                    if direction == "api_to_orig":
                        W, H = binary_orig.shape[1], binary_orig.shape[0]
                        tw = int(binary_api.shape[1] * scale_clamp)
                        th = int(binary_api.shape[0] * scale_clamp)
                        base_bin, tpl_bin = binary_orig, cv2.resize(binary_api, (tw, th), interpolation=cv2.INTER_NEAREST)
                    else:
                        W, H = binary_api.shape[1], binary_api.shape[0]
                        tw = int(binary_orig.shape[1] * scale_clamp)
                        th = int(binary_orig.shape[0] * scale_clamp)
                        base_bin, tpl_bin = binary_api, cv2.resize(binary_orig, (tw, th), interpolation=cv2.INTER_NEAREST)
                    cx = W / 2 + ox * W
                    cy = H / 2 + oy * H
                    pos_x = int(cx - tw / 2)
                    pos_y = int(cy - th / 2)
                    valid_w, valid_h = max(0, W - tw), max(0, H - th)
                    pos_x = max(0, min(valid_w, pos_x))
                    pos_y = max(0, min(valid_h, pos_y))
                    pct1, pct2, combined = _overlap_scores(base_bin, tpl_bin, (pos_x, pos_y), (tw, th))
                    if direction == "api_to_orig":
                        gemini_cfg = {
                            "direction": "api_to_orig", "scale": float(scale_clamp), "rotation": 0,
                            "position": (pos_x, pos_y), "template_size": (tw, th),
                            "score": float(combined), "score_api2orig": pct1, "score_orig2api": pct2,
                        }
                    else:
                        gemini_cfg = {
                            "direction": "orig_to_api", "scale": float(scale_clamp), "rotation": 0,
                            "position": (pos_x, pos_y), "template_size": (tw, th),
                            "score": float(combined), "score_api2orig": pct2, "score_orig2api": pct1,
                        }
                    if scale is not None and 0.2 <= scale <= 2.0 and conf >= 0.3:
                        gemini_candidate = gemini_cfg
                    print(f"      🤖 Gemini: răspuns primit – scale {scale_clamp:.3f}, suprapunere {combined:.2%}, confidence {conf:.2f}" + (" (nu folosit ca candidat)" if conf < 0.3 else ""))
                    # Poza cu alinierea Gemini – mereu salvată când există răspuns valid
                    dr = gemini_cfg["direction"]
                    suf = "api2orig" if dr == "api_to_orig" else "orig2api"
                    _save_step_overlay(
                        binary_orig if dr == "api_to_orig" else binary_api,
                        binary_api if dr == "api_to_orig" else binary_orig,
                        gemini_cfg,
                        brute_steps_dir / f"gemini_alignment_score_{combined:.3f}_{suf}.png",
                    )
            # Fără rafinare: best = câștigătorul din scale search
            best_refined = top_results[0]
            if len(top_results) > 10:
                del top_results[10:]
            # Alegere între căutare vs Gemini (dacă Gemini a returnat candidat valid)
            if gemini_candidate is not None:
                if gemini_candidate["score"] > best_refined["score"]:
                    best_refined = gemini_candidate
                    top_results[0] = gemini_candidate
                    top_results.sort(key=lambda x: x["score"], reverse=True)
                    print(f"      ✅ Variantă Gemini aleasă (aliniere directă): scale {best_refined['scale']:.3f}, suprapunere {best_refined['score']:.2%}")
                else:
                    print(f"      ✅ Variantă căutare aleasă: scale {best_refined['scale']:.3f}, suprapunere {best_refined['score']:.2%} (Gemini: {gemini_candidate['score']:.2%})")
            else:
                print(f"      ✅ Best scale {best_refined['scale']:.3f}, suprapunere {best_refined['score']:.2%}")
    
            # Cel mai bun score per interval de scale (0.1–0.2, 0.2–0.3, …) în brute_steps
            SCALE_INTERVALS = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0), (1.0, 1.1), (1.1, 1.2)]
            for lo, hi in SCALE_INTERVALS:
                cands = [(s, c) for s, c in all_scale_candidates if lo < s <= hi]
                if not cands:
                    continue
                _, best_cfg = max(cands, key=lambda x: x[1]['score'])
                name = f"best_interval_{lo:.1f}_{hi:.1f}_score_{best_cfg['score']:.3f}.png"
                if best_cfg['direction'] == 'api_to_orig':
                    _save_step_overlay(binary_orig, binary_api, best_cfg, brute_steps_dir / name)
                else:
                    _save_step_overlay(binary_api, binary_orig, best_cfg, brute_steps_dir / name)
            print(f"      📁 Best per interval salvat în {brute_steps_dir.name}/ (best_interval_*.png)")
            
            best_first = top_results[0]
            if best_first.get('score_api2orig') is not None and best_first.get('score_orig2api') is not None:
                print(f"      ✅ Best suprapunere: {best_first['score']:.2%} (api→orig: {best_first['score_api2orig']:.2%}, orig→api: {best_first['score_orig2api']:.2%})")
            else:
                print(f"      ✅ Best: {best_first['score']:.2%}")
            
            # Salvare top 5 candidați în brute_steps
            for i, cfg in enumerate(top_results[:5]):
                if cfg['direction'] == 'api_to_orig':
                    _save_step_overlay(binary_orig, binary_api, cfg, brute_steps_dir / f"best_{i+1}_score_{cfg['score']:.3f}_api2orig.png")
                else:
                    _save_step_overlay(binary_api, binary_orig, cfg, brute_steps_dir / f"best_{i+1}_score_{cfg['score']:.3f}_orig2api.png")
            print(f"      📁 Pași salvați în: {brute_steps_dir.name}/ (10 scale steps + top 5, sizing max {MAX_OVERLAY_OUTPUT_SIDE}px)")
    
            # Algoritmi suplimentari: Log-Polar FFT, Affine ECC, Coarse-to-Fine Pyramid
            run_extra_alignment_methods(raster_dir, binary_orig, binary_api)

        best = top_results[0]
        best['mask_w'] = int(api_walls_mask.shape[1])
        best['mask_h'] = int(api_walls_mask.shape[0])
        
        if best['score'] < 0.35:
            print(f"      ⚠️ Suprapunere scăzută ({best['score']:.2%}). Alinierea poate fi incorectă; verifică brute_force_best_overlay.png.")
        
        print(f"\n      🏆 CEL MAI BUN REZULTAT (min % pereți suprapuși):")
        print(f"         Suprapunere: {best['score']:.2%}")
        print(f"         Direcție: {best['direction']}")
        print(f"         Scale: {best['scale']:.3f}x")
        print(f"         Poziție: {best['position']}")
        print(f"         Template size: {best['template_size']}")
        print(f"         Mask size (API): {best['mask_w']}x{best['mask_h']}")
        
        # Salvăm configurația
        config_path = raster_dir / "brute_force_best_config.json"
        try:
            with open(config_path, 'w') as f:
                json.dump(best, f, indent=2)
            print(f"      📄 Salvat: {config_path.name}")
        except OSError as e:
            if e.errno == 28:
                print(f"      ⚠️ Disc plin: nu s-a putut salva {config_path.name}")
            raise
        
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
        overlay_to_save = _maybe_resize_to_standard(overlay_binary)
        try:
            if not cv2.imwrite(str(best_overlay_path), overlay_to_save):
                print(f"      ⚠️ Nu s-a putut salva {best_overlay_path.name} (posibil disc plin)")
        except OSError as e:
            if e.errno == 28:
                print(f"      ⚠️ Disc plin: nu s-a salvat {best_overlay_path.name}")
            else:
                raise
        else:
            print(f"      📄 Salvat: {best_overlay_path.name} (sizing max {MAX_OVERLAY_OUTPUT_SIDE}px)")
        
        # walls_brute.png: același overlay + punct albastru la fiecare capăt de perete din Raster
        # Coordonatele din JSON sunt în REQUEST space (imaginea trimisă la API); trebuie request→mask→overlay
        response_json_path = raster_dir / "response.json"
        walls_brute_img = overlay_binary.copy()
        if response_json_path.exists():
            try:
                with open(response_json_path, 'r') as f:
                    result_data = json.load(f)
                data = result_data.get('data', result_data)
                h_overlay, w_overlay = walls_brute_img.shape[:2]
                # Request vs response: când API returnează alt sizing, JSON e în response space
                req_w = best.get('request_w')
                req_h = best.get('request_h')
                mask_w = best.get('mask_w') or api_walls_mask.shape[1]
                mask_h = best.get('mask_h') or api_walls_mask.shape[0]
                scale_factor = 1.0
                request_info_path = raster_dir / "raster_request_info.json"
                if request_info_path.exists():
                    try:
                        with open(request_info_path, 'r') as f:
                            ri = json.load(f)
                        req_w = ri.get('request_w', req_w)
                        req_h = ri.get('request_h', req_h)
                        mask_w = ri.get('mask_w', mask_w)
                        mask_h = ri.get('mask_h', mask_h)
                        scale_factor = ri.get('scale_factor', 1.0)
                    except Exception:
                        pass
                scale_up = (1.0 / scale_factor) if (scale_factor and 0 < scale_factor < 1.0) else 1.0
                expected_mask_w = (req_w * scale_up) if req_w else 0
                expected_mask_h = (req_h * scale_up) if req_h else 0
                tol = 0.08
                request_matches_response = (
                    mask_w and mask_h and expected_mask_w and expected_mask_h
                    and abs(mask_w - expected_mask_w) <= max(2, expected_mask_w * tol)
                    and abs(mask_h - expected_mask_h) <= max(2, expected_mask_h * tol)
                )
                if request_matches_response:
                    r2m_x = (mask_w / req_w) if (req_w and req_w > 0) else 1.0
                    r2m_y = (mask_h / req_h) if (req_h and req_h > 0) else 1.0
                else:
                    r2m_x = scale_up
                    r2m_y = scale_up
                
                def api_to_overlay_coords(x, y):
                    # Request space → mask space → overlay (original) space
                    x_m = x * r2m_x
                    y_m = y * r2m_y
                    if best['direction'] == 'api_to_orig':
                        ox = x_m * best['scale'] + best['position'][0]
                        oy = y_m * best['scale'] + best['position'][1]
                    else:
                        ox = (x_m - best['position'][0]) / best['scale']
                        oy = (y_m - best['position'][1]) / best['scale']
                    return int(round(ox)), int(round(oy))
                
                radius = max(3, min(12, w_overlay // 200))
                blue_bgr = (255, 0, 0)
                n_pts = 0
                def point_xy(pt):
                    if isinstance(pt, dict):
                        return pt.get('x', 0), pt.get('y', 0)
                    return pt[0], pt[1]
                
                if 'walls' in data and data['walls']:
                    for wall in data['walls']:
                        pos = wall.get('position')
                        if not pos or len(pos) != 2:
                            continue
                        try:
                            x1, y1 = api_to_overlay_coords(*point_xy(pos[0]))
                            x2, y2 = api_to_overlay_coords(*point_xy(pos[1]))
                        except (IndexError, TypeError):
                            continue
                        for (px, py) in [(x1, y1), (x2, y2)]:
                            if 0 <= px < w_overlay and 0 <= py < h_overlay:
                                cv2.circle(walls_brute_img, (px, py), radius, blue_bgr, -1)
                                n_pts += 1
                walls_brute_path = raster_dir / "walls_brute.png"
                cv2.imwrite(str(walls_brute_path), walls_brute_img)
                print(f"      📄 Salvat: {walls_brute_path.name} ({n_pts} capete pereți)")
            except Exception as e:
                import traceback
                print(f"      ⚠️ walls_brute.png: {e}")
                traceback.print_exc()
                cv2.imwrite(str(raster_dir / "walls_brute.png"), overlay_binary)
        else:
            cv2.imwrite(str(raster_dir / "walls_brute.png"), overlay_binary)
            print(f"      📄 Salvat: walls_brute.png (fără response.json)")
        
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
        
        # Request vs response: când API returnează alt sizing decât request, JSON e în response space
        req_w = req_h = mask_w = mask_h = scale_factor = None
        request_info_path = raster_dir / "raster_request_info.json"
        if request_info_path.exists():
            try:
                with open(request_info_path, 'r') as f:
                    ri = json.load(f)
                req_w, req_h = ri.get('request_w'), ri.get('request_h')
                mask_w, mask_h = ri.get('mask_w'), ri.get('mask_h')
                scale_factor = ri.get('scale_factor', 1.0)
            except Exception:
                pass
        if not mask_w or not mask_h:
            mask_w, mask_h = best_config.get('mask_w'), best_config.get('mask_h')
        if not req_w or not req_h:
            req_w, req_h = mask_w, mask_h
        scale_up = (1.0 / scale_factor) if (scale_factor and 0 < scale_factor < 1.0) else 1.0
        expected_mask_w = (req_w * scale_up) if req_w else 0
        expected_mask_h = (req_h * scale_up) if req_h else 0
        tol = 0.08
        request_matches_response = (
            mask_w and mask_h and expected_mask_w and expected_mask_h
            and abs(mask_w - expected_mask_w) <= max(2, expected_mask_w * tol)
            and abs(mask_h - expected_mask_h) <= max(2, expected_mask_h * tol)
        )
        if request_matches_response:
            r2m_x = (mask_w / req_w) if (req_w and req_w > 0) else 1.0
            r2m_y = (mask_h / req_h) if (req_h and req_h > 0) else 1.0
        else:
            r2m_x = r2m_y = scale_up
        
        def api_to_original_coords(x, y):
            """Transformă coordonate din REQUEST space (JSON) la original"""
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
