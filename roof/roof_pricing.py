# roof/roof_pricing.py
"""Generate roof_pricing.json for roof-only and mixed runs."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
try:
    import google.generativeai as genai
except Exception:
    genai = None
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from config.settings import OUTPUT_ROOT, JOBS_ROOT, RUNNER_ROOT
from pricing.db_loader import fetch_pricing_parameters


DEFAULT_DAEMMUNG = "Zwischensparren"
DEFAULT_DACHDECKUNG = "Ziegel"
DEFAULT_UNTERDACH = "Folie"
DEFAULT_DACHSTUHL_TYP = "Sparrendach"
DEFAULT_DECKEN_INNENAUSBAU = "Standard"


DACHFENSTER_TYP_TO_KEYS: dict[str, tuple[str, str]] = {
    "Standard": ("dachfenster_stueck_standard", "roofonly_dachfenster_stueck_standard"),
    "Velux": ("dachfenster_stueck_velux", "roofonly_dachfenster_stueck_velux"),
    "Roto": ("dachfenster_stueck_roto", "roofonly_dachfenster_stueck_roto"),
    "Fakro": ("dachfenster_stueck_fakro", "roofonly_dachfenster_stueck_fakro"),
    "Sonstiges": ("dachfenster_stueck_sonstiges", "roofonly_dachfenster_stueck_sonstiges"),
}
DACHFENSTER_FALLBACK_EUR: dict[str, float] = {
    "Standard": 650.0,
    "Velux": 890.0,
    "Roto": 820.0,
    "Fakro": 850.0,
    "Sonstiges": 750.0,
}


GEMINI_ROOF_SYSTEM_PROMPT = (
    "Ești un motor de calcul geometric pentru acoperișuri. "
    "Primești date numerice pentru un singur dreptunghi de acoperiș și returnezi EXCLUSIV JSON. "
    "Calculează: factor_panta=1/cos(unghi_grade), "
    "amprenta_cu_overhang_m2=amprenta_fara_overhang_m2+(perimetru_fara_overhang_m*overhang_m)+((nr_colturi_exterioare-nr_colturi_interioare)*(overhang_m**2)), "
    "aria_fara_overhang_m2=(amprenta_fara_overhang_m2*factor_panta)-suprafata_geamuri_m2, "
    "aria_cu_overhang_m2=(amprenta_cu_overhang_m2*factor_panta)-suprafata_geamuri_m2. "
    "Returnează JSON cu cheia 'calcule' ce include cheile: factor_panta, amprenta_cu_overhang_m2, aria_fara_overhang_m2, aria_cu_overhang_m2, perimetru_jgheaburi_m."
)


def _count_roof_windows(out_root: Path) -> int:
    """Numără Dachfenster din roof_windows_edited.json (același fișier ca RoofReviewEditor)."""
    p = out_root / "roof" / "roof_3d" / "roof_windows_edited.json"
    if not p.exists():
        return 0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if not isinstance(data, dict):
        return 0
    n = 0
    for _k, arr in data.items():
        if isinstance(arr, list):
            n += len(arr)
    return n


def _point_in_polygon(x: float, y: float, poly: list[list[float]]) -> bool:
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        intersect = ((yi > y) != (yj > y)) and (x < ((xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi))
        if intersect:
            inside = not inside
        j = i
    return inside


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_rectangle_color_mask(mask_img: np.ndarray | None) -> np.ndarray:
    """
    Keep strictly the dominant non-black color from rectangle PNG.
    This avoids mixing blueprint/guide pixels with rectangle fill.
    """
    if mask_img is None or mask_img.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if mask_img.ndim == 2:
        # grayscale fallback: keep positive pixels only
        return (mask_img > 0).astype(np.uint8)

    bgr = mask_img[:, :, :3]
    flat = bgr.reshape(-1, 3)
    non_black = np.any(flat != 0, axis=1)
    if not np.any(non_black):
        return np.zeros(bgr.shape[:2], dtype=np.uint8)
    colors, counts = np.unique(flat[non_black], axis=0, return_counts=True)
    dominant = colors[int(np.argmax(counts))]
    # Strict color match (+/-1 tolerance for compression jitter).
    low = np.maximum(dominant.astype(np.int16) - 1, 0).astype(np.uint8)
    high = np.minimum(dominant.astype(np.int16) + 1, 255).astype(np.uint8)
    mask = cv2.inRange(bgr, low, high)
    return (mask > 0).astype(np.uint8)


def _load_mpp_by_floor_from_scale(out_root: Path) -> dict[str, float]:
    """Fallback map floor_idx -> meters_per_pixel from scale/plan_*/scale_result.json."""
    scale_root = out_root / "scale"
    out: dict[str, float] = {}
    if not scale_root.exists():
        return out
    plan_dirs = sorted([p for p in scale_root.iterdir() if p.is_dir() and p.name.startswith("plan_")], key=lambda p: p.name)
    floor_idx = 0
    for pd in plan_dirs:
        sr = pd / "scale_result.json"
        if not sr.exists():
            continue
        data = _read_json(sr)
        if not isinstance(data, dict):
            continue
        try:
            mpp = float(data.get("meters_per_pixel") or 0.0)
        except Exception:
            mpp = 0.0
        if mpp > 0:
            out[str(floor_idx)] = mpp
            floor_idx += 1
    return out


def _window_area_from_dims(win: dict[str, Any], mpp: float) -> float:
    w = win.get("width_m")
    h = win.get("height_m")
    if isinstance(w, (int, float)) and isinstance(h, (int, float)) and float(w) > 0 and float(h) > 0:
        return float(w) * float(h)
    bbox = win.get("bbox") if isinstance(win.get("bbox"), list) else None
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) * (mpp ** 2)
    return 0.0


def _pixel_perimeter(mask: np.ndarray | None) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    bin_mask = (mask > 0).astype(np.uint8)
    if np.count_nonzero(bin_mask) == 0:
        return 0.0
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(bin_mask, kernel, iterations=1)
    boundary = cv2.subtract(bin_mask, eroded)
    return float(np.count_nonzero(boundary))


def _flood_fill_contour_perimeter_px(mask: np.ndarray | None) -> float:
    """
    Perimeter in pixels by:
      1) flood-fill background to close holes logic,
      2) keep filled object,
      3) extract external contour,
      4) rasterize contour with thickness=1 and count contour pixels.
    """
    if mask is None or mask.size == 0:
        return 0.0
    bin_mask = (mask > 0).astype(np.uint8)
    if np.count_nonzero(bin_mask) == 0:
        return 0.0

    # Flood fill background from outside frame.
    h, w = bin_mask.shape[:2]
    work = (bin_mask * 255).astype(np.uint8)
    ff = work.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(ff, flood_mask, seedPoint=(0, 0), newVal=255)

    # Fill holes: foreground OR holes-inverted-background.
    ff_inv = cv2.bitwise_not(ff)
    filled = cv2.bitwise_or(work, ff_inv)
    filled_bin = (filled > 0).astype(np.uint8)

    contours, _ = cv2.findContours(filled_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    contour_canvas = np.zeros_like(filled_bin, dtype=np.uint8)
    cv2.drawContours(contour_canvas, contours, contourIdx=-1, color=1, thickness=1)
    return float(np.count_nonzero(contour_canvas))


def _polygon_signed_area(pts: list[list[float]]) -> float:
    if len(pts) < 3:
        return 0.0
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = float(pts[i][0]), float(pts[i][1])
        x2, y2 = float(pts[(i + 1) % n][0]), float(pts[(i + 1) % n][1])
        s += (x1 * y2) - (x2 * y1)
    return 0.5 * s


def _count_polygon_corner_types(pts: list[list[float]], eps: float = 1e-7) -> tuple[int, int]:
    """
    Return (exterior_corners, interior_corners) using convex/concave classification.
    For simple polygons: convex == exterior, concave == interior.
    """
    n = len(pts)
    if n < 3:
        return 0, 0
    is_ccw = _polygon_signed_area(pts) > 0
    ext = 0
    intr = 0
    for i in range(n):
        x0, y0 = float(pts[(i - 1) % n][0]), float(pts[(i - 1) % n][1])
        x1, y1 = float(pts[i][0]), float(pts[i][1])
        x2, y2 = float(pts[(i + 1) % n][0]), float(pts[(i + 1) % n][1])
        v1x, v1y = x1 - x0, y1 - y0
        v2x, v2y = x2 - x1, y2 - y1
        cross = (v1x * v2y) - (v1y * v2x)
        if abs(cross) <= eps:
            continue
        is_convex = cross > 0 if is_ccw else cross < 0
        if is_convex:
            ext += 1
        else:
            intr += 1
    # Keep parity with polygon vertices in edge cases.
    if ext + intr == 0:
        return n, 0
    return ext, intr


def _bbox_perimeter_px_from_points(pts: list[list[float]]) -> float:
    if not isinstance(pts, list) or len(pts) < 2:
        return 0.0
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    if w <= 0 or h <= 0:
        return 0.0
    return 2.0 * (w + h)


def _recompute_roof_items_costs(
    items: list[dict[str, Any]],
    area_without_overhang: float,
    area_with_overhang: float,
    tinichigerie_percent: float,
) -> tuple[list[dict[str, Any]], float, float, float]:
    subtotal = 0.0
    skylights_cost = 0.0
    for it in items:
        cat = str(it.get("category") or "")
        if cat == "roof_skylights":
            try:
                skylights_cost += float(it.get("cost") or 0.0)
            except Exception:
                pass
            continue
        if cat == "roof_tinichigerie":
            continue
        unit = str(it.get("unit") or "")
        if unit == "m²":
            applied = str(it.get("applied_area") or "")
            q = area_with_overhang if applied == "with_overhang" else area_without_overhang
            up = float(it.get("unit_price") or 0.0)
            it["quantity"] = round(q, 2)
            it["cost"] = round(q * up, 2)
        try:
            subtotal += float(it.get("cost") or 0.0)
        except Exception:
            pass
    subtotal += skylights_cost
    tin_cost = round(subtotal * tinichigerie_percent, 2)
    for it in items:
        if str(it.get("category") or "") == "roof_tinichigerie":
            it["unit_price"] = tin_cost
            it["cost"] = tin_cost
            break
    total = round(subtotal + tin_cost, 2)
    return items, round(subtotal, 2), tin_cost, total


def _compute_rectangle_inputs_for_gemini(
    out_root: Path,
    meters_per_pixel_by_floor: dict[str, Any] | None,
    overhang_m: float,
) -> list[dict[str, Any]]:
    rect_json = _read_json(out_root / "roof" / "roof_3d" / "roof_rectangles_edited.json")
    rect_walls_json = _read_json(out_root / "roof" / "roof_3d" / "roof_rectangles_edited_walls.json")
    win_json = _read_json(out_root / "roof" / "roof_3d" / "roof_windows_edited.json")
    rect_root = out_root / "roof" / "roof_3d" / "rectangles"
    if not isinstance(rect_json, dict):
        return []
    out: list[dict[str, Any]] = []
    mpp_fallback_by_floor = _load_mpp_by_floor_from_scale(out_root)
    for floor_key, rects in rect_json.items():
        if not isinstance(rects, list):
            continue
        floor_idx = int(floor_key) if str(floor_key).isdigit() else 0
        raw_mpp = float((meters_per_pixel_by_floor or {}).get(str(floor_idx)) or 0.0)
        if raw_mpp <= 0:
            raw_mpp = float(mpp_fallback_by_floor.get(str(floor_idx)) or 0.0)
        if raw_mpp <= 0:
            raw_mpp = 0.01
        windows_floor = win_json.get(str(floor_idx), []) if isinstance(win_json, dict) else []
        walls_floor = rect_walls_json.get(str(floor_idx), []) if isinstance(rect_walls_json, dict) else []
        for rect_idx, rect in enumerate(rects):
            if not isinstance(rect, dict):
                continue
            pts = rect.get("points")
            if not isinstance(pts, list) or len(pts) < 3:
                continue
            pts_walls = None
            if isinstance(walls_floor, list) and rect_idx < len(walls_floor) and isinstance(walls_floor[rect_idx], dict):
                ww = walls_floor[rect_idx].get("points")
                if isinstance(ww, list) and len(ww) >= 3:
                    pts_walls = ww
            mask_path = rect_root / f"floor_{floor_idx}" / f"rectangle_S{rect_idx}.png"
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED) if mask_path.exists() else None
            rect_mask = _extract_rectangle_color_mask(mask_img)
            area_px = float(np.count_nonzero(rect_mask > 0))
            # Requested method: flood fill + contour pixel counting on strict rectangle color.
            perimeter_px = _flood_fill_contour_perimeter_px(rect_mask)
            if area_px <= 0:
                continue
            # Coordinate alignment: editor polygons are often downscaled versus mask/walls space.
            # Convert m/px from editor-space to mask-space using paired rectangle bbox ratio.
            scale_factor = 1.0
            if pts_walls is not None and len(pts_walls) >= 3:
                try:
                    exs = [float(p[0]) for p in pts]
                    eys = [float(p[1]) for p in pts]
                    wxs = [float(p[0]) for p in pts_walls]
                    wys = [float(p[1]) for p in pts_walls]
                    ew = max(exs) - min(exs)
                    eh = max(eys) - min(eys)
                    ww = max(wxs) - min(wxs)
                    wh = max(wys) - min(wys)
                    sx = (ww / ew) if ew > 1e-9 else 1.0
                    sy = (wh / eh) if eh > 1e-9 else 1.0
                    if sx > 0 and sy > 0:
                        scale_factor = (sx + sy) / 2.0
                except Exception:
                    scale_factor = 1.0
            # NOTE: px area/perimeter are measured on mask space; use mask-space mpp directly.
            # coord_scale_factor is kept for diagnostics only.
            mpp = raw_mpp
            area_m2 = area_px * (mpp ** 2)
            perimeter_px_geom = _bbox_perimeter_px_from_points(pts_walls or pts)
            # Active perimeter method requested by user: flood-fill contour pixel count * mpp.
            perimeter_m = perimeter_px * mpp
            cx = sum(float(p[0]) for p in pts) / len(pts)
            cy = sum(float(p[1]) for p in pts) / len(pts)
            n_ext, n_int = _count_polygon_corner_types(pts)

            windows_for_rect: list[dict[str, Any]] = []
            sum_windows_m2 = 0.0
            if isinstance(windows_floor, list):
                for w_idx, win in enumerate(windows_floor):
                    if not isinstance(win, dict):
                        continue
                    bbox = win.get("bbox")
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                    wx = (x1 + x2) / 2.0
                    wy = (y1 + y2) / 2.0
                    if not _point_in_polygon(wx, wy, pts):
                        continue
                    w_area = _window_area_from_dims(win, mpp)
                    sum_windows_m2 += w_area
                    windows_for_rect.append({
                        "window_idx": w_idx,
                        "bbox": [x1, y1, x2, y2],
                        "width_m": win.get("width_m"),
                        "height_m": win.get("height_m"),
                        "area_m2": round(w_area, 4),
                    })

            out.append({
                "floor_idx": floor_idx,
                "rectangle_idx": rect_idx,
                "roof_type": rect.get("roofType"),
                "roof_angle_deg": float(rect.get("roofAngleDeg") or 0.0),
                "center_px": [round(cx, 2), round(cy, 2)],
                "area_px": round(area_px, 4),
                "perimeter_px": round(perimeter_px, 4),
                "perimeter_px_geom": round(perimeter_px_geom, 4),
                "meters_per_pixel_raw": round(raw_mpp, 8),
                "coord_scale_factor": round(scale_factor, 8),
                "meters_per_pixel": round(mpp, 8),
                "amprenta_fara_overhang_m2": round(area_m2, 4),
                "perimetru_fara_overhang_m": round(perimeter_m, 4),
                "overhang_m": round(float(overhang_m), 4),
                "nr_colturi_exterioare": n_ext,
                "nr_colturi_interioare": n_int,
                "suprafata_geamuri_m2": round(sum_windows_m2, 4),
                "windows": windows_for_rect,
            })
    return out


def _call_gemini_for_rectangle(payload: dict[str, Any], api_key: str | None) -> dict[str, Any] | None:
    if not api_key or genai is None:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt_input = {
            "suprafata_utila_m2": payload.get("amprenta_fara_overhang_m2"),
            "perimetru_m": payload.get("perimetru_fara_overhang_m"),
            "unghi_grade": payload.get("roof_angle_deg"),
            "overhang_m": payload.get("overhang_m"),
            "nr_colturi_exterioare": payload.get("nr_colturi_exterioare"),
            "nr_colturi_interioare": payload.get("nr_colturi_interioare"),
            "suprafata_geamuri_m2": payload.get("suprafata_geamuri_m2"),
        }
        resp = model.generate_content(
            [{"role": "user", "parts": [{"text": f"{GEMINI_ROOF_SYSTEM_PROMPT}\n\nINPUT:\n{json.dumps(prompt_input, ensure_ascii=False)}"}]}],
            generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
        )
        txt = (resp.text or "").strip()
        return json.loads(txt) if txt else None
    except Exception:
        return None


def _call_chatgpt_for_rectangle(payload: dict[str, Any], api_key: str | None) -> dict[str, Any] | None:
    if not api_key or OpenAI is None:
        return None
    try:
        client = OpenAI(api_key=api_key)
        model = os.environ.get("OPENAI_ROOF_MODEL", "gpt-4o")
        prompt_input = {
            "suprafata_utila_m2": payload.get("amprenta_fara_overhang_m2"),
            "perimetru_m": payload.get("perimetru_fara_overhang_m"),
            "unghi_grade": payload.get("roof_angle_deg"),
            "overhang_m": payload.get("overhang_m"),
            "nr_colturi_exterioare": payload.get("nr_colturi_exterioare"),
            "nr_colturi_interioare": payload.get("nr_colturi_interioare"),
            "suprafata_geamuri_m2": payload.get("suprafata_geamuri_m2"),
        }
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": GEMINI_ROOF_SYSTEM_PROMPT},
                {"role": "user", "content": f"INPUT:\n{json.dumps(prompt_input, ensure_ascii=False)}"},
            ],
        )
        txt = (response.choices[0].message.content or "").strip()
        if txt.startswith("```json"):
            txt = txt[7:].strip()
        elif txt.startswith("```"):
            txt = txt[3:].strip()
        if txt.endswith("```"):
            txt = txt[:-3].strip()
        return json.loads(txt) if txt else None
    except Exception:
        return None


def _local_roof_calc(payload: dict[str, Any]) -> dict[str, Any]:
    """Deterministic local fallback with the same formula as AI prompt."""
    area = float(payload.get("amprenta_fara_overhang_m2") or 0.0)
    perimeter = float(payload.get("perimetru_fara_overhang_m") or 0.0)
    angle_deg = float(payload.get("roof_angle_deg") or 0.0)
    overhang = float(payload.get("overhang_m") or 0.0)
    n_ext = float(payload.get("nr_colturi_exterioare") or 0.0)
    n_int = float(payload.get("nr_colturi_interioare") or 0.0)
    windows = float(payload.get("suprafata_geamuri_m2") or 0.0)

    # Clamp to physically reasonable interval to avoid exploding factor.
    angle_deg = max(0.0, min(80.0, angle_deg))
    cosv = math.cos(math.radians(angle_deg))
    factor = (1.0 / cosv) if cosv > 1e-8 else 1.0

    added_strip = perimeter * overhang
    corners_corr = (n_ext - n_int) * (overhang ** 2)
    ext_footprint = area + added_strip + corners_corr

    area_no_ov = max(0.0, (area * factor) - windows)
    area_with_ov = max(0.0, (ext_footprint * factor) - windows)
    # Approximate expanded perimeter by adding 2*OH per corner.
    gutter_perimeter = max(0.0, perimeter + (2.0 * overhang * (n_ext - n_int)))

    return {
        "calcule": {
            "factor_panta": round(factor, 6),
            "amprenta_cu_overhang_m2": round(ext_footprint, 6),
            "aria_fara_overhang_m2": round(area_no_ov, 6),
            "aria_cu_overhang_m2": round(area_with_ov, 6),
            "perimetru_jgheaburi_m": round(gutter_perimeter, 6),
        },
        "mentiuni_tehnice": "Fallback local deterministic (same formula as AI prompt).",
    }


def _get_out_root(run_id: str) -> Path | None:
    out = OUTPUT_ROOT / run_id
    if out.exists():
        return out
    out = JOBS_ROOT / run_id / "output"
    if out.exists():
        return out
    out = RUNNER_ROOT / "output" / run_id
    return out if out.exists() else None


def _extract_rectangle_measurements(out_root: Path, meters_per_pixel_by_floor: dict[str, Any] | None, overhang_m: float = 0.4) -> list[dict[str, Any]]:
    """
    Build per-rectangle roof measurements from roof_rectangles_edited.json.
    Returns for each rectangle:
      - roof faces area without overhang (m²)
      - roof faces area with overhang (m²)
      - outer perimeter with overhang (m)
    """
    edited_path = out_root / "roof" / "roof_3d" / "roof_rectangles_edited.json"
    if not edited_path.exists():
        return []
    try:
        edited = json.loads(edited_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(edited, dict):
        return []

    out: list[dict[str, Any]] = []
    for floor_key, rects in edited.items():
        if not isinstance(rects, list):
            continue
        mpp = float((meters_per_pixel_by_floor or {}).get(str(floor_key)) or 0.0)
        if mpp <= 0:
            mpp = 0.01
        for idx, rect in enumerate(rects):
            pts = rect.get("points") if isinstance(rect, dict) else None
            if not isinstance(pts, list) or len(pts) < 4:
                continue
            try:
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                w_px = max(0.0, maxx - minx)
                h_px = max(0.0, maxy - miny)
                if w_px <= 0 or h_px <= 0:
                    continue
                angle_deg = float(rect.get("roofAngleDeg") or 0.0)
                angle_deg = max(0.0, min(80.0, angle_deg))
                cosv = math.cos(math.radians(angle_deg))
                slope_factor = (1.0 / cosv) if cosv > 1e-6 else 1.0

                base_area_m2 = (w_px * h_px) * (mpp ** 2)
                area_without_overhang_m2 = base_area_m2 * slope_factor

                overhang_px = float(overhang_m) / float(mpp)
                w_ov_px = max(0.0, w_px + 2.0 * overhang_px)
                h_ov_px = max(0.0, h_px + 2.0 * overhang_px)
                base_area_with_ov_m2 = (w_ov_px * h_ov_px) * (mpp ** 2)
                area_with_overhang_m2 = base_area_with_ov_m2 * slope_factor

                perimeter_with_overhang_m = (2.0 * (w_ov_px + h_ov_px)) * mpp
                out.append({
                    "floor_idx": int(floor_key) if str(floor_key).isdigit() else 0,
                    "rectangle_idx": idx,
                    "roof_type": rect.get("roofType"),
                    "roof_angle_deg": round(angle_deg, 2),
                    "roof_area_without_overhang_m2": round(area_without_overhang_m2, 4),
                    "roof_area_with_overhang_m2": round(area_with_overhang_m2, 4),
                    "roof_perimeter_with_overhang_m": round(perimeter_with_overhang_m, 4),
                })
            except Exception:
                continue
    return out


def _mask_area_and_contour_px(mask: np.ndarray | None) -> tuple[float, float]:
    if mask is None or mask.size == 0:
        return 0.0, 0.0
    bin_mask = (mask > 0).astype(np.uint8)
    area_px = float(np.count_nonzero(bin_mask))
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_px = float(sum(cv2.arcLength(c, True) for c in contours if len(c) >= 3))
    return area_px, contour_px


def _sum_masks_area_contour(folder: Path) -> tuple[float, float]:
    if not folder.exists() or not folder.is_dir():
        return 0.0, 0.0
    total_area = 0.0
    total_contour = 0.0
    for p in sorted(folder.glob("*.png")):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        a, c = _mask_area_and_contour_px(m)
        total_area += a
        total_contour += c
    return total_area, total_contour


def _extract_rectangle_measurements_from_3d(out_root: Path, meters_per_pixel_by_floor: dict[str, Any] | None) -> list[dict[str, Any]]:
    """
    Preferred path: use per-rectangle 3D unfold masks generated by clean_workflow.
    """
    rect3d_root = out_root / "roof" / "roof_3d" / "rectangles_3d"
    edited_path = out_root / "roof" / "roof_3d" / "roof_rectangles_edited.json"
    if not rect3d_root.exists() or not edited_path.exists():
        return []
    try:
        edited = json.loads(edited_path.read_text(encoding="utf-8"))
    except Exception:
        edited = {}
    out: list[dict[str, Any]] = []
    for floor_dir in sorted(rect3d_root.glob("floor_*")):
        if not floor_dir.is_dir():
            continue
        try:
            floor_idx = int(floor_dir.name.split("_", 1)[1])
        except Exception:
            continue
        mpp = float((meters_per_pixel_by_floor or {}).get(str(floor_idx)) or 0.0)
        if mpp <= 0:
            mpp = 0.01
        floor_rects = edited.get(str(floor_idx)) if isinstance(edited, dict) else None
        for rect_dir in sorted(floor_dir.glob("rectangle_S*")):
            if not rect_dir.is_dir():
                continue
            try:
                rect_idx = int(rect_dir.name.replace("rectangle_S", ""))
            except Exception:
                continue
            roof_type = None
            roof_angle = None
            if isinstance(floor_rects, list) and 0 <= rect_idx < len(floor_rects):
                rr = floor_rects[rect_idx] if isinstance(floor_rects[rect_idx], dict) else {}
                roof_type = rr.get("roofType")
                roof_angle = rr.get("roofAngleDeg")
            chosen_rt = str(roof_type or "2_w")
            unfold_roof_dir = rect_dir / chosen_rt / "unfold_roof"
            unfold_overhang_dir = rect_dir / chosen_rt / "unfold_overhang"
            roof_area_px, _roof_contour_px = _sum_masks_area_contour(unfold_roof_dir)
            overhang_area_px, overhang_contour_px = _sum_masks_area_contour(unfold_overhang_dir)
            # Perimeter requested "with overhang": roof + overhang outer contour from unfolded masks.
            perimeter_with_overhang_m = overhang_contour_px * mpp
            area_without_overhang_m2 = roof_area_px * (mpp ** 2)
            area_with_overhang_m2 = (roof_area_px + overhang_area_px) * (mpp ** 2)
            if area_without_overhang_m2 <= 0 and area_with_overhang_m2 <= 0:
                continue
            out.append({
                "floor_idx": floor_idx,
                "rectangle_idx": rect_idx,
                "roof_type": roof_type,
                "roof_angle_deg": round(float(roof_angle), 2) if roof_angle is not None else None,
                "roof_area_without_overhang_m2": round(area_without_overhang_m2, 4),
                "roof_area_with_overhang_m2": round(area_with_overhang_m2, 4),
                "roof_perimeter_with_overhang_m": round(perimeter_with_overhang_m, 4),
            })
    return out


def generate_roof_pricing(run_id: str, frontend_data: dict | None = None) -> Path | None:
    """
    Generează roof_pricing.json în roof/roof_3d/entire/mixed/.
    Returnează path-ul fișierului sau None dacă nu s-a putut genera.
    """
    out_root = _get_out_root(run_id)
    if not out_root:
        print(f"⚠️ [ROOF PRICING] Output nu există: {run_id}")
        return None

    metrics_path = out_root / "roof" / "roof_3d" / "entire" / "mixed" / "roof_metrics.json"
    if not metrics_path.exists():
        print(f"⚠️ [ROOF PRICING] roof_metrics.json lipsește: {metrics_path}")
        return None

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    # Prefer explicit unfolded metrics to avoid any overlap/double-count issues.
    total = metrics.get("total") or {}
    total_combined = metrics.get("total_combined") or {}
    unfold_roof = metrics.get("unfold_roof") or {}
    unfold_overhang = metrics.get("unfold_overhang") or {}
    roof_total = unfold_roof.get("total") or {}
    overhang_total = unfold_overhang.get("total") or {}
    area_without_overhang = float(roof_total.get("area_m2") or total.get("area_m2") or 0.0)
    overhang_area = float(overhang_total.get("area_m2") or 0.0)
    area_with_overhang = area_without_overhang + overhang_area
    contour_m = float(total_combined.get("contour_m") or total.get("contour_m") or 0.0)
    if area_without_overhang <= 0:
        print(f"⚠️ [ROOF PRICING] area_without_overhang invalid în roof_metrics: {area_without_overhang}")
        return None

    dd = (frontend_data or {}).get("daemmungDachdeckung") or {}
    pd = (frontend_data or {}).get("projektdaten") or {}
    daemmung = dd.get("daemmung") or DEFAULT_DAEMMUNG
    dachdeckung = dd.get("dachdeckung") or DEFAULT_DACHDECKUNG
    unterdach = dd.get("unterdach") or DEFAULT_UNTERDACH
    dachstuhl_typ = dd.get("dachstuhlTyp") or DEFAULT_DACHSTUHL_TYP
    sichtdachstuhl = bool(dd.get("sichtdachstuhl"))
    projektumfang = pd.get("projektumfang") or "Dachstuhl + Dachdeckung"
    nutzung_dachraum = pd.get("nutzungDachraum")
    decken_innenausbau = pd.get("deckenInnenausbau") or DEFAULT_DECKEN_INNENAUSBAU

    roof_cfg = {}
    try:
        tenant_slug = (frontend_data or {}).get("tenant_slug") or "holzbau"
        calc_mode = (frontend_data or {}).get("calc_mode")
        pricing_cfg = fetch_pricing_parameters(str(tenant_slug), str(calc_mode) if calc_mode else None) or {}
        roof_cfg = (pricing_cfg.get("roof") or {})
    except Exception as e:
        print(f"⚠️ [ROOF PRICING] Nu s-au putut încărca parametrii DB, folosesc fallback: {e}")

    def _price(key: str, fallback: float) -> float:
        try:
            return float(roof_cfg.get(key, fallback) or fallback)
        except Exception:
            return fallback

    price_daemmung_map = {
        "Keine": _price("roofonly_daemmung_keine_price", 0.0),
        "Zwischensparren": _price("roofonly_daemmung_zwischensparren_price", 62.0),
        "Aufsparren": _price("roofonly_daemmung_aufsparren_price", 92.0),
        "Kombination": _price("roofonly_daemmung_kombination_price", 108.0),
    }
    price_unterdach_map = {
        "Folie": _price("roofonly_unterdach_folie_price", 14.0),
        "Schalung + Folie": _price("roofonly_unterdach_schalung_folie_price", 32.0),
    }
    price_dachstuhl_map = {
        "Sparrendach": _price("roofonly_dachstuhl_sparrendach_price", 96.0),
        "Pfettendach": _price("roofonly_dachstuhl_pfettendach_price", 114.0),
        "Kehlbalkendach": _price("roofonly_dachstuhl_kehlbalkendach_price", 108.0),
        "Sonderkonstruktion": _price("roofonly_dachstuhl_sonderkonstruktion_price", 138.0),
    }
    price_dachdeckung_map = {
        "Ziegel": _price("roofonly_dachdeckung_ziegel_price", 84.0),
        "Betonstein": _price("roofonly_dachdeckung_betonstein_price", 74.0),
        "Blech": _price("roofonly_dachdeckung_blech_price", 69.0),
        "Schindel": _price("roofonly_dachdeckung_schindel_price", 89.0),
        "Sonstiges": _price("roofonly_dachdeckung_sonstiges_price", 78.0),
    }
    price_decken_innenausbau_map = {
        "Standard": _price("roofonly_decken_innenausbau_standard_price", 38.0),
        "Premium": _price("roofonly_decken_innenausbau_premium_price", 56.0),
        "Exklusiv": _price("roofonly_decken_innenausbau_exklusiv_price", 74.0),
    }
    tinichigerie_percent = _price("roofonly_tinichigerie_percent", 5.0) / 100.0

    price_daemmung = float(price_daemmung_map.get(daemmung, price_daemmung_map[DEFAULT_DAEMMUNG]))
    price_unterdach = float(price_unterdach_map.get(unterdach, price_unterdach_map[DEFAULT_UNTERDACH]))
    price_dachstuhl = float(price_dachstuhl_map.get(dachstuhl_typ, price_dachstuhl_map[DEFAULT_DACHSTUHL_TYP]))
    price_dachdeckung = float(price_dachdeckung_map.get(dachdeckung, price_dachdeckung_map[DEFAULT_DACHDECKUNG]))
    price_sichtdachstuhl = _price("roofonly_sichtdachstuhl_price", 36.0)
    price_decken_innenausbau = float(price_decken_innenausbau_map.get(decken_innenausbau, price_decken_innenausbau_map[DEFAULT_DECKEN_INNENAUSBAU]))

    include_dachstuhl = projektumfang in ("Dachstuhl", "Dachstuhl + Dachdeckung")
    include_dachdeckung = projektumfang in ("Dachdeckung", "Dachstuhl + Dachdeckung")

    items: list[dict[str, Any]] = []
    total_cost = 0.0
    subtotal_before_services = 0.0

    # Dämmung -> roof area WITHOUT overhang
    if include_dachstuhl and price_daemmung > 0:
        cost = round(area_without_overhang * price_daemmung, 2)
        items.append({
            "category": "roof_insulation",
            "name": "Dämmung",
            "details": f"Material: {daemmung}",
            "quantity": round(area_without_overhang, 2),
            "unit": "m²",
            "unit_price": price_daemmung,
            "cost": cost,
            "material": daemmung,
            "applied_area": "without_overhang",
        })
        subtotal_before_services += cost

    # Unterdach -> roof area WITH overhang
    if include_dachstuhl and price_unterdach > 0:
        cost = round(area_with_overhang * price_unterdach, 2)
        items.append({
            "category": "roof_unterdach",
            "name": "Unterspannbahn / Folie",
            "details": f"Material: {unterdach}",
            "quantity": round(area_with_overhang, 2),
            "unit": "m²",
            "unit_price": price_unterdach,
            "cost": cost,
            "material": unterdach,
            "applied_area": "with_overhang",
        })
        subtotal_before_services += cost

    # Dachstuhl-Typ -> roof area WITH overhang
    if include_dachstuhl and price_dachstuhl > 0:
        cost = round(area_with_overhang * price_dachstuhl, 2)
        items.append({
            "category": "roof_structure",
            "name": "Dachstuhl",
            "details": f"Typ: {dachstuhl_typ}",
            "quantity": round(area_with_overhang, 2),
            "unit": "m²",
            "unit_price": price_dachstuhl,
            "cost": cost,
            "material": dachstuhl_typ,
            "applied_area": "with_overhang",
        })
        subtotal_before_services += cost

    # Sichtdachstuhl -> roof area WITHOUT overhang
    if include_dachstuhl and sichtdachstuhl and price_sichtdachstuhl > 0:
        cost = round(area_without_overhang * price_sichtdachstuhl, 2)
        items.append({
            "category": "roof_sichtdachstuhl",
            "name": "Sichtdachstuhl",
            "details": "Sichtqualität Dachstuhl",
            "quantity": round(area_without_overhang, 2),
            "unit": "m²",
            "unit_price": price_sichtdachstuhl,
            "cost": cost,
            "applied_area": "without_overhang",
        })
        subtotal_before_services += cost

    # Dachdeckung -> roof area WITH overhang
    if include_dachdeckung and price_dachdeckung > 0:
        cost = round(area_with_overhang * price_dachdeckung, 2)
        items.append({
            "category": "roof_cover",
            "name": "Dachdeckung",
            "details": f"Material: {dachdeckung}",
            "quantity": round(area_with_overhang, 2),
            "unit": "m²",
            "unit_price": price_dachdeckung,
            "cost": cost,
            "material": dachdeckung,
            "applied_area": "with_overhang",
        })
        subtotal_before_services += cost

    # Decken-Innenausbau only when attic is habitable
    if nutzung_dachraum == "Wohnraum / ausgebaut" and price_decken_innenausbau > 0:
        cost = round(area_without_overhang * price_decken_innenausbau, 2)
        items.append({
            "category": "roof_decken_innenausbau",
            "name": "Decken-Innenausbau",
            "details": f"Ausführung: {decken_innenausbau}",
            "quantity": round(area_without_overhang, 2),
            "unit": "m²",
            "unit_price": price_decken_innenausbau,
            "cost": cost,
            "material": decken_innenausbau,
            "applied_area": "without_overhang",
        })
        subtotal_before_services += cost

    # Dachfenster (Stück): formular „Dachfenster einplanen“ + tip; cantitate din editor (roof_windows_edited.json)
    roof_only_offer = bool((frontend_data or {}).get("roof_only_offer"))
    want_dachfenster = bool(dd.get("dachfensterImDach"))
    dachfenster_typ = str(dd.get("dachfensterTyp") or "").strip() or "Standard"
    if want_dachfenster and out_root:
        n_df = _count_roof_windows(out_root)
        if n_df > 0:
            pair = DACHFENSTER_TYP_TO_KEYS.get(dachfenster_typ, DACHFENSTER_TYP_TO_KEYS["Standard"])
            pk = pair[1] if roof_only_offer else pair[0]
            fb = float(DACHFENSTER_FALLBACK_EUR.get(dachfenster_typ, DACHFENSTER_FALLBACK_EUR["Standard"]))
            price_df = _price(pk, fb)
            if price_df > 0:
                cost_df = round(n_df * price_df, 2)
                items.append({
                    "category": "roof_skylights",
                    "name": "Dachfenster",
                    "details": f"Ausführung: {dachfenster_typ} ({n_df} Stück)",
                    "quantity": float(n_df),
                    "unit": "Stück",
                    "unit_price": price_df,
                    "cost": cost_df,
                    "material": dachfenster_typ,
                })
                subtotal_before_services += cost_df

    # Leistungen enthalten removed from roof-only flow.

    total_cost += subtotal_before_services

    tinichigerie_cost = round(subtotal_before_services * tinichigerie_percent, 2)
    if tinichigerie_cost > 0:
        items.append({
            "category": "roof_tinichigerie",
            "name": "Klempnerarbeiten (Rinnen, Bleche, Anschlüsse)",
            "details": "Dachrinnen, Traufbleche, Anschlüsse",
            "quantity": 1,
            "unit": "Pauschale",
            "unit_price": tinichigerie_cost,
            "cost": tinichigerie_cost,
        })
        total_cost += tinichigerie_cost

    roof_measurements: dict[str, Any] = {
        "area_m2": round(area_without_overhang, 4),
        "contour_m": round(contour_m, 4),
        "roof_area_without_overhang_m2": round(area_without_overhang, 4),
        "roof_area_overhang_only_m2": round(overhang_area, 4),
        "roof_area_with_overhang_m2": round(area_with_overhang, 4),
    }
    if unfold_roof and unfold_roof.get("total"):
        roof_measurements["unfold_roof"] = {
            "area_m2": unfold_roof["total"].get("area_m2"),
            "contour_m": unfold_roof["total"].get("contour_m"),
        }
    if unfold_overhang and unfold_overhang.get("total"):
        roof_measurements["unfold_overhang"] = {
            "area_m2": unfold_overhang["total"].get("area_m2"),
            "contour_m": unfold_overhang["total"].get("contour_m"),
        }
    if total_combined:
        roof_measurements["total_combined"] = {
            "area_m2": total_combined.get("area_m2"),
            "contour_m": total_combined.get("contour_m"),
        }
    by_rect = _extract_rectangle_measurements_from_3d(
        out_root=out_root,
        meters_per_pixel_by_floor=metrics.get("meters_per_pixel_by_floor") or {},
    )
    if not by_rect:
        by_rect = _extract_rectangle_measurements(
            out_root=out_root,
            meters_per_pixel_by_floor=metrics.get("meters_per_pixel_by_floor") or {},
            overhang_m=_price("overhang_m", 0.4),
        )
    roof_measurements["by_rectangle"] = by_rect

    # Rectangle-level inputs and Gemini estimation per rectangle.
    overhang_m = _price("overhang_m", 0.4)
    rect_inputs = _compute_rectangle_inputs_for_gemini(
        out_root=out_root,
        meters_per_pixel_by_floor=metrics.get("meters_per_pixel_by_floor") or {},
        overhang_m=overhang_m,
    )
    roof_measurements["by_rectangle_input"] = rect_inputs

    gemini_key = (frontend_data or {}).get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
    openai_key = (frontend_data or {}).get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
    gemini_by_rect: list[dict[str, Any]] = []
    sum_ai_without = 0.0
    sum_ai_with = 0.0
    for r in rect_inputs:
        g = _call_gemini_for_rectangle(r, gemini_key)
        source = "gemini"
        if g is None:
            g = _call_chatgpt_for_rectangle(r, openai_key)
            source = "chatgpt" if g is not None else "none"
        if g is None:
            g = _local_roof_calc(r)
            source = "local"
        entry = {"floor_idx": r.get("floor_idx"), "rectangle_idx": r.get("rectangle_idx"), "provider": source, "input": r, "output": g}
        gemini_by_rect.append(entry)
        calc = (g or {}).get("calcule") if isinstance(g, dict) else None
        if isinstance(calc, dict):
            try:
                a0 = float(calc.get("aria_fara_overhang_m2") or 0.0)
                a1 = float(calc.get("aria_cu_overhang_m2") or 0.0)
                if a0 > 0:
                    sum_ai_without += a0
                if a1 > 0:
                    sum_ai_with += a1
            except Exception:
                pass
    roof_measurements["gemini_by_rectangle"] = gemini_by_rect
    sum_ai_perimeter = 0.0
    by_rect_ai: list[dict[str, Any]] = []
    for e in gemini_by_rect:
        inp = e.get("input") if isinstance(e, dict) else {}
        outp = e.get("output") if isinstance(e, dict) else {}
        calc = (outp or {}).get("calcule") if isinstance(outp, dict) else None
        if not isinstance(inp, dict):
            continue
        rec = {
            "floor_idx": int(inp.get("floor_idx", 0)),
            "rectangle_idx": int(inp.get("rectangle_idx", 0)),
            "roof_type": inp.get("roof_type"),
            "roof_angle_deg": inp.get("roof_angle_deg"),
            "roof_area_without_overhang_m2": None,
            "roof_area_with_overhang_m2": None,
            "roof_perimeter_with_overhang_m": None,
            "roof_windows_area_m2": float(inp.get("suprafata_geamuri_m2") or 0.0),
            "corners_exterior": int(inp.get("nr_colturi_exterioare") or 0),
            "corners_interior": int(inp.get("nr_colturi_interioare") or 0),
        }
        if isinstance(calc, dict):
            try:
                rec["roof_area_without_overhang_m2"] = round(float(calc.get("aria_fara_overhang_m2") or 0.0), 4)
                rec["roof_area_with_overhang_m2"] = round(float(calc.get("aria_cu_overhang_m2") or 0.0), 4)
                pj = float(calc.get("perimetru_jgheaburi_m") or 0.0)
                rec["roof_perimeter_with_overhang_m"] = round(pj, 4)
            except Exception:
                pass
        by_rect_ai.append(rec)
    if by_rect_ai:
        roof_measurements["by_rectangle"] = by_rect_ai
    if sum_ai_without > 0 and sum_ai_with > 0:
        area_without_overhang = sum_ai_without
        area_with_overhang = sum_ai_with
        overhang_area = max(0.0, area_with_overhang - area_without_overhang)
    for e in gemini_by_rect:
        calc = ((e.get("output") if isinstance(e, dict) else {}) or {}).get("calcule", {})
        if isinstance(calc, dict):
            try:
                pj = float(calc.get("perimetru_jgheaburi_m") or 0.0)
                if pj > 0:
                    sum_ai_perimeter += pj
            except Exception:
                pass
    if sum_ai_perimeter > 0:
        contour_m = sum_ai_perimeter
    roof_measurements["area_m2"] = round(area_without_overhang, 4)
    roof_measurements["contour_m"] = round(contour_m, 4)
    roof_measurements["roof_area_without_overhang_m2"] = round(area_without_overhang, 4)
    roof_measurements["roof_area_overhang_only_m2"] = round(overhang_area, 4)
    roof_measurements["roof_area_with_overhang_m2"] = round(area_with_overhang, 4)

    items, subtotal_before_services, _tin, total_cost = _recompute_roof_items_costs(
        items=items,
        area_without_overhang=area_without_overhang,
        area_with_overhang=area_with_overhang,
        tinichigerie_percent=tinichigerie_percent,
    )

    result = {
        "form_data": {
            "daemmung": daemmung,
            "dachdeckung": dachdeckung,
            "unterdach": unterdach,
            "dachstuhlTyp": dachstuhl_typ,
            "sichtdachstuhl": sichtdachstuhl,
            "dachfensterImDach": bool(dd.get("dachfensterImDach")),
            "dachfensterTyp": dachfenster_typ if want_dachfenster else None,
            "dachfenster_count": _count_roof_windows(out_root) if out_root else 0,
            "projektumfang": projektumfang,
            "nutzungDachraum": nutzung_dachraum,
            "deckenInnenausbau": decken_innenausbau,
        },
        "metrics": {
            "area_m2": area_without_overhang,
            "contour_m": contour_m,
        },
        "roof_measurements": roof_measurements,
        "items": items,
        "detailed_items": items,
        "subtotal_before_services": round(subtotal_before_services, 2),
        "total_cost": round(total_cost, 2),
    }

    out_dir = metrics_path.parent
    out_path = out_dir / "roof_pricing.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ [ROOF PRICING] Generat: {out_path} (total: {total_cost:,.2f} EUR)")
    return out_path
