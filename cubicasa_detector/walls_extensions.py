# -*- coding: utf-8 -*-
"""
Balcon / Wintergarten: interior coloring, parapet wall detection,
and measurements for extensions_measurements.json + 13_extensions.png.

Wintergarten room type is drawn and metered like a normal room (magenta); WG floor /
glass façade are not split into extensions_measurements.

BGR conventions (OpenCV):
  - Balkon interior:  pure blue   (255, 0, 0)
  - Wintergarten:     same magenta as other rooms (no separate WG layer)
  - Other rooms:      magenta     (255, 0, 255)
  - Legacy interior: orange      (0, 165, 255) — still counted as interior for area_px
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

BGR_BALKON_INTERIOR = (255, 0, 0)
BGR_MAIN_INTERIOR = (255, 0, 255)
BGR_ORANGE_LEGACY = (0, 165, 255)
# Parapet / terasa railing highlight on 13_extensions (cyan)
BGR_TERRACE_PARAPET = (255, 255, 0)


def load_room_types_parallel(raster_dir: Path, n_hint: int) -> list[str]:
    """room_types.json = list of German labels, same order as editor rooms."""
    path = raster_dir / "room_types.json"
    if not path.exists():
        return ["Raum"] * max(0, n_hint)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ["Raum"] * max(0, n_hint)
    if not isinstance(raw, list):
        return ["Raum"] * max(0, n_hint)
    out = [str(x).strip() if isinstance(x, str) else "Raum" for x in raw]
    while len(out) < n_hint:
        out.append("Raum")
    return out[: max(n_hint, len(out))] if n_hint > 0 else out


def _norm_type(s: str) -> str:
    t = (s or "").strip().lower()
    if "winter" in t:
        return "wintergarten"
    if "balkon" in t or "teras" in t:
        return "balkon"
    return "other"


def build_typed_interior_layer(
    h: int,
    w: int,
    interior_mask: np.ndarray,
    room_polys: Sequence[np.ndarray],
    room_types: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (vis_bgr, mask_balkon, mask_wg, mask_main) all HxW.
    vis_bgr: colored interior only (black elsewhere).
    """
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    balkon = np.zeros((h, w), dtype=np.uint8)
    wg = np.zeros((h, w), dtype=np.uint8)
    main = np.zeros((h, w), dtype=np.uint8)
    if cv2 is None:
        return vis, balkon, wg, main
    int_u8 = (interior_mask > 0).astype(np.uint8) * 255
    n = min(len(room_polys), len(room_types))
    for i in range(n):
        poly = room_polys[i]
        if poly is None or len(poly) < 3:
            continue
        layer = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(layer, [np.asarray(poly, dtype=np.int32)], 255)
        layer = cv2.bitwise_and(layer, int_u8)
        kind = _norm_type(room_types[i])
        if kind == "balkon":
            vis[layer > 0] = BGR_BALKON_INTERIOR
            balkon[layer > 0] = 255
        elif kind == "wintergarten":
            # Same as a normal heated room: no separate WG floor / glass wall extraction.
            vis[layer > 0] = BGR_MAIN_INTERIOR
            main[layer > 0] = 255
        else:
            vis[layer > 0] = BGR_MAIN_INTERIOR
            main[layer > 0] = 255
    # Interior pixels not covered by any typed polygon → main (mov)
    assigned = (balkon > 0) | (wg > 0) | (main > 0)
    rest = (int_u8 > 0) & (~assigned)
    vis[rest] = BGR_MAIN_INTERIOR
    main[rest] = 255
    return vis, balkon, wg, main


def _dilate_u8(mask: np.ndarray, k: int = 5) -> np.ndarray:
    if cv2 is None:
        return mask
    kk = max(3, int(k))
    if kk % 2 == 0:
        kk += 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
    return cv2.dilate(mask.astype(np.uint8), ker, iterations=1)


def mask_balkon_parapet_walls(
    wall_bin: np.ndarray,
    balkon_interior_mask: np.ndarray,
    main_interior_mask: np.ndarray,
    flood_exterior: np.ndarray,
) -> np.ndarray:
    """
    Parapet / railing: wall pixels that touch both Balkon interior and exterior flood,
    excluding the shared boundary wall between Balkon and main house (both Balkon + main dilated).
    """
    if cv2 is None:
        return np.zeros_like(wall_bin)
    db = _dilate_u8(balkon_interior_mask, 7)
    dm = _dilate_u8(main_interior_mask, 5)
    de = _dilate_u8(flood_exterior.astype(np.uint8) * 255, 5)
    shared_house = (db > 0) & (dm > 0) & (wall_bin > 0)
    parapet = (wall_bin > 0) & (db > 0) & (de > 0) & (~shared_house)
    return (parapet.astype(np.uint8)) * 255


def mask_wintergarten_exterior_glass_walls(
    wall_bin: np.ndarray,
    wg_interior_mask: np.ndarray,
    main_interior_mask: np.ndarray,
    flood_exterior: np.ndarray,
) -> np.ndarray:
    """
    Wintergarten glass shell: wall pixels touching WG interior and exterior flood,
    excluding the WG–house partition (both WG and main dilated).
    """
    if cv2 is None:
        return np.zeros_like(wall_bin)
    dw = _dilate_u8(wg_interior_mask, 7)
    dm = _dilate_u8(main_interior_mask, 5)
    de = _dilate_u8((flood_exterior > 0).astype(np.uint8) * 255, 5)
    shared = (dw > 0) & (dm > 0) & (wall_bin > 0)
    wg_ext = (wall_bin > 0) & (dw > 0) & (de > 0) & (~shared)
    return (wg_ext.astype(np.uint8)) * 255


def compose_01_walls_marked(
    walls_bgr: np.ndarray,
    interior_vis: np.ndarray,
) -> np.ndarray:
    """walls_bgr: 01-style (white walls on black). interior_vis: colors inside rooms only."""
    if cv2 is None:
        return walls_bgr.copy()
    out = walls_bgr.copy()
    int_gray = cv2.cvtColor(interior_vis, cv2.COLOR_BGR2GRAY)
    fill = interior_vis.copy()
    fill[int_gray == 0] = 0
    out = cv2.max(out, fill)
    return out


def compose_13_extensions(
    balkon_wall_mask: np.ndarray,
    wintergarten_wall_mask: np.ndarray,
) -> np.ndarray:
    """Only special extension walls in white on black."""
    if cv2 is None:
        return np.zeros_like(balkon_wall_mask, dtype=np.uint8)
    h, w = balkon_wall_mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    special = (balkon_wall_mask > 0) | (wintergarten_wall_mask > 0)
    out[special] = (255, 255, 255)
    return out


def strip_masks_from_wall_binary(
    wall_bin: np.ndarray,
    strip_parapet: np.ndarray,
    strip_wg_glass: np.ndarray,
) -> np.ndarray:
    out = wall_bin.copy()
    out[strip_parapet > 0] = 0
    out[strip_wg_glass > 0] = 0
    return out


def flood_exterior_from_wall_1px(wall_bin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Same corner/margin flood as raster_processing (4-connectivity on 255-free).
    Returns (flood_exterior_u8 255=exterior, interior_mask_u8 255=inside building).
    """
    h, w = wall_bin.shape[:2]
    ext = np.zeros((h, w), dtype=np.uint8)
    if cv2 is None:
        return ext, np.zeros((h, w), dtype=np.uint8)
    flood_base = np.where(wall_bin > 0, 0, 255).astype(np.uint8)
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    step = max(1, min(50, w // 10, h // 10))
    for px in range(0, w, step):
        seeds.append((px, 0))
        seeds.append((px, h - 1))
    for py in range(0, h, step):
        seeds.append((0, py))
        seeds.append((w - 1, py))
    flood_any = np.zeros((h, w), dtype=np.uint8)
    for cx, cy in seeds:
        if cy >= h or cx >= w:
            continue
        if flood_base[cy, cx] != 255:
            continue
        region_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        img_copy = flood_base.copy()
        cv2.floodFill(img_copy, region_mask, (cx, cy), 128, None, None, cv2.FLOODFILL_MASK_ONLY | 4)
        r = region_mask[1:-1, 1:-1] > 0
        flood_any[r] = 255
    interior = ((flood_any == 0) & (wall_bin == 0)).astype(np.uint8) * 255
    return flood_any, interior


def write_extensions_json(
    path: Path,
    mpp: float | None,
    px_balkon_floor: int,
    px_wg_floor: int,
    px_wg_glass_wall: int,
    px_parapet: int,
    room_height_m: float | None = None,
) -> None:
    mp = float(mpp) if mpp is not None and float(mpp) > 0 else None
    data: dict[str, Any] = {
        "balkon_floor_px": int(px_balkon_floor),
        "wintergarten_floor_px": int(px_wg_floor),
        "wintergarten_glass_wall_px": int(px_wg_glass_wall),
        "balkon_parapet_wall_px": int(px_parapet),
        "mpp": mp,
    }
    if mp:
        data["balkon_floor_m2"] = round(px_balkon_floor * mp * mp, 4)
        data["wintergarten_floor_m2"] = round(px_wg_floor * mp * mp, 4)
        # Pereți 1px: lungime ≈ px × m/px; suprafață vitrată ≈ lungime × înălțime cameră
        data["wintergarten_glass_wall_length_m"] = round(px_wg_glass_wall * mp, 4)
        rh = float(room_height_m) if room_height_m and room_height_m > 0 else None
        if rh:
            data["wintergarten_glass_facade_m2"] = round(float(px_wg_glass_wall) * mp * rh, 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
