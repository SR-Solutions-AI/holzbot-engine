# roof/insulated_area.py
"""Dachfläche gedämmt: subtract uninsulated room polygons (plan view) from roof ohne Überstand, with slope factor."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# Poligoane acoperiș + camere sunt în pixelii canvas-ului review (imageWidth x imageHeight).
# m/px din scale e calibrat pe 01_walls_from_coords.png (altă rezoluție). Aria în px² editor
# trebuie înmulțită cu mpp_walls² * sx * sy ca să coincidă cu aria reală (același lucru ca la jobs._apply_edited_roof_rectangles).

try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
except Exception:  # pragma: no cover
    Polygon = None  # type: ignore[misc, assignment]
    unary_union = None  # type: ignore[misc, assignment]


def room_polygon_is_insulated(room: dict[str, Any]) -> bool:
    if room.get("roomInsulated") is True:
        return True
    rt = str(room.get("roomType") or room.get("room_type") or "").strip()
    rn = str(room.get("roomName") or room.get("room_name") or "").strip()
    return rt == "Raum gedämmt" or rn == "Raum gedämmt"


def _slope_factor(angle_deg: float) -> float:
    a = max(0.0, min(80.0, float(angle_deg)))
    c = math.cos(math.radians(a))
    return (1.0 / c) if c > 1e-6 else 1.0


def _points_to_shapely_poly(pts: list[Any]) -> Any | None:
    if Polygon is None or not isinstance(pts, list) or len(pts) < 3:
        return None
    flat: list[tuple[float, float]] = []
    for p in pts:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        try:
            flat.append((float(p[0]), float(p[1])))
        except (TypeError, ValueError):
            continue
    if len(flat) < 3:
        return None
    if flat[0] != flat[-1]:
        flat = flat + [flat[0]]
    try:
        poly = Polygon(flat)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if poly.area > 0 else None
    except Exception:
        return None


def uninsulated_overlap_on_roof_sloped_m2(
    roof_points: list[Any],
    rooms: list[dict[str, Any]],
    meters_per_pixel: float,
    roof_angle_deg: float,
    *,
    editor_wall_area_scale: float = 1.0,
    stats: dict[str, Any] | None = None,
) -> float:
    """
    Horizontal overlap (plan) between roof polygon and union of uninsulated room polygons,
    converted to sloped roof m² via the same 1/cos(θ) factor as rectangle measurements.

    Intersection is computed in editor/review pixel space; ``editor_wall_area_scale`` should be
    sx*sy (walls_w/ed_w * walls_h/ed_h) so that area_px * mpp**2 * scale matches real m² on the plan.
    """
    if unary_union is None or Polygon is None:
        if stats is not None:
            stats.update({"error": "shapely_unavailable"})
        return 0.0
    mpp = float(meters_per_pixel)
    if mpp <= 0 or mpp > 10.0:
        mpp = 0.01

    scale = float(editor_wall_area_scale)
    if scale <= 0 or scale > 4.0:
        scale = 1.0

    roof_poly = _points_to_shapely_poly(roof_points)
    if roof_poly is None:
        if stats is not None:
            stats.update({"error": "invalid_roof_polygon"})
        return 0.0

    uninsulated: list[Any] = []
    n_rooms_total = 0
    n_rooms_insulated = 0
    for room in rooms:
        if not isinstance(room, dict):
            continue
        n_rooms_total += 1
        if room_polygon_is_insulated(room):
            n_rooms_insulated += 1
            continue
        pts = room.get("points")
        if not isinstance(pts, list):
            continue
        p = _points_to_shapely_poly(pts)
        if p is not None:
            uninsulated.append(p)

    if stats is not None:
        stats.update(
            {
                "rooms_total": n_rooms_total,
                "rooms_insulated": n_rooms_insulated,
                "uninsulated_polygons": len(uninsulated),
            }
        )

    if not uninsulated:
        if stats is not None:
            stats.update({"overlap_editor_px2": 0.0, "overlap_horizontal_m2": 0.0, "overlap_sloped_m2": 0.0})
        return 0.0

    try:
        union_u = unary_union(uninsulated)
        inter = roof_poly.intersection(union_u)
        area_px = float(getattr(inter, "area", 0.0) or 0.0)
    except Exception as e:
        if stats is not None:
            stats.update({"error": f"intersection_failed:{e!s}"})
        return 0.0

    horizontal_m2 = area_px * (mpp**2) * scale
    sf = _slope_factor(roof_angle_deg)
    sloped_m2 = max(0.0, horizontal_m2 * sf)
    if stats is not None:
        stats.update(
            {
                "overlap_editor_px2": round(area_px, 4),
                "mpp_walls": round(mpp, 6),
                "editor_wall_area_scale": round(scale, 6),
                "overlap_horizontal_m2": round(horizontal_m2, 4),
                "slope_factor_1_over_cos": round(sf, 6),
                "overlap_sloped_m2": round(sloped_m2, 4),
            }
        )
    return sloped_m2


def _detections_rooms_json_path(out_root: Path, plan_id: str) -> Path:
    """
    Preferă detections_edited.json (salvat la PATCH din editor) dacă există,
    altfel detections_review_data.json — același ordin ca validateUnifiedReviewBeforeApprove.
    """
    raster_dir = out_root / "scale" / plan_id / "cubicasa_steps" / "raster"
    edited = raster_dir / "detections_edited.json"
    if edited.exists():
        return edited
    return raster_dir / "detections_review_data.json"


def _editor_wall_scale_factors(out_root: Path, plan_id: str) -> tuple[float, float, float, str]:
    """
    sx = walls_w / editor_w, sy = walls_h / editor_h; use sx*sy on plan-view px² when mpp is for wall pixels.
    """
    rooms_path = _detections_rooms_json_path(out_root, plan_id)
    ed_w = ed_h = 0
    if rooms_path.exists():
        try:
            data = json.loads(rooms_path.read_text(encoding="utf-8"))
            ed_w = int(data.get("imageWidth") or 0)
            ed_h = int(data.get("imageHeight") or 0)
        except Exception:
            pass
    if ed_w <= 0 or ed_h <= 0:
        rev = (
            out_root
            / "scale"
            / plan_id
            / "cubicasa_steps"
            / "raster"
            / "detections_review_data.json"
        )
        if rev.exists():
            try:
                data = json.loads(rev.read_text(encoding="utf-8"))
                ed_w = int(data.get("imageWidth") or 0)
                ed_h = int(data.get("imageHeight") or 0)
            except Exception:
                pass
    wall_png = (
        out_root
        / "scale"
        / plan_id
        / "cubicasa_steps"
        / "raster_processing"
        / "walls_from_coords"
        / "01_walls_from_coords.png"
    )
    wh = ww = 0
    if wall_png.exists():
        try:
            import cv2

            img = cv2.imread(str(wall_png), cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size > 0:
                wh, ww = int(img.shape[0]), int(img.shape[1])
        except Exception:
            pass
    if ed_w <= 0 or ed_h <= 0 or ww <= 0 or wh <= 0:
        return (1.0, 1.0, 1.0, "fără editor/walls dims → scale 1 (posibil inexact)")
    sx = float(ww) / float(ed_w)
    sy = float(wh) / float(ed_h)
    prod = sx * sy
    msg = f"editor {ed_w}x{ed_h} → walls {ww}x{wh} (sx={sx:.4f} sy={sy:.4f} area_scale={prod:.6f})"
    return (sx, sy, prod, msg)


def load_rooms_for_plan(out_root: Path, plan_id: str) -> list[dict[str, Any]]:
    p = _detections_rooms_json_path(out_root, plan_id)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    raw = data.get("rooms")
    if not isinstance(raw, list):
        return []
    return [r for r in raw if isinstance(r, dict)]


def _roof_rect_points_at_index(
    edited: dict[str, Any],
    run_id: str,
    floor_idx_roof: int,
    rectangle_idx: int,
) -> list[Any] | None:
    from roof.roof_pricing import _iter_edited_rectangles_roof_order

    for fi, rects in _iter_edited_rectangles_roof_order(edited, run_id):
        if fi != int(floor_idx_roof):
            continue
        if not isinstance(rects, list) or not (0 <= int(rectangle_idx) < len(rects)):
            return None
        row = rects[int(rectangle_idx)]
        if not isinstance(row, dict):
            return None
        pts = row.get("points")
        if isinstance(pts, list) and len(pts) >= 3:
            return pts
        return None
    return None


def apply_roof_insulated_m2_to_rows(
    out_root: Path,
    run_id: str,
    rows: list[dict[str, Any]],
    meters_per_pixel_by_floor: dict[str, Any] | None,
) -> None:
    """
    Mutates each row: sets roof_area_insulated_m2 = max(0, roof_area_without_overhang_m2 - uninsulated_overlap_sloped).
    Falls back to ohne Überstand when geometry is missing or shapely fails.
    """
    from roof.roof_pricing import _load_mpp_by_floor_from_scale, _read_json

    edited = _read_json(out_root / "roof" / "roof_3d" / "roof_rectangles_edited.json")
    if not isinstance(edited, dict):
        edited = {}

    print(
        "       [Dach gedämmt] Intersecție în px editor (canvas review); mpp = m/px pe walls raster; "
        "corecție arie: ×(sx·sy) cu sx=walls_w/editor_w (ca la mască acoperiș). gedämmt = ohne − overlap×1/cos(θ).",
        flush=True,
    )

    rooms_cache: dict[str, list[dict[str, Any]]] = {}
    scale_cache: dict[str, tuple[float, float, float, str]] = {}
    # roof_metrics.json nu include mereu meters_per_pixel_by_floor; același fallback ca la
    # _compute_rectangle_inputs_for_gemini (altfel mpp=0.01 subestimează overlap-ul → gedämmt prea mare).
    mpp_map: dict[str, Any] = dict(meters_per_pixel_by_floor or {})
    for fk, mpp_v in _load_mpp_by_floor_from_scale(out_root, run_id).items():
        try:
            cur = float(mpp_map.get(str(fk)) or 0.0)
        except (TypeError, ValueError):
            cur = 0.0
        if cur <= 0:
            mpp_map[str(fk)] = mpp_v
    if mpp_map:
        print(f"       [Dach gedämmt] mpp pe floor_idx (după merge cu scale): { {k: round(float(v), 6) for k, v in sorted(mpp_map.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0)} }", flush=True)

    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            wo = float(r.get("roof_area_without_overhang_m2") or 0.0)
        except (TypeError, ValueError):
            wo = 0.0
        if wo <= 0:
            continue

        try:
            fi = int(r.get("floor_idx", 0))
            ri = int(r.get("rectangle_idx", 0))
        except (TypeError, ValueError):
            r["roof_area_insulated_m2"] = round(wo, 4)
            continue

        plan_id = str(r.get("plan_id") or "").strip()
        if not plan_id or not run_id:
            r["roof_area_insulated_m2"] = round(wo, 4)
            continue

        try:
            mpp = float(mpp_map.get(str(fi)) or 0.0)
        except (TypeError, ValueError):
            mpp = 0.0
        if mpp <= 0:
            mpp = 0.01

        ang = r.get("roof_angle_deg")
        try:
            ang_f = float(ang) if ang is not None else 0.0
        except (TypeError, ValueError):
            ang_f = 0.0

        pts = _roof_rect_points_at_index(edited, run_id, fi, ri)
        if pts is None:
            r["roof_area_insulated_m2"] = round(wo, 4)
            continue

        if plan_id not in rooms_cache:
            rooms_cache[plan_id] = load_rooms_for_plan(out_root, plan_id)
        rooms = rooms_cache[plan_id]

        if plan_id not in scale_cache:
            scale_cache[plan_id] = _editor_wall_scale_factors(out_root, plan_id)
        _sx, _sy, area_scale, scale_msg = scale_cache[plan_id]

        st: dict[str, Any] = {}
        overlap = uninsulated_overlap_on_roof_sloped_m2(
            pts,
            rooms,
            mpp,
            ang_f,
            editor_wall_area_scale=area_scale,
            stats=st,
        )
        insulated = max(0.0, wo - overlap)
        r["roof_area_insulated_m2"] = round(insulated, 4)

        print(
            f"       [Dach gedämmt] plan={plan_id} floor_idx={fi} rectangle_idx={ri} | "
            f"ohne_Überstand_m2={wo:.4f} | mpp_walls={mpp:.6f} | {scale_msg} | "
            f"camere: total={st.get('rooms_total', 0)} izolate={st.get('rooms_insulated', 0)} "
            f"poligoane_neizolate={st.get('uninsulated_polygons', 0)} | "
            f"overlap_px2_editor={st.get('overlap_editor_px2', 0)} → "
            f"oriz_m2={st.get('overlap_horizontal_m2', 0)} "
            f"(1/cos={st.get('slope_factor_1_over_cos', 1)}) → "
            f"scăzut_panta_m2={st.get('overlap_sloped_m2', overlap):.4f} | "
            f"gedämmt_m2={insulated:.4f}",
            flush=True,
        )

    total_g = 0.0
    for r in rows:
        if isinstance(r, dict) and "roof_area_insulated_m2" in r:
            try:
                total_g += float(r.get("roof_area_insulated_m2") or 0.0)
            except (TypeError, ValueError):
                pass
    print(
        f"       [Dach gedämmt] sumă roof_area_insulated_m2 (toate dreptunghiurile): {total_g:.4f} m²",
        flush=True,
    )
