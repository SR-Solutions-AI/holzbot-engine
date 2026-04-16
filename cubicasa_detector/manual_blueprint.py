# file: cubicasa_detector/manual_blueprint.py
"""Pregătire workspace fără Raster API / brute force: blueprint nativ + pipeline walls după editor."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from config.settings import PlanInfo


def _estimate_initial_walls_mask_1px(original_img: np.ndarray) -> np.ndarray:
    """
    Heuristic mask pentru pereți inițiali când nu avem camere editate.
    Ajută walls_from_coords să segmenteze spații (inclusiv planuri Bestand) în loc de listă goală.
    """
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    # Liniile de plan sunt în general închise; prag adaptiv + cleanup pentru zgomot/text.
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8,
    )
    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    # Subțiem ușor pentru a rămâne aproape de 1px.
    bw = cv2.erode(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    _, bw = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY)
    return bw


def _seed_response_rooms_from_roof_demolitions(raster_dir: Path, w: int, h: int) -> bool:
    """
    Aufstockung/Bestand: dacă editorul nu are rooms, folosim poligoanele Aufstandsfläche
    (`roofDemolitions`) ca seed pentru walls_from_coords.
    """
    edited_path = raster_dir / "detections_edited.json"
    response_path = raster_dir / "response.json"
    if not edited_path.exists() or not response_path.exists():
        return False
    try:
        edited = json.loads(edited_path.read_text(encoding="utf-8"))
        if not isinstance(edited, dict):
            return False
        rooms_raw = edited.get("rooms")
        if isinstance(rooms_raw, list) and len(rooms_raw) > 0:
            return False
        demolitions = edited.get("roofDemolitions")
        if not isinstance(demolitions, list) or len(demolitions) == 0:
            return False
        response = json.loads(response_path.read_text(encoding="utf-8"))
        data = response.get("data", response) if isinstance(response, dict) else {}
        if not isinstance(data, dict):
            data = {}
        seeded_rooms: list[list[dict[str, int]]] = []
        for d in demolitions:
            if not isinstance(d, dict):
                continue
            pts = d.get("points")
            if not isinstance(pts, list) or len(pts) < 3:
                continue
            norm_pts: list[dict[str, int]] = []
            for p in pts:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                try:
                    x = max(0, min(w - 1, int(round(float(p[0])))))
                    y = max(0, min(h - 1, int(round(float(p[1])))))
                except Exception:
                    continue
                norm_pts.append({"x": x, "y": y})
            if len(norm_pts) >= 3:
                seeded_rooms.append(norm_pts)
        if not seeded_rooms:
            return False
        data["rooms"] = seeded_rooms
        if "doors" not in data or not isinstance(data.get("doors"), list):
            data["doors"] = []
        wrapped = {"data": data}
        response_path.write_text(json.dumps(wrapped, indent=2, ensure_ascii=False), encoding="utf-8")
        (raster_dir / "use_visible_m2_sum_mode.flag").write_text("1", encoding="utf-8")
        print(
            f"      🔸 [manual] Seeded {len(seeded_rooms)} rooms din roofDemolitions (Aufstandsfläche) pentru walls_from_coords",
            flush=True,
        )
        return True
    except Exception as e:
        print(f"      ⚠️ [manual] Nu pot seed-ui rooms din roofDemolitions: {e}", flush=True)
        return False


def _should_force_visible_m2_sum_mode(rooms: list, w: int, h: int) -> bool:
    """
    Heuristic pentru planuri unde un crop/poligon poate acoperi mai multe camere tipărite:
    forțăm modul de sumare m² vizibile, în loc de etichetare single-room.
    """
    if not isinstance(rooms, list) or not rooms:
        return False
    if w <= 0 or h <= 0:
        return False

    img_area = float(w * h)
    if img_area <= 0:
        return False

    parsed_polys: list[np.ndarray] = []
    for room in rooms:
        if not isinstance(room, list) or len(room) < 3:
            continue
        poly_pts: list[list[int]] = []
        for p in room:
            if not isinstance(p, dict):
                continue
            try:
                x = max(0, min(w - 1, int(round(float(p.get("x", 0))))))
                y = max(0, min(h - 1, int(round(float(p.get("y", 0))))))
            except Exception:
                continue
            poly_pts.append([x, y])
        if len(poly_pts) >= 3:
            parsed_polys.append(np.array(poly_pts, dtype=np.int32))

    if not parsed_polys:
        return False

    mask_union = np.zeros((h, w), dtype=np.uint8)
    largest_ratio = 0.0
    for poly in parsed_polys:
        area_px = float(abs(cv2.contourArea(poly)))
        if area_px > 0:
            largest_ratio = max(largest_ratio, area_px / img_area)
        cv2.fillPoly(mask_union, [poly], 255)
    union_ratio = float(np.count_nonzero(mask_union)) / img_area

    # Trigger conservator:
    # - puține poligoane, dar foarte mari -> foarte probabil "multi-room per crop"
    # - un singur poligon masiv -> clar sumare vizibilă
    if len(parsed_polys) <= 3 and (largest_ratio >= 0.15 or union_ratio >= 0.18):
        return True
    if len(parsed_polys) == 1 and largest_ratio >= 0.10:
        return True
    return False


def ensure_roof_3d_floor_manifest(run_id: str, plan_ids_in_order: list[str]) -> None:
    """Înainte de editor: mapping etaj → plan_id pentru roof PATCH/GET (fără clean_workflow)."""
    from config.settings import OUTPUT_ROOT

    roof_3d = OUTPUT_ROOT / run_id / "roof" / "roof_3d"
    roof_3d.mkdir(parents=True, exist_ok=True)
    mapping = {str(i): pid for i, pid in enumerate(plan_ids_in_order)}
    (roof_3d / "roof_floor_plan_ids.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def prepare_manual_blueprint_workspace(plan: PlanInfo) -> None:
    """
    Copiază crop-ul blueprint în raster/ și cubicasa_steps fără resize/stretch.
    Scrie response.json gol, raster_request_info identic cu dimensiunea imaginii.
    """
    src = Path(plan.plan_image)
    if not src.exists():
        raise FileNotFoundError(f"Blueprint lipsește: {src}")

    stage = Path(plan.stage_work_dir)
    stage.mkdir(parents=True, exist_ok=True)
    cubicasa = stage / "cubicasa_steps"
    cubicasa.mkdir(parents=True, exist_ok=True)
    raster_dir = cubicasa / "raster"
    raster_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Nu pot citi imaginea: {src}")
    h, w = img.shape[:2]

    # Aceeași imagine ca bază editor (REVIEW_BASE_IMAGE_NAMES)
    for name in ("input_resized_no_filter.png", "input_resized.jpg"):
        cv2.imwrite(str(raster_dir / name), img)
    shutil.copy2(src, cubicasa / "00_original.png")
    shutil.copy2(src, stage / "plan.jpg")

    (raster_dir / "manual_blueprint_mode.flag").write_text("1", encoding="utf-8")

    ri = {
        "request_w": w,
        "request_h": h,
        "mask_w": w,
        "mask_h": h,
        "scale_factor": 1.0,
    }
    (raster_dir / "raster_request_info.json").write_text(
        json.dumps(ri, indent=2), encoding="utf-8"
    )

    response = {"data": {"rooms": [], "doors": []}}
    (raster_dir / "response.json").write_text(
        json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Evităm doors_types goale la început (editor le completează la nevoie)
    (raster_dir / "doors_types.json").write_text("[]", encoding="utf-8")


def write_skipped_no_selections_walls_stub(plan: PlanInfo) -> bool:
    """
    Fără poligoane cameră în editor și fără seed din roofDemolitions: nu rulăm Gemini / flood-fill.
    Scrie artefacte minime + room_scales marcat cu pipeline_skipped (valid pentru _check_raster_complete).
    """
    stage = Path(plan.stage_work_dir)
    cubicasa = stage / "cubicasa_steps"
    wfc = cubicasa / "raster_processing" / "walls_from_coords"
    wfc.mkdir(parents=True, exist_ok=True)
    raster_dir = cubicasa / "raster"
    raster_dir.mkdir(parents=True, exist_ok=True)
    orig_path = cubicasa / "00_original.png"
    if not orig_path.exists():
        return False
    img = cv2.imread(str(orig_path), cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    wall = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(wall, (0, 0), (w - 1, h - 1), 255, 1)
    cv2.imwrite(str(wfc / "01_walls_from_coords.png"), wall)

    rooms_png = raster_dir / "rooms.png"
    blank = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(rooms_png), blank)

    rs = {
        "pipeline_skipped": "no_editor_room_selections",
        "weighted_average_m_px": None,
        "m_px": None,
        "total_area_m2": 0.0,
        "total_area_px": 0,
        "room_scales": {},
    }
    (wfc / "room_scales.json").write_text(json.dumps(rs, indent=2, ensure_ascii=False), encoding="utf-8")

    scale_json = stage / "scale_result.json"
    scale_json.write_text(
        json.dumps(
            {
                "meters_per_pixel": None,
                "method": "no_editor_selections_skipped",
                "confidence": "n/a",
                "rooms_analyzed": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return True


def run_walls_pipeline_after_manual_editor(plan: PlanInfo, roof_only_offer: bool = False) -> bool:
    """
    După apply_detections_edited: generează walls_from_coords + room_scales (Gemini pe camere).
    Folosește mască 1px inițială nulă + camere din response (mod manual).
    """
    from cubicasa_detector.raster_processing import generate_walls_from_room_coordinates

    stage = Path(plan.stage_work_dir)
    cubicasa = stage / "cubicasa_steps"
    raster_dir = cubicasa / "raster"
    orig_path = cubicasa / "00_original.png"
    if not orig_path.exists():
        print(f"      ⚠️ [manual] Lipsă {orig_path}", flush=True)
        return False
    original_img = cv2.imread(str(orig_path), cv2.IMREAD_COLOR)
    if original_img is None:
        return False
    h, w = original_img.shape[:2]
    initial = _estimate_initial_walls_mask_1px(original_img)
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("      ⚠️ [manual] GEMINI_API_KEY lipsește", flush=True)
        return False
    use_roof_visible_m2_scale = roof_only_offer
    response_path = raster_dir / "response.json"
    rooms: list = []
    if response_path.exists():
        try:
            resp = json.loads(response_path.read_text(encoding="utf-8"))
            data = resp.get("data", resp) if isinstance(resp, dict) else {}
            raw = data.get("rooms") if isinstance(data, dict) else None
            if isinstance(raw, list):
                rooms = raw
        except Exception:
            rooms = []

    if len(rooms) >= 1:
        use_roof_visible_m2_scale = False
        if _should_force_visible_m2_sum_mode(rooms, w, h):
            use_roof_visible_m2_scale = True
            print(
                "      🧮 [manual] Rooms mari/aggregate detectate -> forțez Gemini visible_m2_sum per crop.",
                flush=True,
            )
    else:
        seeded_from_demolition = _seed_response_rooms_from_roof_demolitions(raster_dir, w, h)
        if seeded_from_demolition:
            use_roof_visible_m2_scale = True
            try:
                resp = json.loads(response_path.read_text(encoding="utf-8"))
                data = resp.get("data", resp) if isinstance(resp, dict) else {}
                raw = data.get("rooms") if isinstance(data, dict) else None
                rooms = raw if isinstance(raw, list) else []
            except Exception:
                rooms = []

    if len(rooms) == 0:
        print(
            "      ⏭️ [manual] Skip walls_from_coords + Gemini: zero camere (fără poligoane editor / Aufstandsfläche).",
            flush=True,
        )
        return write_skipped_no_selections_walls_stub(plan)

    try:
        generate_walls_from_room_coordinates(
            original_img,
            None,
            raster_dir,
            str(cubicasa),
            gemini_key,
            initial_walls_mask_1px=initial,
            progress_callback=None,
            notify_scale_walls_3d_ui=not roof_only_offer,
            roof_only_offer=use_roof_visible_m2_scale,
        )
        return True
    except Exception as e:
        print(f"      ❌ [manual] walls_from_coords: {e}", flush=True)
        import traceback

        traceback.print_exc()
        return False


def seed_roof_only_rooms_from_roof_polygons(run_id: str, plans: list["PlanInfo"]) -> dict[str, int]:
    """
    roof_only_offer (legacy): construiește 09_interior.png din poligoanele editorului de acoperiș
    și injectează aceleași poligoane în raster/response.json (rooms), per plan.
    Dacă response.json are deja camere (din Grundriss / detections_edited), planul e sărit –
    același flux ca la casă completă, cu Räume gedämmt/ungedämmt.
    Returnează numărul de poligoane aplicate pentru fiecare plan_id.
    """
    from config.settings import OUTPUT_ROOT

    roof_3d = OUTPUT_ROOT / run_id / "roof" / "roof_3d"
    roof_polys_path = roof_3d / "roof_rectangles_edited_walls.json"
    if not roof_polys_path.exists():
        roof_polys_path = roof_3d / "roof_rectangles_edited.json"
    manifest_path = roof_3d / "roof_floor_plan_ids.json"

    if not roof_polys_path.exists() or not manifest_path.exists():
        return {}

    try:
        roof_data = json.loads(roof_polys_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    plan_by_id = {p.plan_id: p for p in plans}
    polygons_by_plan: dict[str, list[list[list[int]]]] = {}

    for floor_key, entries in (roof_data or {}).items():
        if not isinstance(floor_key, str) or floor_key.startswith("_"):
            continue
        if not isinstance(entries, list):
            continue
        plan_id = manifest.get(str(floor_key))
        if not plan_id or plan_id not in plan_by_id:
            continue
        polys = polygons_by_plan.setdefault(plan_id, [])
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            pts = entry.get("points")
            if not isinstance(pts, list):
                continue
            clean: list[list[int]] = []
            for pxy in pts:
                if not isinstance(pxy, (list, tuple)) or len(pxy) < 2:
                    continue
                try:
                    x = int(round(float(pxy[0])))
                    y = int(round(float(pxy[1])))
                except Exception:
                    continue
                clean.append([x, y])
            if len(clean) >= 3:
                polys.append(clean)

    applied_counts: dict[str, int] = {}
    for plan in plans:
        stage = Path(plan.stage_work_dir)
        cubicasa = stage / "cubicasa_steps"
        raster_dir = cubicasa / "raster"
        response_path = raster_dir / "response.json"
        skip_seed = False
        if response_path.exists():
            try:
                current = json.loads(response_path.read_text(encoding="utf-8"))
                data = current.get("data", current) if isinstance(current, dict) else {}
                rooms_existing = data.get("rooms") if isinstance(data, dict) else None
                if isinstance(rooms_existing, list) and len(rooms_existing) >= 1:
                    skip_seed = True
            except Exception:
                pass
        if skip_seed:
            continue

        output_dir = cubicasa / "raster_processing" / "walls_from_coords"
        output_dir.mkdir(parents=True, exist_ok=True)
        orig_path = cubicasa / "00_original.png"
        img = cv2.imread(str(orig_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]

        raw_polys = polygons_by_plan.get(plan.plan_id, [])
        clipped_polys = []
        for pts in raw_polys:
            arr = np.array(
                [
                    [
                        max(0, min(w - 1, int(p[0]))),
                        max(0, min(h - 1, int(p[1]))),
                    ]
                    for p in pts
                ],
                dtype=np.int32,
            )
            if arr.shape[0] >= 3:
                clipped_polys.append(arr)

        interior = np.zeros((h, w, 3), dtype=np.uint8)
        for poly in clipped_polys:
            cv2.fillPoly(interior, [poly], (0, 165, 255))
        cv2.imwrite(str(output_dir / "09_interior.png"), interior)

        response_path = raster_dir / "response.json"
        data = {"rooms": [], "doors": []}
        if response_path.exists():
            try:
                current = json.loads(response_path.read_text(encoding="utf-8"))
                data = current.get("data", current) if isinstance(current, dict) else data
            except Exception:
                pass
        rooms = [
            [{"x": int(p[0]), "y": int(p[1])} for p in poly.tolist()]
            for poly in clipped_polys
        ]
        data["rooms"] = rooms
        wrapped = {"data": data}
        response_path.write_text(json.dumps(wrapped, indent=2, ensure_ascii=False), encoding="utf-8")

        applied_counts[plan.plan_id] = len(clipped_polys)

    return applied_counts


def apply_roof_only_synthetic_walls_and_scale(plan: PlanInfo, meters_per_pixel: float = 0.01) -> bool:
    """
    roof_only_offer: fără camere / fără Gemini walls — mască perete sintetică + room_scales minim
    ca scale și roof să aibă mpp.
    """
    stage = Path(plan.stage_work_dir)
    cubicasa = stage / "cubicasa_steps"
    wfc = cubicasa / "raster_processing" / "walls_from_coords"
    wfc.mkdir(parents=True, exist_ok=True)
    raster_dir = cubicasa / "raster"
    orig_path = cubicasa / "00_original.png"
    if not orig_path.exists():
        return False
    img = cv2.imread(str(orig_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    h, w = img.shape[:2]
    wall = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(wall, (0, 0), (w - 1, h - 1), 255, 1)
    cv2.imwrite(str(wfc / "01_walls_from_coords.png"), wall)

    rooms_png = raster_dir / "rooms.png"
    blank = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(rooms_png), blank)

    mpp = float(meters_per_pixel) if meters_per_pixel > 0 else 0.01
    area_px = float(h * w)
    area_m2 = area_px * (mpp**2)
    rs = {
        "weighted_average_m_px": mpp,
        "m_px": mpp,
        "total_area_m2": max(1.0, area_m2),
        "total_area_px": int(area_px),
        "room_scales": {
            "0": {
                "room_number": 0,
                "room_type": "Dach",
                "room_name": "Dach",
                "area_m2": max(1.0, area_m2),
                "m_px": mpp,
            }
        },
    }
    (wfc / "room_scales.json").write_text(json.dumps(rs, indent=2, ensure_ascii=False), encoding="utf-8")

    scale_json = stage / "scale_result.json"
    scale_json.write_text(
        json.dumps(
            {
                "meters_per_pixel": mpp,
                "method": "roof_only_synthetic",
                "confidence": "low",
                "rooms_analyzed": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return True
