# pdf_generator/roof_measurements_pdf.py
"""PDF cu măsurători detaliate – Dach + Bauteilmaße (DE), grupat pe Stockwerk cu plan."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from PIL import Image as PILImage, ImageEnhance, ImageOps
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image as RLImage,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen.canvas import Canvas

from config.settings import OUTPUT_ROOT, RUNNER_ROOT, JOBS_ROOT, RUNS_ROOT, load_plan_infos, PlansListError, PlanInfo
from config.frontend_loader import load_frontend_data_for_run
from pdf_generator.utils import resolve_editor_blueprint_for_pdf
from pdf_generator.offer_scope import get_offer_inclusions, normalize_nivel_oferta
from area.config import (
    STANDARD_DOOR_HEIGHT_M,
    STANDARD_WINDOW_HEIGHT_M,
    STANDARD_WALL_HEIGHT_M,
    WALL_HEIGHT_EXTRA_STRUCTURE_AND_EXT_FINISH_M,
)
from area.calculator import _infer_window_height_from_width


def _polygon_area_px2_from_points(points: object) -> float:
    """Shoelace area in editor pixel space (same as orchestrator)."""
    if not isinstance(points, list) or len(points) < 3:
        return 0.0
    pts: list[tuple[float, float]] = []
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                pts.append((float(p[0]), float(p[1])))
            except (TypeError, ValueError):
                continue
    if len(pts) < 3:
        return 0.0
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += (x1 * y2) - (x2 * y1)
    return abs(s) * 0.5


def _bbox_area_px2(bbox: object) -> float:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return 0.0
    try:
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
    except (TypeError, ValueError):
        return 0.0


def _stair_opening_edge_lengths_m_from_bbox_and_area(bbox: object, area_m2: float) -> tuple[float | None, float | None]:
    """Kürzere und längere Kantenlänge (m) aus Pixel-BBox und Fläche (m²)."""
    if area_m2 <= 0 or not isinstance(bbox, list) or len(bbox) != 4:
        return None, None
    try:
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        w_px = max(0.0, x2 - x1)
        h_px = max(0.0, y2 - y1)
        apx = w_px * h_px
        if apx <= 0:
            return None, None
        mpp = (float(area_m2) / apx) ** 0.5
        w_m = w_px * mpp
        h_m = h_px * mpp
        return (min(w_m, h_m), max(w_m, h_m))
    except (TypeError, ValueError):
        return None, None


def _scale_mpp_for_plan_dir(plan_dir: Path) -> float | None:
    sr = plan_dir / "scale_result.json"
    if sr.exists():
        try:
            data = json.loads(sr.read_text(encoding="utf-8"))
            mpp = data.get("meters_per_pixel")
            if isinstance(mpp, (int, float)) and float(mpp) > 0:
                return float(mpp)
        except Exception:
            pass
    rs = plan_dir / "cubicasa_steps" / "raster_processing" / "walls_from_coords" / "room_scales.json"
    if rs.exists():
        try:
            data = json.loads(rs.read_text(encoding="utf-8"))
            mpp = data.get("m_px") or data.get("weighted_average_m_px")
            if isinstance(mpp, (int, float)) and float(mpp) > 0:
                return float(mpp)
        except Exception:
            pass
    return None


def _manifest_ordered_scale_plan_dirs(job_root: Path | None, out_root: Path) -> list[Path]:
    """Same plan order as editor/orchestrator (manifest rasterDirs), not sorted plan_01/plan_02 names."""
    roots: list[Path] = []
    if job_root:
        mp = job_root / "detections_review_manifest.json"
        if mp.exists():
            try:
                manifest = json.loads(mp.read_text(encoding="utf-8"))
                raster_dirs = manifest.get("rasterDirs") if isinstance(manifest, dict) else []
                if isinstance(raster_dirs, list):
                    for rd in raster_dirs:
                        rp = Path(str(rd))
                        if rp.name == "raster":
                            roots.append(rp.parent.parent)
            except Exception:
                roots = []
    if not roots:
        scale_root = out_root / "scale"
        if scale_root.is_dir():
            roots = sorted(
                [p for p in scale_root.iterdir() if p.is_dir() and p.name.startswith("plan_")],
                key=lambda p: p.name,
            )
    return roots


def _aufstockung_new_floor_mpp(out_root: Path, floor_kinds: list[str], job_root: Path | None = None) -> float | None:
    """m/px from first plan classified as `new` (Aufstockung rule, same as pricing)."""
    plan_dirs = _manifest_ordered_scale_plan_dirs(job_root, out_root)
    if not plan_dirs:
        return None
    for idx, pd in enumerate(plan_dirs):
        kind = floor_kinds[idx] if idx < len(floor_kinds) else "existing"
        if kind != "new":
            continue
        mpp = _scale_mpp_for_plan_dir(pd)
        if mpp and mpp > 0:
            return mpp
    return None


def _load_aufstockung_floor_kinds(run_id: str, fd_dict: dict[str, Any], job_root: Path | None) -> list[str]:
    """Best-effort floor kinds for Aufstockung in raw plan index order."""
    kinds_raw: Any = []
    extras: dict[str, Any] = {}
    if job_root:
        extras_path = job_root / "detections_review_extras.json"
        if extras_path.exists():
            try:
                extras = json.loads(extras_path.read_text(encoding="utf-8"))
            except Exception:
                extras = {}
    kinds_raw = (
        (extras.get("floorKinds") if isinstance(extras, dict) else None)
        or (fd_dict.get("floorKinds") if isinstance(fd_dict, dict) else None)
        or (fd_dict.get("aufstockungFloorKinds") if isinstance(fd_dict, dict) else None)
        or ((fd_dict.get("structuraCladirii") or {}).get("aufstockungFloorKinds") if isinstance(fd_dict, dict) else None)
        or []
    )
    if not isinstance(kinds_raw, list):
        return []
    return ["new" if str(k).strip().lower() == "new" else "existing" for k in kinds_raw]


def _lista_etaje_from_frontend(fd: dict[str, Any]) -> list[Any]:
    sc = fd.get("structuraCladirii")
    if isinstance(sc, dict):
        le = sc.get("listaEtaje")
        if isinstance(le, list):
            return le
    drafts = fd.get("drafts")
    if isinstance(drafts, dict):
        dsc = drafts.get("structuraCladirii")
        if isinstance(dsc, dict):
            le = dsc.get("listaEtaje")
            if isinstance(le, list):
                return le
    le = fd.get("listaEtaje")
    return le if isinstance(le, list) else []


def _gebaudestruktur_last_floor_is_mansard(lista_etaje: list[Any]) -> bool:
    """mansarda_ohne / mansarda_mit (Gebäudestruktur) → ultimul etaj e Dachgeschoss, nu Obergeschoss."""
    if not lista_etaje:
        return False
    last = lista_etaje[-1]
    return isinstance(last, str) and last.startswith("mansarda")


def _load_floor_plan_order_and_labels_from_manifest(
    run_id: str, job_root: Path | None
) -> tuple[list[int] | None, list[str] | None]:
    """
    floorPlanOrder: order_from_bottom[pos] = original plan index (0 = lowest floor).
    floorLabels: one label per original plan index.
    """
    candidates: list[Path] = []
    if job_root:
        candidates.append(job_root / "detections_review_manifest.json")
    candidates.append(JOBS_ROOT / run_id / "detections_review_manifest.json")
    for mp in candidates:
        if not mp.exists():
            continue
        try:
            data = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        order_raw = data.get("floorPlanOrder")
        labels_raw = data.get("floorLabels")
        labels: list[str] | None = None
        if isinstance(labels_raw, list):
            labels = [str(x) if x is not None else "" for x in labels_raw]
        if not isinstance(order_raw, list) or not order_raw:
            # No order in manifest for this candidate: keep scanning candidates,
            # then fall back to floor_order.json if needed.
            continue
        try:
            order = [int(x) for x in order_raw]
        except (TypeError, ValueError):
            continue
        return order, labels
    # Fallback for older manifests that do not include floorPlanOrder:
    # use engine floor_order.json (same source used by the final offer PDF).
    run_roots = [OUTPUT_ROOT / run_id, RUNNER_ROOT / "output" / run_id, RUNS_ROOT / run_id]
    for run_root in run_roots:
        fo_path = run_root / "floor_order.json"
        if not fo_path.exists():
            continue
        try:
            fo_data = json.loads(fo_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        raw_order = fo_data.get("order_from_bottom")
        if not isinstance(raw_order, list) or not raw_order:
            continue
        try:
            order = [int(x) for x in raw_order]
        except (TypeError, ValueError):
            continue
        if len(set(order)) != len(order):
            continue
        return order, None
    return None, None


def _opening_area_m2_from_row(o: dict) -> float:
    """Pentru measurements_plan: obiectele au adesea doar width_m, nu area_m2."""
    if not isinstance(o, dict):
        return 0.0
    am = o.get("area_m2")
    if am is not None and float(am or 0) > 0:
        return float(am)
    w = float(o.get("width_m", 0) or 0)
    if w <= 0:
        return 0.0
    t = str(o.get("type", "")).lower()
    eh = o.get("height_m")
    if eh is not None and float(eh or 0) > 0:
        return w * float(eh)
    if "window" in t:
        h = _infer_window_height_from_width(w, t)
        return w * float(h)
    if "garage" in t:
        return w * 2.1
    if "door" in t:
        return w * STANDARD_DOOR_HEIGHT_M
    return w * STANDARD_WINDOW_HEIGHT_M


def _load_floor_data_from_pricing(out_root: Path) -> dict[str, dict[str, Any]]:
    """Încarcă măsurători per plan: finisaje din pricing_raw; structură/podea/tavan preferă measurements_plan.json."""
    pricing_dir = out_root / "pricing"
    out: dict[str, dict[str, Any]] = {}
    if not pricing_dir.exists():
        return out
    for plan_dir in sorted(p for p in pricing_dir.iterdir() if p.is_dir() and p.name.startswith("plan_")):
        pr_path = plan_dir / "pricing_raw.json"
        if not pr_path.exists():
            continue
        try:
            with open(pr_path, encoding="utf-8") as f:
                pr = json.load(f)
        except Exception:
            continue
        bd = pr.get("breakdown", {})
        sw = bd.get("structure_walls", {}) or {}
        items_sw = sw.get("detailed_items", []) or sw.get("items", [])
        int_net = ext_net = 0.0
        for it in items_sw:
            cat = str(it.get("category", "")).lower()
            a = float(it.get("area_m2", 0) or 0)
            if cat == "walls_structure_int":
                int_net += a
            elif cat == "walls_structure_ext":
                ext_net += a
        fin = bd.get("finishes", {}) or {}
        items_fin = fin.get("detailed_items", []) or fin.get("items", [])
        fin_int_inner = fin_int_outer = fin_ext = 0.0
        for it in items_fin:
            cat = str(it.get("category", "")).lower()
            a = float(it.get("area_m2", 0) or 0)
            if cat in {"finish_interior", "finish_interior_inner"}:
                fin_int_inner += a
            elif cat == "finish_interior_outer":
                fin_int_outer += a
            elif cat == "finish_exterior":
                fin_ext += a
        fc = bd.get("floors_ceilings", {}) or {}
        items_fc = fc.get("detailed_items", []) or fc.get("items", [])
        floor_area = ceiling_area = 0.0
        for it in items_fc:
            cat = str(it.get("category", "")).lower()
            a = float(it.get("area_m2", 0) or 0)
            if cat == "floor_structure":
                floor_area += a
            elif cat == "ceiling_structure":
                ceiling_area += a
        if floor_area == 0 and items_fc:
            floor_area = float(pr.get("total_area_m2", 0) or 0)
        if ceiling_area == 0:
            ceiling_area = floor_area
        op = bd.get("openings", {}) or {}
        items_op = op.get("items", []) or op.get("detailed_items", [])
        num_doors = num_windows = 0
        openings_area = 0.0
        for it in items_op:
            t = str(it.get("type", "")).lower()
            a = float(it.get("area_m2", 0) or 0)
            if a <= 0:
                a = _opening_area_m2_from_row(it)
            openings_area += a
            if "door" in t:
                num_doors += 1
            elif "window" in t:
                num_windows += 1

        mp_path = plan_dir / "measurements_plan.json"
        if mp_path.exists():
            try:
                mp = json.loads(mp_path.read_text(encoding="utf-8"))
                areas = mp.get("areas") or {}
                walls = areas.get("walls") or {}
                wi = walls.get("interior") or {}
                we = walls.get("exterior") or {}
                s_int = float(wi.get("net_area_m2_structure") or wi.get("gross_area_m2_structure") or 0)
                s_ext = float(we.get("net_area_m2_structure") or we.get("gross_area_m2_structure") or 0)
                if s_int > 0:
                    int_net = s_int
                if s_ext > 0:
                    ext_net = s_ext
                surf = areas.get("surfaces") or {}
                f_m = float(surf.get("floor_m2") or 0)
                c_m = float(surf.get("ceiling_m2") or 0)
                if f_m > 0:
                    floor_area = f_m
                if c_m > 0:
                    ceiling_area = c_m
                elif f_m > 0:
                    ceiling_area = f_m
                op_mp = mp.get("openings")
                if isinstance(op_mp, list) and op_mp:
                    nd = nw = 0
                    oa = 0.0
                    for o in op_mp:
                        if not isinstance(o, dict):
                            continue
                        t = str(o.get("type", "")).lower()
                        oa += _opening_area_m2_from_row(o)
                        if "door" in t:
                            nd += 1
                        elif "window" in t:
                            nw += 1
                    num_doors, num_windows, openings_area = nd, nw, oa
                # Agregate din calculator (afișare corectă când lista are width_m fără area_m2 sau lipsește din pricing)
                wi_o = float(wi.get("openings_area_m2", 0) or 0)
                we_o = float(we.get("openings_area_m2", 0) or 0)
                wall_openings = wi_o + we_o
                if wall_openings > openings_area:
                    openings_area = wall_openings
            except Exception:
                pass

        out[plan_dir.name] = {
            "plan_id": plan_dir.name,
            "structure_int_net_m2": round(int_net, 2),
            "structure_ext_net_m2": round(ext_net, 2),
            "finish_int_inner_net_m2": round(fin_int_inner, 2),
            "finish_int_outer_net_m2": round(fin_int_outer, 2),
            "finish_int_net_m2": round(fin_int_inner + fin_int_outer, 2),
            "finish_ext_net_m2": round(fin_ext, 2),
            "floor_area_m2": round(floor_area, 2),
            "ceiling_area_m2": round(ceiling_area, 2),
            "num_doors": int(num_doors),
            "num_windows": int(num_windows),
            "openings_area_m2": round(openings_area, 2),
        }
    return out


def _read_roof_floor_plan_ids(out_root: Path, run_id: str) -> dict[int, str]:
    for run_root in (out_root, RUNNER_ROOT / "output" / run_id, RUNS_ROOT / run_id):
        rfp = run_root / "roof" / "roof_3d" / "roof_floor_plan_ids.json"
        if rfp.exists():
            try:
                raw = json.loads(rfp.read_text(encoding="utf-8"))
                return {int(k): str(v) for k, v in raw.items()}
            except Exception:
                pass
    return {}


def _roof_floor_mapping_for_pdf(run_id: str, out_root: Path) -> tuple[dict[int, str], dict[str, int]]:
    """
    floor_idx din by_rectangle = index în plans_roof (de sus în jos), ca la _apply_edited_roof_rectangles.
    Nu ne bazăm doar pe fișierul JSON (poate fi din alt run_root sau desincronizat).
    """
    try:
        from roof.jobs import resolve_plans_roof_order

        plans_roof, _, _ = resolve_plans_roof_order(run_id)
        idx_to_plan = {i: p.plan_id for i, p in enumerate(plans_roof)}
        plan_to_idx = {p.plan_id: i for i, p in enumerate(plans_roof)}
        if idx_to_plan:
            return idx_to_plan, plan_to_idx
    except Exception:
        pass
    disk = _read_roof_floor_plan_ids(out_root, run_id)
    plan_to_idx: dict[str, int] = {}
    for i, pid in disk.items():
        plan_to_idx[str(pid)] = int(i)
    return disk, plan_to_idx


def _read_manifest_plan_ids(run_id: str) -> dict[int, str]:
    """Ordinea tab-urilor din review UI (manifest): 0 = primul blueprint afișat în editor."""
    manifest_path = JOBS_ROOT / run_id / "detections_review_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    plan_ids = raw.get("rasterPlanIds") if isinstance(raw, dict) else None
    if not isinstance(plan_ids, list):
        return {}
    out: dict[int, str] = {}
    for idx, value in enumerate(plan_ids):
        pid = str(value or "").strip()
        if pid:
            out[idx] = pid
    return out


def _load_roof_windows_by_plan(out_root: Path, run_id: str) -> dict[str, dict[str, float]]:
    """
    Returnează agregare per plan pentru Dachfenster din roof_windows_edited.json:
    - count: nr. ferestre
    - area_m2: Σ(width_m * height_m)
    """
    out: dict[str, dict[str, float]] = {}
    p = out_root / "roof" / "roof_3d" / "roof_windows_edited.json"
    if not p.exists():
        return out
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return out
    if not isinstance(raw, dict):
        return out
    key_scheme = str(raw.get("_floor_key_scheme") or "").strip().lower()
    if key_scheme == "manifest":
        idx_to_plan = _read_manifest_plan_ids(run_id)
    else:
        idx_to_plan, _ = _roof_floor_mapping_for_pdf(run_id, out_root)
    for floor_key, windows in raw.items():
        if str(floor_key).startswith("_"):
            continue
        try:
            floor_idx = int(floor_key)
        except (TypeError, ValueError):
            continue
        plan_id = idx_to_plan.get(floor_idx)
        if not plan_id:
            continue
        if not isinstance(windows, list):
            continue
        count = 0
        area = 0.0
        for w in windows:
            if not isinstance(w, dict):
                continue
            count += 1
            wm = float(w.get("width_m", 0) or 0)
            hm = float(w.get("height_m", 0) or 0)
            if wm > 0 and hm > 0:
                area += wm * hm
        if count > 0 or area > 0:
            out[plan_id] = {"count": float(count), "area_m2": round(area, 4)}
    return out


def _append_plan_image(
    story: list,
    plan: PlanInfo,
    out_root: Path,
    job_root: Path | None,
    heading_style: ParagraphStyle,
) -> None:
    plan_img_path = resolve_editor_blueprint_for_pdf(plan.plan_image, plan.plan_id, out_root, job_root)
    if not plan_img_path or not plan_img_path.exists():
        return
    try:
        im = PILImage.open(plan_img_path).convert("L")
        im = ImageEnhance.Brightness(im).enhance(0.9)
        im = ImageOps.autocontrast(im)
        width, height = im.size
        aspect = width / height
        target_width = A4[0] - 30 * mm
        if aspect < 1:
            target_width = target_width * 0.65
        img_byte_arr = io.BytesIO()
        im.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        rl_img = RLImage(img_byte_arr)
        rl_img._restrictSize(target_width, 75 * mm)
        rl_img.hAlign = "CENTER"
        story.append(Spacer(1, 2 * mm))
        story.append(rl_img)
        story.append(Spacer(1, 4 * mm))
    except Exception as e:
        story.append(Paragraph(f"<i>(Grundriss konnte nicht geladen werden: {e})</i>", heading_style))


def _roof_rect_table_rows(rec: dict[str, Any]) -> list[list[str]]:
    roof_type_label = {
        "0_w": "Flachdach",
        "1_w": "Pultdach",
        "2_w": "Satteldach",
        "4_w": "Walmdach",
        "4.5_w": "Krüppelwalmdach",
    }
    rows = [["Position", "Wert", "Einheit"]]
    rows.append(
        [
            "Dachneigung",
            f"{float(rec.get('roof_angle_deg')):.1f}" if rec.get("roof_angle_deg") is not None else "—",
            "°",
        ]
    )
    raw_rt = str(rec.get("roof_type") or "")
    rows.append(["Dachtyp", roof_type_label.get(raw_rt, raw_rt or "—"), "—"])
    rows.append(
        [
            "Dachfläche (mit Überstand)",
            f"{float(rec.get('roof_area_with_overhang_m2')):.2f}" if rec.get("roof_area_with_overhang_m2") is not None else "—",
            "m²",
        ]
    )
    rows.append(
        [
            "Dachfläche (ohne Überstand)",
            f"{float(rec.get('roof_area_without_overhang_m2')):.2f}" if rec.get("roof_area_without_overhang_m2") is not None else "—",
            "m²",
        ]
    )
    rows.append(
        [
            "Dachfläche gedämmt",
            f"{float(rec.get('roof_area_insulated_m2')):.2f}" if rec.get("roof_area_insulated_m2") is not None else "—",
            "m²",
        ]
    )
    rows.append(
        [
            "Dachumfang (mit Überstand)",
            f"{float(rec.get('roof_perimeter_with_overhang_m')):.2f}" if rec.get("roof_perimeter_with_overhang_m") is not None else "—",
            "m",
        ]
    )
    return rows


def _append_dachfenster_rows(rows: list[list[str]], roof_windows_count: int, roof_windows_area_m2: float) -> None:
    """Append Dachfenster directly to the roof table rows."""
    if roof_windows_count > 0:
        rows.append(["Anzahl Dachfenster (Öffnungen)", str(max(0, int(roof_windows_count))), "Stk."])
    if roof_windows_area_m2 > 0:
        rows.append(["Dachfensterfläche", f"{max(0.0, float(roof_windows_area_m2)):.2f}", "m²"])


def _style_data_table(t: Table) -> None:
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3E2C22")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F5F0E8")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#8B7355")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
            ]
        )
    )


_ROOF_MEASUREMENTS_FOOTER_DE = (
    "Hinweis: Die angegebenen Maße basieren auf den von Ihnen ausgewählten Optionen im "
    "Formular und im Bearbeitungsfenster sowie auf den von Ihnen gezeichneten Linien bei der "
    "Erkennung der Räume, der Öffnungen und der Dachflächen."
)


def _roof_measurements_footer_canvas(canv: Canvas, doc: SimpleDocTemplate) -> None:
    """Kleiner Disclaimer unten auf jeder Seite (ReportLab-Callback)."""
    canv.saveState()
    try:
        page_w, _page_h = A4
        margin_x = 15 * mm
        max_w = page_w - 2 * margin_x
        font_name = "Helvetica"
        font_size = 7
        lines = simpleSplit(_ROOF_MEASUREMENTS_FOOTER_DE, font_name, font_size, max_w)
        canv.setFont(font_name, font_size)
        canv.setFillColor(colors.HexColor("#5c5c5c"))
        # ReportLab: y grows upward — first split line must get the highest y so reading order is top→bottom.
        line_h = 3.4 * mm
        y_bottom = 8 * mm
        y = y_bottom + (len(lines) - 1) * line_h
        for line in lines:
            canv.drawString(margin_x, y, line)
            y -= line_h
    finally:
        canv.restoreState()


def _measurements_section_title_stockwerk(sw: str) -> str:
    """Pro Stockwerk: Wände/Böden plus Öffnungsfläche (immer, unabhängig vom Angebotsumfang)."""
    return f"{sw} – Wände, Böden, Öffnungen"


def _append_building_measurements_table(
    story: list,
    fd: dict[str, Any],
    sw: str,
    heading_style: ParagraphStyle,
    roof_windows_count: int = 0,
    roof_windows_area_m2: float = 0.0,
    inc: dict | None = None,
) -> None:
    inc = inc or {}
    show_finish = bool(inc.get("finishes"))
    show_win = bool(inc.get("openings"))
    show_doors = bool(inc.get("openings_doors"))

    # Fenster / Öffnungsfläche: Dachfenster werden im Dach-Tabellenblock ausgewiesen.
    total_windows = max(0, int(fd["num_windows"]) - int(roof_windows_count))
    base_oa = max(0.0, float(fd["openings_area_m2"]) - float(roof_windows_area_m2))
    total_openings_area = base_oa
    story.append(Paragraph(f"<b>{_measurements_section_title_stockwerk(sw)}</b>", heading_style))
    rows = [
        ["Element", "Wert", "Einheit"],
        ["Strukturen Innenwände (netto)", f"{fd['structure_int_net_m2']:.2f}", "m²"],
        ["Strukturen Außenwände (netto)", f"{fd['structure_ext_net_m2']:.2f}", "m²"],
    ]
    if show_finish:
        rows.append(["Innenausbau Innenwände (netto)", f"{fd.get('finish_int_inner_net_m2', 0.0):.2f}", "m²"])
        rows.append(["Innenausbau Außenwände (netto)", f"{fd.get('finish_int_outer_net_m2', 0.0):.2f}", "m²"])
        rows.append(["Außenfassade (netto)", f"{fd['finish_ext_net_m2']:.2f}", "m²"])
    rows.extend(
        [
            ["Bodenfläche / Planché", f"{fd['floor_area_m2']:.2f}", "m²"],
            ["Deckenfläche", f"{fd['ceiling_area_m2']:.2f}", "m²"],
        ]
    )
    if show_doors:
        rows.append(["Anzahl Türen", str(fd["num_doors"]), "Stk."])
    if show_win:
        rows.append(["Anzahl Fenster", str(total_windows), "Stk."])
    # Gesamtfläche aller Öffnungen pro Geschoss – immer (Angebotsumfang steuert nur Stückzahlen/Zuschläge, nicht die Mengenermittlung)
    rows.append(["Gesamtfläche Öffnungen", f"{total_openings_area:.2f}", "m²"])
    t = Table(rows, colWidths=[90 * mm, 50 * mm, 25 * mm])
    _style_data_table(t)
    story.append(t)
    story.append(Spacer(1, 8 * mm))


def _append_aufstockung_bestand_indicator_table(
    story: list[Any],
    sw: str,
    heading_style: ParagraphStyle,
    floor_phase1: dict[str, Any],
    *,
    out_root: Path | None = None,
    floor_kinds_for_mpp: list[str] | None = None,
    job_root: Path | None = None,
) -> None:
    """Measurements-only indicator block for Aufstockung existing-floor editor selections (no EUR)."""
    if not isinstance(floor_phase1, dict):
        return
    demo = floor_phase1.get("demolitionSelections") if isinstance(floor_phase1.get("demolitionSelections"), list) else []
    stairs = floor_phase1.get("stairOpenings") if isinstance(floor_phase1.get("stairOpenings"), list) else []
    statik = floor_phase1.get("statikChoice") if isinstance(floor_phase1.get("statikChoice"), dict) else {}
    statik_mode = str(statik.get("mode") or "none").strip().lower()
    new_mpp: float | None = None
    if out_root is not None and floor_kinds_for_mpp:
        new_mpp = _aufstockung_new_floor_mpp(out_root, floor_kinds_for_mpp, job_root)

    def _demo_area_m2(d: dict) -> float:
        a = float(d.get("area_m2") or d.get("area") or 0.0)
        if a <= 0 and new_mpp and new_mpp > 0:
            apx = _polygon_area_px2_from_points(d.get("points"))
            if apx > 0:
                a = float(apx) * (float(new_mpp) ** 2)
        return a

    total_demo_area = sum(_demo_area_m2(d) for d in demo if isinstance(d, dict))
    total_stair_area = 0.0
    for s in stairs:
        if not isinstance(s, dict):
            continue
        a = float(s.get("area_m2") or 0.0)
        if a <= 0 and new_mpp and new_mpp > 0:
            apx = _bbox_area_px2(s.get("bbox"))
            if apx > 0:
                a = float(apx) * (float(new_mpp) ** 2)
        total_stair_area += a
    has_custom_demo_price = isinstance(floor_phase1.get("customDemolitionPrice"), (int, float))

    story.append(Paragraph(f"<b>{sw} – Aufstockung Bestand-Auswahl</b>", heading_style))
    rows = [
        ["Indikator", "Wert", "Einheit"],
        ["Aufstandsflächen (Polygone)", str(len(demo)), "Stk."],
        ["Aufstandsfläche gesamt", f"{total_demo_area:.2f}", "m²"],
        ["Treppenöffnungen (Rechtecke)", str(len(stairs)), "Stk."],
        ["Fläche Treppenöffnungen gesamt", f"{total_stair_area:.2f}", "m²"],
    ]
    for si, s in enumerate(stairs):
        if not isinstance(s, dict):
            continue
        a = float(s.get("area_m2") or 0.0)
        if a <= 0 and new_mpp and new_mpp > 0:
            apx = _bbox_area_px2(s.get("bbox"))
            if apx > 0:
                a = float(apx) * (float(new_mpp) ** 2)
        ow, ol = s.get("opening_width_m"), s.get("opening_length_m")
        wf = float(ow) if isinstance(ow, (int, float)) else None
        lf = float(ol) if isinstance(ol, (int, float)) else None
        if (wf is None or lf is None or wf <= 0 or lf <= 0) and a > 0:
            wf, lf = _stair_opening_edge_lengths_m_from_bbox_and_area(s.get("bbox"), a)
        if (wf is None or lf is None or wf <= 0 or lf <= 0) and new_mpp and new_mpp > 0:
            bbox = s.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                try:
                    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                    w_px = max(0.0, x2 - x1)
                    h_px = max(0.0, y2 - y1)
                    if w_px > 0 and h_px > 0:
                        mpp = float(new_mpp)
                        wf = min(w_px, h_px) * mpp
                        lf = max(w_px, h_px) * mpp
                except (TypeError, ValueError):
                    pass
        dim_str = "—"
        dim_unit = "—"
        if wf is not None and lf is not None and wf > 0 and lf > 0:
            dim_str = f"{max(wf, lf):.2f} × {min(wf, lf):.2f}"
            dim_unit = "m"
        rows.append([f"Treppenöffnung #{si + 1} Öffnungsmaße (L×B)", dim_str, dim_unit])
        rows.append([f"Treppenöffnung #{si + 1} Fläche", f"{a:.2f}" if a > 0 else "—", "m²"])
    rows.extend(
        [
            ["Statik / Verstärkung", statik_mode if statik_mode != "none" else "keine Auswahl", "—"],
            ["Abbruch-Eigenpreis gesetzt", "Ja" if has_custom_demo_price else "Nein", "—"],
        ]
    )
    t = Table(rows, colWidths=[90 * mm, 50 * mm, 25 * mm])
    _style_data_table(t)
    story.append(t)
    story.append(Spacer(1, 8 * mm))


def _append_zubau_new_floor_indicator_table(
    story: list[Any],
    sw: str,
    heading_style: ParagraphStyle,
    floor_entry: dict[str, Any],
) -> None:
    """Measurements-only block for Zubau new-floor editor: Bestand-Marker count + Wandabbruch-Linien."""
    if not isinstance(floor_entry, dict):
        return
    lines = floor_entry.get("zubauWallDemolitionLines") if isinstance(floor_entry.get("zubauWallDemolitionLines"), list) else []
    n_best = int(floor_entry.get("zubauBestandPolygonCount") or 0)
    n_lines = len(lines) if lines else 0
    if n_lines == 0 and n_best == 0:
        return
    total_len = 0.0
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        try:
            total_len += float(ln.get("length_m") or 0.0)
        except (TypeError, ValueError):
            continue
    # Wandabbruch-Streifen: Länge × (Standard-Raumhöhe + Struktur-/Außen-Fertigungsaufschlag), wie Preisrechner
    strip_h_m = float(STANDARD_WALL_HEIGHT_M) + float(WALL_HEIGHT_EXTRA_STRUCTURE_AND_EXT_FINISH_M)
    abriss_flaeche_m2 = round(total_len * strip_h_m, 2) if total_len > 0 else 0.0
    story.append(Paragraph(f"<b>{sw} – Zubau (neues Geschoss)</b>", heading_style))
    rows = [
        ["Indikator", "Wert", "Einheit"],
        ["Bestand-Marker (Polygone)", str(n_best), "Stk."],
        ["Wandabbruch-Linien", str(n_lines), "Stk."],
        ["Summe Linienlänge (Planmaßstab)", f"{total_len:.2f}" if total_len > 0 else "—", "m"],
        [
            "Geschätzte Abrissfläche (Streifen Wandabbruch)",
            f"{abriss_flaeche_m2:.2f}" if abriss_flaeche_m2 > 0 else "—",
            "m²",
        ],
    ]
    t = Table(rows, colWidths=[90 * mm, 50 * mm, 25 * mm])
    _style_data_table(t)
    story.append(t)
    story.append(Spacer(1, 8 * mm))


def generate_roof_measurements_pdf(run_id: str, output_path: Path | None = None) -> Path | None:
    """
    PDF: Stockwerk 1…n = aceeași ordine ca în plans_list și în editor (detections_review);
    Grundriss = aceeași sursă ca fundalul editorului, apoi Dach- und Bauteil-Tabellen.
    Keine separaten Übersicht Dach / Übersicht Gebäude-Textblöcke.
    """
    out_root = OUTPUT_ROOT / run_id
    if not out_root.exists():
        out_root = JOBS_ROOT / run_id / "output"
    if not out_root.exists():
        out_root = RUNNER_ROOT / "output" / run_id
    if not out_root.exists():
        print(f"⚠️ [PDF ROOF] Output nu există: {run_id}")
        return None

    job_root = JOBS_ROOT / run_id
    if not job_root.exists():
        job_root = None
    frontend_data = load_frontend_data_for_run(run_id, job_root)
    roof_only = bool(frontend_data.get("roof_only_offer")) if isinstance(frontend_data, dict) else False
    fd_dict = frontend_data if isinstance(frontend_data, dict) else {}
    aufstockung_phase1 = (fd_dict.get("aufstockungPhase1") or {}) if isinstance(fd_dict, dict) else {}
    # Fallback: measurements PDF runs standalone and may not receive enriched phase1 in frontend_data.
    if not (isinstance(aufstockung_phase1, dict) and isinstance(aufstockung_phase1.get("existingFloors"), list)) and job_root:
        try:
            extras = {}
            extras_path = job_root / "detections_review_extras.json"
            if extras_path.exists():
                extras = json.loads(extras_path.read_text(encoding="utf-8"))
            # Prefer reviewed editor extras first; structuraCladirii often contains stale defaults.
            floor_kinds_raw = (
                (extras.get("floorKinds") if isinstance(extras, dict) else None)
                or (fd_dict.get("floorKinds") if isinstance(fd_dict, dict) else None)
                or (fd_dict.get("aufstockungFloorKinds") if isinstance(fd_dict, dict) else None)
                or ((fd_dict.get("structuraCladirii") or {}).get("aufstockungFloorKinds") if isinstance(fd_dict, dict) else None)
                or []
            )
            floor_kinds = [
                "new" if str(k).strip().lower() == "new" else "existing"
                for k in (floor_kinds_raw if isinstance(floor_kinds_raw, list) else [])
            ]
            fallback_global_statik = extras.get("statikChoice") if isinstance(extras, dict) and isinstance(extras.get("statikChoice"), dict) else {}
            manifest_path = job_root / "detections_review_manifest.json"
            plan_roots: list[Path] = []
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                raster_dirs = manifest.get("rasterDirs") if isinstance(manifest, dict) else []
                if isinstance(raster_dirs, list):
                    for rd in raster_dirs:
                        rp = Path(str(rd))
                        if rp.name == "raster":
                            plan_roots.append(rp.parent.parent)
            existing_floors = []
            for idx, plan_root in enumerate(plan_roots):
                edited_path = plan_root / "cubicasa_steps" / "raster" / "detections_edited.json"
                payload = {}
                if edited_path.exists():
                    try:
                        payload = json.loads(edited_path.read_text(encoding="utf-8"))
                    except Exception:
                        payload = {}
                floor_kind = floor_kinds[idx] if idx < len(floor_kinds) else "existing"
                if floor_kind == "new":
                    continue
                statik = payload.get("statikChoice") if isinstance(payload.get("statikChoice"), dict) else fallback_global_statik
                existing_floors.append({
                    "plan_index": idx,
                    "demolitionSelections": payload.get("roofDemolitions") if isinstance(payload.get("roofDemolitions"), list) else [],
                    "stairOpenings": payload.get("stairOpenings") if isinstance(payload.get("stairOpenings"), list) else [],
                    "customDemolitionPrice": payload.get("customDemolitionPrice"),
                    "statikChoice": statik if isinstance(statik, dict) else {},
                })
            aufstockung_phase1 = {"existingFloors": existing_floors}
        except Exception:
            aufstockung_phase1 = {}
    existing_phase_entries = (
        aufstockung_phase1.get("existingFloors")
        if isinstance(aufstockung_phase1, dict) and isinstance(aufstockung_phase1.get("existingFloors"), list)
        else []
    )
    phase1_by_plan_index: dict[int, dict[str, Any]] = {}
    for entry in existing_phase_entries:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("plan_index"))
        except Exception:
            continue
        phase1_by_plan_index[idx] = entry
    aufstockung_floor_kinds = _load_aufstockung_floor_kinds(run_id, fd_dict, job_root)
    nivel_oferta = normalize_nivel_oferta(fd_dict)
    pdf_inclusions = get_offer_inclusions(nivel_oferta, fd_dict)

    try:
        plan_infos = load_plan_infos(run_id, stage_name="pricing")
    except PlansListError:
        plan_infos = []

    # floorPlanOrder / floor_perm: poziție 0 = cel mai jos etaj (Keller/EG). PDF: aceeași ordine (prima pagină = cel mai jos).
    plans_orig = list(plan_infos)
    n_plans = len(plans_orig)
    floor_perm, manifest_labels = _load_floor_plan_order_and_labels_from_manifest(run_id, job_root)
    if (
        floor_perm
        and n_plans > 0
        and len(floor_perm) == n_plans
        and sorted(floor_perm) == list(range(n_plans))
    ):
        plans_ordered = [plans_orig[i] for i in floor_perm]
    else:
        plans_ordered = plans_orig
        floor_perm = None
    plans_pdf_sequence = plans_ordered

    plan_id_to_stockwerk: dict[str, str] = {}
    for i, p in enumerate(plans_ordered):
        if floor_perm is not None and manifest_labels:
            oi = floor_perm[i]
            if 0 <= oi < len(manifest_labels):
                lbl = (manifest_labels[oi] or "").strip()
                if lbl:
                    plan_id_to_stockwerk[p.plan_id] = lbl
                    continue
        plan_id_to_stockwerk[p.plan_id] = f"Stockwerk {i + 1}"

    # Ultimul plan în ordinea PDF = etajul cel mai de sus. La mansardă (nu pod), DE: Dachgeschoss.
    if _gebaudestruktur_last_floor_is_mansard(_lista_etaje_from_frontend(fd_dict)) and plans_pdf_sequence:
        top_pid = plans_pdf_sequence[-1].plan_id
        plan_id_to_stockwerk[top_pid] = "Dachgeschoss"

    roof_floor_idx_to_plan, plan_id_to_roof_floor_idx = _roof_floor_mapping_for_pdf(run_id, out_root)

    def plan_id_for_roof_floor_idx(floor_idx: int) -> str | None:
        pid = roof_floor_idx_to_plan.get(floor_idx)
        if pid:
            return pid
        if not roof_floor_idx_to_plan and plans_ordered and 0 <= floor_idx < len(plans_ordered):
            return plans_ordered[floor_idx].plan_id
        return None

    def label_for_roof_floor_idx(floor_idx: int) -> str:
        pid = plan_id_for_roof_floor_idx(floor_idx)
        if pid and pid in plan_id_to_stockwerk:
            return plan_id_to_stockwerk[pid]
        return f"Stockwerk {floor_idx + 1}"

    pdf_dir = out_root / "offer_pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = pdf_dir / f"roof_measurements_{run_id}.pdf"

    metrics_path = out_root / "roof" / "roof_3d" / "entire" / "mixed" / "roof_metrics.json"
    roof_pricing_path = out_root / "roof" / "roof_3d" / "entire" / "mixed" / "roof_pricing.json"
    metrics: dict[str, Any] = {}
    by_floor: dict[str, Any] = {}
    roof_pricing: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
            by_floor = metrics.get("by_floor") or {}
        except Exception as e:
            print(f"⚠️ [PDF ROOF] Eroare la citire roof_metrics.json: {e}")
    else:
        print(f"⚠️ [PDF ROOF] roof_metrics.json lipsește – PDF minimal.")
    if roof_pricing_path.exists():
        try:
            with open(roof_pricing_path, encoding="utf-8") as f:
                roof_pricing = json.load(f)
        except Exception:
            roof_pricing = {}

    rp_meas = (roof_pricing.get("roof_measurements") or {}) if isinstance(roof_pricing, dict) else {}
    by_rectangle: list[dict[str, Any]] = list((rp_meas.get("by_rectangle") or []))
    by_floor_roof = (metrics.get("unfold_roof") or {}).get("by_floor") or by_floor

    use_rect_plan_id = any(
        isinstance(r, dict) and str(r.get("plan_id") or "").strip() for r in by_rectangle
    )

    floors_by_plan = _load_floor_data_from_pricing(out_root)
    roof_windows_by_plan = _load_roof_windows_by_plan(out_root, run_id)

    # Tab-Titel im Browser: PDF-/Title-Metadaten (sonst „(anonymous)“ bei ReportLab).
    def _meta_offer_no(fd: dict[str, Any]) -> str:
        raw = str(fd.get("offer_no") or "").strip()
        if raw:
            return raw
        return (run_id[:8] if len(run_id) >= 8 else run_id) or "—"

    def _meta_author(fd: dict[str, Any]) -> str:
        pc = fd.get("pdf_company")
        if isinstance(pc, dict):
            n = str(pc.get("name") or "").strip()
            if n:
                return n
        return "Holzbot"

    _m_title = f"Mengenermittlung {_meta_offer_no(fd_dict)}"
    _m_author = _meta_author(fd_dict)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=22 * mm,
        title=_m_title,
        author=_m_author,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name="RoofTitle",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=8,
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        name="RoofHeading",
        parent=styles["Heading2"],
        fontSize=12,
        spaceAfter=6,
    )

    story: list[Any] = []
    story.append(Paragraph("Mengenermittlung", title_style))
    story.append(Spacer(1, 3 * mm))

    if plans_pdf_sequence:
        for si, plan in enumerate(plans_pdf_sequence):
            if si > 0:
                story.append(PageBreak())
            sw = plan_id_to_stockwerk[plan.plan_id]
            _append_plan_image(story, plan, out_root, job_root, heading_style)

            # Dach: același plan_id ca pricing/măsurători (din roof_pricing dacă există plan_id pe rând).
            if use_rect_plan_id:
                rects = [
                    r
                    for r in by_rectangle
                    if isinstance(r, dict) and str(r.get("plan_id") or "") == plan.plan_id
                ]
            else:
                _ri = plan_id_to_roof_floor_idx.get(plan.plan_id)
                if _ri is not None:
                    rects = [r for r in by_rectangle if int(r.get("floor_idx", -10_000_000)) == _ri]
                else:
                    rects = [
                        r
                        for r in by_rectangle
                        if plan_id_for_roof_floor_idx(int(r.get("floor_idx", 0))) == plan.plan_id
                    ]
            rects.sort(key=lambda r: (int(r.get("rectangle_idx", 0))))
            for rec in rects:
                rect_idx = int(rec.get("rectangle_idx", 0))
                roof_name = "Dach" if rect_idx == 0 else f"Dach {rect_idx + 1}"
                story.append(Paragraph(f"<b>{sw} – {roof_name}</b>", heading_style))
                rows_rect = _roof_rect_table_rows(rec)
                if rect_idx == 0:
                    rw = roof_windows_by_plan.get(plan.plan_id, {})
                    _append_dachfenster_rows(
                        rows_rect,
                        int(rw.get("count", 0) or 0),
                        float(rw.get("area_m2", 0.0) or 0.0),
                    )
                t_rect = Table(rows_rect, colWidths=[90 * mm, 50 * mm, 25 * mm])
                _style_data_table(t_rect)
                story.append(t_rect)
                story.append(Spacer(1, 6 * mm))

            if not rects:
                # Fallback: by_floor aus metrics; Keys können "0","1",… sein
                numeric_floor_keys: list[int] = []
                for k in by_floor_roof.keys():
                    try:
                        numeric_floor_keys.append(int(k))
                    except (TypeError, ValueError):
                        pass
                for fk in sorted({fk for fk in numeric_floor_keys if plan_id_for_roof_floor_idx(fk) == plan.plan_id}):
                    key = str(fk)
                    if key not in by_floor_roof:
                        continue
                    data = by_floor_roof[key]
                    story.append(Paragraph(f"<b>{sw} – Dach</b>", heading_style))
                    area_m2 = data.get("area_m2")
                    contour_m = data.get("contour_m")
                    rows = [
                        ["Element", "Wert", "Einheit"],
                        ["Fläche Dämmzone (gedämmte Dachfläche)", f"{area_m2:.2f}" if area_m2 is not None else "—", "m²"],
                        ["Eindeckfläche (Dachdeckung)", f"{area_m2:.2f}" if area_m2 is not None else "—", "m²"],
                        ["Klempnerarbeiten (Dachumfang)", f"{contour_m:.2f}" if contour_m is not None else "—", "m"],
                    ]
                    if pdf_inclusions.get("finishes"):
                        rows.append(
                            ["Fläche Innenausbau (Innenverkleidung)", f"{area_m2:.2f}" if area_m2 is not None else "—", "m²"]
                        )
                    rows.append(["Fläche Unterspannbahn / Folie", f"{area_m2:.2f}" if area_m2 is not None else "—", "m²"])
                    rw = roof_windows_by_plan.get(plan.plan_id, {})
                    _append_dachfenster_rows(
                        rows,
                        int(rw.get("count", 0) or 0),
                        float(rw.get("area_m2", 0.0) or 0.0),
                    )
                    t = Table(rows, colWidths=[90 * mm, 50 * mm, 25 * mm])
                    _style_data_table(t)
                    story.append(t)
                    story.append(Spacer(1, 8 * mm))

            rw = roof_windows_by_plan.get(plan.plan_id, {})
            rw_count = int(rw.get("count", 0) or 0)
            rw_area = float(rw.get("area_m2", 0.0) or 0.0)
            raw_plan_index = floor_perm[si] if isinstance(floor_perm, list) and si < len(floor_perm) else si
            _wp_pdf = str(fd_dict.get("wizard_package") or "").strip().lower()
            is_existing_aufstockung_floor = (
                _wp_pdf in ("aufstockung", "zubau")
                and (
                    raw_plan_index in phase1_by_plan_index
                    or (
                        0 <= raw_plan_index < len(aufstockung_floor_kinds)
                        and aufstockung_floor_kinds[raw_plan_index] != "new"
                    )
                )
            )
            if not roof_only and plan.plan_id in floors_by_plan and not is_existing_aufstockung_floor:
                _append_building_measurements_table(
                    story,
                    floors_by_plan[plan.plan_id],
                    sw,
                    heading_style,
                    roof_windows_count=rw_count,
                    roof_windows_area_m2=rw_area,
                    inc=pdf_inclusions,
                )
            phase1_entry = phase1_by_plan_index.get(raw_plan_index)
            if phase1_entry:
                _append_aufstockung_bestand_indicator_table(
                    story,
                    sw,
                    heading_style,
                    phase1_entry,
                    out_root=out_root,
                    floor_kinds_for_mpp=aufstockung_floor_kinds,
                    job_root=job_root,
                )
            if _wp_pdf == "zubau" and isinstance(aufstockung_phase1, dict):
                z_nf = aufstockung_phase1.get("newFloors") or []
                z_ent = None
                for ent in z_nf:
                    if isinstance(ent, dict) and int(ent.get("plan_index", -1)) == int(raw_plan_index):
                        z_ent = ent
                        break
                if isinstance(z_ent, dict) and str(z_ent.get("floorKind") or "").strip().lower() == "new":
                    _append_zubau_new_floor_indicator_table(story, sw, heading_style, z_ent)

    else:
        # Kein plans_list: nur Dachtabellen nach roof_floor_idx
        if by_rectangle:
            for rec in sorted(by_rectangle, key=lambda r: (int(r.get("floor_idx", 0)), int(r.get("rectangle_idx", 0)))):
                floor_idx = int(rec.get("floor_idx", 0))
                rect_idx = int(rec.get("rectangle_idx", 0))
                fl = label_for_roof_floor_idx(floor_idx)
                roof_name = "Dach" if rect_idx == 0 else f"Dach {rect_idx + 1}"
                story.append(Paragraph(f"<b>{fl} – {roof_name}</b>", heading_style))
                rows_rect = _roof_rect_table_rows(rec)
                pid = plan_id_for_roof_floor_idx(floor_idx)
                if rect_idx == 0 and pid:
                    rw = roof_windows_by_plan.get(pid, {})
                    _append_dachfenster_rows(
                        rows_rect,
                        int(rw.get("count", 0) or 0),
                        float(rw.get("area_m2", 0.0) or 0.0),
                    )
                t_rect = Table(rows_rect, colWidths=[90 * mm, 50 * mm, 25 * mm])
                _style_data_table(t_rect)
                story.append(t_rect)
                story.append(Spacer(1, 8 * mm))
        else:
            for floor_key in sorted(by_floor_roof.keys(), key=lambda k: int(k) if str(k).isdigit() else 0):
                data = by_floor_roof[floor_key]
                idx = int(floor_key) if str(floor_key).isdigit() else 0
                story.append(Paragraph(f"<b>{label_for_roof_floor_idx(idx)}</b>", heading_style))
                area_m2 = data.get("area_m2")
                contour_m = data.get("contour_m")
                rows = [
                    ["Element", "Wert", "Einheit"],
                    ["Fläche Dämmzone (gedämmte Dachfläche)", f"{area_m2:.2f}" if area_m2 is not None else "—", "m²"],
                    ["Eindeckfläche (Dachdeckung)", f"{area_m2:.2f}" if area_m2 is not None else "—", "m²"],
                    ["Klempnerarbeiten (Dachumfang)", f"{contour_m:.2f}" if contour_m is not None else "—", "m"],
                ]
                if pdf_inclusions.get("finishes"):
                    rows.append(
                        ["Fläche Innenausbau (Innenverkleidung)", f"{area_m2:.2f}" if area_m2 is not None else "—", "m²"]
                    )
                rows.append(["Fläche Unterspannbahn / Folie", f"{area_m2:.2f}" if area_m2 is not None else "—", "m²"])
                pid = plan_id_for_roof_floor_idx(idx)
                if pid:
                    rw = roof_windows_by_plan.get(pid, {})
                    _append_dachfenster_rows(
                        rows,
                        int(rw.get("count", 0) or 0),
                        float(rw.get("area_m2", 0.0) or 0.0),
                    )
                t = Table(rows, colWidths=[90 * mm, 50 * mm, 25 * mm])
                _style_data_table(t)
                story.append(t)
                story.append(Spacer(1, 10 * mm))

        if not roof_only and floors_by_plan:
            story.append(PageBreak())
            for i, (pid, fd) in enumerate(sorted(floors_by_plan.items())):
                if i > 0:
                    story.append(Spacer(1, 6 * mm))
                sw = f"Stockwerk {i + 1}"
                _append_building_measurements_table(story, fd, sw, heading_style, inc=pdf_inclusions)

    doc.build(
        story,
        onFirstPage=_roof_measurements_footer_canvas,
        onLaterPages=_roof_measurements_footer_canvas,
    )
    print(f"✅ [PDF ROOF] Generat: {output_path}")
    return output_path
