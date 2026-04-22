# file: engine/main.py
from __future__ import annotations

import sys
import os

# Limită thread-uri OpenCV/NumPy ca paralelizarea etajelor să nu se bată cu thread-urile interne
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import cv2
cv2.setNumThreads(1)

from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import argparse
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import build_job_root, RUNS_ROOT, RUNNER_ROOT, OUTPUT_ROOT, load_plan_infos, PlanInfo, get_run_dir
from config.frontend_loader import load_frontend_data_for_run
from floor_classifier.basement_scorer import run_basement_scoring
from floor_classifier.floor_order_gemini import run_floor_order_from_gemini
from typing import List, Tuple

# ✅ IMPORT SEGMENTER JOBS
from segmenter.jobs import run_segmentation_for_documents, get_all_classification_results

from floor_classifier import run_floor_classification, FloorClassificationResult
from cubicasa_detector.raster_api import save_detections_review_image, apply_detections_edited
from cubicasa_detector.manual_blueprint import (
    prepare_manual_blueprint_workspace,
    run_walls_pipeline_after_manual_editor,
    ensure_roof_3d_floor_manifest,
    seed_roof_only_rooms_from_roof_polygons,
    apply_roof_only_synthetic_walls_and_scale,
)
from scale import run_scale_detection_for_run
from count_objects import run_count_objects_for_run
from roof.jobs import run_roof_for_run
from pricing.jobs import run_pricing_for_run, PricingJobResult
from offer_builder import build_final_offer
from pdf_generator import generate_complete_offer_pdf, generate_admin_offer_pdf, generate_admin_calculation_method_pdf, generate_roof_measurements_pdf
from roof.roof_pricing import generate_roof_pricing
from segmenter.pdf_utils import convert_pdf_to_png

# =========================================================
# UI NOTIFICATION
# =========================================================
def notify_ui(stage_tag: str, image_path: Path | str | None = None):
    """Trimite un semnal către Backend (NestJS) care va fi propagat în Frontend (LiveFeed)."""
    msg = f">>> UI:STAGE:{stage_tag}"
    if image_path:
        abs_path = Path(image_path).resolve()
        if abs_path.exists():
            msg += f"|IMG:{str(abs_path)}"
        else:
            print(f"⚠️ [UI] Image path not found: {abs_path}", flush=True)
    print(msg, flush=True)
    sys.stdout.flush()


def notify_ui_batch(stage_tag: str, image_paths: list):
    """Trimite un singur eveniment cu toate imaginile (pentru afișare în LiveFeed doar când toate planurile sunt gata)."""
    if not image_paths:
        return
    msg = f">>> UI:STAGE:{stage_tag}"
    for p in image_paths:
        abs_path = Path(p).resolve()
        if abs_path.exists():
            msg += f"|IMG:{str(abs_path)}"
        else:
            print(f"⚠️ [UI] Image path not found: {abs_path}", flush=True)
    print(msg, flush=True)
    sys.stdout.flush()


OFFER_NUMBER_WAIT_TIMEOUT_SEC = 180.0


def wait_for_offer_number_in_job(job_root: Path) -> None:
    """API assigns offer_no and patches jobs/<job_id>/frontend_data.json (tenant counter updated there)."""
    job_file = job_root / "frontend_data.json"
    notify_ui("offer_number_request")
    deadline = time.time() + OFFER_NUMBER_WAIT_TIMEOUT_SEC
    while time.time() < deadline:
        if job_file.exists():
            try:
                data = json.loads(job_file.read_text(encoding="utf-8"))
                ono = data.get("offer_no")
                if isinstance(ono, str) and ono.strip():
                    print(f"✅ [OfferNo] ready: {ono.strip()}", flush=True)
                    return
            except Exception:
                pass
        time.sleep(0.25)
    print(">>> ERROR: Timeout waiting for offer_no (API handshake)", flush=True)
    sys.exit(1)


DETECTIONS_REVIEW_FLAG = "detections_review_approved.flag"
DETECTIONS_REVIEW_WAIT_TIMEOUT_SEC = 3600
DETECTIONS_REVIEW_POLL_INTERVAL_SEC = 1.5
ROOF_REVIEW_FLAG = "roof_review_approved.flag"
ROOF_REVIEW_WAIT_TIMEOUT_SEC = 3600
ROOF_REVIEW_POLL_INTERVAL_SEC = 1.5


def _wait_detections_review_approval(job_root: Path, set_progress) -> None:
    """Blochează până când API-ul scrie flag-ul de aprobare (utilizatorul a confirmat detecțiile) sau timeout.
    În timpul așteptării, crește ușor progresul (cap sub raster_end-1) ca Live Feed-ul să nu rămână înghețat."""
    flag_path = job_root / DETECTIONS_REVIEW_FLAG
    if flag_path.exists():
        return
    print(f"⏸️ [Pipeline] Aștept confirmarea detecțiilor în UI (max {DETECTIONS_REVIEW_WAIT_TIMEOUT_SEC}s)...", flush=True)
    deadline = time.time() + DETECTIONS_REVIEW_WAIT_TIMEOUT_SEC
    # Creep lin: ~8 → max ~54 (sub raster_end-1 = 57), ca utilizatorul să vadă mișcare în timpul editării
    creep_cap = float(_PROGRESS_WEIGHTS["raster_end"] - 3)
    creep_step = 0.28
    creep_interval_sec = 8.0
    stop_creep = threading.Event()

    creep_next = [float(_PROGRESS_WEIGHTS["raster_start"])]

    def _creep_worker():
        while not stop_creep.is_set() and time.time() < deadline:
            if stop_creep.wait(creep_interval_sec):
                break
            if flag_path.exists():
                break
            creep_next[0] = min(creep_cap, creep_next[0] + creep_step)
            set_progress(creep_next[0])
    creep_th = threading.Thread(target=_creep_worker, name="det-review-creep", daemon=True)
    creep_th.start()
    try:
        while time.time() < deadline:
            if flag_path.exists():
                print(f"✅ [Pipeline] Detecții confirmate, continuăm.", flush=True)
                return
            time.sleep(DETECTIONS_REVIEW_POLL_INTERVAL_SEC)
    finally:
        stop_creep.set()
    print(f"⚠️ [Pipeline] Timeout așteptare confirmare detecții; continuăm fără confirmare.", flush=True)


def _wait_roof_review_approval(job_root: Path) -> None:
    """Blochează până când API-ul scrie flag-ul de aprobare pentru editorul de acoperiș (sau timeout)."""
    flag_path = job_root / ROOF_REVIEW_FLAG
    if flag_path.exists():
        return
    print(f"⏸️ [Pipeline] Aștept confirmarea acoperișului în UI (max {ROOF_REVIEW_WAIT_TIMEOUT_SEC}s)...", flush=True)
    deadline = time.time() + ROOF_REVIEW_WAIT_TIMEOUT_SEC
    while time.time() < deadline:
        if flag_path.exists():
            print("✅ [Pipeline] Acoperiș confirmat, continuăm.", flush=True)
            return
        time.sleep(ROOF_REVIEW_POLL_INTERVAL_SEC)
    print("⚠️ [Pipeline] Timeout așteptare confirmare acoperiș; continuăm fără confirmare.", flush=True)


def notify_progress(percent: float):
    """Trimite procentul de progres către Backend/Frontend pentru progress bar (0–100, zecimale ok)."""
    p = max(0.0, min(100.0, float(percent)))
    if abs(p - round(p)) < 1e-6:
        msg = f">>> UI:PROGRESS:{int(round(p))}"
    else:
        msg = f">>> UI:PROGRESS:{p:.2f}"
    print(msg, flush=True)
    sys.stdout.flush()


# Progress monoton: un singur contor 0–100, creștere uniformă pe parcursul rulării.
# Greutăți aproximativ proporționale cu timpul (RasterScan ~50%, Count+Roof ~22%, etc.).
_PROGRESS_WEIGHTS = {
    "seg_end": 3,
    "floor_end": 6,
    "detections_end": 8,
    "raster_start": 8,
    "raster_end": 58,   # 50% pentru RasterScan (8→58)
    "scale_end": 60,
    "count_roof_end": 82,
    "pricing_end": 85,
    "offer_end": 88,
    "pdf_end": 100,
}


def _make_progress_sender():
    """Returnează (set_progress, get_raster_progress_fn).
    set_progress(pct) trimite doar dacă pct > ultimul trimis (monoton).
    get_raster_progress_fn(num_plans) returnează on_raster_tick() pentru RasterScan.
    """
    max_sent = [0.0]

    def set_progress(pct: float):
        pct = max(0.0, min(100.0, float(pct)))
        if pct > max_sent[0] + 1e-9:
            max_sent[0] = pct
            notify_progress(pct)

    def get_raster_progress_fn(num_plans: int):
        total_ticks = 14 * num_plans  # 4 P1 + 2 P2 start+brute + 7 walls_from_coords + 1 P2 end
        raster_done = [0]

        def on_raster_tick():
            raster_done[0] += 1
            pct = _PROGRESS_WEIGHTS["raster_start"] + (
                _PROGRESS_WEIGHTS["raster_end"] - _PROGRESS_WEIGHTS["raster_start"]
            ) * min(raster_done[0], total_ticks) / total_ticks
            set_progress(pct)
        return on_raster_tick

    return set_progress, get_raster_progress_fn


def _notify_all_cubicasa_images_for_plan(plan: PlanInfo) -> None:
    """
    Trimite către UI (admin Details) toate imaginile generate de Cubicasa pentru acest plan:
    pereți, camere, fill, walls_from_coords, openings, wall_repair, raster, etc.
    """
    base = plan.stage_work_dir / "cubicasa_steps"
    if not base.is_dir():
        return
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in sorted(base.rglob(ext)):
            if p.is_file():
                notify_ui("cubicasa_step", p)


def _check_raster_complete(plans: List[PlanInfo], job_root: Path | None = None) -> Tuple[bool, List[str]]:
    """
    Verifică că toate planurile au camerele calculate:
    - room_scales.json există, total_area_m2 > 0, total_area_px > 0
    - cel puțin o cameră în room_scales (sare pentru roof_only_offer sintetic)
    - rooms.png există (masca camerelor)
    Returns (all_ok, failed_plan_ids).
    """
    roof_only = False
    if job_root is not None:
        try:
            fd = load_frontend_data_for_run(job_root.name, job_root)
            roof_only = bool(fd.get("roof_only_offer"))
        except Exception:
            roof_only = False
    failed: List[str] = []
    for plan in plans:
        base = plan.stage_work_dir / "cubicasa_steps"
        room_scales_path = base / "raster_processing" / "walls_from_coords" / "room_scales.json"
        # rooms.png poate fi în raster/rooms.png sau raster_processing/rooms.png
        rooms_png_path_raster = base / "raster" / "rooms.png"
        rooms_png_path_processing = base / "raster_processing" / "rooms.png"
        rooms_png_path = rooms_png_path_raster if rooms_png_path_raster.exists() else rooms_png_path_processing

        if not room_scales_path.exists():
            failed.append(plan.plan_id)
            continue
        try:
            data = json.loads(room_scales_path.read_text(encoding="utf-8"))
            skip_ps = str(data.get("pipeline_skipped") or "").strip()
            if skip_ps == "no_editor_room_selections":
                if not rooms_png_path.exists():
                    failed.append(plan.plan_id)
                continue

            m2 = float(data.get("total_area_m2") or 0)
            px = int(data.get("total_area_px") or 0)
            rooms_data = data.get("room_scales") or data.get("rooms") or {}
            num_rooms = len(rooms_data) if isinstance(rooms_data, dict) else 0

            if m2 <= 0 or px <= 0:
                failed.append(plan.plan_id)
            elif num_rooms < 1 and not roof_only:
                failed.append(plan.plan_id)
            elif not rooms_png_path.exists():
                failed.append(plan.plan_id)
        except Exception:
            failed.append(plan.plan_id)
    return (len(failed) == 0, failed)


def _enrich_frontend_with_aufstockung_phase1(frontend_data: dict, job_root: Path) -> dict:
    """Build canonical Aufstockung phase1 payload from per-plan editor artifacts."""
    if not isinstance(frontend_data, dict):
        return frontend_data
    out = dict(frontend_data)
    wizard_package = str(out.get("wizard_package") or "").strip().lower()
    if wizard_package not in ("aufstockung", "zubau", "zubau_aufstockung"):
        return out

    extras = {}
    extras_path = job_root / "detections_review_extras.json"
    if extras_path.exists():
        try:
            extras = json.loads(extras_path.read_text(encoding="utf-8"))
        except Exception:
            extras = {}

    # Prefer editor extras (detections_review) over structuraCladirii — wizard JSON is often stale vs. review UI.
    floor_kinds_raw = (
        (extras.get("floorKinds") if isinstance(extras, dict) else None)
        or out.get("aufstockungFloorKinds")
        or out.get("floorKinds")
        or (out.get("structuraCladirii") or {}).get("aufstockungFloorKinds")
        or []
    )
    floor_kinds = []
    for k in (floor_kinds_raw if isinstance(floor_kinds_raw, list) else []):
        v = str(k).strip().lower()
        if v in ("new", "zubau"):
            floor_kinds.append("zubau")
        elif v == "aufstockung":
            floor_kinds.append("aufstockung")
        else:
            floor_kinds.append("existing")

    def _polygon_area_px2(points: object) -> float:
        if not isinstance(points, list) or len(points) < 3:
            return 0.0
        pts: list[tuple[float, float]] = []
        for p in points:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    pts.append((float(p[0]), float(p[1])))
                except Exception:
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

    def _load_mpp_for_plan(plan_root: Path) -> float | None:
        scale_result_path = plan_root / "scale_result.json"
        if scale_result_path.exists():
            try:
                scale_data = json.loads(scale_result_path.read_text(encoding="utf-8"))
                mpp_raw = scale_data.get("meters_per_pixel")
                if isinstance(mpp_raw, (int, float)) and float(mpp_raw) > 0:
                    return float(mpp_raw)
            except Exception:
                pass
        room_scales_path = plan_root / "cubicasa_steps" / "raster_processing" / "walls_from_coords" / "room_scales.json"
        if room_scales_path.exists():
            try:
                room_scales = json.loads(room_scales_path.read_text(encoding="utf-8"))
                mpp_raw = room_scales.get("m_px") or room_scales.get("weighted_average_m_px")
                if isinstance(mpp_raw, (int, float)) and float(mpp_raw) > 0:
                    return float(mpp_raw)
            except Exception:
                pass
        return None

    def _infer_mpp_from_doors(p: dict) -> float | None:
        doors = p.get("doors")
        if not isinstance(doors, list):
            return None
        for d in doors:
            if not isinstance(d, dict):
                continue
            h_raw = d.get("height_m")
            if not isinstance(h_raw, (int, float)) or float(h_raw) <= 0:
                continue
            h = float(h_raw)
            bbox = d.get("bbox")
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            try:
                y1, y2 = float(bbox[1]), float(bbox[3])
            except Exception:
                continue
            px_h = abs(y2 - y1)
            if px_h > 1.0:
                return h / px_h
        return None

    def _first_valid_mpp_from_list(mpps: list[float | None]) -> float | None:
        for m in mpps:
            if m is not None and m > 0:
                return float(m)
        return None

    phase1 = dict(out.get("aufstockungPhase1") or {})
    existing_floors = []
    new_floors = []
    demolition_flat = []
    stairs_flat = []
    fallback_global_statik = extras.get("statikChoice") if isinstance(extras, dict) and isinstance(extras.get("statikChoice"), dict) else None

    # Run artifacts live under output/<run_id>/..., not under jobs/<run_id>/scale.
    run_id = str(job_root.name)
    scale_root = OUTPUT_ROOT / run_id / "scale"
    manifest_path = job_root / "detections_review_manifest.json"
    plan_roots = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            raster_dirs = manifest.get("rasterDirs") if isinstance(manifest, dict) else []
            if isinstance(raster_dirs, list):
                for rd in raster_dirs:
                    rp = Path(str(rd))
                    # raster dir => .../scale/<plan>/cubicasa_steps/raster
                    if rp.name == "raster":
                        plan_roots.append(rp.parent.parent)
        except Exception:
            plan_roots = []
    if not plan_roots and scale_root.exists():
        plan_roots = [p for p in sorted(scale_root.iterdir()) if p.is_dir()]

    mpp_disk: list[float | None] = [_load_mpp_for_plan(pr) for pr in plan_roots]

    def _mpp_for_plan_idx(plan_idx: int, payload: dict) -> float | None:
        if 0 <= plan_idx < len(mpp_disk):
            own = mpp_disk[plan_idx]
            if own is not None and own > 0:
                return float(own)
        inferred = _infer_mpp_from_doors(payload)
        if inferred is not None and inferred > 0:
            return float(inferred)
        return _first_valid_mpp_from_list(mpp_disk)

    for idx, plan_root in enumerate(plan_roots):
        edited_path = plan_root / "cubicasa_steps" / "raster" / "detections_edited.json"
        payload = {}
        if edited_path.exists():
            try:
                payload = json.loads(edited_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        floor_kind = floor_kinds[idx] if idx < len(floor_kinds) else "existing"
        roof_demolitions = payload.get("roofDemolitions") if isinstance(payload.get("roofDemolitions"), list) else []
        stair_openings = payload.get("stairOpenings") if isinstance(payload.get("stairOpenings"), list) else []
        custom_demolition_price = (
            float(payload.get("customDemolitionPrice"))
            if isinstance(payload.get("customDemolitionPrice"), (int, float))
            else None
        )
        statik_raw = payload.get("statikChoice") if isinstance(payload.get("statikChoice"), dict) else fallback_global_statik
        statik_mode = str((statik_raw or {}).get("mode") or "none").strip().lower()
        statik_choice = {"mode": statik_mode if statik_mode in ("stahlbetonverbunddecke", "sonderkonstruktion") else "none"}
        if statik_choice["mode"] == "sonderkonstruktion":
            custom_piece = (statik_raw or {}).get("customPiecePrice")
            if isinstance(custom_piece, (int, float)):
                statik_choice["customPiecePrice"] = float(custom_piece)

        plan_mpp = _mpp_for_plan_idx(idx, payload)

        floor_entry = {
            "plan_index": idx,
            "floorKind": floor_kind,
            "demolitionSelections": [],
            "stairOpenings": [],
            "customDemolitionPrice": custom_demolition_price,
            "statikChoice": statik_choice,
        }
        for d in roof_demolitions:
            if not isinstance(d, dict):
                continue
            area_m2 = float(d.get("area_m2") or d.get("area") or 0.0)
            if area_m2 <= 0 and plan_mpp and plan_mpp > 0:
                area_px2 = _polygon_area_px2(d.get("points"))
                if area_px2 > 0:
                    area_m2 = float(area_px2) * (float(plan_mpp) ** 2)
            if area_m2 <= 0:
                continue
            demolition_item = {
                "area_m2": area_m2,
                "price_key": str(d.get("price_key") or "aufstockung_roof_demolition_m2"),
            }
            floor_entry["demolitionSelections"].append(demolition_item)
            if floor_kind == "existing":
                demolition_flat.append({**demolition_item, "plan_index": idx})
        for s in stair_openings:
            if not isinstance(s, dict):
                continue
            bbox = s.get("bbox")
            stair_area_m2 = float(s.get("area_m2") or 0.0)
            if stair_area_m2 <= 0 and plan_mpp and plan_mpp > 0:
                if isinstance(bbox, list) and len(bbox) == 4:
                    try:
                        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                        w = max(0.0, x2 - x1)
                        h = max(0.0, y2 - y1)
                        apx = w * h
                        if apx > 0:
                            stair_area_m2 = float(apx) * (float(plan_mpp) ** 2)
                    except Exception:
                        pass
            pk = str(s.get("price_key") or "").strip()
            if not pk or pk == "aufstockung_stair_opening_m2":
                pk = "aufstockung_stair_opening_piece"
            if stair_area_m2 <= 0:
                continue
            w_m = h_m = None
            if isinstance(bbox, list) and len(bbox) == 4 and plan_mpp and plan_mpp > 0:
                try:
                    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                    w_px = max(0.0, x2 - x1)
                    h_px = max(0.0, y2 - y1)
                    if w_px > 0 and h_px > 0:
                        w_m = w_px * float(plan_mpp)
                        h_m = h_px * float(plan_mpp)
                except Exception:
                    w_m = h_m = None
            stair_item: dict = {
                "area_m2": stair_area_m2,
                "price_key": pk,
            }
            if isinstance(bbox, list) and len(bbox) == 4:
                stair_item["bbox"] = bbox
            if w_m is not None and h_m is not None and w_m > 0 and h_m > 0:
                stair_item["opening_width_m"] = round(min(w_m, h_m), 3)
                stair_item["opening_length_m"] = round(max(w_m, h_m), 3)
            floor_entry["stairOpenings"].append(stair_item)
            if floor_kind == "existing":
                stairs_flat.append({**stair_item, "plan_index": idx})

        if wizard_package in ("zubau", "zubau_aufstockung") and floor_kind == "zubau":
            z_best = payload.get("zubauBestandPolygons") if isinstance(payload.get("zubauBestandPolygons"), list) else []
            z_lines_raw = payload.get("zubauWallDemolitionLines") if isinstance(payload.get("zubauWallDemolitionLines"), list) else []

            def _pt_xy(p: object) -> tuple[float, float] | None:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    try:
                        return float(p[0]), float(p[1])
                    except Exception:
                        return None
                if isinstance(p, dict) and "x" in p and "y" in p:
                    try:
                        return float(p["x"]), float(p["y"])
                    except Exception:
                        return None
                return None

            z_lines_out: list[dict] = []
            for ln in z_lines_raw:
                if not isinstance(ln, dict):
                    continue
                a = _pt_xy(ln.get("a"))
                b = _pt_xy(ln.get("b"))
                if not a or not b:
                    continue
                dx = a[0] - b[0]
                dy = a[1] - b[1]
                len_px = (dx * dx + dy * dy) ** 0.5
                length_m = float(len_px) * float(plan_mpp) if plan_mpp and plan_mpp > 0 else float(ln.get("length_m") or 0.0)
                if length_m <= 0:
                    continue
                pk = str(ln.get("price_key") or "aufstockung_demolition_roof_basic_m2").strip()
                z_lines_out.append(
                    {
                        "a": [round(a[0], 2), round(a[1], 2)],
                        "b": [round(b[0], 2), round(b[1], 2)],
                        "length_m": round(length_m, 4),
                        "price_key": pk,
                    }
                )
            if z_lines_out:
                floor_entry["zubauWallDemolitionLines"] = z_lines_out
            if z_best:
                floor_entry["zubauBestandPolygonCount"] = len(z_best)

        if floor_kind in ("new", "zubau", "aufstockung"):
            new_floors.append(floor_entry)
        else:
            existing_floors.append(floor_entry)

    phase1["existingFloors"] = existing_floors
    phase1["newFloors"] = new_floors
    phase1["demolitionSelections"] = demolition_flat
    phase1["stairOpenings"] = stairs_flat
    if isinstance(extras.get("globalCombinedPrice"), (int, float)):
        phase1["globalCombinedPrice"] = float(extras.get("globalCombinedPrice"))
    out["aufstockungPhase1"] = phase1
    return out


# =========================================================
# TIMER UTILITIES
# =========================================================
class Timer:
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"\n⏱️  START: {self.step_name}", flush=True)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        print(f"✅ FINISH: {self.step_name} ({self.end_time - self.start_time:.2f}s)\n", flush=True)
        return False

class PipelineTimer:
    def __init__(self): 
        self.steps = []
        self.total_start = None
        self.total_end = None
    
    def start(self): 
        self.total_start = time.time()
    
    def add_step(self, name, dur): 
        self.steps.append({"name": name, "duration": dur})
    
    def finish(self): 
        self.total_end = time.time()
        self.print_summary()
    
    def print_summary(self):
        """Afișează rezumatul timpilor pentru fiecare pas."""
        if not self.steps:
            return
        
        total_duration = self.total_end - self.total_start if self.total_start and self.total_end else 0
        
        print("\n" + "="*70)
        print("⏱️  REZUMAT TIMPI PENTRU FIECARE PAS")
        print("="*70)
        
        # Sortăm după durată (descrescător)
        sorted_steps = sorted(self.steps, key=lambda x: x["duration"], reverse=True)
        
        for step in sorted_steps:
            duration = step["duration"]
            percentage = (duration / total_duration * 100) if total_duration > 0 else 0
            print(f"  {step['name']:.<40} {duration:>7.2f}s ({percentage:>5.1f}%)")
        
        print("-"*70)
        print(f"  {'TOTAL':.<40} {total_duration:>7.2f}s (100.0%)")
        print("="*70 + "\n")

pipeline_timer = PipelineTimer()

@dataclass
class ClassifiedPlanInfo:
    job_root: Path
    image_path: Path
    label: str

def _create_run_for_detections(job_root: Path, house_plans: list[ClassifiedPlanInfo]) -> str:
    run_id = job_root.name
    run_dir = RUNS_ROOT / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"plans": [str(p.image_path) for p in house_plans]}
    (run_dir / "plans_list.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_id

# =========================================================
# MAIN LOGIC
# =========================================================

def run_segmentation_and_classification_for_document(
    input_path: str | Path, 
    job_id: str | None = None,
    frontend_data_json: str | None = None
):
    pipeline_timer.start()
    input_path = Path(input_path).resolve()
    job_root = build_job_root(job_id=job_id, prefix="segmentation_job")
    
    # Save frontend data (from env). Skip if file already exists - API may have written full payload to avoid env truncation.
    if frontend_data_json and not (job_root / "frontend_data.json").exists():
        try:
            parsed_data = json.loads(frontend_data_json)
            out_file = job_root / "frontend_data.json"
            out_file.write_text(json.dumps(parsed_data, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"✅ Frontend data saved to {out_file}")
        except Exception as e:
            print(f"⚠️ Failed to save frontend data: {e}")
    elif (job_root / "frontend_data.json").exists():
        print(f"✅ Using existing frontend_data.json from job root (written by API)")

    notify_ui("segmentation_start")
    set_progress, get_raster_progress_fn = _make_progress_sender()
    set_progress(_PROGRESS_WEIGHTS["seg_end"])
    
    # Preview pentru UI (Input files)
    preview_dir = job_root / "_previews"
    preview_dir.mkdir(exist_ok=True)
    
    # Detectează fișierele de intrare
    if input_path.is_dir():
        files_to_process = [f for f in input_path.iterdir() if f.is_file() and not f.name.startswith('.')]
        files_to_process.sort(key=lambda f: f.name)
    else:
        files_to_process = [input_path]
    
    for f in files_to_process:
        file_to_show = f
        if f.suffix.lower() == '.pdf':
            try:
                pngs = convert_pdf_to_png(f, preview_dir)
                if pngs:
                    file_to_show = Path(pngs[0])
            except Exception as e:
                print(f"⚠️ Preview error: {e}")
        notify_ui("segmentation_start", file_to_show)

    segmentation_out_base = job_root / "segmentation"
    segmentation_out_base.mkdir(parents=True, exist_ok=True)
    
    # =========================================================
    # SEGMENTATION & CLASSIFICATION (folosind jobs.py)
    # =========================================================
    with Timer("STEP 1-3: Multi-file Segmentation & Classification") as t:
        def _seg_progress(done: int, total: int):
            if total and total > 0:
                # Interpolare între seg_end și floor_end pentru smoothness
                pct = _PROGRESS_WEIGHTS["seg_end"] + (
                    _PROGRESS_WEIGHTS["floor_end"] - _PROGRESS_WEIGHTS["seg_end"]
                ) * (done / total) * 0.9  # 90% din segmentare, 10% la final
                set_progress(pct)
        seg_results = run_segmentation_for_documents(
            input_path=input_path,
            output_base_dir=segmentation_out_base,
            max_workers=None,
            progress_callback=_seg_progress,
        )
        
        # -------------------------------------------------------------
        # ✅ 1. UI: SOLID WALLS FIRST (Robust Check)
        # -------------------------------------------------------------
        print("📢 [UI] Sending Solid Walls previews (Priority 1)...")
        set_progress(_PROGRESS_WEIGHTS["seg_end"] + 1)
        for i, seg_result in enumerate(seg_results):
            # Calea standard generată de segmenter
            standard_solid_img = seg_result.work_dir / "solid_walls" / "solidified.jpg"
            
            sent = False
            if standard_solid_img.exists():
                notify_ui("segmentation", standard_solid_img)
                sent = True
            else:
                # Fallback: Caută orice JPG în folder (pentru pagini multiple sau naming diferit)
                solid_dir = seg_result.work_dir / "solid_walls"
                if solid_dir.exists():
                    found_imgs = list(solid_dir.glob("*.jpg"))
                    for img in found_imgs:
                        notify_ui("segmentation", img)
                        sent = True
            
            if not sent:
                print(f"⚠️ [UI] Nu am găsit solid walls pentru doc {i}: {seg_result.doc_id}", flush=True)
        
        set_progress(_PROGRESS_WEIGHTS["floor_end"] - 1)
        # -------------------------------------------------------------
        # ✅ 2. UI: CLASSIFICATION -> DOAR BLUEPRINTS
        # -------------------------------------------------------------
        print("📢 [UI] Sending Classified BLUEPRINTS ONLY...")
        for seg_result in seg_results:
            bp_dir = seg_result.work_dir / "classified" / "blueprints"
            
            # Iterăm EXCLUSIV folderul blueprints
            if bp_dir.exists():
                for img in bp_dir.glob("*.*"):
                    notify_ui("classification", img)

    set_progress(_PROGRESS_WEIGHTS["floor_end"])
    pipeline_timer.add_step("Multi-Segmentation", t.end_time - t.start_time)

    # Colectăm rezultatele
    all_cls_results = get_all_classification_results(seg_results)
    plans = [ClassifiedPlanInfo(job_root, r.image_path, r.label) for r in all_cls_results]
    
    # Debug output
    print(f"\n📊 CLASSIFICATION SUMMARY:")
    print(f"   Total plans detected: {len(plans)}")
    from collections import Counter, defaultdict
    label_counts = Counter(p.label for p in plans)
    for label, count in label_counts.items():
        print(f"   {label}: {count}")

    # Blueprint-uri finale (doar house_blueprint) – verificare număr etaje imediat după clasificare
    house_plans = [p for p in plans if p.label == "house_blueprint"]
    print(f"\n🏠 HOUSE PLANS ONLY: {len(house_plans)}/{len(plans)}")
    if len(house_plans) < len(plans):
        print(f"⚠️  {len(plans) - len(house_plans)} plans EXCLUDED (not house_blueprint)\n")

    # Progress: greutăți fixe (seg_end, floor_end, detections_end, raster 8→58, scale→pdf)
    num_plans = len(house_plans)
    set_progress(_PROGRESS_WEIGHTS["detections_end"] - 1)  # după clasificare, înainte de floor/detections

    # ---------- Verificare număr etaje vs formular (chiar după ce știm ce e blueprint) ----------
    frontend_data = load_frontend_data_for_run(job_root.name, job_root)
    roof_only_offer = bool(frontend_data.get("roof_only_offer"))
    measurements_only_offer = bool(frontend_data.get("measurements_only_offer"))
    wizard_package = str(frontend_data.get("wizard_package") or "").strip().lower()
    is_aufstockung_offer = wizard_package in ("aufstockung", "zubau", "zubau_aufstockung")
    our_floors = len(house_plans)
    floors_number = frontend_data.get("floorsNumber")
    basement = frontend_data.get("basement", False)
    structura = frontend_data.get("structuraCladirii", {})
    tip_fundatie_beci = (structura.get("tipFundatieBeci") or "")
    has_basement_form = basement or (
        tip_fundatie_beci and "Keller" in str(tip_fundatie_beci) and "Kein Keller" not in str(tip_fundatie_beci)
    )
    lista_etaje = structura.get("listaEtaje")
    # Mansardă locuibilă (Wohnfläche): Dachgeschoss ohne Kniestock sau mit Kniestock = etaj suplimentar. Pod nu e etaj.
    if isinstance(lista_etaje, list):
        has_mansarda_wohnflaeche = "mansarda_mit" in lista_etaje or "mansarda_ohne" in lista_etaje
    else:
        has_mansarda_wohnflaeche = False

    if floors_number is not None:
        user_expected = (
            int(floors_number)
            + (1 if has_basement_form else 0)
            + (1 if has_mansarda_wohnflaeche else 0)
        )
    else:
        # listaEtaje: "intermediar" = etaj cu plan separat; mansardă (ohne/mit Kniestock, Wohnfläche) = etaj.
        if isinstance(lista_etaje, list):
            etaje_intermediare = sum(1 for e in lista_etaje if e == "intermediar")
            user_expected = (
                1 + etaje_intermediare
                + (1 if has_basement_form else 0)
                + (1 if has_mansarda_wohnflaeche else 0)
            )
        elif isinstance(lista_etaje, dict):
            user_expected = 1 + (1 if has_basement_form else 0)
        else:
            user_expected = 1 + (1 if has_basement_form else 0)
    _extra_suffix = ""
    if isinstance(lista_etaje, list) and has_mansarda_wohnflaeche:
        _extra_suffix += ", Dachgeschoss (Wohnfläche)"
    print(f"   📋 Formular: etaje așteptate (cu beci{_extra_suffix}) = {user_expected}  |  Planuri blueprint = {our_floors}")
    if not roof_only_offer and not measurements_only_offer and our_floors != user_expected:
        print(f"\n⛔ Număr etaje: plan încărcat={our_floors}, formular (cu beci)={user_expected}")
        print(">>> ERROR: Numărul de etaje din planul încărcat nu coincide cu cel ales din formular.")
        sys.exit(1)

    # =========================================================
    # REST OF PIPELINE
    # =========================================================

    with Timer("STEP 4: Floor Classification") as t:
        floor_results = run_floor_classification(job_root, plans)
        notify_ui("floor_classification")
    set_progress(_PROGRESS_WEIGHTS["floor_end"])
    pipeline_timer.add_step("Floor Classification", t.end_time - t.start_time)

    if house_plans:
        # ---------- Ordine etaje (de jos în sus) din etichetele Gemini; beci = primul din ordine ----------
        order_from_bottom = None
        with Timer("STEP 4c: Floor order from labels (Gemini)") as _t:
            order_from_bottom = run_floor_order_from_gemini(house_plans)
        pipeline_timer.add_step("Floor order (Gemini)", _t.end_time - _t.start_time)

        # ---------- Dacă nu avem ordine de la Gemini și userul a ales beci: scor Gemini pentru a alege beciul ----------
        basement_plan_index = None
        if order_from_bottom is not None and has_basement_form and our_floors > 0:
            basement_plan_index = order_from_bottom[0]
            print(f"   💾 Beci din ordine Gemini: plan index {basement_plan_index}")
        elif has_basement_form and our_floors > 0:
            with Timer("STEP 4b: Basement scoring (Gemini)") as _t:
                basement_plan_index = run_basement_scoring(house_plans)
            pipeline_timer.add_step("Basement scoring", _t.end_time - _t.start_time)

        # ---------- Creare run și salvare floor_order / basement_plan_id ----------
        run_id = _create_run_for_detections(job_root, house_plans)
        run_dir = get_run_dir(run_id)
        if order_from_bottom is not None:
            (run_dir / "floor_order.json").write_text(
                json.dumps({"order_from_bottom": order_from_bottom}, indent=2),
                encoding="utf-8",
            )
            print(f"   💾 Salvat: {run_dir / 'floor_order.json'}")
        if basement_plan_index is not None:
            (run_dir / "basement_plan_id.json").write_text(
                json.dumps({"basement_plan_index": basement_plan_index}, indent=2),
                encoding="utf-8",
            )
            print(f"   💾 Salvat: {run_dir / 'basement_plan_id.json'} (beci = plan index {basement_plan_index})")
        
        with Timer("STEP 5: Detections (skipped — no Roboflow)") as t:
            print("📌 [Pipeline] Roboflow / object_crops dezactivate (editor manual).", flush=True)
        set_progress(_PROGRESS_WEIGHTS["detections_end"])
        pipeline_timer.add_step("Detections (skipped)", t.end_time - t.start_time)

        raster_timings: List[tuple] = []
        with Timer("STEP 5.5: Manual blueprint + editor + walls_from_coords") as t:
            print(f"\n{'='*60}")
            print(f"[ManualBlueprint] Run {run_id}: fără Raster API / brute force", flush=True)
            print(f"{'='*60}\n")

            plans = load_plan_infos(run_id, "scale")
            raster_scan_failed = False
            set_progress(_PROGRESS_WEIGHTS["raster_start"])

            for plan in plans:
                try:
                    prepare_manual_blueprint_workspace(plan)
                except Exception as e:
                    print(f"❌ [ManualBlueprint] {plan.plan_id}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    raster_scan_failed = True

            ensure_roof_3d_floor_manifest(run_id, [p.plan_id for p in plans])

            review_paths: List[Path] = []
            for plan in plans:
                raster_dir = plan.stage_work_dir / "cubicasa_steps" / "raster"
                path_base, path_rooms, path_doors = save_detections_review_image(raster_dir)
                if path_base:
                    review_paths.append(path_base)
                if path_rooms:
                    review_paths.append(path_rooms)
                if path_doors:
                    review_paths.append(path_doors)
                dem0 = raster_dir / "detections_review_demolitions.png"
                st0 = raster_dir / "detections_review_stairs.png"
                if dem0.exists():
                    review_paths.append(dem0)
                if st0.exists():
                    review_paths.append(st0)

            if not raster_scan_failed and review_paths:
                run_dir = get_run_dir(run_id)
                order_from_bottom = None
                basement_plan_index = None
                if (run_dir / "floor_order.json").exists():
                    try:
                        fo = json.loads((run_dir / "floor_order.json").read_text(encoding="utf-8"))
                        order_from_bottom = fo.get("order_from_bottom")
                    except Exception:
                        pass
                if (run_dir / "basement_plan_id.json").exists():
                    try:
                        bp = json.loads((run_dir / "basement_plan_id.json").read_text(encoding="utf-8"))
                        basement_plan_index = bp.get("basement_plan_index")
                    except Exception:
                        pass
                n_plans = len(plans)
                if order_from_bottom is not None and len(order_from_bottom) == n_plans:
                    labels_from_bottom = []
                    if basement_plan_index is not None:
                        labels_from_bottom.append("Keller")
                    labels_from_bottom.append("Erdgeschoss")
                    for k in range(1, n_plans - (1 if basement_plan_index is not None else 0)):
                        labels_from_bottom.append(f"{k}. Obergeschoss")
                    plan_index_to_label = {order_from_bottom[pos]: labels_from_bottom[pos] for pos in range(n_plans)}
                    floor_labels = [plan_index_to_label[i] for i in range(n_plans)]
                    try:
                        (job_root / "detections_review_floor_labels.json").write_text(
                            json.dumps(floor_labels, ensure_ascii=False), encoding="utf-8"
                        )
                        print(f"   💾 Etichete etaje pentru UI: {floor_labels}")
                    except Exception as e:
                        print(f"   ⚠️ Nu s-a putut scrie detections_review_floor_labels.json: {e}")

                notify_ui_batch("detections_review", review_paths)
                # UI unificat: la Confirm se scriu detections_review_approved.flag și roof_review_approved.flag.
                _wait_detections_review_approval(job_root, set_progress)

                for plan in plans:
                    raster_dir = plan.stage_work_dir / "cubicasa_steps" / "raster"
                    apply_detections_edited(raster_dir)

                review_paths_after_edit: List[Path] = []
                for plan in plans:
                    raster_dir = plan.stage_work_dir / "cubicasa_steps" / "raster"
                    pb, pr, pd = save_detections_review_image(
                        raster_dir, prefer_edited_room_polygons=True
                    )
                    if pb:
                        review_paths_after_edit.append(pb)
                    if pr:
                        review_paths_after_edit.append(pr)
                    if pd:
                        review_paths_after_edit.append(pd)
                    dem1 = raster_dir / "detections_review_demolitions.png"
                    st1 = raster_dir / "detections_review_stairs.png"
                    if dem1.exists():
                        review_paths_after_edit.append(dem1)
                    if st1.exists():
                        review_paths_after_edit.append(st1)
                if review_paths_after_edit:
                    notify_ui_batch("detections_review", review_paths_after_edit)

                if roof_only_offer:
                    seeded = seed_roof_only_rooms_from_roof_polygons(run_id, plans)
                    if seeded:
                        print("🔸 [Pipeline] roof_only_offer: 09_interior + rooms din poligoanele roof editor.", flush=True)
                        for p in plans:
                            print(f"   - {p.plan_id}: {seeded.get(p.plan_id, 0)} poligoane", flush=True)
                    else:
                        print(
                            "⚠️ [Pipeline] roof_only_offer: nu am găsit poligoane roof editor pentru seed 09_interior; continuăm cu fallback-urile existente.",
                            flush=True,
                        )

                max_w = max(1, min(3, len(plans)))

                def _walls_one(p):
                    return run_walls_pipeline_after_manual_editor(p, roof_only_offer)

                set_progress(55.5)
                with ThreadPoolExecutor(max_workers=max_w) as ex:
                    results = list(ex.map(_walls_one, plans))

                if not all(results):
                    if roof_only_offer:
                        print(
                            "🔸 [Pipeline] roof_only_offer: walls_from_coords + room_scales (Gemini) eșuat pe unele planuri; fallback sintetic per plan.",
                            flush=True,
                        )
                        fallback_failed = False
                        for plan, ok in zip(plans, results):
                            if ok:
                                continue
                            if not apply_roof_only_synthetic_walls_and_scale(plan):
                                fallback_failed = True
                        raster_scan_failed = fallback_failed
                    else:
                        raster_scan_failed = True
                set_progress(56.5)

            set_progress(float(_PROGRESS_WEIGHTS["raster_end"] - 1))

            raster_ok, failed_plan_ids = _check_raster_complete(plans, job_root)
            if failed_plan_ids:
                print(
                    f"⚠️ [ManualBlueprint] Planuri incomplete (room_scales.json + rooms.png): {failed_plan_ids}",
                    flush=True,
                )
            raster_ok = raster_ok and not raster_scan_failed

            print(f"\n{'='*60}")
            print(f"[ManualBlueprint] Complete")
            print(f"{'='*60}\n")
        pipeline_timer.add_step("Manual blueprint + walls", t.end_time - t.start_time)
        
        if not raster_ok:
            if is_aufstockung_offer:
                print(
                    "\n⚠️ [Pipeline] Aufstockung: raster completeness check bypassed (missing room_scales/rooms on some plans).",
                    flush=True,
                )
                print(
                    "   Continuăm pipeline-ul conform workflow-ului Aufstockung (Scale/Count/Roof/Pricing/Offer/PDF).",
                    flush=True,
                )
                raster_ok = True
            else:
                print(f"\n⛔ [Pipeline] Camerele nu sunt calculate pentru toate planurile.", flush=True)
                print(f"   Se sar pașii: Scale, Count Objects, Roof, Pricing, Offer, PDF.", flush=True)
                print(f"   Asigură-te că RasterScan finalizează (room_scales.json + rooms.png) pentru fiecare etaj în ordine, apoi rulează din nou.\n", flush=True)
        
        if raster_ok:
            set_progress(_PROGRESS_WEIGHTS["raster_end"])
            with Timer("STEP 6: Scale") as t:
                run_scale_detection_for_run(run_id)
            set_progress(_PROGRESS_WEIGHTS["scale_end"])
            pipeline_timer.add_step("Scale", t.end_time - t.start_time)

            # STEP 7 (Count Objects) și STEP 13 (Roof) rulează în PARALEL
            set_progress(_PROGRESS_WEIGHTS["scale_end"] + 1)
            def _run_count_objects_timed():
                _start = time.time()
                run_count_objects_for_run(run_id)
                return time.time() - _start
            def _run_roof_timed():
                _start = time.time()
                run_roof_for_run(run_id, notify_ui_events=True)
                return time.time() - _start
            with Timer("STEP 7 + 13: Count Objects || Roof (parallel)") as t_par:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    fut_co = executor.submit(_run_count_objects_timed)
                    fut_roof = executor.submit(_run_roof_timed)
                    t_count_objects = fut_co.result()
                    t_roof = fut_roof.result()
            pipeline_timer.add_step("Count Objects", t_count_objects)
            pipeline_timer.add_step("Roof", t_roof)
            set_progress(_PROGRESS_WEIGHTS["count_roof_end"])

            frontend_data = load_frontend_data_for_run(run_id, job_root)
            frontend_data = _enrich_frontend_with_aufstockung_phase1(frontend_data, job_root)

            set_progress(_PROGRESS_WEIGHTS["pricing_end"] - 2)
            with Timer("STEP 14-15: Pricing") as t:
                pricing_results = run_pricing_for_run(run_id, frontend_data_override=frontend_data)
                notify_ui("pricing")
            set_progress(_PROGRESS_WEIGHTS["pricing_end"])
            pipeline_timer.add_step("Pricing", t.end_time - t.start_time)

            set_progress(_PROGRESS_WEIGHTS["offer_end"] - 2)
            with Timer("STEP 16-17: Offer Generation") as t:
                mf = frontend_data.get("materialeFinisaj") or {}
                sc = frontend_data.get("sistemConstructiv") or {}
                nivel_oferta = sc.get("nivelOferta") or mf.get("nivelOferta")
                if not nivel_oferta:
                    cm = (frontend_data.get("calc_mode") or "").lower()
                    if cm in ("structure", "structura"):
                        nivel_oferta = "Structură"
                    elif cm in ("structure_windows", "structura_ferestre", "structura+ferestre"):
                        nivel_oferta = "Structură + ferestre"
                    elif cm in ("full_house", "full_house_premium", "casa_completa", "casacompleta"):
                        nivel_oferta = "Casă completă"
                    else:
                        nivel_oferta = "Structură"
                print(f"📋 [OFFER] Level: {nivel_oferta}")
                for res in pricing_results:
                    if res.success and res.result_data:
                        try:
                            build_final_offer(
                                res.result_data,
                                nivel_oferta,
                                res.work_dir / "final_offer.json"
                            )
                        except Exception as e:
                            print(f"⚠️ Error building offer for {res.plan_id}: {e}")
                notify_ui("offer_generation")
            set_progress(_PROGRESS_WEIGHTS["offer_end"])
            pipeline_timer.add_step("Offer Generation", t.end_time - t.start_time)

            wait_for_offer_number_in_job(job_root)
            frontend_data = load_frontend_data_for_run(run_id, job_root)
            frontend_data = _enrich_frontend_with_aufstockung_phase1(frontend_data, job_root)

            with Timer("STEP 18-19: PDF GENERATION") as t:
                pdf_path = None
                admin_pdf_path = None
                calc_method_pdf_path = None
                roof_pdf_path = None
                measurements_only = bool(frontend_data.get("measurements_only_offer"))
                try:
                    # Generează roof_pricing.json înainte de PDF (pentru ofertă cu prețuri acoperiș)
                    try:
                        generate_roof_pricing(run_id=run_id, frontend_data=frontend_data)
                    except Exception as rp_err:
                        print(f"⚠️ Roof pricing Error: {rp_err}")
                    set_progress(_PROGRESS_WEIGHTS["pdf_end"] - 12)  # început PDF
                    if measurements_only:
                        print(
                            "🔸 [PDF] measurements_only_offer: skip Kunden-/Admin-/Berechnungs-PDF; nur Mengen-/Maß-PDF.",
                            flush=True,
                        )
                    else:
                        pdf_path = generate_complete_offer_pdf(run_id=run_id, output_path=None, job_root=job_root)
                        print(f"✅ [PDF] User PDF generated: {pdf_path}")
                        notify_ui("pdf_generation", pdf_path)
                        set_progress(_PROGRESS_WEIGHTS["pdf_end"] - 8)
                        admin_pdf_path = generate_admin_offer_pdf(run_id=run_id, output_path=None, job_root=job_root)
                        print(f"✅ [PDF] Admin PDF generated: {admin_pdf_path}")
                        notify_ui("pdf_generation", admin_pdf_path)
                        set_progress(_PROGRESS_WEIGHTS["pdf_end"] - 4)
                        if not bool(frontend_data.get("roof_only_offer")):
                            try:
                                calc_method_pdf_path = generate_admin_calculation_method_pdf(run_id=run_id, output_path=None, job_root=job_root)
                                print(f"✅ [PDF] Calculation Method PDF generated: {calc_method_pdf_path}")
                                notify_ui("pdf_generation", calc_method_pdf_path)
                            except Exception as calc_err:
                                print(f"⚠️ Calculation Method PDF Error: {calc_err}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print("🔸 [PDF] roof_only_offer: skip calculation method PDF (full-house doc)")
                except Exception as e:
                    print(f"⚠️ PDF Error: {e}")
                    import traceback
                    traceback.print_exc()
                    notify_ui("pdf_generation")
                # Roof measurements PDF must be attempted even if user/admin PDF failed.
                try:
                    roof_pdf_path = generate_roof_measurements_pdf(run_id=run_id, output_path=None)
                    if roof_pdf_path:
                        notify_ui("pdf_generation", roof_pdf_path)
                        print(f"✅ [PDF] Roof measurements PDF generated: {roof_pdf_path}")
                except Exception as roof_pdf_err:
                    print(f"⚠️ [PDF] Roof measurements PDF Error: {roof_pdf_err}")
            pipeline_timer.add_step("PDF", t.end_time - t.start_time)

            time.sleep(2.0)
            set_progress(100)
            notify_ui("computation_complete", pdf_path if pdf_path and Path(pdf_path).exists() else None)

        pipeline_timer.finish()
    return job_root, plans, floor_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--no-classification", action="store_true")
    
    args = parser.parse_args()
    frontend_data_json = os.environ.get('FRONTEND_DATA_JSON')
    
    if frontend_data_json:
        print(f"✅ Frontend data received from ENV ({len(frontend_data_json)} chars)")
    else:
        print("⚠️  No frontend data in ENV")
    
    # Segmentare = doar Gemini Crop (coordonate în procente → crop → blueprints/side_views).
    # Pipeline-ul vechi (detect_wall_zones, detect_clusters, classifier) este dezactivat.
    if args.no_classification:
        job_root = build_job_root(job_id=args.job_id, prefix="segmentation_job")
        segmentation_out_base = job_root / "segmentation"
        segmentation_out_base.mkdir(parents=True, exist_ok=True)
        run_segmentation_for_documents(
            input_path=args.input,
            output_base_dir=segmentation_out_base,
            max_workers=None,
        )
    else:
        run_segmentation_and_classification_for_document(
            args.input, 
            job_id=args.job_id,
            frontend_data_json=frontend_data_json
        )