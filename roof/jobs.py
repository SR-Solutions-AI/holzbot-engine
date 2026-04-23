# new/runner/roof/jobs.py
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from config.settings import (
    JOBS_ROOT,
    OUTPUT_ROOT,
    RUNS_ROOT,
    load_plan_infos,
    PlansListError,
    PlanInfo,
)
import cv2
import numpy as np
from config.frontend_loader import load_frontend_data_for_run
from roof.insulated_area import split_mov_extra_roof_border_flood

STAGE_NAME = "roof"

# holzbot-roof e sibling de holzbot-engine (holzbot-dynamic/holzbot-roof, holzbot-engine)
_ENGINE_ROOT = Path(__file__).resolve().parents[1]  # holzbot-engine root
_ROOF_ROOT = _ENGINE_ROOT.parent / "holzbot-roof" 


@dataclass
class RoofJobResult:
    plan_id: str
    work_dir: Path
    success: bool
    message: str
    result_data: dict | None = None


def _load_floor_metadata(job_root: Path, original_name: str, run_dir: Path | None = None, plan_id: str | None = None) -> dict | None:
    """Încarcă metadata pentru a determina floor_type. Încearcă job_root, apoi run_dir (dacă dat).
    Caută mai întâi după plan_id (ex: plan_01_cluster_1), apoi după original_name (stem, ex: cluster_1)."""
    names_to_try = []
    if plan_id:
        names_to_try.append(plan_id)
    if original_name and original_name not in names_to_try:
        names_to_try.append(original_name)
    if not names_to_try:
        names_to_try = [original_name]
    for base in (job_root, run_dir) if run_dir is not None else (job_root,):
        if base is None:
            continue
        for name in names_to_try:
            metadata_file = base / "plan_metadata" / f"{name}.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
    return None


# Ordine pentru sortare: cel mai mic index = beci (dacă există), apoi parter, apoi etaj(e).
# Fără beci: parter = 0, etaj = 1, ...  Cu beci: beci = 0, parter = 1, etaj = 2, ...
_FLOOR_ORDER = {"ground_floor": 0, "intermediate": 1, "top_floor": 2, "unknown": 3}


def _wall_mask_area(run_id: str, plan: PlanInfo) -> int:
    """Aria măștii de pereți în pixeli (pentru fallback: parter = suprafață mai mare). Returnează 0 dacă fișierul lipsește."""
    path = OUTPUT_ROOT / run_id / "scale" / plan.plan_id / "cubicasa_steps" / "raster_processing" / "walls_from_coords" / "01_walls_from_coords.png"
    if not path.exists():
        return 0
    try:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0
        return int((img > 0).sum())
    except Exception:
        return 0


def _floor_type_rank(plan: PlanInfo, job_root: Path, run_dir: Path | None = None) -> int:
    """Rang din clasificare: 0 = parter (ground), 1 = intermediar, 2 = top, 3 = unknown."""
    stem = getattr(plan.plan_image, "stem", None) or (plan.plan_id.split("_", 2)[-1] if "_" in plan.plan_id else plan.plan_id)
    meta = _load_floor_metadata(job_root, stem, run_dir, plan_id=plan.plan_id)
    if not meta:
        return _FLOOR_ORDER["unknown"]
    ft = (meta.get("floor_classification") or {}).get("floor_type", "unknown").lower()
    return _FLOOR_ORDER.get(ft, _FLOOR_ORDER["unknown"])


def _floor_sort_key(
    plan: PlanInfo,
    job_root: Path,
    basement_plan_id: str | None,
    run_id: str,
    run_dir: Path | None,
) -> tuple:
    """
    Cheie sortare: (primar, -arie).
    Primar: beci=0, parter=1, intermediar=2, top=3, unknown=4.
    La egalitate, planul cu suprafață mai mare (parter tipic) vine primul.
    """
    if basement_plan_id and plan.plan_id == basement_plan_id:
        return (0, 0)
    rank = _floor_type_rank(plan, job_root, run_dir)
    area = _wall_mask_area(run_id, plan)
    return (1 + rank, -area)


def _run_for_single_plan(
    run_id: str,
    index: int,
    total: int,
    plan: PlanInfo,
    frontend_data: dict | None,
    total_floors: int,
    job_root: Path,
) -> RoofJobResult:
    """
    Scrie roof_estimation.json cu preț fix 10 EUR pentru top floor, 0 pentru altele.
    """
    work_dir = plan.stage_work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    original_name = plan.plan_image.stem
    metadata = _load_floor_metadata(job_root, original_name, plan_id=plan.plan_id)

    is_top_floor = False
    floor_type = "unknown"

    if metadata:
        floor_class = metadata.get("floor_classification", {})
        floor_type = floor_class.get("floor_type", "unknown").lower()
        is_top_floor = any(keyword in floor_type for keyword in ["top", "roof", "attic", "mansarda"])

    # Fallback: când nu avem metadata (ex: plan_metadata lipsă sau run vechi), ultimul etaj = top floor
    if not metadata and total_floors > 1 and index == total - 1:
        print(f"       🔥 FALLBACK ROOF: Fără plan_metadata – consider ultimul etaj ({plan.plan_id}) ca top floor.", flush=True)
        is_top_floor = True
        floor_type = "top_floor (fallback)"

    if total_floors == 1:
        if not is_top_floor:
            print(f"       🔥 FORCE ROOF: Single plan detected. Forcing is_top_floor=True.", flush=True)
        is_top_floor = True

    final_cost = 10.0 if is_top_floor else 0.0
    result = {
        "plan_id": plan.plan_id,
        "floor_type": floor_type,
        "is_top_floor": is_top_floor,
        "roof_area_sqm": 0.0,
        "roof_final_total_eur": final_cost,
        "inputs": {"roof_type": {"matched_name_de": "Acoperiș din holzbot-roof"}},
    }

    output_file = work_dir / "roof_estimation.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    msg = f"Preț fix: {final_cost:,.0f} EUR" if is_top_floor else f"Nu e top floor ({floor_type}) → 0 EUR"
    print(f"[{STAGE_NAME}] ({index}/{total}) {plan.plan_id} → {msg}", flush=True)
    return RoofJobResult(
        plan_id=plan.plan_id,
        work_dir=work_dir,
        success=True,
        message=msg,
        result_data=result,
    )

def _collect_side_views(job_root: Path) -> List[Path]:
    """Colectează toate imaginile side_view din segmentare."""
    out: List[Path] = []
    seg_base = job_root / "segmentation"
    if not seg_base.is_dir():
        return out
    for sv_dir in seg_base.glob("src_*/classified/side_views"):
        if sv_dir.is_dir():
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                out.extend(sv_dir.glob(ext))
    return sorted(out)


def _run_roof_3d_workflow(
    run_id: str,
    plans: List[PlanInfo],
    floor_roof_types: dict | None = None,
    floor_roof_angles: dict | None = None,
    basement_floor_index_override: int | None = None,
    edited_rectangles_json_path: Path | None = None,
) -> Path | None:
    """
    Rulează holzbot-roof clean_workflow cu wall masks din toate etajele.
    Input: scale/<plan_id>/cubicasa_steps/raster_processing/walls_from_coords/01_walls_from_coords.png
    Returnează directorul de output (roof_3d) sau None dacă nu s-au găsit măști.
    """
    import subprocess

    out_root = OUTPUT_ROOT / run_id
    scale_root = out_root / "scale"
    wall_mask_name = "01_walls_from_coords.png"
    subpath = ["cubicasa_steps", "raster_processing", "walls_from_coords", wall_mask_name]

    paths: List[Path] = []
    for plan in plans:
        p = scale_root / plan.plan_id / Path(*subpath)
        if p.exists():
            paths.append(p)

    if not paths:
        print(f"       ⚠️ [{STAGE_NAME}] Nu s-au găsit wall masks în scale/*/cubicasa_steps/.../{wall_mask_name}", flush=True)
        return None

    roof_3d_dir = out_root / "roof" / "roof_3d"
    roof_3d_dir.mkdir(parents=True, exist_ok=True)
    # Persist explicit mapping floor index -> plan_id used by roof workflow.
    # UI/API must use this mapping instead of assuming raster plan order == floor_X order.
    try:
        floor_plan_ids = {str(i): p.plan_id for i, p in enumerate(plans)}
        (roof_3d_dir / "roof_floor_plan_ids.json").write_text(
            json.dumps(floor_plan_ids, indent=2), encoding="utf-8"
        )
    except Exception as e:
        print(f"       ⚠️ [roof] Nu pot scrie roof_floor_plan_ids.json: {e}", flush=True)
    if floor_roof_types is not None:
        frt_path = roof_3d_dir / "floor_roof_types.json"
        try:
            content = json.dumps({str(k): v for k, v in floor_roof_types.items()}, indent=0)
            frt_path.write_text(content, encoding="utf-8")
            print(f"       [roof] Scris {frt_path.name}: {content!r}", flush=True)
        except Exception as e:
            print(f"       ⚠️ [roof] Eroare la scrierea {frt_path}: {e}", flush=True)
    else:
        print(f"       [roof] Nu s-au scris tipuri (floor_roof_types gol).", flush=True)
    # Index etaj beci (fără acoperiș) – folosit de workflow pentru a nu genera acoperiș pentru acel etaj
    run_dir = RUNS_ROOT / run_id
    idx = basement_floor_index_override
    if idx is None:
        basement_plan_id_path = run_dir / "basement_plan_id.json"
        if basement_plan_id_path.exists():
            try:
                data = json.loads(basement_plan_id_path.read_text(encoding="utf-8"))
                idx = data.get("basement_plan_index")
            except Exception:
                pass
    if idx is not None:
        bfi_path = roof_3d_dir / "basement_floor_index.json"
        bfi_path.write_text(json.dumps({"basement_floor_index": idx}, indent=0), encoding="utf-8")
        print(f"       [roof] Scris {bfi_path.name}: beci = etaj {idx}", flush=True)
    if floor_roof_angles:
        fra_path = roof_3d_dir / "floor_roof_angles.json"
        try:
            content = json.dumps({str(k): float(v) for k, v in floor_roof_angles.items()}, indent=0)
            fra_path.write_text(content, encoding="utf-8")
            print(f"       [roof] Scris {fra_path.name}: {content!r}", flush=True)
        except Exception as e:
            print(f"       ⚠️ [roof] Eroare la scrierea {fra_path}: {e}", flush=True)
    else:
        print(f"       [roof] Nu s-au scris unghiuri (floor_roof_angles gol).", flush=True)

    if len(paths) == 1:
        wall_input = str(paths[0])
        try:
            rp = Path(wall_input).resolve()
            print(f"       [roof] Input mask (algorithm): {rp}", flush=True)
            print(f"       [roof] Input mask URL: file://{rp}", flush=True)
        except Exception:
            pass
    else:
        tmp = tempfile.mkdtemp(prefix="holzbot_roof_floors_")
        try:
            floors_meta = []
            for i, (src, plan) in enumerate(zip(paths, plans)):
                dst = Path(tmp) / f"floor_{i:02d}.png"
                shutil.copy2(src, dst)
                try:
                    print(
                        f"       [roof] Input mask floor_{i:02d} ({plan.plan_id}): {dst.resolve()}",
                        flush=True,
                    )
                    print(f"       [roof] Input mask URL floor_{i:02d}: file://{dst.resolve()}", flush=True)
                except Exception:
                    pass
                interior_path = src.parent / "09_interior.png"
                floors_meta.append({
                    "floor_path": dst.name,
                    "plan_id": plan.plan_id,
                    "interior_mask_path": str(interior_path.resolve()) if interior_path.exists() else None,
                })
            (Path(tmp) / "floors_meta.json").write_text(
                json.dumps(floors_meta, indent=0), encoding="utf-8"
            )
            wall_input = tmp
            try:
                print(f"       [roof] Input folder (algorithm): {Path(wall_input).resolve()}", flush=True)
                print(f"       [roof] Input folder URL: file://{Path(wall_input).resolve()}", flush=True)
            except Exception:
                pass
        except Exception as e:
            print(f"       ⚠️ [{STAGE_NAME}] Eroare la pregătirea etajelor: {e}", flush=True)
            if os.path.exists(tmp):
                shutil.rmtree(tmp, ignore_errors=True)
            return None

    script_path = _ROOF_ROOT / "scripts" / "clean_workflow.py"
    if not script_path.exists():
        print(f"       ⚠️ [{STAGE_NAME}] holzbot-roof/scripts/clean_workflow.py nu există.", flush=True)
        if len(paths) > 1 and os.path.exists(wall_input):
            shutil.rmtree(wall_input, ignore_errors=True)
        return None

    # holzbot-roof/roof_calc uses bytecode from _orig_pyc (Python 3.14); prefer venv cu deps
    _venv_py = _ROOF_ROOT / ".venv314" / "bin" / "python"
    _python = str(_venv_py) if _venv_py.exists() else (shutil.which("python3.14") or sys.executable)

    try:
        # Default: doar rectangles/floor_*; fără roof_types 3D, entire/*, unfold (vezi clean_workflow --rectangles-only).
        _rect_only = os.environ.get("HOLZBOT_ROOF_RECTANGLES_ONLY", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "",
        )
        cmd = [_python, str(script_path)]
        if _rect_only:
            cmd.append("--rectangles-only")
        if edited_rectangles_json_path and edited_rectangles_json_path.exists():
            cmd.append(f"--edited-json={str(edited_rectangles_json_path)}")
        cmd.extend([wall_input, str(roof_3d_dir)])
        result = subprocess.run(
            cmd,
            cwd=str(_ROOF_ROOT),
            capture_output=False,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"       ⚠️ [{STAGE_NAME}] clean_workflow a returnat cod {result.returncode}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"       ⚠️ [{STAGE_NAME}] clean_workflow timeout (300s)", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"       ⚠️ [{STAGE_NAME}] Eroare la clean_workflow: {e}", flush=True)
    finally:
        if len(paths) > 1 and os.path.exists(wall_input):
            shutil.rmtree(wall_input, ignore_errors=True)

    return roof_3d_dir


def _editor_review_dimensions(run_id: str, plan: PlanInfo) -> tuple[int, int] | None:
    """
    Dimensiunile din detections_review_data.json — același spațiu de coordonate ca în RoofReviewEditor
    (GET roof-review-data / canvas). Nu coincid cu 01_walls_from_coords.png (rezoluție nativă).
    """
    p = OUTPUT_ROOT / run_id / "scale" / plan.plan_id / "cubicasa_steps" / "raster" / "detections_review_data.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        ew = int(data.get("imageWidth") or 0)
        eh = int(data.get("imageHeight") or 0)
        if ew > 0 and eh > 0:
            return (ew, eh)
    except Exception:
        pass
    return None


def _meters_per_pixel_for_plan(run_id: str, plan: PlanInfo) -> float | None:
    scale_dir = OUTPUT_ROOT / run_id / "scale" / plan.plan_id
    scale_result = scale_dir / "scale_result.json"
    if scale_result.exists():
        try:
            data = json.loads(scale_result.read_text(encoding="utf-8"))
            mpp = data.get("meters_per_pixel")
            if isinstance(mpp, (int, float)) and float(mpp) > 0:
                return float(mpp)
        except Exception:
            pass
    room_scales = scale_dir / "cubicasa_steps" / "raster_processing" / "walls_from_coords" / "room_scales.json"
    if room_scales.exists():
        try:
            data = json.loads(room_scales.read_text(encoding="utf-8"))
            mpp = data.get("m_px") or data.get("weighted_average_m_px")
            if isinstance(mpp, (int, float)) and float(mpp) > 0:
                return float(mpp)
        except Exception:
            pass
    return None


# Culori BGR aliniate cu visualize_individual_rectangles (holzbot-roof): fill semi-transparent peste masca de pereți.
_ROOF_SECTION_COLORS_BGR: List[tuple[int, int, int]] = [
    (255, 100, 100),  # RGB echivalent ~ (100,100,255)
    (100, 255, 100),
    (100, 100, 255),
    (100, 255, 255),
    (255, 255, 100),
    (255, 100, 255),
]


_ROOF_TYPE_IDS = frozenset({"0_w", "1_w", "2_w", "4_w", "4.5_w"})
_DEFAULT_ROOF_ANGLE_DEG = 16.0
_DEFAULT_ROOF_TYPE = "2_w"


def _polygon_area_xy(pts: list[list[float]]) -> float:
    """Arie poligon în coordonate plane (shoelace)."""
    if len(pts) < 3:
        return 0.0
    s = 0.0
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        s += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(s) * 0.5


def _parse_plan_id_from_raster_dir(raster_dir: str) -> str:
    """Extrage plan_id dintr-o cale .../scale/<plan_id>/cubicasa_steps/raster (fallback pentru manifest)."""
    parts = Path(str(raster_dir).strip()).parts
    for i, part in enumerate(parts):
        if str(part).startswith("plan_") and i + 1 < len(parts) and parts[i + 1] == "cubicasa_steps":
            return str(part)
    for part in reversed(parts):
        if str(part).startswith("plan_"):
            return str(part)
    return ""


def manifest_tab_plan_ids_order(run_id: str) -> list[str] | None:
    """
    Ordinea plan_id din detections_review_manifest.json, cu același dedupe ca GET roof-review-data (Nest).
    Indexul k = cheia JSON din roof_rectangles_edited când _floor_key_scheme == \"manifest\".
    """
    man_path = JOBS_ROOT / run_id / "detections_review_manifest.json"
    if not man_path.exists():
        return None
    try:
        data = json.loads(man_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    raw_dirs = data.get("rasterDirs")
    raw_ids = data.get("rasterPlanIds")
    if not isinstance(raw_dirs, list) or not raw_dirs:
        return None
    seen: set[str] = set()
    out: list[str] = []
    for i, d in enumerate(raw_dirs):
        ds = str(d).strip()
        if not ds:
            continue
        norm = os.path.normpath(ds)
        if norm in seen:
            continue
        seen.add(norm)
        pid = ""
        if isinstance(raw_ids, list) and i < len(raw_ids):
            pid = str(raw_ids[i] or "").strip()
        if not pid:
            pid = _parse_plan_id_from_raster_dir(ds)
        if pid:
            out.append(pid)
    return out or None


def _normalize_roof_type(r: Any) -> str:
    if r is None:
        return _DEFAULT_ROOF_TYPE
    if isinstance(r, str):
        t = r.strip()
        if t in _ROOF_TYPE_IDS:
            return t
    return _DEFAULT_ROOF_TYPE


def _edited_entries_as_roof_indices(
    edited: dict,
    plans_list: List[PlanInfo],
    plans_roof: List[PlanInfo],
    *,
    manifest_tab_plan_ids: list[str] | None = None,
) -> list[tuple[int, list[Any]]]:
    """
    Chei din roof_rectangles_edited.json:
    - _floor_key_scheme == \"manifest\": k = index tab în ordinea din detections_review_manifest (GET roof-review-data),
      mapat la index plans_roof după plan_id. Dacă lipsește manifestul, fallback la plans_list[k] (comportament vechi).
    - altfel (legacy): k este deja index în plans_roof (comportament vechi API: plan_id → acel index).
    """
    scheme = edited.get("_floor_key_scheme")
    out: list[tuple[int, list[Any]]] = []
    for floor_key, rects in edited.items():
        sk = str(floor_key)
        if sk.startswith("_"):
            continue
        if not isinstance(rects, list):
            continue
        try:
            k = int(floor_key)
        except Exception:
            continue
        if scheme == "manifest":
            pid: str | None = None
            if manifest_tab_plan_ids is not None and 0 <= k < len(manifest_tab_plan_ids):
                pid = str(manifest_tab_plan_ids[k]).strip() or None
            if not pid and 0 <= k < len(plans_list):
                pid = plans_list[k].plan_id
            if not pid:
                continue
            roof_i = next((i for i, p in enumerate(plans_roof) if p.plan_id == pid), None)
            if roof_i is None:
                continue
            out.append((roof_i, rects))
        else:
            if k < 0 or k >= len(plans_roof):
                continue
            out.append((k, rects))
    return out


def derive_floor_roof_settings_from_edited(
    edited: dict,
    num_floors: int,
    basement_idx: Optional[int],
    plans_list: List[PlanInfo],
    plans_roof: List[PlanInfo],
    *,
    manifest_tab_plan_ids: list[str] | None = None,
) -> tuple[dict[int, float], dict[int, Optional[str]]]:
    """
    Din roof_rectangles_edited.json: unghi mediu ponderat (suprafață) și tip dominant (suprafață max)
    per etaj. Etaj beci: tip None, unghi 0 (neutilizat de workflow).
    """
    floor_angles: dict[int, float] = {}
    floor_types: dict[int, Optional[str]] = {}

    for floor_idx, rects in _edited_entries_as_roof_indices(
        edited, plans_list, plans_roof, manifest_tab_plan_ids=manifest_tab_plan_ids
    ):
        if floor_idx < 0 or floor_idx >= num_floors:
            continue
        if floor_idx == basement_idx:
            floor_types[floor_idx] = None
            floor_angles[floor_idx] = 0.0
            continue
        if not isinstance(rects, list) or not rects:
            floor_angles[floor_idx] = _DEFAULT_ROOF_ANGLE_DEG
            floor_types[floor_idx] = _DEFAULT_ROOF_TYPE
            continue

        best_area = -1.0
        best_type = _DEFAULT_ROOF_TYPE
        w_sum = 0.0
        w_ang = 0.0

        for r in rects:
            if not isinstance(r, dict):
                continue
            pts = r.get("points")
            if not isinstance(pts, list) or len(pts) < 3:
                continue
            arr: list[list[float]] = []
            for p in pts:
                if isinstance(p, list) and len(p) >= 2:
                    arr.append([float(p[0]), float(p[1])])
            if len(arr) < 3:
                continue
            area = _polygon_area_xy(arr)
            if area <= 0:
                continue
            try:
                ang_f = float(r.get("roofAngleDeg", _DEFAULT_ROOF_ANGLE_DEG))
            except Exception:
                ang_f = _DEFAULT_ROOF_ANGLE_DEG
            ang_f = float(max(0.0, min(60.0, ang_f)))
            rt = _normalize_roof_type(r.get("roofType"))
            w_sum += area
            w_ang += area * ang_f
            if area > best_area:
                best_area = area
                best_type = rt

        if w_sum > 0:
            floor_angles[floor_idx] = w_ang / w_sum
        else:
            floor_angles[floor_idx] = _DEFAULT_ROOF_ANGLE_DEG
        floor_types[floor_idx] = best_type if best_area > 0 else _DEFAULT_ROOF_TYPE

    for i in range(num_floors):
        if i == basement_idx:
            floor_types.setdefault(i, None)
            floor_angles.setdefault(i, 0.0)
        else:
            floor_types.setdefault(i, _DEFAULT_ROOF_TYPE)
            floor_angles.setdefault(i, _DEFAULT_ROOF_ANGLE_DEG)

    return floor_angles, floor_types


def _basement_floor_index_from_roof_3d(roof_3d_dir: Path) -> Optional[int]:
    p = roof_3d_dir / "basement_floor_index.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("basement_floor_index") is not None:
            return int(data["basement_floor_index"])
    except Exception:
        pass
    return None


def _write_floor_roof_json_from_edited(
    roof_3d_dir: Path,
    edited: dict,
    num_floors: int,
    plans_list: List[PlanInfo],
    plans_roof: List[PlanInfo],
    *,
    manifest_tab_plan_ids: list[str] | None = None,
) -> None:
    """Sincronizează floor_roof_angles.json / floor_roof_types.json cu editorul."""
    bi = _basement_floor_index_from_roof_3d(roof_3d_dir)
    angles, types = derive_floor_roof_settings_from_edited(
        edited, num_floors, bi, plans_list, plans_roof, manifest_tab_plan_ids=manifest_tab_plan_ids
    )
    fra_path = roof_3d_dir / "floor_roof_angles.json"
    frt_path = roof_3d_dir / "floor_roof_types.json"
    try:
        fra_path.write_text(
            json.dumps({str(k): float(v) for k, v in sorted(angles.items())}, indent=0),
            encoding="utf-8",
        )
        # JSON null pentru beci
        frt_path.write_text(
            json.dumps({str(k): v for k, v in sorted(types.items())}, indent=0),
            encoding="utf-8",
        )
        print(
            f"       [roof] Sincronizat {fra_path.name} + {frt_path.name} din editor (aggregat per etaj)",
            flush=True,
        )
    except Exception as e:
        print(f"       ⚠️ [roof] Nu pot scrie floor_roof_*.json: {e}", flush=True)


def _interior_footprint_mask_from_09_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Binar 255 = interior clădire (camere) din 09_interior.png — aceleași praguri ca pricing/jobs._compute_interior_mask_area_m2.
    """
    lower_o = np.array([0, 120, 200], dtype=np.uint8)
    upper_o = np.array([80, 210, 255], dtype=np.uint8)
    m_orange = cv2.inRange(img_bgr, lower_o, upper_o)
    lower_p = np.array([200, 0, 200], dtype=np.uint8)
    upper_p = np.array([255, 60, 255], dtype=np.uint8)
    m_purple = cv2.inRange(img_bgr, lower_p, upper_p)
    lower_b = np.array([200, 0, 0], dtype=np.uint8)
    upper_b = np.array([255, 50, 50], dtype=np.uint8)
    m_blue = cv2.inRange(img_bgr, lower_b, upper_b)
    lower_g = np.array([0, 200, 0], dtype=np.uint8)
    upper_g = np.array([50, 255, 50], dtype=np.uint8)
    m_green = cv2.inRange(img_bgr, lower_g, upper_g)
    return cv2.bitwise_or(cv2.bitwise_or(m_orange, m_purple), cv2.bitwise_or(m_blue, m_green))


def _load_interior_footprint_mask(interior_png: Path, target_h: int, target_w: int) -> np.ndarray | None:
    """Mască (H,W) uint8 0/255 aliniată la dimensiunea 01_walls_from_coords; None dacă lipsește / goală."""
    if not interior_png.is_file():
        return None
    img = cv2.imread(str(interior_png), cv2.IMREAD_COLOR)
    if img is None:
        return None
    mh, mw = img.shape[:2]
    if mh != target_h or mw != target_w:
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    mask = _interior_footprint_mask_from_09_bgr(img)
    if int(np.count_nonzero(mask)) <= 0:
        return None
    return mask


def _blend_polygon_over_wall_mask(
    wall_gray: np.ndarray,
    poly: np.ndarray,
    color_bgr: tuple[int, int, int],
    alpha: float = 0.52,
    footprint_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Imagine BGR: masca de pereți + umplere colorată.

    Dacă footprint_mask e dat (poligon ∩ interior 09_interior), culoarea se aplică doar acolo;
    altfel pe tot poligonul umplut.
    """
    h, w = wall_gray.shape[:2]
    base = cv2.cvtColor(wall_gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    fill_layer = np.zeros_like(base)
    cv2.fillPoly(fill_layer, [poly], color_bgr)
    if footprint_mask is not None:
        mask = np.clip(footprint_mask, 0, 255).astype(np.uint8)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
    a = float(np.clip(alpha, 0.0, 1.0))
    idx = mask > 0
    out = base.copy()
    out[idx] = (1.0 - a) * base[idx] + a * fill_layer[idx]
    return np.clip(out, 0, 255).astype(np.uint8)


# Mov = zonă folosită la m²; extra = zonă exclusă din m².
# Culori intenționat foarte contrastante pentru verificare vizuală.
_ROOF_MOV_BGR = (255, 0, 255)      # magenta
_ROOF_EXTRA_BGR = (0, 255, 255)    # galben


def _blend_roof_mov_extra_over_wall_mask(
    wall_gray: np.ndarray,
    mov_mask: np.ndarray,
    extra_mask: np.ndarray,
    alpha_mov: float = 0.70,
    alpha_extra: float = 0.90,
) -> np.ndarray:
    """BGR peste masca de pereți: mov = poligon ∩ interior ∩ zonă închisă de pereți; extra = rest (semi-transparent)."""
    h, w = wall_gray.shape[:2]
    base = cv2.cvtColor(wall_gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    out = base.copy()
    mov_u8 = np.clip(mov_mask, 0, 255).astype(np.uint8)
    extra_u8 = np.clip(extra_mask, 0, 255).astype(np.uint8)
    if mov_u8.shape[:2] != (h, w):
        mov_u8 = cv2.resize(mov_u8, (w, h), interpolation=cv2.INTER_NEAREST)
    if extra_u8.shape[:2] != (h, w):
        extra_u8 = cv2.resize(extra_u8, (w, h), interpolation=cv2.INTER_NEAREST)

    # Desenăm întâi extra, apoi mov (mov are prioritate în zone de overlap accidental).
    a_extra = float(np.clip(alpha_extra, 0.0, 1.0))
    idx_e = extra_u8 > 0
    if np.any(idx_e):
        extra_color = np.array(_ROOF_EXTRA_BGR, dtype=np.float32)
        out[idx_e] = (1.0 - a_extra) * out[idx_e] + a_extra * extra_color

    a_mov = float(np.clip(alpha_mov, 0.0, 1.0))
    idx_m = mov_u8 > 0
    if np.any(idx_m):
        mov_color = np.array(_ROOF_MOV_BGR, dtype=np.float32)
        out[idx_m] = (1.0 - a_mov) * out[idx_m] + a_mov * mov_color
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_edited_roof_rectangles(
    run_id: str,
    roof_3d_dir: Path,
    plans_roof: List[PlanInfo],
    plans_list: List[PlanInfo],
    *,
    allow_roof_rerun: bool = True,
) -> bool:
    """
    Aplică editările din roof_rectangles_edited.json peste roof_3d/rectangles/floor_X
    și recalculează metrici aggregate (entire/mixed/roof_metrics.json) pe noile dreptunghiuri.

    Scrie rectangle_S{i}.png: mov/extra = partiție flood pe canvas-ul rectangle (rectangle_SX):
    traversabil mov+negru (fundal întunecat), barieră = pereți albi; componentă ce atinge
    marginea imaginii = extra.
    """
    edits_path = roof_3d_dir / "roof_rectangles_edited.json"
    if not edits_path.exists():
        return False
    try:
        edited = json.loads(edits_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"       ⚠️ [roof] roof_rectangles_edited.json invalid: {e}", flush=True)
        return False
    if not isinstance(edited, dict):
        return False

    manifest_tabs = manifest_tab_plan_ids_order(run_id)
    if manifest_tabs and edited.get("_floor_key_scheme") == "manifest":
        print(f"       [roof] manifest tab → plan_id: {manifest_tabs}", flush=True)

    rectangles_root = roof_3d_dir / "rectangles"
    rectangles_root.mkdir(parents=True, exist_ok=True)

    total_area_m2 = 0.0
    total_contour_m = 0.0
    # Also persist a walls-coordinate version of the edited polygons for holzbot-roof (3D + overhang).
    edited_walls: dict[str, list[dict]] = {}

    for floor_idx, rects in _edited_entries_as_roof_indices(
        edited, plans_list, plans_roof, manifest_tab_plan_ids=manifest_tabs
    ):
        if floor_idx < 0 or floor_idx >= len(plans_roof):
            continue

        wall_mask = OUTPUT_ROOT / run_id / "scale" / plans_roof[floor_idx].plan_id / "cubicasa_steps" / "raster_processing" / "walls_from_coords" / "01_walls_from_coords.png"
        wall_img = cv2.imread(str(wall_mask), cv2.IMREAD_GRAYSCALE)
        if wall_img is None:
            continue
        h, w = wall_img.shape[:2]
        plan = plans_roof[floor_idx]
        ed = _editor_review_dimensions(run_id, plan)
        if ed:
            sx = float(w) / float(ed[0])
            sy = float(h) / float(ed[1])
            print(
                f"       [roof] floor_{floor_idx} ({plan.plan_id}): scale editor→walls "
                f"{ed[0]}x{ed[1]} → {w}x{h} (sx={sx:.4f}, sy={sy:.4f})",
                flush=True,
            )
        else:
            sx = sy = 1.0
            print(
                f"       ⚠️ [roof] floor_{floor_idx}: lipsă detections_review_data.json — "
                f"coordonate fără scalare (posibil greșit)",
                flush=True,
            )

        floor_dir = rectangles_root / f"floor_{floor_idx}"
        floor_dir.mkdir(parents=True, exist_ok=True)
        for p in floor_dir.glob("*.png"):
            p.unlink(missing_ok=True)

        union_mask = np.zeros((h, w), dtype=np.uint8)
        out_i = 0
        edited_walls[str(floor_idx)] = []
        for r in rects:
            pts = r.get("points") if isinstance(r, dict) else None
            if not isinstance(pts, list) or len(pts) < 3:
                continue
            arr: list[list[float]] = []
            for p in pts:
                if isinstance(p, list) and len(p) >= 2:
                    arr.append([float(p[0]), float(p[1])])
            if len(arr) < 3:
                continue
            arr_scaled = [
                [int(round(x * sx)), int(round(y * sy))] for x, y in arr
            ]
            poly = np.array(arr_scaled, dtype=np.int32)
            poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
            poly_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [poly], 255)
            base_fp = poly_mask
            rectangle_seed_bgr = _blend_polygon_over_wall_mask(
                wall_img,
                poly,
                _ROOF_MOV_BGR,
                alpha=0.70,
                footprint_mask=None,
            )
            mov_mask, extra_mask = split_mov_extra_roof_border_flood(base_fp, rectangle_seed_bgr)
            composite_bgr = _blend_roof_mov_extra_over_wall_mask(wall_img, mov_mask, extra_mask)
            cv2.imwrite(str(floor_dir / f"rectangle_S{out_i}.png"), composite_bgr)
            mov_px = int(np.count_nonzero(mov_mask))
            extra_px = int(np.count_nonzero(extra_mask))
            base_px = int(np.count_nonzero(base_fp))
            print(
                f"       [roof] floor_{floor_idx} rect_S{out_i}: base={base_px}px mov={mov_px}px extra={extra_px}px",
                flush=True,
            )
            row = {"points": arr_scaled}
            if isinstance(r, dict):
                if r.get("roofAngleDeg") is not None:
                    row["roofAngleDeg"] = r.get("roofAngleDeg")
                if r.get("roofType") is not None:
                    row["roofType"] = r.get("roofType")
                if r.get("roofOverhangM") is not None:
                    row["roofOverhangM"] = r.get("roofOverhangM")
                if r.get("roomName") is not None:
                    row["roomName"] = r.get("roomName")
            edited_walls[str(floor_idx)].append(row)
            out_i += 1
            union_mask = np.maximum(union_mask, mov_mask)

        mpp = _meters_per_pixel_for_plan(run_id, plans_roof[floor_idx]) or 0.01
        area_px = float(np.count_nonzero(union_mask))
        contours, _ = cv2.findContours(union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_px = float(sum(cv2.arcLength(c, True) for c in contours))
        total_area_m2 += area_px * (mpp ** 2)
        total_contour_m += contour_px * mpp

    # Preserve any existing 3D-derived overhang metrics (computed by holzbot-roof clean_workflow).
    # The roof editor currently modifies the "roof (insulated) zone" rectangles; we recompute unfold_roof totals
    # from the edited polygons, and keep unfold_overhang from the latest 3D workflow if available.
    prev_overhang_total_area_m2 = 0.0
    prev_overhang_total_contour_m = 0.0
    prev_metrics_path = (roof_3d_dir / "entire" / "mixed" / "roof_metrics.json")
    if prev_metrics_path.exists():
        try:
            prev = json.loads(prev_metrics_path.read_text(encoding="utf-8"))
            uo = (prev.get("unfold_overhang") or {}).get("total") or {}
            prev_overhang_total_area_m2 = float(uo.get("area_m2") or 0.0)
            prev_overhang_total_contour_m = float(uo.get("contour_m") or 0.0)
        except Exception:
            prev_overhang_total_area_m2 = 0.0
            prev_overhang_total_contour_m = 0.0

    roof_area_m2 = round(total_area_m2, 4)
    roof_contour_m = round(total_contour_m, 4)
    overhang_area_m2 = round(prev_overhang_total_area_m2, 4)
    overhang_contour_m = round(prev_overhang_total_contour_m, 4)
    combined_area_m2 = round(float(roof_area_m2) + float(overhang_area_m2), 4)
    combined_contour_m = round(float(roof_contour_m) + float(overhang_contour_m), 4)

    metrics = {
        "unfold_roof": {"total": {"area_m2": roof_area_m2, "contour_m": roof_contour_m}},
        "unfold_overhang": {"total": {"area_m2": overhang_area_m2, "contour_m": overhang_contour_m}},
        "total_combined": {"area_m2": combined_area_m2, "contour_m": combined_contour_m},
    }
    mixed_dir = roof_3d_dir / "entire" / "mixed"
    mixed_dir.mkdir(parents=True, exist_ok=True)
    (mixed_dir / "roof_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_floor_roof_json_from_edited(
        roof_3d_dir,
        edited,
        len(plans_roof),
        plans_list,
        plans_roof,
        manifest_tab_plan_ids=manifest_tabs,
    )

    # Save walls-coordinate polygons for holzbot-roof (expects same coordinates as wall masks).
    try:
        (roof_3d_dir / "roof_rectangles_edited_walls.json").write_text(
            json.dumps(edited_walls, indent=2), encoding="utf-8"
        )
    except Exception as e:
        print(f"       ⚠️ [roof] Nu pot scrie roof_rectangles_edited_walls.json: {e}", flush=True)

    # Re-run holzbot-roof workflow with edited sections so overhang faces/metrics
    # are recalculated from the editor geometry, not kept from a previous run.
    if allow_roof_rerun:
        try:
            basement_idx = _basement_floor_index_from_roof_3d(roof_3d_dir)
            edited_angles, edited_types = derive_floor_roof_settings_from_edited(
                edited,
                len(plans_roof),
                basement_idx,
                plans_list,
                plans_roof,
                manifest_tab_plan_ids=manifest_tabs,
            )
            rerun_dir = _run_roof_3d_workflow(
                run_id,
                plans_roof,
                floor_roof_types=edited_types,
                floor_roof_angles=edited_angles,
                basement_floor_index_override=basement_idx,
                edited_rectangles_json_path=(roof_3d_dir / "roof_rectangles_edited_walls.json"),
            )
            if rerun_dir:
                print("       ✅ [roof] Aplicate editările în holzbot-roof (roof_types + unfold_overhang recalculat)", flush=True)
                # clean_workflow rescrie rectangles/floor_X/*.png; redesenăm overlay-ul flood ca output final.
                return _apply_edited_roof_rectangles(
                    run_id,
                    roof_3d_dir,
                    plans_roof,
                    plans_list,
                    allow_roof_rerun=False,
                )
        except Exception as e:
            print(f"       ⚠️ [roof] Re-run holzbot-roof cu editări a eșuat: {e}", flush=True)

    print("       ✅ [roof] Aplicate editările din roof editor în roof_3d/rectangles + roof_metrics.json", flush=True)
    return True


def resolve_plans_roof_order(run_id: str) -> tuple[List[PlanInfo], Path, List[PlanInfo]]:
    """
    Ordine pentru roof rectangles: de sus în jos (top -> ... -> ground/basement).
    floor_0 devine etajul cel mai de sus.
    Returnează (plans_roof, job_root, plans) — plans = ordinea din load_plan_infos (pentru run_roof_for_run).
    """
    plans = load_plan_infos(run_id, stage_name=STAGE_NAME)
    from config.settings import JOBS_ROOT

    job_root = None
    for jdir in JOBS_ROOT.glob("*"):
        if jdir.is_dir() and run_id in jdir.name:
            job_root = jdir
            break
    if job_root is None:
        job_root = JOBS_ROOT / run_id

    run_dir = RUNS_ROOT / run_id
    basement_plan_id: str | None = None
    order_from_bottom: list[int] | None = None
    if run_dir.exists():
        if (run_dir / "floor_order.json").exists():
            try:
                data = json.loads((run_dir / "floor_order.json").read_text(encoding="utf-8"))
                order_from_bottom = data.get("order_from_bottom")
                if order_from_bottom is not None and len(order_from_bottom) == len(plans) and set(order_from_bottom) == set(range(len(plans))):
                    pass
                else:
                    order_from_bottom = None
            except Exception:
                order_from_bottom = None
        if order_from_bottom is None and (run_dir / "basement_plan_id.json").exists():
            try:
                data = json.loads((run_dir / "basement_plan_id.json").read_text(encoding="utf-8"))
                original_basement_idx = data.get("basement_plan_index")
                if original_basement_idx is not None and 0 <= original_basement_idx < len(plans):
                    basement_plan_id = plans[original_basement_idx].plan_id
            except Exception:
                pass

    if order_from_bottom is not None:
        # floor_order.json este "de jos în sus"; pentru roof rectangles cerința este "de sus în jos".
        plans_roof = [plans[i] for i in reversed(order_from_bottom)]
    else:

        def _key(p):
            return _floor_sort_key(p, job_root, basement_plan_id, run_id, run_dir)

        # Sortarea de bază e de jos în sus; pentru rectangles inversăm în top->down.
        plans_roof = list(reversed(sorted(plans, key=_key)))

    return plans_roof, job_root, plans


def apply_roof_edits_from_disk(run_id: str) -> bool:
    """
    Regenerează rectangle_S*.png din roof_rectangles_edited.json (ex.: după PATCH din UI).
    """
    roof_3d_dir = OUTPUT_ROOT / run_id / "roof" / "roof_3d"
    if not (roof_3d_dir / "roof_rectangles_edited.json").exists():
        return False
    try:
        plans_roof, _, plans_list = resolve_plans_roof_order(run_id)
    except Exception as e:
        print(f"       ⚠️ [roof] apply_roof_edits_from_disk: {e}", flush=True)
        return False
    return _apply_edited_roof_rectangles(run_id, roof_3d_dir, plans_roof, plans_list)


def run_roof_for_run(run_id: str, max_parallel: int | None = None, notify_ui_events: bool = True) -> List[RoofJobResult]:
    """
    Punct de intrare pentru etapa „roof" (calcul acoperiș).

    - Rulează holzbot-roof clean_workflow cu wall masks de pe toate etajele
    - Notifică UI cu filled.png pentru fiecare tip acoperiș (1_w, 2_w, 4_w, 4.5_w)
    - Scrie roof_estimation.json cu preț fix 10 EUR pentru top floor

    Output-uri:
      output/<RUN_ID>/roof/roof_3d/entire/{1_w,2_w,4_w,4.5_w}/filled.png
      output/<RUN_ID>/roof/<plan_id>/roof_estimation.json
    """
    try:
        plans_roof, job_root, plans = resolve_plans_roof_order(run_id)
    except PlansListError as e:
        print(f"❌ [{STAGE_NAME}] {e}")
        return []

    total = len(plans)
    print(f"   [roof] Ordine acoperiș floor_0..floor_X: {[p.plan_id for p in plans_roof]}", flush=True)

    frontend_data = load_frontend_data_for_run(run_id, job_root)

    run_dir = RUNS_ROOT / run_id

    basement_plan_id: str | None = None
    order_from_bottom: list[int] | None = None
    if run_dir.exists():
        if (run_dir / "floor_order.json").exists():
            try:
                data = json.loads((run_dir / "floor_order.json").read_text(encoding="utf-8"))
                ob = data.get("order_from_bottom")
                if ob is not None and len(ob) == len(plans) and set(ob) == set(range(len(plans))):
                    order_from_bottom = ob
            except Exception:
                order_from_bottom = None
        if order_from_bottom is None and (run_dir / "basement_plan_id.json").exists():
            try:
                data = json.loads((run_dir / "basement_plan_id.json").read_text(encoding="utf-8"))
                oi = data.get("basement_plan_index")
                if oi is not None and 0 <= oi < len(plans):
                    basement_plan_id = plans[oi].plan_id
            except Exception:
                pass
        # Keep basement plan id available even when floor_order.json exists.
        if (run_dir / "basement_plan_id.json").exists():
            try:
                data = json.loads((run_dir / "basement_plan_id.json").read_text(encoding="utf-8"))
                oi = data.get("basement_plan_index")
                if oi is not None and 0 <= oi < len(plans):
                    basement_plan_id = plans[oi].plan_id
            except Exception:
                pass

    # Indexul beciului în lista sortată plans_roof.
    basement_idx: int | None = None
    if order_from_bottom is not None and (run_dir / "basement_plan_id.json").exists():
        try:
            data = json.loads((run_dir / "basement_plan_id.json").read_text(encoding="utf-8"))
            raw_bidx = data.get("basement_plan_index")
            if raw_bidx is not None:
                original_bidx = int(raw_bidx)
                if original_bidx in order_from_bottom:
                    # order_from_bottom is bottom->top, but plans_roof is top->down.
                    bottom_idx = order_from_bottom.index(original_bidx)
                    basement_idx = (len(order_from_bottom) - 1) - bottom_idx
                    print(
                        f"   [roof] Beci: plan original index {original_bidx} -> roof index {basement_idx} (din floor_order)",
                        flush=True,
                    )
        except Exception:
            pass
    if basement_idx is None and basement_plan_id:
        for i, p in enumerate(plans_roof):
            if p.plan_id == basement_plan_id:
                basement_idx = i
                print(f"   [roof] Beci: plan {basement_plan_id} la index {basement_idx}", flush=True)
                break
    num_floors_roof = len(plans_roof)
    num_floors_for_gemini = (num_floors_roof - 1) if basement_idx is not None else num_floors_roof

    # Tip acoperiș / unghi: nu mai trimitem Ansichten la Gemini (doar fallback 2_w + unghi default din pipeline).
    floor_roof_types: dict | None = None
    floor_roof_angles: dict | None = None
    side_views: List[Path] = []
    if not side_views or num_floors_for_gemini < 1:
        print(f"\n>>> PAS ROOF GEMINI: NU RULEAZĂ (side_views={len(side_views) if side_views else 0}, num_floors_for_gemini={num_floors_for_gemini}) – folosesc fallback 2_w + default unghi <<<\n", flush=True)
    if side_views and num_floors_for_gemini >= 1:
        print("\n" + "=" * 70, flush=True)
        print(">>> PAS ROOF: GEMINI – clasificare TIP ACOPERIȘ ȘI UNGHI/PANTĂ (side_view) <<<", flush=True)
        print(f">>> Trimitem TOATE cele {len(side_views)} imagini side_view la Gemini. <<<", flush=True)
        print("=" * 70 + "\n", flush=True)
        try:
            from segmenter.classifier import setup_gemini_client
            from roof.roof_type_classifier import classify_roof_types_per_floor
            gemini = setup_gemini_client()
            if gemini:
                types_result, angles_result = classify_roof_types_per_floor(
                    gemini, side_views, num_floors_for_gemini
                )
                # Gemini întoarce indexare semantică "bottom-up" (parter, etaj_1, ...).
                # Workflow-ul roof folosește indexare "top-down" (floor_0 = cel mai sus).
                # Mapăm explicit prin ordinea reală a etajelor, nu prin presupunere simplă pe index.
                roof_indices_bottom_up = list(reversed(list(range(num_floors_roof))))
                if basement_idx is not None:
                    roof_indices_bottom_up = [i for i in roof_indices_bottom_up if i != basement_idx]
                gemini_to_roof_idx = {
                    int(gidx): int(roof_idx)
                    for gidx, roof_idx in enumerate(roof_indices_bottom_up)
                }

                mapped_types: dict[int, str] = {}
                mapped_angles: dict[int, float] = {}
                for gidx, roof_idx in gemini_to_roof_idx.items():
                    if types_result and gidx in types_result:
                        mapped_types[roof_idx] = types_result[gidx]
                    if angles_result and gidx in angles_result:
                        mapped_angles[roof_idx] = float(angles_result[gidx])

                if mapped_types:
                    floor_roof_types = mapped_types
                    print(
                        f"   [roof] FOLOSIM tipuri acoperiș de la Gemini (mapat bottom->top): {floor_roof_types}",
                        flush=True,
                    )
                if mapped_angles:
                    floor_roof_angles = mapped_angles
                    print(
                        f"   [roof] FOLOSIM unghiuri acoperiș de la Gemini (mapat bottom->top): {floor_roof_angles}",
                        flush=True,
                    )
                if not floor_roof_types:
                    print(f"   [roof] NU avem tipuri de la Gemini – vom folosi fallback 2_w pentru toate etajele.", flush=True)
                if not floor_roof_angles:
                    print(f"   [roof] NU avem unghiuri de la Gemini – folosesc default 30° per etaj.", flush=True)
            else:
                print(">>> PAS ROOF GEMINI: client Gemini nu s-a inițializat (GEMINI_API_KEY?) – fallback 2_w + 30° <<<", flush=True)
        except Exception as e:
            print(f"       ⚠️ [{STAGE_NAME}] Roof type classifier: {e}", flush=True)
    DEFAULT_ROOF_ANGLE_DEG = 30.0
    if not floor_roof_types and num_floors_roof >= 1:
        floor_roof_types = {i: "2_w" for i in range(num_floors_roof)}
        print(f"   [roof] Aplicat fallback: floor_roof_types={floor_roof_types}", flush=True)
    elif floor_roof_types is not None and basement_idx is not None and num_floors_roof >= 1:
        # Beci: fără acoperiș; celelalte etaje rămân la valorile deja mapate pe roof-index.
        full_types: dict[int, str | None] = {}
        for i in range(num_floors_roof):
            if i == basement_idx:
                full_types[i] = None
            else:
                full_types[i] = floor_roof_types.get(i, "2_w")
        floor_roof_types = full_types
        print(
            f"   [roof] Beci fără acoperiș la index {basement_idx}; floor_roof_types={floor_roof_types}",
            flush=True,
        )

    # Unghiuri: folosim cele de la Gemini; lipsă = default 30° (sau 0 pentru 0_w)
    if floor_roof_angles is None or not floor_roof_angles:
        floor_roof_angles = {}
    for i in range(num_floors_roof):
        if i not in floor_roof_angles:
            t = (floor_roof_types or {}).get(i)
            floor_roof_angles[i] = 0.0 if t == "0_w" else DEFAULT_ROOF_ANGLE_DEG
    if basement_idx is not None and floor_roof_angles:
        # Fixăm explicit unghiul beciului; restul cheilor sunt deja mapate pe roof-index.
        floor_roof_angles[basement_idx] = 0.0
    if num_floors_roof >= 1:
        floor_roof_angles = {i: float(floor_roof_angles.get(i, DEFAULT_ROOF_ANGLE_DEG)) for i in range(num_floors_roof)}
    else:
        floor_roof_angles = None

    # Dacă utilizatorul a salvat din editorul de acoperiș, suprascriem Gemini/fallback cu agregat per etaj.
    roof_3d_edits = OUTPUT_ROOT / run_id / "roof" / "roof_3d" / "roof_rectangles_edited.json"
    if roof_3d_edits.exists() and num_floors_roof >= 1 and floor_roof_angles is not None and floor_roof_types is not None:
        try:
            edited_ov = json.loads(roof_3d_edits.read_text(encoding="utf-8"))
            if isinstance(edited_ov, dict) and edited_ov:
                d_ang, d_typ = derive_floor_roof_settings_from_edited(
                    edited_ov,
                    num_floors_roof,
                    basement_idx,
                    plans,
                    plans_roof,
                    manifest_tab_plan_ids=manifest_tab_plan_ids_order(run_id),
                )
                floor_roof_angles = {i: float(d_ang[i]) for i in range(num_floors_roof)}
                floor_roof_types = {i: d_typ[i] for i in range(num_floors_roof)}
                print(
                    f"   [roof] Override din roof editor (aggregat per etaj): "
                    f"angles={floor_roof_angles}, types={floor_roof_types}",
                    flush=True,
                )
        except Exception as e:
            print(f"   [roof] Override editor ignorate: {e}", flush=True)

    print(f"   [roof] Valorile trimise la workflow: floor_roof_types={floor_roof_types}, floor_roof_angles={floor_roof_angles}", flush=True)
    print(f"\n⚙️  [{STAGE_NAME}] Acoperiș pentru {total} plan{'uri' if total > 1 else ''} (inclusiv beci în 3D, preț fix 10 EUR)...")

    roof_3d_dir = OUTPUT_ROOT / run_id / "roof" / "roof_3d"
    skip_auto_rectangles = False
    if roof_3d_edits.exists():
        try:
            edited_probe = json.loads(roof_3d_edits.read_text(encoding="utf-8"))
            skip_auto_rectangles = isinstance(edited_probe, dict) and len(edited_probe) > 0
        except Exception:
            skip_auto_rectangles = False
    if skip_auto_rectangles:
        roof_3d_dir.mkdir(parents=True, exist_ok=True)
        try:
            floor_plan_ids = {str(i): p.plan_id for i, p in enumerate(plans_roof)}
            (roof_3d_dir / "roof_floor_plan_ids.json").write_text(
                json.dumps(floor_plan_ids, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            print(f"       ⚠️ [roof] roof_floor_plan_ids.json: {e}", flush=True)
        if basement_idx is not None:
            try:
                (roof_3d_dir / "basement_floor_index.json").write_text(
                    json.dumps({"basement_floor_index": basement_idx}, indent=0), encoding="utf-8"
                )
            except Exception:
                pass
        _apply_edited_roof_rectangles(run_id, roof_3d_dir, plans_roof, plans)
        print("       [roof] Sărit clean_workflow (poligoane doar din editor).", flush=True)
    else:
        roof_3d_dir = _run_roof_3d_workflow(
            run_id, plans_roof, floor_roof_types, floor_roof_angles,
            basement_floor_index_override=basement_idx,
        )
        if roof_3d_dir:
            _apply_edited_roof_rectangles(run_id, roof_3d_dir, plans_roof, plans)
    if roof_3d_dir and notify_ui_events:
        try:
            from orchestrator import notify_ui
            mixed_png = roof_3d_dir / "entire" / "mixed" / "filled.png"
            if mixed_png.exists():
                notify_ui(STAGE_NAME, mixed_png)
            else:
                for rt in ("0_w", "1_w", "2_w", "4_w", "4.5_w"):
                    p = roof_3d_dir / "entire" / rt / "filled.png"
                    if p.exists():
                        notify_ui(STAGE_NAME, p)
        except ImportError:
            pass
    
    if max_parallel is None:
        cpu_count = os.cpu_count() or 4
        max_parallel = min(cpu_count, total)
    
    results: List[RoofJobResult] = []
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(
                _run_for_single_plan,
                run_id,
                idx,
                total,
                plan,
                frontend_data,
                total,
                job_root,
            ): plan
            for idx, plan in enumerate(plans, start=1)
        }
        
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            status = "✅" if res.success else "❌"
            print(f"{status} [{STAGE_NAME}] {res.plan_id} → {res.message}")
    
    # ==========================================
    # REZUMAT FINAL
    # ==========================================
    
    print(f"\n{'─'*70}")
    print("🏠 REZUMAT ACOPERIȘ:")
    print(f"{'─'*70}")
    
    total_roof_cost = 0.0
    for r in results:
        if r.success and r.result_data:
            cost = r.result_data.get("roof_final_total_eur", 0)
            total_roof_cost += cost
            
            if r.result_data.get("is_top_floor"):
                print(f"  ✅ {r.plan_id}: {cost:,.0f} EUR")
            else:
                print(f"  ℹ️  {r.plan_id}: Nu e top floor → 0 EUR")
    
    print(f"{'─'*70}")
    print(f"💰 TOTAL ACOPERIȘ: {total_roof_cost:,.2f} EUR")
    print(f"{'─'*70}\n")
    
    return results


if __name__ == '__main__':
    # Regenerează rectangle_S*.png din roof_rectangles_edited.json: python -m roof.jobs <run_id>
    if len(sys.argv) < 2:
        print('Usage: python -m roof.jobs <run_id>', flush=True)
        raise SystemExit(2)
    _ok = apply_roof_edits_from_disk(sys.argv[1])
    raise SystemExit(0 if _ok else 1)