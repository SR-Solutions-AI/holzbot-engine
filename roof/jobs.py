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
from typing import List

from config.settings import (
    OUTPUT_ROOT,
    RUNS_ROOT,
    load_plan_infos,
    PlansListError,
    PlanInfo,
)
from config.frontend_loader import load_frontend_data_for_run

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


def _load_floor_metadata(job_root: Path, original_name: str) -> dict | None:
    """Încarcă metadata pentru a determina floor_type."""
    metadata_file = job_root / "plan_metadata" / f"{original_name}.json"
    if not metadata_file.exists():
        return None
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


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
    metadata = _load_floor_metadata(job_root, original_name)

    is_top_floor = False
    floor_type = "unknown"

    if metadata:
        floor_class = metadata.get("floor_classification", {})
        floor_type = floor_class.get("floor_type", "unknown").lower()
        is_top_floor = any(keyword in floor_type for keyword in ["top", "roof", "attic", "mansarda"])

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
    if floor_roof_types:
        frt_path = roof_3d_dir / "floor_roof_types.json"
        try:
            content = json.dumps({str(k): v for k, v in floor_roof_types.items()}, indent=0)
            frt_path.write_text(content, encoding="utf-8")
            print(f"       [roof] Scris {frt_path.name}: {content!r}", flush=True)
        except Exception as e:
            print(f"       ⚠️ [roof] Eroare la scrierea {frt_path}: {e}", flush=True)
    else:
        print(f"       [roof] Nu s-au scris tipuri (floor_roof_types gol).", flush=True)
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
    else:
        tmp = tempfile.mkdtemp(prefix="holzbot_roof_floors_")
        try:
            for i, src in enumerate(paths):
                dst = Path(tmp) / f"floor_{i:02d}.png"
                shutil.copy2(src, dst)
            wall_input = tmp
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
        result = subprocess.run(
            [_python, str(script_path), wall_input, str(roof_3d_dir)],
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


def run_roof_for_run(run_id: str, max_parallel: int | None = None) -> List[RoofJobResult]:
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
        plans = load_plan_infos(run_id, stage_name=STAGE_NAME)
    except PlansListError as e:
        print(f"❌ [{STAGE_NAME}] {e}")
        return []

    total = len(plans)

    from config.settings import JOBS_ROOT
    job_root = None
    for jdir in JOBS_ROOT.glob("*"):
        if jdir.is_dir() and run_id in jdir.name:
            job_root = jdir
            break
    if job_root is None:
        job_root = JOBS_ROOT / run_id

    frontend_data = load_frontend_data_for_run(run_id, job_root)

    # Excludem beci din roof workflow
    run_dir = RUNS_ROOT / run_id
    basement_idx: int | None = None
    if (run_dir / "basement_plan_id.json").exists():
        try:
            data = json.loads((run_dir / "basement_plan_id.json").read_text(encoding="utf-8"))
            basement_idx = data.get("basement_plan_index")
        except Exception:
            pass
    plans_roof = [p for i, p in enumerate(plans) if i != basement_idx] if basement_idx is not None else plans
    num_floors_roof = len(plans_roof)

    # Tip acoperiș per etaj: DOAR din Gemini (side_view). Panta/unghi: din formular (detalii acoperiș).
    floor_roof_types: dict | None = None
    side_views = _collect_side_views(job_root)
    if not side_views or num_floors_roof < 1:
        print(f"\n>>> PAS ROOF GEMINI: NU RULEAZĂ (side_views={len(side_views) if side_views else 0}, num_floors_roof={num_floors_roof}) – folosesc fallback 2_w <<<\n", flush=True)
    if side_views and num_floors_roof >= 1:
        print("\n" + "=" * 70, flush=True)
        print(">>> PAS ROOF: GEMINI – clasificare DOAR TIP ACOPERIȘ (side_view) <<<", flush=True)
        print(">>> Panta acoperișului se introduce în formular (Dämmung & Dachdeckung) <<<", flush=True)
        print("=" * 70 + "\n", flush=True)
        try:
            from segmenter.classifier import setup_gemini_client
            from roof.roof_type_classifier import classify_roof_types_per_floor
            gemini = setup_gemini_client()
            if gemini:
                floor_roof_types = classify_roof_types_per_floor(
                    gemini, side_views, num_floors_roof
                )
                if floor_roof_types:
                    print(f"   [roof] FOLOSIM tipuri acoperiș de la Gemini: {floor_roof_types}", flush=True)
                else:
                    print(f"   [roof] NU avem tipuri de la Gemini – vom folosi fallback 2_w pentru toate etajele.", flush=True)
            else:
                print(">>> PAS ROOF GEMINI: client Gemini nu s-a inițializat (GEMINI_API_KEY?) – fallback 2_w <<<", flush=True)
        except Exception as e:
            print(f"       ⚠️ [{STAGE_NAME}] Roof type classifier: {e}", flush=True)
    if not floor_roof_types and num_floors_roof >= 1:
        floor_roof_types = {i: "2_w" for i in range(num_floors_roof)}
        print(f"   [roof] Aplicat fallback: floor_roof_types={floor_roof_types}", flush=True)

    # Panta acoperișului: din formular (pasul Dämmung & Dachdeckung), o valoare pentru toate etajele
    DEFAULT_ROOF_ANGLE_DEG = 30.0
    angle_deg = DEFAULT_ROOF_ANGLE_DEG
    dd = frontend_data.get("daemmungDachdeckung") or {}
    raw = frontend_data.get("pantaAcoperis") or frontend_data.get("dachneigung") or dd.get("pantaAcoperis") or dd.get("dachneigung")
    print(f"   [roof] Sursă pantă: frontend_data.pantaAcoperis={frontend_data.get('pantaAcoperis')!r}, "
          f"frontend_data.dachneigung={frontend_data.get('dachneigung')!r}, "
          f"daemmungDachdeckung.pantaAcoperis={dd.get('pantaAcoperis')!r}, "
          f"daemmungDachdeckung.dachneigung={dd.get('dachneigung')!r} → raw={raw!r}", flush=True)
    if raw is not None:
        try:
            v = float(raw)
            if 10 <= v <= 70:
                angle_deg = v
                print(f"   [roof] Folosesc panta din formular: {angle_deg}°", flush=True)
            else:
                print(f"   [roof] Valoare raw={v} în afara [10,70] – folosesc default {DEFAULT_ROOF_ANGLE_DEG}°", flush=True)
        except (TypeError, ValueError) as e:
            print(f"   [roof] Nu pot converti raw={raw!r} la float: {e} – folosesc default {DEFAULT_ROOF_ANGLE_DEG}°", flush=True)
    else:
        print(f"   [roof] Nicio pantă în formular – folosesc default {DEFAULT_ROOF_ANGLE_DEG}° pentru toate etajele.", flush=True)
    floor_roof_angles: dict | None = {i: angle_deg for i in range(num_floors_roof)} if num_floors_roof >= 1 else None

    print(f"   [roof] Valorile trimise la workflow: floor_roof_types={floor_roof_types}, floor_roof_angles={floor_roof_angles}", flush=True)
    print(f"\n⚙️  [{STAGE_NAME}] Acoperiș pentru {total} plan{'uri' if total > 1 else ''} (preț fix 10 EUR)...")

    # 1. Rulează holzbot-roof 3D workflow (fără beci, cu floor_roof_types și floor_roof_angles)
    roof_3d_dir = _run_roof_3d_workflow(
        run_id, plans_roof, floor_roof_types, floor_roof_angles
    )
    if roof_3d_dir:
        try:
            from orchestrator import notify_ui
            mixed_png = roof_3d_dir / "entire" / "mixed" / "filled.png"
            if mixed_png.exists():
                notify_ui(STAGE_NAME, mixed_png)
            else:
                for rt in ("1_w", "2_w", "4_w", "4.5_w"):
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