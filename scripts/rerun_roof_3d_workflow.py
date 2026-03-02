#!/usr/bin/env python3
"""
Re-rulare DOAR a workflow-ului roof 3D (clean_workflow) cu ordinea corectă a etajelor.

Engine-ul ordonează planurile: beci=floor_0, parter=floor_1, etaj=floor_2 (sau parter=0, etaj=1
dacă nu e beci). Acest script încarcă planurile, le sortează la fel ca run_roof_for_run,
apoi rulează clean_workflow cu floor_00.png = primul plan, floor_01.png = al doilea, etc.
Astfel rectangles/ și roof_types/ rămân aliniate pe etaje.

Dacă runs/<RUN_ID> nu există, folosește un fallback: planurile din output/<RUN_ID>/scale/
sunt ordonate după aria măștii de pereți (descrescător), deci parter (arie mai mare) = floor_0.

Usage:
  cd holzbot-engine && .venv/bin/python scripts/rerun_roof_3d_workflow.py <RUN_ID>

Exemplu:
  .venv/bin/python scripts/rerun_roof_3d_workflow.py 8107c3a3-1492-4945-8aea-63008de2adbc
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Ca să putem importa din config (și eventual roof.jobs)
_ENGINE_ROOT = Path(__file__).resolve().parents[1]
_ROOF_ROOT = _ENGINE_ROOT.parent / "holzbot-roof"
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))
os.chdir(_ENGINE_ROOT)

from config.settings import OUTPUT_ROOT, RUNS_ROOT


def _wall_mask_area(path: Path) -> int:
    """Aria măștii de pereți în pixeli. Returnează 0 dacă lipsește."""
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0
        return int((img > 0).sum())
    except Exception:
        return 0


def _discover_plans_from_scale(run_id: str):
    """Când runs/ nu există: descoperă planurile din output/run_id/scale/, ordonate după arie (desc)."""
    scale_root = OUTPUT_ROOT / run_id / "scale"
    if not scale_root.is_dir():
        return None
    subpath = ["cubicasa_steps", "raster_processing", "walls_from_coords", "01_walls_from_coords.png"]
    pairs = []
    for plan_dir in scale_root.iterdir():
        if not plan_dir.is_dir():
            continue
        wall_path = plan_dir / Path(*subpath)
        if not wall_path.exists():
            continue
        area = _wall_mask_area(wall_path)
        pairs.append((plan_dir.name, wall_path, area))
    if not pairs:
        return None
    # Parter (arie mai mare) = floor_0; etaj (arie mai mică) = floor_1
    pairs.sort(key=lambda x: -x[2])
    return [(plan_id, path) for plan_id, path, _ in pairs]


def _run_workflow_with_ordered_paths(run_id: str, ordered_plan_paths, roof_3d_dir: Path) -> bool:
    """Construiește folderul de input în ordinea dată și rulează clean_workflow (fără a depinde de roof.jobs)."""
    if len(ordered_plan_paths) == 1:
        wall_input = str(ordered_plan_paths[0][1])
    else:
        tmp = tempfile.mkdtemp(prefix="holzbot_roof_floors_")
        try:
            floors_meta = []
            for i, (plan_id, src) in enumerate(ordered_plan_paths):
                dst = Path(tmp) / f"floor_{i:02d}.png"
                shutil.copy2(src, dst)
                interior_path = src.parent / "09_interior.png"
                floors_meta.append({
                    "floor_path": dst.name,
                    "plan_id": plan_id,
                    "interior_mask_path": str(interior_path.resolve()) if interior_path.exists() else None,
                })
            (Path(tmp) / "floors_meta.json").write_text(
                json.dumps(floors_meta, indent=0), encoding="utf-8"
            )
            wall_input = tmp
        except Exception as e:
            print(f"       ⚠️ [roof] Eroare la pregătirea etajelor: {e}", file=sys.stderr)
            if os.path.exists(tmp):
                shutil.rmtree(tmp, ignore_errors=True)
            return False

    script_path = _ROOF_ROOT / "scripts" / "clean_workflow.py"
    if not script_path.exists():
        print(f"       ⚠️ [roof] holzbot-roof/scripts/clean_workflow.py nu există.", file=sys.stderr)
        if len(ordered_plan_paths) > 1 and os.path.exists(wall_input):
            shutil.rmtree(wall_input, ignore_errors=True)
        return False

    _venv_py = _ROOF_ROOT / ".venv314" / "bin" / "python"
    _python = str(_venv_py) if _venv_py.exists() else (shutil.which("python3.14") or sys.executable)
    try:
        result = subprocess.run(
            [_python, str(script_path), wall_input, str(roof_3d_dir)],
            cwd=str(_ROOF_ROOT),
            capture_output=False,
            timeout=300,
        )
        ok = result.returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    except Exception as e:
        print(f"       ⚠️ [roof] Eroare la clean_workflow: {e}", file=sys.stderr)
        ok = False
    finally:
        if len(ordered_plan_paths) > 1 and os.path.exists(wall_input):
            shutil.rmtree(wall_input, ignore_errors=True)
    return ok


def main():
    if len(sys.argv) < 2:
        print("Usage: .venv/bin/python scripts/rerun_roof_3d_workflow.py <RUN_ID>", file=sys.stderr)
        sys.exit(1)
    run_id = sys.argv[1].strip()

    roof_3d_dir = OUTPUT_ROOT / run_id / "roof" / "roof_3d"
    roof_3d_dir.mkdir(parents=True, exist_ok=True)

    run_dir = RUNS_ROOT / run_id
    ordered_plan_paths = None  # list of (plan_id, path) in engine order

    # 1) Încercare cu runs/ și ordinea engine (beci, parter, etaj)
    if run_dir.exists():
        try:
            from config.settings import load_plan_infos, PlansListError
            from roof.jobs import (
                STAGE_NAME,
                _floor_sort_key,
                _run_roof_3d_workflow,
            )
            plans = load_plan_infos(run_id, stage_name="roof")
            from config.settings import JOBS_ROOT
            job_root = None
            for jdir in JOBS_ROOT.glob("*"):
                if jdir.is_dir() and run_id in jdir.name:
                    job_root = jdir
                    break
            if job_root is None:
                job_root = JOBS_ROOT / run_id
            basement_plan_id = None
            if (run_dir / "basement_plan_id.json").exists():
                try:
                    data = json.loads((run_dir / "basement_plan_id.json").read_text(encoding="utf-8"))
                    idx = data.get("basement_plan_index")
                    if idx is not None and 0 <= idx < len(plans):
                        basement_plan_id = plans[idx].plan_id
                except Exception:
                    pass
            plans_roof = sorted(
                plans,
                key=lambda p: _floor_sort_key(p, job_root, basement_plan_id, run_id, run_dir),
            )
            print("   [roof] Ordine etaje (din runs/ + metadata engine):")
            for i, p in enumerate(plans_roof):
                print(f"      floor_{i} ← {p.plan_id}")
            # Citim tipuri/unghiuri existente
            floor_roof_types = None
            floor_roof_angles = None
            basement_idx = None
            if basement_plan_id is not None:
                for i, p in enumerate(plans_roof):
                    if p.plan_id == basement_plan_id:
                        basement_idx = i
                        break
            for f in (roof_3d_dir / "floor_roof_types.json", roof_3d_dir / "floor_roof_angles.json"):
                if f.exists():
                    try:
                        data = json.loads(f.read_text(encoding="utf-8"))
                        d = {int(k): v for k, v in data.items()}
                        if "angle" in f.stem:
                            floor_roof_angles = {k: float(v) for k, v in d.items()}
                        else:
                            floor_roof_types = d
                        print(f"   [roof] Citit {f.name}")
                    except Exception:
                        pass
            roof_3d_dir = _run_roof_3d_workflow(
                run_id,
                plans_roof,
                floor_roof_types,
                floor_roof_angles,
                basement_floor_index_override=basement_idx,
            )
            if roof_3d_dir:
                print(f"\n✅ Workflow roof 3D finalizat. Output: {roof_3d_dir}")
            else:
                print("\n❌ Workflow roof 3D a eșuat sau nu s-au găsit wall masks.", file=sys.stderr)
                sys.exit(1)
            return
        except PlansListError as e:
            print(f"   ⚠️ [roof] {e} – folosesc fallback din scale/", flush=True)
        except Exception as e:
            print(f"   ⚠️ [roof] Eroare engine: {e} – folosesc fallback din scale/", flush=True)

    # 2) Fallback: ordonare după arie (parter = mai mare = floor_0)
    ordered_plan_paths = _discover_plans_from_scale(run_id)
    if not ordered_plan_paths:
        print("❌ [roof] Nu s-au găsit planuri (nici runs/, nici scale/ cu 01_walls_from_coords.png).", file=sys.stderr)
        sys.exit(1)
    print("   [roof] Ordine etaje (fallback: după arie măștii, descrescător → parter=floor_0):")
    for i, (plan_id, path) in enumerate(ordered_plan_paths):
        print(f"      floor_{i} ← {plan_id}")

    ok = _run_workflow_with_ordered_paths(run_id, ordered_plan_paths, roof_3d_dir)
    if ok:
        print(f"\n✅ Workflow roof 3D finalizat. Output: {roof_3d_dir}")
    else:
        print("\n❌ Workflow roof 3D a eșuat.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
