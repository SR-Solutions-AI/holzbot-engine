#!/usr/bin/env python3
"""
Benchmark openings detection variants across historical commits.

Runs `scripts/run_walls_from_coords_only.py` for the same RUN_ID/PLAN_ID set in
multiple git revisions, then snapshots output artifacts and writes a summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable


ENGINE_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ENGINE_ROOT / "output"
WORKTREES_ROOT = ENGINE_ROOT / ".bench_worktrees"
DEFAULT_COMMITS = [
    "582eeb4",  # big update
    "d80278c",  # segm upd
    "37d5f43",  # speed upd
    "d318e5c",  # editor
    "56a7a8f",  # door win
    "HEAD",     # current
]


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _latest_run_id() -> str:
    runs = [p for p in OUTPUT_ROOT.iterdir() if p.is_dir()]
    if not runs:
        raise RuntimeError(f"No run directories found in {OUTPUT_ROOT}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0].name


def _plan_ids(run_id: str) -> list[str]:
    scale_dir = OUTPUT_ROOT / run_id / "scale"
    plans = [p.name for p in scale_dir.iterdir() if p.is_dir() and p.name.startswith("plan_")]
    if not plans:
        raise RuntimeError(f"No plans found in {scale_dir}")
    return sorted(plans)


def _ensure_worktree(commit: str) -> Path:
    WORKTREES_ROOT.mkdir(parents=True, exist_ok=True)
    safe = commit.replace("/", "_")
    wt = WORKTREES_ROOT / safe
    if not wt.exists():
        _run(["git", "worktree", "add", str(wt), commit], cwd=ENGINE_ROOT)
    output_link = wt / "output"
    if output_link.exists() or output_link.is_symlink():
        if output_link.resolve() != OUTPUT_ROOT.resolve():
            if output_link.is_symlink() or output_link.is_file():
                output_link.unlink()
            else:
                shutil.rmtree(output_link)
            output_link.symlink_to(OUTPUT_ROOT, target_is_directory=True)
    else:
        output_link.symlink_to(OUTPUT_ROOT, target_is_directory=True)
    return wt


def _count_types(detections_review_json: Path) -> Counter:
    data = json.loads(detections_review_json.read_text(encoding="utf-8"))
    doors = data.get("doors") or []
    c: Counter = Counter()
    for d in doors:
        if isinstance(d, dict):
            t = str(d.get("type") or "unknown").strip().lower()
            c[t] += 1
    c["total"] = len(doors)
    return c


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _iter_commits(commits_csv: str | None) -> Iterable[str]:
    if not commits_csv:
        return DEFAULT_COMMITS
    return [x.strip() for x in commits_csv.split(",") if x.strip()]


def _run_walls_for_commit(wt: Path, run_id: str, plan_id: str) -> tuple[bool, str]:
    code = r"""
from pathlib import Path
import os
import sys
import cv2

worktree = Path(sys.argv[1])
engine_root = Path(sys.argv[2])
run_id = sys.argv[3]
plan_id = sys.argv[4]
sys.path.insert(0, str(worktree))

output_root = engine_root / "output"
plan_dir = output_root / run_id / "scale" / plan_id
steps_dir = plan_dir / "cubicasa_steps"
raster_dir = steps_dir / "raster"
original_path = steps_dir / "00_original.png"
walls_path = steps_dir / "02_ai_walls_closed.png"

if not original_path.exists() or not walls_path.exists() or not (raster_dir / "response.json").exists():
    print("missing_required_inputs")
    raise SystemExit(3)

original_img = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
walls_mask = cv2.imread(str(walls_path), cv2.IMREAD_GRAYSCALE)
if original_img is None or walls_mask is None:
    print("image_read_error")
    raise SystemExit(4)
if walls_mask.shape[:2] != original_img.shape[:2]:
    walls_mask = cv2.resize(walls_mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)

from cubicasa_detector.raster_processing import generate_walls_from_room_coordinates
res = generate_walls_from_room_coordinates(
    original_img,
    {},
    raster_dir,
    str(steps_dir),
    gemini_api_key=os.environ.get("GEMINI_API_KEY"),
    initial_walls_mask_1px=walls_mask,
)
if not res:
    print("walls_from_coords_none")
    raise SystemExit(5)
try:
    from cubicasa_detector.raster_api import save_detections_review_image
    save_detections_review_image(raster_dir)
except Exception as e:
    print(f"save_detections_review_image_error:{e}")
print("ok")
"""
    cmd = [
        str(ENGINE_ROOT / "venv" / "bin" / "python"),
        "-c",
        code,
        str(wt),
        str(ENGINE_ROOT),
        run_id,
        plan_id,
    ]
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = ""
    p = subprocess.run(cmd, cwd=str(wt), env=env, text=True, capture_output=True, timeout=600)
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode == 0, out.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="", help="Run id to benchmark (default: latest in output/)")
    parser.add_argument("--plans", default="", help="Comma-separated plan ids (default: all plan_* from run/scale)")
    parser.add_argument(
        "--commits",
        default="",
        help="Comma-separated commits to test (default: curated list + HEAD)",
    )
    args = parser.parse_args()

    run_id = args.run_id.strip() or _latest_run_id()
    plans = [p.strip() for p in args.plans.split(",") if p.strip()] if args.plans else _plan_ids(run_id)
    commits = list(_iter_commits(args.commits))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_root = ENGINE_ROOT / "output" / run_id / "bench_openings" / ts
    report_root.mkdir(parents=True, exist_ok=True)

    venv_py = ENGINE_ROOT / "venv" / "bin" / "python"
    if not venv_py.exists():
        raise RuntimeError(f"Python venv not found: {venv_py}")

    rows: list[dict] = []
    for commit in commits:
        wt = _ensure_worktree(commit)
        for plan_id in plans:
            print(f"[bench] commit={commit} plan={plan_id} ...", flush=True)
            ok, logs = _run_walls_for_commit(wt, run_id, plan_id)

            det_json = OUTPUT_ROOT / run_id / "scale" / plan_id / "cubicasa_steps" / "raster" / "detections_review_data.json"
            counts = _count_types(det_json) if det_json.exists() else Counter()
            row = {
                "commit": commit,
                "plan_id": plan_id,
                "ok": int(ok),
                "total": counts.get("total", 0),
                "door": counts.get("door", 0),
                "window": counts.get("window", 0),
                "stairs": counts.get("stairs", 0),
                "garage_door": counts.get("garage_door", 0),
                "unknown": counts.get("unknown", 0),
                "note": logs[:500],
            }
            rows.append(row)
            print(
                f"[bench] commit={commit} plan={plan_id} ok={ok} "
                f"total={row['total']} door={row['door']} window={row['window']} stairs={row['stairs']}",
                flush=True,
            )

            snap_dir = report_root / "snapshots" / commit / plan_id
            _copy_if_exists(det_json, snap_dir / "detections_review_data.json")
            _copy_if_exists(
                OUTPUT_ROOT / run_id / "scale" / plan_id / "cubicasa_steps" / "raster" / "detections_review_base.png",
                snap_dir / "detections_review_base.png",
            )
            _copy_if_exists(
                OUTPUT_ROOT / run_id / "scale" / plan_id / "cubicasa_steps" / "raster" / "detections_review_doors.png",
                snap_dir / "detections_review_doors.png",
            )
            _copy_if_exists(
                OUTPUT_ROOT / run_id / "scale" / plan_id / "cubicasa_steps" / "raster" / "detections_review_rooms.png",
                snap_dir / "detections_review_rooms.png",
            )
            _copy_if_exists(
                OUTPUT_ROOT / run_id / "scale" / plan_id / "cubicasa_steps" / "raster_processing" / "openings" / "01_openings.png",
                snap_dir / "01_openings.png",
            )
            _copy_if_exists(
                OUTPUT_ROOT / run_id / "scale" / plan_id / "cubicasa_steps" / "raster_processing" / "walls_from_coords" / "01_walls_from_coords.png",
                snap_dir / "01_walls_from_coords.png",
            )

    (report_root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (report_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["commit", "plan_id", "ok", "total", "door", "window", "stairs", "garage_door", "unknown", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Benchmark complete. Report: {report_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

