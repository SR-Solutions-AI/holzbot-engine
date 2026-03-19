#!/usr/bin/env python3
"""
Reapelează Gemini rooms per crop (1 apel / imagine, fără batch) pe crop-urile existente,
actualizează room_scales.json, regenerează detections_review_data și imaginea cu labels.
Pipeline: Gemini (strict OCR) → validare → fallback "Raum".

  cd holzbot-engine
  .venv/bin/python scripts/rerun_rooms_labels.py [path]

  path = cubicasa_steps sau .../raster_processing/walls_from_coords
  Necesită GEMINI_API_KEY.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINE_ROOT))


def _load_dotenv():
    for p in (ENGINE_ROOT / ".env", Path.cwd() / ".env", Path.home() / ".env"):
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            except Exception:
                pass
            break


def _collect_room_batch_paths(walls_from_coords: Path) -> list[Path]:
    by_index = {}
    for f in walls_from_coords.glob("room_*_batch.png"):
        m = re.match(r"room_(\d+)_batch\.png", f.name, re.I)
        if m:
            by_index[int(m.group(1))] = f
    return [by_index[i] for i in sorted(by_index)]


def main() -> int:
    _load_dotenv()
    if len(sys.argv) >= 2:
        p = Path(sys.argv[1]).resolve()
        if (p / "walls_from_coords").exists():
            p = p / "walls_from_coords"
        elif p.name == "walls_from_coords" and (p / "room_scales.json").exists():
            pass
        else:
            p = p / "raster_processing" / "walls_from_coords" if (p / "raster_processing").exists() else p
        walls_from_coords = p
    else:
        # default: ultimul run plan_01
        out = ENGINE_ROOT / "output"
        walls_from_coords = None
        for run_dir in sorted(out.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            wfc = run_dir / "scale" / "plan_01_cluster_1" / "cubicasa_steps" / "raster_processing" / "walls_from_coords"
            if wfc.exists() and (wfc / "room_scales.json").exists():
                walls_from_coords = wfc
                break
        if not walls_from_coords or not walls_from_coords.exists():
            print("Nu s-a găsit walls_from_coords cu room_scales.json. Da path: cubicasa_steps sau .../walls_from_coords")
            return 1

    paths = _collect_room_batch_paths(walls_from_coords)
    if not paths:
        print(f"Niciun room_*_batch.png în {walls_from_coords}")
        return 1

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY_1")
    if not api_key:
        print("Setează GEMINI_API_KEY")
        return 1

    from cubicasa_detector.scale_detection import call_gemini_rooms_per_crop

    print(f"Apel Gemini per crop pentru {len(paths)} camere (1 apel / imagine)...")
    per_crop_results = call_gemini_rooms_per_crop([str(p) for p in paths], api_key)
    if not per_crop_results or len(per_crop_results) != len(paths):
        print(f"Gemini a returnat {len(per_crop_results) if per_crop_results else 0} (așteptat {len(paths)})")
        return 1

    rs_path = walls_from_coords / "room_scales.json"
    with open(rs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rooms = data.get("rooms") or {}
    room_scales = data.get("room_scales") or {}
    for i, res in enumerate(per_crop_results):
        key = str(i)
        rn = (res.get("room_name") or "").strip() or f"Room_{i}"
        rt = (res.get("room_type") or "Raum").strip() or "Raum"
        if key in rooms:
            rooms[key]["room_name"] = rn
            rooms[key]["room_type"] = rt
        if key in room_scales:
            room_scales[key]["room_name"] = rn
            room_scales[key]["room_type"] = rt
    data["rooms"] = rooms
    data["room_scales"] = room_scales
    with open(rs_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Actualizat: {rs_path}")

    raster_dir = walls_from_coords.parent.parent / "raster"
    if not raster_dir.exists():
        raster_dir = walls_from_coords.parent / "raster"
    if not raster_dir.exists():
        print("Nu s-a găsit raster dir.")
        return 1

    from cubicasa_detector.raster_api import save_detections_review_image
    print("Regenerez detections_review_data și imagini...")
    save_detections_review_image(raster_dir)
    print("Regenerez imaginea cu labels...")
    argv_save = sys.argv
    sys.argv = ["draw_room_labels_overlay.py", str(raster_dir)]
    try:
        from scripts.draw_room_labels_overlay import main as draw_main
        return draw_main()
    finally:
        sys.argv = argv_save


if __name__ == "__main__":
    sys.exit(main())
