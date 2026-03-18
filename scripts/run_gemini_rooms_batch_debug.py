#!/usr/bin/env python3
"""
Rulează DOAR apelul Gemini batch pentru camere (call_gemini_rooms_batch) pe crop-uri existente
și salvează: (1) JSON cu paths + răspuns, (2) imagini overlay cu index + room_type | room_name
pentru verificare vizuală că ordinea nu e amestecată.

Utilizare:
  # Din folder unde există room_*_batch.png sau room_*_temp_for_gemini.png (după un run)
  python scripts/run_gemini_rooms_batch_debug.py --dir output/RUN_ID/scale/plan_01_cluster_1/cubicasa_steps/raster_processing

  # Sau cu path-uri explicite (ordinea din listă = Image 1, 2, ... pentru Gemini)
  python scripts/run_gemini_rooms_batch_debug.py --paths crop0.png crop1.png crop2.png --output-dir ./gemini_rooms_debug

Necesită: GEMINI_API_KEY în environment sau .env.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_ENGINE = Path(__file__).resolve().parents[1]
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))
os.chdir(_ENGINE)


def _load_dotenv():
    for p in (_ENGINE / ".env", Path.cwd() / ".env", Path.home() / ".env"):
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


def _collect_room_crop_paths(dir_path: Path) -> list[Path]:
    """Caută room_0_batch.png, room_1_batch.png, ... sau room_*_temp_for_gemini.png (în dir sau subdir) și sortează după index."""
    dir_path = dir_path.resolve()
    if not dir_path.is_dir():
        return []
    by_index: dict[int, Path] = {}
    for pat in ("room_*_batch.png", "room_*_temp_for_gemini.png"):
        for f in dir_path.rglob(pat):
            m = re.match(r"room_(\d+)_(?:batch|temp_for_gemini)\.png", f.name, re.I)
            if m:
                idx = int(m.group(1))
                if idx not in by_index:
                    by_index[idx] = f
    return [by_index[i] for i in sorted(by_index)]


def main():
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="Rulează doar Gemini rooms batch pe crop-uri și salvează JSON + overlay-uri pentru debug.",
    )
    parser.add_argument(
        "--dir", "-d",
        type=Path,
        default=None,
        help="Folder unde sunt room_*_batch.png sau room_*_temp_for_gemini.png (ordine după index).",
    )
    parser.add_argument(
        "--paths", "-p",
        nargs="+",
        type=Path,
        default=None,
        help="Listă explicită de path-uri către crop-uri (ordinea = Image 1, 2, ...).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Folder unde se salvează gemini_rooms_batch_debug.json și overlay-urile (default: același ca --dir sau ./gemini_rooms_debug).",
    )
    args = parser.parse_args()

    if args.dir is None and not args.paths:
        print("❌ Specifică fie --dir (folder cu room_*_batch.png), fie --paths crop1.png crop2.png ...")
        sys.exit(1)
    if args.dir is not None and args.paths:
        print("❌ Folosește fie --dir, fie --paths, nu ambele.")
        sys.exit(1)

    if args.paths:
        paths = [Path(p).resolve() for p in args.paths]
        for p in paths:
            if not p.exists():
                print(f"❌ Fișier negăsit: {p}")
                sys.exit(1)
        base_dir = paths[0].parent if paths else Path.cwd()
    else:
        paths = _collect_room_crop_paths(args.dir)
        base_dir = args.dir.resolve()
        if not paths:
            print(f"❌ Nu am găsit niciun room_*_batch.png sau room_*_temp_for_gemini.png în {args.dir}")
            sys.exit(1)
        print(f"   📁 Găsite {len(paths)} crop-uri în {args.dir}: {[p.name for p in paths]}")

    out_dir = (args.output_dir or base_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY_1")
    if not api_key:
        print("❌ Setează GEMINI_API_KEY în environment sau .env")
        sys.exit(1)

    from cubicasa_detector.scale_detection import call_gemini_rooms_batch

    path_strs = [str(p) for p in paths]
    print(f"   🧠 Apel Gemini rooms batch pentru {len(path_strs)} imagini...")
    batch_results = call_gemini_rooms_batch(path_strs, api_key)

    if batch_results is None or len(batch_results) != len(paths):
        print(f"   ❌ Gemini a returnat {len(batch_results) if batch_results else 0} rezultate (așteptat {len(paths)}).")
        sys.exit(1)

    # Salvare JSON debug
    debug_data = {
        "paths": path_strs,
        "response": [
            {
                "room_name": r.get("room_name") or "",
                "area_m2": r.get("area_m2"),
                "room_type": r.get("room_type") or "Raum",
            }
            for r in batch_results
        ],
    }
    import json
    json_path = out_dir / "gemini_rooms_batch_debug.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, indent=2, ensure_ascii=False)
    print(f"   ✅ JSON salvat: {json_path}")

    # Overlay pe fiecare crop: index + room_type | room_name
    try:
        import cv2
        for idx, (path, r) in enumerate(zip(paths, batch_results)):
            img = cv2.imread(str(path))
            if img is None:
                continue
            room_type = (r.get("room_type") or "Raum").strip() or "Raum"
            room_name = (r.get("room_name") or "").strip() or ""
            area_m2 = r.get("area_m2")
            label = f"{idx}: {room_type}"
            if room_name:
                label += f" | {room_name}"
            if area_m2 is not None:
                try:
                    label += f"  {float(area_m2):.1f}m²"
                except (TypeError, ValueError):
                    pass
            # Text în colț
            h, w = img.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.4, min(w, h) / 400.0 * 0.5)
            thick = max(1, int(scale * 2))
            (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
            x1, y1 = 8, 8 + th + 4
            cv2.rectangle(img, (4, 4), (x1 + tw + 4, y1 + 4), (0, 0, 0), -1)
            cv2.rectangle(img, (4, 4), (x1 + tw + 4, y1 + 4), (0, 200, 0), 2)
            cv2.putText(img, label, (8, 8 + th), font, scale, (0, 255, 0), thick, cv2.LINE_AA)
            out_name = f"room_{idx}_gemini_debug.png"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), img)
        print(f"   ✅ Overlay-uri salvate în {out_dir}: room_0_gemini_debug.png ...")
    except Exception as e:
        print(f"   ⚠️ Overlay-uri nu s-au putut salva: {e}")

    # Afișare rezumat în consolă
    print("\n   Rezumat (ordine = Image 1, 2, ...):")
    for idx, r in enumerate(batch_results):
        rt = (r.get("room_type") or "Raum").strip() or "Raum"
        rn = (r.get("room_name") or "").strip()
        am = r.get("area_m2")
        am_str = f" {am}m²" if am is not None else ""
        print(f"      [{idx}] {Path(paths[idx]).name} -> room_type={rt!r}  room_name={rn!r}{am_str}")


if __name__ == "__main__":
    main()
