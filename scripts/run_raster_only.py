#!/usr/bin/env python3
"""
Rulează doar apelul Raster API: trimite imaginea la Raster și salvează răspunsul.
Fără Phase 1 (AI walls), fără brute force, fără overlay pe original.

Output în <output>/<plan_id>/cubicasa_steps/raster/:
  - response.json       (răspunsul complet de la API)
  - raster_request.png  (imaginea trimisă)
  - raster_response.png / processed_image.jpg (imaginea returnată de API)
  - output.svg, output.dxf (dacă API le returnează)
  - raster_request_info.json

Utilizare:
  # O singură imagine
  python scripts/run_raster_only.py --input plan.png --output output/raster_only

  # Toate imaginile dintr-un folder
  python scripts/run_raster_only.py --input folder_cu_planuri --output output/raster_only

Necesită: RASTER_API_KEY în environment (sau .env).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

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


def main():
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="Doar Raster API: trimite imaginea și salvează răspunsul (response.json + imagini)."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Cale către o imagine (jpg/png) sau folder cu imagini",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("output/raster_only"),
        help="Folder de output (default: output/raster_only)",
    )
    parser.add_argument(
        "--plan-id",
        type=str,
        default="plan_01",
        help="Prefix pentru subfoldere când input e folder (plan_01, plan_02, ...)",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_base = args.output.resolve()

    if not input_path.exists():
        print(f"❌ Nu există: {input_path}")
        return 1

    if not os.environ.get("RASTER_API_KEY"):
        print("❌ RASTER_API_KEY lipsește. Setează variabila sau adaugă în .env.")
        return 1

    # Colectăm liste de (plan_id, image_path)
    if input_path.is_file():
        exts = (".jpg", ".jpeg", ".png")
        if input_path.suffix.lower() not in exts:
            print(f"❌ Fișierul nu e imagine (folosește .jpg sau .png): {input_path}")
            return 1
        items = [(args.plan_id, input_path)]
    else:
        exts = (".jpg", ".jpeg", ".png")
        images = sorted(
            [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in exts],
            key=lambda p: p.name,
        )
        if not images:
            print(f"❌ Nu am găsit imagini (extensii: {exts}) în {input_path}")
            return 1
        items = [
            (f"{args.plan_id}_{i:02d}" if len(images) > 1 else args.plan_id, img)
            for i, img in enumerate(images, start=1)
        ]

    from cubicasa_detector.raster_api import call_raster_api, save_overlay_on_original_from_response

    output_base.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output: {output_base}")
    print(f"🖼️  Imagini: {len(items)}\n")

    for plan_id, img_path in items:
        print(f"{'='*60}")
        print(f"[{plan_id}] {img_path.name}")
        print(f"{'='*60}")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Nu am putut citi imaginea: {img_path}")
            continue

        work_dir = output_base / plan_id
        work_dir.mkdir(parents=True, exist_ok=True)
        steps_dir = work_dir / "cubicasa_steps"
        steps_dir.mkdir(parents=True, exist_ok=True)

        result = call_raster_api(img, str(steps_dir))
        if result is not None:
            raster_dir = Path(result["raster_dir"])
            print(f"✅ Răspuns salvat în: {raster_dir}")
            print(f"   response.json, raster_request.png, raster_response.png, output.svg, output.dxf")
            overlay_req = raster_dir / "overlay_on_request.png"
            if overlay_req.exists():
                print(f"   overlay_on_request.png (pereți/camere/uși pe imaginea scale-down trimisă la Raster, 1:1)")
            else:
                print(f"   ⚠️ overlay_on_request.png nu s-a generat (verifică response.json / raster_request.png)")
            if save_overlay_on_original_from_response(raster_dir, img):
                print(f"   overlay_on_original.png (camere + uși pe planul original)")
            else:
                print(f"   ⚠️ overlay_on_original.png nu s-a putut genera")
        else:
            print(f"❌ Apel Raster eșuat pentru {img_path.name}")

    print(f"\n✅ Gata. Output: {output_base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
