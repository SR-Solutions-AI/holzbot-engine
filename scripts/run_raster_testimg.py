#!/usr/bin/env python3
"""
Rulează direct Raster (API + mască pereți) și Brute force pe imaginile din folderul testimg.

Utilizare:
  cd holzbot-engine && python scripts/run_raster_testimg.py
  python scripts/run_raster_testimg.py --input /cale/la/testimg --output /cale/output

Implicit: input = holzbot-roof/testimg (sibling al holzbot-dynamic), output = output/raster_testimg.
Necesită: GEMINI_API_KEY, RASTER_API_KEY în environment (sau .env).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ca să importăm din holzbot-engine
_ENGINE = Path(__file__).resolve().parents[1]
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))
os.chdir(_ENGINE)

# Încarcă .env dacă există
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

    default_input = _ENGINE.parent / "holzbot-roof" / "testimg"
    default_output = _ENGINE / "output" / "raster_testimg"

    parser = argparse.ArgumentParser(description="Raster + Brute force pe imagini din testimg")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=default_input,
        help=f"Folder cu imagini (default: {default_input})",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=default_output,
        help=f"Folder de output (default: {default_output})",
    )
    parser.add_argument(
        "--plan-id",
        type=str,
        default="plan_01",
        help="Prefix pentru subfoldere (plan_01, plan_02, ...)",
    )
    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_base = args.output.resolve()

    if not input_dir.is_dir():
        print(f"❌ Folderul de input nu există: {input_dir}")
        print(f"   Creează folderul sau folosește: --input /cale/la/testimg")
        return 1

    exts = (".jpg", ".jpeg", ".png")
    images = sorted(
        [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in exts],
        key=lambda p: p.name,
    )
    if not images:
        print(f"❌ Nu am găsit imagini (extensii: {exts}) în {input_dir}")
        return 1

    if not os.environ.get("RASTER_API_KEY"):
        print("❌ RASTER_API_KEY lipsește. Setează variabila sau adaugă în .env.")
        return 1
    if not os.environ.get("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY lipsește. Setează variabila sau adaugă în .env.")
        return 1

    from cubicasa_detector.jobs import run_cubicasa_phase1, run_cubicasa_phase2

    output_base.mkdir(parents=True, exist_ok=True)
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_base}")
    print(f"🖼️  Imagini: {len(images)}")
    print()

    for i, img_path in enumerate(images, start=1):
        plan_id = f"{args.plan_id}_{i:02d}" if len(images) > 1 else args.plan_id
        work_dir = output_base / plan_id
        work_dir.mkdir(parents=True, exist_ok=True)
        steps_dir = work_dir / "cubicasa_steps"
        steps_dir.mkdir(parents=True, exist_ok=True)

        print(f"{'='*60}")
        print(f"[{i}/{len(images)}] {img_path.name} → {plan_id}")
        print(f"{'='*60}")

        try:
            # Phase 1: Raster API + AI walls → 02_ai_walls_closed.png
            run_cubicasa_phase1(
                plan_image=img_path,
                output_dir=work_dir,
            )
            # Phase 2: Brute force + overlay + crop + walls from coords
            run_cubicasa_phase2(output_dir=work_dir)
            print(f"✅ {plan_id}: Raster + Brute force finalizat.")
            print(f"   Rezultate: {steps_dir / 'raster'} și {steps_dir / 'raster_processing'}")
        except Exception as e:
            print(f"❌ {plan_id}: Eroare: {e}")
            import traceback
            traceback.print_exc()

    print()
    print(f"✅ Gata. Output: {output_base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
