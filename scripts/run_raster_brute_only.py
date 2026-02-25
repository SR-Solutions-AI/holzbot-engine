#!/usr/bin/env python3
"""
Rulează doar pașii de brute force (fără Raster din nou, fără crop, walls_from_coords, garaj etc.).

Structura output raster (definită în detector / raster_api):
  <work_dir>/
    cubicasa_steps/
      00_original.png           # imaginea originală (necesară la pornirea phase 2 + overlay)
      02_ai_walls_closed.png    # pereți AI închisi (masca „original walls” pentru brute force)
      raster/
        api_walls_mask.png      # masca pereți de la RasterScan API (input la brute force)
        response.json           # răspuns API (folosit la overlay pe original)
        raster_request.png, raster_response.png, walls.png, doors.png, combined.png, overlay.png, ...
        brute_steps/            # creat de brute force (overlay per interval)
        brute_force_best_config.json, brute_force_best_overlay.png, walls_brute.png, ...

Mod 1 – folosește output existent (nu rulează Raster):
  python scripts/run_raster_brute_only.py --use-existing --output output/<RUN_ID>/scale
  Sau direct pe output/raster_brute_only dacă ai rulat deja Phase 1 acolo.
  Rulează doar brute force pe fiecare subdirector care are toate fișierele necesare (v. mai jos).

Mod 2 – imagini din folder (rulează Phase 1 + brute only):
  python scripts/run_raster_brute_only.py --input /cale/testimg --output /cale/output
  (implicit: input = holzbot-roof/testimg, output = output/raster_brute_only)

Necesită: GEMINI_API_KEY (și RASTER_API_KEY doar în mod 2).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
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


def main():
    _load_dotenv()

    default_input = _ENGINE.parent / "holzbot-roof" / "testimg"
    default_output = _ENGINE / "output" / "raster_brute_only"

    parser = argparse.ArgumentParser(
        description="Doar brute force (opțional pe output existent, fără Raster din nou)"
    )
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Folosește output existent: nu rula Raster, doar brute force pe planurile din --output (dir scale: output/RUN_ID/scale)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="(Deprecated: acum implicit nu folosim cache.) Recalculează mereu alinierea.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Folosește brute_force_best_config.json dacă există. Implicit: nu folosim cache (recalculăm mereu).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=default_output,
        help="Mod use-existing: director scale (output/RUN_ID/scale). Altfel: folder unde se scriu rezultatele.",
    )
    parser.add_argument("--input", "-i", type=Path, default=default_input, help="Folder cu imagini (doar fără --use-existing)")
    parser.add_argument("--plan-id", type=str, default="plan_01", help="Prefix subfoldere (doar fără --use-existing)")
    args = parser.parse_args()

    # Implicit nu folosim cache; doar cu --use-cache folosim brute_force_best_config.json
    no_cache = not args.use_cache

    output_base = args.output.resolve()

    if args.use_existing:
        # Mod: folosim output-ul existent, nu rulăm Raster
        if not output_base.is_dir():
            print(f"❌ Directorul de output nu există: {output_base}")
            print(f"   Exemplu: --output output/97c711c4-b589-4c89-a07d-aef522e7148c/scale")
            return 1

        if not os.environ.get("GEMINI_API_KEY"):
            print("❌ GEMINI_API_KEY lipsește.")
            return 1

        from cubicasa_detector.jobs import run_cubicasa_phase2_brute_only

        # Fișiere necesare pentru phase 2 brute-only (conform detector: 00_original, 02_ai_walls_closed, raster/api_walls_mask, raster/response.json)
        plan_dirs = sorted([d for d in output_base.iterdir() if d.is_dir()])
        if not plan_dirs:
            print(f"❌ Nu am găsit subdirectoare (planuri) în {output_base}")
            return 1

        to_run = []
        for plan_dir in plan_dirs:
            steps = plan_dir / "cubicasa_steps"
            raster_dir = steps / "raster"
            missing = []
            if not (steps / "00_original.png").exists():
                missing.append("00_original.png")
            if not (steps / "02_ai_walls_closed.png").exists():
                missing.append("02_ai_walls_closed.png")
            if not raster_dir.is_dir():
                missing.append("raster/")
            elif not (raster_dir / "api_walls_mask.png").exists():
                missing.append("raster/api_walls_mask.png")
            elif not (raster_dir / "response.json").exists():
                missing.append("raster/response.json")
            if not missing:
                to_run.append(plan_dir)
            else:
                print(f"   ⏭️  Omis {plan_dir.name}: lipsește {', '.join(missing)}")

        if not to_run:
            print(f"❌ Niciun plan din {output_base} nu are toate fișierele necesare.")
            print(f"   Necesare: 00_original.png, 02_ai_walls_closed.png, raster/, raster/api_walls_mask.png, raster/response.json")
            return 1

        print(f"📁 Output existent: {output_base}")
        print(f"🖼️  Planuri cu Raster deja rulat: {len(to_run)} (doar brute force)\n")

        for i, work_dir in enumerate(to_run, start=1):
            plan_id = work_dir.name
            print(f"{'='*60}")
            print(f"[{i}/{len(to_run)}] {plan_id}")
            print(f"{'='*60}")
            t0 = time.perf_counter()
            try:
                run_cubicasa_phase2_brute_only(output_dir=work_dir, no_cache=no_cache)
                elapsed = time.perf_counter() - t0
                print(f"✅ {plan_id}: brute force gata.")
                print(f"   ⏱️  Timp: {elapsed:.1f}s ({elapsed/60:.1f} min)")
                print(f"   Rezultate: {work_dir / 'cubicasa_steps' / 'raster'} (incl. brute_steps/)")
            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f"❌ {plan_id}: {e}")
                print(f"   ⏱️  Timp până la eroare: {elapsed:.1f}s")
                import traceback
                traceback.print_exc()

        print(f"\n✅ Gata. Output: {output_base}")
        return 0

    # Mod: imagini din folder → Phase 1 + brute only
    input_dir = args.input.resolve()
    if not input_dir.is_dir():
        print(f"❌ Folderul de input nu există: {input_dir}")
        return 1

    exts = (".jpg", ".jpeg", ".png")
    images = sorted(
        [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in exts],
        key=lambda p: p.name,
    )
    if not images:
        print(f"❌ Nu am găsit imagini în {input_dir}")
        return 1

    if not os.environ.get("RASTER_API_KEY"):
        print("❌ RASTER_API_KEY lipsește.")
        return 1
    if not os.environ.get("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY lipsește.")
        return 1

    from cubicasa_detector.jobs import run_cubicasa_phase1, run_cubicasa_phase2_brute_only

    output_base.mkdir(parents=True, exist_ok=True)
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_base}")
    print(f"🖼️  Imagini: {len(images)} (Phase 1 + brute force)\n")

    for i, img_path in enumerate(images, start=1):
        plan_id = f"{args.plan_id}_{i:02d}" if len(images) > 1 else args.plan_id
        work_dir = output_base / plan_id
        work_dir.mkdir(parents=True, exist_ok=True)
        steps_dir = work_dir / "cubicasa_steps"
        steps_dir.mkdir(parents=True, exist_ok=True)

        print(f"{'='*60}")
        print(f"[{i}/{len(images)}] {img_path.name} → {plan_id}")
        print(f"{'='*60}")

        t0 = time.perf_counter()
        try:
            run_cubicasa_phase1(plan_image=img_path, output_dir=work_dir)
            t1 = time.perf_counter()
            run_cubicasa_phase2_brute_only(output_dir=work_dir, no_cache=no_cache)
            elapsed = time.perf_counter() - t0
            phase2_time = time.perf_counter() - t1
            print(f"✅ {plan_id}: Raster API + brute force gata.")
            print(f"   ⏱️  Timp total: {elapsed:.1f}s ({elapsed/60:.1f} min), din care brute force: {phase2_time:.1f}s")
            print(f"   Rezultate: {steps_dir / 'raster'} (incl. brute_steps/)")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"❌ {plan_id}: {e}")
            print(f"   ⏱️  Timp până la eroare: {elapsed:.1f}s")
            import traceback
            traceback.print_exc()

    print(f"\n✅ Gata. Output: {output_base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
