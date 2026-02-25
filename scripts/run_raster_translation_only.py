#!/usr/bin/env python3
"""
Script nou: trimite imaginea la Raster, generează 02_ai_walls_closed pe imaginea scale-down
trimisă la Raster, apoi aplică brute force DOAR cu translații (fără scale) între:
  - masca de pereți primită din răspunsul Raster (api_walls_from_json)
  - masca 02_ai_walls_closed generată pe aceeași imagine scale-down.

Flux:
  1. Încarcă imaginea, preprocesează (eliminare linii subțiri) și scale-down (max 1000px).
  2. Salvează imaginea scale-down și rulează Phase 1 (Raster API + AI walls) cu ea ca input.
     → Rezultat: 02_ai_walls_closed.png la dimensiunea request + raster/ cu response.json.
  3. Construiește masca API din response.json (api_walls_from_json.png).
  4. Brute force doar translații între cele două măști (aceeași dimensiune).
  5. Salvează: translation_only_config.json, translation_only_overlay.png în raster/brute_steps/.

Utilizare:
  python scripts/run_raster_translation_only.py --input ../holzbot-roof/testimg --output output/raster_translation_only
  python scripts/run_raster_translation_only.py --input plan.png --output output/raster_translation_only --plan-id plan_01
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

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


def _preprocess_for_raster(img: np.ndarray) -> np.ndarray:
    """Aceeași preprocesare ca în call_raster_api: eliminare linii subțiri (inpaint)."""
    api_img = img.copy()
    gray = cv2.cvtColor(api_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thin_lines_mask = np.zeros_like(gray)
    min_line_area = (gray.shape[0] * gray.shape[1]) * 0.0001
    for contour in contours:
        if cv2.contourArea(contour) < min_line_area:
            cv2.drawContours(thin_lines_mask, [contour], -1, 255, -1)
    api_img = cv2.inpaint(api_img, thin_lines_mask, 3, cv2.INPAINT_TELEA)
    return api_img


def _scale_to_max_side(img: np.ndarray, max_side: int = 1000) -> np.ndarray:
    """Scale-down ca la Raster: max 1000px pe latura lungă."""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="Raster API + 02_ai_walls pe imaginea scale-down + brute force doar translații."
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Imagine sau folder cu imagini")
    parser.add_argument("--output", "-o", type=Path, default=Path("output/raster_translation_only"), help="Folder output")
    parser.add_argument("--plan-id", type=str, default="plan_01", help="Prefix plan când input e folder")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_base = args.output.resolve()

    if not input_path.exists():
        print(f"❌ Nu există: {input_path}")
        return 1

    if not os.environ.get("RASTER_API_KEY"):
        print("❌ RASTER_API_KEY lipsește. Setează variabila sau .env.")
        return 1
    if not os.environ.get("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY lipsește (necesar pentru Phase 1 / AI walls). Setează variabila sau .env.")
        return 1

    # Colectăm (plan_id, image_path)
    if input_path.is_file():
        if input_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            print(f"❌ Nu e imagine: {input_path}")
            return 1
        items = [(args.plan_id, input_path)]
    else:
        images = sorted(
            [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")],
            key=lambda p: p.name,
        )
        if not images:
            print(f"❌ Nu am găsit imagini în {input_path}")
            return 1
        items = [
            (f"{args.plan_id}_{i:02d}" if len(images) > 1 else args.plan_id, img)
            for i, img in enumerate(images, start=1)
        ]

    from cubicasa_detector.jobs import run_cubicasa_phase1
    from cubicasa_detector.raster_api import (
        build_api_walls_mask_from_json,
        brute_force_translation_only,
    )

    output_base.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output: {output_base}")
    print(f"🖼️  Imagini: {len(items)}\n")

    for plan_id, img_path in items:
        print(f"{'='*60}")
        print(f"[{plan_id}] {img_path.name}")
        print(f"{'='*60}")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Nu am putut citi: {img_path}")
            continue

        work_dir = output_base / plan_id
        work_dir.mkdir(parents=True, exist_ok=True)
        steps_dir = work_dir / "cubicasa_steps"
        steps_dir.mkdir(parents=True, exist_ok=True)

        # 1) Preprocesare + scale-down (identic cu ce se trimite la Raster)
        print("   🧹 Preprocesare + scale-down (max 1000px)...")
        api_img = _preprocess_for_raster(img)
        request_img = _scale_to_max_side(api_img, 1000)
        request_input_path = work_dir / "request_input.png"
        cv2.imwrite(str(request_input_path), request_img)
        req_h, req_w = request_img.shape[:2]
        print(f"      Dimensiune request: {req_w}x{req_h}")

        # 2) Run Phase 1 cu imaginea scale-down → 02_ai_walls_closed la request size + apel Raster
        print("   🤖 Phase 1 (Raster API + AI walls) pe imaginea scale-down...")
        try:
            run_cubicasa_phase1(request_input_path, work_dir)
        except Exception as e:
            print(f"❌ Phase 1 eșuat: {e}")
            import traceback
            traceback.print_exc()
            continue

        ac_path = steps_dir / "02_ai_walls_closed.png"
        if not ac_path.exists():
            print(f"❌ Lipsă {ac_path.name} după Phase 1.")
            continue

        raster_dir = steps_dir / "raster"
        if not raster_dir.exists() or not (raster_dir / "response.json").exists():
            print(f"❌ Lipsă raster/ sau response.json.")
            continue

        # 3) Ref walls = 02_ai_walls_closed (deja la dimensiunea request)
        ref_walls = cv2.imread(str(ac_path), cv2.IMREAD_GRAYSCALE)
        if ref_walls is None:
            print(f"❌ Nu am putut citi {ac_path.name}")
            continue
        _, ref_binary = cv2.threshold(ref_walls, 127, 255, cv2.THRESH_BINARY)

        # 4) Mască API din JSON (request space)
        request_info_path = raster_dir / "raster_request_info.json"
        if request_info_path.exists():
            with open(request_info_path, "r") as f:
                ri = json.load(f)
            rw = ri.get("request_w") or ref_binary.shape[1]
            rh = ri.get("request_h") or ref_binary.shape[0]
        else:
            rw, rh = ref_binary.shape[1], ref_binary.shape[0]

        api_walls = build_api_walls_mask_from_json(raster_dir, rw, rh)
        if api_walls is None:
            print("❌ Nu s-a putut construi api_walls din response.json.")
            continue
        if api_walls.shape != ref_binary.shape:
            api_walls = cv2.resize(api_walls, (ref_binary.shape[1], ref_binary.shape[0]), interpolation=cv2.INTER_NEAREST)
        _, api_binary = cv2.threshold(api_walls, 127, 255, cv2.THRESH_BINARY)

        # 5) Brute force doar translații
        print("   📐 Brute force doar translații (fără scale)...")
        result = brute_force_translation_only(ref_binary, api_binary)
        if result is None:
            print("❌ brute_force_translation_only a returnat None.")
            continue

        # 6) Salvare rezultate în raster/brute_steps/
        brute_steps_dir = raster_dir / "brute_steps"
        brute_steps_dir.mkdir(parents=True, exist_ok=True)

        config_path = brute_steps_dir / "translation_only_config.json"
        with open(config_path, "w") as f:
            json.dump({
                "position": list(result["position"]),
                "score": result["score"],
                "direction": "translation_only",
                "template_size": list(result["template_size"]),
            }, f, indent=2)
        print(f"      📄 Salvat: {config_path.relative_to(raster_dir)}")

        # Overlay: ref = roșu, api deplasat = verde
        tx, ty = result["position"]
        h_r, w_r = ref_binary.shape[:2]
        overlay_req = np.zeros((h_r, w_r, 3), dtype=np.uint8)
        overlay_req[:, :, 2] = ref_binary
        x_dst = max(0, tx)
        y_dst = max(0, ty)
        x_src = max(0, -tx)
        y_src = max(0, -ty)
        w_c = min(w_r - x_dst, api_binary.shape[1] - x_src)
        h_c = min(h_r - y_dst, api_binary.shape[0] - y_src)
        if w_c > 0 and h_c > 0:
            overlay_req[y_dst : y_dst + h_c, x_dst : x_dst + w_c, 0] = api_binary[y_src : y_src + h_c, x_src : x_src + w_c]
            overlay_req[y_dst : y_dst + h_c, x_dst : x_dst + w_c, 1] = api_binary[y_src : y_src + h_c, x_src : x_src + w_c]
        overlay_path = brute_steps_dir / "translation_only_overlay.png"
        cv2.imwrite(str(overlay_path), overlay_req)
        print(f"      📄 Salvat: {overlay_path.relative_to(raster_dir)}")

        print(f"      ✅ Offset (tx, ty) = {result['position']}, score = {result['score']:.2%}")
        print(f"✅ {plan_id}: gata. Rezultate în {raster_dir} (brute_steps/)")

    print(f"\n✅ Gata. Output: {output_base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
