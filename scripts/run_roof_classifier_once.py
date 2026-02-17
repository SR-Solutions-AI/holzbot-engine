#!/usr/bin/env python3
"""Rulează o singură dată apelul Gemini pentru tip/unghi acoperiș, pe ultima rulare.
   Folosește doar PIL + google.generativeai (fără segmenter/cv2)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ca să importăm din holzbot-engine
_ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ENGINE))
os.chdir(_ENGINE)

RUN_ID = "50752e39-5c6f-4a05-8af1-fc2ced52ceb4"


def _collect_side_views(job_root: Path):
    out = []
    seg_base = job_root / "segmentation"
    if not seg_base.is_dir():
        return out
    for sv_dir in seg_base.glob("src_*/classified/side_views"):
        if sv_dir.is_dir():
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                out.extend(sv_dir.glob(ext))
    return sorted(out)


def _prep_for_vlm(img_path: Path, min_long_edge: int = 1280):
    """PIL-only (fără cv2)."""
    from PIL import Image, ImageFilter
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    long_edge = max(w, h)
    if long_edge < min_long_edge:
        scale = min_long_edge / float(long_edge)
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    return im


def _load_dotenv():
    """Încarcă .env în os.environ (fără pachet dotenv)."""
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


def _setup_gemini():
    """Inițializare Gemini fără a importa segmenter."""
    _load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        for model_name in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"):
            try:
                return genai.GenerativeModel(model_name, safety_settings=safety_settings)
            except Exception:
                continue
    except Exception as e:
        print(f"Eroare init Gemini: {e}", flush=True)
    return None


def main():
    from config.settings import JOBS_ROOT, RUNS_ROOT, load_plan_infos
    from roof.roof_type_classifier import (
        PROMPT_ROOF_TYPE_PER_FLOOR,
        _floor_names,
        _parse_roof_types_response,
    )
    STAGE_NAME = "roof"

    job_root = None
    for jdir in JOBS_ROOT.glob("*"):
        if jdir.is_dir() and RUN_ID in jdir.name:
            job_root = jdir
            break
    if job_root is None:
        job_root = JOBS_ROOT / RUN_ID

    if not job_root.is_dir():
        print(f"Job root nu există: {job_root}", flush=True)
        sys.exit(1)

    side_views = _collect_side_views(job_root)
    print(f"Side views găsite: {len(side_views)}", flush=True)
    for p in side_views:
        print(f"  - {p.name}", flush=True)

    if not side_views:
        print("Nu există side_view-uri. Ieșire.", flush=True)
        sys.exit(0)

    # num_floors_roof (excludem beci)
    try:
        plans = load_plan_infos(RUN_ID, stage_name=STAGE_NAME)
    except Exception as e:
        print(f"load_plan_infos: {e}. Folosesc num_floors=1.", flush=True)
        plans = []
    run_dir = RUNS_ROOT / RUN_ID
    basement_idx = None
    if (run_dir / "basement_plan_id.json").exists():
        try:
            data = json.loads((run_dir / "basement_plan_id.json").read_text(encoding="utf-8"))
            basement_idx = data.get("basement_plan_index")
        except Exception:
            pass
    plans_roof = [p for i, p in enumerate(plans) if i != basement_idx] if basement_idx is not None else plans
    num_floors_roof = len(plans_roof) if plans_roof else 1
    print(f"Num floors (roof): {num_floors_roof}", flush=True)

    gemini = _setup_gemini()
    if not gemini:
        print("Gemini client nu s-a inițializat (lipsă GEMINI_API_KEY?).", flush=True)
        sys.exit(1)

    floor_keys_list = ["parter"] + [f"etaj_{i}" for i in range(1, num_floors_roof)]
    prompt = PROMPT_ROOF_TYPE_PER_FLOOR.format(
        num_floors=num_floors_roof,
        floor_names=_floor_names(num_floors_roof),
        floor_keys=", ".join(floor_keys_list),
    )
    parts = [prompt]
    for p in side_views:
        if not p.exists():
            continue
        try:
            parts.append(_prep_for_vlm(p))
        except Exception as e:
            print(f"  Skip {p.name}: {e}", flush=True)
    if len(parts) < 2:
        print("Nicio imagine validă.", flush=True)
        sys.exit(1)

    gen_config = {
        "temperature": 0.0,
        "max_output_tokens": 512,
        "response_mime_type": "application/json",
    }
    print("Apel Gemini classify_roof_types_per_floor (doar tipuri)...", flush=True)
    try:
        response = gemini.generate_content(parts, generation_config=gen_config)
        text = (response.text or "").strip() if response and response.parts else ""
        if not text:
            print("Răspuns Gemini gol.", flush=True)
            sys.exit(0)
        types_result = _parse_roof_types_response(text, num_floors_roof)
        print(f"   [RoofTypeClassifier] Gemini: types={types_result}", flush=True)
        print(f"Rezultat types: {types_result}", flush=True)
        if len(types_result) == num_floors_roof:
            print("OK – apelul Gemini a returnat date complete.", flush=True)
        else:
            print("Gemini a returnat date incomplete (fallback va fi folosit în pipeline).", flush=True)
    except Exception as e:
        print(f"Eroare Gemini: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
