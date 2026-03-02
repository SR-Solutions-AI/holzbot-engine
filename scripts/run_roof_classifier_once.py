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

# Încarcă doar roof_type_classifier fără roof.jobs (care cere cv2)
import importlib.util
_rfc_spec = importlib.util.spec_from_file_location(
    "roof_type_classifier",
    _ENGINE / "roof" / "roof_type_classifier.py",
)
_rfc = importlib.util.module_from_spec(_rfc_spec)
_rfc_spec.loader.exec_module(_rfc)

RUN_ID = "78842a74-2ef7-4af3-87f2-2fb8b419dbc1"


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
    """Simulare: același prompt și etichete ca în roof_type_classifier, apel direct Gemini, afișează răspunsul."""
    from config.settings import JOBS_ROOT, RUNS_ROOT, load_plan_infos

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
        print("Gemini client nu s-a inițializat (GEMINI_API_KEY?).", flush=True)
        sys.exit(1)

    floor_names_text = _rfc._floor_names(num_floors_roof)
    prompt = _rfc.PROMPT_ROOF_TYPE_PER_FLOOR.format(
        num_floors=num_floors_roof,
        floor_names=floor_names_text,
    )
    parts = [prompt]
    img_count = 0
    for p in side_views:
        if not p.exists():
            continue
        try:
            img_count += 1
            label = f"\n[Image {img_count} of {len(side_views)} side-view/cross-section]\n"
            parts.append(label)
            parts.append(_prep_for_vlm(p))
        except Exception as e:
            print(f"  Skip {p.name}: {e}", flush=True)
    if len(parts) < 2:
        print("Nicio imagine validă.", flush=True)
        sys.exit(1)

    print("Apel Gemini (același prompt ca în pipeline, cu etichete imagini)...", flush=True)
    gen_config = {"temperature": 0.2, "max_output_tokens": 2048}
    try:
        response = gemini.generate_content(parts, generation_config=gen_config)
        # Extrage textul din TOATE părțile (uneori .text e doar prima parte)
        text = ""
        if response and response.candidates:
            c = response.candidates[0]
            finish_reason = getattr(c, "finish_reason", None)
            print(f"  finish_reason: {finish_reason!r}", flush=True)
            if getattr(c, "content", None) and getattr(c.content, "parts", None):
                for part in c.content.parts:
                    text += getattr(part, "text", "") or ""
        if not text and response:
            text = (response.text or "").strip()
        text = text.strip()
        if not text:
            print("Răspuns Gemini gol.", flush=True)
            sys.exit(0)
        # Salvează răspunsul complet (uneori afișarea în terminal taie)
        raw_path = _ENGINE / "gemini_roof_response_raw.txt"
        raw_path.write_text(text, encoding="utf-8")
        print(f"\nRăspuns complet salvat în: {raw_path} ({len(text)} caractere)", flush=True)
        print("--- RĂSPUNS GEMINI (raw) ---", flush=True)
        print(text, flush=True)
        print("--- SFÂRȘIT RAW ---\n", flush=True)
        types_result = _rfc._parse_roof_types_response(text, num_floors_roof)
        data = None
        json_str = _rfc._extract_json_object(text)
        if json_str:
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                pass
        types_out, angles_out = _rfc._dict_to_floor_result(data or {}, num_floors_roof)
        if not types_out:
            types_out = types_result
        if not angles_out and types_out:
            for i in range(num_floors_roof):
                angles_out[i] = 0.0 if types_out.get(i) == "0_w" else 30.0
        print("=== REZULTAT SIMULARE ===", flush=True)
        print(f"  types:  {types_out}", flush=True)
        print(f"  angles: {angles_out}", flush=True)
    except Exception as e:
        print(f"Eroare: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
