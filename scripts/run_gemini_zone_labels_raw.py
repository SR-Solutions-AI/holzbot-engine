#!/usr/bin/env python3
"""
Apelează DOAR Gemini pentru zone labels SAU blacklist words și salvează răspunsul brut într-un JSON.

Moduri: zone_labels (garage, terasa, balcon, intrare_acoperita) | blacklist (pool, piscina, etc.)

Utilizare:
  # Zone labels
  python scripts/run_gemini_zone_labels_raw.py --mode zone_labels --image plan.png --output gemini_zone_labels_raw_response.json

  # Blacklist (termeni din cubicasa_detector/blacklist_words.json)
  python scripts/run_gemini_zone_labels_raw.py --mode blacklist --image plan.png --output gemini_blacklist_raw_response.json

Necesită: GEMINI_API_KEY în environment sau .env.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
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


def _load_blacklist_terms():
    path = _ENGINE / "cubicasa_detector" / "blacklist_words.json"
    if not path.exists():
        return ["pool", "Pool", "piscina", "Schwimmbad", "swimming", "bazin"]
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("blacklist_terms", []) or ["pool", "piscina", "Schwimmbad"]
    except Exception:
        return ["pool", "piscina", "Schwimmbad"]


def _build_blacklist_prompt(terms: list[str]) -> str:
    terms_str = ", ".join(repr(t) for t in (terms[:25] if len(terms) > 25 else terms))
    return f"""You are analyzing an architectural floor plan image. Find any of these BLACKLIST words/labels written on the plan (they usually indicate a pool or similar zone we want to detect):

Terms to search for (in any language): {terms_str}

For each occurrence you find, return the CENTER of that text in percentage coordinates:
- x_center_pct = (center_x / image_width) * 100
- y_center_pct = (center_y / image_height) * 100
(both 0–100).

Use "label" as the exact word or a normalized form (e.g. "pool", "Pool", "piscina") that you found.

Return ONLY a JSON object (no markdown):
{{
  "detections": [
    {{ "label": "pool", "x_center_pct": 50.0, "y_center_pct": 30.0 }}
  ]
}}
If none of these words appear on the plan, return {{ "detections": [] }}.
"""


# Același prompt ca în scale_detection.GEMINI_PROMPT_ZONE_LABELS (fără a încărca torch etc.)
GEMINI_PROMPT_ZONE_LABELS = """
You are analyzing an architectural floor plan image. Your task is to READ THE TEXT WRITTEN ON THE PLAN (room labels, zone names, annotations) and locate the following zone types. For each one you find, return the CENTER of that written label in percentage coordinates: x_center_pct = (center_x / image_width) * 100, y_center_pct = (center_y / image_height) * 100, both in range 0–100.

Search the plan for these zone labels. Look for the exact words or common abbreviations as written on the drawing (German, English, Romanian, etc.):

1. **garage** – Look for: Garage, Garaj, Gar., Carport, Parking, Stellplatz, Stellpl., Garáž, Garaž, Parkplatz, etc. Return the center of that text.

2. **terasa** – Look for: Terrasse, Terr., Terrace, Terasa, Terasă, Patio, Garten (when it is a terrace/patio area), etc. Return the center of that text.

3. **balcon** – Look for: Balkon, Balk., Balcon, Balcony, Loggia, etc. Return the center of that text.

4. **intrare_acoperita** – Look ONLY for: "Eingang überdacht", "überdachter Eingang", "Eing. überdacht", or equivalent "covered entrance". Do NOT match: Foyer, Flur, Diele, Hall, Eingang alone, Entrance, Corridor. We need specifically the covered entrance zone.

Rules:
- Scan the whole image for these words/abbreviations. Return the geometric center of the text bounding box for each label found.
- If a zone is not present on the plan, omit it from the array.
- Output ONLY valid JSON. Use exactly these label strings: "garage", "terasa", "balcon", "intrare_acoperita".

Return ONLY a JSON object (no markdown, no explanation):
{
  "detections": [
    { "label": "garage", "x_center_pct": 25.5, "y_center_pct": 62.0 },
    { "label": "terasa", "x_center_pct": 70.0, "y_center_pct": 15.0 }
  ]
}
"""
GEMINI_MODEL = "gemini-3-flash-preview"


def main():
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="Apel Gemini zone labels sau blacklist: salvează răspunsul brut într-un JSON.",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=("zone_labels", "blacklist"),
        default="zone_labels",
        help="zone_labels = garage/terasa/balcon/intrare_acoperita; blacklist = termeni din blacklist_words.json",
    )
    parser.add_argument(
        "--image", "-i",
        type=Path,
        required=True,
        help="Cale către imaginea de plan (ex: plan_for_gemini_zones.png sau 00_original.png)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Fișier JSON de output (default: gemini_zone_labels_raw_response.json sau gemini_blacklist_raw_response.json)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="maxOutputTokens pentru API (default: 2048)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = Path("gemini_blacklist_raw_response.json" if args.mode == "blacklist" else "gemini_zone_labels_raw_response.json")

    image_path = args.image.resolve()
    if not image_path.exists():
        print(f"❌ Imagine negăsită: {image_path}")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY_1")
    if not api_key:
        print("❌ Setează GEMINI_API_KEY în environment sau .env")
        sys.exit(1)

    if args.mode == "blacklist":
        blacklist_terms = _load_blacklist_terms()
        prompt = _build_blacklist_prompt(blacklist_terms) + "\n\nIMPORTANT: Răspunde DOAR cu un obiect JSON valid, fără text înainte sau după."
        print(f"   [BLACKLIST] Termeni: {blacklist_terms[:6]}{'...' if len(blacklist_terms) > 6 else ''}")
    else:
        prompt = GEMINI_PROMPT_ZONE_LABELS + "\n\nIMPORTANT: Răspunde DOAR cu un obiect JSON valid, fără text înainte sau după."

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = image_path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/jpeg")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    gen_config = {
        "temperature": 0.0,
        "maxOutputTokens": args.max_tokens,
        "responseMimeType": "application/json",
    }
    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": mime_type, "data": image_data}},
                {"text": prompt},
            ]
        }],
        "generationConfig": gen_config,
    }

    import requests
    headers = {"Content-Type": "application/json"}
    print(f"📤 [{args.mode.upper()}] Trimit imagine: {image_path.name} ({image_path.stat().st_size // 1024} KB)")
    print(f"   maxOutputTokens: {args.max_tokens}")
    response = requests.post(url, json=payload, headers=headers, timeout=60)

    out_data = {
        "mode": args.mode,
        "image_path": str(image_path),
        "http_status": response.status_code,
        "raw_text": None,
        "raw_text_length": 0,
        "error": None,
    }

    if response.status_code != 200:
        print(f"⚠️  Gemini HTTP {response.status_code}")
        out_data["error"] = response.text[:500] if response.text else "no body"
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Salvat: {args.output}")
        sys.exit(1)

    result = response.json()
    if "candidates" not in result or not result["candidates"]:
        out_data["error"] = "no candidates in response"
        out_data["full_response_keys"] = list(result.keys()) if isinstance(result, dict) else []
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print("⚠️  Răspuns fără candidates")
        print(f"💾 Salvat: {args.output}")
        sys.exit(1)

    text = result["candidates"][0]["content"]["parts"][0].get("text", "").strip()
    out_data["raw_text"] = text
    out_data["raw_text_length"] = len(text)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f"✅ [{args.mode.upper()}] Răspuns primit, lungime: {len(text)} caractere")
    print(f"   Primele 200 caractere: {text[:200]!r}")
    print(f"💾 Salvat: {args.output}")


if __name__ == "__main__":
    main()
