# new/runner/scale/openai_scale.py
# Detectare scară cu Gemini (REST). Numele fișierului rămâne pentru compatibilitate importuri vechi.
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from common.gemini_rest import DEFAULT_GEMINI_MODEL, gemini_api_key, generate_json, image_part_from_path


SCALE_DETECTION_PROMPT = """
Imaginea atașată este un plan arhitectural generic, utilizat doar pentru analiză vizuală și estimare.
Scopul este să **estimezi vizual scara** imaginii (metri/pixel) pe baza oricăror informații observabile:
- etichete numerice (ex: dimensiuni în metri),
- text cu suprafețe (m²),
- scară grafică,
- sau proporții între camere.

Nu trebuie să efectuezi calcule exacte de măsurare, doar o **estimare logică bazată pe observații vizuale**.
Dacă există mai multe indicii, alege cea mai coerentă valoare și explică scurt metoda în JSON.

Returnează strict un JSON cu structura următoare:

{
  "image_width_px": <int>,
  "image_height_px": <int>,
  "reference_measurement": {
    "segment_label": "<string>",
    "pixel_length_estimated": <float>,
    "real_length_meters": <float>
  },
  "meters_per_pixel": <float>,
  "verification": {
    "room_example": {
      "label": "<string>",
      "approx_dimensions": "<string>",
      "expected_area": "<string>",
      "validation": "<string>"
    }
  }
}
"""


def detect_scale_with_openai(image_path: Path) -> dict[str, Any]:
    """
    Trimite imaginea planului către Gemini (vision) pentru detectare scară.

    Args:
        image_path: Path către imaginea planului (plan.jpg)

    Returns:
        Dict cu meters_per_pixel și detalii despre estimare
    """
    api_key = gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY lipsește din environment")

    print(f"  📐 Trimit {image_path.name} către Gemini ({DEFAULT_GEMINI_MODEL}) pentru detectare scară...")

    parts = [
        image_part_from_path(image_path),
        {"text": SCALE_DETECTION_PROMPT + "\n\nReturn ONLY valid JSON matching the schema above."},
    ]
    try:
        result = generate_json(
            api_key,
            parts,
            system_instruction=(
                "Ești un expert în arhitectură și interpretare vizuală a planurilor de construcții. "
                "Estimează scara imaginilor în mod descriptiv și rațional. Răspunde doar cu JSON valid."
            ),
            model=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            temperature=0.0,
            max_output_tokens=4096,
            timeout=120,
        )
    except Exception as e:
        raise RuntimeError(f"Eroare la apelul Gemini: {e}") from e

    if not isinstance(result, dict):
        raise ValueError("Răspunsul Gemini nu este un obiect JSON")

    if "meters_per_pixel" not in result:
        raise ValueError("Răspunsul Gemini nu conține cheia 'meters_per_pixel'")

    print(f"  ✅ Scară detectată: {float(result['meters_per_pixel']):.6f} m/pixel")

    return result


# Alias explicit pentru cod nou
detect_scale_with_gemini = detect_scale_with_openai
