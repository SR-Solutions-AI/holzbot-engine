# new/runner/perimeter/gemini_measure.py
from __future__ import annotations

import json
import os
import math
from pathlib import Path

from common.gemini_rest import DEFAULT_GEMINI_MODEL, gemini_api_key, generate_json, image_part_from_path


PERIMETER_PROMPT = """
You are analyzing an architectural floor plan (top-down view).

Your task is to estimate:
1. Total length of INTERIOR walls (walls between rooms, excluding exterior walls)
2. Total length of EXTERIOR walls (building perimeter/outline)
3. Total building PERIMETER (outer boundary length)

Use BOTH methods:

METHOD 1 - Pixel-based:
- Identify wall lines on the plan
- Estimate total length in pixels for each category
- Convert using scale: {meters_per_pixel:.6f} m/pixel
- Formula: length_m = length_px × meters_per_pixel

METHOD 2 - Proportion-based:
- Identify room dimensions (if labeled)
- Estimate wall lengths from building shape and proportions
- Calculate perimeter from total area: P ≈ 4√A

METHOD 3 - Calculate AVERAGE of both methods.

DEFINITIONS:
- Interior walls = walls between rooms (bathrooms, bedrooms, kitchen)
- Exterior walls = building outer walls
- Perimeter = total length of outer boundary

VALIDATION (typical single-family home 80-120 m²):
- Interior walls: 30-60 m
- Exterior walls: 30-50 m
- Perimeter: 30-45 m

CRITICAL: You MUST respond with ONLY valid JSON. No markdown, no explanations, ONLY JSON.

OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "scale_meters_per_pixel": {meters_per_pixel:.6f},
  "estimations": {{
    "by_pixels": {{
      "interior_meters": <float>,
      "exterior_meters": <float>,
      "total_perimeter_meters": <float>,
      "method_notes": "<string: explain measurement approach>"
    }},
    "by_proportion": {{
      "interior_meters": <float>,
      "exterior_meters": <float>,
      "total_perimeter_meters": <float>,
      "method_notes": "<string: explain logic>"
    }},
    "average_result": {{
      "interior_meters": <float>,
      "exterior_meters": <float>,
      "total_perimeter_meters": <float>
    }}
  }},
  "confidence": "high | medium | low",
  "verification_notes": "<string: consistency check>"
}}

REMEMBER: 
- Perimeter MUST be ≤ exterior walls length
- Interior walls typically 0.8-1.5× exterior walls
- ALL values MUST be realistic for a single-family home
- Output MUST be valid JSON ONLY (no markdown blocks, no text before/after)
"""


def _fallback_estimation(meters_per_pixel: float, house_area_m2: float = 100.0) -> dict:
    """
    Fallback estimation când GPT-4o refuză să analizeze imagini.
    Folosește formula simplă: P ≈ 4√A
    """
    perimeter_est = 4.0 * math.sqrt(house_area_m2)
    interior_est = perimeter_est * 1.2  # interior = ~1.2× perimetru
    exterior_est = perimeter_est * 1.0
    
    return {
        "scale_meters_per_pixel": meters_per_pixel,
        "estimations": {
            "by_pixels": {
                "interior_meters": interior_est,
                "exterior_meters": exterior_est,
                "total_perimeter_meters": perimeter_est,
                "method_notes": "Fallback estimation (Gemini refused or failed image analysis)"
            },
            "by_proportion": {
                "interior_meters": interior_est,
                "exterior_meters": exterior_est,
                "total_perimeter_meters": perimeter_est,
                "method_notes": f"Fallback: P ≈ 4√A, using estimated area {house_area_m2:.1f}m²"
            },
            "average_result": {
                "interior_meters": interior_est,
                "exterior_meters": exterior_est,
                "total_perimeter_meters": perimeter_est
            }
        },
        "confidence": "low",
        "verification_notes": "Fallback estimation used (API refused or failed image analysis)"
    }


def measure_perimeter_with_gemini(
    plan_image: Path,
    scale_data: dict
) -> dict:
    """
    Trimite planul la Gemini (vision) pentru măsurarea lungimilor pereților.

    Args:
        plan_image: Path către plan.jpg
        scale_data: Dict cu scale_result.json (conține meters_per_pixel)

    Returns:
        Dict cu structura de estimări perimetru
    """
    api_key = gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY lipsește din environment")

    meters_per_pixel = float(scale_data.get("meters_per_pixel", 0.0))
    if meters_per_pixel <= 0:
        raise ValueError("Scara invalidă în scale_result.json")

    print(f"       📐 Măsurare pereți cu Gemini ({DEFAULT_GEMINI_MODEL}, scala: {meters_per_pixel:.6f} m/px)...")

    parts = [
        image_part_from_path(plan_image),
        {"text": PERIMETER_PROMPT.format(meters_per_pixel=meters_per_pixel) + "\n\nReturn ONLY valid JSON as specified."},
    ]
    try:
        result = generate_json(
            api_key,
            parts,
            system_instruction=(
                "You are an expert in precise measurements on 2D architectural plans. "
                "You MUST respond with valid JSON only."
            ),
            model=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            temperature=0.0,
            max_output_tokens=4096,
            timeout=120,
        )
    except Exception as e:
        print(f"       ⚠️  Eroare la apelul Gemini: {e}")
        print("       🔄 Folosesc fallback estimation...")
        return _fallback_estimation(meters_per_pixel)

    if not isinstance(result, dict):
        print("       ⚠️  Răspuns Gemini invalid (non-object)")
        print("       🔄 Folosesc fallback estimation...")
        return _fallback_estimation(meters_per_pixel)

    # Validare structură
    if "estimations" not in result:
        print("       ⚠️  Răspunsul Gemini nu conține cheia 'estimations'")
        print("       🔄 Folosesc fallback estimation...")
        return _fallback_estimation(meters_per_pixel)
    
    avg = result["estimations"].get("average_result", {})
    int_m = avg.get("interior_meters", 0)
    ext_m = avg.get("exterior_meters", 0)
    per_m = avg.get("total_perimeter_meters", 0)
    
    print(f"       ✅ Măsurare completă:")
    print(f"          • Pereți interiori: {int_m:.1f} m")
    print(f"          • Pereți exteriori: {ext_m:.1f} m")
    print(f"          • Perimetru: {per_m:.1f} m")
    
    return result