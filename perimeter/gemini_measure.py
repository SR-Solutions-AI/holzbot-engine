# new/runner/perimeter/gemini_measure.py
from __future__ import annotations

import base64
import json
import os
import math
from pathlib import Path

from openai import OpenAI


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
                "method_notes": "Fallback estimation (GPT-4o refused image analysis)"
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
    Trimite planul la GPT-4o pentru măsurarea lungimilor pereților.
    
    Args:
        plan_image: Path către plan.jpg
        scale_data: Dict cu scale_result.json (conține meters_per_pixel)
    
    Returns:
        Dict cu structura de estimări perimetru
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY lipsește din environment")
    
    client = OpenAI(api_key=api_key)
    
    meters_per_pixel = float(scale_data.get("meters_per_pixel", 0.0))
    if meters_per_pixel <= 0:
        raise ValueError("Scara invalidă în scale_result.json")
    
    print(f"       📐 Măsurare pereți cu GPT-4o (scala: {meters_per_pixel:.6f} m/px)...")
    
    # Codificare imagine
    with open(plan_image, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in precise measurements on 2D architectural plans. You MUST respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": PERIMETER_PROMPT.format(meters_per_pixel=meters_per_pixel)
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            temperature=0,
            max_tokens=2000,
            response_format={"type": "json_object"}  # Forțează JSON
        )
    except Exception as e:
        print(f"       ⚠️  Eroare la apelul OpenAI: {e}")
        print(f"       🔄 Folosesc fallback estimation...")
        return _fallback_estimation(meters_per_pixel)
    
    reply = response.choices[0].message.content.strip()
    
    # Verificare refuz explicit
    if "unable to analyze" in reply.lower() or "cannot analyze" in reply.lower() or "i'm unable" in reply.lower():
        print(f"       ⚠️  GPT-4o a refuzat analiza imagini")
        print(f"       🔄 Folosesc fallback estimation...")
        return _fallback_estimation(meters_per_pixel)
    
    # Curăță JSON (elimină markdown dacă există)
    if reply.startswith("```"):
        lines = reply.split("\n")
        # Elimină liniile cu ```
        lines = [l for l in lines if not l.strip().startswith("```")]
        reply = "\n".join(lines).strip()
    
    try:
        result = json.loads(reply)
    except json.JSONDecodeError as e:
        print(f"       ⚠️  Răspuns invalid de la GPT-4o:")
        print(reply[:500])
        print(f"       🔄 Folosesc fallback estimation...")
        return _fallback_estimation(meters_per_pixel)
    
    # Validare structură
    if "estimations" not in result:
        print(f"       ⚠️  Răspunsul GPT-4o nu conține cheia 'estimations'")
        print(f"       🔄 Folosesc fallback estimation...")
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