# new/runner/scale/openai_scale.py
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Tuple

from openai import OpenAI


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


def detect_scale_with_openai(image_path: Path) -> dict:
    """
    Trimite imaginea planului către GPT-4o pentru detectare scară.
    
    Args:
        image_path: Path către imaginea planului (plan.jpg)
    
    Returns:
        Dict cu meters_per_pixel și detalii despre estimare
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY lipsește din environment")
    
    client = OpenAI(api_key=api_key)
    
    print(f"  📐 Trimit {image_path.name} către GPT-4o pentru detectare scară...")
    
    # Codificare imagine în base64
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "Ești un expert în arhitectură și interpretare vizuală a planurilor de construcții. Estimează scara imaginilor în mod descriptiv și rațional."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SCALE_DETECTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }
            ]
        )
    except Exception as e:
        raise RuntimeError(f"Eroare la apelul OpenAI: {e}")
    
    reply = response.choices[0].message.content.strip()
    
    # Curăță JSON (elimină markdown code fences)
    if reply.startswith("```json"):
        reply = reply[7:].strip()
    elif reply.startswith("```"):
        reply = reply[3:].strip()
    
    if reply.endswith("```"):
        reply = reply[:-3].strip()
    
    try:
        result = json.loads(reply)
    except json.JSONDecodeError as e:
        print("⚠️  Răspuns invalid de la GPT-4o:")
        print(reply[:500])
        raise ValueError(f"Nu pot parsa JSON-ul returnat de GPT-4o: {e}")
    
    # Validare structură răspuns
    if "meters_per_pixel" not in result:
        raise ValueError("Răspunsul GPT-4o nu conține cheia 'meters_per_pixel'")
    
    print(f"  ✅ Scară detectată: {result['meters_per_pixel']:.6f} m/pixel")
    
    return result