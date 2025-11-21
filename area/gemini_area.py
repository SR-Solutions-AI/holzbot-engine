# new/runner/area/gemini_area.py
from __future__ import annotations

import os
import json
import google.generativeai as genai
from pathlib import Path

def estimate_house_area_with_gemini(
    image_path: Path,
    scale_json_path: Path,
    api_key: str | None = None
) -> dict:
    """
    Estimează aria casei folosind Gemini (geometric + semantic).
    Returnează dicționarul JSON complet primit de la AI.
    """
    
    # 1. Configurare API
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found (env or arg).")
        
    genai.configure(api_key=api_key)

    # 2. Citire Scară
    if not scale_json_path.exists():
        raise FileNotFoundError(f"Scale file missing: {scale_json_path}")
        
    with open(scale_json_path, "r", encoding="utf-8") as f:
        scale_data = json.load(f)

    # Încercăm diverse chei posibile pentru scară
    meters_per_pixel = scale_data.get("meters_per_pixel")
    if meters_per_pixel is None:
         # Fallback dacă scara e salvată altfel
         meters_per_pixel = scale_data.get("scale", {}).get("meters_per_pixel")
         
    if not meters_per_pixel:
        raise ValueError(f"Could not find 'meters_per_pixel' in {scale_json_path}")

    # 3. Citire Imagine
    if not image_path.exists():
        raise FileNotFoundError(f"Image file missing: {image_path}")
        
    with open(image_path, "rb") as f:
        plan_bytes = f.read()

    # 4. Prompt
    prompt = f"""
Imaginea atașată este un plan arhitectural de casă.
Scopul tău este să estimezi **suprafața totală a casei în metri pătrați** (Amprenta construită desfășurată pentru acest nivel).

Fă asta în două moduri independente:

1️⃣ **Metoda bazată pe scară (geometrică)**:
   - Folosește valoarea scării: **{meters_per_pixel:.6f} m/pixel**.
   - Estimează dimensiunile exterioare ale clădirii și calculează aria totală (inclusiv camere, pereți, fără curte).
   - Aceasta este aria brută (Gross Floor Area).

2️⃣ **Metoda bazată pe etichete și legende (semantică)**:
   - Caută texte cu valori de suprafețe: m², „Gesamtfläche”, „Wohnfläche”, „Essen/Wohnen”, etc.
   - Adună toate valorile numerice care par a fi suprafețe de camere.
   - Dacă există o valoare totală (Gesamtfläche / Total), folosește-o prioritar.

3️⃣ **Analiză comparativă și selecție inteligentă**:
   - Dacă cele două metode diferă cu peste 25%, **NU face media**.
   - În schimb, alege metoda mai plauzibilă și explică motivul în "verification_notes".
   - Dacă diferența este rezonabilă (<25%), poți face media sau alege valoarea geometrică dacă planul e clar.

4️⃣ **Rezultat final**:
   - Returnează DOAR JSON, fără text suplimentar, cu această structură:

{{
  "scale_meters_per_pixel": {meters_per_pixel:.6f},
  "surface_estimation": {{
    "by_scale_m2": <float sau null>,
    "by_labels_m2": <float sau null>,
    "final_area_m2": <float>,
    "method_used": "<string: 'scale', 'labels', 'average' sau 'hybrid'>"
  }},
  "confidence": "<string: 'high', 'medium', 'low'>",
  "verification_notes": "<string>"
}}
"""

    # 5. Apelare Model
    # Încercăm Pro, apoi Flash
    model_name = "gemini-2.0-flash" # Sau 1.5-pro, în funcție de acces
    try:
        model = genai.GenerativeModel(model_name)
    except:
        model = genai.GenerativeModel("gemini-1.5-flash")

    response = model.generate_content(
        [
            {"role": "user", "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": plan_bytes}},
            ]}
        ],
        generation_config={"temperature": 0.0, "response_mime_type": "application/json"}
    )

    # 6. Procesare Răspuns
    reply = response.text.strip()
    
    # Curățare markdown ```json ... ```
    if reply.startswith("```"):
        lines = reply.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        reply = "\n".join(lines)

    try:
        return json.loads(reply)
    except json.JSONDecodeError:
        # Fallback simplu în caz de eroare de parse
        print(f"⚠️ Gemini Area JSON Decode Error. Raw: {reply}")
        raise