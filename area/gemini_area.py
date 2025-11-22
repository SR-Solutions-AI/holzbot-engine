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

    # 4. Prompt Modificat cu Logica Strictă 30%
    prompt = f"""
Imaginea atașată este un plan arhitectural de casă.
Scopul tău este să estimezi **suprafața totală a casei în metri pătrați** (Amprenta construită desfășurată pentru acest nivel).

Te rog să efectuezi calculele în pași, apoi să aplici regula de decizie de la final.

1️⃣ **Metoda 1: Bazată pe scară (Geometrică)**:
   - Folosește valoarea scării: **{meters_per_pixel:.6f} m/pixel**.
   - Identifică conturul exterior al pereților (fără terase neacoperite/curte).
   - Calculează aria brută (Gross Floor Area).

2️⃣ **Metoda 2: Bazată pe etichete (Semantică)**:
   - Caută texte cu valori de suprafețe (ex: "15.4 m²", "Wohnfläche", "Gesamt").
   - Adună toate valorile camerelor.
   - Dacă există un text explicit "Total" sau "Gesamtfläche", folosește-l ca valoare principală pentru această metodă.

3️⃣ **REGULĂ STRICTĂ DE DECIZIE (Algoritm)**:
   Compare cele două rezultate:
   - Dacă `Metoda 2 (Labels)` este 0 sau nu s-au găsit texte -> Alege `Metoda 1`.
   - Altfel, calculează diferența procentuală dintre ele.
   
   **CONDITII:**
   A. Dacă diferența este **<= 30%** (mai mică sau egală cu 30%):
      -> `final_area_m2` TREBUIE să fie **MEDIA Aritmetică** dintre Metoda 1 și Metoda 2.
      
   B. Dacă diferența este **> 30%** (mai mare de 30%):
      -> `final_area_m2` TREBUIE să fie valoarea de la **Metoda 2 (Labels)**.

4️⃣ **Rezultat final**:
   - Returnează DOAR JSON, fără text suplimentar, cu această structură:

{{
  "scale_meters_per_pixel": {meters_per_pixel:.6f},
  "surface_estimation": {{
    "by_scale_m2": <float sau null>,
    "by_labels_m2": <float sau null>,
    "diff_percentage": <float (ex: 15.5)>,
    "final_area_m2": <float>,
    "method_used": "<string: 'average_under_30', 'labels_over_30', 'scale_fallback'>"
  }},
  "confidence": "<string: 'high', 'medium', 'low'>",
  "verification_notes": "<string: explică scurt decizia luată>"
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