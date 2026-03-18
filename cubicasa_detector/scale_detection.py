# file: engine/cubicasa_detector/scale_detection.py
"""
Module pentru detectarea scalei din planuri.
Conține funcții pentru apelul Gemini API și detectarea scalei din label-uri de camere.
"""

from __future__ import annotations

import json
import math
import base64
import os
import re
import requests
import cv2
import numpy as np
from pathlib import Path
from typing import Callable

from .config import GEMINI_MODEL


GEMINI_PROMPT_CROP = """
Analizează această imagine de plan de casă și identifică:
1. Numele camerei/camerelor (ex: "Living Room", "Bedroom", "Kitchen", "Terrasse", etc.)
2. Suprafața camerei/camerelor în metri pătrați (m²) - DOAR dacă este explicit indicată PENTRU O CAMERĂ CONCRETĂ

REGULI STRICTE PENTRU EXTRAGEREA SUPRAFEȚEI:

⚠️ CRITICAL: Extrage DOAR valori care sunt EXPLICIT etichetate ca metri pătrați (m²) ȘI sunt asociate cu O CAMERĂ CONCRETĂ (ex: "Terrasse 62,82 m²", "Küche 15 m²").

❌ IGNORĂ COMPLET – NU extrage și NU aduna:
   - Texte informativ de tip TOTAL / SUPRAFAȚĂ UTILIZABILĂ pentru ÎNTREGUL NIVEL, nu pentru o cameră:
     • "NNF = ca. X m²", "NNF=ca. 200 m²", "NNF ca. X m²"
     • "Nutzfläche ca. X m²", "Gesamtnutzfläche", "Hausfläche ca. X m²"
     • "net floor area", "suprafață utilizabilă", "total usable area", "ca. X m²" (fără nume de cameră lângă)
   - Acestea sunt TOTALURI pentru etaj/nivel, NU suprafețe ale unei camere. Nu le include în room_name sau area_m2.
   - Numere din scale (ex: "1:100", "1:50"), dimensiuni (ex: "3.5 x 4.2"), coordonate sau adrese.

✅ Extrage DOAR: nume de cameră + suprafața în m² când sunt ÎMPREUNĂ (ex: "Terrasse Feinsteinzeug 62,82 m²" → room_name: "Terrasse", area_m2: 62.82).

IMPORTANT:
- Zonele negre din imagine NU fac parte din cameră.
- Dacă singura suprafață în m² din imagine este un total (NNF / Nutzfläche / ca. X m² fără nume de cameră), returnează null pentru area_m2.
- Dacă sunt mai multe camere cu suprafețe în crop, adună DOAR suprafețele care aparțin unor camere concrete (nu totalul de nivel).

Returnează JSON:
{
  "room_name": "numele camerei sau 'Multiple rooms' dacă sunt mai multe",
  "area_m2": 15.5
}
Dacă nu găsești o suprafață asociată unei cameri concrete (doar total NNF/Nutzfläche etc.), returnează null pentru area_m2.
"""

# Prompt pentru batch de camere (N imagini → un singur răspuns JSON cu array)
# room_number = numărul desenat pe imagine (room_no:N) – pentru asociere cu poligoanele din 09_interior (OCR)
# room_name și room_type = EXACT textul de pe plan, aceeași limbă (fără traducere). Planurile sunt adesea în germană.
GEMINI_PROMPT_ROOMS_BATCH = """
You receive {n} images of floor plan room crops (in order: Image 1, Image 2, ... Image {n}).
Each image has in the TOP-LEFT corner two labels: "area_px:NNNNN" and "room_no:N" (N = room number 0, 1, 2, ...). READ both numbers.

CRITICAL – room_number and EXACT text from plan:
- room_number = the integer you see in "room_no:N" on that image. Return it exactly (required for matching).
- room_name and room_type = copy the EXACT word(s) written ON THE PLAN for that room. Same language as the plan. DO NOT translate. If the plan is in German you get German (Küche, Bad, Flur, Wohnzimmer, Schlafzimmer, Abstellraum, etc.). If the plan is in English you get English. Copy character for character. Use "Raum" only if no readable label or multiple labels in one crop.

For EACH image in order, extract:
1. room_number – integer from "room_no:N" in the corner. Required.
2. area_px – integer from "area_px:NNNNN". Required.
3. room_name – EXACT text from the plan (same language, no translation). "Mehrere Räume" only if several room names in crop and plan is German; otherwise "Multiple rooms" etc. as on plan.
4. area_m2 – number only if m² is explicitly written for that room; ignore NNF/Nutzfläche totals.
5. room_type – EXACT room label as written on the plan. Do not translate.

RULES:
- room_number and area_px = copy from corner labels.
- room_name and room_type = copy EXACTLY from plan, same language.
- Black areas are not part of the room.

Return ONLY this JSON (no markdown):
{{"rooms": [{{"room_number": 0, "area_px": 33060, "room_name": "<exact from plan>", "area_m2": 12.5, "room_type": "<exact from plan>"}}, ...]}}
with exactly {n} objects. Every object MUST have room_number (int), area_px (int), room_name (string), and room_type (string).
"""


def is_informational_total_result(result: dict) -> bool:
    """
    Returnează True dacă rezultatul Gemini pare a fi un text informativ de tip total (NNF, Nutzfläche etc.),
    nu o cameră concretă. Astfel de rezultate nu trebuie folosite la calculul scalei / sumă camere.
    """
    if not result or not isinstance(result, dict):
        return True
    room_name = (result.get("room_name") or "").strip().upper()
    area_m2 = result.get("area_m2")
    total_keywords = (
        "NNF", "NUTZFLÄCHE", "NUTZFLAECHE", "GESAMT", "HAUSFLÄCHE", "HAUSFLAECHE",
        "NET FLOOR", "TOTAL AREA", "SUPRAFAȚĂ UTILIZABILĂ", "SUPRAFATA UTILIZABILA",
        "ERDGESCHOSS 1:100", "1:100"
    )
    if any(kw in room_name for kw in total_keywords):
        return True
    if area_m2 is not None:
        try:
            a = float(area_m2)
            # Suprafață foarte rotundă (100, 150, 200) cu nume lipsă sau doar număr = probabil total etaj
            if a >= 80 and a == round(a) and (not room_name or room_name.isdigit() or room_name in ("TOTAL", "SUM", "AREA")):
                return True
        except (TypeError, ValueError):
            pass
    return False


GEMINI_PROMPT_TOTAL_SUM = """
Analizează această imagine de plan de casă și identifică suma totală a suprafețelor tuturor camerelor (în metri pătrați, m²).

Returnează JSON cu formatul:
{
  "total_sum_m2": 120.5
}
Dacă nu găsești suma totală, returnează null.
"""

# Prompt pentru detectarea zonelor: garaj, terasă, balcon, intrare acoperită (coordonate în procente)
GEMINI_PROMPT_ZONE_LABELS = """
You are analyzing an architectural floor plan image. Your task is to READ THE TEXT WRITTEN ON THE PLAN (room labels, zone names, annotations) and locate the following zone types. For each one you find, return the CENTER of that written label in percentage coordinates: x_center_pct = (center_x / image_width) * 100, y_center_pct = (center_y / image_height) * 100, both in range 0–100.

Search the plan for these zone labels. Look for the exact words or common abbreviations as written on the drawing (German, English, Romanian, etc.):

1. **garage** – Look for: Garage, Garaj, Gar., Carport, Parking, Stellplatz, Stellpl., Garáž, Garaž, Parkplatz, etc. Return the center of that text.

2. **terasa** – Look for: Terrasse, Terr., Terrace, Terasa, Terasă, Patio, Garten (when it is a terrace/patio area), etc. Return the center of that text.

3. **balcon** – Look for: Balkon, Balk., Balcon, Balcony, Loggia, etc. Return the center of that text.

4. **wintergarden** – Look for: Wintergarten, Wintergarden, Winter garden, Glasanbau, etc. Return the center of that text.

5. **intrare_acoperita** – Look ONLY for: "Eingang überdacht", "überdachter Eingang", "Eing. überdacht", or equivalent "covered entrance". Do NOT match: Foyer, Flur, Diele, Hall, Eingang alone, Entrance, Corridor. We need specifically the covered entrance zone.

Rules:
- Scan the whole image for these words/abbreviations. Return the geometric center of the text bounding box for each label found.
- If a zone is not present on the plan, omit it from the array.
- Output ONLY valid JSON. Use exactly these label strings: "garage", "terasa", "balcon", "wintergarden", "intrare_acoperita".

Return ONLY a JSON object (no markdown, no explanation):
{
  "detections": [
    { "label": "garage", "x_center_pct": 25.5, "y_center_pct": 62.0 },
    { "label": "terasa", "x_center_pct": 70.0, "y_center_pct": 15.0 }
  ]
}
"""


# Prompt pentru ChatGPT: primește raw return de la Gemini, returnează JSON valid (structură detections).
# Acoladele literale sunt escapate {{ }} ca să nu fie interpretate la .format(raw_text=...).
PROMPT_REPAIR_ZONE_LABELS = """You receive the raw text returned by another AI (Gemini). It was supposed to be a JSON object with a "detections" array. The response may be truncated or invalid JSON.

Your task: output ONLY a valid JSON object with this exact structure (no markdown, no explanation, no code fence):

{{"detections": [
  {{"label": "garage", "x_center_pct": 25.5, "y_center_pct": 62.0}},
  {{"label": "terasa", "x_center_pct": 70.0, "y_center_pct": 15.0}}
]}}

Rules:
- "detections" must be an array. Each item: "label" (string), "x_center_pct" (number 0–100), "y_center_pct" (number 0–100).
- Allowed labels only: "garage", "terasa", "balcon", "wintergarden", "intrare_acoperita".
- From the raw text below, extract every zone you can identify (even from truncated content) and put them in the array.
- If nothing usable: return {{"detections": []}}.
- Output ONLY the JSON object. Start with {{ and end with }}.

Raw response from the other AI:
---
{raw_text}
---"""


def _extract_json_object_from_text(text: str) -> str | None:
    """Extrage primul obiect JSON complet din text (primul { până la } potrivit)."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_zone_labels_object(json_str: str) -> dict | None:
    """
    Parsează un string JSON care ar trebui să fie un obiect cu cheia detections (array).
    Returnează un dict cu cheia "detections" (listă) sau None. Nu folosește niciodată data["detections"].
    """
    if not json_str or not json_str.strip():
        return None
    json_str = json_str.strip()
    # Fix trailing comma în array/object (ca la segmentare)
    json_str = re.sub(r",\s*]", "]", json_str)
    json_str = re.sub(r",\s*}", "}", json_str)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    detections = None
    for k, v in data.items():
        if k and "detections" in (k.strip().lower()) and isinstance(v, list):
            detections = v
            break
    if detections is None:
        detections = []
    out = {"detections": []}
    for item in detections:
        if not isinstance(item, dict):
            continue
        label = (item.get("label") or "").strip().lower()
        if label not in ("garage", "terasa", "balcon", "wintergarden", "intrare_acoperita"):
            continue
        try:
            x_pct = float(item.get("x_center_pct"))
            y_pct = float(item.get("y_center_pct"))
        except (TypeError, ValueError):
            continue
        out["detections"].append({"label": label, "x_center_pct": x_pct, "y_center_pct": y_pct})
    return out


def _try_parse_truncated_zone_labels(raw_text: str) -> dict | None:
    """Încearcă reparare locală: răspuns trunchiat Gemini – închide ] și } lipsă, apoi parsează (ca la segmentare)."""
    raw = (raw_text or "").strip()
    if not raw or "detections" not in raw.lower():
        return None
    start = raw.find("{")
    if start < 0:
        return None
    s = raw[start:]
    open_br, open_sq = 0, 0
    for c in s:
        if c == "{":
            open_br += 1
        elif c == "}":
            open_br -= 1
        elif c == "[":
            open_sq += 1
        elif c == "]":
            open_sq -= 1
    s = s + "]" * max(0, open_sq) + "}" * max(0, open_br)
    return _parse_zone_labels_object(s)


def _repair_zone_labels_json_with_chatgpt(raw_text: str) -> dict | None:
    """
    Același pattern ca la segmentare (Gemini Crop): trimitem raw la ChatGPT, extragem obiect JSON din răspuns,
    parsează fără acces direct la chei. Nu propagăm nicio excepție.
    """
    try:
        # 1) Întâi încercăm reparare locală (trunchiere), ca la segmentare când _extract_json_array e gol
        parsed = _try_parse_truncated_zone_labels(raw_text or "")
        if parsed is not None:
            print("   [Gemini zone labels] JSON trunchiat reparat local (paranteze închise).")
            return parsed

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("   [Gemini zone labels] OPENAI_API_KEY lipsește – nu pot repara cu ChatGPT.")
            return None

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        except Exception:
            return None

        prompt = PROMPT_REPAIR_ZONE_LABELS.format(raw_text=(raw_text or "")[:8000])
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You output only valid JSON. No markdown, no explanation, no extra text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
        )

        content = ""
        if response and getattr(response, "choices", None) and len(response.choices) > 0:
            msg = getattr(response.choices[0], "message", None)
            if msg is not None:
                content = (getattr(msg, "content", None) or "").strip()

        if not content:
            return None

        # Elimină markdown code block (ca la segmentare: regex pentru ```json ... ```)
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if m:
            content = m.group(1).strip()
        if "```" in content:
            for part in content.split("```"):
                part = part.strip()
                if part.startswith("{"):
                    content = part
                    break

        json_str = _extract_json_object_from_text(content)
        if not json_str:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
        if not json_str:
            return None

        parsed = _parse_zone_labels_object(json_str)
        if parsed is not None:
            n = len(parsed.get("detections") or [])
            if n > 0:
                print(f"   [Gemini zone labels] ChatGPT a construit JSON valid din răspunsul Gemini ({n} zone).")
            else:
                print("   [Gemini zone labels] ChatGPT a returnat JSON valid dar fără zone (detections gol) → se va folosi OCR.")
            return parsed
        return None
    except Exception as e:
        print(f"   [Gemini zone labels] Eroare la reparare: {e}")
        return None


def call_gemini_zone_labels(
    image_path: str | Path,
    api_key: str,
    image_w: int,
    image_h: int,
) -> dict[str, list[tuple[int, int]]]:
    """
    Apelează Gemini pentru detectarea etichetelor de zone (garage, terasa, balcon, intrare_acoperita).
    Returnează un dict: label -> listă de (cx_px, cy_px) centre în pixeli (pot exista mai multe terase, garaje etc.).
    Salvează JSON-ul primit (detections cu procente) în același director cu imaginea.
    """
    out: dict[str, list[tuple[int, int]]] = {}
    try:
        result = call_gemini(
            str(image_path),
            GEMINI_PROMPT_ZONE_LABELS,
            api_key,
            max_retries=1,
            repair_callback=_repair_zone_labels_json_with_chatgpt,
            max_output_tokens=2048,
        )
        if not result or "detections" not in result:
            return out
        # Salvează JSON-ul (detections cu procente) pentru inspecție
        try:
            save_dir = Path(image_path).parent
            json_path = save_dir / "gemini_zone_labels.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"   [Gemini zone labels] Salvat: {json_path.name}")
        except Exception as e:
            print(f"   [Gemini zone labels] Nu s-a putut salva JSON: {e}")
        for d in result.get("detections") or []:
            label = (d.get("label") or "").strip().lower()
            if label not in ("garage", "terasa", "balcon", "wintergarden", "intrare_acoperita"):
                continue
            x_pct = d.get("x_center_pct")
            y_pct = d.get("y_center_pct")
            if x_pct is None or y_pct is None:
                continue
            try:
                x_pct = float(x_pct)
                y_pct = float(y_pct)
            except (TypeError, ValueError):
                continue
            cx = int(round(x_pct / 100.0 * max(1, image_w)))
            cy = int(round(y_pct / 100.0 * max(1, image_h)))
            cx = max(0, min(image_w - 1, cx))
            cy = max(0, min(image_h - 1, cy))
            if label not in out:
                out[label] = []
            out[label].append((cx, cy))
    except Exception as e:
        print(f"   [Gemini zone labels] Eroare: {e}")
    return out


def _build_blacklist_prompt(terms: list[str]) -> str:
    """Construiește promptul Gemini pentru detectarea cuvintelor blacklist pe plan."""
    terms_str = ", ".join(repr(t) for t in (terms[:25] if len(terms) > 25 else terms))
    return f"""You are analyzing an architectural floor plan image. Find any of these BLACKLIST words/labels written on the plan (e.g. pool, piscina, Schwimmbad):

Terms to search for (any language): {terms_str}

For each occurrence found, return the CENTER of that text in percentage coordinates:
- x_center_pct = (center_x / image_width) * 100
- y_center_pct = (center_y / image_height) * 100
(both 0–100). Use "label" as the word found (e.g. "pool", "Pool", "piscina").

Return ONLY a valid JSON object (no markdown):
{{"detections": [{{ "label": "pool", "x_center_pct": 50.0, "y_center_pct": 30.0 }}]}}
If none of these words appear, return {{"detections": []}}."""


def call_gemini_blacklist(
    image_path: str | Path,
    api_key: str,
    image_w: int,
    image_h: int,
    blacklist_terms: list[str],
) -> list[tuple[int, int, str]]:
    """
    Apelează Gemini pentru detectarea cuvintelor blacklist pe plan.
    Returnează lista de (cx_px, cy_px, label) pentru fiecare detecție, sau listă goală.
    """
    if not blacklist_terms:
        return []
    out: list[tuple[int, int, str]] = []
    try:
        prompt = _build_blacklist_prompt(blacklist_terms)
        result = call_gemini(
            str(image_path),
            prompt,
            api_key,
            max_retries=1,
            repair_callback=None,
            max_output_tokens=2048,
        )
        if not result or "detections" not in result:
            return out
        try:
            save_dir = Path(image_path).parent
            json_path = save_dir / "gemini_blacklist.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"   [Gemini blacklist] Salvat: {json_path.name}")
        except Exception as e:
            print(f"   [Gemini blacklist] Nu s-a putut salva JSON: {e}")
        for d in result.get("detections") or []:
            label = (d.get("label") or "").strip()
            if not label:
                continue
            x_pct = d.get("x_center_pct")
            y_pct = d.get("y_center_pct")
            if x_pct is None or y_pct is None:
                continue
            try:
                x_pct = float(x_pct)
                y_pct = float(y_pct)
            except (TypeError, ValueError):
                continue
            cx = int(round(x_pct / 100.0 * max(1, image_w)))
            cy = int(round(y_pct / 100.0 * max(1, image_h)))
            cx = max(0, min(image_w - 1, cx))
            cy = max(0, min(image_h - 1, cy))
            out.append((cx, cy, label))
    except Exception as e:
        print(f"   [Gemini blacklist] Eroare: {e}")
    return out


def _repair_scale_json_with_gpt(raw_text: str, prompt: str):
    """Fallback: GPT construiește JSON valid din răspunsul raw Gemini."""
    schema_hint = (
        "Object with room_name (str), area_m2 (float or null). "
        "Or object with total_sum_m2 (float or null)."
    )
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from common.json_repair import repair_json_with_gpt
        return repair_json_with_gpt(raw_text, schema_hint)
    except Exception as e:
        print(f"   [scale] GPT repair failed: {e}")
        return None


def call_gemini(image_path, prompt, api_key, max_retries=2, repair_callback: Callable[[str], dict | None] | None = None, max_output_tokens: int = 1000):
    """API REST Direct (v1beta) pentru Gemini. Retry la non-JSON. repair_callback: dacă e dat, la JSON invalid se folosește pentru reparare (ex. zone labels). max_output_tokens: limită tokeni (default 1000; pentru zone labels se recomandă 2048)."""
    try:
        for attempt in range(max_retries + 1):
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            ext = Path(image_path).suffix.lower()
            mime_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}
            mime_type = mime_map.get(ext, 'image/jpeg')

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"

            gen_config = {"temperature": 0.0, "maxOutputTokens": max_output_tokens, "responseMimeType": "application/json"}
            payload = {
                "contents": [{
                    "parts": [
                        {"inline_data": {"mime_type": mime_type, "data": image_data}},
                        {"text": prompt + "\n\nIMPORTANT: Răspunde DOAR cu un obiect JSON valid, fără text înainte sau după."}
                    ]
                }],
                "generationConfig": gen_config
            }

            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=payload, headers=headers, timeout=30)

            if response.status_code != 200:
                print(f"⚠️  Gemini HTTP {response.status_code}")
                return None

            result = response.json()
            if 'candidates' not in result or not result['candidates']:
                return None

            text = result['candidates'][0]['content']['parts'][0].get('text', '').strip()
            if not text:
                return None

            if text.startswith("```"):
                lines = text.splitlines()
                if lines and lines[0].lstrip().startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines).strip()

            start = text.find("{")
            end = text.rfind("}") + 1
            candidate = text
            if start != -1 and end > start:
                candidate = text[start:end]

            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                preview = " ".join(text.split())[:160]
                print(f"⚠️  Gemini returned non-JSON: {preview!r}, încerc GPT repair...")
                repaired = None
                try:
                    if repair_callback is not None:
                        repaired = repair_callback(text)
                    else:
                        repaired = _repair_scale_json_with_gpt(text, prompt)
                except Exception as repair_err:
                    print(f"   [repair] Callback eșuat: {repair_err}")
                if repaired is not None:
                    return repaired
                if attempt < max_retries:
                    continue
                return None
        return None
    except Exception as e:
        print(f"⚠️  Gemini error: {e}")
        return None


def _repair_json_rooms_response(raw: str) -> str:
    """
    Încearcă reparații locale pe textul JSON (fără apel extern).
    - Elimină virgula trailing înainte de ] sau }
    - Închide bracket-uri lipsă la sfârșit (răspuns trunchiat)
    """
    if not raw or not raw.strip():
        return raw
    # Trailing comma înainte de ] sau }
    s = re.sub(r",\s*([\]}])", r"\1", raw)
    # Stack de închideri: la [ push ']', la { push '}'
    stack = []
    in_string = False
    escape = False
    quote = None
    i = 0
    while i < len(s):
        c = s[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == quote:
                in_string = False
            i += 1
            continue
        if c in ('"', "'"):
            in_string = True
            quote = c
            i += 1
            continue
        if c == "{":
            stack.append("}")
        elif c == "[":
            stack.append("]")
        elif c == "}" or c == "]":
            if stack and stack[-1] == c:
                stack.pop()
            # altfel e închidere neașteptată, nu push
        i += 1
    # Adaugă închiderile lipsă în ordine corectă
    s += "".join(reversed(stack))
    return s


def _parse_rooms_batch_text(text: str, n: int):
    """
    Extrage și curăță textul JSON din răspuns, parsează și validează.
    Returnează (list[dict] pentru out, None) la succes, sau (None, err) la eșec.
    """
    if not text or not text.strip():
        return None, "empty text"
    # Markdown code block
    if "```" in text:
        lines = text.splitlines()
        if lines and "```" in lines[0]:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return None, "no JSON object"
    text = text[start:end]
    # Încercare parsare directă, apoi cu repair
    for raw in (text, _repair_json_rooms_response(text)):
        try:
            data = json.loads(raw)
            break
        except json.JSONDecodeError:
            data = None
    if data is None:
        return None, "JSON decode failed"
    rooms = data.get("rooms")
    if not isinstance(rooms, list):
        return None, "missing or invalid 'rooms' array"
    if len(rooms) != n:
        return None, f"rooms length {len(rooms)} != {n}"
    out = []
    for pos, r in enumerate(rooms):
        if not isinstance(r, dict):
            out.append({"room_number": pos, "room_name": "", "area_m2": None, "room_type": "Raum", "area_px": None})
        else:
            apx = r.get("area_px")
            if apx is not None:
                try:
                    apx = int(apx)
                except (TypeError, ValueError):
                    apx = None
            rnum = r.get("room_number")
            if rnum is not None:
                try:
                    rnum = int(rnum)
                except (TypeError, ValueError):
                    rnum = pos
            else:
                rnum = pos
            out.append({
                "room_number": rnum,
                "room_name": r.get("room_name") or "",
                "area_m2": r.get("area_m2"),
                "room_type": (r.get("room_type") or "Raum").strip() or "Raum",
                "area_px": apx,
            })
    return out, None


def call_gemini_rooms_batch(image_paths: list, api_key: str, max_retries: int = 2):
    """
    Trimite toate crop-urile de camere într-un singur apel Gemini.
    Returnează list[dict] în aceeași ordine (fiecare dict: room_name, area_m2, room_type, area_px) sau None la eșec.
    """
    if not image_paths or not api_key:
        return None
    image_paths = [Path(p) for p in image_paths]
    n = len(image_paths)
    parts = []
    for p in image_paths:
        with open(p, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        ext = p.suffix.lower()
        mime_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}
        mime_type = mime_map.get(ext, 'image/png')
        parts.append({"inline_data": {"mime_type": mime_type, "data": data}})
    prompt = GEMINI_PROMPT_ROOMS_BATCH.format(n=n)
    parts.append({"text": prompt + "\n\nIMPORTANT: Răspunde DOAR cu un obiect JSON valid (rooms array cu exact " + str(n) + " elemente)."})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    gen_config = {"temperature": 0.0, "maxOutputTokens": 2048, "responseMimeType": "application/json"}
    payload = {"contents": [{"parts": parts}], "generationConfig": gen_config}
    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=90)
            if response.status_code != 200:
                return None
            result = response.json()
            if 'candidates' not in result or not result['candidates']:
                return None
            text = result['candidates'][0]['content']['parts'][0].get('text', '').strip()
            out, err = _parse_rooms_batch_text(text, n)
            if out is not None:
                return out
            # Reparare cu GPT dacă e disponibil
            if attempt == max_retries and os.getenv("OPENAI_API_KEY"):
                try:
                    import sys
                    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                    from common.json_repair import repair_json_with_gpt
                    schema = (
                        'Object with key "rooms" (array of exactly ' + str(n) + ' objects). '
                        'Each object: room_number (int), area_px (int), room_name (str), area_m2 (number or null), room_type (str).'
                    )
                    repaired = repair_json_with_gpt(text, schema)
                    if repaired and isinstance(repaired.get("rooms"), list) and len(repaired["rooms"]) == n:
                        out, _ = _parse_rooms_batch_text(json.dumps(repaired), n)
                        if out is not None:
                            return out
                except Exception as gpt_err:
                    print(f"⚠️  Gemini rooms batch GPT repair failed: {gpt_err}")
            if attempt < max_retries:
                continue
            print(f"⚠️  Gemini rooms batch error: {err}")
            return None
        except (KeyError, TypeError) as e:
            if attempt < max_retries:
                continue
            print(f"⚠️  Gemini rooms batch error: {e}")
            return None
        except Exception as e:
            print(f"⚠️  Gemini rooms batch error: {e}")
            return None
    return None

# Batch clasificare uși/ferestre (N imagini → un JSON cu array de tipuri)
GEMINI_PROMPT_DOORS_BATCH = """You receive {n} images of floor plan openings. For each image in order (Image 1, 2, ... {n}), classify as exactly ONE type. You MUST use both "door" and "window" where appropriate – do NOT return "door" for all.

CRITICAL – when to use "window":
- "window" = glass opening: parallel lines or grid inside, usually narrower than doors, often horizontal or vertical strip. Use "window" for any opening that clearly shows glass/fenestration. Many images will be windows.
- "door" = passage for people: single or double leaf, may show swing arc, typically wider than windows.
- "garage_door" = very wide (2-3x a normal door), multiple panels, ground level.
- "stairs" = visible step pattern (parallel lines), connects floors.

Rule: If the opening looks like a window (narrow, lines/grid, glass) → "window". If it looks like a door (passage, swing) → "door". Vary your answers.

Return ONLY this JSON (no markdown): {{"types": ["door", "window", "garage_door", "stairs", ...]}}
with exactly {n} strings, one per image in order.
"""


def _parse_doors_batch_text(text: str, n: int):
    """
    Extrage și curăță JSON din răspuns, parsează și validează.
    Returnează (list[str] pentru types, None) la succes, sau (None, err) la eșec.
    """
    if not text or not text.strip():
        return None, "empty text"
    if "```" in text:
        lines = text.splitlines()
        if lines and "```" in lines[0]:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return None, "no JSON object"
    text = text[start:end]
    valid_types = {'door', 'window', 'garage_door', 'stairs'}
    for raw in (text, _repair_json_rooms_response(text)):
        try:
            data = json.loads(raw)
            break
        except json.JSONDecodeError:
            data = None
    if data is None:
        return None, "JSON decode failed"
    types_raw = data.get("types")
    if not isinstance(types_raw, list):
        return None, "missing or invalid 'types' array"
    if len(types_raw) != n:
        return None, f"types length {len(types_raw)} != {n}"
    out = []
    for t in types_raw:
        s = (t if isinstance(t, str) else str(t)).lower().strip()
        out.append(s if s in valid_types else 'door')
    return out, None


def call_gemini_doors_batch(image_paths: list, api_key: str, max_retries: int = 2) -> list[str] | None:
    """
    Trimite toate crop-urile de deschideri (doors) într-un singur apel Gemini.
    Returnează list[str] de tipuri ('door', 'window', 'garage_door', 'stairs') în aceeași ordine, sau None.
    """
    if not image_paths or not api_key:
        return None
    image_paths = [Path(p) for p in image_paths]
    n = len(image_paths)
    parts = []
    for p in image_paths:
        with open(p, 'rb') as f:
            data = base64.b64encode(f.read()).decode('utf-8')
        ext = p.suffix.lower()
        mime_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}
        mime_type = mime_map.get(ext, 'image/png')
        parts.append({"inline_data": {"mime_type": mime_type, "data": data}})
    prompt = GEMINI_PROMPT_DOORS_BATCH.format(n=n)
    parts.append({"text": prompt + "\n\nIMPORTANT: Reply only with JSON object with key 'types' and an array of exactly " + str(n) + " strings."})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    gen_config = {"temperature": 0.0, "maxOutputTokens": 1024, "responseMimeType": "application/json"}
    payload = {"contents": [{"parts": parts}], "generationConfig": gen_config}
    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            if response.status_code != 200:
                return None
            result = response.json()
            if 'candidates' not in result or not result['candidates']:
                return None
            text = result['candidates'][0]['content']['parts'][0].get('text', '').strip()
            out, err = _parse_doors_batch_text(text, n)
            if out is not None:
                return out
            if attempt == max_retries and os.getenv("OPENAI_API_KEY"):
                try:
                    import sys
                    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                    from common.json_repair import repair_json_with_gpt
                    schema = (
                        'Object with key "types" (array of exactly ' + str(n) + ' strings). '
                        'Each string is one of: "door", "window", "garage_door", "stairs".'
                    )
                    repaired = repair_json_with_gpt(text, schema)
                    if repaired and isinstance(repaired.get("types"), list) and len(repaired["types"]) == n:
                        out, _ = _parse_doors_batch_text(json.dumps(repaired), n)
                        if out is not None:
                            return out
                except Exception as gpt_err:
                    print(f"⚠️  Gemini doors batch GPT repair failed: {gpt_err}")
            if attempt < max_retries:
                continue
            print(f"⚠️  Gemini doors batch error: {err}")
            return None
        except (KeyError, TypeError) as e:
            if attempt < max_retries:
                continue
            print(f"⚠️  Gemini doors batch error: {e}")
            return None
        except Exception as e:
            print(f"⚠️  Gemini doors batch error: {e}")
            return None
    return None


GEMINI_PROMPT_WALL_ALIGNMENT = """
You are given TWO images of floor plan wall masks (binary or grayscale: white/light = walls, black = background).

- IMAGE 1 (first image): REFERENCE mask = "original" walls. This is the ground truth we want to align to.
- IMAGE 2 (second image): TEMPLATE mask = "detected" walls (e.g. from an API). This image must be scaled and positioned to overlay exactly on IMAGE 1.

Your task: determine the exact scale and position so that when IMAGE 2 is scaled and placed over IMAGE 1, the walls match as much as possible.

Assume:
- Both images show the SAME floor plan; they may differ in resolution, scale, and position.
- No rotation: only scaling and translation (horizontal/vertical shift) are needed.
- Coordinates: origin (0,0) is top-left; x increases right, y increases down.

Return a JSON object with these exact keys:

{
  "scale": <float between 0.2 and 2.0>,
  "offset_x_pct": <float between -0.5 and 0.5>,
  "offset_y_pct": <float between -0.5 and 0.5>,
  "direction": "api_to_orig",
  "confidence": <float between 0 and 1>
}

Meaning:
- scale: factor to apply to IMAGE 2 width and height so its size matches IMAGE 1. Example: 1.0 = same size; 0.6 = scale IMAGE 2 to 60% so it fits inside IMAGE 1.
- offset_x_pct: horizontal shift of the CENTER of (scaled) IMAGE 2 relative to the CENTER of IMAGE 1, as a fraction of IMAGE 1 width. Positive = move IMAGE 2 right; negative = left.
- offset_y_pct: vertical shift of the CENTER of (scaled) IMAGE 2 relative to the CENTER of IMAGE 1, as a fraction of IMAGE 1 height. Positive = move IMAGE 2 down; negative = up.
- direction: use "api_to_orig" if IMAGE 2 (template) is scaled and placed ONTO IMAGE 1 (reference). Use "orig_to_api" if instead IMAGE 1 is scaled and placed onto IMAGE 2.
- confidence: how sure you are (0 = guess, 1 = very confident). Use 0.5–0.7 for typical alignment, 0.8+ only if the match is obvious.

Reply with ONLY the JSON object, no other text.
"""


def call_gemini_wall_alignment(
    reference_image_path: str | Path,
    template_image_path: str | Path,
    api_key: str,
    max_retries: int = 1,
    timeout: int = 45,
):
    """
    Trimite cele două măști (referință + template) la Gemini și primește scale + offset pentru aliniere.
    Rulează în paralel cu căutarea brută; rezultatul poate fi folosit ca hint pentru refinare.

    Returns:
        dict cu scale, offset_x_pct, offset_y_pct, direction, confidence; sau None la eroare.
    """
    try:
        ref_path = Path(reference_image_path)
        tpl_path = Path(template_image_path)
        if not ref_path.is_file() or not tpl_path.is_file():
            return None
        with open(ref_path, "rb") as f:
            ref_b64 = base64.b64encode(f.read()).decode("utf-8")
        with open(tpl_path, "rb") as f:
            tpl_b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = ref_path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
        mime = mime_map.get(ext, "image/png")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
        gen_config = {"temperature": 0.1, "maxOutputTokens": 512, "responseMimeType": "application/json"}
        prompt_text = (
            "IMAGE 1 (first image) = REFERENCE = original walls mask. "
            "IMAGE 2 (second image) = TEMPLATE = detected walls to align onto IMAGE 1.\n\n"
            + GEMINI_PROMPT_WALL_ALIGNMENT
            + "\n\nReply with ONLY the JSON object, no markdown."
        )
        payload = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": mime, "data": ref_b64}},
                    {"inline_data": {"mime_type": mime, "data": tpl_b64}},
                    {"text": prompt_text}
                ]
            }],
            "generationConfig": gen_config
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return None
        result = response.json()
        if not result.get("candidates"):
            return None
        text = (result["candidates"][0].get("content", {}).get("parts") or [{}])[0].get("text", "").strip()
        if not text:
            return None
        start, end = text.find("{"), text.rfind("}") + 1
        if start == -1 or end <= start:
            return None
        out = json.loads(text[start:end])
        if not isinstance(out, dict):
            return None
        # Normalize
        scale = out.get("scale")
        if scale is not None:
            scale = float(scale)
            scale = max(0.2, min(2.0, scale))
        out["scale"] = scale
        for k in ("offset_x_pct", "offset_y_pct", "confidence"):
            if out.get(k) is not None:
                out[k] = float(out[k])
        out["offset_x_pct"] = max(-0.5, min(0.5, out.get("offset_x_pct", 0) or 0))
        out["offset_y_pct"] = max(-0.5, min(0.5, out.get("offset_y_pct", 0) or 0))
        out["confidence"] = max(0.0, min(1.0, out.get("confidence", 0) or 0))
        out.setdefault("direction", "api_to_orig")
        return out
    except Exception as e:
        print(f"      ⚠️ Gemini wall alignment: {e}")
        return None


def flood_fill_room(indoor_mask, seed_pt, search_radius=10):
    """Flood fill pentru segmentare camere."""
    h, w = indoor_mask.shape[:2]
    x0, y0 = seed_pt
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))

    if indoor_mask[y0, x0] == 0:
        found = False
        for dy in range(-search_radius, search_radius + 1):
            yy = y0 + dy
            if 0 <= yy < h:
                for dx in range(-search_radius, search_radius + 1):
                    xx = x0 + dx
                    if 0 <= xx < w:
                        if indoor_mask[yy, xx] > 0:
                            x0, y0 = xx, yy
                            found = True
                            break
                if found:
                    break
        if not found:
            return np.zeros_like(indoor_mask), 0, seed_pt

    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    temp = indoor_mask.copy()
    try:
        _, _, mask, _ = cv2.floodFill(
            temp, mask, (x0, y0), 255,
            loDiff=(0,), upDiff=(0,),
            flags=cv2.FLOODFILL_MASK_ONLY | 4
        )
    except:
        return np.zeros_like(indoor_mask), 0, (x0, y0)

    room_mask = (mask[1:-1, 1:-1] != 0).astype(np.uint8) * 255
    area_px = int(np.count_nonzero(room_mask))
    return room_mask, area_px, (x0, y0)


def save_step(name, img, steps_dir):
    """Salvează un step de debug."""
    if steps_dir:
        path = Path(steps_dir) / f"{name}.png"
        cv2.imwrite(str(path), img)


def detect_scale_from_room_labels(image_path, indoor_mask, walls_mask, steps_dir, min_room_area=1.0, max_room_area=300.0, api_key=None):
    """Detectează scala din label-uri de camere."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None: return None
    
    h, w = img_bgr.shape[:2]

    # A. SEGMENTARE CAMERE
    room_candidates = []
    min_dim = min(h, w)
    kernel_size = max(3, int(min_dim / 100))
    if kernel_size % 2 == 0: kernel_size += 1
    
    kernel_dynamic = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    walls_dilated = cv2.dilate(walls_mask, kernel_dynamic, iterations=1)
    fillable_mask = cv2.bitwise_and(cv2.bitwise_not(walls_dilated), indoor_mask)
    unvisited_mask = fillable_mask.copy()
    
    sample_stride = max(5, int(w / 200))
    MIN_PIXEL_AREA = max(100, int((w * h) * 0.0005))
    room_idx = 0
    debug_iteration = img_bgr.copy()

    print(f"   🔍 Segmentare camere (Kernel {kernel_size}px)...")
    
    for y in range(0, h, sample_stride):
        for x in range(0, w, sample_stride):
            # Verificăm dacă coordonatele sunt în limitele valide
            if y >= unvisited_mask.shape[0] or x >= unvisited_mask.shape[1]:
                continue
            if unvisited_mask[y, x] > 0:
                room_mask, area_px, _ = flood_fill_room(unvisited_mask, (x, y))
                
                if area_px < MIN_PIXEL_AREA:
                    unvisited_mask[room_mask > 0] = 0
                    continue
                
                room_idx += 1
                coords = np.where(room_mask > 0)
                if not coords[0].size: continue
                
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                padding = max(5, int(min_dim / 100))
                y_s, y_e = max(0, y_min - padding), min(h, y_max + 1 + padding)
                x_s, x_e = max(0, x_min - padding), min(w, x_max + 1 + padding)
                
                crop_path = Path(steps_dir) / f"crop_{room_idx}.png"
                cv2.imwrite(str(crop_path), img_bgr[y_s:y_e, x_s:x_e])
                
                print(f"      ⏳ Segment {room_idx} ({area_px} px)...")
                
                res = call_gemini(crop_path, GEMINI_PROMPT_CROP, api_key)
                if res and is_informational_total_result(res):
                    res = None
                if res and 'area_m2' in res and res['area_m2'] is not None:
                    try:
                        area_m2 = float(res['area_m2'])
                        if min_room_area <= area_m2 <= max_room_area:
                            room_candidates.append({
                                "index": room_idx,
                                "area_m2_label": area_m2,
                                "area_px": int(area_px),
                                "room_name": res.get('room_name', 'Unknown')
                            })
                            print(f"         ✅ {res.get('room_name')} -> {area_m2} m²")
                            cv2.rectangle(debug_iteration, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        else:
                            cv2.rectangle(debug_iteration, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
                    except ValueError: pass
                else:
                    cv2.rectangle(debug_iteration, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
                
                unvisited_mask[room_mask > 0] = 0
    
    save_step("debug_segmentation_final", debug_iteration, steps_dir)
    
    if not room_candidates:
        print("      ❌ Nu s-au găsit camere valide")
        return None

    # B. CALCUL SCARĂ
    area_labels = np.array([r["area_m2_label"] for r in room_candidates])
    area_pixels = np.array([r["area_px"] for r in room_candidates])
    
    num = np.sum(area_pixels * area_labels)
    den = np.sum(area_pixels ** 2)
    
    if den == 0: return None
    
    m_px_local = float(np.sqrt(num / den))
    print(f"   📊 Scara Locală: {m_px_local:.9f} m/px")

    total_indoor_px = int(np.count_nonzero(indoor_mask))
    gemini_total = call_gemini(image_path, GEMINI_PROMPT_TOTAL_SUM, api_key)
    
    sum_labels_m2 = sum(area_labels)
    if gemini_total and 'total_sum_m2' in gemini_total:
        try: sum_labels_m2 = float(gemini_total['total_sum_m2'])
        except: pass

    m_px_target = math.sqrt(sum_labels_m2 / float(total_indoor_px)) if total_indoor_px > 0 else m_px_local
    suprafata_cu_camere = total_indoor_px * (m_px_local ** 2)
    
    final_m_px = m_px_local
    method_note = "Scară Locală"
    
    if suprafata_cu_camere < sum_labels_m2:
        final_m_px = m_px_target
        method_note = "Forțat pe Suma Totală"
    
    print(f"   ✅ Scară Finală: {final_m_px:.9f} m/px")

    return {
        "method": "cubicasa_gemini",
        "meters_per_pixel": float(final_m_px),
        "rooms_used": len(room_candidates),
        "optimization_info": {"method_note": method_note},
        "per_room": room_candidates
    }
