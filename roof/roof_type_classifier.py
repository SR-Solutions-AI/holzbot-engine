# roof/roof_type_classifier.py
# Clasificare tip acoperiș ȘI unghi/pantă per etaj cu Gemini (pe baza imaginilor side_view).
# Unghiul folosit la construcție este cel returnat de Gemini (per etaj).

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOF_TYPES = ("0_w", "1_w", "2_w", "4_w", "4.5_w")
DEFAULT_ROOF_ANGLE_DEG = 30.0


PROMPT_ROOF_TYPE_PER_FLOOR = """You are an expert in roof types and roof pitch for residential buildings.

You will receive ALL side-view images of the building (facades, cross-sections / Schnitt, perspective). Use every image to decide the roof type and pitch per floor.

CRITICAL for cross-sections / section views (e.g. "Schnitt", "Section"):
- The roof drawn ABOVE the top floor (etaj_1, etaj_2, ...) is THAT floor's roof. If that roof has visible slope, a ridge line, or triangular/gable shape, it is NOT flat: use 2_w (gable) or 4_w (hip) with angle_deg 15–45°, not 0_w.
- Only use 0_w (flat, angle 0) when the roof above that floor is clearly horizontal with no slope.
- A single-story annex with a flat roof does not change the main building's top-floor roof: if the section shows a pitched roof above the upper floor of the main building, that floor must be 2_w (or 4_w) with an angle.

The building has {num_floors} floor(s) that have a roof (ignore basement – no roof there). Floors from bottom to top:
{floor_names}

For each floor you must provide:
1) Roof type (use exactly this code):
   - 0_w = flat roof, no slope / Flachdach (perfectly flat → angle 0)
   - 1_w = single slope / Satteldach einseitig
   - 2_w = gable, two slopes / Satteldach
   - 4_w = hip, four slopes / Walmdach
   - 4.5_w = half-hip / Krüppelwalmdach

2) Roof pitch / angle in degrees (Dachneigung): the slope of the roof relative to horizontal.
   - Flat roof (0_w): use 0.
   - Sloped roofs: typical range 15°–45°. Estimate from the section/side view.
   - One number per floor, in degrees (e.g. 25, 30, 35).

Return a JSON object with one key per floor (parter, etaj_1, ...). Each value is an object with:
- "type": one of 0_w, 1_w, 2_w, 4_w, 4.5_w
- "angle_deg": number (0 for flat, 10–70 for sloped roofs)

Example: {{ "parter": {{ "type": "0_w", "angle_deg": 0 }}, "etaj_1": {{ "type": "2_w", "angle_deg": 28 }} }}"""


def _floor_names(num_floors: int) -> str:
    names = ["parter"]
    for i in range(1, num_floors):
        names.append(f"etaj_{i}")
    return "\n".join(f"- {n}" for n in names)


def _roof_types_json_schema(num_floors: int) -> dict:
    """Schema JSON pentru răspuns structurat Gemini (response_mime_type=application/json)."""
    floor_names = ["parter"] + [f"etaj_{i}" for i in range(1, num_floors)]
    properties = {
        name: {
            "type": "string",
            "enum": list(ROOF_TYPES),
            "description": f"Roof type for {name}",
        }
        for name in floor_names
    }
    return {
        "type": "object",
        "properties": properties,
        "required": floor_names,
    }


def _extract_json_object(text: str) -> str | None:
    """Extrage primul obiect JSON complet din text (inclusiv cu prefixe)."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    quote = None
    for i, c in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if not in_string:
            if c in ('"', "'"):
                in_string = True
                quote = c
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        else:
            if c == quote:
                in_string = False
    return None


def _dict_to_floor_result(data: dict, num_floors: int) -> Tuple[Dict[int, str], Dict[int, float]]:
    """Convertește dict parsabil în (Dict[floor_idx, roof_type], Dict[floor_idx, angle_deg])."""
    types_out: Dict[int, str] = {}
    angles_out: Dict[int, float] = {}
    if not isinstance(data, dict):
        return types_out, angles_out
    floor_names = ["parter"] + [f"etaj_{i}" for i in range(1, num_floors)]
    for idx, name in enumerate(floor_names):
        val = data.get(name)
        if isinstance(val, dict):
            t = val.get("type")
            a = val.get("angle_deg")
            if isinstance(t, str) and t in ROOF_TYPES:
                types_out[idx] = t
            elif isinstance(t, str):
                nv = t.replace(".", "_").replace(" ", "_").strip().lower()
                mapping = {"0w": "0_w", "1w": "1_w", "2w": "2_w", "4w": "4_w", "45w": "4.5_w", "4_5w": "4.5_w"}
                if nv in mapping:
                    types_out[idx] = mapping[nv]
            if a is not None:
                try:
                    angles_out[idx] = max(0, min(70, float(a)))
                except (TypeError, ValueError):
                    angles_out[idx] = DEFAULT_ROOF_ANGLE_DEG if types_out.get(idx) != "0_w" else 0.0
            elif types_out.get(idx) == "0_w":
                angles_out[idx] = 0.0
            else:
                angles_out[idx] = DEFAULT_ROOF_ANGLE_DEG
        elif isinstance(val, str) and val in ROOF_TYPES:
            types_out[idx] = val
            angles_out[idx] = 0.0 if val == "0_w" else DEFAULT_ROOF_ANGLE_DEG
        elif isinstance(val, str):
            nv = val.replace(".", "_").replace(" ", "_").strip().lower()
            mapping = {"0w": "0_w", "1w": "1_w", "2w": "2_w", "4w": "4_w", "45w": "4.5_w", "4_5w": "4.5_w"}
            if nv in mapping:
                types_out[idx] = mapping[nv]
                angles_out[idx] = 0.0 if mapping[nv] == "0_w" else DEFAULT_ROOF_ANGLE_DEG
    return types_out, angles_out


def _parse_roof_types_natural_language(text: str, num_floors: int) -> Dict[int, str]:
    """Extrage tipuri acoperiș din text natural (ex: 'parter ... flat roofs (0_w)', 'etaj_1 ... 2_w')."""
    out: Dict[int, str] = {}
    if not text or num_floors < 1:
        return out
    # Caută per etaj: nume etaj urmat (în ~200 caractere) de un cod 0_w, 1_w, 2_w, 4_w, 4.5_w
    text_lower = text.lower()
    for idx in range(num_floors):
        if idx == 0:
            names = ["parter", "ground floor", "ground", "eg ", "erdgeschoss", "floor 0"]
        else:
            names = [f"etaj_{idx}", f"etaj {idx}", f"floor {idx}", f"floor_{idx}"]
        for name in names:
            pos = text_lower.find(name)
            if pos < 0:
                continue
            snippet = text[pos : pos + 220]
            for code in ROOF_TYPES:
                if code in snippet:
                    out[idx] = code
                    break
            if idx not in out and ("flat" in snippet or "flachdach" in snippet):
                out[idx] = "0_w"
            if idx in out:
                break
    return out


def _parse_roof_types_response(text: str, num_floors: int) -> Dict[int, str]:
    """Parsează răspunsul Gemini în Dict[floor_idx, roof_type]."""
    out: Dict[int, str] = {}
    if not text:
        return out
    original_text = text.strip()
    text = original_text
    # Elimină prefixe tip "Here is the JSON requested:" etc. (oricare ordine cuvinte)
    for prefix in (
        "here is the json requested:",
        "here is the json:",
        "here's the json:",
        "here is the requested json:",
        "json:",
    ):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    # Extrage obiectul JSON complet (rezolvă prefixe + texte după })
    json_str = _extract_json_object(text)
    if json_str:
        text = json_str
    elif "{" in text:
        text = text[text.find("{") :]
    data: dict = {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        if not data:
            m2 = re.search(r"\{[^}]*\"parter\"[^}]*\"(?:0_w|1_w|2_w|4_w|4\.5_w)\"[^}]*\}", text)
            if m2:
                try:
                    data = json.loads(m2.group(0))
                except json.JSONDecodeError:
                    pass
            if not data:
                return _parse_roof_types_natural_language(original_text, num_floors)
    if not isinstance(data, dict):
        return _parse_roof_types_natural_language(original_text, num_floors)
    types_out, _ = _dict_to_floor_result(data, num_floors)
    if len(types_out) < num_floors:
        nl = _parse_roof_types_natural_language(original_text, num_floors)
        for i in range(num_floors):
            if i not in types_out and i in nl:
                types_out[i] = nl[i]
    return types_out


def _build_roof_types_and_angles_json_with_gpt(gemini_raw_text: str, num_floors: int) -> Tuple[Dict[int, str], Dict[int, float]]:
    """Trimite răspunsul Gemini la ChatGPT; acesta construiește JSON cu tip + angle_deg per etaj."""
    floor_keys = ["parter"] + [f"etaj_{i}" for i in range(1, num_floors)]
    schema_hint = (
        f"JSON object with keys: {', '.join(floor_keys)}. "
        "Each value must be an object with 'type' (one of: 0_w, 1_w, 2_w, 4_w, 4.5_w) and 'angle_deg' (number 0 for flat, 10-70 for sloped roofs). "
        "Example: {\"parter\": {\"type\": \"0_w\", \"angle_deg\": 0}, \"etaj_1\": {\"type\": \"2_w\", \"angle_deg\": 28}}"
    )
    try:
        from common.json_repair import repair_json_with_gpt
        data = repair_json_with_gpt(gemini_raw_text, schema_hint)
        if data:
            types_out, angles_out = _dict_to_floor_result(data, num_floors)
            if len(types_out) == num_floors:
                for i in range(num_floors):
                    if i not in angles_out:
                        angles_out[i] = 0.0 if types_out.get(i) == "0_w" else DEFAULT_ROOF_ANGLE_DEG
                print(f"   [RoofTypeClassifier] ChatGPT a construit JSON: types={types_out}, angles={angles_out}", flush=True)
                return types_out, angles_out
    except Exception as e:
        print(f"   [RoofTypeClassifier] Eroare ChatGPT (build JSON): {e}", flush=True)
    return {}, {}


def classify_roof_types_per_floor(
    gemini_client: Any,
    side_view_images: List[Path],
    num_floors: int,
) -> Tuple[Dict[int, str], Dict[int, float]]:
    """
    Trimite imaginile side_view la Gemini și returnează tipul acoperișului ȘI unghiul (panta) per etaj.
    Unghiul folosit la construcție este cel returnat de Gemini (per etaj).

    Args:
        gemini_client: client Gemini (genai.GenerativeModel)
        side_view_images: listă de path-uri către imagini side_view
        num_floors: număr de etaje (fără beci)

    Returns:
        (floor_roof_types, floor_roof_angles): ({0: "2_w", ...}, {0: 0.0, 1: 28.0, ...}).
        La eșec tipuri: (dict gol, dict gol); apelantul folosește fallback (2_w, default 30°).
    """
    empty_types: Dict[int, str] = {}
    empty_angles: Dict[int, float] = {}
    if not gemini_client or num_floors < 1:
        return empty_types, empty_angles
    if not side_view_images:
        return empty_types, empty_angles

    from segmenter.classifier import prep_for_vlm

    floor_keys_list = ["parter"] + [f"etaj_{i}" for i in range(1, num_floors)]
    floor_names_text = _floor_names(num_floors)
    prompt = PROMPT_ROOF_TYPE_PER_FLOOR.format(
        num_floors=num_floors,
        floor_names=floor_names_text,
    )

    parts: list = [prompt]
    added = 0
    for i, p in enumerate(side_view_images):
        if not Path(p).exists():
            continue
        try:
            pil = prep_for_vlm(p)
            # Etichetează fiecare imagine ca să asocieze view-urile cu etajele
            label = f"\n[Image {added + 1} of {len(side_view_images)} side-view/cross-section]\n"
            parts.append(label)
            parts.append(pil)
            added += 1
        except Exception:
            continue

    if len(parts) < 2:
        print(f"   [RoofTypeClassifier] Nu s-au putut încărca imagini side_view (încărcate: {added}, totale: {len(side_view_images)}) – NU RULEAZĂ.", flush=True)
        return empty_types, empty_angles
    print(f"   [RoofTypeClassifier] Trimit la Gemini {added} imagini side_view (din {len(side_view_images)} găsite).", flush=True)

    gen_config = {
        "temperature": 0.2,
        "max_output_tokens": 2048,
    }
    try:
        print("   [RoofTypeClassifier] Trimit imaginile la Gemini (generate_content)...", flush=True)
        response = gemini_client.generate_content(parts, generation_config=gen_config)
        print("   [RoofTypeClassifier] Răspuns Gemini primit.", flush=True)
        if not response:
            print("   [RoofTypeClassifier] NU FOLOSIM date Gemini: răspuns gol.", flush=True)
            return empty_types, empty_angles
        # Text din toate părțile (uneori response.text e trunchiat la prima parte)
        text = ""
        if response.candidates:
            c = response.candidates[0]
            if getattr(c, "content", None) and getattr(c.content, "parts", None):
                for part in c.content.parts:
                    text += getattr(part, "text", "") or ""
        if not text:
            text = (response.text or "").strip()
        text = text.strip()
        if not response.candidates or not text:
            print("   [RoofTypeClassifier] NU FOLOSIM date Gemini: răspuns gol (fără candidates sau text).", flush=True)
            return empty_types, empty_angles
        if len(text) > 400:
            print(f"   [RoofTypeClassifier] Raw Gemini (primele 400 chr): {text[:400]!r}...", flush=True)
        else:
            print(f"   [RoofTypeClassifier] Raw Gemini: {text!r}", flush=True)
        if os.environ.get("ROOF_DEBUG_FULL_RESPONSE"):
            print("   [RoofTypeClassifier] FULL Raw Gemini:", text, flush=True)
        if not text:
            print(f"   [RoofTypeClassifier] Răspuns Gemini gol – NU FOLOSIM.", flush=True)
            return empty_types, empty_angles
        # Trimitem răspunsul Gemini la ChatGPT pentru JSON canonic (tip + angle_deg per etaj)
        print(f"   [RoofTypeClassifier] Trimit răspunsul Gemini la ChatGPT pentru construirea JSON (tip + unghi)...", flush=True)
        built_types, built_angles = _build_roof_types_and_angles_json_with_gpt(text, num_floors)
        if built_types:
            return built_types, built_angles
        # Fallback: parse local doar tipuri; unghiuri = default (și 0 pentru 0_w)
        types_result = _parse_roof_types_response(text, num_floors)
        if len(types_result) == num_floors:
            angles_result = {}
            for i in range(num_floors):
                angles_result[i] = 0.0 if types_result.get(i) == "0_w" else DEFAULT_ROOF_ANGLE_DEG
            print(f"   [RoofTypeClassifier] Fallback parse local: types={types_result}, angles={angles_result} (default) – FOLOSIM.", flush=True)
            return types_result, angles_result
        if types_result:
            for i in range(num_floors):
                if i not in types_result:
                    types_result[i] = "2_w"
            angles_result = {i: (0.0 if types_result[i] == "0_w" else DEFAULT_ROOF_ANGLE_DEG) for i in range(num_floors)}
            print(f"   [RoofTypeClassifier] Parse parțial + completare: types={types_result}, angles={angles_result} – FOLOSIM.", flush=True)
            return types_result, angles_result
        print(f"   [RoofTypeClassifier] Nici ChatGPT nu a putut construi rezultat complet – NU FOLOSIM.", flush=True)
    except Exception as e:
        print(f"   [RoofTypeClassifier] Eroare Gemini: {e}", flush=True)
    return empty_types, empty_angles
