# roof/roof_type_classifier.py
# Clasificare DOAR tip acoperiș per etaj cu Gemini (pe baza imaginilor side_view).
# Unghiul/panta acoperișului se introduce în formular (detalii acoperiș), nu de la Gemini.

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

ROOF_TYPES = ("0_w", "1_w", "2_w", "4_w", "4.5_w")


PROMPT_ROOF_TYPE_PER_FLOOR = """You are an expert in roof types for residential buildings.

These images show exterior views (side views) of a building with {num_floors} floor(s) that have a roof. Ignore basement – no roof there.

Floors from bottom to top: {floor_names}

For each floor, say what roof type you see. Roof types (you can use the code or describe in words):
- 0_w = flat roof, no slope / Flachdach (perfectly flat, no pitch)
- 1_w = single slope / Satteldach einseitig
- 2_w = gable, two slopes / Satteldach
- 4_w = hip, four slopes / Walmdach
- 4.5_w = half-hip / Krüppelwalmdach

Describe briefly for each floor (parter, etaj_1, etc.). Any format is fine – we will convert it to structured data later."""


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


def _dict_to_floor_result(data: dict, num_floors: int) -> Dict[int, str]:
    """Convertește dict parsabil în Dict[floor_idx, roof_type]."""
    out: Dict[int, str] = {}
    if not isinstance(data, dict):
        return out
    floor_names = ["parter"] + [f"etaj_{i}" for i in range(1, num_floors)]
    for idx, name in enumerate(floor_names):
        val = data.get(name)
        if isinstance(val, dict):
            t = val.get("type")
            if isinstance(t, str) and t in ROOF_TYPES:
                out[idx] = t
            elif isinstance(t, str):
                nv = t.replace(".", "_").replace(" ", "_").strip().lower()
                mapping = {"0w": "0_w", "1w": "1_w", "2w": "2_w", "4w": "4_w", "45w": "4.5_w", "4_5w": "4.5_w"}
                if nv in mapping:
                    out[idx] = mapping[nv]
        elif isinstance(val, str) and val in ROOF_TYPES:
            out[idx] = val
        elif isinstance(val, str):
            nv = val.replace(".", "_").replace(" ", "_").strip().lower()
            mapping = {"0w": "0_w", "1w": "1_w", "2w": "2_w", "4w": "4_w", "45w": "4.5_w", "4_5w": "4.5_w"}
            if nv in mapping:
                out[idx] = mapping[nv]
    return out


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
    out = _dict_to_floor_result(data, num_floors)
    if len(out) < num_floors:
        nl = _parse_roof_types_natural_language(original_text, num_floors)
        for i in range(num_floors):
            if i not in out and i in nl:
                out[i] = nl[i]
    return out


def _build_roof_types_json_with_gpt(gemini_raw_text: str, num_floors: int) -> Dict[int, str]:
    """Trimite mereu răspunsul Gemini la ChatGPT; acesta construiește JSON-ul canonic (chei etaj -> tip acoperiș)."""
    floor_keys = ["parter"] + [f"etaj_{i}" for i in range(1, num_floors)]
    schema_hint = (
        f"JSON object with exactly these keys: {', '.join(floor_keys)}. "
        "Each value must be one of: 0_w, 1_w, 2_w, 4_w, 4.5_w (roof type codes)."
    )
    try:
        from common.json_repair import repair_json_with_gpt
        data = repair_json_with_gpt(gemini_raw_text, schema_hint)
        if data:
            out = _dict_to_floor_result(data, num_floors)
            if len(out) == num_floors:
                print(f"   [RoofTypeClassifier] ChatGPT a construit JSON: {out}", flush=True)
                return out
    except Exception as e:
        print(f"   [RoofTypeClassifier] Eroare ChatGPT (build JSON): {e}", flush=True)
    return {}


def classify_roof_types_per_floor(
    gemini_client: Any,
    side_view_images: List[Path],
    num_floors: int,
) -> Dict[int, str]:
    """
    Trimite imaginile side_view la Gemini și returnează DOAR tipul acoperișului per etaj.
    Unghiul/panta se introduce în formular (detalii acoperiș), nu se extrage de aici.

    Args:
        gemini_client: client Gemini (genai.GenerativeModel)
        side_view_images: listă de path-uri către imagini side_view
        num_floors: număr de etaje (fără beci)

    Returns:
        floor_roof_types: {0: "2_w", 1: "4_w", ...}. La eșec: dict gol; apelantul folosește fallback (2_w).
    """
    empty: Dict[int, str] = {}
    if not gemini_client or num_floors < 1:
        return empty
    if not side_view_images:
        return empty

    from segmenter.classifier import prep_for_vlm

    floor_keys_list = ["parter"] + [f"etaj_{i}" for i in range(1, num_floors)]
    prompt = PROMPT_ROOF_TYPE_PER_FLOOR.format(
        num_floors=num_floors,
        floor_names=", ".join(floor_keys_list),
    )

    parts: list = [prompt]
    for p in side_view_images:
        if not Path(p).exists():
            continue
        try:
            pil = prep_for_vlm(p)
            parts.append(pil)
        except Exception:
            continue

    if len(parts) < 2:
        return empty

    gen_config = {
        "temperature": 0.2,
        "max_output_tokens": 1024,
    }
    try:
        print("   [RoofTypeClassifier] Trimit imaginile la Gemini (generate_content)...", flush=True)
        response = gemini_client.generate_content(parts, generation_config=gen_config)
        print("   [RoofTypeClassifier] Răspuns Gemini primit.", flush=True)
        if not response or not response.parts:
            print("   [RoofTypeClassifier] NU FOLOSIM date Gemini: răspuns gol (response sau response.parts lipsă).", flush=True)
            return empty
        text = (response.text or "").strip()
        if len(text) > 400:
            print(f"   [RoofTypeClassifier] Raw Gemini (primele 400 chr): {text[:400]!r}...", flush=True)
        else:
            print(f"   [RoofTypeClassifier] Raw Gemini: {text!r}", flush=True)
        if not text:
            print(f"   [RoofTypeClassifier] Răspuns Gemini gol – NU FOLOSIM.", flush=True)
            return empty
        # Mereu trimitem răspunsul Gemini la ChatGPT pentru a construi JSON-ul canonic (structură etaj -> tip)
        print(f"   [RoofTypeClassifier] Trimit răspunsul Gemini la ChatGPT pentru construirea JSON-ului...", flush=True)
        built = _build_roof_types_json_with_gpt(text, num_floors)
        if built:
            return built
        # Fallback: parse local (JSON sau limbaj natural: "parter ... flat roofs (0_w)" etc.)
        types_result = _parse_roof_types_response(text, num_floors)
        if len(types_result) == num_floors:
            print(f"   [RoofTypeClassifier] Fallback parse local: types={types_result} – FOLOSIM.", flush=True)
            return types_result
        if types_result:
            for i in range(num_floors):
                if i not in types_result:
                    types_result[i] = "2_w"
            print(f"   [RoofTypeClassifier] Parse parțial + completare 2_w: types={types_result} – FOLOSIM.", flush=True)
            return types_result
        print(f"   [RoofTypeClassifier] Nici ChatGPT nu a putut construi rezultat complet – NU FOLOSIM.", flush=True)
    except Exception as e:
        print(f"   [RoofTypeClassifier] Eroare Gemini: {e}", flush=True)
    return empty
