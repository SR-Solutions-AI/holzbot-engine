# file: engine/segmenter/gemini_crop.py
"""
Extragere planuri (etaje) și side view-uri prin Gemini: coordonate 0–1000 → crop → salvare.
Este singurul flux de „segmentare”: pipeline-ul vechi (clusters, classifier) este dezactivat.
Format așteptat: [{"box_2d": [ymin, xmin, ymax, xmax], "label": "..."}, ...] cu box_2d în scară 0–1000
(sau 0–1); la parsare normalizăm tot la 0–1 pentru crop.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image

from common.gemini_rest import DEFAULT_GEMINI_MODEL, gemini_api_key, generate_json

from .classifier import ClassificationResult, ConfidenceLevel, setup_gemini_client
from .common import STEP_DIRS

_GEMINI_SAFETY = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

PROMPT_BOXES = """You are an expert in visual analysis for architectural drawings. Analyze the attached image and identify ALL of the following:

1) FLOOR PLANS (Blueprints): Top-down plans of individual floors — e.g. GRUNDRISS KG, GRUNDRISS EG, GRUNDRISS OG, or any "Grundriss" / floor plan. For each one, return the exact label (e.g. "GRUNDRISS KG", "GRUNDRISS EG") and its bounding box.
2) SIDE VIEWS & SECTIONS: Elevations (Ansicht Ost, Ansicht Süd, Ansicht West, Ansicht Nord) and sections (Schnitt A-A, etc.). For each such view, return its label and bounding box.

CRITICAL FOR FLOOR PLANS: For each floor plan (blueprint), the bounding box must include the ENTIRE drawn floor plan: the building/house AND all attached areas (garage, terrace, balcony, covered entrance / Eingang überdacht, loggia, patio). Do NOT cut off garages, terraces, balconies or covered entrances. You may trim large empty margins and unrelated context (e.g. distant pools, separate gardens, legend text), but the box must show the full outline of the house including garage and any attached outdoor spaces so that room detection and scale work correctly.

FLOOR ORDER ON THIS PAGE: You must also return the stacking order of the FLOOR PLANS on this page from BOTTOM to TOP (building levels: basement lowest, then ground floor, then upper floors). The floor plans appear in your "items" array; some items are floor plans, some are side views. Ignore side views for ordering. Number the FLOOR plans in the order they appear in the array as 0, 1, 2, ... (first floor plan = 0, second = 1, etc.). Return "floor_order": an array of these indices in order from BOTTOM to TOP. Example: if you have 3 floor plans at array positions 0, 1, 2 and they are KG, EG, OG, then floor_order must be [0, 1, 2]. If the order on the page is EG, KG, OG (positions 0,1,2), then floor_order must be [1, 0, 2] (basement KG first, then EG, then OG).

IGNORE: the general site/lot plan. Do NOT include Lageplan / Abstandsflächenplan / Situationsplan as floor plans. If the whole page is one big site map, still extract any smaller building floor-plan or elevation boxes inside it. Exclude 3D color renderings and project data tables in corners. Focus on technical drawings only.

COORDINATE SYSTEM: Use bounding boxes in format [ymin, xmin, ymax, xmax] normalized on a scale of 0 to 1000. So the image width and height each map to 0–1000. Example: left half of image ≈ xmin 0, xmax 500; top 10% ≈ ymin 0, ymax 100.

Return a single JSON OBJECT with two keys:
- "items": array of elements. Each element must have:
  - "box_2d": [ymin, xmin, ymax, xmax] — integers or numbers in 0–1000 range.
  - "label": string — the exact label you see (e.g. "GRUNDRISS EG", "Ansicht Süd", "Schnitt A-A").
- "floor_order": array of integers — the indices of the FLOOR plan items (0 = first floor plan in "items", 1 = second floor plan, ...) in order from BOTTOM to TOP. If there is only one floor plan, use [0]. If there are no floor plans, use [].

You MUST include EVERY floor plan and every side view/section on the page. Do not omit any. Output ONLY this JSON object, no markdown, no code block, no text before or after. Start with { and end with }.

Example format:
{"items": [{"box_2d": [30, 15, 550, 485], "label": "GRUNDRISS KG"}, {"box_2d": [30, 510, 340, 970], "label": "Ansicht Süd"}, {"box_2d": [570, 15, 980, 550], "label": "GRUNDRISS EG"}], "floor_order": [0, 2]}

(Here the first and third items are floor plans; KG is bottom so index 0 first, then EG index 2. Side view is not in floor_order.)

Output ONLY the raw JSON object."""

# Fallback prompt când primul apel clasifică totul ca side_view:
# forțăm extragerea planurilor de etaj (blueprints) cu reguli mai stricte.
PROMPT_BOXES_FLOOR_RETRY = """You previously failed to extract floor plans correctly.
Now do a second strict pass and prioritize FLOOR PLANS (top-down blueprints).

Task:
1) Extract ALL FLOOR PLANS on the page (Grundriss KG/EG/OG, floor plans).
2) Also extract side views/sections if present, but DO NOT replace floor plans with side views.

Critical:
- If the page contains floor plans, you MUST return at least one item labeled as floor.
- A floor plan can have attached garage/terrace/balcony/covered entrance; keep them inside the box.
- If uncertain between floor vs side_view, prefer floor when it is top-down architectural plan geometry.
- NEVER classify site plans as floors: Lageplan, Abstandsflächenplan (Abstandsflaechenplan), Situationsplan must be side_view/non-floor.

Return ONLY this JSON object:
{
  "items": [{"box_2d":[ymin,xmin,ymax,xmax], "label":"floor"|"side_view"}],
  "floor_order": [indices of floor items from BOTTOM to TOP]
}
Coordinates must be normalized 0..1000 scale.
No markdown, no explanation."""


# Aliniere două măști de pereți: Gemini returnează procente (scara 0–1000), ca la box_2d
PROMPT_WALL_ALIGNMENT = """You are an expert in aligning floor plan wall masks. You see TWO images:

- IMAGE 1 = REFERENCE (original walls). This is the ground truth.
- IMAGE 2 = TEMPLATE (detected walls, e.g. from API). It must be SCALED and SHIFTED to overlay exactly on IMAGE 1.

Both show the SAME floor plan. No rotation: only scale and translation. Origin is top-left; x right, y down.

Return a single JSON object with these keys. Use scale 0–1000 for percentages (like our coordinate standard):

- "scale_1000": number 200–2000. Meaning: scale factor for IMAGE 2 so it matches IMAGE 1 size. 1000 = 1.0 (same size), 600 = 0.6 (shrink to 60%), 1200 = 1.2 (enlarge to 120%).
- "offset_x_1000": number -500 to 500. Horizontal shift of the CENTER of (scaled) IMAGE 2 relative to the CENTER of IMAGE 1, in tenths of percent of IMAGE 1 width. 0 = centered; -500 = template center 50% of width to the left; 500 = 50% to the right.
- "offset_y_1000": number -500 to 500. Same for vertical. 0 = centered; negative = up, positive = down.
- "direction": "api_to_orig" (scale and place IMAGE 2 onto IMAGE 1) or "orig_to_api" (place IMAGE 1 onto IMAGE 2).
- "confidence_1000": number 0–1000. How sure you are (0 = guess, 1000 = very confident).

Reply with ONLY the JSON object. No markdown, no text before or after."""


def get_gemini_wall_alignment(
    reference_image_path: str | Path,
    template_image_path: str | Path,
) -> dict[str, Any] | None:
    """
    Trimite cele două măști (referință + template) la Gemini; returnează cum suprapune template peste referință.
    Folosește același client și pattern ca get_gemini_boxes_for_page: procente în scară 0–1000.

    Returns:
        dict cu: scale (float 0.2–2.0), offset_x_pct, offset_y_pct (fracțiuni -0.5..0.5),
        direction ("api_to_orig" | "orig_to_api"), confidence (0–1). Sau None la eroare.
    """
    client = setup_gemini_client()
    if not client:
        return None

    ref_path = Path(reference_image_path)
    tpl_path = Path(template_image_path)
    if not ref_path.is_file() or not tpl_path.is_file():
        return None

    try:
        img_ref = Image.open(ref_path).convert("RGB")
        img_tpl = Image.open(tpl_path).convert("RGB")
    except Exception:
        return None

    try:
        response = client.generate_content(
            [PROMPT_WALL_ALIGNMENT, img_ref, img_tpl],
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 512,
            },
            safety_settings=_GEMINI_SAFETY,
        )
    except Exception as e:
        print(f"      ⚠️ Gemini wall alignment: {e}")
        return None

    raw = _get_response_text(response)
    if not raw or not raw.strip():
        return None

    raw = raw.strip()
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        out = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None
    if not isinstance(out, dict):
        return None

    # Normalize din 0–1000 la valorile folosite de raster_api
    scale_1000 = out.get("scale_1000")
    if scale_1000 is not None:
        scale = float(scale_1000) / 1000.0
        scale = max(0.2, min(2.0, scale))
    else:
        scale = 1.0
    ox = out.get("offset_x_1000")
    oy = out.get("offset_y_1000")
    offset_x_pct = (float(ox) / 1000.0) if ox is not None else 0.0  # -0.5 .. 0.5
    offset_y_pct = (float(oy) / 1000.0) if oy is not None else 0.0
    offset_x_pct = max(-0.5, min(0.5, offset_x_pct))
    offset_y_pct = max(-0.5, min(0.5, offset_y_pct))
    conf = out.get("confidence_1000")
    confidence = (float(conf) / 1000.0) if conf is not None else 0.5
    confidence = max(0.0, min(1.0, confidence))
    direction = "api_to_orig" if str(out.get("direction", "api_to_orig")).strip().lower() != "orig_to_api" else "orig_to_api"

    return {
        "scale": scale,
        "offset_x_pct": offset_x_pct,
        "offset_y_pct": offset_y_pct,
        "direction": direction,
        "confidence": confidence,
    }


def _get_response_text(response: Any) -> str:
    """Extrage textul din răspunsul Gemini (compatibil cu diverse versiuni API)."""
    if not response:
        return ""
    try:
        if getattr(response, "text", None):
            return response.text
    except Exception:
        pass
    try:
        if getattr(response, "candidates", None) and response.candidates:
            c0 = response.candidates[0]
            content = getattr(c0, "content", None)
            if content and getattr(content, "parts", None) and content.parts:
                for part in content.parts:
                    if getattr(part, "text", None):
                        return part.text
    except Exception:
        pass
    return ""


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    if not text or not text.strip():
        return []
    text = text.strip()
    # Elimină markdown code block dacă există
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    # Caută primul '[' valid (start of array)
    start = text.find("[")
    if start == -1:
        return []
    depth = 0
    end = -1
    in_string = False
    escape = False
    quote_char = None
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == quote_char:
                in_string = False
            continue
        if c in ('"', "'"):
            in_string = True
            quote_char = c
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return []
    chunk = text[start:end]
    try:
        data = json.loads(chunk)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        pass
    # Retry: trailing comma sau newline-uri care deranjează
    chunk_clean = re.sub(r",\s*]", "]", chunk)
    chunk_clean = re.sub(r",\s*}", "}", chunk_clean)
    chunk_clean = chunk_clean.strip()
    try:
        data = json.loads(chunk_clean)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _extract_json_object(text: str) -> tuple[list[dict[str, Any]], list[int]]:
    """
    Încearcă să parseze răspunsul ca obiect JSON cu "items" și "floor_order".
    Returnează (items, floor_order). Dacă nu e obiect valid, returnează ([], []).
    """
    if not text or not text.strip():
        return [], []
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    if start == -1:
        return [], []
    depth = 0
    end = -1
    in_string = False
    escape = False
    quote_char = None
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == quote_char:
                in_string = False
            continue
        if c in ('"', "'"):
            in_string = True
            quote_char = c
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return [], []
    chunk = text[start:end]
    try:
        data = json.loads(chunk)
        if not isinstance(data, dict):
            return [], []
        items = data.get("items")
        if not isinstance(items, list):
            items = []
        order = data.get("floor_order")
        if isinstance(order, list):
            order = [int(x) for x in order if isinstance(x, (int, float))]
        else:
            order = []
        return items, order
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return [], []


PROMPT_REPAIR_JSON = """The following text was returned by an AI that was asked to output a JSON array of bounding boxes for architectural floor plans and side views. The response is malformed or in the wrong format.

Your task: output ONLY a valid JSON array. Each element must have:
- "box_2d": [ymin, xmin, ymax, xmax] — four numbers. Accept either scale 0-1000 (Gemini standard) or 0.0-1.0 (fractions). ymin=top, xmin=left, ymax=bottom, xmax=right.
- "label": string — use "floor" for floor plans (Grundriss KG/EG/OG, etc.) or "side_view" for elevations/sections (Ansicht Ost/Süd, Schnitt A-A, etc.). You can keep the exact label (e.g. "GRUNDRISS EG") or use "floor"/"side_view".

If the text contains coordinates in pixels or 0-100, convert to either 0-1000 or 0.0-1.0 consistently. If the text was truncated, include ALL items you can identify (e.g. three floor plans: GRUNDRISS KG, EG, OG = three separate "floor" entries). If there is no usable content, return [].

Return ONLY the JSON array, no markdown, no explanation. Start with [ and end with ].

Text from the other AI:
---
{raw_text}
---"""


def _repair_json_with_gemini_second_pass(raw_gemini_text: str) -> list[dict[str, Any]]:
    """La eroare de parsare, al doilea apel Gemini (REST) reconstruiește JSON-ul (array de cutii)."""
    api_key = gemini_api_key()
    if not api_key:
        print("   [Gemini Crop] GEMINI_API_KEY lipsește – nu pot repara JSON.")
        return []
    prompt = PROMPT_REPAIR_JSON.format(raw_text=raw_gemini_text[:25000])
    try:
        result = generate_json(
            api_key,
            [{"text": prompt}],
            system_instruction="Return ONLY a JSON array as specified. No markdown.",
            model=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            temperature=0.0,
            max_output_tokens=8192,
            timeout=120,
        )
    except Exception as e:
        print(f"   [Gemini Crop] Eroare la reparare JSON (Gemini): {e}")
        return []

    if isinstance(result, list):
        items = [x for x in result if isinstance(x, dict)]
        if items:
            print("   [Gemini Crop] Reparare Gemini: array JSON valid.")
        return items
    if isinstance(result, dict) and "items" in result and isinstance(result["items"], list):
        items = [x for x in result["items"] if isinstance(x, dict)]
        if items:
            print("   [Gemini Crop] Reparare Gemini: extras din cheia 'items'.")
        return items
    return []


def get_gemini_boxes_for_page(image_path: Path) -> list[dict[str, Any]]:
    """
    Trimite imaginea la Gemini si primeste lista de cutii in procente (0-1) + label.
    Fiecare element: {"box_2d": [ymin, xmin, ymax, xmax], "label": "floor" | "side_view"}
    """
    client = setup_gemini_client()
    if not client:
        print("⚠️ [Gemini Crop] Gemini nu este disponibil (GEMINI_API_KEY).")
        return []

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ [Gemini Crop] Nu am putut deschide imaginea: {image_path} – {e}")
        return []

    try:
        response = client.generate_content(
            [PROMPT_BOXES, img],
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 8192,
            },
            safety_settings=_GEMINI_SAFETY,
        )
    except Exception as e:
        print(f"⚠️ [Gemini Crop] Eroare la apel Gemini: {e}")
        return []

    raw = _get_response_text(response)
    if not raw:
        print("⚠️ [Gemini Crop] Raspuns gol de la Gemini (sau blocat).")
        if getattr(response, "candidates", None) and response.candidates:
            c0 = response.candidates[0]
            fr = getattr(c0, "finish_reason", None)
            if fr and fr != 1:
                print(f"   finish_reason={fr} (1=STOP, 2=SAFETY, 3=MAX_TOKENS)")
        return []

    raw = raw.strip()
    items, floor_order = _extract_json_object(raw)
    if not items:
        items = _extract_json_array(raw)
        floor_order = []
    if not items:
        preview = (raw[:500] + "…") if len(raw) > 500 else raw
        print("⚠️ [Gemini Crop] Nu am putut parsa JSON din raspuns. Primele caractere:")
        print(f"   {preview!r}")
        print("   [Gemini Crop] Încerc reparare cu ChatGPT...")
        items = _repair_json_with_gemini_second_pass(raw)
        floor_order = []
    if not items:
        print("   [Gemini Crop] Retry: re-apel Gemini pentru aceeași pagină...")
        try:
            response = client.generate_content(
                [PROMPT_BOXES, img],
                generation_config={"temperature": 0.0, "max_output_tokens": 8192},
                safety_settings=_GEMINI_SAFETY,
            )
            raw2 = _get_response_text(response)
            if raw2:
                raw2 = raw2.strip()
                items, floor_order = _extract_json_object(raw2)
                if not items:
                    items = _extract_json_array(raw2)
                    floor_order = []
                if not items:
                    items = _repair_json_with_gemini_second_pass(raw2)
                    floor_order = []
        except Exception as e:
            print(f"   [Gemini Crop] Retry Gemini eroare: {e}")
    if not items:
        return []

    # Debug: ce a returnat Gemini (înainte de validare)
    for i, it in enumerate(items):
        lbl = (it.get("label") or "") if isinstance(it, dict) else "?"
        box = it.get("box_2d") if isinstance(it, dict) else None
        print(f"   [Gemini Crop] Brut {i+1}: label={lbl!r} box_2d={box}")
    print(f"   [Gemini Crop] Total brut de la Gemini: {len(items)} zone")

    def _validate_items(item_list: list) -> list[dict[str, Any]]:
        out = []
        for i, item in enumerate(item_list):
            if not isinstance(item, dict):
                print(f"   [Gemini Crop] Item {i}: skip (nu e dict)")
                continue
            box = item.get("box_2d")
            raw_label = (item.get("label") or "").strip()
            label_lower = raw_label.lower()
            label_norm = label_lower.replace("\u00df", "ss")

            # Excludere explicită pentru planuri de sit care NU trebuie să intre la etaje.
            site_plan_markers = (
                "lageplan", "site_plan", "site plan",
                "situationsplan", "situation plan",
                "abstandsflachenplan", "abstandsflaechenplan", "abstandsflächenplan",
                "abstandsflache", "abstandsflaeche", "abstandsfläche",
            )
            if any(k in label_norm for k in site_plan_markers):
                label = "side_view"
            elif label_lower in ("floor", "side_view"):
                label = label_lower
            elif any(k in label_norm for k in (
                "grundriss", "etaj", "floor", "plan", " eg ", " eg.", " og ", " og.", ".og", "blueprint",
                "parter", "level", "nivel",
                "erdgeschoss", "obergeschoss", "geschoss",
                "keller", "kellergeschoss",
                "dach", "dachgeschoss",
            )) or " kg " in label_norm or " kg." in label_norm or label_norm.startswith("kg "):
                label = "floor"
            else:
                label = "side_view"
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                print(f"   [Gemini Crop] Item {i} label={raw_label!r}: skip (box invalid sau len != 4)")
                continue
            try:
                ymin, xmin, ymax, xmax = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            except (TypeError, ValueError):
                print(f"   [Gemini Crop] Item {i} label={raw_label!r}: skip (box nu sunt numere)")
                continue
            mx = max(ymin, xmin, ymax, xmax)
            if mx > 1.0 and mx <= 100.0:
                ymin, xmin, ymax, xmax = ymin / 100.0, xmin / 100.0, ymax / 100.0, xmax / 100.0
            elif mx > 100.0 and mx <= 1000.0:
                ymin, xmin, ymax, xmax = ymin / 1000.0, xmin / 1000.0, ymax / 1000.0, xmax / 1000.0
            elif mx > 1000.0:
                print(f"   [Gemini Crop] Item {i} label={raw_label!r} box={box}: skip (coordonate >1000)")
                continue
            ymin = max(0.0, min(1.0, ymin))
            xmin = max(0.0, min(1.0, xmin))
            ymax = max(0.0, min(1.0, ymax))
            xmax = max(0.0, min(1.0, xmax))
            if ymax <= ymin or xmax <= xmin:
                print(f"   [Gemini Crop] Item {i} label={raw_label!r}: skip (box invalid: ymax<=ymin sau xmax<=xmin)")
                continue
            out.append({"box_2d": [ymin, xmin, ymax, xmax], "label": label, "raw_label": raw_label})
        return out

    valid = _validate_items(items)
    # Dacă Gemini a returnat zone dar toate au box invalid: retry Gemini o dată, apoi eventual reparare ChatGPT
    if len(items) > 0 and len(valid) == 0:
        print("   [Gemini Crop] Toate zonele au box invalid → retry Gemini pentru aceeași pagină...")
        try:
            response = client.generate_content(
                [PROMPT_BOXES, img],
                generation_config={"temperature": 0.0, "max_output_tokens": 8192},
                safety_settings=_GEMINI_SAFETY,
            )
            raw_retry = _get_response_text(response)
            if raw_retry:
                raw_retry = raw_retry.strip()
                items, floor_order = _extract_json_object(raw_retry)
                if not items:
                    items = _extract_json_array(raw_retry)
                    floor_order = []
                if not items:
                    items = _repair_json_with_gemini_second_pass(raw_retry)
                    floor_order = []
                if items:
                    valid = _validate_items(items)
                    if valid:
                        print(f"   [Gemini Crop] După retry: {len(valid)} zone valide.")
        except Exception as e:
            print(f"   [Gemini Crop] Retry Gemini (box invalid) eroare: {e}")
        if len(valid) == 0 and len(items) > 0:
            print("   [Gemini Crop] Încerc reparare cu ChatGPT (text → JSON)...")
            items_repaired = _repair_json_with_gemini_second_pass(raw)
            if items_repaired:
                items = items_repaired
                valid = _validate_items(items)
                if valid:
                    print(f"   [Gemini Crop] După reparare ChatGPT: {len(valid)} zone valide.")
    # Dacă după validare nu avem niciun floor, facem încă un apel Gemini mai strict.
    n_floor_now = sum(1 for v in valid if v.get("label") == "floor")
    if n_floor_now == 0:
        print("   [Gemini Crop] 0 floor detectate → fallback strict Gemini pentru extragere floor plans...")
        try:
            response_floor = client.generate_content(
                [PROMPT_BOXES_FLOOR_RETRY, img],
                generation_config={"temperature": 0.0, "max_output_tokens": 8192},
                safety_settings=_GEMINI_SAFETY,
            )
            raw_floor = _get_response_text(response_floor)
            if raw_floor:
                raw_floor = raw_floor.strip()
                items_floor, order_floor = _extract_json_object(raw_floor)
                if not items_floor:
                    items_floor = _extract_json_array(raw_floor)
                    order_floor = []
                if not items_floor:
                    items_floor = _repair_json_with_gemini_second_pass(raw_floor)
                    order_floor = []
                if items_floor:
                    valid_floor = _validate_items(items_floor)
                    floor_count_retry = sum(1 for v in valid_floor if v.get("label") == "floor")
                    if floor_count_retry > 0:
                        print(f"   [Gemini Crop] Fallback strict a recuperat {floor_count_retry} floor plan(s).")
                        valid = valid_floor
                        floor_order = order_floor
                    else:
                        print("   [Gemini Crop] Fallback strict încă fără floor plans.")
        except Exception as e:
            print(f"   [Gemini Crop] Fallback strict Gemini eroare: {e}")

    # Ordine etaje pe pagină (de la Gemini): adăugăm position_from_bottom la fiecare plan floor
    floor_indices = [i for i, v in enumerate(valid) if v.get("label") == "floor"]
    n_floor = len(floor_indices)
    if n_floor > 0 and isinstance(floor_order, list) and len(floor_order) == n_floor and set(floor_order) == set(range(n_floor)):
        for k, idx in enumerate(floor_indices):
            valid[idx]["position_from_bottom"] = floor_order.index(k)
        print(f"   [Gemini Crop] floor_order pe pagină: {floor_order} → position_from_bottom setat")
    elif n_floor > 0:
        for k, idx in enumerate(floor_indices):
            valid[idx]["position_from_bottom"] = k
    if items:
        n_floor = sum(1 for v in valid if v["label"] == "floor")
        n_side = len(valid) - n_floor
        print(f"   [Gemini Crop] Parsat: {len(items)} zone → {n_floor} floor, {n_side} side_view (valid total: {len(valid)})")
    return valid


def crop_and_save(
    image_path: Path,
    work_dir: Path,
    boxes: list[dict[str, Any]],
) -> tuple[list[ClassificationResult], list[dict[str, Any]]]:
    """
    Script general: decupează imaginea după coordonatele primite (box_2d în 0.0–1.0) și salvează
    crop-urile. Fiecare element din boxes: {"box_2d": [...], "label": "floor"|"side_view", "raw_label": "..."}.
    - label "floor" → work_dir/classified/blueprints/cluster_1.jpg, cluster_2.jpg, ...
    - label "side_view" → work_dir/classified/side_views/cluster_1.jpg, ...
    Returnează (lista de ClassificationResult, lista de {file, raw_label} pentru floor crops).
    """
    bp_dir = work_dir / STEP_DIRS["classified"]["blueprints"]
    sv_dir = work_dir / STEP_DIRS["classified"]["side_views"]
    bp_dir.mkdir(parents=True, exist_ok=True)
    sv_dir.mkdir(parents=True, exist_ok=True)

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ [Gemini Crop] Eroare la deschidere imagine: {e}")
        return [], []

    width, height = img.size
    results: list[ClassificationResult] = []
    floor_labels: list[dict[str, Any]] = []
    floor_idx = 0
    side_view_idx = 0

    for item in boxes:
        ymin, xmin, ymax, xmax = item["box_2d"]
        label = item["label"]
        raw_label = item.get("raw_label") or item.get("label") or ""
        left = xmin * width
        top = ymin * height
        right = xmax * width
        bottom = ymax * height
        left = max(0, min(left, width - 1))
        right = max(left + 1, min(right, width))
        top = max(0, min(top, height - 1))
        bottom = max(top + 1, min(bottom, height))
        try:
            cropped = img.crop((int(left), int(top), int(right), int(bottom)))
        except Exception as e:
            print(f"⚠️ [Gemini Crop] Crop esuat: {e}")
            continue

        if label == "floor":
            floor_idx += 1
            filename = f"cluster_{floor_idx}.jpg"
            dst = bp_dir / filename
            our_label = "house_blueprint"
            floor_labels.append({
                "file": filename,
                "raw_label": raw_label,
                "position_from_bottom": item.get("position_from_bottom", 0),
            })
        else:
            side_view_idx += 1
            filename = f"cluster_{side_view_idx}.jpg"
            dst = sv_dir / filename
            our_label = "side_view"

        try:
            cropped.save(dst, "JPEG", quality=92)
        except Exception as e:
            print(f"⚠️ [Gemini Crop] Salvare esuata {dst}: {e}")
            continue

        results.append(
            ClassificationResult(
                image_path=dst,
                label=our_label,
                confidence=ConfidenceLevel.HIGH,
                gemini_vote=our_label,
            )
        )
        print(f"   ✅ {our_label}: {filename} (dimensiune: {cropped.size})")

    if floor_labels:
        try:
            labels_path = bp_dir / "crop_labels.json"
            labels_path.write_text(json.dumps(floor_labels, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            print(f"⚠️ [Gemini Crop] Nu am putut scrie crop_labels.json: {e}")

    return results, floor_labels
