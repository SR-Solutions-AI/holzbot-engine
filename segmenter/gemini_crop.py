# file: engine/segmenter/gemini_crop.py
"""
Extragere planuri (etaje) și side view-uri prin Gemini: coordonate 0–1000 → crop → salvare.
Este singurul flux de „segmentare”: pipeline-ul vechi (clusters, classifier) este dezactivat.
Format așteptat: [{"box_2d": [ymin, xmin, ymax, xmax], "label": "..."}, ...] cu box_2d în scară 0–1000
(sau 0–1); la parsare normalizăm tot la 0–1 pentru crop.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from .classifier import ClassificationResult, ConfidenceLevel, setup_gemini_client, setup_openai_client
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

IGNORE: the general site/lot plan (if the whole page is one big site map, still extract any smaller floor-plan or elevation boxes inside it). Exclude 3D color renderings and project data tables in corners. Focus on technical drawings only.

COORDINATE SYSTEM: Use bounding boxes in format [ymin, xmin, ymax, xmax] normalized on a scale of 0 to 1000. So the image width and height each map to 0–1000. Example: left half of image ≈ xmin 0, xmax 500; top 10% ≈ ymin 0, ymax 100.

Return ONLY a JSON array. No markdown, no code block, no text before or after. Each element must have:
- "box_2d": [ymin, xmin, ymax, xmax] — integers or numbers in 0–1000 range.
- "label": string — the exact label you see (e.g. "GRUNDRISS EG", "Ansicht Süd", "Schnitt A-A").

Example format:
[{"box_2d": [30, 15, 550, 485], "label": "GRUNDRISS KG"}, {"box_2d": [30, 510, 340, 970], "label": "Ansicht Süd"}, {"box_2d": [670, 15, 980, 550], "label": "Schnitt A-A"}]

Output ONLY the raw JSON array. Start with [ and end with ]."""


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


def _repair_json_with_chatgpt(raw_gemini_text: str) -> list[dict[str, Any]]:
    """La eroare de parsare, trimite răspunsul Gemini la ChatGPT ca să returneze JSON corect."""
    client = setup_openai_client()
    if not client:
        print("   [Gemini Crop] ChatGPT indisponibil (OPENAI_API_KEY) – nu pot repara JSON.")
        return []
    # Trimitem tot răspunsul (până la 25k caractere) ca ChatGPT să vadă toate zonele (ex. KG, EG, OG)
    prompt = PROMPT_REPAIR_JSON.format(raw_text=raw_gemini_text[:25000])
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"   [Gemini Crop] Eroare la apel ChatGPT pentru reparare: {e}")
        return []
    if not text:
        return []
    items = _extract_json_array(text)
    if items:
        print("   [Gemini Crop] ChatGPT a reparat JSON-ul din răspunsul Gemini.")
    return items


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
                "max_output_tokens": 4096,
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
    items = _extract_json_array(raw)
    if not items:
        preview = (raw[:500] + "…") if len(raw) > 500 else raw
        print("⚠️ [Gemini Crop] Nu am putut parsa JSON din raspuns. Primele caractere:")
        print(f"   {preview!r}")
        print("   [Gemini Crop] Încerc reparare cu ChatGPT...")
        items = _repair_json_with_chatgpt(raw)
        if not items:
            return []

    # Debug: ce a returnat Gemini (înainte de validare)
    for i, it in enumerate(items):
        lbl = (it.get("label") or "") if isinstance(it, dict) else "?"
        box = it.get("box_2d") if isinstance(it, dict) else None
        print(f"   [Gemini Crop] Brut {i+1}: label={lbl!r} box_2d={box}")
    print(f"   [Gemini Crop] Total brut de la Gemini: {len(items)} zone")

    valid = []
    n_floor, n_side = 0, 0
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            print(f"   [Gemini Crop] Item {i}: skip (nu e dict)")
            continue
        box = item.get("box_2d")
        raw_label = (item.get("label") or "").strip()
        label_lower = raw_label.lower()
        # Normalize ß → ss so "geschoss" matches "geschoß"
        label_norm = label_lower.replace("\u00df", "ss")
        # Map to "floor" (blueprint) or "side_view" (elevation/section)
        if label_lower in ("floor", "side_view"):
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
            label = "side_view"  # Ansicht, Schnitt, section, elevation, etc.
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            print(f"   [Gemini Crop] Item {i} label={raw_label!r}: skip (box invalid sau len != 4)")
            continue
        try:
            ymin, xmin, ymax, xmax = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        except (TypeError, ValueError):
            print(f"   [Gemini Crop] Item {i} label={raw_label!r}: skip (box nu sunt numere)")
            continue
        # Normalize to 0-1: accept 0-1000 (Gemini standard), 0-100 (%), or 0-1 (fractions)
        mx = max(ymin, xmin, ymax, xmax)
        if mx > 1.0 and mx <= 100.0:
            ymin, xmin, ymax, xmax = ymin / 100.0, xmin / 100.0, ymax / 100.0, xmax / 100.0
        elif mx > 100.0 and mx <= 1000.0:
            ymin, xmin, ymax, xmax = ymin / 1000.0, xmin / 1000.0, ymax / 1000.0, xmax / 1000.0
        elif mx > 1000.0:
            print(f"   [Gemini Crop] Item {i} label={raw_label!r} box={box}: skip (coordonate >1000, probabil pixeli)")
            continue  # likely pixels; skip
        ymin = max(0.0, min(1.0, ymin))
        xmin = max(0.0, min(1.0, xmin))
        ymax = max(0.0, min(1.0, ymax))
        xmax = max(0.0, min(1.0, xmax))
        if ymax <= ymin or xmax <= xmin:
            print(f"   [Gemini Crop] Item {i} label={raw_label!r}: skip (box invalid: ymax<=ymin sau xmax<=xmin)")
            continue
        valid.append({"box_2d": [ymin, xmin, ymax, xmax], "label": label})
        if label == "floor":
            n_floor += 1
        else:
            n_side += 1
    if items:
        print(f"   [Gemini Crop] Parsat: {len(items)} zone → {n_floor} floor, {n_side} side_view (valid total: {len(valid)})")
    return valid


def crop_and_save(
    image_path: Path,
    work_dir: Path,
    boxes: list[dict[str, Any]],
) -> list[ClassificationResult]:
    """
    Script general: decupează imaginea după coordonatele primite (box_2d în 0.0–1.0) și salvează
    crop-urile. Fiecare element din boxes: {"box_2d": [ymin, xmin, ymax, xmax], "label": "floor"|"side_view"}.
    - label "floor" → work_dir/classified/blueprints/cluster_1.jpg, cluster_2.jpg, ...
    - label "side_view" → work_dir/classified/side_views/cluster_1.jpg, ...
    Returnează lista de ClassificationResult pentru fiecare crop salvat.
    """
    bp_dir = work_dir / STEP_DIRS["classified"]["blueprints"]
    sv_dir = work_dir / STEP_DIRS["classified"]["side_views"]
    bp_dir.mkdir(parents=True, exist_ok=True)
    sv_dir.mkdir(parents=True, exist_ok=True)

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ [Gemini Crop] Eroare la deschidere imagine: {e}")
        return []

    width, height = img.size
    results: list[ClassificationResult] = []
    floor_idx = 0
    side_view_idx = 0

    for item in boxes:
        ymin, xmin, ymax, xmax = item["box_2d"]
        label = item["label"]
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

    return results
