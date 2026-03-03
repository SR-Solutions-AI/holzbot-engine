# floor_classifier/floor_order_gemini.py
# Ordinea etajelor: la segmentare Gemini returnează floor_order pe pagină (position_from_bottom).
# Aici facem merge la nivel global din etichete + position_from_bottom (fără al doilea apel Gemini).

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List

PROMPT_FLOOR_ORDER = """You are an expert in architectural floor plans and building levels.

You are given a list of floor plan LABELS (text written on the drawings), one per floor. Your task is to return the correct stacking order from BOTTOM to TOP (index 0 = lowest level, last index = top).

Rules:
- BASEMENT (Keller / KG / Untergeschoss / cellar) is ALWAYS the lowest (index 0).
- GROUND FLOOR (Parter / EG / Erdgeschoss) is always the lowest above basement (index 0 if no basement, else index 1).
- Other floors (OG, 1. OG, 2. OG, Dachgeschoss, etc.) go above ground floor; order them logically (1. OG, 2. OG, etc.).

Input: a JSON array of objects with "index" (0-based position in the list) and "label" (the exact text from the drawing).
Output: a JSON array of indices in order from BOTTOM to TOP. Example: [2, 0, 1] means: the plan at index 2 is basement (bottom), then index 0, then index 1 (top).

Reply with ONLY the JSON array, no other text. Example: [2, 0, 1]"""


def _floor_type_rank(label: str) -> int:
    """Rang pentru sortare globală: 0=beci, 1=parter, 2+=etaj, 99=unknown."""
    if not label or not isinstance(label, str):
        return 99
    s = label.strip().upper().replace("\u00df", "SS")
    if "KG" in s or "KELLER" in s or "KELLERGESCHOSS" in s or "UNTERGESCHOSS" in s or "CELLAR" in s:
        return 0
    if "EG" in s or "PARTER" in s or "ERDGESCHOSS" in s or "GROUND" in s:
        return 1
    if "2. OG" in s or "2.OG" in s:
        return 4
    if "1. OG" in s or "1.OG" in s:
        return 3
    if "OG" in s or "OBERGESCHOSS" in s or "DACHGESCHOSS" in s or "DACH" in s:
        return 2
    return 99


def _load_crop_label_for_plan(image_path: Path) -> tuple[str, int]:
    """Încarcă raw_label și position_from_bottom din crop_labels.json. Returnează (raw_label, position_from_bottom)."""
    labels_file = image_path.parent / "crop_labels.json"
    if not labels_file.exists():
        return "", 0
    try:
        data = json.loads(labels_file.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return "", 0
        name = image_path.name
        for item in data:
            if isinstance(item, dict) and item.get("file") == name:
                raw = str(item.get("raw_label") or "")
                pos = int(item.get("position_from_bottom", 0))
                return raw, pos
    except Exception:
        pass
    return "", 0


def _load_raw_label_for_plan(image_path: Path) -> str:
    """Încarcă raw_label pentru un plan din crop_labels.json din același folder cu imaginea."""
    labels_file = image_path.parent / "crop_labels.json"
    if not labels_file.exists():
        return ""
    try:
        data = json.loads(labels_file.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return ""
        name = image_path.name
        for item in data:
            if isinstance(item, dict) and item.get("file") == name:
                return str(item.get("raw_label") or "")
    except Exception:
        pass
    return ""


def _merge_order_from_labels_and_position(house_plans: List[Any]) -> List[int]:
    """Ordine globală din etichete + position_from_bottom. Sort: (floor_type_rank, work_dir, position_from_bottom)."""
    rows: List[tuple[int, str, str, int]] = []
    for i, plan in enumerate(house_plans):
        img_path = getattr(plan, "image_path", None)
        if img_path is None:
            rows.append((i, "", "", 0))
            continue
        p = Path(img_path)
        raw_label, position_from_bottom = _load_crop_label_for_plan(p)
        work_dir = str(p.parent.parent.parent) if len(p.parts) >= 3 else ""
        rows.append((i, raw_label, work_dir, position_from_bottom))
    rows.sort(key=lambda r: (r[2], r[3], r[0]))
    rows.sort(key=lambda r: (_floor_type_rank(r[1]), r[2], r[3]))
    return [r[0] for r in rows]


def run_floor_order_from_gemini(house_plans: List[Any]) -> List[int] | None:
    """
    Ordine etaje de jos în sus. Preferă merge din datele de la segmentare (position_from_bottom + labels).
    Fallback: apel Gemini cu doar label-urile.
    """
    if not house_plans:
        return None
    # Folosim mereu merge din etichete + position_from_bottom (date de la segmentare)
    order = _merge_order_from_labels_and_position(house_plans)
    has_any_label = any(
        _load_crop_label_for_plan(Path(getattr(p, "image_path", "")))[0]
        for p in house_plans
        if getattr(p, "image_path", None)
    )
    if has_any_label:
        print(f"   [FloorOrder] Ordine din segmentare (etichete + position_from_bottom): {order}")
        return order
    return _run_floor_order_gemini_fallback(house_plans)


def _run_floor_order_gemini_fallback(house_plans: List[Any]) -> List[int] | None:
    """Apel Gemini cu (index, label) când nu avem suficiente date din segmentare."""
    from segmenter.classifier import setup_gemini_client

    if not house_plans:
        return None

    labels: List[dict] = []
    for i, plan in enumerate(house_plans):
        img_path = getattr(plan, "image_path", None)
        if img_path is None:
            labels.append({"index": i, "label": ""})
            continue
        raw, _ = _load_crop_label_for_plan(Path(img_path))
        if not raw:
            raw = _load_raw_label_for_plan(Path(img_path))
        labels.append({"index": i, "label": raw or f"Plan {i + 1}"})
        print(f"   [FloorOrder] Plan {i}: {Path(img_path).name} → label={raw!r}")

    if all(not lb.get("label", "").strip() for lb in labels):
        print("   ⚠️ [FloorOrder] Niciun raw_label disponibil – skip Gemini floor order.")
        return None

    client = setup_gemini_client()
    if client is None:
        print("   ⚠️ [FloorOrder] Gemini indisponibil.")
        return None

    try:
        prompt = PROMPT_FLOOR_ORDER + "\n\nInput labels:\n" + json.dumps(labels, ensure_ascii=False)
        response = client.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 256,
            },
        )
        if not response or not response.parts:
            return None
        text = response.text.strip()
        # Extrage array JSON (poate fi înconjurat de markdown)
        m = re.search(r"\[[\d\s,]+\]", text)
        if not m:
            print(f"   ⚠️ [FloorOrder] Răspuns invalid (nu am găsit array): {text[:200]}")
            return None
        order = json.loads(m.group(0))
        if not isinstance(order, list) or len(order) != len(house_plans):
            print(f"   ⚠️ [FloorOrder] Ordine invalidă (lungime {len(order) if isinstance(order, list) else 0} != {len(house_plans)}).")
            return None
        if set(order) != set(range(len(house_plans))):
            print(f"   ⚠️ [FloorOrder] Ordine invalidă (nu e permutare 0..N-1).")
            return None
        print(f"   ✅ [FloorOrder] Ordine de jos în sus (Gemini fallback): {order}")
        return order
    except Exception as e:
        print(f"   ⚠️ [FloorOrder] Eroare: {e}")
        return None
