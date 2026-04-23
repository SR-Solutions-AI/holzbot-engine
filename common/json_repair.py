# common/json_repair.py
# Repară răspunsuri Gemini invalide (non-JSON) folosind un al doilea apel Gemini (model rapid, text-only).

from __future__ import annotations

import os
from typing import Any, Optional

from common.gemini_rest import DEFAULT_GEMINI_MODEL_FAST, gemini_api_key, generate_json


PROMPT_JSON_REPAIR = """Another model returned the following text instead of valid JSON.
Your task: extract or reconstruct a valid JSON object from it.

Raw response:
---
{raw_text}
---

Expected format hint: {schema_hint}

Return ONLY the JSON object. No markdown, no preamble, no explanation. Just the raw JSON."""


def repair_json_with_gemini(
    raw_text: str,
    schema_hint: str = "A JSON object with relevant keys (e.g. room_name, area_m2, parter, etaj_1, total_sum_m2)",
) -> Optional[dict[str, Any]]:
    """
    Trimite textul corupt la Gemini (model rapid, fără imagini) și cere JSON valid.
    """
    api_key = gemini_api_key()
    if not api_key:
        return None
    prompt = PROMPT_JSON_REPAIR.format(
        raw_text=(raw_text or "")[:8000],
        schema_hint=schema_hint,
    )
    try:
        parsed = generate_json(
            api_key,
            [{"text": prompt}],
            system_instruction="You extract or fix JSON. Return ONLY a valid JSON object, no other text.",
            model=DEFAULT_GEMINI_MODEL_FAST,
            temperature=0.0,
            max_output_tokens=2048,
            timeout=60,
        )
        if isinstance(parsed, dict):
            return parsed
        return None
    except Exception as e:
        print(f"   [json_repair] Gemini repair failed: {e}", flush=True)
        return None


# Alias pentru cod existent (scale_detection, roof_type_classifier, etc.)
repair_json_with_gpt = repair_json_with_gemini
