# common/json_repair.py
# Folosește GPT pentru a repara răspunsuri Gemini invalide (non-JSON) în JSON corect.

from __future__ import annotations

import json
import os
from typing import Any, Optional


PROMPT_JSON_REPAIR = """Another AI (Gemini) returned the following text instead of valid JSON. 
Your task: extract or reconstruct a valid JSON object from it.

Raw response from Gemini:
---
{raw_text}
---

Expected format hint: {schema_hint}

Return ONLY the JSON object. No markdown, no preamble, no explanation. Just the raw JSON."""


def repair_json_with_gpt(
    raw_text: str,
    schema_hint: str = "A JSON object with relevant keys (e.g. room_name, area_m2, parter, etaj_1, total_sum_m2)",
) -> Optional[dict[str, Any]]:
    """
    Trimite răspunsul raw de la Gemini la GPT și obține un JSON valid.
    
    Args:
        raw_text: Textul raw returnat de Gemini (posibil non-JSON)
        schema_hint: Descriere scurtă a formatului JSON așteptat
    
    Returns:
        Dict parsabil sau None dacă reparația eșuează
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception:
        return None

    prompt = PROMPT_JSON_REPAIR.format(
        raw_text=raw_text[:4000],  # limit pentru a evita token overflow
        schema_hint=schema_hint,
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract or fix JSON. Return ONLY valid JSON, no other text."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.0,
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            return None
        # Elimină markdown code fences dacă există
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].lstrip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        # Extrage { ... }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
        return json.loads(text)
    except Exception as e:
        print(f"   [json_repair] GPT repair failed: {e}", flush=True)
        return None
