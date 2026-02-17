# ai/json_repair.py
# Repară răspunsuri Gemini invalide/non-JSON folosind GPT ca model secundar.

from __future__ import annotations

import json
import os
from typing import Any, Optional


def repair_json_with_gpt(
    raw_text: str,
    schema_description: str,
    api_key: Optional[str] = None,
) -> Optional[dict]:
    """
    Trimite răspunsul brut (posibil invalid) de la Gemini la GPT și cere
    construirea unui obiect JSON valid.

    Args:
        raw_text: Textul brut returnat de Gemini (poate conține prefixe, markdown, etc.)
        schema_description: Descrierea formatului JSON așteptat (ex: '{"parter": "2_w", "etaj_1": "1_w"}')
        api_key: OPENAI_API_KEY (opțional, citit din env dacă lipsește)

    Returns:
        dict parsabil sau None dacă reparația eșuează
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception:
        return None

    prompt = f"""Another LLM returned the following response, which failed to parse as valid JSON.
Your task: extract or construct a valid JSON object from this response.

Expected format / schema: {schema_description}

Raw LLM response:
---
{raw_text[:4000]}
---

Return ONLY the valid JSON object. No markdown code fences, no preamble like "Here is...", no explanation. Just the raw JSON."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You output only valid JSON. No other text."},
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
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
        return json.loads(text)
    except Exception as e:
        print(f"   [json_repair] GPT repair failed: {e}", flush=True)
        return None
