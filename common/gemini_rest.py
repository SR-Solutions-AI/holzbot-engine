# common/gemini_rest.py
# Apeluri REST către Google Generative Language API (v1beta), același stil ca cubicasa_detector/scale_detection.call_gemini.
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import requests

# Aliniat cu holzbot-engine/cubicasa_detector/config.py — poate fi suprascris din .env
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
# Pentru sarcini ușoare (fost gpt-4o-mini): reparare JSON text-only, fără imagini
DEFAULT_GEMINI_MODEL_FAST = os.environ.get("GEMINI_MODEL_FAST", "gemini-2.0-flash")


def resolve_vision_model(model: str | None) -> str:
    return (model or "").strip() or DEFAULT_GEMINI_MODEL


def resolve_text_model(model: str | None) -> str:
    return (model or "").strip() or DEFAULT_GEMINI_MODEL_FAST


def gemini_api_key() -> str | None:
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    return key or None


def _mime_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")


def image_part_from_path(image_path: str | Path) -> dict[str, Any]:
    p = Path(image_path)
    with open(p, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return {"inline_data": {"mime_type": _mime_for_path(p), "data": data}}


def image_part_from_bytes(data: bytes, mime_type: str = "image/png") -> dict[str, Any]:
    return {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(data).decode("utf-8")}}


def _extract_text_from_response(result: dict[str, Any]) -> str | None:
    cands = result.get("candidates")
    if not cands:
        return None
    parts = (cands[0].get("content") or {}).get("parts") or []
    if not parts:
        return None
    return (parts[0].get("text") or "").strip() or None


def generate_content_raw(
    api_key: str,
    parts: list[dict[str, Any]],
    *,
    system_instruction: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int = 8192,
    response_mime_type: str | None = None,
    timeout: int = 120,
) -> str | None:
    """
    POST generateContent; returnează textul brut din primul candidate sau None.
    """
    m = resolve_vision_model(model)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
    gen_cfg: dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
    }
    if response_mime_type:
        gen_cfg["responseMimeType"] = response_mime_type
    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": gen_cfg,
    }
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    try:
        resp = requests.post(url, json=body, headers={"Content-Type": "application/json"}, timeout=timeout)
        if resp.status_code != 200:
            try:
                err = resp.json()
            except Exception:
                err = resp.text[:500]
            print(f"⚠️  Gemini REST HTTP {resp.status_code}: {err}", flush=True)
            return None
        data = resp.json()
        return _extract_text_from_response(data)
    except Exception as e:
        print(f"⚠️  Gemini REST error: {e}", flush=True)
        return None


def parse_json_loose(text: str | None) -> Any | None:
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    start = s.find("{")
    start_arr = s.find("[")
    if start_arr >= 0 and (start < 0 or start_arr < start):
        start = start_arr
    if start < 0:
        return None
    end_br = s.rfind("}") + 1
    end_sq = s.rfind("]") + 1
    end = max(end_br, end_sq)
    if end <= start:
        return None
    chunk = s[start:end]
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        return None


def generate_json(
    api_key: str,
    parts: list[dict[str, Any]],
    *,
    system_instruction: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int = 8192,
    timeout: int = 120,
) -> Any | None:
    """responseMimeType=application/json; parsează JSON sau None."""
    raw = generate_content_raw(
        api_key,
        parts,
        system_instruction=system_instruction,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        timeout=timeout,
    )
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return parse_json_loose(raw)


def generate_text(
    api_key: str,
    parts: list[dict[str, Any]],
    *,
    system_instruction: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    timeout: int = 120,
) -> str | None:
    return generate_content_raw(
        api_key,
        parts,
        system_instruction=system_instruction,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type=None,
        timeout=timeout,
    )


def first_integer_in_text(text: str | None) -> int | None:
    if not text:
        return None
    m = re.search(r"\d+", text.strip())
    return int(m.group()) if m else None
