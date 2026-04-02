# holzbot-engine/validator.py
# Validare plan(uri): cel puțin un Grundriss (floor plan); Ansicht/Schnitt e opțional (informativ în JSON).
import sys
import json
import os
import requests
import fitz  # PyMuPDF
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI

# Config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MAX_PDF_PAGES = 20  # limită pentru numărul de imagini trimise într-un singur request

SYSTEM_PROMPT = """You are an architect assistant. You will receive one or more images (e.g. pages of a PDF or a single image).
Your task: determine across ALL provided images:
1. Is there at least one image that shows a FLOOR PLAN (Grundriss) – room labels, walls, clear top-down structure?
2. Optionally: is there a SIDE VIEW / ELEVATION / SECTION (Ansicht, Schnitt, Fassade)? This is informational only.

Return JSON with exactly these keys:
- "has_floor_plan": true if at least one image is a floor plan, else false.
- "has_side_view": true if at least one image is a side view / elevation / section, else false.
- "valid": true if has_floor_plan is true. Side views are NOT required for validity.
- "reason": short explanation in the same language as the document (e.g. if valid=false: no recognizable floor plan).

If the content looks like a photo, generic sketch without structure, or unrelated, set has_floor_plan false and valid=false.
Return ONLY this JSON object, no markdown, no code block."""


def _collect_images_from_pdf(file_bytes):
    """Extrage toate paginile PDF ca liste de (BytesIO png, width, height)."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    n = min(len(doc), MAX_PDF_PAGES)
    out = []
    for i in range(n):
        page = doc.load_page(i)
        pix = page.get_pixmap()
        buf = BytesIO(pix.tobytes("png"))
        out.append((buf, pix.width, pix.height))
    doc.close()
    return out


def _run_vision(client, content_list, num_images):
    """Apelează Vision cu content_list (text + N image_url). content_list = [text_block, img1, img2, ...]."""
    chat_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content_list},
        ],
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    result_text = chat_response.choices[0].message.content
    data = json.loads(result_text)
    hp = data.get("has_floor_plan", False)
    hv = data.get("has_side_view", False)
    if "valid" not in data:
        data["valid"] = bool(hp)
    if "reason" not in data:
        data["reason"] = "OK" if data["valid"] else ("No floor plan (Grundriss) found in the provided pages." if num_images > 1 else "No floor plan (Grundriss) found.")
    return data


def validate_plan(file_url):
    """
    Validează un singur document (URL): PDF (toate paginile, până la MAX_PDF_PAGES) sau o imagine.
    Cerință: în cel puțin una dintre pagini/imagine să existe un floor plan (Grundriss). Ansicht/Schnitt nu e obligatoriu.
    Returnează: { "valid": bool, "reason": str, "has_floor_plan": bool, "has_side_view": bool }
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        response = requests.get(file_url)
        response.raise_for_status()
        file_bytes = response.content
        content_type = response.headers.get("Content-Type", "").lower()

        images_for_api = []

        if "pdf" in content_type or file_url.lower().endswith(".pdf"):
            try:
                images_for_api = _collect_images_from_pdf(file_bytes)
            except Exception as e:
                return {"valid": False, "reason": f"PDF Corrupt: {str(e)}", "has_floor_plan": False, "has_side_view": False}
        else:
            try:
                img = Image.open(BytesIO(file_bytes))
                buf = BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                images_for_api = [(buf, img.width, img.height)]
            except Exception as e:
                return {"valid": False, "reason": "Invalid Image Format", "has_floor_plan": False, "has_side_view": False}

        if not images_for_api:
            return {"valid": False, "reason": "No pages in PDF", "has_floor_plan": False, "has_side_view": False}

        text = (
            f"Validate these {len(images_for_api)} image(s). "
            "Across all of them, there must be at least one FLOOR PLAN (Grundriss). Side views are optional."
        )
        content_list = [{"type": "text", "text": text}]
        for buf, _w, _h in images_for_api:
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("utf-8")
            data_url = f"data:image/png;base64,{b64}"
            content_list.append({"type": "image_url", "image_url": {"url": data_url, "detail": "high"}})

        result = _run_vision(client, content_list, len(images_for_api))
        result.setdefault("has_floor_plan", False)
        result.setdefault("has_side_view", False)
        result["valid"] = bool(result.get("has_floor_plan"))
        return result

    except Exception as e:
        return {"valid": False, "reason": f"System Error: {str(e)}", "has_floor_plan": False, "has_side_view": False}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"valid": False, "reason": "No URL provided", "has_floor_plan": False, "has_side_view": False}))
        sys.exit(1)

    url = sys.argv[1]
    result = validate_plan(url)
    print(json.dumps(result))
