# file: engine/new/runner/segmenter/classifier.py
# ------------------------------------------------------------
# Clasificare planuri cu OpenAI (ChatGPT Vision) + fallback local.
# Refolosește logica ta originală din plan_segmentation.py.
# ------------------------------------------------------------

from __future__ import annotations

import os
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageFilter, ImageFile

from .common import STEP_DIRS, get_output_dir, debug_print, safe_imread

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# tipuri de label permise
LabelType = Literal["house_blueprint", "site_blueprint", "side_view", "text_area"]

OPENAI_PROMPT = (
    "You are an extremely strict architectural image classifier.\n"
    "Return exactly ONE lowercase label from this set: house_blueprint | site_blueprint | side_view | text_area.\n\n"
    "Definitions:\n"
    "- house_blueprint = HOUSE FLOOR PLAN (top-down 2D) with interior walls forming rooms, door/window symbols, "
    "dimension lines. Not an elevation, not a 3D render, not a site plan.\n"
    "- site_blueprint  = SITE/LOT plan: plot/property boundaries, setbacks, streets/road names, north arrow/compass rose, "
    "driveway/landscaping, terrain contours.\n"
    "- side_view       = exterior façade/elevation or 3D perspective view.\n"
    "- text_area       = page or crop dominated by paragraphs, tables, legends or mostly text.\n"
    "Output: house_blueprint OR site_blueprint OR side_view OR text_area."
)


@dataclass
class ClassificationResult:
    """
    Rezultatul clasificării pentru UN plan.
    """
    image_path: Path
    label: LabelType


# ==============================
# Helpers pentru OpenAI Vision
# ==============================
def _prep_for_vlm(img_path: str | Path, min_long_edge: int = 1280) -> Image.Image:
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    long_edge = max(w, h)
    if long_edge < min_long_edge:
        scale = min_long_edge / float(long_edge)
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    im = im.filter(ImageFilter.UnsharpMask(radius=0.8, percent=110, threshold=2))
    return im


def _pil_to_base64(pil_img: Image.Image) -> str:
    import io
    import base64

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_openai_text(resp_obj) -> str:
    # helper-ul tău original – îl păstrăm ca fallback generic
    try:
        if hasattr(resp_obj, "output_text"):
            return resp_obj.output_text
        if hasattr(resp_obj, "output") and hasattr(resp_obj.output, "text"):
            return resp_obj.output.text
    except Exception:
        pass
    try:
        if resp_obj and getattr(resp_obj, "choices", None):
            choice0 = resp_obj.choices[0]
            # support atât dict cât și obiect
            msg = getattr(choice0, "message", None)
            if isinstance(msg, dict):
                return msg.get("content", "")
            if msg is not None and hasattr(msg, "content"):
                return msg.content
    except Exception:
        pass
    return ""


# ==============================
# Heuristici locale (fallback)
# ==============================
def _count_rect_rooms_and_lines(img_gray: np.ndarray) -> tuple[int, float, float, int]:
    g = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Canny(g, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=40, maxLineGap=8)
    angs: list[float] = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            ang = abs(math.degrees(math.atan2((y2 - y1), (x2 - x1)))) % 180.0
            angs.append(ang)

    angs_arr = np.array(angs) if angs else np.array([])

    def is_ortho(a: float) -> bool:
        return (min(abs(a - 0), abs(a - 180)) < 8) or (abs(a - 90) < 8)

    def is_diag(a: float) -> bool:
        return (30 <= a <= 60) or (120 <= a <= 150)

    if angs_arr.size > 0:
        ortho = np.mean([1.0 if is_ortho(a) else 0.0 for a in angs_arr])
        diag = np.mean([1.0 if is_diag(a) else 0.0 for a in angs_arr])
    else:
        ortho, diag = 0.0, 0.0

    binv = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 21, 10
    )
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binv, 8)
    cc_small = 0
    for x in stats[1:]:
        a = x[cv2.CC_STAT_AREA]
        if 15 <= a <= 800:
            cc_small += 1

    cnts, _ = cv2.findContours(
        cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    rooms_like = 0
    h, w = img_gray.shape[:2]
    area_img = h * w
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.0002 * area_img or a > 0.25 * area_img:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            rooms_like += 1

    return rooms_like, float(ortho), float(diag), int(cc_small)


def local_classify(img_path: str | Path) -> LabelType:
    im = safe_imread(img_path)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rooms_like, ortho_ratio, diag_ratio, cc_small = _count_rect_rooms_and_lines(g)

    debug_print(
        f"🧪 LOCAL {Path(img_path).name}: "
        f"rooms={rooms_like} ortho={ortho_ratio:.2f} "
        f"diag={diag_ratio:.2f} textCC={cc_small}"
    )

    # logică din scriptul tău
    if rooms_like <= 1 and cc_small >= 1800 and ortho_ratio <= 0.55:
        return "text_area"
    if rooms_like >= 5 and ortho_ratio >= 0.55 and diag_ratio <= 0.25:
        return "house_blueprint"
    if rooms_like <= 2 and diag_ratio >= 0.35:
        return "side_view"
    if rooms_like <= 2 and (cc_small >= 350 or ortho_ratio <= 0.45):
        return "site_blueprint"
    return "side_view"


# ==============================
# Clasificare cu OpenAI
# ==============================

ALLOWED: set[str] = {"house_blueprint", "site_blueprint", "side_view", "text_area"}


def _parse_label(txt: str) -> str:
    t = (txt or "").strip().lower()
    if t in ALLOWED:
        return t
    for w in ALLOWED:
        if w in t:
            return w
    return ""


def _build_openai_client():
    """
    Creează clientul OpenAI, dacă e posibil. Altfel întoarce (None, False)
    """
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ Eroare: Lipsă OPENAI_API_KEY în .env sau env vars. Folosesc DOAR fallback local.")
        return None, False

    try:
        from openai import OpenAI

        client = OpenAI(api_key=openai_key)
        use_responses_api = hasattr(client, "responses")
        return client, use_responses_api
    except Exception as e:
        print(f"⚠️ OpenAI SDK indisponibil ({e}). Folosesc DOAR fallback local.")
        return None, False


def _classify_one_with_openai(client, use_responses_api: bool, img_path: str | Path) -> str:
    """
    Returnează label de la OpenAI sau "" dacă nu reușește.
    """
    if client is None:
        return ""

    try:
        pil_img = _prep_for_vlm(img_path)
        b64 = _pil_to_base64(pil_img)

        if use_responses_api:
            # Nou API Responses
            resp = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": OPENAI_PROMPT},
                            {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                        ],
                    }
                ],
                temperature=0.0,
                max_output_tokens=64,
            )
            out = _extract_openai_text(resp)
            label = _parse_label(out)
            if label:
                return label

            # fallback prompt mai scurt
            resp2 = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Return one label: house_blueprint | site_blueprint | side_view | text_area",
                            },
                            {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                        ],
                    }
                ],
                temperature=0.0,
                max_output_tokens=64,
            )
            return _parse_label(_extract_openai_text(resp2))

        else:
            # vechiul chat.completions
            msg = [
                {"role": "system", "content": "You are a careful vision classifier."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OPENAI_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                },
            ]
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=msg,
                temperature=0.0,
                max_tokens=64,
            )
            out = _extract_openai_text(resp)
            label = _parse_label(out)
            if label:
                return label

            msg[1]["content"][0]["text"] = (
                "Return one label: house_blueprint | site_blueprint | side_view | text_area"
            )
            resp2 = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=msg,
                temperature=0.0,
                max_tokens=64,
            )
            return _parse_label(_extract_openai_text(resp2))

    except Exception as e:
        debug_print(f"⚠️ OpenAI exception {Path(img_path).name}: {e}")
        return ""


# ==============================
# Funcția principală de clasificare
# ==============================

def classify_segmented_plans(segmentation_out: str | Path) -> list[ClassificationResult]:
    """
    Clasifică planurile decupate de segmenter (clusters/plan_crops) în:
      - house_blueprint
      - site_blueprint
      - side_view
      - text_area

    Folosește:
      - OpenAI Vision (gpt-4o-mini) dacă există OPENAI_API_KEY
      - fallback local_classify(...) + post-validare pe folderul 'blueprints'

    segmentation_out:
      - trebuie să fie același output_dir pe care l-ai dat lui segment_document(...)
        (de ex: job_root / 'segmentation').

    Returnează lista de ClassificationResult.
    """
    segmentation_out = Path(segmentation_out).resolve()

    # nu resetăm nimic aici, doar spunem segmenter-ului că OUTPUT_DIR este segmentation_out
    from .common import set_output_dir  # evităm import circular în top

    set_output_dir(segmentation_out)

    # directoare
    crops_dir = segmentation_out / STEP_DIRS["clusters"]["crops"]

    bp_dir = segmentation_out / STEP_DIRS["classified"]["blueprints"]
    sp_dir = segmentation_out / STEP_DIRS["classified"]["siteplan"]
    sv_dir = segmentation_out / STEP_DIRS["classified"]["side_views"]
    tx_dir = segmentation_out / STEP_DIRS["classified"]["text"]

    for d in (bp_dir, sp_dir, sv_dir, tx_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not crops_dir.is_dir():
        print(f"ℹ️ Nu există crops_dir cu planuri: {crops_dir}")
        return []

    client, use_responses_api = _build_openai_client()

    results: list[ClassificationResult] = []

    print("\n[STEP 8] Clasificare cu OpenAI (gpt-4o-mini) + fallback local + post-validare...")

    for img_file in sorted(crops_dir.iterdir()):
        if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        img_path = img_file
        # 1) încercăm cu OpenAI
        label = _classify_one_with_openai(client, use_responses_api, img_path)
        # 2) dacă nu reușește, fallback local
        if not label:
            label = local_classify(img_path)

        # mapare în foldere
        if label == "house_blueprint":
            dst = bp_dir / img_file.name
            shutil.copy(str(img_path), str(dst))
            print(f"🏗 house_blueprint: {img_file.name}")
        elif label == "site_blueprint":
            dst = sp_dir / img_file.name
            shutil.copy(str(img_path), str(dst))
            print(f"🗺 site_blueprint: {img_file.name}")
        elif label == "side_view":
            dst = sv_dir / img_file.name
            shutil.copy(str(img_path), str(dst))
            print(f"🏠 side_view: {img_file.name}")
        elif label == "text_area":
            dst = tx_dir / img_file.name
            shutil.copy(str(img_path), str(dst))
            print(f"📝 text_area: {img_file.name}")
        else:
            dst = tx_dir / img_file.name
            shutil.copy(str(img_path), str(dst))
            print(f"📝 text_area*(fallback): {img_file.name}")
            label = "text_area"

        results.append(ClassificationResult(image_path=dst, label=label))  # type: ignore[arg-type]

    print("✅ Clasificare inițială finalizată!\n")

    # ==========================
    # STEP 8B – Post-validare
    # ==========================
    print("[STEP 8B] Post-validare folder 'blueprints'...")

    moved = 0
    for img_file in sorted(bp_dir.iterdir()):
        if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        img_path = img_file
        lbl = local_classify(img_path)

        if lbl in ("side_view", "site_blueprint", "text_area"):
            if lbl == "side_view":
                dst_dir = sv_dir
            elif lbl == "site_blueprint":
                dst_dir = sp_dir
            else:
                dst_dir = tx_dir

            dst = dst_dir / img_file.name
            shutil.move(str(img_path), str(dst))
            moved += 1
            print(f"↪️  mutat din blueprints în {dst_dir.name}: {img_file.name}")

            # actualizăm și rezultatele pentru imaginea asta
            for r in results:
                if r.image_path == img_path:
                    r.image_path = dst
                    r.label = lbl  # type: ignore[assignment]
                    break

    print(f"✅ Post-validare terminată. Mutate din blueprints: {moved}\n")

    return results
