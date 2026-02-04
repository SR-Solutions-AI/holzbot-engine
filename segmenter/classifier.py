# file: engine/runner/segmenter/classifier.py
# ------------------------------------------------------------
# Clasificare ULTRA-ROBUSTĂ cu dual-model validation
# GPT-4o + Gemini (Auto-Fallback) + Fast Failover Logic
# + Detectare și Eliminare Duplicate Clusters (Simple AI Comparison)
# + Verificare Măsurători Camere pentru Confirmare Duplicate
# ------------------------------------------------------------

from __future__ import annotations

import os
import math
import shutil
import io
import base64
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Any
from enum import Enum
from collections import defaultdict, Counter

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageFilter, ImageFile

from .common import STEP_DIRS, get_output_dir, debug_print, safe_imread

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Tipuri de label
LabelType = Literal["house_blueprint", "site_blueprint", "side_view", "text_area"]

ALLOWED_LABELS: set[str] = {"house_blueprint", "site_blueprint", "side_view", "text_area"}


class ConfidenceLevel(Enum):
    """Nivel de încredere în clasificare"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ClassificationResult:
    """Rezultat clasificare cu nivel de încredere"""
    image_path: Path
    label: LabelType
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH
    gpt_vote: Optional[str] = None
    gemini_vote: Optional[str] = None
    arbitration_rounds: int = 0


@dataclass
class DuplicateGroup:
    """Grup de clustere duplicate (păstrăm cel mai mare)"""
    keep_cluster: Path
    remove_clusters: list[Path]
    ai_response: str


# ============================================================================
# PROMPTURI PENTRU FIECARE RUNDĂ DE ARBITRAJ
# ============================================================================

PROMPT_ROUND_1_BASE = """You are an EXPERT architectural drawing classifier.

CRITICAL: Return EXACTLY ONE label from this list: house_blueprint | site_blueprint | side_view | text_area

1. house_blueprint = TOP-DOWN FLOOR PLAN of interior. Must show interior walls, rooms, doors.
2. site_blueprint = SITE/LOT PLAN. Shows property boundaries, streets, context. NO interior rooms.
3. side_view = EXTERIOR ELEVATION or 3D PERSPECTIVE. Façade view.
4. text_area = PAGE DOMINATED BY TEXT (>40%). Specifications, legends.

Return ONLY the label."""


PROMPT_ROUND_2_EXPLICIT = """You are a FORENSIC architectural document analyzer.
Return EXACTLY ONE: house_blueprint | site_blueprint | side_view | text_area

DISTINCTIONS:
- Interior walls visible? -> house_blueprint
- Only property lines/streets? -> site_blueprint
- Looking at the building from the side? -> side_view
- Mostly text? -> text_area

Output format: [label_only]"""


PROMPT_ROUND_3_EXTREME = """FINAL DECISION.
Choose: house_blueprint | site_blueprint | side_view | text_area

- house_blueprint: I see rooms (kitchen, bed, bath) from above.
- site_blueprint: I see the lot, roads, and building outline (no rooms).
- side_view: I see the roof pitch and windows from the outside.
- text_area: I see paragraphs of text.

Return ONLY the label."""


# ============================================================================
# HELPERS PENTRU IMAGINI
# ============================================================================

def prep_for_vlm(img_path: str | Path, min_long_edge: int = 1280) -> Image.Image:
    """Pregătește imaginea pentru VLM: resize + sharpen"""
    try:
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        long_edge = max(w, h)
        
        if long_edge < min_long_edge:
            scale = min_long_edge / float(long_edge)
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        
        im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
        return im
    except Exception as e:
        debug_print(f"⚠️ Eroare la procesarea imaginii {img_path}: {e}")
        return Image.new('RGB', (500, 500), color='black')


def pil_to_base64(pil_img: Image.Image) -> str:
    """Convertește PIL Image la base64"""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def parse_label(text: str) -> Optional[str]:
    """Extrage label valid din răspunsul modelului"""
    if not text:
        return None
    
    text_clean = text.strip().lower()
    text_clean = text_clean.replace("```", "").replace("[", "").replace("]", "").strip()
    
    if text_clean in ALLOWED_LABELS:
        return text_clean
    
    found_labels = []
    for label in ALLOWED_LABELS:
        if label in text_clean:
            found_labels.append(label)
    
    if len(found_labels) == 1:
        return found_labels[0]
        
    return None


# ============================================================================
# CLIENT SETUP
# ============================================================================

def setup_openai_client():
    """Inițializează clientul OpenAI"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return None
    
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception as e:
        debug_print(f"⚠️  Eroare OpenAI init: {e}")
        return None


def setup_gemini_client():
    """Inițializează clientul Gemini cu FALLBACK AUTOMAT de modele"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return None
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
        
        models_to_try = [
            'gemini-2.5-flash',
            'gemini-2.0-flash',
            'gemini-1.5-flash'
        ]
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name, safety_settings=safety_settings)
                debug_print(f"✅ Gemini: Model selectat '{model_name}'")
                return model
            except Exception:
                continue
                
        debug_print("❌ Gemini: Niciun model flash disponibil.")
        return None

    except Exception as e:
        debug_print(f"⚠️  Eroare Gemini init: {e}")
        return None


# ============================================================================
# CLASIFICARE CU FIECARE MODEL
# ============================================================================

def classify_with_gpt(client, img_path: Path, prompt: str) -> Optional[str]:
    """Clasificare cu GPT-4o"""
    if client is None:
        return None
    
    try:
        pil_img = prep_for_vlm(img_path)
        b64 = pil_to_base64(pil_img)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                        }
                    ]
                }
            ],
            temperature=0.0,
            max_tokens=50
        )
        
        result = response.choices[0].message.content
        return parse_label(result)
    
    except Exception as e:
        debug_print(f"❌ GPT error for {img_path.name}: {e}")
        return None


def classify_with_gemini(client, img_path: Path, prompt: str) -> Optional[str]:
    """Clasificare cu Gemini"""
    if client is None:
        return None
    
    try:
        pil_img = prep_for_vlm(img_path)
        
        for _ in range(2):
            try:
                response = client.generate_content(
                    [prompt, pil_img],
                    generation_config={
                        "temperature": 0.0,
                        "max_output_tokens": 50,
                    }
                )
                if response.parts:
                    return parse_label(response.text)
                time.sleep(1)
            except Exception:
                pass
        
        return None
    
    except Exception as e:
        debug_print(f"❌ Gemini error for {img_path.name}: {e}")
        return None


# ============================================================================
# HEURISTICI LOCALE (FALLBACK FINAL)
# ============================================================================

def count_features(img_gray: np.ndarray) -> tuple[int, float, float, int]:
    """Extrage features pentru clasificare locală"""
    h, w = img_gray.shape[:2]
    area_img = h * w
    
    g = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Canny(g, 50, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                           minLineLength=40, maxLineGap=8)
    
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.degrees(math.atan2(y2-y1, x2-x1))) % 180
            angles.append(angle)
    
    def is_ortho(a): return min(abs(a), abs(a-180)) < 8 or abs(a-90) < 8
    def is_diag(a): return (30 <= a <= 60) or (120 <= a <= 150)
    
    ortho_ratio = np.mean([is_ortho(a) for a in angles]) if angles else 0.0
    diag_ratio = np.mean([is_diag(a) for a in angles]) if angles else 0.0
    
    binv = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 21, 10)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binv, 8)
    
    text_components = sum(1 for stat in stats[1:] 
                         if 15 <= stat[cv2.CC_STAT_AREA] <= 800)
    
    _, thresh = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    room_like = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.0002 * area_img or area > 0.25 * area_img:
            continue
        
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        
        if len(approx) == 4 and cv2.isContourConvex(approx):
            room_like += 1
    
    return room_like, ortho_ratio, diag_ratio, text_components


def local_classify(img_path: Path) -> str:
    """Clasificare locală cu heuristici"""
    try:
        img = safe_imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rooms, ortho, diag, text_cc = count_features(gray)
        
        debug_print(f"   └─ Local features: rooms={rooms}, ortho={ortho:.2f}, "
                    f"diag={diag:.2f}, text_cc={text_cc}")
        
        if text_cc >= 1800 and rooms <= 1 and ortho <= 0.55:
            return "text_area"
        
        if rooms >= 5 and ortho >= 0.55 and diag <= 0.25:
            return "house_blueprint"
        
        if rooms <= 2 and diag >= 0.35:
            return "side_view"
        
        if rooms <= 2 and (text_cc >= 350 or ortho <= 0.45):
            return "site_blueprint"
        
        return "side_view"
    except Exception:
        return "text_area"


# ============================================================================
# SISTEM DE ARBITRAJ
# ============================================================================

def arbitrate_classification(
    img_path: Path,
    gpt_client,
    gemini_client,
    round_num: int
) -> tuple[Optional[str], Optional[str]]:
    """Rundă de arbitraj pentru clasificare"""
    prompts = [
        PROMPT_ROUND_1_BASE,
        PROMPT_ROUND_2_EXPLICIT,
        PROMPT_ROUND_3_EXTREME
    ]
    
    prompt = prompts[min(round_num, 2)]
    
    gpt_label = classify_with_gpt(gpt_client, img_path, prompt)
    gemini_label = classify_with_gemini(gemini_client, img_path, prompt)
    
    return gpt_label, gemini_label


# ============================================================================
# CLASIFICARE PRINCIPALĂ CU DUAL-MODEL + FAST FAILOVER
# ============================================================================

def classify_image_robust(
    img_path: Path,
    gpt_client,
    gemini_client
) -> ClassificationResult:
    """Clasificare cu sistem dual-model + Fast Failover"""
    img_name = img_path.name
    debug_print(f"\n🔍 Clasificare: {img_name}")
    
    gpt_label = classify_with_gpt(gpt_client, img_path, PROMPT_ROUND_1_BASE)
    gemini_label = classify_with_gemini(gemini_client, img_path, PROMPT_ROUND_1_BASE)
    
    debug_print(f"   Round 1 → GPT: {gpt_label}, Gemini: {gemini_label}")
    
    if gpt_label and gemini_label and gpt_label == gemini_label:
        debug_print(f"   ✅ Acord imediat: {gpt_label}")
        return ClassificationResult(
            image_path=img_path,
            label=gpt_label,  # type: ignore
            confidence=ConfidenceLevel.HIGH,
            gpt_vote=gpt_label,
            gemini_vote=gemini_label,
            arbitration_rounds=0
        )

    if not gemini_label and gpt_label:
        debug_print(f"   ⚠️ Gemini failed. Trusting GPT: {gpt_label}")
        return ClassificationResult(
            image_path=img_path,
            label=gpt_label,  # type: ignore
            confidence=ConfidenceLevel.HIGH,
            gpt_vote=gpt_label,
            gemini_vote=None,
            arbitration_rounds=0
        )

    if not gpt_label and gemini_label:
        debug_print(f"   ⚠️ GPT failed. Trusting Gemini: {gemini_label}")
        return ClassificationResult(
            image_path=img_path,
            label=gemini_label,  # type: ignore
            confidence=ConfidenceLevel.HIGH,
            gpt_vote=None,
            gemini_vote=gemini_label,
            arbitration_rounds=0
        )

    if not gpt_label and not gemini_label:
        debug_print("   ❌ Ambele AI au eșuat. Execut fallback heuristici...")
        local_label = local_classify(img_path)
        return ClassificationResult(
            image_path=img_path,
            label=local_label, # type: ignore
            confidence=ConfidenceLevel.LOW,
            gpt_vote=None,
            gemini_vote=None,
            arbitration_rounds=0
        )

    debug_print("   ⚔️  Dezbatere necesară (Modelele nu sunt de acord).")
    
    for round_num in range(1, 3):
        debug_print(f"   ⚖️  Round {round_num + 1} - Arbitraj...")
        
        gpt_label, gemini_label = arbitrate_classification(
            img_path, gpt_client, gemini_client, round_num
        )
        
        debug_print(f"   Round {round_num + 1} → GPT: {gpt_label}, Gemini: {gemini_label}")
        
        if gpt_label and gemini_label and gpt_label == gemini_label:
            debug_print(f"   ✅ Acord după arbitraj: {gpt_label}")
            return ClassificationResult(
                image_path=img_path,
                label=gpt_label,  # type: ignore
                confidence=ConfidenceLevel.MEDIUM,
                gpt_vote=gpt_label,
                gemini_vote=gemini_label,
                arbitration_rounds=round_num + 1
            )
        
        if gpt_label and not gemini_label:
             return ClassificationResult(image_path, gpt_label, ConfidenceLevel.MEDIUM, gpt_label, None, round_num + 1) # type: ignore
        if gemini_label and not gpt_label:
             return ClassificationResult(image_path, gemini_label, ConfidenceLevel.MEDIUM, None, gemini_label, round_num + 1) # type: ignore

    debug_print("   🔧 Fallback la heuristici locale (Arbitraj eșuat)...")
    local_label = local_classify(img_path)
    
    votes = [v for v in [gpt_label, gemini_label, local_label] if v]
    
    if len(votes) >= 2:
        vote_counts = Counter(votes)
        final_label = vote_counts.most_common(1)[0][0]
    else:
        final_label = local_label
    
    debug_print(f"   ⚠️  Voting final: {final_label} (votes: {votes})")
    
    return ClassificationResult(
        image_path=img_path,
        label=final_label,  # type: ignore
        confidence=ConfidenceLevel.LOW,
        gpt_vote=gpt_label,
        gemini_vote=gemini_label,
        arbitration_rounds=3
    )


# ============================================================================
# EXTRAGERE MĂSURĂTORI CAMERE PENTRU VERIFICARE DUPLICATE
# ============================================================================

def extract_room_measurements(img_path: Path, ai_client) -> dict:
    """
    Extrage măsurătorile camerelor dintr-un plan pentru verificare duplicate.
    
    Returns:
        dict cu structura:
        {
            "rooms": [
                {"name": "Bedroom", "area_m2": 14.5},
                {"name": "Kitchen", "area_m2": 12.3},
                ...
            ],
            "total_area": 85.4
        }
    """
    
    PROMPT = """You are an expert at reading architectural floor plans.

Extract ALL room measurements from this floor plan image.

For each room, identify:
1. Room name (e.g., "Bedroom", "Kitchen", "Bath", "Living", etc.)
2. Area in square meters (the number next to m²)

Return ONLY a JSON object with this EXACT structure:
{
  "rooms": [
    {"name": "Bedroom", "area_m2": 14.5},
    {"name": "Kitchen", "area_m2": 12.3}
  ]
}

CRITICAL:
- Extract ALL rooms you can see
- Use exact numbers from the plan
- Use decimal point (.), not comma
- Return ONLY valid JSON, no explanation
"""

    try:
        # Încercăm GPT
        if ai_client and hasattr(ai_client, 'chat'):
            try:
                pil_img = prep_for_vlm(img_path, min_long_edge=1280)
                b64 = pil_to_base64(pil_img)
                
                response = ai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": PROMPT},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                            ]
                        }
                    ],
                    temperature=0.0,
                    max_tokens=500
                )
                
                result = response.choices[0].message.content.strip()
                
                # Parse JSON
                if result.startswith("```"):
                    start = result.find('{')
                    end = result.rfind('}') + 1
                    if start != -1 and end > start:
                        result = result[start:end]
                
                data = json.loads(result)
                
                # Calculăm total
                if "rooms" in data:
                    total = sum(r.get("area_m2", 0) for r in data["rooms"])
                    data["total_area"] = round(total, 2)
                
                return data
                
            except Exception as e:
                debug_print(f"⚠️ GPT room extraction failed: {e}")
                return {"rooms": [], "total_area": 0}
        
        # Încercăm Gemini
        elif ai_client and hasattr(ai_client, 'generate_content'):
            try:
                pil_img = prep_for_vlm(img_path, min_long_edge=1280)
                
                response = ai_client.generate_content(
                    [PROMPT, pil_img],
                    generation_config={"temperature": 0.0, "max_output_tokens": 500}
                )
                
                if response.parts:
                    result = response.text.strip()
                    
                    # Parse JSON
                    if result.startswith("```"):
                        start = result.find('{')
                        end = result.rfind('}') + 1
                        if start != -1 and end > start:
                            result = result[start:end]
                    
                    data = json.loads(result)
                    
                    # Calculăm total
                    if "rooms" in data:
                        total = sum(r.get("area_m2", 0) for r in data["rooms"])
                        data["total_area"] = round(total, 2)
                    
                    return data
                
                return {"rooms": [], "total_area": 0}
                
            except Exception as e:
                debug_print(f"⚠️ Gemini room extraction failed: {e}")
                return {"rooms": [], "total_area": 0}
        
        return {"rooms": [], "total_area": 0}
    
    except Exception as e:
        debug_print(f"❌ Eroare la extragere măsurători: {e}")
        return {"rooms": [], "total_area": 0}


def compare_room_measurements(data1: dict, data2: dict, tolerance: float = 0.15) -> tuple[bool, str]:
    """
    Compară măsurătorile camerelor din 2 planuri.
    
    Args:
        data1, data2: Dict-uri cu structura {"rooms": [...], "total_area": X}
        tolerance: Toleranță procentuală (0.15 = 15%)
    
    Returns:
        (are_same_rooms, explanation)
    """
    
    rooms1 = data1.get("rooms", [])
    rooms2 = data2.get("rooms", [])
    
    total1 = data1.get("total_area", 0)
    total2 = data2.get("total_area", 0)
    
    # Dacă nu am măsurători din ambele, returnăm incert
    if not rooms1 or not rooms2:
        return (False, "NO_MEASUREMENTS")
    
    # Verificăm numărul de camere
    if abs(len(rooms1) - len(rooms2)) > 1:
        return (False, f"DIFFERENT_ROOM_COUNT: {len(rooms1)} vs {len(rooms2)}")
    
    # Verificăm suprafața totală
    if total1 > 0 and total2 > 0:
        diff_pct = abs(total1 - total2) / max(total1, total2)
        
        if diff_pct > tolerance:
            return (False, f"DIFFERENT_TOTAL_AREA: {total1:.1f}m² vs {total2:.1f}m² (diff: {diff_pct*100:.1f}%)")
    
    # Sortăm camerele după arie
    rooms1_sorted = sorted(rooms1, key=lambda r: r.get("area_m2", 0), reverse=True)
    rooms2_sorted = sorted(rooms2, key=lambda r: r.get("area_m2", 0), reverse=True)
    
    # Comparăm camerele pereche cu pereche
    matched = 0
    unmatched = []
    
    for i, r1 in enumerate(rooms1_sorted):
        area1 = r1.get("area_m2", 0)
        
        if i < len(rooms2_sorted):
            r2 = rooms2_sorted[i]
            area2 = r2.get("area_m2", 0)
            
            if area1 > 0 and area2 > 0:
                diff = abs(area1 - area2) / max(area1, area2)
                
                if diff <= tolerance:
                    matched += 1
                else:
                    unmatched.append(f"{area1:.1f}m² vs {area2:.1f}m² (diff: {diff*100:.1f}%)")
    
    # Dacă majoritatea camerelor se potrivesc
    match_ratio = matched / max(len(rooms1), len(rooms2))
    
    if match_ratio >= 0.7:  # 70% din camere match
        return (True, f"SAME_ROOMS: {matched}/{max(len(rooms1), len(rooms2))} rooms match")
    else:
        return (False, f"DIFFERENT_ROOMS: Only {matched}/{max(len(rooms1), len(rooms2))} match. Unmatched: {', '.join(unmatched[:3])}")


# ============================================================================
# DETECTARE DUPLICATE CLUSTERS (SIMPLE AI COMPARISON + ROOM MEASUREMENTS)
# ============================================================================

def check_duplicate_with_ai(
    img1_path: Path,
    img2_path: Path,
    ai_client
) -> tuple[bool, str]:
    """
    Întreabă AI-ul dacă cele 2 planuri sunt același lucru la scale diferit.
    ✅ ACUM verifică și măsurătorile camerelor pentru confirmare!
    
    Returns: (is_duplicate, ai_response_text)
    """
    
    PROMPT = """Look at these two architectural floor plans.

Question: Do these two images represent THE SAME floor plan, just at different scales or with different annotations/labels?

Consider:
- Same room layout and structure
- Same number of rooms
- Same wall positions
- Different scale is OK
- Different annotations/text is OK
- Minor rotation is OK

Answer with EXACTLY ONE WORD:
- "YES" if they are the same plan (just different scale/annotations)
- "NO" if they are different plans

Your answer:"""

    try:
        # STEP 1: Verificare vizuală AI (ca înainte)
        is_duplicate_visual = False
        visual_response = "NO_CHECK"
        
        # Încercăm GPT
        if ai_client and hasattr(ai_client, 'chat'):
            try:
                pil_img1 = prep_for_vlm(img1_path, min_long_edge=1024)
                pil_img2 = prep_for_vlm(img2_path, min_long_edge=1024)
                
                b64_1 = pil_to_base64(pil_img1)
                b64_2 = pil_to_base64(pil_img2)
                
                response = ai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": PROMPT},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_1}"}},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_2}"}}
                            ]
                        }
                    ],
                    temperature=0.0,
                    max_tokens=10
                )
                
                visual_response = response.choices[0].message.content.strip().upper()
                is_duplicate_visual = "YES" in visual_response
                
            except Exception as e:
                debug_print(f"⚠️ GPT visual check failed: {e}")
        
        # Încercăm Gemini
        elif ai_client and hasattr(ai_client, 'generate_content'):
            try:
                pil_img1 = prep_for_vlm(img1_path, min_long_edge=1024)
                pil_img2 = prep_for_vlm(img2_path, min_long_edge=1024)
                
                response = ai_client.generate_content(
                    [PROMPT, pil_img1, "---SECOND IMAGE---", pil_img2],
                    generation_config={"temperature": 0.0, "max_output_tokens": 10}
                )
                
                if response.parts:
                    visual_response = response.text.strip().upper()
                    is_duplicate_visual = "YES" in visual_response
                
            except Exception as e:
                debug_print(f"⚠️ Gemini visual check failed: {e}")
        
        # STEP 2: Dacă AI-ul vizual zice YES, verificăm și măsurătorile
        if is_duplicate_visual:
            debug_print(f"      → Visual check: YES - Verific măsurători...")
            
            # Extragem măsurători din ambele planuri
            measurements1 = extract_room_measurements(img1_path, ai_client)
            measurements2 = extract_room_measurements(img2_path, ai_client)
            
            debug_print(f"         Plan 1: {len(measurements1.get('rooms', []))} rooms, total {measurements1.get('total_area', 0):.1f}m²")
            debug_print(f"         Plan 2: {len(measurements2.get('rooms', []))} rooms, total {measurements2.get('total_area', 0):.1f}m²")
            
            # Comparăm măsurătorile
            are_same, explanation = compare_room_measurements(measurements1, measurements2, tolerance=0.15)
            
            if are_same:
                final_response = f"YES (Visual + Measurements Match: {explanation})"
                debug_print(f"      ✅ CONFIRMED DUPLICATE: {explanation}")
                return (True, final_response)
            else:
                final_response = f"NO (Visual YES but Measurements Differ: {explanation})"
                debug_print(f"      ❌ NOT DUPLICATE: {explanation}")
                return (False, final_response)
        
        # Dacă AI-ul vizual zice NO, e clar diferit
        return (is_duplicate_visual, visual_response)
    
    except Exception as e:
        debug_print(f"❌ Eroare la verificare duplicate: {e}")
        return (False, f"ERROR: {str(e)}")


# Numărul maxim de blueprint-uri luate în calcul pentru detectare duplicate.
# Comparările se fac DOAR între clusterele finale (blueprint); dacă sunt prea multe,
# luăm doar cele mai mari (după suprafață) ca să nu ajungem la sute de perechi.
MAX_BLUEPRINTS_FOR_DUPLICATE_CHECK = 10


def detect_and_remove_duplicates(
    crops_dir: Path,
    ai_client
) -> list[DuplicateGroup]:
    """
    Detectează și elimină blueprint-urile duplicate prin comparație AI directă.
    Compară DOAR între clusterele finale din folderul blueprints (nu toate clusterele).
    Dacă sunt mai multe de MAX_BLUEPRINTS_FOR_DUPLICATE_CHECK, se iau doar cele mai mari
    după suprafață, ca să limităm numărul de perechi.
    """
    
    print("\n[STEP 7B] Detectare și eliminare duplicate blueprint-uri finale...")
    
    image_files = sorted(crops_dir.glob("*.jpg")) + sorted(crops_dir.glob("*.png"))
    
    if len(image_files) < 2:
        print("ℹ️ Prea puține blueprint-uri pentru verificare duplicate.")
        return []
    
    if not ai_client:
        print("⚠️ Niciun client AI disponibil - skip detectare duplicate.")
        return []
    
    # Obținem dimensiunile fișierelor
    image_data = []
    for img_file in image_files:
        try:
            img = cv2.imread(str(img_file))
            if img is not None:
                h, w = img.shape[:2]
                file_size = img_file.stat().st_size
                image_data.append({
                    'path': img_file,
                    'size': file_size,
                    'width': w,
                    'height': h,
                    'pixels': w * h
                })
        except Exception as e:
            debug_print(f"⚠️ Eroare procesare {img_file.name}: {e}")
    
    # Sortăm descrescător după suprafață; pentru duplicate check folosim doar primele N
    image_data.sort(key=lambda x: x['pixels'], reverse=True)
    to_compare = image_data[:MAX_BLUEPRINTS_FOR_DUPLICATE_CHECK]
    skipped = len(image_data) - len(to_compare)
    if skipped > 0:
        print(f"📊 Blueprint-uri în folder: {len(image_data)}. Pentru duplicate verific doar cele {len(to_compare)} cele mai mari (max {MAX_BLUEPRINTS_FOR_DUPLICATE_CHECK}).")
    else:
        print(f"📊 Verificare {len(to_compare)} blueprint-uri pentru duplicate...")
    total_comparisons = len(to_compare) * (len(to_compare) - 1) // 2
    print(f"🔍 Total perechi de verificat: {total_comparisons}\n")
    
    confirmed_duplicates = []
    current_comparison = 0
    
    for i in range(len(to_compare)):
        for j in range(i + 1, len(to_compare)):
            current_comparison += 1
            
            img1 = to_compare[i]
            img2 = to_compare[j]
            
            path1 = img1['path']
            path2 = img2['path']
            
            print(f"[{current_comparison}/{total_comparisons}] Comparare:")
            print(f"  • {path1.name} ({img1['width']}x{img1['height']}, {img1['size']/1024:.1f} KB)")
            print(f"  • {path2.name} ({img2['width']}x{img2['height']}, {img2['size']/1024:.1f} KB)")
            
            # Întrebăm AI-ul
            is_duplicate, ai_response = check_duplicate_with_ai(path1, path2, ai_client)
            
            print(f"  → AI răspuns: {ai_response}")
            
            if is_duplicate:
                print(f"  ✅ DUPLICATE detectat!\n")
                confirmed_duplicates.append({
                    'img1': img1,
                    'img2': img2,
                    'ai_response': ai_response
                })
            else:
                print(f"  ❌ Nu sunt duplicate\n")
    
    # Dacă nu am găsit nimic
    if not confirmed_duplicates:
        print("✅ Nu s-au găsit duplicate!\n")
        return []
    
    # Construim grupuri de duplicate
    print(f"🗑️ Procesare {len(confirmed_duplicates)} perechi duplicate...\n")
    
    graph = defaultdict(set)
    
    for dup in confirmed_duplicates:
        p1 = dup['img1']['path']
        p2 = dup['img2']['path']
        graph[p1].add(p2)
        graph[p2].add(p1)
    
    # Găsim componente conectate (grupuri de duplicate)
    visited = set()
    duplicate_groups = []
    
    def dfs(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, component)
    
    for node in graph:
        if node not in visited:
            component = []
            dfs(node, component)
            if len(component) > 1:
                # Sortăm după număr de pixeli (păstrăm cel mai mare)
                component_with_data = []
                for path in component:
                    img_data = next((d for d in image_data if d['path'] == path), None)
                    if img_data:
                        component_with_data.append(img_data)
                
                component_with_data.sort(key=lambda x: x['pixels'], reverse=True)
                
                keep = component_with_data[0]['path']
                remove = [d['path'] for d in component_with_data[1:]]
                
                # Găsim răspunsul AI pentru grup
                ai_responses = [d['ai_response'] for d in confirmed_duplicates 
                               if (d['img1']['path'] in component and d['img2']['path'] in component)]
                
                duplicate_groups.append(DuplicateGroup(
                    keep_cluster=keep,
                    remove_clusters=remove,
                    ai_response=", ".join(ai_responses) if ai_responses else "YES"
                ))
    
    # Eliminăm clusterele duplicate
    total_removed = 0
    
    for group in duplicate_groups:
        keep_data = next((d for d in image_data if d['path'] == group.keep_cluster), None)
        
        print(f"🔹 Grup duplicate (AI: {group.ai_response}):")
        if keep_data:
            print(f"   ✅ PĂSTRAT: {group.keep_cluster.name}")
            print(f"      └─ Dimensiuni: {keep_data['width']}x{keep_data['height']} ({keep_data['size']/1024:.1f} KB)")
        
        for remove_path in group.remove_clusters:
            remove_data = next((d for d in image_data if d['path'] == remove_path), None)
            try:
                remove_path.unlink()
                if remove_data:
                    print(f"   🗑️ ELIMINAT: {remove_path.name}")
                    print(f"      └─ Dimensiuni: {remove_data['width']}x{remove_data['height']} ({remove_data['size']/1024:.1f} KB)")
                total_removed += 1
            except Exception as e:
                debug_print(f"⚠️ Eroare la eliminare {remove_path.name}: {e}")
        
        print()
    
    print(f"✅ Eliminare duplicate finalizată!")
    print(f"   • Grupuri duplicate: {len(duplicate_groups)}")
    print(f"   • Clustere eliminate: {total_removed}")
    print(f"   • Clustere rămase: {len(image_files) - total_removed}\n")
    
    return duplicate_groups


# ============================================================================
# PADDING ALB PENTRU BLUEPRINTS
# ============================================================================

def add_white_padding(src: Path, dst: Path, padding_pct: float = 0.10) -> None:
    """Adaugă un contur alb procentual în jurul imaginii"""
    try:
        img = cv2.imread(str(src))
        if img is None:
            shutil.copy(str(src), str(dst))
            return

        h, w = img.shape[:2]
        pad_h = int(h * padding_pct)
        pad_w = int(w * padding_pct)

        img_padded = cv2.copyMakeBorder(
            img, 
            pad_h, pad_h, pad_w, pad_w, 
            cv2.BORDER_CONSTANT, 
            value=[255, 255, 255]
        )
        
        cv2.imwrite(str(dst), img_padded)
        
    except Exception as e:
        debug_print(f"⚠️ Eroare la adăugare padding pentru {src.name}: {e}")
        shutil.copy(str(src), str(dst))


# ============================================================================
# FUNCȚIA PRINCIPALĂ DE CLASIFICARE
# ============================================================================

def classify_segmented_plans(segmentation_out: str | Path) -> list[ClassificationResult]:
    """Clasifică planurile decupate de segmenter"""
    segmentation_out = Path(segmentation_out).resolve()
    
    from .common import set_output_dir
    set_output_dir(segmentation_out)
    
    crops_dir = segmentation_out / STEP_DIRS["clusters"]["crops"]
    
    bp_dir = segmentation_out / STEP_DIRS["classified"]["blueprints"]
    sp_dir = segmentation_out / STEP_DIRS["classified"]["siteplan"]
    sv_dir = segmentation_out / STEP_DIRS["classified"]["side_views"]
    tx_dir = segmentation_out / STEP_DIRS["classified"]["text"]
    
    for d in (bp_dir, sp_dir, sv_dir, tx_dir):
        d.mkdir(parents=True, exist_ok=True)
    
    if not crops_dir.is_dir():
        debug_print(f"ℹ️ Nu există crops_dir cu planuri: {crops_dir}")
        return []
    
    print("\n[STEP 7A] Inițializare AI clients...")
    gpt_client = setup_openai_client()
    gemini_client = setup_gemini_client()
    
    clients = []
    if gpt_client: clients.append("GPT-4o")
    if gemini_client: clients.append("Gemini (Auto-Detect)")
    
    if not clients:
        print("⚠️  Niciun client AI disponibil - folosesc DOAR heuristici locale")
    else:
        print(f"✅ Clienți activi: {', '.join(clients)}")
    
    # CLASIFICARE
    print("[STEP 8] Clasificare cu Dual-Model Validation (GPT-4o + Gemini)...")
    
    results: list[ClassificationResult] = []
    
    image_files = sorted(crops_dir.glob("*.jpg")) + sorted(crops_dir.glob("*.png"))
    
    if not image_files:
        debug_print(f"ℹ️ Nicio imagine rămasă în {crops_dir}")
        return []
    
    print(f"📊 Clasificare {len(image_files)} planuri...\n")
    
    for img_file in image_files:
        result = classify_image_robust(img_file, gpt_client, gemini_client)
        
        label_to_dir = {
            "house_blueprint": bp_dir,
            "site_blueprint": sp_dir,
            "side_view": sv_dir,
            "text_area": tx_dir
        }
        
        dst_dir = label_to_dir[result.label]
        dst = dst_dir / img_file.name
        
        if result.label == "house_blueprint":
            add_white_padding(img_file, dst, padding_pct=0.10)
        else:
            shutil.copy(str(img_file), str(dst))
        
        result.image_path = dst
        results.append(result)
        
        confidence_emoji = {
            ConfidenceLevel.HIGH: "🟢",
            ConfidenceLevel.MEDIUM: "🟡",
            ConfidenceLevel.LOW: "🔴"
        }
        
        label_emoji = {
            "house_blueprint": "🏗",
            "site_blueprint": "🗺",
            "side_view": "🏠",
            "text_area": "📝"
        }
        
        print(f"{confidence_emoji[result.confidence]} {label_emoji[result.label]} "
              f"{result.label:20} | {img_file.name}")
    
    print("\n✅ Clasificare finalizată!\n")
    
    # DETECTARE ȘI ELIMINARE DUPLICATE (DOAR PENTRU BLUEPRINT-URI FINALE)
    # Folosim primul client disponibil (prioritate GPT)
    ai_client = gpt_client if gpt_client else gemini_client
    
    if ai_client and bp_dir.exists():
        blueprint_files = list(bp_dir.glob("*.jpg")) + list(bp_dir.glob("*.png"))
        if len(blueprint_files) >= 2:
            print(f"[STEP 7B] Detectare duplicate în blueprint-uri finale ({len(blueprint_files)} fișiere)...")
            duplicate_groups = detect_and_remove_duplicates(
                crops_dir=bp_dir,
                ai_client=ai_client
            )
        else:
            print(f"[STEP 7B] Prea puține blueprint-uri ({len(blueprint_files)}) pentru verificare duplicate.")
    else:
        if not ai_client:
            print("[STEP 7B] ⚠️ Niciun client AI disponibil - skip detectare duplicate.")
        if not bp_dir.exists():
            print(f"[STEP 7B] ⚠️ Folder blueprint-uri nu există: {bp_dir}")
    
    # POST-VALIDARE
    print("[STEP 8B] Post-validare folder 'blueprints' cu heuristici...")
    
    moved = 0
    for img_file in sorted(bp_dir.iterdir()):
        if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        
        lbl = local_classify(img_file)
        
        if lbl in ("side_view", "site_blueprint", "text_area"):
            if lbl == "side_view":
                dst_dir = sv_dir
            elif lbl == "site_blueprint":
                dst_dir = sp_dir
            else:
                dst_dir = tx_dir
            
            dst = dst_dir / img_file.name
            shutil.move(str(img_file), str(dst))
            moved += 1
            print(f"↪️  mutat din blueprints în {dst_dir.name}: {img_file.name}")
            
            for r in results:
                if r.image_path == img_file:
                    r.image_path = dst
                    r.label = lbl  # type: ignore[assignment]
                    r.confidence = ConfidenceLevel.LOW
                    break
    
    print(f"✅ Post-validare terminată. Mutate din blueprints: {moved}\n")
    
    return results