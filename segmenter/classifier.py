# file: engine/runner/segmenter/classifier.py
# ------------------------------------------------------------
# Clasificare ULTRA-ROBUSTĂ cu dual-model validation
# GPT-4o + Gemini + sistem de arbitraj în 3 runde
# Integrat în structura existentă engine/runner/segmenter
# ------------------------------------------------------------

from __future__ import annotations

import os
import math
import shutil
import io
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
from enum import Enum

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
    HIGH = "high"  # Ambele modele de acord
    MEDIUM = "medium"  # De acord după arbitraj
    LOW = "low"  # Doar heuristici locale


@dataclass
class ClassificationResult:
    """Rezultat clasificare cu nivel de încredere"""
    image_path: Path
    label: LabelType
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH
    gpt_vote: Optional[str] = None
    gemini_vote: Optional[str] = None
    arbitration_rounds: int = 0


# ============================================================================
# PROMPTURI PENTRU FIECARE RUNDĂ DE ARBITRAJ
# ============================================================================

PROMPT_ROUND_1_BASE = """You are an EXPERT architectural drawing classifier with 20 years of experience.

CRITICAL: Return EXACTLY ONE label from this list: house_blueprint | site_blueprint | side_view | text_area

DEFINITIONS (Read carefully):

1. house_blueprint = TOP-DOWN FLOOR PLAN of a HOUSE/BUILDING interior
   ✓ MUST HAVE: Interior walls forming rooms, door/window symbols, dimension lines
   ✓ MUST BE: 2D top-down view (bird's eye view)
   ✗ NOT: Exterior elevations, 3D views, site plans, sections
   
2. site_blueprint = SITE/LOT PLAN showing property boundaries and context
   ✓ MUST HAVE: Property/lot boundaries, plot lines, street names or roads
   ✓ OFTEN HAS: North arrow, compass rose, setback lines, landscaping, driveways
   ✗ NOT: Interior floor plans, building elevations
   
3. side_view = EXTERIOR ELEVATION or 3D PERSPECTIVE
   ✓ MUST BE: Side/front/rear view of building exterior OR 3D rendering
   ✓ Shows: Façade, windows, roof profile, exterior materials
   ✗ NOT: Top-down floor plans, site plans
   
4. text_area = PAGE DOMINATED BY TEXT
   ✓ MUST HAVE: Primarily paragraphs, tables, legends, specifications
   ✓ More than 40% of image is text blocks
   ✗ NOT: Drawings with labels (those are blueprints)

STRICT RULES:
- If you see interior rooms from above → house_blueprint
- If you see property boundaries and streets → site_blueprint  
- If you see building from the side → side_view
- If mostly text → text_area

Return ONLY the label, nothing else."""


PROMPT_ROUND_2_EXPLICIT = """You are a FORENSIC architectural document analyzer. This is CRITICAL classification.

Return EXACTLY ONE: house_blueprint | site_blueprint | side_view | text_area

ULTRA-PRECISE DEFINITIONS:

🏠 house_blueprint (INTERIOR floor plan, top-down):
  REQUIRED FEATURES:
  ✓ Multiple interior walls creating rooms/spaces
  ✓ Door symbols (arcs, gaps in walls)
  ✓ Window symbols (parallel lines in walls)
  ✓ Dimension lines with measurements
  ✓ View is from DIRECTLY ABOVE (plan view)
  
  NEGATIVE INDICATORS (if present → NOT house_blueprint):
  ✗ Building shown from side/angle
  ✗ Only exterior boundaries visible
  ✗ Streets or property lines
  ✗ 3D perspective or shading

🗺️ site_blueprint (PROPERTY/LOT plan):
  REQUIRED FEATURES:
  ✓ Property boundary lines (often bold perimeter)
  ✓ Street names or road indicators
  ✓ Setback dimensions from property lines
  ✓ Building footprint outline (not interior details)
  ✓ North arrow or orientation indicator
  
  NEGATIVE INDICATORS:
  ✗ Interior room divisions
  ✗ Detailed door/window symbols
  ✗ Furniture layout

🏢 side_view (ELEVATION or 3D view):
  REQUIRED FEATURES:
  ✓ Building viewed from side/front/rear angle
  ✓ Visible roof profile/pitch
  ✓ Vertical walls showing height
  ✓ Windows shown as rectangles on wall surface
  ✓ May have perspective/depth
  
  NEGATIVE INDICATORS:
  ✗ Top-down bird's eye view
  ✗ Interior floor layout visible

📄 text_area (TEXT-HEAVY page):
  REQUIRED FEATURES:
  ✓ >40% of image is text paragraphs/tables
  ✓ Specifications, notes, schedules
  ✓ Minimal or no drawings
  
  NEGATIVE INDICATORS:
  ✗ Clear architectural drawing with labels

CRITICAL DECISION TREE:
1. Can you see interior rooms from above? → house_blueprint
2. Can you see property lines + streets? → site_blueprint
3. Is building shown from the side? → side_view
4. Mostly text? → text_area

Output format: [label_only]"""


PROMPT_ROUND_3_EXTREME = """FINAL ARBITRATION - Maximum precision required.

You MUST choose: house_blueprint | site_blueprint | side_view | text_area

DECISION MATRIX (follow exactly):

Step 1: Identify the VIEW ANGLE
- If viewing from DIRECTLY ABOVE (bird's eye) → Continue to Step 2
- If viewing from SIDE/ANGLE (elevation/3D) → side_view
- If no clear drawing, mostly text → text_area

Step 2: What do you see from above?
- If you see INTERIOR WALLS dividing rooms + door/window symbols → house_blueprint
- If you see PROPERTY BOUNDARIES + street names → site_blueprint
- If unclear → Continue to Step 3

Step 3: Final checks
- Count interior rooms visible: If ≥2 rooms → house_blueprint
- See property perimeter + roads: → site_blueprint
- Building from exterior angle: → side_view
- Text dominates: → text_area

VISUAL CLUES:
🏠 house_blueprint: "I can see the kitchen, bathroom, bedroom layout from above"
🗺️ site_blueprint: "I can see where the building sits on the lot with streets"
🏢 side_view: "I can see the front/side of the building with roof"
📄 text_area: "This is mostly text specifications"

Return ONLY the label."""


# ============================================================================
# HELPERS PENTRU IMAGINI
# ============================================================================

def prep_for_vlm(img_path: str | Path, min_long_edge: int = 1280) -> Image.Image:
    """Pregătește imaginea pentru VLM: resize + sharpen"""
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    long_edge = max(w, h)
    
    if long_edge < min_long_edge:
        scale = min_long_edge / float(long_edge)
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    
    # Sharpening pentru detalii mai clare
    im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    return im


def pil_to_base64(pil_img: Image.Image) -> str:
    """Convertește PIL Image la base64"""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def parse_label(text: str) -> Optional[str]:
    """Extrage label valid din răspunsul modelului"""
    if not text:
        return None
    
    text_clean = text.strip().lower()
    
    # Verificare directă
    if text_clean in ALLOWED_LABELS:
        return text_clean
    
    # Căutare în text
    for label in ALLOWED_LABELS:
        if label in text_clean:
            return label
    
    return None


# ============================================================================
# CLIENT SETUP
# ============================================================================

def setup_openai_client():
    """Inițializează clientul OpenAI"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        debug_print("⚠️  Lipsă OPENAI_API_KEY în .env")
        return None
    
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception as e:
        debug_print(f"⚠️  Eroare OpenAI init: {e}")
        return None


def setup_gemini_client():
    """Inițializează clientul Gemini"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        debug_print("⚠️  Lipsă GEMINI_API_KEY în .env")
        return None
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.5-flash')
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
                            "image_url": {"url": f"data:image/png;base64,{b64}"}
                        }
                    ]
                }
            ],
            temperature=0.0,
            max_tokens=100
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
        
        response = client.generate_content(
            [prompt, pil_img],
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 100,
            }
        )
        
        return parse_label(response.text)
    
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
    
    # Edge detection
    g = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Canny(g, 50, 150)
    
    # Detectare linii
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                           minLineLength=40, maxLineGap=8)
    
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.degrees(math.atan2(y2-y1, x2-x1))) % 180
            angles.append(angle)
    
    # Calcul orientări
    def is_ortho(a): return min(abs(a), abs(a-180)) < 8 or abs(a-90) < 8
    def is_diag(a): return (30 <= a <= 60) or (120 <= a <= 150)
    
    ortho_ratio = np.mean([is_ortho(a) for a in angles]) if angles else 0.0
    diag_ratio = np.mean([is_diag(a) for a in angles]) if angles else 0.0
    
    # Detectare componente mici (posibil text)
    binv = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 21, 10)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binv, 8)
    
    text_components = sum(1 for stat in stats[1:] 
                         if 15 <= stat[cv2.CC_STAT_AREA] <= 800)
    
    # Detectare forme dreptunghiulare (camere)
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
    """Clasificare locală cu heuristici (fallback când AI-ul nu merge)"""
    img = safe_imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rooms, ortho, diag, text_cc = count_features(gray)
    
    debug_print(f"   └─ Local features: rooms={rooms}, ortho={ortho:.2f}, "
                f"diag={diag:.2f}, text_cc={text_cc}")
    
    # Logică de clasificare
    if text_cc >= 1800 and rooms <= 1 and ortho <= 0.55:
        return "text_area"
    
    if rooms >= 5 and ortho >= 0.55 and diag <= 0.25:
        return "house_blueprint"
    
    if rooms <= 2 and diag >= 0.35:
        return "side_view"
    
    if rooms <= 2 and (text_cc >= 350 or ortho <= 0.45):
        return "site_blueprint"
    
    return "side_view"


# ============================================================================
# SISTEM DE ARBITRAJ
# ============================================================================

def arbitrate_classification(
    img_path: Path,
    gpt_client,
    gemini_client,
    round_num: int
) -> tuple[Optional[str], Optional[str]]:
    """
    Rundă de arbitraj cu prompturi mai detaliate
    Returnează (gpt_label, gemini_label)
    """
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
# CLASIFICARE PRINCIPALĂ CU DUAL-MODEL
# ============================================================================

def classify_image_robust(
    img_path: Path,
    gpt_client,
    gemini_client
) -> ClassificationResult:
    """
    Clasificare cu sistem dual-model + arbitraj în 3 runde
    """
    img_name = img_path.name
    debug_print(f"\n🔍 Clasificare: {img_name}")
    
    # RUNDA 1: Clasificare inițială
    gpt_label = classify_with_gpt(gpt_client, img_path, PROMPT_ROUND_1_BASE)
    gemini_label = classify_with_gemini(gemini_client, img_path, PROMPT_ROUND_1_BASE)
    
    debug_print(f"   Round 1 → GPT: {gpt_label}, Gemini: {gemini_label}")
    
    # Verificare acord
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
    
    # RUNDA 2-3: Arbitraj cu prompturi mai detaliate
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
    
    # FALLBACK: Heuristici locale + voting
    debug_print("   🔧 Fallback la heuristici locale...")
    local_label = local_classify(img_path)
    
    # Voting: 2/3 wins
    votes = [v for v in [gpt_label, gemini_label, local_label] if v]
    
    if len(votes) >= 2:
        from collections import Counter
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
# FUNCȚIA PRINCIPALĂ DE CLASIFICARE (INTERFAȚĂ COMPATIBILĂ)
# ============================================================================

def classify_segmented_plans(segmentation_out: str | Path) -> list[ClassificationResult]:
    """
    Clasifică planurile decupate de segmenter (clusters/plan_crops) în:
      - house_blueprint
      - site_blueprint
      - side_view
      - text_area

    Folosește:
      - Dual-model validation (GPT-4o + Gemini)
      - Arbitraj în 3 runde dacă diferă
      - Fallback la heuristici locale + voting

    Args:
        segmentation_out: output_dir de la segment_document(...)
                         (e.g., job_root / 'segmentation')

    Returns:
        Lista de ClassificationResult
    """
    segmentation_out = Path(segmentation_out).resolve()
    
    # Directoare
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
    
    # Setup clienți AI
    print("\n[STEP 8] Clasificare cu Dual-Model Validation (GPT-4o + Gemini)...")
    gpt_client = setup_openai_client()
    gemini_client = setup_gemini_client()
    
    if gpt_client is None and gemini_client is None:
        print("⚠️  Niciun client AI disponibil - folosesc DOAR heuristici locale")
        print("   Setează OPENAI_API_KEY sau GEMINI_API_KEY în .env pentru precizie maximă")
    elif gpt_client is None:
        print("⚠️  OpenAI indisponibil - folosesc Gemini + heuristici")
    elif gemini_client is None:
        print("⚠️  Gemini indisponibil - folosesc GPT + heuristici")
    else:
        print("✅ Ambii clienți AI disponibili (precizie maximă)")
    
    results: list[ClassificationResult] = []
    
    # Procesare imagini
    image_files = sorted(crops_dir.glob("*.jpg")) + sorted(crops_dir.glob("*.png"))
    
    if not image_files:
        debug_print(f"ℹ️ Nicio imagine găsită în {crops_dir}")
        return []
    
    print(f"📊 Clasificare {len(image_files)} planuri...\n")
    
    for img_file in image_files:
        # Clasificare cu sistem robust
        result = classify_image_robust(img_file, gpt_client, gemini_client)
        
        # Mapare în foldere
        label_to_dir = {
            "house_blueprint": bp_dir,
            "site_blueprint": sp_dir,
            "side_view": sv_dir,
            "text_area": tx_dir
        }
        
        dst_dir = label_to_dir[result.label]
        dst = dst_dir / img_file.name
        shutil.copy(str(img_file), str(dst))
        
        # Update path în rezultat
        result.image_path = dst
        results.append(result)
        
        # Status logging
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
              f"{result.label:20} | Rounds: {result.arbitration_rounds} | {img_file.name}")
    
    print("\n✅ Clasificare finalizată!\n")
    
    # ==========================
    # STEP 8B – Post-validare
    # ==========================
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
            
            # Actualizăm rezultatele
            for r in results:
                if r.image_path == img_file:
                    r.image_path = dst
                    r.label = lbl  # type: ignore[assignment]
                    r.confidence = ConfidenceLevel.LOW  # downgrade confidence
                    break
    
    print(f"✅ Post-validare terminată. Mutate din blueprints: {moved}\n")
    
    # Statistici finale
    from collections import Counter
    label_counts = Counter(r.label for r in results)
    confidence_counts = Counter(r.confidence for r in results)
    
    print("📊 STATISTICI FINALE:")
    print("-" * 50)
    print("\nDistribuție labels:")
    for label, count in label_counts.items():
        print(f"  {label:20}: {count:3} imagini")
    
    print("\nNivel încredere:")
    for conf, count in confidence_counts.items():
        print(f"  {conf.value:10}: {count:3} imagini")
    
    if results:
        avg_rounds = sum(r.arbitration_rounds for r in results) / len(results)
        print(f"\nRunde arbitraj medii: {avg_rounds:.1f}")
    
    return results