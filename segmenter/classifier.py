# file: engine/runner/segmenter/classifier.py
# ------------------------------------------------------------
# Clasificare ULTRA-ROBUSTĂ cu dual-model validation
# GPT-4o + Gemini (Auto-Fallback) + Fast Failover Logic
# ------------------------------------------------------------

from __future__ import annotations

import os
import math
import shutil
import io
import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Any
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
    HIGH = "high"      # Acord unanim sau Single Model Success (celălalt a eșuat)
    MEDIUM = "medium"  # De acord după arbitraj
    LOW = "low"        # Doar heuristici locale


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
        
        # Sharpening pentru detalii mai clare
        im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
        return im
    except Exception as e:
        debug_print(f"⚠️ Eroare la procesarea imaginii {img_path}: {e}")
        # Returnăm o imagine goală neagră în caz de eroare gravă
        return Image.new('RGB', (500, 500), color='black')


def pil_to_base64(pil_img: Image.Image) -> str:
    """Convertește PIL Image la base64"""
    buf = io.BytesIO()
    # JPEG cu calitate 85 este mult mai rapid la upload decât PNG
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def parse_label(text: str) -> Optional[str]:
    """Extrage label valid din răspunsul modelului"""
    if not text:
        return None
    
    text_clean = text.strip().lower()
    
    # Elimină caractere nedorite (markdown, punctuație)
    text_clean = text_clean.replace("```", "").replace("[", "").replace("]", "").strip()
    
    # Verificare directă
    if text_clean in ALLOWED_LABELS:
        return text_clean
    
    # Căutare în text (dacă modelul e guraliv)
    found_labels = []
    for label in ALLOWED_LABELS:
        if label in text_clean:
            found_labels.append(label)
    
    # Dacă găsim exact un label în text, îl returnăm
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
        
        # Încercăm modelele în ordine: cel mai nou -> cel mai stabil
        models_to_try = [
            'gemini-2.5-flash',
            'gemini-2.0-flash',
            'gemini-1.5-flash'
        ]
        
        for model_name in models_to_try:
            try:
                # Test simplu de instanțiere
                model = genai.GenerativeModel(model_name, safety_settings=safety_settings)
                # Dacă nu a dat eroare la creare, presupunem că e bun.
                # Eroarea reală de acces apare la generate_content, dar o prindem acolo.
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
        
        # Retry logic simplu
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
        
        # Dacă ajungem aici, Gemini nu a returnat nimic valid
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
    try:
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
    except Exception:
        return "text_area" # Cel mai safe fallback


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
    Rundă de arbitraj. Se apelează doar dacă ambele modele au răspuns în runda anterioară
    dar nu s-au pus de acord.
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
# CLASIFICARE PRINCIPALĂ CU DUAL-MODEL + FAST FAILOVER
# ============================================================================

def classify_image_robust(
    img_path: Path,
    gpt_client,
    gemini_client
) -> ClassificationResult:
    """
    Clasificare cu sistem dual-model + Fast Failover.
    """
    img_name = img_path.name
    debug_print(f"\n🔍 Clasificare: {img_name}")
    
    # RUNDA 1: Clasificare inițială
    gpt_label = classify_with_gpt(gpt_client, img_path, PROMPT_ROUND_1_BASE)
    gemini_label = classify_with_gemini(gemini_client, img_path, PROMPT_ROUND_1_BASE)
    
    debug_print(f"   Round 1 → GPT: {gpt_label}, Gemini: {gemini_label}")
    
    # ----------------------------------------------------
    # FAST FAILOVER LOGIC
    # ----------------------------------------------------
    
    # CAZ 1: Ambele modele funcționează și sunt de acord
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

    # CAZ 2: Gemini a eșuat (None), dar GPT a reușit -> TRUST GPT
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

    # CAZ 3: GPT a eșuat (None), dar Gemini a reușit -> TRUST GEMINI
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

    # CAZ 4: Ambele au eșuat (None) -> Fallback la Heuristici
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

    # CAZ 5: Ambele au răspuns, dar DIFERIT -> Arbitraj
    # Doar aici intrăm în bucla de așteptare/retry
    debug_print("   ⚔️  Dezbatere necesară (Modelele nu sunt de acord).")
    
    for round_num in range(1, 3):
        debug_print(f"   ⚖️  Round {round_num + 1} - Arbitraj...")
        
        gpt_label, gemini_label = arbitrate_classification(
            img_path, gpt_client, gemini_client, round_num
        )
        
        debug_print(f"   Round {round_num + 1} → GPT: {gpt_label}, Gemini: {gemini_label}")
        
        # Verificăm din nou condițiile de acord
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
        
        # Fast failover în timpul arbitrajului (dacă unul moare pe parcurs)
        if gpt_label and not gemini_label:
             return ClassificationResult(image_path, gpt_label, ConfidenceLevel.MEDIUM, gpt_label, None, round_num + 1) # type: ignore
        if gemini_label and not gpt_label:
             return ClassificationResult(image_path, gemini_label, ConfidenceLevel.MEDIUM, None, gemini_label, round_num + 1) # type: ignore

    
    # FALLBACK FINAL: Heuristici locale + voting din ultimele rezultate AI
    debug_print("   🔧 Fallback la heuristici locale (Arbitraj eșuat)...")
    local_label = local_classify(img_path)
    
    # Voting: 2/3 wins (Ultimul GPT, Ultimul Gemini, Local)
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
    Clasifică planurile decupate de segmenter.
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
    
    # Logging clienți activi
    clients = []
    if gpt_client: clients.append("GPT-4o")
    if gemini_client: clients.append("Gemini (Auto-Detect)")
    
    if not clients:
        print("⚠️  Niciun client AI disponibil - folosesc DOAR heuristici locale")
    else:
        print(f"✅ Clienți activi: {', '.join(clients)}")
    
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
              f"{result.label:20} | {img_file.name}")
    
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
    
    return results