# floor_classifier/basement_scorer.py
# Scoring cu Gemini: care plan este cel mai probabil să fie beciul (Keller).

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Any

# ClassifiedPlanInfo din orchestrator (job_root, image_path, label)
# image_path este Path către imaginea planului


PROMPT_BASEMENT_SCORE = """You are an expert in architectural floor plans and building levels.

This image shows ONE floor plan from a multi-floor building. Your task is to rate how likely this specific floor plan is to be the BASEMENT (Keller / Untergeschoss / cellar).

Context:
- A basement is typically the lowest level, often partly or fully below ground.
- Typical basement indicators: few or no large windows; utility rooms (heating, storage); simpler layout; often rectangular/cellular rooms; possible parking or storage zones; no main entrance from street; sometimes smaller ceiling height suggested by layout density.
- Upper floors typically have: more/bigger windows, living rooms, main entrance, more open or complex layout.

Respond with ONLY a single integer between 1 and 100:
- 100 = very likely this is the basement (strong indicators: no large windows, utilities, storage-like layout).
- 1 = very unlikely (e.g. clearly upper floor with many windows, main living areas).
- 50 = uncertain or could be any floor.

Output format: one line containing only the number, e.g. 75"""


def _parse_score_from_response(text: str) -> int:
    """Extrage un întreg 1-100 din răspunsul modelului."""
    if not text:
        return 50
    text = text.strip()
    # Caută numere în text și ia primul în [1, 100]
    numbers = re.findall(r"\b([1-9][0-9]?|100)\b", text)
    for n in numbers:
        val = int(n)
        if 1 <= val <= 100:
            return val
    # Fallback: orice număr de 1-3 cifre
    any_num = re.search(r"\b(\d{1,3})\b", text)
    if any_num:
        val = int(any_num.group(1))
        return max(1, min(100, val))
    return 50


def score_plan_basement_likelihood(gemini_client: Any, image_path: Path) -> int:
    """Returnează un scor 1-100 pentru cât de probabil e ca planul să fie beci."""
    from segmenter.classifier import prep_for_vlm

    if gemini_client is None:
        return 50
    try:
        pil_img = prep_for_vlm(image_path)
        response = gemini_client.generate_content(
            [PROMPT_BASEMENT_SCORE, pil_img],
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 20,
            },
        )
        if response and response.parts:
            return _parse_score_from_response(response.text)
    except Exception as e:
        print(f"   ⚠️ [BasementScorer] Eroare pentru {image_path.name}: {e}")
    return 50


def run_basement_scoring(house_plans: List[Any]) -> int:
    """
    Pentru fiecare plan din house_plans (listă de obiecte cu .image_path),
    obține un scor Gemini 1-100 pentru probabilitatea că e beci.
    Returnează indexul (0-based) al planului cu scorul cel mai mare.
    """
    from segmenter.classifier import setup_gemini_client

    if not house_plans:
        return 0
    client = setup_gemini_client()
    if client is None:
        print("   ⚠️ [BasementScorer] Gemini indisponibil – aleg primul plan ca beci.")
        return 0

    scores: List[int] = []
    for i, plan in enumerate(house_plans):
        img_path = getattr(plan, "image_path", None)
        if img_path is None or not Path(img_path).exists():
            scores.append(50)
            print(f"   [BasementScorer] Plan {i + 1}: lipsă imagine → 50")
            continue
        score = score_plan_basement_likelihood(client, Path(img_path))
        scores.append(score)
        print(f"   [BasementScorer] Plan {i + 1} ({Path(img_path).name}): {score}")

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    print(f"   ✅ [BasementScorer] Beci ales: plan index {best_idx} (scor {scores[best_idx]})")
    return best_idx
