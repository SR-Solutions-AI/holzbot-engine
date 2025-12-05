# new/runner/count_objects/gemini_verification.py
from __future__ import annotations

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai

from .preprocessing import preprocess_for_ai
from .config import MAX_GEMINI_WORKERS


def _init_gemini():
    """Inițializează modelul Gemini."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing in environment")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash-exp")


def ask_gemini_comparison(gemini_model, reference_path: Path, candidate_path: Path, label: str, temp_dir: Path) -> bool:
    """
    Verifică dacă 'candidate' este același tip de obiect ca 'reference'.
    Reference = obiectul cu cel mai mare confidence găsit în plan.
    """
    try:
        prompt = (
            f"You are an expert architectural plan analyzer. "
            f"Image 1 is a CONFIRMED {label} from this specific floor plan (high confidence reference). "
            f"Image 2 is a CANDIDATE crop from the same plan that needs verification. "
            f"Task: Look at the visual style, line thickness, and geometry. "
            f"Does Image 2 represent the same type of architectural element ({label}) as Image 1? "
            f"Ignore rotation (it might be rotated). Ignore slight cropping differences. "
            f"Return strict 'DA' if it is the same object type, or 'NU' if it is noise/wall/text."
        )
        
        # Preprocesăm ambele imagini (contrast, resize)
        ref_proc = preprocess_for_ai(reference_path, temp_dir)
        cand_proc = preprocess_for_ai(candidate_path, temp_dir)
        
        response = gemini_model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": open(ref_proc, "rb").read()},
            {"mime_type": "image/jpeg", "data": open(cand_proc, "rb").read()},
        ])
        
        text = (response.text or "").strip().upper()
        return "DA" in text
    
    except Exception as e:
        print(f"       [Gemini ERR] {e}")
        return False


def verify_candidates_parallel(candidates: list[dict], reference_path: Path, temp_dir: Path) -> dict:
    """Verifică mai mulți candidați în paralel comparându-i cu referința."""
    if not candidates:
        return {}
    
    gemini_model = _init_gemini()
    results = {}
    
    def verify_one(cand):
        try:
            # Dacă avem referință din plan, o folosim. Altfel folosim template-ul generic.
            ref_to_use = reference_path
            
            is_valid = ask_gemini_comparison(
                gemini_model,
                ref_to_use,
                cand["tmp_path"],
                cand["label"],
                temp_dir
            )
            return (cand["idx"], is_valid)
        except Exception as e:
            print(f"       [ERR] Gemini #{cand['idx']}: {e}")
            return (cand["idx"], False)
    
    with ThreadPoolExecutor(max_workers=MAX_GEMINI_WORKERS) as executor:
        futures = {executor.submit(verify_one, cand): cand for cand in candidates}
        
        for future in as_completed(futures):
            idx, is_valid = future.result()
            results[idx] = is_valid
    
    return results