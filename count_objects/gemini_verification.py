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


def ask_gemini_single(gemini_model, template_path: Path, candidate_path: Path, label: str, temp_dir: Path) -> bool:
    """Verifică cu Gemini dacă obiectul candidat e același tip dar rotit."""
    try:
        prompt = (
            f"Ești expert în interpretarea planurilor arhitecturale 2D. "
            f"Prima imagine arată un {label} standard, drept (neînclinat). "
            f"A doua imagine este un extras dintr-un plan tehnic. "
            f"Determină dacă a doua imagine reprezintă același tip de obiect, "
            f"dar rotit față de orizontală/verticală (ex. 30–60°). "
            f"Răspunde strict 'DA' sau 'NU'."
        )
        
        temp_proc = preprocess_for_ai(template_path, temp_dir)
        cand_proc = preprocess_for_ai(candidate_path, temp_dir)
        
        response = gemini_model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": open(temp_proc, "rb").read()},
            {"mime_type": "image/jpeg", "data": open(cand_proc, "rb").read()},
        ])
        
        text = (response.text or "").strip().upper()
        return "DA" in text
    
    except Exception as e:
        print(f"       [Gemini ERR] {e}")
        return False


def verify_candidates_parallel(candidates: list[dict], template_path: Path, temp_dir: Path) -> dict:
    """Verifică mai mulți candidați în paralel cu Gemini."""
    if not candidates:
        return {}
    
    gemini_model = _init_gemini()
    results = {}
    
    def verify_one(cand):
        """Helper pentru verificare paralelă."""
        try:
            is_valid = ask_gemini_single(
                gemini_model,
                template_path,
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