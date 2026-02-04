# file: engine/count_objects/gemini_verification.py
from __future__ import annotations

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai

from .preprocessing import preprocess_for_ai
from .config import MAX_GEMINI_WORKERS


def _init_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing")
    genai.configure(api_key=api_key)
    for model_name in ("gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"):
        try:
            return genai.GenerativeModel(model_name)
        except Exception:
            continue
    raise RuntimeError("No Gemini flash model available")


def ask_gemini_comparison(gemini_model, reference_path: Path, candidate_path: Path, label: str, temp_dir: Path) -> bool:
    try:
        prompt = (
            f"You are an architectural expert. "
            f"Image 1 is a CONFIRMED {label} (reference). "
            f"Image 2 is a candidate. "
            f"Does Image 2 represent the SAME TYPE of architectural element as Image 1? "
            f"Ignore rotation. Strict checking."
            f"Return strict 'DA' or 'NU'."
        )
        
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
    if not candidates: return {}
    
    gemini_model = _init_gemini()
    results = {}
    
    def verify_one(cand):
        try:
            is_valid = ask_gemini_comparison(gemini_model, reference_path, cand["tmp_path"], cand["label"], temp_dir)
            return (cand["idx"], is_valid)
        except Exception as e:
            return (cand["idx"], False)
    
    with ThreadPoolExecutor(max_workers=MAX_GEMINI_WORKERS) as executor:
        futures = {executor.submit(verify_one, cand): cand for cand in candidates}
        for future in as_completed(futures):
            idx, is_valid = future.result()
            results[idx] = is_valid
    
    return results