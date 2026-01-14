# new/runner/cubicasa_detector/jobs.py
from __future__ import annotations

import os
import torch
from pathlib import Path

from .detector import run_cubicasa_detection


def run_cubicasa_for_plan(
    plan_image: Path,
    output_dir: Path,
    gemini_api_key: str | None = None
) -> dict:
    """
    Wrapper pentru detector CubiCasa.
    
    Args:
        plan_image: Path către imaginea planului (jpg/png)
        output_dir: Directory unde se salvează rezultatele
        gemini_api_key: Optional API key pentru Gemini
        
    Returns:
        dict identic cu ce returnează run_cubicasa_detection:
        {
            "scale_result": {...},
            "measurements": {...},
            "masks": {...}
        }
    """
    
    # 1. GĂSEȘTE WEIGHTS
    weights_candidates = [
        Path(__file__).parent / "model_weights.pth",
        Path(__file__).parent.parent / "model_weights.pth",
        Path("model_weights.pth"),
    ]
    
    weights_file = None
    for candidate in weights_candidates:
        if candidate.exists():
            weights_file = candidate
            break
    
    if not weights_file:
        raise FileNotFoundError(
            "Nu găsesc model_weights.pth. Plasează-l în runner/cubicasa_detector/"
        )
    
    # 2. API KEY
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY lipsește!")
    
    # 3. DEVICE DETECTION
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"🤖 CubiCasa: Procesez {plan_image.name}")
    print(f"   Model: {weights_file.name}")
    print(f"   Device: {device}")
    
    # 4. RULEAZĂ DETECTION
    # ✅ CRITICAL: Trimite parametrii cu numele CORECT
    return run_cubicasa_detection(
        image_path=str(plan_image),           # ✅ image_path (NU plan_image)
        model_weights_path=str(weights_file), # ✅ model_weights_path (NU weights_file)
        output_dir=str(output_dir),           # ✅ output_dir
        gemini_api_key=gemini_api_key,        # ✅ gemini_api_key
        device=device,                        # ✅ device
        save_debug_steps=True                 # ✅ save_debug_steps
    )