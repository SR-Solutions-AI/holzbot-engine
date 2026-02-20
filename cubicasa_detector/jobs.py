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


def _get_weights_and_device():
    """Comun pentru phase1/phase2: găsește weights și device."""
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
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return weights_file, device


def run_cubicasa_phase1(
    plan_image: Path,
    output_dir: Path,
    gemini_api_key: str | None = None,
    raster_timings: list | None = None,
) -> dict:
    """
    Rulează doar faza 1 (Raster API + AI walls → 02_ai_walls_closed).
    Fiecare apel își încarcă propriul model/device (potrivit pentru execuție paralelă).
    raster_timings: listă mutabilă în care se adaugă (nume_pas, durată) pentru raportare timpi.
    """
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY lipsește!")
    weights_file, device = _get_weights_and_device()
    print(f"🤖 CubiCasa Phase 1: {plan_image.name} (model: {weights_file.name}, device: {device})")
    return run_cubicasa_detection(
        image_path=str(plan_image),
        model_weights_path=str(weights_file),
        output_dir=str(output_dir),
        gemini_api_key=gemini_api_key,
        device=device,
        save_debug_steps=True,
        run_phase=1,
        reused_model=None,
        reused_device=None,
        raster_timings=raster_timings,
    )


def run_cubicasa_phase2(
    output_dir: Path,
    gemini_api_key: str | None = None,
    raster_timings: list | None = None,
) -> dict:
    """
    Rulează doar faza 2 (brute force + crop + walls from coords + restul pipeline-ului).
    Presupune că phase1 a rulat deja pentru acest output_dir (există 02_ai_walls_closed.png etc).
    raster_timings: listă mutabilă în care se adaugă (nume_pas, durată) pentru raportare timpi.
    Returns:
        dict identic cu run_cubicasa_detection (scale_result, measurements, masks).
    """
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY lipsește!")
    weights_file, device = _get_weights_and_device()
    print(f"🤖 CubiCasa Phase 2: {output_dir} (continuare din 02_ai_walls_closed)")
    return run_cubicasa_detection(
        image_path="",  # nefolosit la phase 2
        model_weights_path=str(weights_file),
        output_dir=str(output_dir),
        gemini_api_key=gemini_api_key,
        device=device,
        save_debug_steps=True,
        run_phase=2,
        raster_timings=raster_timings,
    )