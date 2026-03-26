# new/runner/cubicasa_detector/jobs.py
from __future__ import annotations

import os
import torch
from pathlib import Path

from .detector import run_cubicasa_detection
from .config import DEBUG


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
        image_path=str(plan_image),
        model_weights_path=str(weights_file),
        output_dir=str(output_dir),
        gemini_api_key=gemini_api_key,
        device=device,
        save_debug_steps=DEBUG,
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
    progress_callback: callable | None = None,
) -> dict:
    """
    Rulează doar faza 1 (Raster API + AI walls → 02_ai_walls_closed).
    raster_timings: listă mutabilă în care se adaugă (nume_pas, durată) pentru raportare timpi.
    progress_callback: opțional, apelat cu (sub_step: int) 0=start, 1=raster API done, 2=phase1 end.
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
        save_debug_steps=DEBUG,
        run_phase=1,
        reused_model=None,
        reused_device=None,
        raster_timings=raster_timings,
        progress_callback=progress_callback,
    )


def run_cubicasa_phase2(
    output_dir: Path,
    gemini_api_key: str | None = None,
    raster_timings: list | None = None,
    use_translation_only_raster: bool = True,
    reuse_cached_translation_only: bool = False,
    progress_callback: callable | None = None,
) -> dict:
    """
    Rulează doar faza 2 (brute force + crop + walls from coords + restul pipeline-ului).
    Presupune că phase1 a rulat deja pentru acest output_dir (există 02_ai_walls_closed.png etc).
    raster_timings: listă mutabilă în care se adaugă (nume_pas, durată) pentru raportare timpi.
    progress_callback: opțional, apelat cu (sub_step: int) 0=phase2 start, 1=brute done, 2=walls from coords done, 3=phase2 end.
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
        save_debug_steps=DEBUG,
        run_phase=2,
        raster_timings=raster_timings,
        use_translation_only_raster=use_translation_only_raster,
        reuse_cached_translation_only=reuse_cached_translation_only,
        progress_callback=progress_callback,
    )


def run_cubicasa_phase2_brute_only(
    output_dir: Path,
    gemini_api_key: str | None = None,
    raster_timings: list | None = None,
    no_cache: bool = False,
) -> dict:
    """
    Rulează doar brute force + overlay (fără crop, walls_from_coords, garaj, interior/exterior etc.).
    Presupune că phase1 a rulat deja (există 02_ai_walls_closed.png și raster/).
    Pentru script minimal: doar Raster API + pașii brute force.
    no_cache: dacă True, nu folosește brute_force_best_config.json; recalculează mereu.
    """
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY lipsește!")
    weights_file, device = _get_weights_and_device()
    print(f"🤖 CubiCasa Phase 2 (brute only): {output_dir}")
    return run_cubicasa_detection(
        image_path="",
        model_weights_path=str(weights_file),
        output_dir=str(output_dir),
        gemini_api_key=gemini_api_key,
        device=device,
        save_debug_steps=DEBUG,
        run_phase=3,  # doar brute force + overlay
        raster_timings=raster_timings,
        brute_force_no_cache=no_cache,
    )