# file: engine/cubicasa_detector/alignment_methods.py
"""
Algoritmi suplimentari de aliniere pentru planuri vs mască API:
- Log-Polar FFT (scale + rotație)
- Affine ECC (transformare afină iterativă)
- Coarse-to-Fine Brute Force (IoU pyramid)
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any


def align_log_polar_fft(
    ref_binary: np.ndarray,
    api_binary: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """
    Detectează scara și rotația între referință și API folosind FFT în spațiul log-polar.
    ref = planul de referință (base), api = masca API (template).
    Returnează (scale_factor, angle_deg) sau None la eroare.
    """
    img_ref = ref_binary.astype(np.uint8)
    img_api = api_binary.astype(np.uint8)
    h, w = img_ref.shape
    img_api = cv2.resize(img_api, (w, h), interpolation=cv2.INTER_LINEAR)

    f_ref = np.fft.fft2(img_ref.astype(np.float32))
    f_api = np.fft.fft2(img_api.astype(np.float32))
    f_ref_shift = np.fft.fftshift(f_ref)
    f_api_shift = np.fft.fftshift(f_api)

    mag_ref = cv2.magnitude(f_ref_shift.real, f_ref_shift.imag)
    mag_api = cv2.magnitude(f_api_shift.real, f_api_shift.imag)

    center = (w // 2, h // 2)
    max_radius = float(np.sqrt(center[0] ** 2 + center[1] ** 2))
    if max_radius < 1:
        return None
    flags = cv2.WARP_POLAR_LOG | cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS

    try:
        lp_ref = cv2.warpPolar(mag_ref, (w, h), center, max_radius, flags)
        lp_api = cv2.warpPolar(mag_api, (w, h), center, max_radius, flags)
    except cv2.error:
        return None

    try:
        shift, _ = cv2.phaseCorrelate(
            lp_ref.astype(np.float32), lp_api.astype(np.float32)
        )
    except cv2.error:
        return None

    # shift[0] = dx în log-polar (corelat cu log(scale)), shift[1] = dy (unghi)
    scale_factor = np.exp(shift[0] * np.log(max_radius) / w) if w else 1.0
    angle = shift[1] * 360.0 / h if h else 0.0
    return (float(scale_factor), float(angle))


def align_affine_ecc(
    ref_binary: np.ndarray,
    api_binary: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Calculează transformarea afină care aliniază api la ref (Enhanced Correlation Coefficient).
    Returnează matricea 2x3 (warp) sau None la eșec.
    """
    img_ref = ref_binary.astype(np.uint8)
    img_api = api_binary.astype(np.uint8)
    h, w = img_ref.shape
    img_api = cv2.resize(img_api, (w, h), interpolation=cv2.INTER_LINEAR)

    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-7)

    try:
        _, warp_matrix = cv2.findTransformECC(
            img_ref, img_api, warp_matrix, warp_mode, criteria
        )
        return warp_matrix
    except cv2.error:
        return None


def align_brute_force_pyramid(
    ref_binary: np.ndarray,
    api_binary: np.ndarray,
) -> Tuple[float, int, int, float]:
    """
    Coarse-to-fine piramidă: multe scale-uri la rezoluție redusă, apoi rafinare scară + poziție la full res.
    ref = base, api = template plasat pe ref.
    Returnează (scale, tx, ty, iou).
    """
    _, ref_full = cv2.threshold(ref_binary, 127, 255, cv2.THRESH_BINARY)
    _, api_full = cv2.threshold(api_binary, 127, 255, cv2.THRESH_BINARY)

    small_factor = 0.05
    ref_small = cv2.resize(
        ref_full, None, fx=small_factor, fy=small_factor, interpolation=cv2.INTER_AREA
    )
    h_ref, w_ref = ref_full.shape
    h_api, w_api = api_full.shape

    best_iou = 0.0
    best_scale = 1.0
    best_tx, best_ty = 0, 0

    # Coarse: scale-uri 0.1–10 (150 pași; 300 era prea lent)
    COARSE_SCALE_STEPS = 150
    for s in np.linspace(0.1, 10.0, COARSE_SCALE_STEPS):
        sw = int(w_api * s * small_factor)
        sh = int(h_api * s * small_factor)
        if sw <= 0 or sh <= 0:
            continue

        api_small_scaled = cv2.resize(
            api_full, (sw, sh), interpolation=cv2.INTER_NEAREST
        )
        if (
            api_small_scaled.shape[0] > ref_small.shape[0]
            or api_small_scaled.shape[1] > ref_small.shape[1]
        ):
            continue

        res = cv2.matchTemplate(ref_small, api_small_scaled, cv2.TM_CCORR_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        tx, ty = max_loc

        canvas = np.zeros_like(ref_small)
        y_end = min(ty + sh, ref_small.shape[0])
        x_end = min(tx + sw, ref_small.shape[1])
        canvas[ty:y_end, tx:x_end] = api_small_scaled[: y_end - ty, : x_end - tx]
        inter = np.logical_and(ref_small, canvas).sum()
        union = np.logical_or(ref_small, canvas).sum()
        iou = float(inter / union) if union > 0 else 0.0

        if iou > best_iou:
            best_iou = iou
            best_scale = float(s)
            best_tx, best_ty = tx, ty

    # Rafinare scară: în jurul best_scale, pas fin (ex. ±5% cu pas 0.2%)
    FINE_SCALE_RADIUS = 0.05
    FINE_SCALE_STEP = 0.002
    scale_lo = max(0.1, best_scale - FINE_SCALE_RADIUS)
    scale_hi = min(10.0, best_scale + FINE_SCALE_RADIUS)
    fine_scales = np.arange(scale_lo, scale_hi + FINE_SCALE_STEP / 2, FINE_SCALE_STEP)
    real_tx = int(best_tx / small_factor)
    real_ty = int(best_ty / small_factor)

    for s in fine_scales:
        sw = int(w_api * s * small_factor)
        sh = int(h_api * s * small_factor)
        if sw <= 0 or sh <= 0:
            continue
        api_small_scaled = cv2.resize(api_full, (sw, sh), interpolation=cv2.INTER_NEAREST)
        if (
            api_small_scaled.shape[0] > ref_small.shape[0]
            or api_small_scaled.shape[1] > ref_small.shape[1]
        ):
            continue
        res = cv2.matchTemplate(ref_small, api_small_scaled, cv2.TM_CCORR_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        tx, ty = max_loc
        canvas = np.zeros_like(ref_small)
        y_end = min(ty + sh, ref_small.shape[0])
        x_end = min(tx + sw, ref_small.shape[1])
        canvas[ty:y_end, tx:x_end] = api_small_scaled[: y_end - ty, : x_end - tx]
        inter = np.logical_and(ref_small, canvas).sum()
        union = np.logical_or(ref_small, canvas).sum()
        iou = float(inter / union) if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_scale = float(s)
            best_tx, best_ty = tx, ty
            real_tx = int(tx / small_factor)
            real_ty = int(ty / small_factor)

    # Fără rafinare la full res: rezultatul e (scale, tx, ty) din rezoluția redusă (5%), scalat la coordonate full
    return (best_scale, real_tx, real_ty, best_iou)


def build_config_from_log_polar(
    scale: float,
    angle_deg: float,
    ref_binary: np.ndarray,
    api_binary: np.ndarray,
    direction: str = "api_to_orig",
) -> Dict[str, Any]:
    """Construiește config (position, template_size, score) pentru Log-Polar: aplică scale+rotație pe api și îl centrează pe ref."""
    h_ref, w_ref = ref_binary.shape[:2]
    h_api, w_api = api_binary.shape[:2]
    tw = max(1, int(w_api * scale))
    th = max(1, int(h_api * scale))
    api_scaled = cv2.resize(api_binary, (tw, th), interpolation=cv2.INTER_NEAREST)
    if abs(angle_deg) > 0.01:
        M = cv2.getRotationMatrix2D((tw / 2, th / 2), angle_deg, 1.0)
        api_scaled = cv2.warpAffine(api_scaled, M, (tw, th))
    # Centrare pe ref
    x_pos = max(0, (w_ref - tw) // 2)
    y_pos = max(0, (h_ref - th) // 2)
    placed = np.zeros((h_ref, w_ref), dtype=np.uint8)
    y_end = min(y_pos + th, h_ref)
    x_end = min(x_pos + tw, w_ref)
    placed[y_pos:y_end, x_pos:x_end] = api_scaled[: y_end - y_pos, : x_end - x_pos]
    inter = np.logical_and(ref_binary, placed).sum()
    union = np.logical_or(ref_binary, placed).sum()
    score = float(inter / union) if union > 0 else 0.0
    return {
        "direction": direction,
        "scale": scale,
        "angle_deg": angle_deg,
        "position": (x_pos, y_pos),
        "template_size": (tw, th),
        "score": score,
        "method": "log_polar_fft",
    }


def build_config_from_ecc(
    warp_matrix: np.ndarray,
    ref_binary: np.ndarray,
    api_binary: np.ndarray,
    direction: str = "api_to_orig",
) -> Dict[str, Any]:
    """Construiește config și score pentru ECC: warp pe api, overlap cu ref."""
    h, w = ref_binary.shape[:2]
    api_warped = cv2.warpAffine(api_binary, warp_matrix, (w, h))
    inter = np.logical_and(ref_binary, api_warped).sum()
    union = np.logical_or(ref_binary, api_warped).sum()
    score = float(inter / union) if union > 0 else 0.0
    return {
        "direction": direction,
        "warp_matrix": warp_matrix.tolist(),
        "position": (0, 0),
        "template_size": (w, h),
        "score": score,
        "method": "affine_ecc",
    }


def build_config_from_pyramid(
    scale: float,
    tx: int,
    ty: int,
    iou: float,
    api_binary: np.ndarray,
    direction: str = "api_to_orig",
) -> Dict[str, Any]:
    """Construiește config pentru Coarse-to-Fine (scale, position, template_size, score=IoU)."""
    h_api, w_api = api_binary.shape[:2]
    tw = max(1, int(w_api * scale))
    th = max(1, int(h_api * scale))
    return {
        "direction": direction,
        "scale": scale,
        "position": (tx, ty),
        "template_size": (tw, th),
        "score": iou,
        "method": "coarse_to_fine_pyramid",
    }
