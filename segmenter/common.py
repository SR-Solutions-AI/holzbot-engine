# file: engine/runner/segmenter/common.py
from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================================
# CONFIG SEGMENTER
# =========================================

# Logging de debug
DEBUG: bool = True

# DPI-uri cerute (segmenter încearcă în ordine până reușește)
REQUESTED_DPI = [900, 600, 450]

# Downsample final (dacă imaginea este randată la DPI mai mare)
DOWNSAMPLE_TARGET_DPI = 450

# Limită maximă pe latura imaginii în pixeli pentru randarea PDF-urilor
MAX_RENDER_DIM = 32750

# Latura maximă a planurilor exportate (crop-uri)
MAX_PLAN_EXPORT_LONG_EDGE = 4500  # px

# ======================================================
# Structură foldere – nume mai sugestive, fără "stepX"
# ======================================================

STEP_DIRS = {
    # imagini după eliminarea textului
    "text": "text_removed",

    # imagini după eliminarea hașurilor
    "hatch": "hatching_removed",

    # hărți de contururi (Canny)
    "outline": "edges",

    # masca liniilor groase
    "thick": "thick_lines",

    # pereți „solidificați”
    "solid": "solid_walls",

    # zone de pereți / interior-exterior
    "walls": "wall_zones",

    # pipeline de clustere (planuri brute)
    "clusters": {
        "root": "clusters",
        "initial": "clusters/clean_mask",          # masca clean folosită la connected components
        "split": "clusters/split_candidates",      # clustere mari care pot fi splituite
        "merged": "clusters/merged_boxes",         # după merge overlapped
        "expanded": "clusters/expanded_boxes",     # după expand_cluster
        "final": "clusters/annotated_preview",     # preview cu dreptunghiuri numerotate
        "crops": "clusters/plan_crops",            # AICI se salvează planurile crop-uite
    },

    # clasificare (OpenAI + heuristici) – le vei folosi în classifier.py
    "classified": {
        "root": "classified",
        "blueprints": "classified/blueprints",
        "side_views": "classified/side_views",
        "text": "classified/text",
        "siteplan": "classified/siteplan",
    },

    # rafinare blueprint-uri (inside-only + auto-crop) – pentru refiner.py
    "bp_refined": {
        "root": "blueprints_refined",
        "debug": "blueprints_refined/debug",
        "crops": "blueprints_refined/crops",
    },
}

# OUTPUT_DIR global al segmenter-ului (setat de reset_output_folders)
OUTPUT_DIR: Path = Path("segmenter_out")


def debug_print(msg: str) -> None:
    if DEBUG:
        print(msg)


def set_output_dir(output_dir: str | Path) -> None:
    global OUTPUT_DIR
    OUTPUT_DIR = Path(output_dir)


def get_output_dir() -> Path:
    return OUTPUT_DIR


def reset_output_folders(output_dir: str | Path) -> None:
    """
    Resetează complet structura de foldere pentru segmentare.
    """
    set_output_dir(output_dir)
    root = get_output_dir()

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    for _, v in STEP_DIRS.items():
        if isinstance(v, dict):
            for sub in v.values():
                (root / sub).mkdir(parents=True, exist_ok=True)
        else:
            (root / v).mkdir(parents=True, exist_ok=True)

    debug_print(f"🧹 Folderul de output '{root}' a fost resetat complet.\n")


def save_debug(img: np.ndarray, subfolder: str, name: str) -> None:
    """
    Salvează imagini de debug în subfolder relativ la OUTPUT_DIR.
    E folosit peste tot unde aveai save_debug înainte.
    """
    if not DEBUG:
        return

    root = get_output_dir()
    folder_path = root / subfolder
    folder_path.mkdir(parents=True, exist_ok=True)

    out_path = folder_path / name
    cv2.imwrite(str(out_path), img)
    debug_print(f"📸 Saved: {out_path}")


def safe_imread(path: str | Path) -> np.ndarray:
    """
    Citește robust o imagine (exact helper-ul tău).
    """
    arr = np.fromfile(str(Path(path)), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Nu s-a putut citi imaginea: {path}")
    return img


def resize_bgr_max_side(bgr: np.ndarray, max_side: int = MAX_PLAN_EXPORT_LONG_EDGE) -> np.ndarray:
    h, w = bgr.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return bgr
    scale = max_side / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    
    # 🟢 MODIFICARE: Folosim INTER_LANCZOS4 pentru a păstra detaliile fine (scrisul)
    debug_print(f"🔻 Resize plan: {(w, h)} -> {(new_w, new_h)} (max_side={max_side})")
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)