# new/runner/count_objects/preprocessing.py
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def preprocess_for_ai(img_path: Path, temp_dir: Path, size: int = 128) -> str:
    """Preprocesează imagine pentru comparație AI."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Imagine invalidă: {img_path}")
    
    img = cv2.equalizeHist(img)
    img = cv2.convertScaleAbs(img, alpha=1.4, beta=15)
    
    if np.mean(img > 200) < 0.3:
        img = cv2.bitwise_not(img)
    
    img = cv2.resize(img, (size, size))
    _, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    
    processed_path = temp_dir / f"proc_{img_path.name}"
    cv2.imwrite(str(processed_path), img)
    return str(processed_path)


def load_templates(root_dir: Path) -> list[dict]:
    """Încarcă template-uri și le rotește în 4 direcții."""
    templates = []
    if not root_dir.exists():
        return templates
    
    for f in root_dir.glob("*.png"):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        base = cv2.equalizeHist(img)
        
        for ang in [0, 90, 180, 270]:
            M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), ang, 1.0)
            rotated = cv2.warpAffine(base, M, (img.shape[1], img.shape[0]), borderValue=255)
            templates.append({"name": f.name, "image": rotated})
    
    return templates