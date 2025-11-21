# new/runner/count_objects/stairs_detection.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .roboflow_api import infer_roboflow


def detect_stairs(image_path: Path, api_key: str, workspace: str, project: str, version: int) -> dict | None:
    """
    Detectează scări folosind modelul standard Roboflow.
    Returnează doar detecția cu cel mai mare confidence.
    """
    print(f"       [STAIRS] Detectare scări (model v{version})...")
    
    try:
        result = infer_roboflow(
            image_path,
            api_key,
            workspace,
            project,
            version,
            confidence=0.1
        )
        
        preds = result.get("predictions", [])
        if not preds:
            print(f"       [STAIRS] Nicio scară detectată")
            return None
        
        # Sortează după confidence și ia prima
        best = max(preds, key=lambda p: float(p.get("confidence", 0.0)))
        conf = float(best.get("confidence", 0.0))
        
        print(f"       [STAIRS] ✅ Scară detectată (confidence: {conf:.2f})")
        return best
    
    except Exception as e:
        print(f"       [STAIRS] ⚠️ Eroare: {e}")
        return None


def process_stairs(plan_image: Path, api_key: str) -> Tuple[dict | None, dict]:
    """
    Detectează scara cu cel mai mare confidence.
    
    Returns:
        (stairs_bbox_dict or None, stairs_export_dict)
    """
    from .config import (
        ROBOFLOW_STAIRS_PROJECT,
        ROBOFLOW_STAIRS_VERSION,
        ROBOFLOW_STAIRS_WORKSPACE
    )
    
    stairs_pred = detect_stairs(
        plan_image,
        api_key,
        ROBOFLOW_STAIRS_WORKSPACE,
        ROBOFLOW_STAIRS_PROJECT,
        ROBOFLOW_STAIRS_VERSION
    )
    
    if not stairs_pred:
        return None, {}
    
    x = int(stairs_pred.get("x", 0))
    y = int(stairs_pred.get("y", 0))
    w = int(stairs_pred.get("width", 0))
    h = int(stairs_pred.get("height", 0))
    conf = float(stairs_pred.get("confidence", 0.0))
    
    x1 = max(0, x - w//2)
    y1 = max(0, y - h//2)
    x2 = x + w//2
    y2 = y + h//2
    
    stairs_bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    
    stairs_export = {
        "type": "stairs",
        "status": "confirmed",
        "confidence": round(conf, 3),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2
    }
    
    return stairs_bbox, stairs_export