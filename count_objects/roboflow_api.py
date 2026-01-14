# new/runner/count_objects/roboflow_api.py
from __future__ import annotations

import time
import requests
from pathlib import Path

from .config import CONF_THRESHOLD, OVERLAP


def infer_roboflow(
    image_path: Path,
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    confidence: float = CONF_THRESHOLD,
    overlap: int = OVERLAP
) -> dict:
    """
    Apel Roboflow pentru detecții YOLO (modele cu versiune standard).
    Funcționează pentru TOATE modelele standard (doors/windows/stairs).
    """
    max_retries = 5
    timeout = 60
    conf_percent = int(confidence * 100)
    
    # 1) Încearcă infer.roboflow.com
    infer_url = f"https://infer.roboflow.com/{workspace}/{project}/{version}"
    infer_url += f"?confidence={conf_percent}&overlap={overlap}"
    
    headers = {
        "Authorization": f"Key {api_key}",
        "Accept": "application/json",
        "Content-Type": "image/jpeg",
    }
    
    img_bytes = image_path.read_bytes()
    
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(infer_url, headers=headers, data=img_bytes, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict) and "predictions" in data:
                    return data
                return {"predictions": data.get("predictions", [])}
            elif r.status_code in (401, 403, 404, 405):
                print(f"       [INFO] infer.roboflow.com → {r.status_code}, trying detect")
                break
            else:
                print(f"       [WARN] infer {r.status_code}")
        except Exception as e:
            print(f"       [ERR] infer attempt {attempt}: {e}")
        time.sleep(1.2)
    
    # 2) Fallback pe detect.roboflow.com
    detect_url = f"https://detect.roboflow.com/{project}/{version}"
    params = {"api_key": api_key, "confidence": conf_percent, "overlap": overlap}
    
    for attempt in range(1, max_retries + 1):
        try:
            with open(image_path, "rb") as f:
                files = {"file": (image_path.name, f, "image/jpeg")}
                r = requests.post(detect_url, params=params, files=files, timeout=timeout)
            
            if r.status_code == 200:
                return {"predictions": r.json().get("predictions", [])}
        except Exception as e:
            print(f"       [ERR] detect attempt {attempt}: {e}")
        time.sleep(1.2)
    
    raise RuntimeError("Failed to get predictions from Roboflow")