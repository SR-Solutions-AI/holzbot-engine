# utils.py - Funcții helper pentru formatare și I/O
from __future__ import annotations
import json
from pathlib import Path

def format_money(value: float | int | None, currency: str = "EUR") -> str:
    """Formatează o sumă de bani cu separatori corecți"""
    if value is None:
        return "—"
    try:
        v = float(value)
        # Format: 1.234,56 EUR
        formatted = f"{v:,.2f}".replace(",", " ").replace(".", ",").replace(" ", ".")
        return f"{formatted} {currency}"
    except Exception:
        return "—"

def format_area(value: float | int | None) -> str:
    """Formatează o suprafață în m²"""
    if value is None:
        return "—"
    try:
        v = float(value)
        return f"{v:,.2f}".replace(",", " ").replace(".", ",").replace(" ", ".") + " m²"
    except Exception:
        return "—"

def format_length(value: float | int | None) -> str:
    """Formatează o lungime în m"""
    if value is None:
        return "—"
    try:
        v = float(value)
        return f"{v:,.2f}".replace(",", " ").replace(".", ",").replace(" ", ".") + " m"
    except Exception:
        return "—"

def safe_get(data: dict, *keys, default=None):
    """Obține o valoare din dict nested în siguranță"""
    cur = data
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur

def load_json_safe(path: Path) -> dict:
    """Încarcă un fișier JSON în siguranță, returnând dict gol dacă eșuează"""
    if not path or not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def get_plan_image_path(plan_id: str, base_dir: Path) -> Path | None:
    """Caută imaginea unui plan în mai multe locații posibile"""
    candidates = [
        base_dir / plan_id / "plan.jpg",
        base_dir / plan_id / "plan.png",
        base_dir / "classified" / "blueprints" / f"{plan_id}.jpg",
        base_dir / "segmentation" / "classified" / "blueprints" / f"{plan_id}.jpg",
        base_dir / "detections" / f"{plan_id}" / "plan.jpg",
    ]
    
    # Caută și în subfolderele de tip plan_X_cluster_Y
    if base_dir.exists():
        for item in base_dir.rglob("plan.jpg"):
            if plan_id.lower() in str(item.parent).lower():
                return item
    
    for c in candidates:
        if c.exists():
            return c
    
    return None