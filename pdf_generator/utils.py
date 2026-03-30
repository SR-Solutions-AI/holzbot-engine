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
        # Raster/scale output (planuri procesate de pipeline)
        base_dir / "scale" / plan_id / "cubicasa_steps" / "raster" / "input_resized.jpg",
    ]
    
    # Caută și în subfolderele de tip plan_X_cluster_Y
    if base_dir.exists():
        for item in base_dir.rglob("plan.jpg"):
            if plan_id.lower() in str(item.parent).lower():
                return item
        for item in base_dir.rglob("input_resized.jpg"):
            if plan_id.lower() in str(item).lower():
                return item
    
    for c in candidates:
        if c.exists():
            return c
    
    return None


def resolve_plan_image_for_pdf(plan_image: Path, plan_id: str, output_root: Path, job_root: Path | None) -> Path | None:
    """
    Returnează path-ul imaginii planului pentru PDF. Folosește imaginea din output-ul
    pipeline-ului (scale/raster) ca sursă principală, astfel încât în ofertă apară
    întotdeauna planul procesat. Dacă nu există, încearcă plan_image și job_root.
    """
    # Sursă principală: imaginea din scale/raster (output pipeline) – astfel apare în PDF
    scale_path = output_root / "scale" / plan_id / "cubicasa_steps" / "raster" / "input_resized.jpg"
    if scale_path.exists():
        return scale_path
    if plan_image and plan_image.exists():
        return plan_image
    if job_root:
        p = get_plan_image_path(plan_id, job_root)
        if p:
            return p
    p = get_plan_image_path(plan_id, output_root)
    if p:
        return p
    return None


# Aceeași bază ca în raster_api._load_review_base_image + detections_review_base.png (LiveFeed / editor).
_EDITOR_BLUEPRINT_CANDIDATES = (
    "detections_review_base.png",
    "input_resized_no_filter.png",
    "input_resized.jpg",
    "raster_request.png",
)


def resolve_editor_blueprint_for_pdf(
    plan_image: Path,
    plan_id: str,
    output_root: Path,
    job_root: Path | None,
) -> Path | None:
    """
    Imaginea folosită în editorul de verificare (aceleași pixeli ca detections_review_base /
    input_resized_no_filter înainte de filtrul API). Fallback la resolve_plan_image_for_pdf.
    """
    raster_dir = output_root / "scale" / plan_id / "cubicasa_steps" / "raster"
    for name in _EDITOR_BLUEPRINT_CANDIDATES:
        p = raster_dir / name
        if p.exists():
            return p
    return resolve_plan_image_for_pdf(plan_image, plan_id, output_root, job_root)