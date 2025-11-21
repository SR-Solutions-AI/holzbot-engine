# new/runner/detections/crop_scripts/crop_door.py
import json
from pathlib import Path
from PIL import Image


def crop_from_pred(img: Image.Image, pred: dict) -> Image.Image:
    """Extrage crop-ul pe baza predicției (format cxcywh)."""
    x, y = pred["x"], pred["y"]
    w, h = pred["width"], pred["height"]
    
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)
    
    # Clamp la dimensiunile imaginii
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    
    return img.crop((x1, y1, x2, y2))


def export_rotations(crop: Image.Image, out_dir: Path, base_name: str) -> int:
    """
    Exportă crop-ul cu toate rotațiile și flip-urile (data augmentation).
    Returns: numărul de imagini generate.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    angles = [0, 90, 180, 270]
    flips = [
        ("none", None),
        ("flipH", Image.FLIP_LEFT_RIGHT),
        ("flipV", Image.FLIP_TOP_BOTTOM),
        ("flipHV", "both"),
    ]
    
    count = 0
    for ang in angles:
        rot = crop.rotate(ang, expand=True)
        
        for flip_name, flip_mode in flips:
            if flip_mode == "both":
                img_out = rot.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
            elif flip_mode is not None:
                img_out = rot.transpose(flip_mode)
            else:
                img_out = rot
            
            out_path = out_dir / f"{base_name}_{ang}_{flip_name}.png"
            img_out.save(out_path)
            count += 1
    
    return count


def process(image_path: Path, json_path: Path, out_root: Path) -> int:
    """
    Procesează toate detecțiile de tip 'door' (fără 'double').
    Returns: numărul TOTAL de imagini generate (cu augmentări).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    predictions = [
        p for p in data.get("predictions", [])
        if "door" in str(p.get("class", "")).lower()
        and "double" not in str(p.get("class", "")).lower()
    ]
    
    if not predictions:
        return 0
    
    img = Image.open(image_path)
    out_dir = out_root / "door"
    
    total_count = 0
    for i, pred in enumerate(predictions):
        crop = crop_from_pred(img, pred)
        count = export_rotations(crop, out_dir, f"door_{i:03d}")
        total_count += count
    
    return total_count