#!/usr/bin/env python3
"""
Generează o imagine compusă: fiecare crop de cameră trimis la Gemini + labelul detectat.
Poate genera:
  1) PNG via PIL (dacă Pillow e instalat)
  2) HTML cu imaginile în grid (mereu, folosind path-uri relative din output/)
Usage: python scripts/gemini_crops_labels_image.py <run_id>
       python scripts/gemini_crops_labels_image.py ae488c37-11d8-4a30-81c3-1c0e997ade59
Output: output/<run_id>/scale/gemini_room_crops_with_labels.png (dacă PIL)
        output/<run_id>/scale/gemini_room_crops_with_labels.html (mereu)
"""
import base64
import json
import sys
from pathlib import Path


def load_plan_cells(output_root):
    """Returnează list of (plan_name, list of (path_to_crop, label))."""
    all_cells = []
    for plan_dir in sorted(output_root.iterdir()):
        if not plan_dir.is_dir() or not plan_dir.name.startswith("plan_"):
            continue
        wfc = plan_dir / "cubicasa_steps" / "raster_processing" / "walls_from_coords"
        room_scales_path = wfc / "room_scales.json"
        if not room_scales_path.exists():
            continue
        with open(room_scales_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rooms = data.get("rooms") or data.get("room_scales") or {}
        if not rooms:
            continue
        plan_cells = []
        for i in sorted(rooms.keys(), key=lambda x: int(x)):
            r = rooms[i]
            crop_name = r.get("crop_image") or f"room_{i}_crop.png"
            crop_path = wfc / crop_name
            if not crop_path.exists():
                crop_path = wfc / f"room_{i}_temp_for_gemini.png"
            label = r.get("room_type") or r.get("room_name") or f"Room {i}"
            if isinstance(label, str) and "(" in label:
                label = label.split("(")[0].strip() or label
            plan_cells.append((crop_path, label))
        if plan_cells:
            all_cells.append((plan_dir.name, plan_cells))
    return all_cells


def write_html(output_root, all_cells, run_id, embed_images=True):
    """Scrie HTML cu grid de crop-uri + label-uri.
    Dacă embed_images=True, imaginile sunt încorporate ca base64 (HTML funcționează în orice preview).
    Dacă False, folosește path-uri relative (funcționează doar când deschizi fișierul din folderul scale/).
    """
    out_path = output_root / "gemini_room_crops_with_labels.html"
    lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Gemini room crops – " + run_id + "</title>",
        "<style>",
        "body { font-family: sans-serif; background: #f5f5f5; padding: 16px; }",
        "h1 { font-size: 1.2rem; margin-bottom: 16px; }",
        ".plan { margin-bottom: 24px; }",
        ".plan h2 { font-size: 1rem; color: #333; margin-bottom: 8px; }",
        ".grid { display: flex; flex-wrap: wrap; gap: 12px; }",
        ".cell { width: 200px; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }",
        ".cell img { width: 200px; height: 140px; object-fit: contain; background: #fafafa; display: block; }",
        ".cell .label { padding: 8px; font-size: 13px; color: #111; text-align: center; }",
        "</style></head><body>",
        "<h1>Cropped rooms sent to Gemini and detected labels</h1>",
    ]
    for plan_name, cells in all_cells:
        lines.append(f"<div class='plan'><h2>{plan_name}</h2><div class='grid'>")
        for crop_path, label in cells:
            if embed_images and crop_path.exists():
                raw = crop_path.read_bytes()
                b64 = base64.b64encode(raw).decode("ascii")
                suffix = crop_path.suffix.lower()
                mime = "image/png" if suffix == ".png" else "image/jpeg"
                src = f"data:{mime};base64,{b64}"
            elif not embed_images:
                rel = crop_path.relative_to(output_root)
                src = rel.as_posix()
            else:
                src = "data:image/svg+xml," + base64.b64encode(
                    b'<svg xmlns="http://www.w3.org/2000/svg" width="200" height="140"><rect fill="#eee" width="200" height="140"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#999">lipsa</text></svg>'
                ).decode("ascii")
            # escape single quotes in label for HTML attr
            label_esc = label.replace("'", "&#39;")
            lines.append(
                f"<div class='cell'><img src='{src}' alt='' loading='lazy'/>"
                f"<div class='label'>{label_esc}</div></div>"
            )
        lines.append("</div></div>")
    lines.append("</body></html>")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def write_png(output_root, all_cells):
    """Generează PNG compus cu PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    cell_max_h = 140
    cell_max_w = 200
    label_h = 28
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
            font_title = font

    out_parts = []
    for plan_name, cells in all_cells:
        row_images = []
        for crop_path, label in cells:
            if not crop_path.exists():
                img = Image.new("RGB", (160, 120), color=(40, 40, 40))
                d = ImageDraw.Draw(img)
                d.text((10, 50), "lipsa", fill=(200, 200, 200))
            else:
                img = Image.open(crop_path).convert("RGB")
            w, h = img.size
            scale = min(cell_max_w / w, cell_max_h / h, 1.0)
            nw, nh = int(w * scale), int(h * scale)
            small = img.resize((nw, nh), Image.Resampling.LANCZOS)
            cell = Image.new("RGB", (cell_max_w, cell_max_h + label_h), color=(248, 248, 248))
            x0 = (cell_max_w - nw) // 2
            y0 = (cell_max_h - nh) // 2
            cell.paste(small, (x0, y0))
            d = ImageDraw.Draw(cell)
            d.text((cell_max_w // 2, cell_max_h + 8), label, fill=(0, 0, 0), font=font, anchor="mt")
            row_images.append(cell)
        row_w = sum(c.size[0] for c in row_images)
        row_h = row_images[0].size[1]
        row = Image.new("RGB", (row_w, row_h), color=(248, 248, 248))
        x = 0
        for c in row_images:
            row.paste(c, (x, 0))
            x += c.size[0]
        title_h = 36
        block = Image.new("RGB", (row_w, title_h + row_h), color=(220, 220, 220))
        dt = ImageDraw.Draw(block)
        dt.text((10, title_h // 2), plan_name, fill=(0, 0, 0), font=font_title, anchor="lm")
        block.paste(row, (0, title_h))
        out_parts.append(block)

    total_w = max(p.size[0] for p in out_parts)
    total_h = sum(p.size[1] for p in out_parts)
    out = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    y = 0
    for p in out_parts:
        out.paste(p, (0, y))
        y += p.size[1]
    out_path = output_root / "gemini_room_crops_with_labels.png"
    out.save(str(out_path))
    return out_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python gemini_crops_labels_image.py <run_id> [--no-embed]")
        sys.exit(1)
    run_id = sys.argv[1].strip()
    embed = "--no-embed" not in sys.argv
    base = Path(__file__).resolve().parent.parent
    output_root = base / "output" / run_id / "scale"
    if not output_root.exists():
        print(f"Nu există: {output_root}")
        sys.exit(1)

    all_cells = load_plan_cells(output_root)
    if not all_cells:
        print("Niciun plan cu room_scales.json / crop-uri găsit.")
        sys.exit(1)

    html_path = write_html(output_root, all_cells, run_id, embed_images=embed)
    print(f"Salvat HTML: {html_path}" + (" (imagini încorporate)" if embed else " (path-uri relative)"))

    png_path = write_png(output_root, all_cells)
    if png_path:
        print(f"Salvat PNG:  {png_path}")
    else:
        print("PNG nesalvat (instalează Pillow pentru PNG). Deschide fișierul HTML în browser.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
