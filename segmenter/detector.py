# file: engine/runner/segmenter/detector.py
from __future__ import annotations

from pathlib import Path

from .common import reset_output_folders, safe_imread
from .pdf_utils import convert_pdf_to_png
from .preprocess import (
    remove_text_regions,
    remove_hatched_areas,
    detect_outlines,
    filter_thick_lines,
    solidify_walls,
)
from .clusters import detect_wall_zones


def segment_page_image(page_path: str | Path) -> list[str]:
    """
    Rulează pipeline-ul de segmentare pe O singură imagine (pagini deja în PNG).
    RETURN: listă de path-uri (str) către planurile decupate.
    """
    page_path = Path(page_path)
    print(f"\n🖼 Procesare imagine pagină: {page_path}")
    img = safe_imread(page_path)

    no_text = remove_text_regions(img)
    gray = cv2.cvtColor(no_text, cv2.COLOR_BGR2GRAY)
    no_hatch = remove_hatched_areas(gray)
    outlines = detect_outlines(no_hatch)
    thick = filter_thick_lines(outlines)
    solid = solidify_walls(thick)
    crop_paths = detect_wall_zones(img, solid)
    print("🏁 Procesare pagină completă!\n")
    return crop_paths


# trebuie importat cv2 aici pentru segment_page_image
import cv2  # noqa: E402


def segment_document(input_path: str | Path, output_dir: str | Path) -> list[str]:
    """
    input_path poate fi:
      - path către o imagine (png/jpg/pdf)
      - path către un folder cu imagini/pdf

    output_dir:
      - folderul în care se vor crea TOATE subfolderele stepX_...
      - planurile vor fi în: <output_dir>/step7_clusters/crops

    return:
      - listă de path-uri (str) către TOATE planurile decupate
    """
    input_path = Path(input_path)

    # 1) pregătim OUTPUT_DIR pentru acest job
    reset_output_folders(output_dir)

    all_plan_paths: list[str] = []

    # 2) strângem fișierele de intrare
    if input_path.is_dir():
        files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf")
        ]
    else:
        files = [input_path]

    # 3) rulăm pipeline-ul pe fiecare fișier
    from .common import get_output_dir  # import local ca să evităm dependență circulară

    for f in files:
        ext = f.suffix.lower()
        if ext == ".pdf":
            pages_dir = get_output_dir() / "pdf_pages"
            png_pages = convert_pdf_to_png(f, pages_dir)
            for pth in png_pages:
                plan_paths = segment_page_image(pth)
                all_plan_paths.extend(plan_paths)
        else:
            plan_paths = segment_page_image(f)
            all_plan_paths.extend(plan_paths)

    print(f"📦 Total planuri detectate: {len(all_plan_paths)}")
    return all_plan_paths


# CLI simplu pentru test local:
if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Segmentare document în planuri (crop-uri)")
    parser.add_argument("input", help="Path către PDF sau imagine (sau folder cu mai multe)")
    parser.add_argument(
        "--output-dir",
        help="Folder de output pentru job (default: ./segmenter_out)",
        default="segmenter_out",
    )
    args = parser.parse_args()

    plans = segment_document(args.input, args.output_dir)
    print("\nPlanuri detectate:")
    for p in plans:
        print("  -", p)
