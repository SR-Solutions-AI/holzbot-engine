# file: engine/segmenter/jobs.py
"""
Job runner: SEGMENTARE = doar fluxul Gemini Crop. Toți pașii vechi sunt DEZACTIVAȚI:
- nu se mai rulează: detect_wall_zones, detect_clusters, remove_hatched_areas, classifier rounds, etc.
Pentru fiecare pagină PDF/imagine: trimitem imaginea la Gemini → primim coordonate în procente
(box_2d: [ymin, xmin, ymax, xmax] în 0.0–1.0) + label (floor | side_view) → crop cu PIL → salvare
în classified/blueprints (etaje) și classified/side_views (vederi laterale / secțiuni).
"""
from __future__ import annotations

import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .common import set_output_dir, STEP_DIRS
from .pdf_utils import convert_pdf_to_png
from .gemini_crop import get_gemini_boxes_for_page, crop_and_save
from .classifier import ClassificationResult


@dataclass
class SegmentationJobResult:
    """Rezultatul procesării unui document în etapa de segmentare."""
    doc_id: str
    doc_path: Path
    work_dir: Path
    success: bool
    message: str
    total_clusters: int
    classification_results: List[ClassificationResult]


def _segment_single_page(
    page_path: Path,
    work_dir: Path,
    doc_id: str,
    doc_path: Path,
    task_idx: int,
    total_tasks: int,
) -> SegmentationJobResult:
    """
    Procesează O singură pagină (imagine) în propriul work_dir.
    Folosit pentru: o imagine standalone SAU o pagină dintr-un PDF (câte un folder src_*_page_001, src_*_page_002, ...).
    """
    try:
        print(f"\n[Gemini Crop] ({task_idx}/{total_tasks}) {doc_id}: {page_path.name}", flush=True)
        set_output_dir(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        for sub in (STEP_DIRS["classified"]["blueprints"], STEP_DIRS["classified"]["side_views"],
                    STEP_DIRS["classified"]["siteplan"], STEP_DIRS["classified"]["text"],
                    "solid_walls"):
            (work_dir / sub).mkdir(parents=True, exist_ok=True)
        solid_preview = work_dir / "solid_walls" / "solidified.jpg"
        try:
            shutil.copy(str(page_path), str(solid_preview))
        except Exception:
            pass

        boxes = get_gemini_boxes_for_page(page_path)
        print(f"  [Gemini Crop] {doc_id}: {len(boxes)} zone (etaje + side views)", flush=True)
        classification_results = crop_and_save(page_path, work_dir, boxes)
        n_bp = len([r for r in classification_results if r.label == "house_blueprint"])
        n_sv = len([r for r in classification_results if r.label == "side_view"])
        print(f"  [Gemini Crop] {doc_id} → blueprints: {n_bp}, side_views: {n_sv}", flush=True)

        return SegmentationJobResult(
            doc_id=doc_id,
            doc_path=doc_path,
            work_dir=work_dir,
            success=True,
            message=f"Gemini Crop: {len(boxes)} zone → {len(classification_results)} planuri",
            total_clusters=len(classification_results),
            classification_results=classification_results,
        )
    except Exception as e:
        error_msg = f"Error processing {page_path.name}: {e}"
        print(f"  [Segmenter] ❌ {doc_id}: {error_msg}", flush=True)
        traceback.print_exc()
        return SegmentationJobResult(
            doc_id=doc_id,
            doc_path=doc_path,
            work_dir=work_dir,
            success=False,
            message=error_msg,
            total_clusters=0,
            classification_results=[],
        )


def run_segmentation_for_documents(
    input_path: str | Path,
    output_base_dir: str | Path,
    max_workers: int | None = None
) -> List[SegmentationJobResult]:
    """
    Procesează documente: câte un work_dir (src_* sau src_*_page_001, _page_002, ...) per pagină.
    - Imagine unică → un folder: src_0_input_0
    - PDF cu N pagini → N foldere: src_0_input_0_page_001, src_0_input_0_page_002, ...
    """
    input_path = Path(input_path).resolve()
    output_base_dir = Path(output_base_dir).resolve()
    output_base_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        files_to_process = [
            f for f in input_path.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]
        files_to_process.sort(key=lambda f: f.name)
    else:
        files_to_process = [input_path]

    if not files_to_process:
        print("⚠️  [Segmenter] No input files found!")
        return []

    # Construim lista de task-uri: (page_path, work_dir, doc_id, doc_path)
    tasks: List[tuple] = []
    for idx, fpath in enumerate(files_to_process):
        if fpath.suffix.lower() == ".pdf":
            pages_dir = output_base_dir / f"src_{idx}_{fpath.stem}_pdf_pages"
            pages_dir.mkdir(parents=True, exist_ok=True)
            png_pages = convert_pdf_to_png(fpath, pages_dir)
            print(f"  [Segmenter] PDF {fpath.name} → {len(png_pages)} pagini → foldere src_{idx}_{fpath.stem}_page_001, ...", flush=True)
            for p in range(1, len(png_pages) + 1):
                page_path = Path(png_pages[p - 1])
                work_dir = output_base_dir / f"src_{idx}_{fpath.stem}_page_{p:03d}"
                work_dir.mkdir(parents=True, exist_ok=True)
                doc_id = f"src_{idx}_{fpath.stem}_page_{p:03d}"
                tasks.append((page_path, work_dir, doc_id, fpath))
        else:
            work_dir = output_base_dir / f"src_{idx}_{fpath.stem}"
            work_dir.mkdir(parents=True, exist_ok=True)
            doc_id = f"src_{idx}_{fpath.stem}"
            tasks.append((fpath, work_dir, doc_id, fpath))

    total_tasks = len(tasks)
    print(f"\n{'='*60}")
    print(f"[Gemini Crop] Starting: {total_tasks} pagini/fișiere (din {len(files_to_process)} documente)")
    print(f"{'='*60}\n")

    if max_workers is None:
        import os
        max_workers = min(total_tasks, os.cpu_count() or 4)

    results: List[SegmentationJobResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for task_idx, (page_path, work_dir, doc_id, doc_path) in enumerate(tasks, start=1):
            future = executor.submit(
                _segment_single_page,
                page_path,
                work_dir,
                doc_id,
                doc_path,
                task_idx,
                total_tasks,
            )
            futures[future] = (page_path, work_dir, doc_id, doc_path)

        for future in as_completed(futures):
            page_path, work_dir, doc_id, doc_path = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"❌ [Segmenter] Unexpected error for {doc_id}: {e}")
                traceback.print_exc()
                results.append(
                    SegmentationJobResult(
                        doc_id=doc_id,
                        doc_path=doc_path,
                        work_dir=work_dir,
                        success=False,
                        message=f"Unexpected error: {e}",
                        total_clusters=0,
                        classification_results=[],
                    )
                )

    successful = sum(1 for r in results if r.success)
    failed = total_tasks - successful
    total_clusters = sum(r.total_clusters for r in results)
    total_classified = sum(len(r.classification_results) for r in results)

    print(f"\n{'='*60}")
    print(f"[Gemini Crop] Complete")
    print(f"  ✅ Success: {successful}/{total_tasks}")
    if failed > 0:
        print(f"  ❌ Failed: {failed}/{total_tasks}")
    print(f"  📦 Total crops: {total_clusters}")
    print(f"  🏷️  Total classified: {total_classified}")
    print(f"{'='*60}\n")

    return results


def get_all_classification_results(
    segmentation_results: List[SegmentationJobResult]
) -> List[ClassificationResult]:
    """
    Helper pentru a extrage toate rezultatele de clasificare dintr-o listă de SegmentationJobResult.
    """
    all_results = []
    for seg_result in segmentation_results:
        all_results.extend(seg_result.classification_results)
    return all_results