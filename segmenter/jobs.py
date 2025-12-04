# file: engine/segmenter/jobs.py
"""
Job runner pentru segmentare și clasificare paralelă a documentelor.
✅ THREAD-SAFE: Fiecare document primește propriul OUTPUT_DIR izolat prin thread-local storage.
"""
from __future__ import annotations

import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2

from .common import reset_output_folders, safe_imread, set_output_dir, STEP_DIRS
from .pdf_utils import convert_pdf_to_png
from .preprocess import (
    remove_text_regions,
    remove_hatched_areas,
    detect_outlines,
    filter_thick_lines,
    solidify_walls,
)
from .clusters import detect_wall_zones
from .classifier import classify_segmented_plans, ClassificationResult


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


def _segment_single_document(
    doc_path: Path,
    work_dir: Path,
    doc_idx: int,
    total_docs: int
) -> SegmentationJobResult:
    """
    ✅ THREAD-SAFE: Procesează UN singur document în propriul OUTPUT_DIR izolat.
    
    Args:
        doc_path: Path către document (PDF sau imagine)
        work_dir: Directory unic pentru acest document (e.g., segmentation/src_0_input_0)
        doc_idx: Index document (pentru logging)
        total_docs: Total documente (pentru logging)
    
    Returns:
        SegmentationJobResult cu rezultatele segmentării și clasificării
    """
    doc_id = f"src_{doc_idx}_{doc_path.stem}"
    
    try:
        print(f"\n[Segmenter] ({doc_idx}/{total_docs}) Processing: {doc_path.name}", flush=True)
        
        # ✅ CRITICAL: Set OUTPUT_DIR thread-local IMEDIAT
        reset_output_folders(work_dir)
        set_output_dir(work_dir)
        
        # Conversie PDF → PNG dacă e necesar
        if doc_path.suffix.lower() == '.pdf':
            pages_dir = work_dir / "pdf_pages"
            pages_dir.mkdir(exist_ok=True)
            png_pages = convert_pdf_to_png(doc_path, pages_dir)
            page_paths = [Path(p) for p in png_pages]
        else:
            page_paths = [doc_path]
        
        all_crops = []
        
        # Procesează fiecare pagină
        for page_path in page_paths:
            print(f"  [Segmenter] Processing page: {page_path.name}", flush=True)
            
            # ✅ Re-confirm OUTPUT_DIR pentru siguranță (deși e thread-local)
            set_output_dir(work_dir)
            
            # Încarcă imaginea
            img = safe_imread(page_path)
            
            # Pipeline de procesare
            no_text = remove_text_regions(img)
            gray = cv2.cvtColor(no_text, cv2.COLOR_BGR2GRAY)
            no_hatch = remove_hatched_areas(gray)
            edges = detect_outlines(no_hatch)
            thick = filter_thick_lines(edges)
            solid = solidify_walls(thick)
            
            # Detectează clustere (planuri)
            crops = detect_wall_zones(img, solid)
            all_crops.extend(crops)
        
        # Verifică că crops-urile au fost salvate corect
        crops_dir = work_dir / STEP_DIRS["clusters"]["crops"]
        if crops_dir.exists():
            actual_crops = list(crops_dir.glob("*.jpg"))
            print(f"  [Segmenter] ✅ Saved {len(actual_crops)} clusters in {crops_dir}", flush=True)
        else:
            print(f"  [Segmenter] ⚠️  No crops directory created!", flush=True)
        
        print(f"  [Segmenter] Total clusters detected: {len(all_crops)}", flush=True)
        
        # Clasificare (thread-safe prin work_dir)
        set_output_dir(work_dir)
        classification_results = classify_segmented_plans(work_dir)
        
        # Debug: Arată unde au fost clasificate planurile
        bp_dir = work_dir / "classified" / "blueprints"
        sp_dir = work_dir / "classified" / "siteplan"
        sv_dir = work_dir / "classified" / "side_views"
        tx_dir = work_dir / "classified" / "text"
        
        print(f"  [Segmenter] Classification results:", flush=True)
        print(f"     blueprints: {len(list(bp_dir.glob('*.*'))) if bp_dir.exists() else 0}", flush=True)
        print(f"     siteplan: {len(list(sp_dir.glob('*.*'))) if sp_dir.exists() else 0}", flush=True)
        print(f"     side_views: {len(list(sv_dir.glob('*.*'))) if sv_dir.exists() else 0}", flush=True)
        print(f"     text: {len(list(tx_dir.glob('*.*'))) if tx_dir.exists() else 0}", flush=True)
        
        return SegmentationJobResult(
            doc_id=doc_id,
            doc_path=doc_path,
            work_dir=work_dir,
            success=True,
            message=f"Successfully processed {len(all_crops)} clusters, classified {len(classification_results)} plans",
            total_clusters=len(all_crops),
            classification_results=classification_results
        )
        
    except Exception as e:
        error_msg = f"Error processing {doc_path.name}: {e}"
        print(f"  [Segmenter] ❌ {error_msg}", flush=True)
        traceback.print_exc()
        
        return SegmentationJobResult(
            doc_id=doc_id,
            doc_path=doc_path,
            work_dir=work_dir,
            success=False,
            message=error_msg,
            total_clusters=0,
            classification_results=[]
        )


def run_segmentation_for_documents(
    input_path: str | Path,
    output_base_dir: str | Path,
    max_workers: int | None = None
) -> List[SegmentationJobResult]:
    """
    ✅ THREAD-SAFE: Procesează multiple documente în paralel cu OUTPUT_DIR izolat per thread.
    
    Args:
        input_path: Path către un document sau folder cu documente
        output_base_dir: Directory de bază pentru output (se vor crea subdirectoare per document)
        max_workers: Număr maxim de thread-uri (default: numărul de fișiere sau CPU count)
    
    Returns:
        Lista de SegmentationJobResult pentru fiecare document
    """
    input_path = Path(input_path).resolve()
    output_base_dir = Path(output_base_dir).resolve()
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Detectează fișierele de intrare
    if input_path.is_dir():
        files_to_process = [
            f for f in input_path.iterdir()
            if f.is_file() and not f.name.startswith('.')
        ]
        files_to_process.sort(key=lambda f: f.name)
    else:
        files_to_process = [input_path]
    
    if not files_to_process:
        print("⚠️  [Segmenter] No input files found!")
        return []
    
    print(f"\n{'='*60}")
    print(f"[Segmenter] Starting segmentation for {len(files_to_process)} document(s)")
    print(f"{'='*60}\n")
    
    # Determină numărul de workers
    if max_workers is None:
        import os
        max_workers = min(len(files_to_process), os.cpu_count() or 4)
    
    results: List[SegmentationJobResult] = []
    total = len(files_to_process)
    
    # Procesare paralelă cu thread-local storage
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        for idx, fpath in enumerate(files_to_process):
            # Creează work_dir unic pentru fiecare document
            work_dir = output_base_dir / f"src_{idx}_{fpath.stem}"
            work_dir.mkdir(parents=True, exist_ok=True)
            
            future = executor.submit(
                _segment_single_document,
                fpath,
                work_dir,
                idx + 1,
                total
            )
            futures[future] = fpath
        
        # Colectează rezultatele pe măsură ce se termină
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                fpath = futures[future]
                print(f"❌ [Segmenter] Unexpected error for {fpath.name}: {e}")
                traceback.print_exc()
                results.append(
                    SegmentationJobResult(
                        doc_id=f"src_X_{fpath.stem}",
                        doc_path=fpath,
                        work_dir=output_base_dir / f"src_X_{fpath.stem}",
                        success=False,
                        message=f"Unexpected error: {e}",
                        total_clusters=0,
                        classification_results=[]
                    )
                )
    
    # Summary
    successful = sum(1 for r in results if r.success)
    failed = total - successful
    total_clusters = sum(r.total_clusters for r in results)
    total_classified = sum(len(r.classification_results) for r in results)
    
    print(f"\n{'='*60}")
    print(f"[Segmenter] Segmentation Complete")
    print(f"  ✅ Success: {successful}/{total}")
    if failed > 0:
        print(f"  ❌ Failed: {failed}/{total}")
    print(f"  📦 Total clusters: {total_clusters}")
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