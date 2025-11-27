from __future__ import annotations

import sys
import os

# --- FIX: Adăugăm directorul curent în sys.path pentru a permite importuri absolute ---
# Acest lucru rezolvă problema când scriptul este rulat din NestJS sau din alt folder.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import argparse
import time
from datetime import datetime

from config.settings import build_job_root, RUNS_ROOT, RUNNER_ROOT
from config.frontend_loader import load_frontend_data_for_run  # ✅ NOU
from segmenter import segment_document, classify_segmented_plans
from segmenter.classifier import ClassificationResult
from floor_classifier import run_floor_classification, FloorClassificationResult
from detections.jobs import run_detections_for_run
from scale import run_scale_detection_for_run
from count_objects import run_count_objects_for_run
from exterior_doors.jobs import run_exterior_doors_for_run
from measure_objects.jobs import run_measure_objects_for_run
from perimeter.jobs import run_perimeter_for_run
from area.jobs import run_area_for_run
from roof.jobs import run_roof_for_run
from pricing.jobs import run_pricing_for_run, PricingJobResult
from offer_builder import build_final_offer
from pdf_generator import generate_complete_offer_pdf


# =========================================================
# UI NOTIFICATION HELPER
# =========================================================
def notify_ui(stage_tag: str, image_path: Path | str | None = None):
    """
    Trimite un semnal către Backend (NestJS) care va fi propagat în Frontend (LiveFeed).
    Format protocol: >>> UI:STAGE:<tag>|IMG:<path>
    """
    msg = f">>> UI:STAGE:{stage_tag}"
    if image_path:
        abs_path = Path(image_path).resolve()
        if abs_path.exists():
            msg += f"|IMG:{str(abs_path)}"
    
    print(msg, flush=True)


# =========================================================
# TIMER UTILITIES
# =========================================================

class Timer:
    """Context manager pentru măsurarea timpului."""
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"\n⏱️ START: {self.step_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        print(f"✅ FINISH: {self.step_name} ({self.end_time - self.start_time:.2f}s)\n")
        return False

class PipelineTimer:
    def __init__(self): 
        self.steps = []
        self.total_start = None
        self.total_end = None
    
    def start(self): 
        self.total_start = time.time()
    
    def add_step(self, name, dur): 
        self.steps.append({"name": name, "duration": dur})
    
    def finish(self): 
        self.total_end = time.time()
    
    def print_summary(self): 
        pass

pipeline_timer = PipelineTimer()

@dataclass
class ClassifiedPlanInfo:
    """Plan clasificat cu tip (house_blueprint, site_plan, etc.)"""
    job_root: Path
    image_path: Path
    label: str

def _create_run_for_detections(job_root: Path, house_plans: list[ClassifiedPlanInfo]) -> str:
    run_id = job_root.name
    run_dir = RUNS_ROOT / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {"plans": [str(p.image_path) for p in house_plans]}
    (run_dir / "plans_list.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_id


# ✅ ȘTERS: _load_frontend_data() - folosim funcția centralizată


# =========================================================
# MAIN LOGIC
# =========================================================

def run_segmentation_for_document(input_path: str | Path, job_id: str | None = None):
    input_path = Path(input_path).resolve()
    job_root = build_job_root(job_id=job_id, prefix="segmentation_job")
    segmentation_out = job_root / "segmentation"
    segmentation_out.mkdir(parents=True, exist_ok=True)
    
    with Timer("1. SEGMENTATION") as t:
        plan_paths = segment_document(input_path, segmentation_out)
        
        # Căutăm solidified.jpg
        preview = segmentation_out / "solid_walls" / "solidified.jpg"
        if not preview.exists(): 
            preview = segmentation_out / "clusters" / "annotated_preview" / "final_clusters.jpg"
        
        notify_ui("segmentation", preview if preview.exists() else None)
        
    pipeline_timer.add_step("Segmentation", t.end_time - t.start_time)
    plans = [PlanInfo(job_root, Path(p).resolve()) for p in plan_paths]
    return job_root, plans

def run_segmentation_and_classification_for_document(
    input_path: str | Path, 
    job_id: str | None = None,
    frontend_data_json: str | None = None
):
    pipeline_timer.start()
    input_path = Path(input_path).resolve()
    job_root = build_job_root(job_id=job_id, prefix="segmentation_job")
    
    # =========================================================
    # 0. SAVE FRONTEND DATA (Dacă e trimis din Backend)
    # =========================================================
    if frontend_data_json:
        try:
            parsed_data = json.loads(frontend_data_json)
            out_file = job_root / "frontend_data.json"
            out_file.write_text(json.dumps(parsed_data, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"✅ Frontend data saved to {out_file}")
        except Exception as e:
            print(f"⚠️ Failed to save frontend data passed via args: {e}")

    # =========================================================
    # STEP 1 & 2: SEGMENTATION (afișăm text 1, poza, text 2)
    # =========================================================
    
    # ✅ Trimitem PRIMUL text ÎNAINTE de a începe procesarea
    notify_ui("segmentation_start")
    
    with Timer("STEP 1-2: Segmentation") as t:
        segmentation_out = job_root / "segmentation"
        segmentation_out.mkdir(parents=True, exist_ok=True)
        
        segment_document(input_path, segmentation_out)
        
        # Trimitem notificare pentru segmentation (POZA + text 2)
        solidified_img = segmentation_out / "solid_walls" / "solidified.jpg"
        if solidified_img.exists():
            notify_ui("segmentation", solidified_img)
        else:
            preview = segmentation_out / "clusters" / "annotated_preview" / "final_clusters.jpg"
            notify_ui("segmentation", preview if preview.exists() else None)
        
    pipeline_timer.add_step("Segmentation", t.end_time - t.start_time)

    # =========================================================
    # STEP 3: CLASSIFICATION (text 3 + toate pozele din blueprints)
    # =========================================================
    with Timer("STEP 3: Classification") as t:
        cls_results = classify_segmented_plans(segmentation_out)
        
        # Trimitem notificare pentru classification + pozele
        bp_dir = segmentation_out / "classified" / "blueprints"
        if bp_dir.exists():
            for img in bp_dir.glob("*.*"):
                notify_ui("classification", img)
        else:
            notify_ui("classification")
        
    pipeline_timer.add_step("Classification", t.end_time - t.start_time)

    plans = [ClassifiedPlanInfo(job_root, r.image_path, r.label) for r in cls_results]

    # =========================================================
    # STEP 4: FLOOR CLASSIFICATION (text 4)
    # =========================================================
    with Timer("STEP 4: Floor Classification") as t:
        floor_results = run_floor_classification(job_root, plans)
        notify_ui("floor_classification")
    pipeline_timer.add_step("Floor Classification", t.end_time - t.start_time)

    house_plans = [p for p in plans if p.label == "house_blueprint"]
    
    if house_plans:
        run_id = _create_run_for_detections(job_root, house_plans)
        
        # =========================================================
        # STEP 5: DETECTIONS (text 5)
        # =========================================================
        with Timer("STEP 5: Detections") as t:
            run_detections_for_run(run_id)
            notify_ui("detections")
        pipeline_timer.add_step("Detections", t.end_time - t.start_time)
        
        # =========================================================
        # STEP 6: SCALE (text 6)
        # =========================================================
        with Timer("STEP 6: Scale") as t:
            run_scale_detection_for_run(run_id)
            notify_ui("scale")
        pipeline_timer.add_step("Scale", t.end_time - t.start_time)

        # =========================================================
        # STEP 7: COUNT OBJECTS (text 7 + plan_detected_all_hybrid.jpg)
        # =========================================================
        with Timer("STEP 7: Count Objects") as t:
            run_count_objects_for_run(run_id)
            
            out_dir = RUNNER_ROOT / "output" / run_id / "count_objects"
            if out_dir.exists():
                for img in out_dir.rglob("*plan_detected_all_hybrid.jpg"):
                    notify_ui("count_objects", img)
            else:
                notify_ui("count_objects")
        pipeline_timer.add_step("Count Objects", t.end_time - t.start_time)

        # =========================================================
        # STEP 8 & 9: EXTERIOR DOORS (text 8 + blue_overlay.jpg + text 9)
        # =========================================================
        with Timer("STEP 8-9: Exterior Doors") as t:
            run_exterior_doors_for_run(run_id)
            
            out_dir = RUNNER_ROOT / "output" / run_id / "exterior_doors"
            if out_dir.exists():
                for img in out_dir.rglob("*blue_overlay.jpg"):
                    notify_ui("exterior_doors", img)
            else:
                notify_ui("exterior_doors")
        pipeline_timer.add_step("Exterior Doors", t.end_time - t.start_time)

        # =========================================================
        # STEP 10: MEASURE (text 10)
        # =========================================================
        with Timer("STEP 10: Measure") as t:
            run_measure_objects_for_run(run_id)
            notify_ui("measure_objects")
        pipeline_timer.add_step("Measure", t.end_time - t.start_time)

        # =========================================================
        # STEP 11: PERIMETER (text 11)
        # =========================================================
        with Timer("STEP 11: Perimeter") as t:
            run_perimeter_for_run(run_id)
            notify_ui("perimeter")
        pipeline_timer.add_step("Perimeter", t.end_time - t.start_time)

        # =========================================================
        # STEP 12: AREA (text 12)
        # =========================================================
        with Timer("STEP 12: Area") as t:
            run_area_for_run(run_id)
            notify_ui("area")
        pipeline_timer.add_step("Area", t.end_time - t.start_time)

        # =========================================================
        # STEP 13: ROOF (text 13)
        # =========================================================
        with Timer("STEP 13: Roof") as t:
            run_roof_for_run(run_id)
            notify_ui("roof")
        pipeline_timer.add_step("Roof", t.end_time - t.start_time)

        # =========================================================
        # STEP 14 & 15: PRICING (text 14 + text 15)
        # =========================================================
        # ✅ CORECTAT: Aici se face load combinat (Fallback + Job Specific)
        frontend_data = load_frontend_data_for_run(run_id, job_root)
        
        with Timer("STEP 14-15: Pricing") as t:
            pricing_results = run_pricing_for_run(
                run_id, 
                frontend_data_override=frontend_data
            )
            notify_ui("pricing")
        pipeline_timer.add_step("Pricing", t.end_time - t.start_time)

        # =========================================================
        # STEP 16 & 17: OFFER GENERATION (text 16 + text 17)
        # =========================================================
        with Timer("STEP 16-17: Offer Generation") as t:
            # ✅ CITEȘTE din frontend_data (fără default hardcodat)
            nivel_oferta = frontend_data.get("materialeFinisaj", {}).get("nivelOferta")
            
            # Dacă lipsește (pentru siguranță), pune un default
            if not nivel_oferta:
                nivel_oferta = "Structură"
                print(f"⚠️ nivelOferta lipsește, folosesc default: {nivel_oferta}")
            
            print(f"📋 [OFFER] Building offers with level: {nivel_oferta}")
            
            for res in pricing_results:
                if res.success and res.result_data:
                    try:
                        build_final_offer(
                            res.result_data, 
                            nivel_oferta,  # ✅ Valoarea citită din JSON
                            res.work_dir / "final_offer.json"
                        )
                    except Exception as e:
                        print(f"⚠️ Error building offer for {res.plan_id}: {e}")
            
            notify_ui("offer_generation")
        pipeline_timer.add_step("Offer Generation", t.end_time - t.start_time)
        
        # =========================================================
        # STEP 18 & 19: PDF GENERATION (text 18 + text 19 + PDF PATH)
        # =========================================================
        with Timer("STEP 18-19: PDF") as t:
            pdf_path = None
            try:
                pdf_path = generate_complete_offer_pdf(run_id=run_id, output_path=None)
                notify_ui("pdf_generation")
            except Exception as e:
                print(f"⚠️ PDF Error: {e}")
                notify_ui("pdf_generation")
        pipeline_timer.add_step("PDF", t.end_time - t.start_time)
        
        # =========================================================
        # FINAL: Semnalează că TOTUL s-a terminat
        # =========================================================
        time.sleep(2.0)
        notify_ui("computation_complete", pdf_path if pdf_path and Path(pdf_path).exists() else None)

    pipeline_timer.finish()
    return job_root, plans, floor_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--no-classification", action="store_true")
    
    args = parser.parse_args()
    
    frontend_data_json = os.environ.get('FRONTEND_DATA_JSON')
    
    if frontend_data_json:
        print(f"✅ Frontend data received from ENV ({len(frontend_data_json)} chars)")
    else:
        print("⚠️  No frontend data in ENV")
    
    if args.no_classification:
        run_segmentation_for_document(args.input, job_id=args.job_id)
    else:
        run_segmentation_and_classification_for_document(
            args.input, 
            job_id=args.job_id,
            frontend_data_json=frontend_data_json
        )