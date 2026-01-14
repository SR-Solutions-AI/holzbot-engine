# file: engine/main.py
from __future__ import annotations

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import argparse
import time

from config.settings import build_job_root, RUNS_ROOT, RUNNER_ROOT
from config.frontend_loader import load_frontend_data_for_run

# ✅ IMPORT SEGMENTER JOBS
from segmenter.jobs import run_segmentation_for_documents, get_all_classification_results

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
from pdf_generator import generate_complete_offer_pdf, generate_admin_offer_pdf, generate_admin_calculation_method_pdf
from segmenter.pdf_utils import convert_pdf_to_png

# =========================================================
# UI NOTIFICATION
# =========================================================
def notify_ui(stage_tag: str, image_path: Path | str | None = None):
    """Trimite un semnal către Backend (NestJS) care va fi propagat în Frontend (LiveFeed)."""
    msg = f">>> UI:STAGE:{stage_tag}"
    if image_path:
        abs_path = Path(image_path).resolve()
        if abs_path.exists():
            msg += f"|IMG:{str(abs_path)}"
        else:
            print(f"⚠️ [UI] Image path not found: {abs_path}", flush=True)
    print(msg, flush=True)
    sys.stdout.flush()

# =========================================================
# TIMER UTILITIES
# =========================================================
class Timer:
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"\n⏱️  START: {self.step_name}", flush=True)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        print(f"✅ FINISH: {self.step_name} ({self.end_time - self.start_time:.2f}s)\n", flush=True)
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

pipeline_timer = PipelineTimer()

@dataclass
class ClassifiedPlanInfo:
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

# =========================================================
# MAIN LOGIC
# =========================================================

def run_segmentation_and_classification_for_document(
    input_path: str | Path, 
    job_id: str | None = None,
    frontend_data_json: str | None = None
):
    pipeline_timer.start()
    input_path = Path(input_path).resolve()
    job_root = build_job_root(job_id=job_id, prefix="segmentation_job")
    
    # Save frontend data
    if frontend_data_json:
        try:
            parsed_data = json.loads(frontend_data_json)
            out_file = job_root / "frontend_data.json"
            out_file.write_text(json.dumps(parsed_data, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"✅ Frontend data saved to {out_file}")
        except Exception as e:
            print(f"⚠️ Failed to save frontend data: {e}")

    notify_ui("segmentation_start")
    
    # Preview pentru UI (Input files)
    preview_dir = job_root / "_previews"
    preview_dir.mkdir(exist_ok=True)
    
    # Detectează fișierele de intrare
    if input_path.is_dir():
        files_to_process = [f for f in input_path.iterdir() if f.is_file() and not f.name.startswith('.')]
        files_to_process.sort(key=lambda f: f.name)
    else:
        files_to_process = [input_path]
    
    for f in files_to_process:
        file_to_show = f
        if f.suffix.lower() == '.pdf':
            try:
                pngs = convert_pdf_to_png(f, preview_dir)
                if pngs:
                    file_to_show = Path(pngs[0])
            except Exception as e:
                print(f"⚠️ Preview error: {e}")
        notify_ui("segmentation_start", file_to_show)

    segmentation_out_base = job_root / "segmentation"
    segmentation_out_base.mkdir(parents=True, exist_ok=True)
    
    # =========================================================
    # SEGMENTATION & CLASSIFICATION (folosind jobs.py)
    # =========================================================
    with Timer("STEP 1-3: Multi-file Segmentation & Classification") as t:
        seg_results = run_segmentation_for_documents(
            input_path=input_path,
            output_base_dir=segmentation_out_base,
            max_workers=None  # auto-detect
        )
        
        # -------------------------------------------------------------
        # ✅ 1. UI: SOLID WALLS FIRST (Robust Check)
        # -------------------------------------------------------------
        print("📢 [UI] Sending Solid Walls previews (Priority 1)...")
        for i, seg_result in enumerate(seg_results):
            # Calea standard generată de segmenter
            standard_solid_img = seg_result.work_dir / "solid_walls" / "solidified.jpg"
            
            sent = False
            if standard_solid_img.exists():
                notify_ui("segmentation", standard_solid_img)
                sent = True
            else:
                # Fallback: Caută orice JPG în folder (pentru pagini multiple sau naming diferit)
                solid_dir = seg_result.work_dir / "solid_walls"
                if solid_dir.exists():
                    found_imgs = list(solid_dir.glob("*.jpg"))
                    for img in found_imgs:
                        notify_ui("segmentation", img)
                        sent = True
            
            if not sent:
                print(f"⚠️ [UI] Nu am găsit solid walls pentru doc {i}: {seg_result.doc_id}", flush=True)
        
        # -------------------------------------------------------------
        # ✅ 2. UI: CLASSIFICATION -> DOAR BLUEPRINTS
        # -------------------------------------------------------------
        print("📢 [UI] Sending Classified BLUEPRINTS ONLY...")
        for seg_result in seg_results:
            bp_dir = seg_result.work_dir / "classified" / "blueprints"
            
            # Iterăm EXCLUSIV folderul blueprints
            if bp_dir.exists():
                for img in bp_dir.glob("*.*"):
                    notify_ui("classification", img)

    pipeline_timer.add_step("Multi-Segmentation", t.end_time - t.start_time)

    # Colectăm rezultatele
    all_cls_results = get_all_classification_results(seg_results)
    plans = [ClassifiedPlanInfo(job_root, r.image_path, r.label) for r in all_cls_results]
    
    # Debug output
    print(f"\n📊 CLASSIFICATION SUMMARY:")
    print(f"   Total plans detected: {len(plans)}")
    from collections import Counter
    label_counts = Counter(p.label for p in plans)
    for label, count in label_counts.items():
        print(f"   {label}: {count}")

    # =========================================================
    # REST OF PIPELINE
    # =========================================================
    
    with Timer("STEP 4: Floor Classification") as t:
        floor_results = run_floor_classification(job_root, plans)
        notify_ui("floor_classification")
    pipeline_timer.add_step("Floor Classification", t.end_time - t.start_time)

    house_plans = [p for p in plans if p.label == "house_blueprint"]
    
    print(f"\n🏠 HOUSE PLANS ONLY: {len(house_plans)}/{len(plans)}")
    if len(house_plans) < len(plans):
        print(f"⚠️  {len(plans) - len(house_plans)} plans EXCLUDED (not house_blueprint)\n")
    
    if house_plans:
        run_id = _create_run_for_detections(job_root, house_plans)
        
        with Timer("STEP 5: Detections") as t:
            run_detections_for_run(run_id)
            notify_ui("detections")
        pipeline_timer.add_step("Detections", t.end_time - t.start_time)
        
        with Timer("STEP 6: Scale") as t:
            run_scale_detection_for_run(run_id)
            notify_ui("scale")
        pipeline_timer.add_step("Scale", t.end_time - t.start_time)

        with Timer("STEP 7: Count Objects") as t:
            run_count_objects_for_run(run_id)
            out_dir = RUNNER_ROOT / "output" / run_id / "count_objects"
            if out_dir.exists():
                for img in out_dir.rglob("*plan_detected_all_hybrid.jpg"):
                    notify_ui("count_objects", img)
            else:
                notify_ui("count_objects")
        pipeline_timer.add_step("Count Objects", t.end_time - t.start_time)

        with Timer("STEP 8-9: Exterior Doors") as t:
            run_exterior_doors_for_run(run_id)
            out_dir = RUNNER_ROOT / "output" / run_id / "exterior_doors"
            if out_dir.exists():
                for img in out_dir.rglob("*blue_overlay.jpg"):
                    notify_ui("exterior_doors", img)
            else:
                notify_ui("exterior_doors")
        pipeline_timer.add_step("Exterior Doors", t.end_time - t.start_time)

        with Timer("STEP 10: Measure") as t:
            run_measure_objects_for_run(run_id)
            notify_ui("measure_objects")
        pipeline_timer.add_step("Measure", t.end_time - t.start_time)

        with Timer("STEP 11: Perimeter") as t:
            run_perimeter_for_run(run_id)
            notify_ui("perimeter")
        pipeline_timer.add_step("Perimeter", t.end_time - t.start_time)

        with Timer("STEP 12: Area") as t:
            # Încărcăm frontend_data pentru a folosi înălțimea pereților din formular
            frontend_data = load_frontend_data_for_run(run_id, job_root)
            run_area_for_run(run_id, frontend_data=frontend_data)
            
            area_out_dir = RUNNER_ROOT / "output" / run_id / "scale"
            
            found_any_image = False
            if area_out_dir.exists():
                for plan_dir in area_out_dir.iterdir():
                    if plan_dir.is_dir():
                        img_3d = plan_dir / "walls_3d_view.png"
                        if img_3d.exists():
                            notify_ui("area", img_3d)
                            found_any_image = True
                            continue
                        
                        img_2d = plan_dir / "final_viz.png"
                        if img_2d.exists():
                            notify_ui("area", img_2d)
                            found_any_image = True
            
            if not found_any_image:
                notify_ui("area")
                
        pipeline_timer.add_step("Area", t.end_time - t.start_time)
        
        with Timer("STEP 13: Roof") as t:
            run_roof_for_run(run_id)
            notify_ui("roof")
        pipeline_timer.add_step("Roof", t.end_time - t.start_time)

        frontend_data = load_frontend_data_for_run(run_id, job_root)
        
        with Timer("STEP 14-15: Pricing") as t:
            pricing_results = run_pricing_for_run(run_id, frontend_data_override=frontend_data)
            notify_ui("pricing")
        pipeline_timer.add_step("Pricing", t.end_time - t.start_time)

        with Timer("STEP 16-17: Offer Generation") as t:
            # Offer level should follow user intent. If the form doesn't include `nivelOferta`,
            # fall back to calc_mode/offer_type selection coming from the API.
            nivel_oferta = frontend_data.get("materialeFinisaj", {}).get("nivelOferta")
            if not nivel_oferta:
                cm = (frontend_data.get("calc_mode") or "").lower()
                if cm in ("structure", "structura"):
                    nivel_oferta = "Structură"
                elif cm in ("structure_windows", "structura_ferestre", "structura+ferestre"):
                    nivel_oferta = "Structură + ferestre"
                elif cm in ("full_house", "full_house_premium", "casa_completa", "casacompleta"):
                    nivel_oferta = "Casă completă"
                else:
                    nivel_oferta = "Structură"
            print(f"📋 [OFFER] Level: {nivel_oferta}")
            
            for res in pricing_results:
                if res.success and res.result_data:
                    try:
                        build_final_offer(
                            res.result_data, 
                            nivel_oferta, 
                            res.work_dir / "final_offer.json"
                        )
                    except Exception as e:
                        print(f"⚠️ Error building offer for {res.plan_id}: {e}")
            
            notify_ui("offer_generation")
        pipeline_timer.add_step("Offer Generation", t.end_time - t.start_time)
        
        with Timer("STEP 18-19: PDF GENERATION") as t:
            pdf_path = None
            admin_pdf_path = None
            calc_method_pdf_path = None
            try:
                # Generate user PDF (with branding)
                pdf_path = generate_complete_offer_pdf(run_id=run_id, output_path=None)
                print(f"✅ [PDF] User PDF generated: {pdf_path}")
                notify_ui("pdf_generation", pdf_path)
                
                # Generate admin PDF (strict, no branding)
                admin_pdf_path = generate_admin_offer_pdf(run_id=run_id, output_path=None)
                print(f"✅ [PDF] Admin PDF generated: {admin_pdf_path}")
                notify_ui("pdf_generation", admin_pdf_path)
                
                # Generate calculation method PDF (English, detailed explanations)
                try:
                    calc_method_pdf_path = generate_admin_calculation_method_pdf(run_id=run_id, output_path=None)
                    print(f"✅ [PDF] Calculation Method PDF generated: {calc_method_pdf_path}")
                    notify_ui("pdf_generation", calc_method_pdf_path)
                except Exception as calc_err:
                    print(f"⚠️ Calculation Method PDF Error: {calc_err}")
                    import traceback
                    traceback.print_exc()
            except Exception as e:
                print(f"⚠️ PDF Error: {e}")
                import traceback
                traceback.print_exc()
                notify_ui("pdf_generation")
        pipeline_timer.add_step("PDF", t.end_time - t.start_time)
        
        time.sleep(2.0)
        # Notify completion with user PDF (for backward compatibility)
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
        from segmenter import segment_document
        segment_document(args.input, build_job_root(job_id=args.job_id, prefix="segmentation_job") / "segmentation")
    else:
        run_segmentation_and_classification_for_document(
            args.input, 
            job_id=args.job_id,
            frontend_data_json=frontend_data_json
        )