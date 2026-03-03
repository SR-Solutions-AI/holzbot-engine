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

from config.settings import build_job_root, RUNS_ROOT, RUNNER_ROOT, load_plan_infos, PlanInfo, get_run_dir
from config.frontend_loader import load_frontend_data_for_run
from floor_classifier.basement_scorer import run_basement_scoring
from floor_classifier.floor_order_gemini import run_floor_order_from_gemini
from typing import List, Tuple

# ✅ IMPORT SEGMENTER JOBS
from segmenter.jobs import run_segmentation_for_documents, get_all_classification_results

from floor_classifier import run_floor_classification, FloorClassificationResult
from detections.jobs import run_detections_for_run
from cubicasa_detector.jobs import run_cubicasa_for_plan, run_cubicasa_phase1, run_cubicasa_phase2
from scale import run_scale_detection_for_run
from count_objects import run_count_objects_for_run
from roof.jobs import run_roof_for_run
from pricing.jobs import run_pricing_for_run, PricingJobResult
from offer_builder import build_final_offer
from pdf_generator import generate_complete_offer_pdf, generate_admin_offer_pdf, generate_admin_calculation_method_pdf, generate_roof_measurements_pdf
from roof.roof_pricing import generate_roof_pricing
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


def notify_progress(percent: int):
    """Trimite procentul de progres către Backend/Frontend pentru progress bar."""
    msg = f">>> UI:PROGRESS:{percent}"
    print(msg, flush=True)
    sys.stdout.flush()


def _notify_all_cubicasa_images_for_plan(plan: PlanInfo) -> None:
    """
    Trimite către UI (admin Details) toate imaginile generate de Cubicasa pentru acest plan:
    pereți, camere, fill, walls_from_coords, openings, wall_repair, raster, etc.
    """
    base = plan.stage_work_dir / "cubicasa_steps"
    if not base.is_dir():
        return
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in sorted(base.rglob(ext)):
            if p.is_file():
                notify_ui("cubicasa_step", p)


def _check_raster_complete(plans: List[PlanInfo]) -> Tuple[bool, List[str]]:
    """
    Verifică că toate planurile au camerele calculate:
    - room_scales.json există, total_area_m2 > 0, total_area_px > 0
    - cel puțin o cameră în room_scales/rooms
    - rooms.png există (masca camerelor)
    Returns (all_ok, failed_plan_ids).
    """
    failed: List[str] = []
    for plan in plans:
        base = plan.stage_work_dir / "cubicasa_steps"
        room_scales_path = base / "raster_processing" / "walls_from_coords" / "room_scales.json"
        # rooms.png poate fi în raster/rooms.png sau raster_processing/rooms.png
        rooms_png_path_raster = base / "raster" / "rooms.png"
        rooms_png_path_processing = base / "raster_processing" / "rooms.png"
        rooms_png_path = rooms_png_path_raster if rooms_png_path_raster.exists() else rooms_png_path_processing

        if not room_scales_path.exists():
            failed.append(plan.plan_id)
            continue
        try:
            data = json.loads(room_scales_path.read_text(encoding="utf-8"))
            m2 = float(data.get("total_area_m2") or 0)
            px = int(data.get("total_area_px") or 0)
            rooms_data = data.get("room_scales") or data.get("rooms") or {}
            num_rooms = len(rooms_data) if isinstance(rooms_data, dict) else 0

            if m2 <= 0 or px <= 0:
                failed.append(plan.plan_id)
            elif num_rooms < 1:
                failed.append(plan.plan_id)
            elif not rooms_png_path.exists():
                failed.append(plan.plan_id)
        except Exception:
            failed.append(plan.plan_id)
    return (len(failed) == 0, failed)


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
        self.print_summary()
    
    def print_summary(self):
        """Afișează rezumatul timpilor pentru fiecare pas."""
        if not self.steps:
            return
        
        total_duration = self.total_end - self.total_start if self.total_start and self.total_end else 0
        
        print("\n" + "="*70)
        print("⏱️  REZUMAT TIMPI PENTRU FIECARE PAS")
        print("="*70)
        
        # Sortăm după durată (descrescător)
        sorted_steps = sorted(self.steps, key=lambda x: x["duration"], reverse=True)
        
        for step in sorted_steps:
            duration = step["duration"]
            percentage = (duration / total_duration * 100) if total_duration > 0 else 0
            print(f"  {step['name']:.<40} {duration:>7.2f}s ({percentage:>5.1f}%)")
        
        print("-"*70)
        print(f"  {'TOTAL':.<40} {total_duration:>7.2f}s (100.0%)")
        print("="*70 + "\n")

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
    
    # Save frontend data (from env). Skip if file already exists - API may have written full payload to avoid env truncation.
    if frontend_data_json and not (job_root / "frontend_data.json").exists():
        try:
            parsed_data = json.loads(frontend_data_json)
            out_file = job_root / "frontend_data.json"
            out_file.write_text(json.dumps(parsed_data, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"✅ Frontend data saved to {out_file}")
        except Exception as e:
            print(f"⚠️ Failed to save frontend data: {e}")
    elif (job_root / "frontend_data.json").exists():
        print(f"✅ Using existing frontend_data.json from job root (written by API)")

    notify_ui("segmentation_start")
    notify_progress(2)
    
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
        def _seg_progress(done: int, total: int):
            if total and total > 0:
                notify_progress(round(2 + 3 * (done / total)))
        seg_results = run_segmentation_for_documents(
            input_path=input_path,
            output_base_dir=segmentation_out_base,
            max_workers=None,
            progress_callback=_seg_progress,
        )
        
        # -------------------------------------------------------------
        # ✅ 1. UI: SOLID WALLS FIRST (Robust Check)
        # -------------------------------------------------------------
        print("📢 [UI] Sending Solid Walls previews (Priority 1)...")
        notify_progress(3)  # segmentare rulată
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
        
        notify_progress(4)  # solid walls trimise
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

    notify_progress(5)  # segmentare + clasificare (înainte de a ști numărul de etaje)
    pipeline_timer.add_step("Multi-Segmentation", t.end_time - t.start_time)

    # Colectăm rezultatele
    all_cls_results = get_all_classification_results(seg_results)
    plans = [ClassifiedPlanInfo(job_root, r.image_path, r.label) for r in all_cls_results]
    
    # Debug output
    print(f"\n📊 CLASSIFICATION SUMMARY:")
    print(f"   Total plans detected: {len(plans)}")
    from collections import Counter, defaultdict
    label_counts = Counter(p.label for p in plans)
    for label, count in label_counts.items():
        print(f"   {label}: {count}")

    # Blueprint-uri finale (doar house_blueprint) – verificare număr etaje imediat după clasificare
    house_plans = [p for p in plans if p.label == "house_blueprint"]
    print(f"\n🏠 HOUSE PLANS ONLY: {len(house_plans)}/{len(plans)}")
    if len(house_plans) < len(plans):
        print(f"⚠️  {len(plans) - len(house_plans)} plans EXCLUDED (not house_blueprint)\n")

    # Progress dinamic: 3 seg, 4 floor, 5 detections, 6 + 8*etaje (raster) + 13 (scale→pdf)
    num_plans = len(house_plans)
    CUBICASA_SUB_STEPS_PER_PLAN = 8  # start, preprocess, raster_api_done, phase1_end, phase2_start, brute_done, walls_done, phase2_end
    total_progress_steps = 19 + CUBICASA_SUB_STEPS_PER_PLAN * num_plans
    def _progress(done: int) -> int:
        return round((done / total_progress_steps) * 100)
    notify_progress(_progress(3))  # seg_start, segmentation, classification

    # ---------- Verificare număr etaje vs formular (chiar după ce știm ce e blueprint) ----------
    frontend_data = load_frontend_data_for_run(job_root.name, job_root)
    our_floors = len(house_plans)
    floors_number = frontend_data.get("floorsNumber")
    basement = frontend_data.get("basement", False)
    structura = frontend_data.get("structuraCladirii", {})
    tip_fundatie_beci = (structura.get("tipFundatieBeci") or "")
    has_basement_form = basement or (
        tip_fundatie_beci and "Keller" in str(tip_fundatie_beci) and "Kein Keller" not in str(tip_fundatie_beci)
    )
    lista_etaje = structura.get("listaEtaje")
    # Mansardă locuibilă (Wohnfläche): Dachgeschoss ohne Kniestock sau mit Kniestock = etaj suplimentar. Pod nu e etaj.
    if isinstance(lista_etaje, list):
        has_mansarda_wohnflaeche = "mansarda_mit" in lista_etaje or "mansarda_ohne" in lista_etaje
    else:
        has_mansarda_wohnflaeche = False

    if floors_number is not None:
        user_expected = (
            int(floors_number)
            + (1 if has_basement_form else 0)
            + (1 if has_mansarda_wohnflaeche else 0)
        )
    else:
        # listaEtaje: "intermediar" = etaj cu plan separat; mansardă (ohne/mit Kniestock, Wohnfläche) = etaj.
        if isinstance(lista_etaje, list):
            etaje_intermediare = sum(1 for e in lista_etaje if e == "intermediar")
            user_expected = (
                1 + etaje_intermediare
                + (1 if has_basement_form else 0)
                + (1 if has_mansarda_wohnflaeche else 0)
            )
        elif isinstance(lista_etaje, dict):
            user_expected = 1 + (1 if has_basement_form else 0)
        else:
            user_expected = 1 + (1 if has_basement_form else 0)
    _extra_suffix = ""
    if isinstance(lista_etaje, list) and has_mansarda_wohnflaeche:
        _extra_suffix += ", Dachgeschoss (Wohnfläche)"
    print(f"   📋 Formular: etaje așteptate (cu beci{_extra_suffix}) = {user_expected}  |  Planuri blueprint = {our_floors}")
    if our_floors != user_expected:
        print(f"\n⛔ Număr etaje: plan încărcat={our_floors}, formular (cu beci)={user_expected}")
        print(">>> ERROR: Numărul de etaje din planul încărcat nu coincide cu cel ales din formular.")
        sys.exit(1)

    # =========================================================
    # REST OF PIPELINE
    # =========================================================

    with Timer("STEP 4: Floor Classification") as t:
        floor_results = run_floor_classification(job_root, plans)
        notify_ui("floor_classification")
    notify_progress(_progress(4))
    pipeline_timer.add_step("Floor Classification", t.end_time - t.start_time)

    if house_plans:
        # ---------- Ordine etaje (de jos în sus) din etichetele Gemini; beci = primul din ordine ----------
        order_from_bottom = None
        with Timer("STEP 4c: Floor order from labels (Gemini)") as _t:
            order_from_bottom = run_floor_order_from_gemini(house_plans)
        pipeline_timer.add_step("Floor order (Gemini)", _t.end_time - _t.start_time)

        # ---------- Dacă nu avem ordine de la Gemini și userul a ales beci: scor Gemini pentru a alege beciul ----------
        basement_plan_index = None
        if order_from_bottom is not None and has_basement_form and our_floors > 0:
            basement_plan_index = order_from_bottom[0]
            print(f"   💾 Beci din ordine Gemini: plan index {basement_plan_index}")
        elif has_basement_form and our_floors > 0:
            with Timer("STEP 4b: Basement scoring (Gemini)") as _t:
                basement_plan_index = run_basement_scoring(house_plans)
            pipeline_timer.add_step("Basement scoring", _t.end_time - _t.start_time)

        # ---------- Creare run și salvare floor_order / basement_plan_id ----------
        run_id = _create_run_for_detections(job_root, house_plans)
        run_dir = get_run_dir(run_id)
        if order_from_bottom is not None:
            (run_dir / "floor_order.json").write_text(
                json.dumps({"order_from_bottom": order_from_bottom}, indent=2),
                encoding="utf-8",
            )
            print(f"   💾 Salvat: {run_dir / 'floor_order.json'}")
        if basement_plan_index is not None:
            (run_dir / "basement_plan_id.json").write_text(
                json.dumps({"basement_plan_index": basement_plan_index}, indent=2),
                encoding="utf-8",
            )
            print(f"   💾 Salvat: {run_dir / 'basement_plan_id.json'} (beci = plan index {basement_plan_index})")
        
        with Timer("STEP 5: Detections") as t:
            run_detections_for_run(run_id)
            # Notificarea UI pentru detections este trimisă direct din raster_processing.py
            # după generarea 01_openings.png (în timpul scale detection - STEP 6)
        notify_progress(_progress(5))
        pipeline_timer.add_step("Detections", t.end_time - t.start_time)
        
        # ✅ STEP 5.5: RasterScan Processing (generează room_scales.json)
        # Acest pas apelează RasterScan API și generează pereții corecți + lista de camere + room_scales.json
        # DUPĂ ce avem lista de camere, putem calcula scale detection (STEP 6)
        raster_timings = []  # (nume_pas, durata) din fiecare plan, agregate la final
        with Timer("STEP 5.5: RasterScan Processing") as t:
            print(f"\n{'='*60}")
            print(f"[RasterScan] Starting RasterScan processing for run: {run_id}")
            print(f"{'='*60}\n")
            
            # Încărcăm planurile pentru acest run
            plans = load_plan_infos(run_id, "scale")  # Folosim "scale" pentru că output_dir va fi în scale/
            raster_scan_failed = False
            notify_progress(_progress(6))  # început RasterScan
            # Un singur contor în ordinea executării: la fiecare callback incrementăm și raportăm (progres mereu crescător)
            raster_done = [6]  # 6 = început RasterScan; apoi +1 la fiecare sub-pas (phase1 plan0, phase1 plan1, phase2 plan0, phase2 plan1)
            def _raster_step():
                raster_done[0] += 1
                notify_progress(_progress(raster_done[0]))
            # Phase 1 secvențial (NU în paralel) ca să nu trimitem toate planurile la Raster API în același timp
            def do_phase1(plan, plan_index: int):
                def on_progress(sub_step: int):
                    _raster_step()  # la fiecare sub_step (0,1,2,3) un pas completat
                print(f"[RasterScan] Phase 1 {plan.plan_id} → Raster API + 02_ai_walls_closed", flush=True)
                run_cubicasa_phase1(
                    plan_image=plan.plan_image,
                    output_dir=plan.stage_work_dir,
                    raster_timings=raster_timings,
                    progress_callback=on_progress,
                )
                print(f"✅ [RasterScan] {plan.plan_id}: Phase 1 OK", flush=True)
            for i, plan in enumerate(plans):
                try:
                    do_phase1(plan, i)
                except Exception as e:
                    print(f"❌ [RasterScan] {plan.plan_id}: Eroare Phase 1: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    raster_scan_failed = True
            if not raster_scan_failed:
                # Phase 2 secvențial (un plan după altul): brute force + room_scales.json, ca să nu se blocheze
                def do_phase2(plan, plan_index: int):
                    def on_progress(sub_step: int):
                        _raster_step()  # la fiecare sub_step (0,1,2,3) un pas completat
                    print(f"[RasterScan] Phase 2 {plan.plan_id} → brute force + room_scales.json", flush=True)
                    run_cubicasa_phase2(
                        output_dir=plan.stage_work_dir,
                        raster_timings=raster_timings,
                        progress_callback=on_progress,
                    )
                    print(f"✅ [RasterScan] {plan.plan_id}: room_scales.json generat cu succes", flush=True)
                for i, plan in enumerate(plans):
                    try:
                        do_phase2(plan, i)
                    except Exception as e:
                        print(f"❌ [RasterScan] {plan.plan_id}: Eroare Phase 2: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                        raster_scan_failed = True
                # Trimite toate imaginile Cubicasa (pereți, camere, fill, openings, etc.) la admin Details
                if not raster_scan_failed:
                    for plan in plans:
                        _notify_all_cubicasa_images_for_plan(plan)
            
            raster_ok, failed_plan_ids = _check_raster_complete(plans)
            if failed_plan_ids:
                print(f"⚠️ [RasterScan] Planuri fără camere calculate (room_scales.json + rooms.png): {failed_plan_ids}", flush=True)
            raster_ok = raster_ok and not raster_scan_failed
            
            print(f"\n{'='*60}")
            print(f"[RasterScan] RasterScan Processing Complete")
            print(f"{'='*60}\n")
        pipeline_timer.add_step("RasterScan Processing", t.end_time - t.start_time)
        # Pași detaliați Raster (agregare pe toate planurile)
        if raster_timings:
            by_name = defaultdict(float)
            for name, dur in raster_timings:
                by_name[name] += dur
            for name, dur in sorted(by_name.items(), key=lambda x: -x[1]):
                pipeline_timer.add_step(f"  └ {name}", dur)
        
        if not raster_ok:
            print(f"\n⛔ [Pipeline] Camerele nu sunt calculate pentru toate planurile.", flush=True)
            print(f"   Se sar pașii: Scale, Count Objects, Roof, Pricing, Offer, PDF.", flush=True)
            print(f"   Asigură-te că RasterScan finalizează (room_scales.json + rooms.png) pentru fiecare etaj în ordine, apoi rulează din nou.\n", flush=True)
        
        if raster_ok:
            notify_progress(_progress(6 + CUBICASA_SUB_STEPS_PER_PLAN * num_plans))
            with Timer("STEP 6: Scale") as t:
                run_scale_detection_for_run(run_id)
            notify_progress(_progress(7 + CUBICASA_SUB_STEPS_PER_PLAN * num_plans))
            pipeline_timer.add_step("Scale", t.end_time - t.start_time)

            notify_progress(_progress(8 + CUBICASA_SUB_STEPS_PER_PLAN * num_plans))
            with Timer("STEP 7: Count Objects") as t:
                run_count_objects_for_run(run_id)
            notify_progress(_progress(9 + CUBICASA_SUB_STEPS_PER_PLAN * num_plans))
            pipeline_timer.add_step("Count Objects", t.end_time - t.start_time)

            notify_progress(_progress(10 + CUBICASA_SUB_STEPS_PER_PLAN * num_plans))
            with Timer("STEP 13: Roof") as t:
                run_roof_for_run(run_id)
                notify_ui("roof")
            notify_progress(_progress(11 + CUBICASA_SUB_STEPS_PER_PLAN * num_plans))
            pipeline_timer.add_step("Roof", t.end_time - t.start_time)

            frontend_data = load_frontend_data_for_run(run_id, job_root)

            notify_progress(_progress(12 + CUBICASA_SUB_STEPS_PER_PLAN * num_plans))
            with Timer("STEP 14-15: Pricing") as t:
                pricing_results = run_pricing_for_run(run_id, frontend_data_override=frontend_data)
                notify_ui("pricing")
            notify_progress(_progress(13 + CUBICASA_SUB_STEPS_PER_PLAN * num_plans))
            pipeline_timer.add_step("Pricing", t.end_time - t.start_time)

            notify_progress(_progress(14 + CUBICASA_SUB_STEPS_PER_PLAN * num_plans))
            with Timer("STEP 16-17: Offer Generation") as t:
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
            notify_progress(_progress(15 + CUBICASA_SUB_STEPS_PER_PLAN * num_plans))  # sfârșit Offer
            pipeline_timer.add_step("Offer Generation", t.end_time - t.start_time)

            with Timer("STEP 18-19: PDF GENERATION") as t:
                pdf_path = None
                admin_pdf_path = None
                calc_method_pdf_path = None
                try:
                    # Generează roof_pricing.json înainte de PDF (pentru ofertă cu prețuri acoperiș)
                    try:
                        generate_roof_pricing(run_id=run_id, frontend_data=frontend_data)
                    except Exception as rp_err:
                        print(f"⚠️ Roof pricing Error: {rp_err}")
                    notify_progress(_progress(16 + CUBICASA_SUB_STEPS_PER_PLAN * num_plans))  # început PDF
                    pdf_path = generate_complete_offer_pdf(run_id=run_id, output_path=None, job_root=job_root)
                    print(f"✅ [PDF] User PDF generated: {pdf_path}")
                    notify_ui("pdf_generation", pdf_path)
                    notify_progress(_progress(17 + CUBICASA_SUB_STEPS_PER_PLAN * num_plans))  # după PDF utilizator
                    admin_pdf_path = generate_admin_offer_pdf(run_id=run_id, output_path=None, job_root=job_root)
                    print(f"✅ [PDF] Admin PDF generated: {admin_pdf_path}")
                    notify_ui("pdf_generation", admin_pdf_path)
                    notify_progress(_progress(18 + CUBICASA_SUB_STEPS_PER_PLAN * num_plans))  # după PDF admin
                    try:
                        calc_method_pdf_path = generate_admin_calculation_method_pdf(run_id=run_id, output_path=None, job_root=job_root)
                        print(f"✅ [PDF] Calculation Method PDF generated: {calc_method_pdf_path}")
                        notify_ui("pdf_generation", calc_method_pdf_path)
                    except Exception as calc_err:
                        print(f"⚠️ Calculation Method PDF Error: {calc_err}")
                        import traceback
                        traceback.print_exc()
                    notify_progress(_progress(19 + CUBICASA_SUB_STEPS_PER_PLAN * num_plans))  # după PDF calc method
                    roof_pdf_path = generate_roof_measurements_pdf(run_id=run_id, output_path=None)
                    if roof_pdf_path:
                        notify_ui("pdf_generation", roof_pdf_path)
                except Exception as e:
                    print(f"⚠️ PDF Error: {e}")
                    import traceback
                    traceback.print_exc()
                    notify_ui("pdf_generation")
            pipeline_timer.add_step("PDF", t.end_time - t.start_time)

            time.sleep(2.0)
            notify_progress(100)
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
    
    # Segmentare = doar Gemini Crop (coordonate în procente → crop → blueprints/side_views).
    # Pipeline-ul vechi (detect_wall_zones, detect_clusters, classifier) este dezactivat.
    if args.no_classification:
        job_root = build_job_root(job_id=args.job_id, prefix="segmentation_job")
        segmentation_out_base = job_root / "segmentation"
        segmentation_out_base.mkdir(parents=True, exist_ok=True)
        run_segmentation_for_documents(
            input_path=args.input,
            output_base_dir=segmentation_out_base,
            max_workers=None,
        )
    else:
        run_segmentation_and_classification_for_document(
            args.input, 
            job_id=args.job_id,
            frontend_data_json=frontend_data_json
        )