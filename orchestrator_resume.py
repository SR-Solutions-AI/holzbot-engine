"""
Resume pipelines for „Angebot bearbeiten“ (variables-only or post–detection-editor).

  python -m orchestrator_resume --job-id <offer_uuid> --engine-run-id <output_run_id> --mode variables|post_editor

variables: pricing + offer JSON + PDFs (reuses existing output/<engine_run_id>).
post_editor: from apply_detections_edited through scale/count/roof/pricing/PDF.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config.frontend_loader import load_frontend_data_for_run
from config.settings import JOBS_ROOT, load_plan_infos
from cubicasa_detector.raster_api import apply_detections_edited
from cubicasa_detector.manual_blueprint import (
    run_walls_pipeline_after_manual_editor,
    seed_roof_only_rooms_from_roof_polygons,
    apply_roof_only_synthetic_walls_and_scale,
)
from count_objects import run_count_objects_for_run
from offer_builder import build_final_offer
from pdf_generator import (
    generate_complete_offer_pdf,
    generate_admin_offer_pdf,
    generate_admin_calculation_method_pdf,
    generate_roof_measurements_pdf,
)
from pricing.jobs import run_pricing_for_run
from roof.jobs import run_roof_for_run
from roof.roof_pricing import generate_roof_pricing
from scale import run_scale_detection_for_run

from orchestrator import (
    PipelineTimer,
    Timer,
    _PROGRESS_WEIGHTS,
    _check_raster_complete,
    _enrich_frontend_with_aufstockung_phase1,
    _make_progress_sender,
    notify_ui,
    wait_for_offer_number_in_job,
)


def _pdf_block(run_id: str, job_root: Path, frontend_data: dict, set_progress) -> tuple:
    """Returns (pdf_path,)."""
    pdf_path = None
    admin_pdf_path = None
    calc_method_pdf_path = None
    measurements_only = bool(frontend_data.get("measurements_only_offer"))
    try:
        try:
            generate_roof_pricing(run_id=run_id, frontend_data=frontend_data)
        except Exception as rp_err:
            print(f"⚠️ Roof pricing Error: {rp_err}", flush=True)
        set_progress(_PROGRESS_WEIGHTS["pdf_end"] - 12)
        if measurements_only:
            print(
                "🔸 [PDF] measurements_only_offer: skip Kunden-/Admin-/Berechnungs-PDF; nur Mengen-/Maß-PDF.",
                flush=True,
            )
        else:
            pdf_path = generate_complete_offer_pdf(run_id=run_id, output_path=None, job_root=job_root)
            print(f"✅ [PDF] User PDF generated: {pdf_path}", flush=True)
            notify_ui("pdf_generation", pdf_path)
            set_progress(_PROGRESS_WEIGHTS["pdf_end"] - 8)
            admin_pdf_path = generate_admin_offer_pdf(run_id=run_id, output_path=None, job_root=job_root)
            print(f"✅ [PDF] Admin PDF generated: {admin_pdf_path}", flush=True)
            notify_ui("pdf_generation", admin_pdf_path)
            set_progress(_PROGRESS_WEIGHTS["pdf_end"] - 4)
            if not bool(frontend_data.get("roof_only_offer")):
                try:
                    calc_method_pdf_path = generate_admin_calculation_method_pdf(
                        run_id=run_id, output_path=None, job_root=job_root
                    )
                    print(f"✅ [PDF] Calculation Method PDF generated: {calc_method_pdf_path}", flush=True)
                    notify_ui("pdf_generation", calc_method_pdf_path)
                except Exception as calc_err:
                    print(f"⚠️ Calculation Method PDF Error: {calc_err}", flush=True)
                    import traceback

                    traceback.print_exc()
            else:
                print("🔸 [PDF] roof_only_offer: skip calculation method PDF (full-house doc)", flush=True)
    except Exception as e:
        print(f"⚠️ PDF Error: {e}", flush=True)
        import traceback

        traceback.print_exc()
        notify_ui("pdf_generation")
    try:
        roof_pdf_path = generate_roof_measurements_pdf(run_id=run_id, output_path=None)
        if roof_pdf_path:
            notify_ui("pdf_generation", roof_pdf_path)
            print(f"✅ [PDF] Roof measurements PDF generated: {roof_pdf_path}", flush=True)
    except Exception as roof_pdf_err:
        print(f"⚠️ [PDF] Roof measurements PDF Error: {roof_pdf_err}", flush=True)
    return (pdf_path,)


def run_variables(engine_run_id: str, job_id: str) -> int:
    job_root = JOBS_ROOT / job_id
    if not job_root.is_dir():
        print(f">>> ERROR: job_root missing: {job_root}", flush=True)
        return 1
    out_root = Path(__file__).resolve().parent / "output" / engine_run_id
    if not out_root.exists():
        print(f">>> ERROR: output run not found: {out_root}", flush=True)
        return 1

    pt = PipelineTimer()
    pt.start()
    set_progress, _ = _make_progress_sender()
    set_progress(_PROGRESS_WEIGHTS["pricing_end"] - 2)

    frontend_data = load_frontend_data_for_run(engine_run_id, job_root)
    frontend_data = _enrich_frontend_with_aufstockung_phase1(frontend_data, job_root)

    with Timer("RESUME: Pricing") as t:
        pricing_results = run_pricing_for_run(engine_run_id, frontend_data_override=frontend_data)
        notify_ui("pricing")
    set_progress(_PROGRESS_WEIGHTS["pricing_end"])
    pt.add_step("Pricing", t.end_time - t.start_time)

    set_progress(_PROGRESS_WEIGHTS["offer_end"] - 2)
    with Timer("RESUME: Offer Generation") as t:
        mf = frontend_data.get("materialeFinisaj") or {}
        sc = frontend_data.get("sistemConstructiv") or {}
        nivel_oferta = sc.get("nivelOferta") or mf.get("nivelOferta")
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
        print(f"📋 [OFFER] Level: {nivel_oferta}", flush=True)
        for res in pricing_results:
            if res.success and res.result_data:
                try:
                    build_final_offer(
                        res.result_data,
                        nivel_oferta,
                        res.work_dir / "final_offer.json",
                    )
                except Exception as e:
                    print(f"⚠️ Error building offer for {res.plan_id}: {e}", flush=True)
        notify_ui("offer_generation")
    set_progress(_PROGRESS_WEIGHTS["offer_end"])
    pt.add_step("Offer Generation", t.end_time - t.start_time)

    wait_for_offer_number_in_job(job_root)
    frontend_data = load_frontend_data_for_run(engine_run_id, job_root)
    frontend_data = _enrich_frontend_with_aufstockung_phase1(frontend_data, job_root)

    with Timer("RESUME: PDF") as t:
        pdf_path, *_rest = _pdf_block(engine_run_id, job_root, frontend_data, set_progress)
    pt.add_step("PDF", t.end_time - t.start_time)

    time.sleep(2.0)
    set_progress(100)
    notify_ui("computation_complete", pdf_path if pdf_path and Path(pdf_path).exists() else None)
    pt.finish()
    return 0


def run_post_editor(engine_run_id: str, job_id: str) -> int:
    job_root = JOBS_ROOT / job_id
    if not job_root.is_dir():
        print(f">>> ERROR: job_root missing: {job_root}", flush=True)
        return 1

    pt = PipelineTimer()
    pt.start()
    set_progress, _ = _make_progress_sender()

    run_id = engine_run_id
    frontend_data = load_frontend_data_for_run(run_id, job_root)
    frontend_data = _enrich_frontend_with_aufstockung_phase1(frontend_data, job_root)
    roof_only_offer = bool(frontend_data.get("roof_only_offer"))
    wizard_package = str(frontend_data.get("wizard_package") or "").strip().lower()
    is_aufstockung_offer = wizard_package in ("aufstockung", "zubau", "zubau_aufstockung")

    plans = load_plan_infos(run_id, "scale")
    if not plans:
        print(">>> ERROR: No plans for post_editor resume (load_plan_infos empty)", flush=True)
        return 1

    raster_scan_failed = False
    with Timer("RESUME: apply detections + walls") as t:
        for plan in plans:
            raster_dir = plan.stage_work_dir / "cubicasa_steps" / "raster"
            apply_detections_edited(raster_dir)

        if roof_only_offer:
            seeded = seed_roof_only_rooms_from_roof_polygons(run_id, plans)
            if seeded:
                print("🔸 [resume] roof_only_offer: seeded rooms from roof editor.", flush=True)
            else:
                print(
                    "⚠️ [resume] roof_only_offer: no roof polygons for seed; continuing.",
                    flush=True,
                )

        max_w = max(1, min(3, len(plans)))

        def _walls_one(p):
            return run_walls_pipeline_after_manual_editor(p, roof_only_offer)

        set_progress(55.5)
        with ThreadPoolExecutor(max_workers=max_w) as ex:
            results = list(ex.map(_walls_one, plans))

        if not all(results):
            if roof_only_offer:
                print("🔸 [resume] roof_only: walls fallback per plan.", flush=True)
                fallback_failed = False
                for plan, ok in zip(plans, results):
                    if ok:
                        continue
                    if not apply_roof_only_synthetic_walls_and_scale(plan):
                        fallback_failed = True
                raster_scan_failed = fallback_failed
            else:
                raster_scan_failed = True
        set_progress(56.5)

    set_progress(float(_PROGRESS_WEIGHTS["raster_end"] - 1))
    raster_ok, failed_plan_ids = _check_raster_complete(plans, job_root)
    if failed_plan_ids:
        print(f"⚠️ [resume] Incomplete raster: {failed_plan_ids}", flush=True)
    raster_ok = raster_ok and not raster_scan_failed
    pt.add_step("Manual blueprint + walls", t.end_time - t.start_time)

    if not raster_ok:
        if is_aufstockung_offer:
            print(
                "\n⚠️ [resume] Aufstockung: raster completeness check bypassed; continuing pipeline.",
                flush=True,
            )
            raster_ok = True
        else:
            print(
                "\n⛔ [resume] Raster incomplete — cannot continue to scale/pricing.\n",
                flush=True,
            )
            return 2

    set_progress(_PROGRESS_WEIGHTS["raster_end"])
    with Timer("RESUME: Scale") as t:
        run_scale_detection_for_run(run_id)
    set_progress(_PROGRESS_WEIGHTS["scale_end"])
    pt.add_step("Scale", t.end_time - t.start_time)

    set_progress(_PROGRESS_WEIGHTS["scale_end"] + 1)

    def _run_count_objects_timed():
        _start = time.time()
        run_count_objects_for_run(run_id)
        return time.time() - _start

    def _run_roof_timed():
        _start = time.time()
        run_roof_for_run(run_id, notify_ui_events=True)
        return time.time() - _start

    with Timer("RESUME: Count Objects || Roof") as t_par:
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_co = executor.submit(_run_count_objects_timed)
            fut_roof = executor.submit(_run_roof_timed)
            t_count_objects = fut_co.result()
            t_roof = fut_roof.result()
    pt.add_step("Count Objects", t_count_objects)
    pt.add_step("Roof", t_roof)
    set_progress(_PROGRESS_WEIGHTS["count_roof_end"])

    frontend_data = load_frontend_data_for_run(run_id, job_root)
    frontend_data = _enrich_frontend_with_aufstockung_phase1(frontend_data, job_root)

    set_progress(_PROGRESS_WEIGHTS["pricing_end"] - 2)
    with Timer("RESUME: Pricing") as t:
        pricing_results = run_pricing_for_run(run_id, frontend_data_override=frontend_data)
        notify_ui("pricing")
    set_progress(_PROGRESS_WEIGHTS["pricing_end"])
    pt.add_step("Pricing", t.end_time - t.start_time)

    set_progress(_PROGRESS_WEIGHTS["offer_end"] - 2)
    with Timer("RESUME: Offer Generation") as t:
        mf = frontend_data.get("materialeFinisaj") or {}
        sc = frontend_data.get("sistemConstructiv") or {}
        nivel_oferta = sc.get("nivelOferta") or mf.get("nivelOferta")
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
        for res in pricing_results:
            if res.success and res.result_data:
                try:
                    build_final_offer(
                        res.result_data,
                        nivel_oferta,
                        res.work_dir / "final_offer.json",
                    )
                except Exception as e:
                    print(f"⚠️ Error building offer for {res.plan_id}: {e}", flush=True)
        notify_ui("offer_generation")
    set_progress(_PROGRESS_WEIGHTS["offer_end"])
    pt.add_step("Offer Generation", t.end_time - t.start_time)

    wait_for_offer_number_in_job(job_root)
    frontend_data = load_frontend_data_for_run(run_id, job_root)

    with Timer("RESUME: PDF") as t:
        pdf_path, *_ = _pdf_block(run_id, job_root, frontend_data, set_progress)
    pt.add_step("PDF", t.end_time - t.start_time)

    time.sleep(2.0)
    set_progress(100)
    notify_ui("computation_complete", pdf_path if pdf_path and Path(pdf_path).exists() else None)
    pt.finish()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--engine-run-id", required=True)
    ap.add_argument("--mode", choices=["variables", "post_editor"], required=True)
    args = ap.parse_args()
    if args.mode == "variables":
        return run_variables(args.engine_run_id, args.job_id)
    return run_post_editor(args.engine_run_id, args.job_id)


if __name__ == "__main__":
    sys.exit(main())
