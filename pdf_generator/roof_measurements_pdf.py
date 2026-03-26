# pdf_generator/roof_measurements_pdf.py
"""PDF cu măsurători detaliate – Dämmung & Dachdeckung + Bauteilmaße (DE)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

from config.settings import OUTPUT_ROOT, RUNNER_ROOT, JOBS_ROOT, RUNS_ROOT, load_plan_infos, PlansListError
from config.frontend_loader import load_frontend_data_for_run


def _load_floor_data_from_pricing(out_root: Path) -> list[dict[str, Any]]:
    """Încarcă măsurători per etaj din pricing_raw.json (structuri, finisaje, podele, tavan, deschideri)."""
    pricing_dir = out_root / "pricing"
    if not pricing_dir.exists():
        return []
    floors_data: list[dict[str, Any]] = []
    plan_dirs = sorted(p for p in pricing_dir.iterdir() if p.is_dir() and p.name.startswith("plan_"))
    for plan_dir in plan_dirs:
        pr_path = plan_dir / "pricing_raw.json"
        if not pr_path.exists():
            continue
        try:
            with open(pr_path, encoding="utf-8") as f:
                pr = json.load(f)
        except Exception:
            continue
        bd = pr.get("breakdown", {})
        # Structuri interiori/exteriori (walls_structure_int, walls_structure_ext)
        sw = bd.get("structure_walls", {}) or {}
        items_sw = sw.get("detailed_items", []) or sw.get("items", [])
        int_net = ext_net = 0.0
        for it in items_sw:
            cat = str(it.get("category", "")).lower()
            a = float(it.get("area_m2", 0) or 0)
            if cat == "walls_structure_int":
                int_net += a
            elif cat == "walls_structure_ext":
                ext_net += a
        # Finisaje interior/exterior (finish_interior, finish_exterior)
        fin = bd.get("finishes", {}) or {}
        items_fin = fin.get("detailed_items", []) or fin.get("items", [])
        fin_int = fin_ext = 0.0
        for it in items_fin:
            cat = str(it.get("category", "")).lower()
            a = float(it.get("area_m2", 0) or 0)
            if cat == "finish_interior":
                fin_int += a
            elif cat == "finish_exterior":
                fin_ext += a
        # Podele și tavan (floor_structure, ceiling_structure)
        fc = bd.get("floors_ceilings", {}) or {}
        items_fc = fc.get("detailed_items", []) or fc.get("items", [])
        floor_area = ceiling_area = 0.0
        for it in items_fc:
            cat = str(it.get("category", "")).lower()
            a = float(it.get("area_m2", 0) or 0)
            if cat == "floor_structure":
                floor_area += a
            elif cat == "ceiling_structure":
                ceiling_area += a
        if floor_area == 0 and items_fc:
            floor_area = float(pr.get("total_area_m2", 0) or 0)
        if ceiling_area == 0:
            ceiling_area = floor_area
        # Deschideri: număr uși, geamuri, suprafață totală (type: door, window)
        op = bd.get("openings", {}) or {}
        items_op = op.get("items", []) or op.get("detailed_items", [])
        num_doors = num_windows = 0
        openings_area = 0.0
        for it in items_op:
            t = str(it.get("type", "")).lower()
            a = float(it.get("area_m2", 0) or 0)
            openings_area += a
            if "door" in t:
                num_doors += 1
            elif "window" in t:
                num_windows += 1
        floors_data.append({
            "plan_id": plan_dir.name,
            "structure_int_net_m2": round(int_net, 2),
            "structure_ext_net_m2": round(ext_net, 2),
            "finish_int_net_m2": round(fin_int, 2),
            "finish_ext_net_m2": round(fin_ext, 2),
            "floor_area_m2": round(floor_area, 2),
            "ceiling_area_m2": round(ceiling_area, 2),
            "num_doors": int(num_doors),
            "num_windows": int(num_windows),
            "openings_area_m2": round(openings_area, 2),
        })
    return floors_data


def generate_roof_measurements_pdf(run_id: str, output_path: Path | None = None) -> Path | None:
    """
    Generează PDF cu măsurători acoperiș (Dämmung & Dachdeckung), în germană.
    Conține per etaj: suprafață izolată, suprafață acoperire, tinichigerie, suprafață finisaje, suprafață folie.
    """
    # Folosim aceeași logică de path ca generatorul principal: output/run_id sau jobs/run_id
    out_root = OUTPUT_ROOT / run_id
    if not out_root.exists():
        out_root = JOBS_ROOT / run_id / "output"
    if not out_root.exists():
        out_root = RUNNER_ROOT / "output" / run_id
    if not out_root.exists():
        print(f"⚠️ [PDF ROOF] Output nu există: {run_id}")
        return None

    job_root = JOBS_ROOT / run_id
    if not job_root.exists():
        job_root = None
    frontend_data = load_frontend_data_for_run(run_id, job_root)
    roof_only = bool(frontend_data.get("roof_only_offer")) if isinstance(frontend_data, dict) else False
    sistem_constructiv = frontend_data.get("sistemConstructiv", {})
    tip_acoperis = sistem_constructiv.get("tipAcoperis", "—")
    projektdaten = frontend_data.get("projektdaten", {}) or {}
    dd = frontend_data.get("daemmungDachdeckung", {}) or {}

    # Ordine etaje și etichete (Keller, Erdgeschoss, 1. Obergeschoss, Mansardă, etc.)
    plan_id_to_label: dict[str, str] = {}
    roof_floor_labels: list[str] = []  # etichete pentru by_floor "0", "1", "2" (doar etaje cu acoperiș)
    try:
        plan_infos = load_plan_infos(run_id, stage_name="pricing")
    except PlansListError:
        plan_infos = []
    order_from_bottom = None
    basement_plan_index = None
    for run_dir_candidate in (out_root, RUNNER_ROOT / "output" / run_id, RUNS_ROOT / run_id):
        run_dir = Path(run_dir_candidate)
        if not run_dir.exists():
            continue
        if order_from_bottom is None:
            fo_path = run_dir / "floor_order.json"
            if fo_path.exists():
                try:
                    data = json.loads(fo_path.read_text(encoding="utf-8"))
                    ob = data.get("order_from_bottom")
                    if ob and len(ob) == len(plan_infos) and set(ob) == set(range(len(plan_infos))):
                        order_from_bottom = ob
                except Exception:
                    pass
        if basement_plan_index is None:
            bp_path = run_dir / "basement_plan_id.json"
            if bp_path.exists():
                try:
                    bp_data = json.loads(bp_path.read_text(encoding="utf-8"))
                    raw = bp_data.get("basement_plan_index")
                    if raw is not None and 0 <= int(raw) < len(plan_infos):
                        basement_plan_index = int(raw)
                except Exception:
                    pass
        if order_from_bottom is not None and basement_plan_index is not None:
            break

    if plan_infos:
        # Construim enriched_plans cu floor_type din plan_metadata
        enriched_plans = []
        for i, plan in enumerate(plan_infos):
            plan_name_parts = plan.plan_id.split("_")
            meta_filename = f"{plan_name_parts[-2]}_{plan_name_parts[-1]}.json" if len(plan_name_parts) >= 2 and plan_name_parts[-2] == "cluster" else f"{plan.plan_id}.json"
            meta_path = out_root / "plan_metadata" / meta_filename
            floor_type = "unknown"
            if meta_path.exists():
                try:
                    raw = json.loads(meta_path.read_text(encoding="utf-8"))
                    floor_type = (raw.get("floor_classification", {}).get("floor_type", "unknown") or "unknown").strip().lower()
                except Exception:
                    pass
            if floor_type == "unknown":
                if "parter" in plan.plan_id.lower() or "ground" in plan.plan_id.lower():
                    floor_type = "ground_floor"
                elif "etaj" in plan.plan_id.lower() or "top" in plan.plan_id.lower():
                    floor_type = "top_floor"
            sort_key = 0 if floor_type == "ground_floor" else 1
            enriched_plans.append({"plan": plan, "floor_type": floor_type, "sort": sort_key, "index": i})
        enriched_plans.sort(key=lambda x: x["sort"])
        enriched_by_idx = {p["index"]: p for p in enriched_plans}
        enriched_ordered = [enriched_by_idx[i] for i in order_from_bottom] if order_from_bottom else enriched_plans
        if order_from_bottom and len(enriched_ordered) != len(enriched_plans):
            enriched_ordered = enriched_plans

        def _floor_display_label(pos: int, p_data: dict, basement_idx: int | None, ordered_list: list) -> str:
            if basement_idx is not None and p_data.get("index") == basement_idx:
                return "Keller"
            ft = (p_data.get("floor_type") or "").strip().lower()
            if ft == "ground_floor":
                return "Erdgeschoss"
            if ft == "top_floor":
                return "Mansardă" if len(ordered_list) > 1 else "Dachgeschoss"
            if ft == "intermediate":
                inter_count = sum(1 for i in range(pos + 1) if (ordered_list[i].get("floor_type") or "").strip().lower() == "intermediate")
                return f"{inter_count}. Obergeschoss"
            return "Erdgeschoss" if pos == 0 else f"Obergeschoss {pos}"

        for pos, p_data in enumerate(enriched_ordered):
            plan = p_data["plan"]
            label = _floor_display_label(pos, p_data, basement_plan_index, enriched_ordered)
            plan_id_to_label[plan.plan_id] = label
        # Etichete doar pentru etajele cu acoperiș (fără Keller)
        for pos, p_data in enumerate(enriched_ordered):
            if basement_plan_index is not None and p_data.get("index") == basement_plan_index:
                continue
            label = _floor_display_label(pos, p_data, basement_plan_index, enriched_ordered)
            roof_floor_labels.append(label)
    else:
        roof_floor_labels = ["Erdgeschoss", "1. Obergeschoss / Dachgeschoss", "2. Obergeschoss", "3. Obergeschoss"]

    pdf_dir = out_root / "offer_pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = pdf_dir / f"roof_measurements_{run_id}.pdf"

    metrics_path = out_root / "roof" / "roof_3d" / "entire" / "mixed" / "roof_metrics.json"
    roof_pricing_path = out_root / "roof" / "roof_3d" / "entire" / "mixed" / "roof_pricing.json"
    metrics = {}
    by_floor = {}
    roof_pricing = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
            by_floor = metrics.get("by_floor") or {}
        except Exception as e:
            print(f"⚠️ [PDF ROOF] Eroare la citire roof_metrics.json: {e}")
    else:
        print(f"⚠️ [PDF ROOF] roof_metrics.json lipsește – se generează PDF minimal pentru toți utilizatorii.")
    if roof_pricing_path.exists():
        try:
            with open(roof_pricing_path, encoding="utf-8") as f:
                roof_pricing = json.load(f)
        except Exception:
            roof_pricing = {}

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name="RoofTitle",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=8,
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        name="RoofHeading",
        parent=styles["Heading2"],
        fontSize=12,
        spaceAfter=6,
    )
    body = styles["BodyText"]

    story = []
    story.append(Paragraph("Mengenermittlung", title_style))
    story.append(Spacer(1, 3 * mm))

    unfold_roof_total = (metrics.get("unfold_roof") or {}).get("total") or {}
    unfold_overhang_total = (metrics.get("unfold_overhang") or {}).get("total") or {}
    total_combined = metrics.get("total_combined") or {}
    rp_meas = (roof_pricing.get("roof_measurements") or {}) if isinstance(roof_pricing, dict) else {}

    roof_area_without_overhang_m2 = rp_meas.get("roof_area_without_overhang_m2")
    roof_area_with_overhang_m2 = rp_meas.get("roof_area_with_overhang_m2")
    overhang_area_m2 = rp_meas.get("roof_area_overhang_only_m2")
    if roof_area_without_overhang_m2 is None:
        roof_area_without_overhang_m2 = unfold_roof_total.get("area_m2")
    if overhang_area_m2 is None:
        overhang_area_m2 = unfold_overhang_total.get("area_m2")
    if roof_area_with_overhang_m2 is None:
        if roof_area_without_overhang_m2 is not None and overhang_area_m2 is not None:
            roof_area_with_overhang_m2 = float(roof_area_without_overhang_m2) + float(overhang_area_m2)
        else:
            roof_area_with_overhang_m2 = total_combined.get("area_m2")
    roof_angles_str = "—"
    for candidate in (out_root, RUNNER_ROOT / "output" / run_id, RUNS_ROOT / run_id):
        ang_path = Path(candidate) / "roof" / "roof_3d" / "floor_roof_angles.json"
        if ang_path.exists():
            try:
                ang_data = json.loads(ang_path.read_text(encoding="utf-8"))
                vals = [float(ang_data[k]) for k in sorted(ang_data.keys(), key=lambda x: int(x)) if ang_data.get(k) is not None]
                roof_angles_str = ", ".join(f"{v:.0f}°" for v in vals) if vals else "—"
            except Exception:
                pass
            break
    story.append(Paragraph("<b>Übersicht Dach</b>", heading_style))
    story.append(Paragraph(
        f"<b>Dachfläche (Dämmzone, ohne Überstand):</b> {roof_area_without_overhang_m2:.2f} m²" if roof_area_without_overhang_m2 is not None else "<b>Dachfläche (Dämmzone, ohne Überstand):</b> —",
        body
    ))
    story.append(Paragraph(
        f"<b>Überstand-Fläche (nur Überstand):</b> {overhang_area_m2:.2f} m²" if overhang_area_m2 is not None else "<b>Überstand-Fläche (nur Überstand):</b> —",
        body
    ))
    story.append(Paragraph(f"<b>Dachtyp:</b> {tip_acoperis}", body))
    story.append(Paragraph(f"<b>Dachneigung:</b> {roof_angles_str}", body))
    story.append(Spacer(1, 4 * mm))

    floor_labels_fallback = {
        "0": "Erdgeschoss",
        "1": "1. Obergeschoss / Dachgeschoss",
        "2": "2. Obergeschoss",
        "3": "3. Obergeschoss",
    }

    # Per-rectangle roof measurements (preferred for roof-only flows).
    by_rectangle = ((roof_pricing.get("roof_measurements") or {}).get("by_rectangle") or [])
    if by_rectangle:
        roof_type_label = {
            "0_w": "Flachdach",
            "1_w": "Pultdach",
            "2_w": "Satteldach",
            "4_w": "Walmdach",
            "4.5_w": "Krüppelwalmdach",
        }
        for rec in sorted(by_rectangle, key=lambda r: (int(r.get("floor_idx", 0)), int(r.get("rectangle_idx", 0)))):
            floor_idx = int(rec.get("floor_idx", 0))
            rect_idx = int(rec.get("rectangle_idx", 0))
            floor_label = roof_floor_labels[floor_idx] if floor_idx < len(roof_floor_labels) else f"Etage {floor_idx}"
            roof_name = "Dach" if rect_idx == 0 else f"Dach {rect_idx + 1}"
            story.append(Paragraph(f"<b>{floor_label} – {roof_name}</b>", heading_style))
            rows = [["Position", "Wert", "Einheit"]]
            rows.append(["Dachneigung", f"{float(rec.get('roof_angle_deg')):.1f}" if rec.get("roof_angle_deg") is not None else "—", "°"])
            raw_rt = str(rec.get("roof_type") or "")
            rows.append(["Dachtyp", roof_type_label.get(raw_rt, raw_rt or "—"), "—"])
            rows.append(["Dachfläche gesamt (mit Überstand)", f"{float(rec.get('roof_area_with_overhang_m2')):.2f}" if rec.get("roof_area_with_overhang_m2") is not None else "—", "m²"])
            rows.append(["Dachfläche gedämmt (ohne Überstand)", f"{float(rec.get('roof_area_without_overhang_m2')):.2f}" if rec.get("roof_area_without_overhang_m2") is not None else "—", "m²"])
            rows.append(["Dachumfang (mit Überstand)", f"{float(rec.get('roof_perimeter_with_overhang_m')):.2f}" if rec.get("roof_perimeter_with_overhang_m") is not None else "—", "m"])
            t_rect = Table(rows, colWidths=[90 * mm, 50 * mm, 25 * mm])
            t_rect.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3E2C22")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F5F0E8")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#8B7355")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
            ]))
            story.append(t_rect)
            story.append(Spacer(1, 8 * mm))

    # Per-floor data fallback.
    by_floor_roof = (metrics.get("unfold_roof") or {}).get("by_floor") or by_floor
    if not by_rectangle:
        for floor_key in sorted(by_floor_roof.keys(), key=lambda k: int(k) if str(k).isdigit() else 0):
            data = by_floor_roof[floor_key]
            idx = int(floor_key) if str(floor_key).isdigit() else 0
            label = roof_floor_labels[idx] if idx < len(roof_floor_labels) else floor_labels_fallback.get(str(floor_key), f"Etaj {floor_key}")

            story.append(Paragraph(f"<b>{label}</b>", heading_style))

            area_m2 = data.get("area_m2")
            contour_m = data.get("contour_m")

            rows = [
                ["Element", "Wert", "Einheit"],
                ["Fläche Dämmzone (gedämmte Dachfläche)", f"{area_m2:.2f}" if area_m2 is not None else "—", "m²"],
                ["Eindeckfläche (Dachdeckung)", f"{area_m2:.2f}" if area_m2 is not None else "—", "m²"],
                ["Klempnerarbeiten (Dachumfang)", f"{contour_m:.2f}" if contour_m is not None else "—", "m"],
                ["Fläche Innenausbau (Innenverkleidung)", f"{area_m2:.2f}" if area_m2 is not None else "—", "m²"],
                ["Fläche Unterspannbahn / Folie", f"{area_m2:.2f}" if area_m2 is not None else "—", "m²"],
            ]

            t = Table(rows, colWidths=[90 * mm, 50 * mm, 25 * mm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3E2C22")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F5F0E8")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#8B7355")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
            ]))
            story.append(t)
            story.append(Spacer(1, 10 * mm))

    total_area = unfold_roof_total.get("area_m2")
    total_contour = unfold_roof_total.get("contour_m")
    if (total_area is not None or total_contour is not None) and not by_rectangle:
        story.append(Paragraph("<b>Gesamtsumme (alle Etagen)</b>", heading_style))
        rows_total = [
            ["Element", "Wert", "Einheit"],
            ["Gesamtfläche Dämmung / Eindeckung / Innenausbau / Folie", f"{total_area:.2f}" if total_area is not None else "—", "m²"],
            ["Gesamtumfang Klempnerarbeiten", f"{total_contour:.2f}" if total_contour is not None else "—", "m"],
        ]
        t2 = Table(rows_total, colWidths=[90 * mm, 50 * mm, 25 * mm])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3E2C22")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#E8DCC8")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#8B7355")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
        ]))
        story.append(t2)

    # Secțiune nouă: Übersicht Gebäude (Wände, Böden, Decken, Öffnungen) din pricing_raw.json
    floors_data = _load_floor_data_from_pricing(out_root)
    if floors_data and not roof_only:
        story.append(PageBreak())
        story.append(Paragraph("Übersicht Gebäude", title_style))
        story.append(Paragraph(
            "Netto-Flächen für Strukturen, Innenausbau, Böden und Decken sowie Anzahl und Gesamtfläche der Öffnungen, "
            "sortiert nach Stockwerken. Basierend auf den Planungsdaten.",
            body
        ))
        story.append(Spacer(1, 8 * mm))
        for idx, fd in enumerate(floors_data):
            label = plan_id_to_label.get(fd.get("plan_id", ""), floor_labels_fallback.get(str(idx), f"Stockwerk {idx}"))
            story.append(Paragraph(f"<b>{label}</b>", heading_style))
            rows = [
                ["Element", "Wert", "Einheit"],
                ["Strukturen Innenwände (netto)", f"{fd['structure_int_net_m2']:.2f}", "m²"],
                ["Strukturen Außenwände (netto)", f"{fd['structure_ext_net_m2']:.2f}", "m²"],
                ["Innenausbau Innen (netto)", f"{fd['finish_int_net_m2']:.2f}", "m²"],
                ["Innenausbau Außen (netto)", f"{fd['finish_ext_net_m2']:.2f}", "m²"],
                ["Bodenfläche / Planché", f"{fd['floor_area_m2']:.2f}", "m²"],
                ["Deckenfläche", f"{fd['ceiling_area_m2']:.2f}", "m²"],
                ["Anzahl Türen", str(fd['num_doors']), "Stk."],
                ["Anzahl Fenster", str(fd['num_windows']), "Stk."],
                ["Gesamtfläche Öffnungen", f"{fd['openings_area_m2']:.2f}", "m²"],
            ]
            t = Table(rows, colWidths=[90 * mm, 50 * mm, 25 * mm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3E2C22")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F5F0E8")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#8B7355")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
            ]))
            story.append(t)
            story.append(Spacer(1, 10 * mm))

    doc.build(story)
    print(f"✅ [PDF ROOF] Generat: {output_path}")
    return output_path
