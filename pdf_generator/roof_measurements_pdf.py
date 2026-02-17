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

from config.settings import OUTPUT_ROOT, RUNNER_ROOT, JOBS_ROOT


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

    metrics_path = out_root / "roof" / "roof_3d" / "entire" / "mixed" / "roof_metrics.json"
    if not metrics_path.exists():
        print(f"⚠️ [PDF ROOF] roof_metrics.json lipsește: {metrics_path}")
        return None

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    by_floor = metrics.get("by_floor") or {}
    if not by_floor:
        print(f"⚠️ [PDF ROOF] by_floor gol în roof_metrics.json")
        return None

    pdf_dir = out_root / "offer_pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = pdf_dir / f"roof_measurements_{run_id}.pdf"

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
    )
    heading_style = ParagraphStyle(
        name="RoofHeading",
        parent=styles["Heading2"],
        fontSize=12,
        spaceAfter=6,
    )
    body = styles["BodyText"]

    story = []
    story.append(Paragraph("Dämmung & Dachdeckung – Dachmaße", title_style))
    story.append(Paragraph(
        "Detaillierte Messungen der Dachflächen für Dämmung, Eindeckung, Klempnerarbeiten und Folien. "
        "Alle Angaben basieren auf der 3D-Roof-Analyse.",
        body
    ))
    story.append(Spacer(1, 8 * mm))

    floor_labels = {
        "0": "Erdgeschoss",
        "1": "1. Obergeschoss / Dachgeschoss",
        "2": "2. Obergeschoss",
        "3": "3. Obergeschoss",
    }

    for floor_key in sorted(by_floor.keys(), key=lambda k: int(k) if str(k).isdigit() else 0):
        data = by_floor[floor_key]
        label = floor_labels.get(str(floor_key), f"Etaj {floor_key}")

        story.append(Paragraph(f"<b>{label}</b>", heading_style))

        area_m2 = data.get("area_m2")
        contour_m = data.get("contour_m")

        rows = [
            ["Kennzahl", "Wert", "Einheit"],
            ["Fläche Dämmzone (gedämmte Dachfläche)", f"{area_m2:.2f}" if area_m2 is not None else "—", "m²"],
            ["Eindeckfläche (Dachdeckung)", f"{area_m2:.2f}" if area_m2 is not None else "—", "m²"],
            ["Klempnerarbeiten (Dachumfang)", f"{contour_m:.2f}" if contour_m is not None else "—", "m"],
            ["Fläche Finisagen (Innenverkleidung)", f"{area_m2:.2f}" if area_m2 is not None else "—", "m²"],
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

    total = metrics.get("total") or {}
    total_area = total.get("area_m2")
    total_contour = total.get("contour_m")
    if total_area is not None or total_contour is not None:
        story.append(Paragraph("<b>Gesamtsumme (alle Etagen)</b>", heading_style))
        rows_total = [
            ["Kennzahl", "Wert", "Einheit"],
            ["Gesamtfläche Dämmung / Eindeckung / Finisagen / Folie", f"{total_area:.2f}" if total_area is not None else "—", "m²"],
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

    # Secțiune nouă: Bauteilmaße (Wände, Böden, Decken, Öffnungen) din pricing_raw.json
    floors_data = _load_floor_data_from_pricing(out_root)
    if floors_data:
        story.append(PageBreak())
        story.append(Paragraph("Bauteilmaße (Wände, Böden, Decken, Öffnungen)", title_style))
        story.append(Paragraph(
            "Netto-Flächen für Strukturen, Finisagen, Böden und Decken sowie Anzahl und Gesamtfläche der Öffnungen, "
            "sortiert nach Etagen. Basierend auf den Planungsdaten.",
            body
        ))
        story.append(Spacer(1, 8 * mm))
        for idx, fd in enumerate(floors_data):
            label = floor_labels.get(str(idx), f"Etaj {idx}")
            story.append(Paragraph(f"<b>{label}</b>", heading_style))
            rows = [
                ["Kennzahl", "Wert", "Einheit"],
                ["Strukturen Innenwände (netto)", f"{fd['structure_int_net_m2']:.2f}", "m²"],
                ["Strukturen Außenwände (netto)", f"{fd['structure_ext_net_m2']:.2f}", "m²"],
                ["Finisagen Innen (netto)", f"{fd['finish_int_net_m2']:.2f}", "m²"],
                ["Finisagen Außen (netto)", f"{fd['finish_ext_net_m2']:.2f}", "m²"],
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
