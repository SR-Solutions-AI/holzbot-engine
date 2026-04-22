from __future__ import annotations
import json
import io
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from datetime import datetime
import random
import contextvars

# --- NEW: OpenAI Import ---
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfmetrics import stringWidth

from PIL import Image as PILImage, ImageEnhance, ImageOps

from config.settings import load_plan_infos, PlansListError, RUNNER_ROOT, PROJECT_ROOT, RUNS_ROOT, OUTPUT_ROOT, JOBS_ROOT
from config.frontend_loader import load_frontend_data_for_run
from pdf_generator.utils import resolve_plan_image_for_pdf
from branding.db_loader import fetch_tenant_branding

from .offer_scope import (
    normalize_nivel_oferta as _normalize_nivel_oferta,
    get_offer_inclusions as _get_offer_inclusions,
    roof_items_for_pdf_table as _roof_items_for_pdf_table,
    baukosten_position_label_de as _baukosten_position_label_de,
)
from .project_overview_fields import (
    build_selected_form_overview_items,
    merge_neues_geschoss_plan1_finishes_into_erdgeschoss,
    merge_plan0_form_keys_into_ground_display,
)
from .roof_measurements_pdf import _load_floor_plan_order_and_labels_from_manifest


def _load_aufstockung_floor_kinds(run_id: str, frontend_data: dict, job_root: Path | None) -> list[str]:
    """Best-effort floor kinds in raw plan index order."""
    extras: dict = {}
    if job_root:
        extras_path = Path(job_root) / "detections_review_extras.json"
        if extras_path.exists():
            try:
                extras = json.loads(extras_path.read_text(encoding="utf-8"))
            except Exception:
                extras = {}
    kinds_raw = (
        (extras.get("floorKinds") if isinstance(extras, dict) else None)
        or (frontend_data.get("floorKinds") if isinstance(frontend_data, dict) else None)
        or (frontend_data.get("aufstockungFloorKinds") if isinstance(frontend_data, dict) else None)
        or ((frontend_data.get("structuraCladirii") or {}).get("aufstockungFloorKinds") if isinstance(frontend_data, dict) else None)
        or []
    )
    if not isinstance(kinds_raw, list):
        return []
    out: list[str] = []
    for k in kinds_raw:
        v = str(k).strip().lower()
        if v in ("new", "zubau", "aufstockung"):
            # Editor sends "new" for an extension storey; treat as Zubau (aligned with roof_measurements_pdf + pricing UI).
            out.append(v if v != "new" else "zubau")
        else:
            out.append("existing")
    return out


def _floor_labels_equivalent(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a.strip().lower() == b.strip().lower():
        return True
    return bool(_expand_floor_label_for_form_omit(a) & _expand_floor_label_for_form_omit(b))


def _canonical_finishes_floor_key(label: str) -> str | None:
    """Map manifest / UI label to keys used in _finishes_per_floor_from_form."""
    if not (label or "").strip():
        return None
    candidates = (
        ["Keller", "Erdgeschoss"]
        + [f"Obergeschoss {i}" for i in range(1, 10)]
        + ["Mansardă", "Dachgeschoss"]
    )
    for cand in candidates:
        if _floor_labels_equivalent(label, cand):
            return cand
    return None


def _wand_suffix_for_canonical_floor(canon: str) -> str | None:
    """Middle part for wand/boden keys (außenwande_*, innenwande_*, bodenaufbau_*), aligned with project_overview_fields static rows."""
    c = (canon or "").strip()
    if not c:
        return None
    low = c.lower()
    if c == "Erdgeschoss" or low == "erdgeschoss":
        return "ground"
    m = re.match(r"^Obergeschoss\s+(\d+)$", c, re.I)
    if m:
        return f"floor_{max(1, int(m.group(1)))}"
    if c in ("Mansardă", "Dachgeschoss") or "mansard" in low:
        return "Mansarda"
    if c == "Keller" or low == "keller":
        return "Beci"
    if "zubau" in low and "plan" in low:
        return "plan_0"
    return None


def _wand_außen_key(suffix: str) -> str:
    if suffix == "Mansarda":
        return "außenwandeMansarda"
    if suffix == "Beci":
        return "außenwandeBeci"
    return f"außenwande_{suffix}"


def _wand_innen_key(suffix: str) -> str:
    if suffix == "Mansarda":
        return "innenwandeMansarda"
    if suffix == "Beci":
        return "innenwandeBeci"
    return f"innenwande_{suffix}"


def _boden_decken_belag_key(category: str, suffix: str) -> str | None:
    if suffix == "Mansarda":
        if category == "bodenaufbau":
            return "bodenaufbauMansarda"
        if category == "deckenaufbau":
            return "deckenaufbauMansarda"
        if category == "bodenbelag":
            return "bodenbelagMansarda"
        return None
    if suffix == "Beci":
        if category == "deckenaufbau":
            return "deckenaufbauBeci"
        if category == "bodenbelag":
            return "bodenbelagBeci"
        return None
    if category == "bodenaufbau":
        return f"bodenaufbau_{suffix}"
    if category == "deckenaufbau":
        return f"deckenaufbau_{suffix}"
    if category == "bodenbelag":
        return f"bodenbelag_{suffix}"
    return None


def _material_from_pricing_breakdown_item(item: dict) -> str:
    m = item.get("material")
    if m is not None and str(m).strip():
        return str(m).strip()
    name = str(item.get("name") or "")
    mm = re.search(r"\(([^)]+)\)\s*$", name)
    if mm:
        return mm.group(1).strip()
    return ""


def _floor_label_for_pricing_breakdown_row(entry: dict, item: dict) -> str:
    fl = str(item.get("floor_label") or "").strip()
    if fl:
        return fl
    pricing = entry.get("pricing") or {}
    fins = ((pricing.get("breakdown") or {}).get("finishes") or {}).get("detailed_items") or []
    for fit in fins:
        if not isinstance(fit, dict):
            continue
        fl2 = str(fit.get("floor_label") or "").strip()
        if fl2:
            return fl2
    ent = str(entry.get("floor_label") or "").strip()
    if ent:
        return ent
    entry_type = (entry.get("type") or "").strip().lower()
    if entry_type == "ground_floor":
        return "Erdgeschoss"
    if "top" in entry_type or "mansard" in entry_type:
        return "Dachgeschoss"
    return "Obergeschoss"


def _wand_boden_overlay_from_plans_pricing(frontend_data: dict, plans_data: list | None) -> tuple[dict, dict]:
    """Fill empty per-floor wand/boden keys from pricing breakdown (same source as Innenausbau in Projektübersicht)."""
    wa = dict(merge_plan0_form_keys_into_ground_display(dict(frontend_data.get("wandaufbau") or {})))
    bd = dict(merge_plan0_form_keys_into_ground_display(dict(frontend_data.get("bodenDeckeBelag") or {})))
    for entry in plans_data or []:
        if not isinstance(entry, dict):
            continue
        breakdown = (entry.get("pricing") or {}).get("breakdown") or {}
        structure = breakdown.get("structure_walls") or {}
        for item in structure.get("detailed_items") or []:
            if not isinstance(item, dict):
                continue
            cat = str(item.get("category") or "")
            if cat not in ("walls_structure_int", "walls_structure_ext"):
                continue
            mat = _material_from_pricing_breakdown_item(item)
            if not mat:
                continue
            raw_lab = _floor_label_for_pricing_breakdown_row(entry, item)
            canon = _canonical_finishes_floor_key(raw_lab) or raw_lab.strip()
            suf = _wand_suffix_for_canonical_floor(canon)
            if not suf:
                continue
            if cat == "walls_structure_int":
                key = _wand_innen_key(suf)
                if not wa.get(key) or str(wa.get(key)).strip() == "":
                    wa[key] = mat
            else:
                key = _wand_außen_key(suf)
                if not wa.get(key) or str(wa.get(key)).strip() == "":
                    wa[key] = mat
        floors = breakdown.get("floors_ceilings") or {}
        for item in floors.get("detailed_items") or []:
            if not isinstance(item, dict):
                continue
            cat = str(item.get("category") or "")
            if cat not in ("bodenaufbau", "deckenaufbau", "bodenbelag"):
                continue
            mat = _material_from_pricing_breakdown_item(item)
            if not mat:
                continue
            raw_lab = _floor_label_for_pricing_breakdown_row(entry, item)
            canon = _canonical_finishes_floor_key(raw_lab) or raw_lab.strip()
            suf = _wand_suffix_for_canonical_floor(canon)
            if not suf:
                continue
            key = _boden_decken_belag_key(cat, suf)
            if not key:
                continue
            if not bd.get(key) or str(bd.get(key)).strip() == "":
                bd[key] = mat
    wa = merge_plan0_form_keys_into_ground_display(wa)
    bd = merge_plan0_form_keys_into_ground_display(bd)
    return wa, bd


def _finish_and_wandboden_keys_for_manifest_index(
    manifest_plan_index: int, *, manifest_floor_kind: str | None = None
) -> tuple[str, str, str, str, str, str]:
    """
    Map plan index in manifest (floorKinds order) to form dict keys for finishes + Wandaufbau/Boden.
    Aligns with pricing / Preisdatenbank: index 0 → Erdgeschoss, 1 → Obergeschoss 1, etc.
    Zubau pe primul plan: chei `plan_0` (editor), nu `ground`, ca să nu se suprapună cu EG.
    """
    fk0 = str(manifest_floor_kind or "").strip().lower()
    if int(manifest_plan_index) == 0 and fk0 == "zubau":
        return (
            "Neues Geschoss (Plan 1)",
            "außenwande_plan_0",
            "innenwande_plan_0",
            "bodenaufbau_plan_0",
            "deckenaufbau_plan_0",
            "bodenbelag_plan_0",
        )
    tiers: list[tuple[str, str, str, str, str, str]] = [
        (
            "Erdgeschoss",
            "außenwande_ground",
            "innenwande_ground",
            "bodenaufbau_ground",
            "deckenaufbau_ground",
            "bodenbelag_ground",
        ),
        (
            "Obergeschoss 1",
            "außenwande_floor_1",
            "innenwande_floor_1",
            "bodenaufbau_floor_1",
            "deckenaufbau_floor_1",
            "bodenbelag_floor_1",
        ),
        (
            "Obergeschoss 2",
            "außenwande_floor_2",
            "innenwande_floor_2",
            "bodenaufbau_floor_2",
            "deckenaufbau_floor_2",
            "bodenbelag_floor_2",
        ),
        (
            "Obergeschoss 3",
            "außenwande_floor_3",
            "innenwande_floor_3",
            "bodenaufbau_floor_3",
            "deckenaufbau_floor_3",
            "bodenbelag_floor_3",
        ),
        (
            "Dachgeschoss",
            "außenwandeMansarda",
            "innenwandeMansarda",
            "bodenaufbauMansarda",
            "deckenaufbauMansarda",
            "bodenbelagMansarda",
        ),
    ]
    i = max(0, int(manifest_plan_index))
    if i >= len(tiers):
        return tiers[-1]
    return tiers[i]


def _plan_id_to_aufstockung_kind_and_label(
    run_id: str,
    plan_infos_ordered: list,
    job_root: Path | None,
    frontend_data: dict,
) -> tuple[dict[str, str], dict[str, str]]:
    """Align floorKinds / labels with plan_id (plans_list order), not enriched sort index."""
    kinds = _load_aufstockung_floor_kinds(run_id, frontend_data, job_root)
    pid_to_kind: dict[str, str] = {}
    for i, pinfo in enumerate(plan_infos_ordered):
        k = str(kinds[i]).strip().lower() if i < len(kinds) else "existing"
        pid_to_kind[pinfo.plan_id] = k
    _, manifest_labels = _load_floor_plan_order_and_labels_from_manifest(run_id, job_root)
    pid_to_label: dict[str, str] = {}
    for i, pinfo in enumerate(plan_infos_ordered):
        lab = ""
        if manifest_labels and i < len(manifest_labels):
            lab = str(manifest_labels[i] or "").strip()
        fk = pid_to_kind.get(pinfo.plan_id, "existing")
        # If editor marks floor as Zubau/Aufstockung, prefer explicit label over generic EG/OG labels.
        if fk in ("zubau", "aufstockung"):
            generic = {"erdgeschoss", "dachgeschoss", "mansardă", "mansarda", f"stockwerk {i + 1}"}
            if not lab or lab.strip().lower() in generic or "obergeschoss" in lab.strip().lower():
                lab = "Zubau" if fk == "zubau" else "Aufstockung"
        if not lab:
            lab = f"Stockwerk {i + 1}"
        pid_to_label[pinfo.plan_id] = lab
    return pid_to_kind, pid_to_label


def _append_extension_form_selection_section(
    story,
    styles,
    frontend_data: dict,
    plans_data: list,
    enforcer: GermanEnforcer,
) -> None:
    """Show wand/boden/finish form choices for Zubau / Aufstockung storeys (customer PDF)."""
    wp = str((frontend_data or {}).get("wizard_package") or "").strip().lower()
    if wp not in ("aufstockung", "zubau", "zubau_aufstockung"):
        return
    wa = frontend_data.get("wandaufbau", {}) or {}
    bd = frontend_data.get("bodenDeckeBelag", {}) or {}
    finishes_all = _finishes_per_floor_from_form(frontend_data.get("materialeFinisaj", {}) or {})

    def _emit(
        parts: list,
        prefix: str,
        mapping: list[tuple[str, str, str | None]],
        values: dict,
    ) -> None:
        for key, label_de, field_tag in mapping:
            val = values.get(key)
            if val is None or str(val).strip() == "":
                continue
            rendered = str(val).strip()
            if field_tag:
                rendered = enforcer.get(rendered) or rendered
            parts.append(
                Paragraph(
                    f"<b>{prefix} ({label_de}):</b> <b>{rendered}</b>",
                    styles["Body"],
                )
            )

    def _emit_fb(
        parts: list,
        prefix: str,
        label_de: str,
        field_tag: str | None,
        values: dict,
        primary_key: str,
        fallback_key: str | None,
    ) -> None:
        raw = values.get(primary_key)
        if (raw is None or str(raw).strip() == "") and fallback_key:
            raw = values.get(fallback_key)
        if raw is None or str(raw).strip() == "":
            return
        rendered = str(raw).strip()
        if field_tag:
            rendered = enforcer.get(rendered) or rendered
        parts.append(
            Paragraph(
                f"<b>{prefix} ({label_de}):</b> <b>{rendered}</b>",
                styles["Body"],
            )
        )

    for entry in plans_data or []:
        fk = str(entry.get("floor_kind") or "").strip().lower()
        if fk not in ("zubau", "aufstockung"):
            continue
        fl_raw = str(entry.get("floor_label") or "").strip()
        if not fl_raw:
            continue
        mi = int(entry.get("manifest_plan_index", 0))
        tier_finish, wa_out_k, wa_in_k, bd_bo_k, bd_de_k, bd_be_k = _finish_and_wandboden_keys_for_manifest_index(
            mi, manifest_floor_kind=fk
        )
        parts: list = []
        title = "Zubau" if fk == "zubau" else "Aufstockung"
        label_head = fl_raw if fl_raw.lower() != title.lower() else title
        fin = dict(finishes_all.get(tier_finish, {}) or {}) if tier_finish else {}
        # Date vechi / popup: primul plan Zubau poate fi încă pe chei *_ground (înainte de plan_0); PDF citea doar plan_0 → gol.
        if mi == 0 and fk == "zubau":
            eg_fin = finishes_all.get("Erdgeschoss") or {}
            for subk in ("interior_inner", "interior_outer", "exterior"):
                cur = fin.get(subk)
                if (cur is None or str(cur).strip() == "") and eg_fin.get(subk):
                    fin[subk] = eg_fin[subk]
        if fin.get("interior_inner"):
            parts.append(
                Paragraph(
                    f"<b>Innenausbau Innenwände ({fl_raw}):</b> <b>{enforcer.get(str(fin['interior_inner'])) or fin['interior_inner']}</b>",
                    styles["Body"],
                )
            )
        if fin.get("interior_outer"):
            parts.append(
                Paragraph(
                    f"<b>Innenausbau Außenwände ({fl_raw}):</b> <b>{enforcer.get(str(fin['interior_outer'])) or fin['interior_outer']}</b>",
                    styles["Body"],
                )
            )
        if fin.get("exterior"):
            parts.append(
                Paragraph(
                    f"<b>{enforcer.get('Fațadă')} ({fl_raw}):</b> <b>{enforcer.get(str(fin['exterior'])) or fin['exterior']}</b>",
                    styles["Body"],
                )
            )

        if mi == 0 and fk == "zubau":
            _emit_fb(parts, "Wandaufbau Außenwände", fl_raw, "wandaufbau_aussen", wa, wa_out_k, "außenwande_ground")
            _emit_fb(parts, "Wandaufbau Innenwände", fl_raw, "wandaufbau_innen", wa, wa_in_k, "innenwande_ground")
            _emit_fb(parts, "Bodenaufbau", fl_raw, "bodenaufbau", bd, bd_bo_k, "bodenaufbau_ground")
            _emit_fb(parts, "Deckenaufbau", fl_raw, "deckenaufbau", bd, bd_de_k, "deckenaufbau_ground")
            _emit_fb(parts, "Bodenbelag", fl_raw, "bodenbelag", bd, bd_be_k, "bodenbelag_ground")
        else:
            if wa_out_k:
                _emit(parts, "Wandaufbau Außenwände", [(wa_out_k, fl_raw, "wandaufbau_aussen")], wa)
            if wa_in_k:
                _emit(parts, "Wandaufbau Innenwände", [(wa_in_k, fl_raw, "wandaufbau_innen")], wa)
            if bd_bo_k:
                _emit(parts, "Bodenaufbau", [(bd_bo_k, fl_raw, "bodenaufbau")], bd)
            if bd_de_k:
                _emit(parts, "Deckenaufbau", [(bd_de_k, fl_raw, "deckenaufbau")], bd)
            if bd_be_k:
                _emit(parts, "Bodenbelag", [(bd_be_k, fl_raw, "bodenbelag")], bd)

        if not parts:
            continue
        story.append(Spacer(1, 2 * mm))
        story.append(
            Paragraph(
                f"<b>{title} – {label_head}: gewählte Ausführung (Formular)</b>",
                styles["H3"],
            )
        )
        for p in parts:
            story.append(p)


def _load_detections_review_floor_labels_list(run_id: str, job_root: Path | None) -> list[str]:
    paths: list[Path] = []
    if job_root:
        paths.append(Path(job_root) / "detections_review_floor_labels.json")
    paths.append(JOBS_ROOT / run_id / "detections_review_floor_labels.json")
    for p in paths:
        if not p.exists():
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
    return []


def _expand_floor_label_for_form_omit(de_label: str) -> set[str]:
    """Map job/manifest floor labels to keys used in Projektübersicht (finishes / Wandaufbau / Boden)."""
    s = (de_label or "").strip()
    out: set[str] = set()
    if not s:
        return out
    out.add(s)
    low = s.lower()
    if low == "keller":
        out.add("Keller")
    if low == "erdgeschoss":
        out.add("Erdgeschoss")
    m = re.match(r"^(\d+)\.\s*Obergeschoss(?:\s*/\s*Dachgeschoss)?$", s, re.I)
    if m:
        n = int(m.group(1))
        out.add(f"Obergeschoss {n}")
    m2 = re.match(r"^Obergeschoss\s+(\d+)$", s, re.I)
    if m2:
        n = int(m2.group(1))
        out.add(f"{n}. Obergeschoss")
    if "dachgeschoss" in low:
        out.add("Dachgeschoss")
    if "mansard" in low:
        out.add("Mansardă")
        out.add("Dachgeschoss")
    return out


def _merge_finishes_per_floor_by_canonical(finishes: dict) -> dict:
    """Unifică chei echivalente (ex. «1. Obergeschoss» vs «Obergeschoss 1») ca să nu apară același etaj de două ori în Projektübersicht."""
    if not finishes:
        return {}
    merged: dict[str, dict] = {}
    for k, v in finishes.items():
        ks = str(k or "").strip()
        if not ks:
            continue
        ck = _canonical_finishes_floor_key(ks) or ks
        if not isinstance(v, dict):
            v = {}
        if ck not in merged:
            merged[ck] = {"interior_inner": None, "interior_outer": None, "exterior": None}
        tgt = merged[ck]
        for field in ("interior_inner", "interior_outer", "exterior"):
            if tgt.get(field) is None and v.get(field):
                tgt[field] = v[field]
    return merged


def _overview_floor_key_matches_omit(key: str, omit: frozenset[str]) -> bool:
    if not key or not omit:
        return False
    k = key.strip()
    if k in omit:
        return True
    m = re.match(r"^Obergeschoss\s+(\d+)$", k, re.I)
    if m:
        if f"{m.group(1)}. Obergeschoss" in omit:
            return True
    if k == "Mansardă":
        return bool(omit & {"Mansardă", "Dachgeschoss"})
    if k == "Dachgeschoss":
        return bool(omit & {"Dachgeschoss", "Mansardă"})
    return False


def _aufstockung_form_overview_omit_floor_labels(
    run_id: str,
    frontend_data: dict | None,
    job_root: Path | None,
) -> frozenset[str]:
    """Do not omit any storey from Projektübersicht form lines.

    Older behaviour hid Bestand (existing) storeys for Zubau/Aufstockung, which removed Erdgeschoss
    Wandaufbau/Boden/Finishes from the customer PDF while Obergeschoss rows still appeared — confusing.
    """
    return frozenset()


def _popup_price_key_label_de(price_key: str) -> str:
    key = str(price_key or "").strip().lower()
    labels = {
        "aufstockung_demolition_roof_basic_m2": "Dach-Rueckbau (Standard/Flach)",
        "aufstockung_demolition_roof_complex_m2": "Dach-Rueckbau (Komplex/Steil)",
        "aufstockung_demolition_roof_special_m2": "Dach-Rueckbau (Sonderlage)",
        "aufstockung_stair_opening_piece": "Treppenoeffnung (Stueck)",
        "aufstockung_stair_opening_m2": "Treppenoeffnung",
    }
    return labels.get(key, price_key)


def _build_editor_popup_overview_items(frontend_data: dict, plans_data: list, enforcer: "GermanEnforcer") -> list[str]:
    """Expose all key selections from detections editor popup in customer offer."""
    if not isinstance(frontend_data, dict):
        return []
    phase1 = frontend_data.get("aufstockungPhase1")
    if not isinstance(phase1, dict):
        return []

    idx_to_label: dict[int, str] = {}
    for i, entry in enumerate(plans_data or []):
        try:
            idx = int(entry.get("manifest_plan_index", i))
        except Exception:
            idx = i
        lab = str(entry.get("floor_label") or f"Stockwerk {i + 1}").strip()
        idx_to_label[idx] = lab

    items: list[str] = []

    def _kind_label(v: str) -> str:
        vv = str(v or "").strip().lower()
        if vv == "zubau":
            return "Zubau"
        if vv == "aufstockung":
            return "Aufstockung"
        return "Bestand"

    floors = []
    for key in ("existingFloors", "newFloors"):
        vals = phase1.get(key)
        if isinstance(vals, list):
            floors.extend(vals)

    for floor in floors:
        if not isinstance(floor, dict):
            continue
        try:
            pidx = int(floor.get("plan_index"))
        except Exception:
            continue
        fl = idx_to_label.get(pidx, f"Stockwerk {pidx + 1}")
        fk = _kind_label(str(floor.get("floorKind") or "existing"))
        items.append(f"<b>Editor-Popup ({fl}):</b> <b>{fk}</b>")

        demo = floor.get("demolitionSelections")
        if isinstance(demo, list):
            for d in demo:
                if not isinstance(d, dict):
                    continue
                area = float(d.get("area_m2") or 0.0)
                if area <= 0:
                    continue
                pk = _popup_price_key_label_de(str(d.get("price_key") or ""))
                items.append(f"• Rueckbau: {enforcer.get(pk) or pk} – {area:.2f} m²")

        stairs = floor.get("stairOpenings")
        if isinstance(stairs, list) and stairs:
            items.append(f"• Treppenoeffnungen: {len(stairs)}")
            for s in stairs:
                if not isinstance(s, dict):
                    continue
                pk = _popup_price_key_label_de(str(s.get("price_key") or ""))
                w = s.get("opening_width_m")
                l = s.get("opening_length_m")
                if isinstance(w, (int, float)) and isinstance(l, (int, float)) and w > 0 and l > 0:
                    items.append(f"  - {enforcer.get(pk) or pk}: {float(w):.2f} x {float(l):.2f} m")
                else:
                    items.append(f"  - {enforcer.get(pk) or pk}")

        cdp = floor.get("customDemolitionPrice")
        if isinstance(cdp, (int, float)) and float(cdp) > 0:
            items.append(f"• Pauschalpreis Rueckbau: {_money(float(cdp))}")

        statik = floor.get("statikChoice")
        if isinstance(statik, dict):
            mode = str(statik.get("mode") or "").strip().lower()
            if mode in ("stahlbetonverbunddecke", "sonderkonstruktion"):
                mode_label = "Stahlbetonverbunddecke" if mode == "stahlbetonverbunddecke" else "Sonderkonstruktion"
                items.append(f"• Statik: {mode_label}")
                cpp = statik.get("customPiecePrice")
                if mode == "sonderkonstruktion" and isinstance(cpp, (int, float)) and float(cpp) > 0:
                    items.append(f"  - Preis pro Stueck: {_money(float(cpp))}")

        z_cnt = floor.get("zubauBestandPolygonCount")
        if isinstance(z_cnt, int) and z_cnt > 0:
            items.append(f"• Aufstandsflaeche Marker: {z_cnt}")
        z_lines = floor.get("zubauWallDemolitionLines")
        if isinstance(z_lines, list) and z_lines:
            total_len = 0.0
            for ln in z_lines:
                if isinstance(ln, dict) and isinstance(ln.get("length_m"), (int, float)):
                    total_len += float(ln.get("length_m"))
            if total_len > 0:
                items.append(f"• Zubau Wandrueckbau: {len(z_lines)} Linien, {total_len:.2f} m")

    gcp = phase1.get("globalCombinedPrice")
    if isinstance(gcp, (int, float)) and float(gcp) > 0:
        items.append(f"<b>Editor-Popup Gesamtpreis:</b> <b>{_money(float(gcp))}</b>")

    return items


# ---------- 1. DICTIONAR STATIC EXTINS (CONSTANTE) ----------
STATIC_TRANSLATIONS = {
    # Unități
    "buc": "Stk.",
    "buc.": "Stk.",
    "bucata": "Stk.",
    "piece": "Stk.",
    "pieces": "Stk.",
    "mp": "m²",
    "m2": "m²",
    "ml": "m",
    "kg": "kg",
    "tone": "t",
    "ora": "Std.",
    "manopera": "Arbeitsleistung",
    
    # Structură & Nivele
    "Parter": "Erdgeschoss",
    "Ground Floor": "Erdgeschoss",
    "ground_floor": "Erdgeschoss",
    "Etaj": "Obergeschoss",
    "top_floor": "Obergeschoss / Dachgeschoss",
    "Etaj 1": "1. Obergeschoss",
    "Mansarda": "Dachgeschoss",
    "Acoperis": "Dach",
    "Dachfläche": "Dachfläche",
    "Fundație": "Fundament",
    "Fundament (Sockel)": "Fundament (Sockel)",
    "Placa": "Bodenplatte",
    
    # ELEMENTE SCAPATE ANTERIOR (FIX)
    "Structură Tavan": "Deckenkonstruktion",
    "Structura Tavan": "Deckenkonstruktion",
    "Bodenstruktur": "Bodenaufbau",
    "Deckenstruktur": "Deckenaufbau",
    "Structura Podea": "Bodenkonstruktion",
    
    # Elemente Constructive
    "Pereti": "Wände",
    "Pereti Exteriori": "Außenverkleidung",
    "Pereti Interiori": "Innenverkleidung",
    "Planseu": "Geschossdecke",
    "Grinda": "Holzbalken",
    "Stalp": "Stütze",
    "Scara": "Treppe",
    "Balustrada": "Geländer",
    
    # Materiale
    "Beton": "Beton",
    "Lemn": "Holz",
    "Caramida": "Ziegel",
    "Fier": "Stahl",
    "Vata": "Dämmung",
    "Rigips": "Gipskarton",
    
    # Categorii Oferta
    "Structură": "Rohbau / Konstruktion",
    "Arhitectura": "Architektur",
    "Instalatii": "Haustechnik",
    "Finisaje": "Innenausbau",
    "Casă completă": "Schlüsselfertig",
    "La rosu": "Rohbau",
    
    # Utilități
    "Electrice": "Elektroinstallation",
    "Sanitare": "Sanitärinstallation",
    "Termice": "Heizungstechnik",
    "Canalizare": "Abwasser",
    
    # Deschideri
    "Fereastra": "Fenster",
    "Usa": "Tür",
    "Usa intrare": "Haustür",
    "Dublu": "Zweiflügelig",
    "Simplu": "Einflügelig",
    "Double": "Zweiflügelig",
    "Single": "Einflügelig",
    
    # Diverse & Hardcoded Labels (Extins)
    "Total": "Gesamt",
    "Pret": "Preis",
    "Cantitate": "Menge",
    "Descriere": "Beschreibung",
    "Angebot": "Angebot", 
    "Nr.": "Nr.",
    "Bauherr / Kunde": "Bauherr / Kunde",
    "Ort / Bauort": "Ort / Bauort",
    "Telefon": "Telefon",
    "E-Mail": "E-Mail",
    "Bauvorhaben": "Bauvorhaben",
    "Angebot für Ihr Chiemgauer Massivholzhaus": "Angebot für Ihr Holzhaus",
    "Angebot für Ihr Chiemgauer Holzhaus": "Angebot für Ihr Holzhaus",
    "Angebot für Ihr Holzhaus": "Angebot für Ihr Holzhaus",
    "Sehr geehrte Damen und Herren,": "Sehr geehrte Damen und Herren,", 
    "vielen Dank für Ihre Anfrage. Nachfolgend erhalten Sie unsere detaillierte Kostenschätzung.": "vielen Dank für Ihre Anfrage. Nachfolgend erhalten Sie unsere detaillierte Kostenschätzung.",
    "HINWEIS: Unverbindliche Kostenschätzung. Kein verbindliches Angebot.": "HINWEIS: Unverbindliche Kostenschätzung. Kein verbindliches Angebot.",
    "Planungsebene": "Planungsstand",
    "Gesamtkostenzusammenstellung": "Gesamtkostenzusammenstellung",
    "Baukosten (Konstruktion, Ausbau, Technik)": "Baukosten (Konstruktion, Ausbau, Technik)",
    "Baustelleneinrichtung, Logistik & Planung (10%)": "Baustelleneinrichtung, Logistik & Planung",
    "Bauleitung, Koordination & Gewinn (10%)": "Baustelleneinrichtung, Logistik & Planung",
    "Nettosumme (exkl. MwSt.)": "Nettosumme (ohne MwSt.)",
    "Nettosumme (ohne MwSt.)": "Nettosumme (ohne MwSt.)",
    "MwSt. (19%)": "MwSt. (19%)",
    "GESAMTSUMME BRUTTO": "GESAMTSUMME BRUTTO",
    "Position": "Position",
    "Betrag": "Betrag",
    "Annahmen & Vorbehalte": "Annahmen & Vorbehalte",
    "Die vorliegende Kalkulation basiert auf den übermittelten Planunterlagen.": "Die vorliegende Kalkulation basiert auf den übermittelten Planunterlagen.",
    "Bauteil": "Bauteil", 
    "Fläche": "Fläche", 
    "Preis/m²": "Preis/m²", 
    "Gesamt": "Gesamt", 
    "SUMME": "SUMME",
    "Dachkonstruktion – Detail": "Dachkonstruktion – Detail", 
    "Komponente": "Komponente", 
    "Bemerkung": "Bemerkung", 
    "Menge": "Menge", 
    "Preis": "Preis", 
    "SUMME DACH": "SUMME DACH",
    "Treppenanlagen": "Treppenanlagen", 
    "Beschreibung": "Beschreibung", 
    "SUMME TREPPEN": "SUMME TREPPEN", 
    "Zusammenfassung Fenster & Türen": "Zusammenfassung Fenster & Türen",
    "Stückzahl": "Stückzahl", 
    "Ø Preis/Stk.": "Ø Preis/Stk.", 
    "SUMME ÖFFNUNGEN": "SUMME ÖFFNUNGEN", 
    "Zusammenfassung Haustechnik & Installationen": "Zusammenfassung Haustechnik & Installationen",
    "Gewerk / Kategorie": "Gewerk / Kategorie", 
    "Gesamtpreis": "Gesamtpreis", 
    "SUMME HAUSTECHNIK": "SUMME HAUSTECHNIK", 
    "Außentür": "Außentür", 
    "Innentür": "Innentür", 
    "Fenster": "Fenster",
    "Türen": "Türen",
    "Zweiflügelig": "Zweiflügelig", 
    "Einflügelig": "Einflügelig",
    "Fundament / Bodenplatte": "Fundament / Bodenplatte",
    "Tragwerkskonstruktion – Wände": "Tragwerkskonstruktion – Wände",
    "Geschossdecken & Balken": "Geschossdecken & Balken",
    "Oberflächen & Ausbau": "Oberflächen & Ausbau",
    "Kategorie": "Kategorie",
    
    # Admin PDF translations
    "Ventilație": "Ventilation",
    "Măsurători Pereți": "Wandmaße",
    # Wall measurements table translations
    "Tip Perete": "Wandtyp",
    "Utilizare": "Verwendung",
    "Lungime (m)": "Länge (m)",
    "Arie Brută (m²)": "Bruttofläche (m²)",
    "Deschideri (m²)": "Öffnungen (m²)",
    "Arie Netă (m²)": "Nettofläche (m²)",
    "Pereți Exteriori": "Außenwände",
    "Pereți Interiori": "Innenwände",
    "Structură (Skeleton)": "Struktur (Skeleton)",
    "Finisaje (Outline)": "Ausbau",
    # Note: "Structură" and "Finisaje" are used in wall measurements table with specific translations
    # For wall measurements context: "Structură" = "Struktur", "Finisaje" = "Ausbau"
    # These are handled directly in the table code to avoid conflicts with general translations
    
    # NOU: Textele de extins
    "Vielen Dank für Ihre Anfrage. Nachfolgend erhalten Sie unsere detaillierte Kostenschätzung, basierend auf den übermittelten Planunterlagen.": "Vielen Dank für Ihre Anfrage. Nachfolgend erhalten Sie unsere detaillierte Kostenschätzung, basierend auf den übermittelten Planunterlagen.",
    "Diese Dokumentation soll Ihnen eine klare Übersicht der notwendigen Investition bieten. Sollten Sie Fragen zur Kalkulation, den Komponenten oder wünschen Sie eine individuelle Anpassung, stehen wir Ihnen jederzeit gerne zur Verfügung.": "Diese Dokumentation soll Ihnen eine klare Übersicht der notwendigen Investition bieten. Sollten Sie Fragen zur Kalkulation, den Komponenten oder wünschen Sie eine individuelle Anpassung, stehen wir Ihnen jederzeit gerne zur Verfügung.",
    
    # Secțiuni noi pentru PDF simplificat
    "Klärung des Zwecks (richtige Erwartungen setzen)": "Klärung des Zwecks (richtige Erwartungen setzen)",
    "Diese Schätzung dient der Orientierung und ist für die erste Diskussion mit dem Auftraggeber bestimmt.": "Diese Schätzung dient der Orientierung und ist für die erste Diskussion mit dem Auftraggeber bestimmt.",
    "Die Schätzung basiert auf den bereitgestellten Informationen und auf der automatischen Analyse der Pläne.": "Die Schätzung basiert auf den bereitgestellten Informationen und auf der automatischen Analyse der Pläne.",
    "Das Dokument stellt kein verbindliches Angebot dar, sondern hilft bei:": "Das Dokument stellt kein verbindliches Angebot dar, sondern hilft bei:",
    "schnelles Erhalten eines realistischen Budgetüberblicks": "schnelles Erhalten eines realistischen Budgetüberblicks",
    "Vermeidung von Zeitverlust bei Projekten, die finanziell nicht machbar sind": "Vermeidung von Zeitverlust bei Projekten, die finanziell nicht machbar sind",
    "Projektübersicht (leicht verständlich)": "Projektübersicht (leicht verständlich)",
    "Projektübersicht": "Projektübersicht",
    "Allgemeine Baudaten (Auszug):": "Allgemeine Baudaten (Auszug):",
    "Nutzfläche": "Nutzfläche",
    "Anzahl der Ebenen": "Anzahl der Stockwerke",
    "Anzahl der Stockwerke": "Anzahl der Stockwerke",
    "Bausystem": "Bausystem",
    "Dachtyp": "Dachtyp",
    "Dachmaterial": "Dachmaterial",
    "Heizsystem": "Heizsystem",
    "Fertigstellungsgrad": "Fertigstellungsgrad",
    "gemäß verfügbaren Plänen und Informationen": "gemäß verfügbaren Plänen und Informationen",
    "Kostenstruktur (vereinfacht, klar)": "Kostenstruktur (vereinfacht, klar)",
    "Kostenstruktur": "Kostenstruktur",
    "Komponente": "Komponente",
    "Geschätzte Kosten": "Geschätzte Kosten",
    "Hausstruktur (Wände, Decken, Dach)": "Hausstruktur (Wände, Decken, Dach)",
    "Fenster & Türen": "Fenster & Türen",
    "Innenausbau": "Innenausbau",
    "Installationen & Technik": "Installationen & Technik",
    "GESAMT": "GESAMT",
    "Was NICHT enthalten ist (sehr wichtig)": "Was NICHT enthalten ist (sehr wichtig)",
    "Nicht in dieser Schätzung enthalten:": "Nicht in dieser Schätzung enthalten:",
    "Grundstückskosten": "Grundstückskosten",
    "Außenanlagen (Zaun, Hof, Wege)": "Außenanlagen (Zaun, Hof, Wege)",
    "Küche und Möbel": "Küche und Möbel",
    "Sonderausstattung oder individuelle Anforderungen": "Sonderausstattung oder individuelle Anforderungen",
    "Steuern, Genehmigungen, Anschlüsse": "Steuern, Genehmigungen, Anschlüsse",
    "Diese werden in den nächsten Phasen besprochen.": "Diese werden in den nächsten Phasen besprochen.",
    "Genauigkeit der Schätzung (rechtliche Sicherheit)": "Genauigkeit der Schätzung (rechtliche Sicherheit)",
    "Die Schätzung liegt erfahrungsgemäß in einem Bereich von ±10-15 %, abhängig von den finalen Ausführungs- und Planungsdetails.": "Die endgültigen Kosten können je nach Planungs- und Ausführungsdetails abweichen.",
    "Rechtlicher Hinweis / Haftungsausschluss": "Rechtlicher Hinweis / Haftungsausschluss",
    "Dieses Dokument ist eine unverbindliche Kostenschätzung zur ersten Budgetorientierung und ersetzt kein verbindliches Angebot.": "Dieses Dokument ist eine unverbindliche Kostenschätzung zur ersten Budgetorientierung und ersetzt kein verbindliches Angebot.",
    "Die dargestellten Werte basieren auf den vom Nutzer bereitgestellten Informationen und typischen Erfahrungswerten der jeweiligen Holzbaufirma.": "Die dargestellten Werte basieren auf den vom Nutzer bereitgestellten Informationen und typischen Erfahrungswerten der jeweiligen Holzbaufirma.",
    "Abweichungen durch Planänderungen, Ausführungsdetails, Grundstücksgegebenheiten, behördliche Auflagen oder individuelle Wünsche sind möglich.": "Abweichungen durch Planänderungen, Ausführungsdetails, Grundstücksgegebenheiten, behördliche Auflagen oder individuelle Wünsche sind möglich.",
    "Nicht Bestandteil dieser Schätzung sind insbesondere:": "Nicht Bestandteil dieser Schätzung sind insbesondere:",
    "Außenanlagen (z. B. Einfriedungen, Einfahrten, Garten- und Landschaftsgestaltung)": "Außenanlagen (z. B. Einfriedungen, Einfahrten, Garten- und Landschaftsgestaltung)",
    "statische Berechnungen": "statische Berechnungen",
    "bauphysikalische Nachweise": "bauphysikalische Nachweise",
    "Grundstücks- und Bodenbeschaffenheit": "Grundstücks- und Bodenbeschaffenheit",
    "Förderungen, Gebühren und Abgaben": "Förderungen, Gebühren und Abgaben",
    "behördliche oder rechtliche Prüfungen": "behördliche oder rechtliche Prüfungen",
    "Die endgültige Preisfestlegung erfolgt ausschließlich im Rahmen eines individuellen Angebots nach detaillierter Planung und Prüfung durch die ausführende Holzbaufirma.": "Die endgültige Preisfestlegung erfolgt ausschließlich im Rahmen eines individuellen Angebots nach detaillierter Planung und Prüfung durch die ausführende Holzbaufirma.",
    "Planungsebenen": "Planungsebenen",
    
    # Traduceri pentru valorile din formular
    # Sistem constructiv
    "CLT": "CLT",
    "Holzrahmen": "Holzrahmen",
    "HOLZRAHMEN": "Holzrahmen",
    "Massivholz": "Massivholz",
    "Panouri": "Paneele",
    "Module": "Module",
    "Montaj pe șantier": "Montage auf der Baustelle",
    "Placă": "Platte",
    "Piloți": "Pfähle",
    "Soclu": "Sockel",
    "Drept": "Flachdach",
    "Două ape": "Satteldach",
    "Patru ape": "Walmdach",
    "Mansardat": "Mansarddach",
    "Șarpantă complexă": "Komplexes Dach",
    "Satteldach": "Satteldach",
    # Materiale finisaj
    "Tencuială": "Putz",
    "Lemn": "Holz",
    "Fibrociment": "Faserzement",
    "Mix": "Mix",
    "Lemn-Aluminiu": "Holz-Aluminium",
    "PVC": "PVC",
    "Aluminiu": "Aluminium",
    # Acoperiș
    "Țiglă": "Dachziegel",
    "Tablă": "Dachblech",
    "Membrană": "Dachmembran",
    # Performanță energetică
    "Standard": "Standard",
    "KfW 55": "KfW 55",
    "KfW 40": "KfW 40",
    "KfW 40+": "KfW 40+",
    "Gaz": "Gas",
    "Pompa de căldură": "Wärmepumpe",
    "Electric": "Elektrisch",
    "Kamin": "Kamin",
    "Kaminabzug": "Kaminabzug",
    "Kamin & Kaminabzug": "Kamin & Kaminabzug",
    "Ja": "Ja",
    "Treppe": "Treppe",
    # Nivel ofertă
    "Structură": "Rohbau / Konstruktion",
    "Structură + ferestre": "Rohbau + Fenster",
    "Rohbau/Tragwerk": "Rohbau / Tragwerk",
    "Tragwerk + Fenster": "Tragwerk + Fenster",
    "Schlüsselfertig": "Schlüsselfertig",
    "Schlüsselfertiges Haus": "Schlüsselfertiges Haus",
    # Număr ferestre și uși
    "Anzahl Fenster": "Anzahl Fenster",
    "Anzahl Türen": "Anzahl Türen",
    "detektiert": "detektiert",
    "Baustellenzufahrt": "Baustellenzufahrt",
    "Tip fundație": "Fundamenttyp",
    "Sistem constructiv": "Bausystem",
    "Tip acoperiș": "Dachtyp",
    "Tip acoperis": "Dachtyp",
    "Nivel ofertă": "Gewünschter Angebotsumfang",
    "Nivel oferta": "Gewünschter Angebotsumfang",
    "Nivel de ofertă dorit": "Gewünschter Angebotsumfang",
    "Nivel de oferta dorit": "Gewünschter Angebotsumfang",
    # Termeni din formular care trebuie traduse
    "Blockbau": "Blockbau",
    "Holzrahmen": "Holzrahmen",
    "Massivholz": "Massivholz",
    "Placă": "Bodenplatte",
    "Piloți": "Pfähle",
    "Soclu": "Sockel",
    "Structură": "Rohbau / Konstruktion",
    "Structură + ferestre": "Rohbau + Fenster",
    "Casă completă": "Schlüsselfertig",
    "Ușor (camion 40t)": "Leicht (LKW 40t)",
    "Mediu": "Mittel",
    "Dificil": "Schwierig",
    "Leicht (LKW 40t)": "Leicht (LKW 40t)",
    "Mittel": "Mittel",
    "Schwierig": "Schwierig",
    "Plan": "Eben",
    "Pantă ușoară": "Leichte Hanglage",
    "Pantă mare": "Starke Hanglage",
    # Termeni pentru finisaje
    "Tencuială": "Putz",
    "Fibrociment": "Faserzement",
    "Mix": "Mischung",
    "Lemn-Aluminiu": "Holz-Aluminium",
    "PVC": "Kunststoff",
    "Aluminiu": "Aluminium",
    # Termeni pentru acoperiș
    "Țiglă": "Dachziegel",
    "Țiglă ceramică": "Tondachziegel",
    "Țiglă beton": "Betondachstein",
    "Tablă": "Blech",
    "Tablă fălțuită": "Stehfalzblech",
    "Șindrilă bituminoasă": "Bitumschindel",
    "Membrană": "Membranbahn",
    "Membrană PVC": "PVC-Bahn",
    "Hidroizolație bitum": "Bitumenabdichtung",
    # Termeni pentru încălzire
    "Gaz": "Gas",
    "Pompa de căldură": "Wärmepumpe",
    "Electric": "Elektrisch",
    # Termeni pentru nivel energetic
    "KfW 55": "KfW 55",
    "KfW 40": "KfW 40",
    "KfW 40+": "KfW 40+",
    # Termeni pentru grad prefabricare
    "Montaj pe șantier": "Montage auf der Baustelle",
    "Prefabricare parțială": "Teilweise Vorfertigung",
    "Prefabricare completă": "Vollständige Vorfertigung",
    # Termeni pentru tipuri de acoperiș
    "Drept": "Flachdach",
    "Două ape": "Satteldach",
    "Patru ape": "Walmdach",
    "Mansardat": "Mansarddach",
    "Șarpantă complexă": "Komplexes Dach",
    # Termeni pentru finisaje per etaj
    "Finisaj interior": "Innenausbau",
    "Fațadă": "Fassade",
    "Finisaj interior (Parter)": "Innenausbau (Erdgeschoss)",
    "Fațadă (Parter)": "Fassade (Erdgeschoss)",
    "Finisaj interior (Etaj 1)": "Innenausbau (1. Obergeschoss)",
    "Fațadă (Etaj 1)": "Fassade (1. Obergeschoss)",
    "Finisaj interior (Etaj 2)": "Innenausbau (2. Obergeschoss)",
    # Termeni pentru finisaje interioare și exterioare (în loc de pereți)
    "Innenverkleidung": "Innenverkleidung",
    "Außenverkleidung": "Außenverkleidung",
    "Fațadă (Etaj 2)": "Fassade (2. Obergeschoss)",
    "Finisaj interior (Etaj 3)": "Innenausbau (3. Obergeschoss)",
    "Fațadă (Etaj 3)": "Fassade (3. Obergeschoss)",
    "Finisaj interior (Mansardă)": "Innenausbau (Mansarde)",
    "Fațadă (Mansardă)": "Fassade (Mansarde)",
    "Finisaj interior (Beci)": "Innenausbau (Keller)",
    # Termeni pentru etaje
    "Erdgeschoss": "Erdgeschoss",
    "Obergeschoss": "Obergeschoss",
    "1. Obergeschoss": "1. Obergeschoss",
    "2. Obergeschoss": "2. Obergeschoss",
    "3. Obergeschoss": "3. Obergeschoss",
    "Dachgeschoss": "Dachgeschoss",
    "Mansardă": "Mansarde",
    "Planungsebene": "Planungsebene",
    "Finisaj interior": "Innenausbau",
    "Fațadă": "Fassade",
    "Fenster & Türen (Material)": "Fenster & Türen (Material)",
    "Nivel energetic": "Energiestandard",
    "Gesamtnutzfläche": "Hausfläche",
    "Hausfläche": "Hausfläche",
    "Bruttogeschossfläche": "Bruttogeschossfläche",
    "Allgemeine Projektinformationen": "Allgemeine Projektinformationen",
}

# ---------- 2. AI TRANSLATION SERVICE (AGRESIV) ----------
class GermanEnforcer:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if OpenAI and api_key else None
        self.cache = STATIC_TRANSLATIONS.copy()

    def get(self, text):
        """Returnează traducerea din cache. Dacă nu există, returnează originalul."""
        if not text or not isinstance(text, str): 
            return text
        
        translated = self.cache.get(text)
        
        if translated and '(' in translated and ')' in translated and translated != text:
            translated = re.sub(r'\s*\([^)]*\)\s*', '', translated).strip()
            return translated

        return translated if translated else text

    def translate_table_batch(self, texts: list[str]) -> dict[str, str]:
        """
        Traduce un batch de texte DIRECT pentru tabele.
        Filtrează automat: > 3 caractere, nu e preț/număr.
        """
        if not self.client:
            return {t: t for t in texts}
        
        # Filtrează ce merită tradus
        to_translate = []
        for t in texts:
            if not isinstance(t, str):
                continue
            t_clean = t.strip()
            if len(t_clean) <= 3:  # ✅ Skip < 3 caractere
                continue
            if re.match(r'^[\d\.,\s€]+$', t_clean):  # ✅ Skip prețuri/numere
                continue
            to_translate.append(t_clean)
        
        if not to_translate:
            return {t: t for t in texts}
        
        print(f"🇩🇪 [TableTranslation] Translating {len(to_translate)} items...")
        
        # Verifică cache-ul întâi
        results = {}
        missing = []
        
        for text in to_translate:
            if text in self.cache:
                results[text] = self.cache[text]
            else:
                missing.append(text)
        
        # Traduce ce lipsește
        if missing:
            prompt = (
                "You are a professional technical translator for the German construction industry. "
                "Translate the following JSON list from Romanian/English to German. "
                "Keep technical precision (e.g. 'Parter' -> 'Erdgeschoss', 'Structura Tavan' -> 'Deckenkonstruktion', "
                "'Pereti Exteriori' -> 'Außenverkleidung', 'Lemn' -> 'Holz'). "
                "Output ONLY a valid JSON object where keys are input text and values are German translations."
            )
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(missing)}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3
                )
                
                translations = json.loads(response.choices[0].message.content)
                self.cache.update(translations)
                results.update(translations)
                
                print(f"✅ [TableTranslation] Cached {len(translations)} new translations")
                
            except Exception as e:
                print(f"⚠️ [TableTranslation] Error: {e}")
                for text in missing:
                    results[text] = text
        
        # Returnează dicționar complet
        final_results = {}
        for t in texts:
            if isinstance(t, str):
                t_clean = t.strip()
                final_results[t] = results.get(t_clean, t)
            else:
                final_results[t] = t
        
        return final_results

# ---------- FONTS & ASSETS ----------
FONTS_DIR = Path(__file__).parent.parent / "pdf_assets" / "fonts"
FONT_REG = FONTS_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONTS_DIR / "DejaVuSans-Bold.ttf"
BASE_FONT, BOLD_FONT = "DejaVuSans", "DejaVuSans-Bold"

try:
    if FONT_REG.exists() and FONT_BOLD.exists():
        pdfmetrics.registerFont(TTFont(BASE_FONT, str(FONT_REG)))
        pdfmetrics.registerFont(TTFont(BOLD_FONT, str(FONT_BOLD)))
except Exception:
    BASE_FONT, BOLD_FONT = "Helvetica", "Helvetica-Bold"

COMPANY = {
    "name": "Chiemgauer Holzhaus",
    "legal": "LSP Holzbau GmbH & Co KG",
    "addr_lines": ["Seiboldsdorfer Mühle 1a", "83278 Traunstein"],
    "phone": "+49 (0) 861 / 166 192 0",
    "fax":   "+49 (0) 861 / 166 192 20",
    "email": "info@chiemgauer-holzhaus.de",
    "web":   "www.chiemgauer-holzhaus.de",
    "footer_left":  "Chiemgauer Holzhaus\nLSP Holzbau GmbH & Co KG\nRegistergericht Traunstein HRA Nr. 7311\nGeschäftsführer Bernhard Oeggl",
    "footer_mid":   "LSP Verwaltungs GmbH\nPersönlich haftende Gesellschafterin\nRegistergericht Traunstein HRB Nr. 13146",
    "footer_right": "Volksbank Raiffeisenbank Oberbayern Südost eG\nKto.Nr. 7 313 640  ·  BLZ 710 900 00\nIBAN: DE81 7109 0000 0007 3136 40   BIC: GENODEF1BGL   USt-ID: DE131544091",
}

IMG_IDENTITY = PROJECT_ROOT / "offer_identity.png"
IMG_LOGOS = PROJECT_ROOT / "offer_logos.png"

def _apply_branding(tenant_slug: str | None) -> dict:
    """
    Returns an overlay dict with:
      - company overrides
      - assets overrides
      - offer_prefix / handler_name
    """
    if not tenant_slug:
        return {}
    try:
        return fetch_tenant_branding(tenant_slug) or {}
    except Exception as e:
        print(f"⚠️ [PDF] Failed to load branding for tenant '{tenant_slug}': {e}", flush=True)
        return {}

def _asset_path(filename: str | None) -> Path | None:
    if not filename:
        return None
    p = PROJECT_ROOT / filename
    return p if p.exists() else None


def _logo_path_from_assets(assets: dict | None) -> Path | None:
    """
    Returnează path-ul logo-ului pentru PDF. Dacă userul a încărcat logo (logo_base64 sau logo_url),
    folosim DOAR acel logo – nu cădem niciodată la identity_image sau fișiere din branding.
    """
    if not assets:
        return None
    import base64
    has_user_logo = bool(assets.get("logo_base64") or (assets.get("logo_url") or "").strip())
    logo_b64 = assets.get("logo_base64")
    if logo_b64 and isinstance(logo_b64, str):
        try:
            data = base64.b64decode(logo_b64)
            ext = ".png"
            mime = (assets.get("logo_mime") or "").lower()
            if "jpeg" in mime or "jpg" in mime:
                ext = ".jpg"
            fd, path = tempfile.mkstemp(suffix=ext, prefix="tenant_logo_")
            os.close(fd)
            Path(path).write_bytes(data)
            return Path(path)
        except Exception as e:
            print(f"⚠️ [PDF] Failed to decode tenant logo_base64: {e}", flush=True)
    logo_url = (assets.get("logo_url") or "").strip()
    if logo_url and logo_url.startswith(("http://", "https://")):
        try:
            req = urllib.request.Request(logo_url, headers={"User-Agent": "Holzbot-PDF-Engine/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                ctype = (resp.headers.get("Content-Type") or "").lower()
            ext = ".png"
            if "image/jpeg" in ctype or logo_url.lower().endswith((".jpg", ".jpeg")):
                ext = ".jpg"
            fd, path = tempfile.mkstemp(suffix=ext, prefix="tenant_logo_")
            os.close(fd)
            Path(path).write_bytes(data)
            return Path(path)
        except Exception as e:
            print(f"⚠️ [PDF] Failed to download tenant logo from URL: {e}", flush=True)
            return None
    if has_user_logo:
        return None
    identity_file = assets.get("identity_image")
    if identity_file:
        p = _asset_path(identity_file)
        if p and p.exists():
            return p
    return None


def _identity_path_for_pdf(assets: dict | None) -> Path | None:
    """
    Prefer uploaded logo (logo_base64 / logo_url). If that is missing or cannot be decoded,
    fall back to identity_image from tenant branding, then bundled IMG_IDENTITY.
    """
    a = assets or {}
    from_preisdatenbank = a.get("logo_base64") or a.get("logo_url")
    logo_path = _logo_path_from_assets(a)
    if logo_path and logo_path.exists():
        return logo_path
    # If Preisdatenbank had a logo URL/base64 but decode failed, still show tenant identity strip.
    identity_file = a.get("identity_image")
    return (_asset_path(identity_file) if identity_file else None) or (IMG_IDENTITY if IMG_IDENTITY.exists() else None)


def _clean_multiline_text(v: object, *, max_lines: int = 6, max_line_len: int = 72) -> str:
    """
    Clean text coming from DB config for footer/header blocks.
    - Convert literal '\\n' into real newlines
    - Strip non-printable chars
    - Cap number of lines and line length to avoid footer blowups
    """
    if v is None:
        return ""
    s = str(v)
    # Convert escaped newlines from JSON/SQL inserts
    s = s.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    # Normalize actual CRLF too
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Strip control chars except newline
    s = "".join(ch for ch in s if ch == "\n" or (32 <= ord(ch) < 127) or ch in "ÄÖÜäöüß€")
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    lines = lines[:max_lines]
    clipped: list[str] = []
    for ln in lines:
        if len(ln) > max_line_len:
            clipped.append(ln[: max_line_len - 1].rstrip() + "…")
        else:
            clipped.append(ln)
    return "\n".join(clipped)

def _build_form_preisdatenbank_rows(frontend_data: dict, pricing_coeffs: dict, enforcer) -> list[tuple[str, str, str, str]]:
    """
    Builds rows for the Admin PDF table: (Kategorie, Parameter, Im Formular gewählt, Preis Preisdatenbank).
    Covers all variables that influence the final calculation.
    """
    # Mapare etichete germane din formular la chei din pricing (db_loader folosește uneori variante în română)
    _finish_to_key = {
        "Putz": "Tencuială", "Holz": "Lemn", "Faserzement": "Fibrociment", "Mix": "Mix",
        "Tencuială": "Tencuială", "Lemn": "Lemn", "Fibrociment": "Fibrociment",
    }

    def _fv(*keys_and_default):
        """Get first present value from frontend_data by path; default last item. Pairs (step, field) then default."""
        if not keys_and_default:
            return "—"
        default = keys_and_default[-1]
        keys = keys_and_default[:-1]
        for i in range(0, len(keys) - 1, 2):
            step = keys[i]
            field = keys[i + 1]
            d = (frontend_data or {}).get(step)
            if isinstance(d, dict):
                v = d.get(field)
                if v is not None and v != "":
                    return enforcer.get(str(v)) if enforcer else str(v)
        return default

    def _price(expr):
        """Evaluate price from pricing_coeffs; return formatted string or '—'."""
        try:
            if callable(expr):
                v = expr()
            elif isinstance(expr, tuple):
                v = next((x for x in expr if x is not None and x != ""), None)
                if v is None:
                    return "—"
                v = float(v)
            elif isinstance(expr, (int, float)):
                v = float(expr)
            else:
                v = float(expr)
            return f"{v:,.2f} €" if v is not None else "—"
        except Exception:
            return "—"

    rows = []
    pc = pricing_coeffs or {}
    sist = (frontend_data or {}).get("sistemConstructiv") or {}
    struct = (frontend_data or {}).get("structuraCladirii") or {}
    mat = (frontend_data or {}).get("materialeFinisaj") or {}
    perf = (frontend_data or {}).get("performantaEnergetica") or (frontend_data or {}).get("performanta") or {}
    fer = (frontend_data or {}).get("ferestreUsi") or {}
    dd = (frontend_data or {}).get("daemmungDachdeckung") or {}
    _nivel_pd = _normalize_nivel_oferta(frontend_data or {})
    _incl_pd = _get_offer_inclusions(_nivel_pd, frontend_data)

    # --- Fundație / Keller (einmal: Untergeschoss, einmal: Fundamenttyp) ---
    _acces_to_key = {
        "Leicht (LKW 40t)": "Leicht (LKW 40t)", "Mittel": "Mittel", "Schwierig": "Schwierig",
        "Ușor (camion 40t)": "Leicht (LKW 40t)", "Mediu": "Mittel", "Dificil": "Schwierig",
    }
    _teren_to_key = {
        "Eben": "Eben", "Leichte Hanglage": "Leichte Hanglage", "Starke Hanglage": "Starke Hanglage",
        "Plan": "Eben", "Pantă ușoară": "Leichte Hanglage", "Pantă mare": "Starke Hanglage",
    }
    tip_beci = struct.get("tipFundatieBeci") or sist.get("tipFundatieBeci")
    if str(tip_beci or "").strip() == "Placă":
        tip_beci_lookup = "Kein Keller (nur Bodenplatte)"
    else:
        tip_beci_lookup = tip_beci
    f_coeff = pc.get("foundation", {}).get("unit_price_per_m2", {})
    if _incl_pd.get("foundation"):
        price_beci = f_coeff.get(tip_beci_lookup) or f_coeff.get(tip_beci) if tip_beci else None
        rows.append((
            "Fundament",
            "Untergeschoss / Fundament",
            _fv("structuraCladirii", "tipFundatieBeci", "sistemConstructiv", "tipFundatieBeci", "—"),
            _price(price_beci) if price_beci is not None else "—"
        ))
        tip_fund = sist.get("tipFundatie") or struct.get("tipFundatie")
        price_fund = f_coeff.get(tip_fund) if tip_fund else None
        rows.append((
            "Fundament",
            "Fundamenttyp (Platte / Piloți / Sockel)",
            _fv("sistemConstructiv", "tipFundatie", "structuraCladirii", "tipFundatie", "—"),
            _price(price_fund) if price_fund is not None else "—"
        ))

    # --- Sistem constructiv ---
    if _incl_pd.get("structure_walls"):
        tip_sistem = sist.get("tipSistem")
        sys_prices = pc.get("system", {}).get("base_unit_prices", {}).get(tip_sistem, {}) if tip_sistem else {}
        p_int = sys_prices.get("interior") if isinstance(sys_prices, dict) else None
        p_ext = sys_prices.get("exterior") if isinstance(sys_prices, dict) else None
        price_sys = f"{p_int:,.2f} / {p_ext:,.2f} €/m²" if p_int is not None and p_ext is not None else "—"
        rows.append((
            "Tragwerk",
            "Bausystem (Innen / Außen €/m²)",
            _fv("sistemConstructiv", "tipSistem", "—"),
            price_sys
        ))

    # --- Acces șantier, teren (mit Normalisierung wie im Rechner) ---
    acces_raw = sist.get("accesSantier") or (frontend_data or {}).get("logistica", {}).get("accesSantier")
    acces = _acces_to_key.get(str(acces_raw or "").strip(), (acces_raw or "").strip())
    sist_c = pc.get("sistem_constructiv", {})
    acces_factors = sist_c.get("acces_santier_factor", {})
    fac_acces = acces_factors.get(acces) if acces else None
    rows.append((
        "Logistik",
        "Baustellenzufahrt",
        _fv("sistemConstructiv", "accesSantier", "logistica", "accesSantier", "—"),
        _price(fac_acces) if fac_acces is not None else "— (Faktor)"
    ))
    teren_raw = sist.get("teren")
    teren = _teren_to_key.get(str(teren_raw or "").strip(), (teren_raw or "").strip())
    teren_factors = sist_c.get("teren_factor", {})
    fac_teren = teren_factors.get(teren) if teren else None
    rows.append((
        "Logistik",
        "Gelände",
        _fv("sistemConstructiv", "teren", "—"),
        _price(fac_teren) if fac_teren is not None else "— (Faktor)"
    ))
    if _incl_pd.get("utilities"):
        util_price = sist_c.get("utilitati_anschluss_price")
        has_util = sist.get("utilitati") or (frontend_data or {}).get("logistica", {}).get("utilitati")
        rows.append((
            "Anschlüsse",
            "Versorgungsanschlüsse",
            "Ja" if has_util else "Nein",
            _price(util_price) if has_util and util_price is not None else "—"
        ))

    # --- Înălțime etaje ---
    if _incl_pd.get("structure_walls"):
        inaltime = struct.get("inaltimeEtaje")
        area_coeff = pc.get("area", {}).get("floor_height_m", {})
        h_m = area_coeff.get(inaltime) if inaltime else None
        rows.append((
            "Struktur",
            "Raumhöhe",
            _fv("structuraCladirii", "inaltimeEtaje", "—"),
            f"{float(h_m):.2f} m" if h_m is not None else "—"
        ))

    roof_c = pc.get("roof", {})
    _pd_roof_finish = _incl_pd.get("finishes") or _incl_pd.get("utilities")

    # --- Acoperiș: tip, material, dämmung, unterdach, dachstuhl ---
    if _incl_pd.get("roof"):
        rows.append((
            "Dach",
            "Dachform",
            _fv("sistemConstructiv", "tipAcoperis", "—"),
            "—"
        ))
    if _incl_pd.get("finishes"):
        rows.append((
            "Dach",
            "Dachmaterial",
            _fv("materialeFinisaj", "materialAcoperis", "—"),
            _price(roof_c.get("tile_price_per_m2") or roof_c.get("metal_price_per_m2") or roof_c.get("membrane_price_per_m2"))
        ))
    if _pd_roof_finish:
        daemm = dd.get("daemmung")
        roof_daemm_map = {"Keine": roof_c.get("daemmung_keine_price"), "Zwischensparren": roof_c.get("daemmung_zwischensparren_price"), "Aufsparren": roof_c.get("daemmung_aufsparren_price")}
        daemm_price = None
        if daemm:
            for k, p in roof_daemm_map.items():
                if k in str(daemm) or (daemm and k.lower() in str(daemm).lower()):
                    daemm_price = p
                    break
        if daemm_price is None:
            daemm_price = roof_c.get("daemmung_zwischensparren_price") or roof_c.get("daemmung_aufsparren_price")
        rows.append((
            "Dach",
            "Dämmung",
            _fv("daemmungDachdeckung", "daemmung", "—"),
            _price(daemm_price)
        ))
        unterdach = dd.get("unterdach")
        u_price = None
        if unterdach and "Schalung" in str(unterdach):
            u_price = roof_c.get("unterdach_schalung_folie_price")
        if u_price is None:
            u_price = roof_c.get("unterdach_folie_price") or roof_c.get("unterdach_schalung_folie_price")
        rows.append((
            "Dach",
            "Unterdach",
            _fv("daemmungDachdeckung", "unterdach", "—"),
            _price(u_price)
        ))
        dachstuhl = dd.get("dachstuhlTyp")
        dachstuhl_key = None
        if dachstuhl:
            s = str(dachstuhl).lower()
            if "sparren" in s:
                dachstuhl_key = "dachstuhl_sparrendach_price"
            elif "pfetten" in s:
                dachstuhl_key = "dachstuhl_pfettendach_price"
            elif "kehl" in s:
                dachstuhl_key = "dachstuhl_kehlbalkendach_price"
            elif "sonder" in s:
                dachstuhl_key = "dachstuhl_sonderkonstruktion_price"
        if not dachstuhl_key:
            dachstuhl_key = "dachstuhl_sparrendach_price"
        rows.append((
            "Dach",
            "Dachstuhl-Typ",
            _fv("daemmungDachdeckung", "dachstuhlTyp", "—"),
            _price(roof_c.get(dachstuhl_key))
        ))
    if _incl_pd.get("openings"):
        df_im = dd.get("dachfensterImDach")
        df_typ = (dd.get("dachfensterTyp") or "").strip()
        df_price = None
        if df_im and df_typ:
            _df_map = {
                "Standard": "dachfenster_stueck_standard",
                "Velux": "dachfenster_stueck_velux",
                "Roto": "dachfenster_stueck_roto",
                "Fakro": "dachfenster_stueck_fakro",
                "Sonstiges": "dachfenster_stueck_sonstiges",
            }
            _pk = _df_map.get(df_typ, "dachfenster_stueck_standard")
            df_price = roof_c.get(_pk)
        rows.append((
            "Dach",
            "Dachfenster (Stückpreis gewählte Ausführung)",
            "Ja, " + df_typ if df_im and df_typ else ("Nein" if not df_im else "Ja"),
            _price(df_price) if df_im and df_typ else "—",
        ))

    # --- Ferestre / uși (nur inkl. wenn Angebotsumfang Fenster & Türen) ---
    if _incl_pd.get("openings"):
        window_qual = fer.get("windowQuality") or mat.get("tamplarie")
        win_prices = pc.get("openings", {}).get("windows_price_per_m2", {})
        wp = win_prices.get(window_qual) if window_qual else None
        rows.append((
            "Fenster & Türen",
            "Fensterart (€/m²)",
            _fv("ferestreUsi", "windowQuality", "materialeFinisaj", "tamplarie", "—"),
            _price(wp)
        ))
        door_int_prices = pc.get("openings", {}).get("door_interior_prices", {})
        door_ext_prices = pc.get("openings", {}).get("door_exterior_prices", {})
        door_material_int = (fer.get("doorMaterialInterior") or "Standard").strip()
        door_material_ext = (fer.get("doorMaterialExterior") or "Standard").strip()
        price_door_int = door_int_prices.get(door_material_int) if door_int_prices else None
        price_door_ext = door_ext_prices.get(door_material_ext) if door_ext_prices else None
        if price_door_int is None and door_int_prices:
            price_door_int = door_int_prices.get("Standard")
        if price_door_ext is None and door_ext_prices:
            price_door_ext = door_ext_prices.get("Standard")
        rows.append((
            "Fenster & Türen",
            "Innentüren Türtyp (€/Stück)",
            _fv("ferestreUsi", "doorMaterialInterior", "Standard"),
            _price(price_door_int)
        ))
        rows.append((
            "Fenster & Türen",
            "Außentüren Türtyp (€/Stück)",
            _fv("ferestreUsi", "doorMaterialExterior", "Standard"),
            _price(price_door_ext)
        ))

    # --- Finisaje (exemple: parter interior/exterior) ---
    if _incl_pd.get("finishes"):
        fi_ground_inner = mat.get("finisajInteriorInnen_ground") or mat.get("finisajInterior_ground") or mat.get("finisajInterior")
        fi_ground_outer = mat.get("finisajInteriorAussen_ground") or mat.get("finisajInterior_ground") or mat.get("finisajInterior")
        fa_ground = mat.get("fatada_ground") or mat.get("fatada")
        fin_c = pc.get("finishes", {})
        fi_inner_key = _finish_to_key.get((fi_ground_inner or "").strip(), (fi_ground_inner or "").strip())
        fi_outer_key = _finish_to_key.get((fi_ground_outer or "").strip(), (fi_ground_outer or "").strip())
        fa_key = _finish_to_key.get((fa_ground or "").strip(), (fa_ground or "").strip())
        fi_inner_price = fin_c.get("interior_inner", {}).get(fi_inner_key) if fi_inner_key else None
        fi_outer_price = fin_c.get("interior_outer", {}).get(fi_outer_key) if fi_outer_key else None
        fa_price = fin_c.get("exterior", {}).get(fa_key) if fa_key else None
        rows.append((
            "Oberflächen",
            "Innenwände / Außenwände / Fassade (EG)",
            f"{_fv('materialeFinisaj', 'finisajInteriorInnen_ground', 'finisajInterior_ground', 'finisajInterior', '—')} / {_fv('materialeFinisaj', 'finisajInteriorAussen_ground', 'finisajInterior_ground', 'finisajInterior', '—')} / {_fv('materialeFinisaj', 'fatada_ground', 'fatada', '—')}",
            f"{_price(fi_inner_price)} / {_price(fi_outer_price)} / {_price(fa_price)}"
        ))

    # --- Performanță energetică, încălzire, ventilație ---
    if _incl_pd.get("utilities"):
        nivel_energ = perf.get("nivelEnergetic")
        tip_inc = perf.get("tipIncalzire") or (frontend_data or {}).get("incalzire", {}).get("tipIncalzire")
        rows.append((
            "Technik",
            "Nivel energetic",
            _fv("performantaEnergetica", "nivelEnergetic", "performanta", "nivelEnergetic", "—"),
            "— (Modifikator)"
        ))
        rows.append((
            "Technik",
            "Heizsystem",
            _fv("performantaEnergetica", "tipIncalzire", "performanta", "tipIncalzire", "incalzire", "tipIncalzire", "—"),
            "— (Modifikator)"
        ))
        vent = perf.get("ventilatie")
        vent_base = pc.get("utilities", {}).get("ventilation", {}).get("coefficient_ventilation_per_m2")
        rows.append((
            "Technik",
            "Ventilation",
            "Ja" if vent else "Nein",
            _price(vent_base) if vent and vent_base is not None else "—"
        ))

    # --- Semineu / Kamin ---
    if _incl_pd.get("utilities"):
        tip_sem = perf.get("tipSemineu") or (frontend_data or {}).get("incalzire", {}).get("tipSemineu")
        fp = pc.get("fireplace", {})
        fire_prices = fp.get("prices", {})
        sem_price = fire_prices.get(tip_sem) if tip_sem else None
        rows.append((
            "Kamin / Ofen",
            "Kaminart",
            _fv("performantaEnergetica", "tipSemineu", "performanta", "tipSemineu", "incalzire", "tipSemineu", "—"),
            _price(sem_price)
        ))
        horn_pf = fp.get("horn_per_floor")
        rows.append((
            "Kaminabzug",
            "Horn pro Geschoss (€)",
            "—",
            _price(horn_pf)
        ))

    # --- Scări ---
    if _incl_pd.get("floors_ceilings"):
        stair_c = pc.get("stairs", {})
        p_stair = stair_c.get("price_per_stair_unit")
        p_rail = stair_c.get("railing_price_per_stair")
        rows.append((
            "Treppen",
            "Preis pro Treppeneinheit / Geländer (€)",
            "—",
            f"{_price(p_stair)} / {_price(p_rail)}"
        ))

    # --- Area (Geschossdecke / Decke) ---
    if _incl_pd.get("floors_ceilings"):
        area_c = pc.get("area", {})
        floor_coeff = area_c.get("floor_coefficient_per_m2")
        ceiling_coeff = area_c.get("ceiling_coefficient_per_m2")
        rows.append(("Geschossdecken", "Boden (€/m²)", "—", _price(floor_coeff)))
        rows.append(("Geschossdecken", "Decke (€/m²)", "—", _price(ceiling_coeff)))

    # --- Wandaufbau (pereți interiori/exteriori dacă folosit) ---
    if _incl_pd.get("structure_walls"):
        wb = pc.get("wandaufbau", {})
        wand = (frontend_data or {}).get("wandaufbau", {})
        wb_aussen = wb.get("aussen", {})
        wb_innen = wb.get("innen", {})
        for label, key_aussen, key_innen in [
            ("EG Außen/Innen", "außenwande_ground", "innenwande_ground"),
            ("Mansarda Außen/Innen", "außenwandeMansarda", "innenwandeMansarda"),
            ("Keller Außen/Innen", "außenwandeBeci", "innenwandeBeci"),
        ]:
            sel_aussen = (wand.get(key_aussen) or "").strip()
            sel_innen = (wand.get(key_innen) or "").strip()
            p_aussen = wb_aussen.get(sel_aussen) if sel_aussen else None
            p_innen = wb_innen.get(sel_innen) if sel_innen else None
            form_val = f"{enforcer.get(sel_aussen) if sel_aussen else '—'} / {enforcer.get(sel_innen) if sel_innen else '—'}"
            rows.append(("Wandaufbau", label, form_val, f"{_price(p_aussen)} / {_price(p_innen)}"))

    # --- Boden/Decke/Belag (Bodenaufbau, Deckenaufbau, Bodenbelag) ---
    if _incl_pd.get("finishes"):
        bdb = pc.get("boden_decke_belag", {})
        bdb_data = (frontend_data or {}).get("bodenDeckeBelag", {})
        bodenaufbau_map = bdb.get("bodenaufbau", {})
        deckenaufbau_map = bdb.get("deckenaufbau", {})
        bodenbelag_map = bdb.get("bodenbelag", {})
        for suffix, key_boden, key_decken, key_belag in [
            ("EG", "bodenaufbau_ground", "deckenaufbau_ground", "bodenbelag_ground"),
            ("OG1", "bodenaufbau_floor_1", "deckenaufbau_floor_1", "bodenbelag_floor_1"),
            ("Mansarda", "bodenaufbauMansarda", "deckenaufbauMansarda", "bodenbelagMansarda"),
            ("Keller", None, "deckenaufbauBeci", "bodenbelagBeci"),
        ]:
            opt_b = (bdb_data.get(key_boden) or "").strip() if key_boden else ""
            opt_d = (bdb_data.get(key_decken) or "").strip() if key_decken else ""
            opt_bl = (bdb_data.get(key_belag) or "").strip() if key_belag else ""
            p_b = bodenaufbau_map.get(opt_b) if opt_b else None
            p_d = deckenaufbau_map.get(opt_d) if opt_d else None
            p_bl = bodenbelag_map.get(opt_bl) if opt_bl else None
            form_val = f"{enforcer.get(opt_b) or '—'} / {enforcer.get(opt_d) or '—'} / {enforcer.get(opt_bl) or '—'}"
            rows.append(("Boden/Decke/Belag", f"Bodenaufbau/Decke/Belag ({suffix})", form_val, f"{_price(p_b)} / {_price(p_d)} / {_price(p_bl)}"))

    # --- Utilities: Strom, Heizung, Kanalisation (Basispreise + Modifikatoren) ---
    if _incl_pd.get("utilities"):
        util = pc.get("utilities", {})
        elec = util.get("electricity", {})
        heat = util.get("heating", {})
        rows.append(("Technik", "Strom Basis (€/m²)", "—", _price(elec.get("coefficient_electricity_per_m2"))))
        for niv in ["Standard", "KfW 55", "KfW 40", "KfW 40+"]:
            mod = elec.get("energy_performance_modifiers", {}).get(niv)
            rows.append(("Technik", f"Strom Modifikator ({niv})", "—", _price(mod) if mod is not None else "—"))
        rows.append(("Technik", "Heizung Basis (€/m²)", "—", _price(heat.get("coefficient_heating_per_m2"))))
        for ht in ["Gas", "Wärmepumpe", "Elektrisch", "Gaz"]:
            tc = heat.get("type_coefficients", {}).get(ht)
            if tc is not None:
                rows.append(("Technik", f"Heizung Typ ({ht})", "—", _price(tc) if isinstance(tc, (int, float)) else f"{float(tc):.3f} (Faktor)"))
        rows.append(("Technik", "Kanalisation (€/m²)", "—", _price(util.get("sewage", {}).get("coefficient_sewage_per_m2"))))

    # --- Dach: Sichtdachstuhl (finisaj), pantă (structură acoperiș) ---
    if _pd_roof_finish:
        rows.append(("Dach", "Sichtdachstuhl Zuschlag (€/m²)", _fv("daemmungDachdeckung", "sichtdachstuhl", "—"), _price(roof_c.get("sichtdachstuhl_zuschlag_price"))))
    if _incl_pd.get("roof"):
        rows.append(("Dach", "Dachneigung Zuschlag (€/Grad)", "—", _price(roof_c.get("panta_acoperis_zuschlag_per_grad"))))

    # --- Wintergärten & Balkone (extras tip finisaj / amenajări) ---
    if _incl_pd.get("finishes"):
        wg_balk = (frontend_data or {}).get("wintergaertenBalkone", {})
        wb_coeff = pc.get("wintergaerten_balkone", {})
        wg_typ = (wg_balk.get("wintergartenTyp") or "").strip()
        wg_price = (wb_coeff.get("wintergarten") or {}).get(wg_typ) if wg_typ else None
        rows.append(("Wintergarten", "Typ", _fv("wintergaertenBalkone", "wintergartenTyp", "—"), _price(wg_price)))
        for btyp, blabel in [("Holzgeländer", "Holz"), ("Stahlgeländer", "Stahl"), ("Glasgeländer", "Glas")]:
            p = (wb_coeff.get("balkon") or {}).get(btyp)
            rows.append(("Balkon", f"Balkon {blabel} (€)", "—", _price(p)))
        balk_typ = (wg_balk.get("balkonTyp") or "").strip()
        balk_price = (wb_coeff.get("balkon") or {}).get(balk_typ) if balk_typ else None
        rows.append(("Balkon", "Gewählt", _fv("wintergaertenBalkone", "balkonTyp", "—"), _price(balk_price)))

    # --- Angebotsumfang (nur Anzeige, kein Preis) ---
    nivel_oferta = _normalize_nivel_oferta(frontend_data)
    rows.append(("Angebotsumfang", "Niveau", enforcer.get(nivel_oferta) if nivel_oferta else _fv("sistemConstructiv", "nivelOferta", "materialeFinisaj", "nivelOferta", "—"), "—"))

    return rows


def _finishes_per_floor_from_form(materiale_finisaj: dict) -> dict:
    """Construiește dicționar finisaje per etaj din datele formularului (chei materialeFinisaj)."""
    m = materiale_finisaj or {}
    # Ordine etaje: Parter, Etaj 1, 2, 3, Mansardă, Beci
    out = {}
    max_og = 3
    for k in m:
        if not isinstance(k, str):
            continue
        mm = re.search(r"_floor_(\d+)$", k, re.I)
        if mm:
            max_og = max(max_og, int(mm.group(1)))
    max_og = min(20, max(3, max_og))
    # Zubau / neues Geschoss pe primul plan (editor: *_plan_0)
    fi_plan0 = m.get("finisajInteriorInnen_plan_0") or m.get("finisajInterior_plan_0")
    fo_plan0 = m.get("finisajInteriorAussen_plan_0") or m.get("finisajInterior_plan_0")
    fa_plan0 = m.get("fatada_plan_0")
    if fi_plan0 or fo_plan0 or fa_plan0:
        out["Neues Geschoss (Plan 1)"] = {
            "interior_inner": fi_plan0,
            "interior_outer": fo_plan0,
            "exterior": fa_plan0,
        }
    # Parter / Erdgeschoss
    fi_ground = m.get("finisajInteriorInnen_ground") or m.get("finisajInterior_ground") or m.get("finisajInterior")
    fi_ground_outer = m.get("finisajInteriorAussen_ground") or m.get("finisajInterior_ground") or m.get("finisajInterior")
    fa_ground = m.get("fatada_ground") or m.get("fatada")
    if fi_ground or fi_ground_outer or fa_ground:
        out["Erdgeschoss"] = {"interior_inner": fi_ground, "interior_outer": fi_ground_outer, "exterior": fa_ground}
    for i in range(1, max_og + 1):
        fi = m.get(f"finisajInteriorInnen_floor_{i}") or m.get(f"finisajInterior_floor_{i}")
        fo = m.get(f"finisajInteriorAussen_floor_{i}") or m.get(f"finisajInterior_floor_{i}")
        fa = m.get(f"fatada_floor_{i}")
        if fi or fo or fa:
            out[f"Obergeschoss {i}"] = {"interior_inner": fi, "interior_outer": fo, "exterior": fa}
    # Mansardă
    fi_mans = m.get("finisajInteriorInnenMansarda") or m.get("finisajInteriorMansarda")
    fo_mans = m.get("finisajInteriorAussenMansarda") or m.get("finisajInteriorMansarda")
    fa_mans = m.get("fatadaMansarda")
    if fi_mans or fo_mans or fa_mans:
        out["Mansardă"] = {"interior_inner": fi_mans, "interior_outer": fo_mans, "exterior": fa_mans}
    # Beci
    fi_beci = m.get("finisajInteriorBeci")
    if fi_beci:
        out["Keller"] = {"interior_inner": fi_beci, "interior_outer": None, "exterior": None}
    return out


# ---------- STYLES ----------
def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(name="H1", fontName=BOLD_FONT, fontSize=12, leading=22, spaceAfter=10, spaceBefore=6))
    s.add(ParagraphStyle(name="H2", fontName=BOLD_FONT, fontSize=11, leading=14, spaceBefore=8, spaceAfter=4))
    s.add(ParagraphStyle(name="H3", fontName=BOLD_FONT, fontSize=10, leading=13, spaceBefore=4, spaceAfter=2))
    s.add(ParagraphStyle(name="Body", fontName=BASE_FONT, fontSize=10, leading=14, spaceAfter=3))
    s.add(ParagraphStyle(name="Small", fontName=BASE_FONT, fontSize=7.2, leading=9.2))
    s.add(ParagraphStyle(name="Disclaimer", fontName=BASE_FONT, fontSize=8.5, leading=11, textColor=colors.HexColor("#333333")))
    # Prevent mid-word breaks/hyphenation inside table cells.
    s.add(ParagraphStyle(name="Cell", fontName=BASE_FONT, fontSize=9.5, leading=12, splitLongWords=0))
    s.add(ParagraphStyle(name="CellBold", fontName=BOLD_FONT, fontSize=9.5, leading=12, splitLongWords=0))
    s.add(ParagraphStyle(name="CellSmall", fontName=BASE_FONT, fontSize=9, leading=11, splitLongWords=0))
    return s

def _pretty_label(v: str) -> str:
    """Turn ALL CAPS-ish tokens into nicer casing for PDF display (keeps common acronyms)."""
    if v is None:
        return "—"
    s = str(v).strip()
    if not s:
        return "—"

    # Normalize separators for nicer wrapping.
    s = s.replace("_", " ").replace(" - ", " – ").replace("-", "–")
    parts = [p.strip() for p in s.split("–")]

    keep_upper = {"PVC", "HPL", "OSB", "BSH", "KVH", "CLT", "AI"}

    def nice_word(w: str) -> str:
        if not w:
            return w
        if w.upper() in keep_upper:
            return w.upper()
        # If it's all-caps (or mostly), title-case it.
        if w.isupper():
            return w.capitalize()
        return w

    out_parts = []
    for p in parts:
        words = [nice_word(w) for w in p.split()]
        out_parts.append(" ".join(words))

    return " – ".join(out_parts)

def P(text, style_name="Cell"):
    return Paragraph((str(text) or "").replace("\n", "<br/>"), _styles()[style_name])

_pdf_currency_suffix: contextvars.ContextVar[str] = contextvars.ContextVar(
    "pdf_currency_suffix", default="\u00a0€"
)


def _parse_display_currency_code(fd: dict | None) -> str:
    if not isinstance(fd, dict):
        return "EUR"
    c = str(fd.get("display_currency") or fd.get("currency") or "EUR").strip().upper()
    return "CHF" if c == "CHF" else "EUR"


def _parse_vat_rate_decimal(fd: dict | None) -> float:
    """Effective MwSt. as decimal (e.g. 0.19). JSON may store 19 or 0.19. Default 0.19 = backward compatible."""
    if not isinstance(fd, dict):
        return 0.19
    raw = fd.get("vat_rate")
    if raw is None:
        return 0.19
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.19
    if v > 1.0:
        v = v / 100.0
    return max(0.0, min(0.5, v))


def _mwst_label_de(rate_decimal: float) -> str:
    pct = rate_decimal * 100.0
    if abs(pct - round(pct)) < 1e-9:
        pct_str = str(int(round(pct)))
    else:
        s = f"{pct:.2f}".replace(".", ",").rstrip("0").rstrip(",")
        pct_str = s
    return f"MwSt. ({pct_str} %)"


def _vat_line_label_for_offer(fd: dict | None, rate_decimal: float) -> str:
    """DE/CH: MwSt. (x %). AT (vat_preset): „20% USt.“"""
    preset = ""
    if isinstance(fd, dict):
        preset = str(fd.get("vat_preset") or "").strip().upper()
    pct = rate_decimal * 100.0
    if abs(pct - round(pct)) < 1e-9:
        pct_str = str(int(round(pct)))
    else:
        s = f"{pct:.2f}".replace(".", ",").rstrip("0").rstrip(",")
        pct_str = s
    if preset == "AT":
        return f"{pct_str}% USt."
    return _mwst_label_de(rate_decimal)


def _resolve_selected_baustelleneinrichtung_percent(frontend_data: dict | None, summary_params: dict | None) -> tuple[str, float]:
    """
    Use ONLY the user-selected Baustelleneinrichtung option from form data.
    Legacy form key fallback: accesSantier.
    """
    fd = frontend_data if isinstance(frontend_data, dict) else {}
    sist = fd.get("sistemConstructiv") or {}
    log = fd.get("logistica") or {}
    raw_choice = (
        sist.get("baustelleneinrichtung")
        or log.get("baustelleneinrichtung")
        or sist.get("accesSantier")
        or log.get("accesSantier")
        or ""
    )
    choice = str(raw_choice or "").strip().lower()
    key = "baustelleneinrichtung_standard_percent"
    if "erschwert" in choice or "mittel" in choice or "mediu" in choice:
        key = "baustelleneinrichtung_erschwert_percent"
    elif "sondertransport" in choice or "kranlogistik" in choice or "schwierig" in choice or "dificil" in choice:
        key = "baustelleneinrichtung_sondertransport_percent"
    elif "standard" in choice or "leicht" in choice or "ușor" in choice or "usor" in choice:
        key = "baustelleneinrichtung_standard_percent"
    try:
        value = float((summary_params or {}).get(key, 0.0) or 0.0)
    except Exception:
        value = 0.0
    return key, value


def _money(x):
    try:
        v = float(x)
        s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{s}{_pdf_currency_suffix.get()}"
    except: return "—"

def _fmt_m2(v):
    try:
        val = float(v)
        s = f"{val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        # NBSP so "m²" never goes to a new line
        return f"{s}\u00a0m²"
    except: return "—"

def _fmt_qty(v, unit=""):
    try:
        val = float(v)
        s = f"{val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        u = (unit or "").strip()
        ul = u.lower()
        if ul in {"buc", "buc.", "bucata", "bucată", "piece", "pieces", "stk", "stk.", "stück", "stücke"}:
            u = "Stk."
        if ul in {"ml", "ml.", "lfm"}:
            u = "m"
        # NBSP so the unit never goes to a new line
        return f"{s}\u00a0{u}".strip()
    except: return "—"

def _auto_col_widths(rows: list[list[str]], total_width_pt: float, font_name: str, font_size: float, pad_pt: float = 14.0, min_col_pt: float = 30.0) -> list[float]:
    """Auto-fit table columns to content, keeping within total_width_pt."""
    if not rows:
        return []
    ncols = max(len(r) for r in rows)
    maxw = [0.0] * ncols
    for r in rows:
        for i in range(ncols):
            txt = str(r[i]) if i < len(r) else ""
            w = stringWidth(txt, font_name, font_size) + pad_pt
            if w > maxw[i]:
                maxw[i] = w
    # Add a bit of breathing room so columns don't feel cramped
    widths = [max(min_col_pt, w + 4.0) for w in maxw]
    total = sum(widths)
    if total <= total_width_pt:
        return widths
    # scale down but keep min_col_pt
    flex = [max(0.0, w - min_col_pt) for w in widths]
    flex_total = sum(flex)
    if flex_total <= 0:
        return [min_col_pt] * ncols
    over = total - total_width_pt
    ratio = over / flex_total
    return [w - max(0.0, w - min_col_pt) * ratio for w in widths]

# ---------- PDF DRAWING ----------
def _draw_ribbon(canv: Canvas):
    """Same ribbon placement as legacy layout; only the label text is shortened to «Preisschätzung» (centered)."""
    canv.saveState()
    x, y = 18 * mm, A4[1] - 23 * mm
    rw, rh = A4[0] - 36 * mm, 9 * mm
    canv.setFillColor(colors.HexColor("#1c1c1c"))
    canv.rect(x, y, rw, rh, stroke=0, fill=1)
    canv.setFillColor(colors.white)
    canv.setFont(BOLD_FONT, 10)
    text = "Preisschätzung"
    tw = stringWidth(text, BOLD_FONT, 10)
    canv.drawString(x + (rw - tw) / 2.0, y + 2.35 * mm, text)
    canv.restoreState()

def _draw_footer(canv: Canvas):
    canv.saveState()
    y = 9*mm
    colw = (A4[0]-36*mm)/3.0
    x0 = 18*mm
    canv.setFont(BASE_FONT, 6.6)
    canv.setFillColor(colors.black)
    for i, block in enumerate((COMPANY["footer_left"], COMPANY["footer_mid"], COMPANY["footer_right"])):
        tx = x0 + i*colw
        lines = block.split("\n")
        for idx, line in enumerate(lines):
            canv.drawString(tx, y + (len(lines)-idx-1)*3.0*mm, line)
    canv.restoreState()

def _draw_firstpage_right_box(canv: Canvas, offer_no: str, handler: str):
    canv.saveState()
    box_x = A4[0]-18*mm-65*mm
    box_y = A4[1]-62*mm
    cw = 65*mm
    row_h = 8.2*mm
    # Bearbeiter = person who generated the offer; Kunden-Nr. (Fibu-Info) removed per requirement
    rows = [("Datum", datetime.now().strftime("%d.%m.%Y")), ("Bearbeiter", handler), ("Auftrag", offer_no)]
    canv.setFont(BASE_FONT, 9)
    canv.setStrokeColor(colors.black)
    canv.rect(box_x, box_y - row_h*len(rows), cw, row_h*len(rows), stroke=1, fill=0)
    for i, (k, v) in enumerate(rows):
        y = box_y - (i+1)*row_h + 2.6*mm
        canv.drawString(box_x+3*mm, y, k)
        canv.drawRightString(box_x+cw-3*mm, y, v)
    canv.restoreState()

def _first_page_canvas(offer_no: str, handler: str, assets: dict | None = None, tenant_slug: str = None):
    def _inner(canv: Canvas, doc):
        _draw_ribbon(canv)
        a = assets or {}
        show_logos = a.get("show_offer_logos", True)
        logos_file = a.get("offer_logos_image")

        identity_path = _identity_path_for_pdf(a)
        if identity_path and identity_path.exists():
            canv.drawImage(str(identity_path), A4[0]-18*mm-85*mm, A4[1]-53*mm, 85*mm, 22*mm, preserveAspectRatio=True, mask='auto')

        # Holzbau / Eder / Betonbau: fără banda offer_logos (doar branding propriu / identity)
        is_holzbau = tenant_slug and tenant_slug.lower() == "holzbau"
        is_eder = tenant_slug and tenant_slug.lower() in ("eder", "ederholzbau")
        is_betonbau = tenant_slug and tenant_slug.lower() == "betonbau"
        if not is_holzbau and not is_eder and not is_betonbau and show_logos:
            logos_path = _asset_path(logos_file) if logos_file else IMG_LOGOS
            if logos_path and logos_path.exists():
                canv.drawImage(str(logos_path), 18*mm, A4[1]-55*mm, 80*mm, 26*mm, preserveAspectRatio=True, mask='auto', anchor='sw')
        _draw_firstpage_right_box(canv, offer_no, handler)
        _draw_footer(canv)
    return _inner

def _later_pages_canvas(canv: Canvas, doc):
    _draw_ribbon(canv)
    _draw_footer(canv)

def _header_block(story, styles, offer_no: str, client: dict, enforcer, assets: dict | None = None, tenant_slug: str = None):
    # Firmenname (COMPANY["name"]) prima linie; fallback la legal
    first_line = (COMPANY.get("name") or COMPANY["legal"] or "").strip() or COMPANY["legal"]
    left_lines = [first_line, *COMPANY["addr_lines"], "", f"Tel. {COMPANY['phone']}", f"Fax {COMPANY['fax']}", "", COMPANY["email"], COMPANY["web"]]
    tbl = Table([[P("<br/>".join(left_lines), "Small"), P("", "Small")]], colWidths=[95*mm, A4[0]-36*mm-95*mm])
    tbl.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
    
    # Verifică dacă există iconițe (logo din Preisdatenbank sau identity_image din branding, offer_logos)
    a = assets or {}
    show_logos = a.get("show_offer_logos", True)
    logos_file = a.get("offer_logos_image")

    has_identity = False
    has_logos = False

    identity_path = _identity_path_for_pdf(a)
    has_identity = identity_path is not None and identity_path.exists()

    if show_logos:
        if logos_file:
            logos_path = _asset_path(logos_file)
            has_logos = logos_path.exists() if logos_path else False
        else:
            has_logos = IMG_LOGOS.exists()

    # Pentru holzbau / eder / betonbau banda offer_logos nu se desenează pe canvas → nu rezervăm înălțime pentru ea.
    is_holzbau = tenant_slug and tenant_slug.lower() == "holzbau"
    is_eder = tenant_slug and tenant_slug.lower() in ("eder", "ederholzbau")
    is_betonbau = tenant_slug and tenant_slug.lower() == "betonbau"
    skip_left_offer_logos_strip = bool(is_holzbau or is_eder or is_betonbau)
    effective_has_logos = bool(has_logos and not skip_left_offer_logos_strip)

    # Fără identity + fără bandă logos în layout: spațiu mic; altfel spațiu pentru dreapta (identity) / logos.
    if (not has_identity and not effective_has_logos) or skip_left_offer_logos_strip:
        # După eliminarea benzii offer_logos, textul companiei poate urca (fără banda de ~26 mm „rezervată” în flow).
        story.append(Spacer(1, 0 if skip_left_offer_logos_strip else 5 * mm))
    else:
        story.append(Spacer(1, 36 * mm))
    
    story.append(tbl)
    story.append(Spacer(1, 6*mm))
    
    story.append(Paragraph(f"{enforcer.get('Angebot')} • {enforcer.get('Nr.')}: {offer_no}", styles["H1"]))
    story.append(Spacer(1, 3*mm))
    
    # Datele clientului - verificăm mai multe surse și adăugăm debug
    name = client.get("nume") or client.get("name") or "—"
    city = client.get("localitate") or client.get("city") or "—"
    telefon = client.get("telefon") or "—"
    email = client.get("email") or "—"
    
    print(f"🔍 [PDF] Client data: name='{name}', city='{city}', telefon='{telefon}', email='{email}'")
    print(f"🔍 [PDF] Client dict keys: {list(client.keys()) if isinstance(client, dict) else 'Not a dict'}")
    
    name = enforcer.get(name.strip()) if name and name != "—" else "—"
    city = enforcer.get(city.strip()) if city and city != "—" else "—"
    
    lines = [
        f"<b>{enforcer.get('Bauherr / Kunde')}:</b> {name}", 
        f"<b>{enforcer.get('Ort / Bauort')}:</b> {city}", 
        f"<b>{enforcer.get('Telefon')}:</b> {telefon}", 
        f"<b>{enforcer.get('E-Mail')}:</b> {email}"
    ]
    story.append(Paragraph("<br/>".join(lines), _styles()["Cell"]))
    story.append(Spacer(1, 6*mm))

def _intro(
    story,
    styles,
    client: dict,
    enforcer: GermanEnforcer,
    offer_title: str | None = None,
    intro_content: str | None = None,
):
    title = offer_title or enforcer.get("Angebot für Ihr Holzhaus")
    title = enforcer.get(title) if title else enforcer.get("Angebot für Ihr Holzhaus")
    if title and "Chiemgauer" in title:
        title = "Angebot für Ihr Holzhaus"
    intro_block = [Paragraph(title, styles["H2"])]

    intro_paragraphs = [
        enforcer.get("Sehr geehrte Damen und Herren,"),
        enforcer.get("Vielen Dank für Ihre Anfrage. Nachfolgend erhalten Sie unsere detaillierte Kostenschätzung, basierend auf den übermittelten Planunterlagen."),
        enforcer.get("Diese Dokumentation soll Ihnen eine klare Übersicht der notwendigen Investition bieten. Sollten Sie Fragen zur Kalkulation, den Komponenten oder wünschen Sie eine individuelle Anpassung, stehen wir Ihnen jederzeit gerne zur Verfügung."),
    ]
    if intro_content:
        custom_paragraphs = [part.strip() for part in str(intro_content).split("\n\n") if part.strip()]
        if custom_paragraphs:
            intro_paragraphs = [enforcer.get(part) for part in custom_paragraphs]

    for paragraph in intro_paragraphs:
        intro_block.append(Paragraph(paragraph, styles["Body"]))

    story.append(KeepTogether(intro_block))
    story.append(Spacer(1, 6*mm))

# ---------- TABLES (TRADUCERE DIRECTĂ) ----------

def _table_standard(story, styles, title: str, data_dict: dict, enforcer: GermanEnforcer, show_mod_column: bool | None = None):
    items = data_dict.get("items", []) or data_dict.get("detailed_items", [])
    if not items: 
        return

    def _split_label_and_material(raw: str) -> tuple[str, str]:
        """Split 'Bauteil (Material)' into ('Bauteil', 'Material')."""
        if not raw or not isinstance(raw, str):
            return ("—", "—")
        mats = re.findall(r"\(([^)]+)\)", raw)
        if mats:
            base = re.sub(r"\s*\([^)]*\)\s*", " ", raw).strip()
            # also remove stray parens (some inputs are messy)
            base = re.sub(r"[()]+", " ", base).strip()
            return (base if base else raw, mats[-1].strip() if mats[-1].strip() else "—")
        # remove stray parens even if we don't have a match
        raw_clean = re.sub(r"[()]+", " ", raw).strip()
        return (raw_clean if raw_clean else raw, "—")

    # Detect if we should render Material / Mod columns for this table.
    has_any_material = False
    has_any_mode = False
    for it in items:
        name_raw = it.get("name", "")
        _label, mat_from_name = _split_label_and_material(name_raw)
        mat = it.get("material") or mat_from_name
        mode = it.get("construction_mode") or it.get("mode")
        if mat and mat != "—":
            has_any_material = True
        if mode:
            has_any_mode = True
    show_material = has_any_material
    # If show_mod_column is explicitly set, use it; otherwise auto-detect
    show_mode = has_any_mode if show_mod_column is None else show_mod_column
    
    # ✅ COLECTEAZĂ tot textul
    all_texts = [title, "Bauteil", "Fläche", "Preis/m²", "Gesamt", "SUMME"]
    if show_material:
        all_texts.append("Material")
    if show_mode:
        all_texts.append("Mod")
    for it in items:
        name_raw = it.get("name", "")
        name_clean, mat_from_name = _split_label_and_material(name_raw)
        all_texts.append(name_clean)
        if show_material:
            all_texts.append(_pretty_label(str(it.get("material") or mat_from_name or "—")))
        if show_mode:
            all_texts.append(_pretty_label(str(it.get("construction_mode") or it.get("mode") or "—")))
    
    # ✅ TRADUCE în batch
    translations = enforcer.translate_table_batch(all_texts)
    
    # ✅ CONSTRUIEȘTE tabelul
    title_para = Paragraph(translations[title], styles["H2"])
    
    head = [P(translations["Bauteil"], "CellBold")]
    if show_material:
        head.append(P(translations.get("Material", "Material"), "CellBold"))
    if show_mode:
        head.append(P(translations.get("Mod", "Mod"), "CellBold"))
    head.extend([
        P(translations["Fläche"], "CellBold"), 
        P(translations["Preis/m²"], "CellBold"), 
        P(translations["Gesamt"], "CellBold"),
    ])
    
    data = []
    floor_header_rows = []  # Track which rows are floor headers
    
    # Pentru finisaje, grupăm pe etaje
    is_finishes = "Oberflächen" in title or "Ausbau" in title or "Finisaje" in title or any(it.get("category", "").startswith("finish") for it in items)
    
    if is_finishes:
        # Grupăm items după floor_label
        items_by_floor = {}
        items_without_floor = []
        
        for it in items:
            floor_label = it.get("floor_label", "")
            if floor_label:
                if floor_label not in items_by_floor:
                    items_by_floor[floor_label] = []
                items_by_floor[floor_label].append(it)
            else:
                items_without_floor.append(it)
        
        # Sortăm etajele
        floor_order = ["Erdgeschoss"]
        for i in range(1, 10):
            floor_order.append(f"Obergeschoss {i}")
        floor_order.extend(["Mansardă", "Dachgeschoss"])
        
        # Adăugăm items grupate pe etaje
        for floor_label in floor_order:
            if floor_label in items_by_floor:
                # Adăugăm header pentru etaj (dacă avem mai multe etaje)
                if len(items_by_floor) > 1 or items_without_floor:
                    floor_row = [P(f"<b>{floor_label}</b>", "CellBold")]
                    if show_material:
                        floor_row.append(P(""))
                    if show_mode:
                        floor_row.append(P(""))
                    floor_row.extend([P(""), P(""), P("")])
                    data.append(floor_row)
                    floor_header_rows.append(len(data) - 1)  # Track this row index (0-based in data, +1 for head)
                
                for it in items_by_floor[floor_label]:
                    name_original = it.get("name", "—")
                    # Eliminăm floor_label din nume dacă este deja inclus
                    if f" - {floor_label}" in name_original:
                        name_original = name_original.replace(f" - {floor_label}", "")
                    
                    name_clean, mat_from_name = _split_label_and_material(name_original)
                    name_translated = translations.get(name_clean, name_clean)

                    row = [P(name_translated)]
                    if show_material:
                        mat_val_raw = str(it.get("material") or mat_from_name or "—")
                        mat_val = _pretty_label(mat_val_raw)
                        row.append(P(translations.get(mat_val, mat_val), "CellSmall"))
                    if show_mode:
                        mode_val_raw = str(it.get("construction_mode") or it.get("mode") or "—")
                        mode_val = _pretty_label(mode_val_raw)
                        row.append(P(translations.get(mode_val, mode_val), "CellSmall"))

                    row.extend([
                        P(_fmt_m2(it.get("area_m2", 0))), 
                        P(_money(it.get("unit_price", 0)), "CellSmall"), 
                        P(_money(it.get("cost", 0) or it.get("total_cost", 0)), "CellBold"),
                    ])
                    data.append(row)
        
        # Adăugăm items fără floor_label (fallback pentru compatibilitate)
        for it in items_without_floor:
            name_original = it.get("name", "—")
            name_clean, mat_from_name = _split_label_and_material(name_original)
            name_translated = translations.get(name_clean, name_clean)

            row = [P(name_translated)]
            if show_material:
                mat_val_raw = str(it.get("material") or mat_from_name or "—")
                mat_val = _pretty_label(mat_val_raw)
                row.append(P(translations.get(mat_val, mat_val), "CellSmall"))
            if show_mode:
                mode_val_raw = str(it.get("construction_mode") or it.get("mode") or "—")
                mode_val = _pretty_label(mode_val_raw)
                row.append(P(translations.get(mode_val, mode_val), "CellSmall"))

            row.extend([
                P(_fmt_m2(it.get("area_m2", 0))), 
                P(_money(it.get("unit_price", 0)), "CellSmall"), 
                P(_money(it.get("cost", 0) or it.get("total_cost", 0)), "CellBold"),
            ])
            data.append(row)
    else:
        # Comportament normal pentru alte categorii
        for it in items:
            name_original = it.get("name", "—")
            name_clean, mat_from_name = _split_label_and_material(name_original)
            name_translated = translations.get(name_clean, name_clean)

            row = [P(name_translated)]
            if show_material:
                mat_val_raw = str(it.get("material") or mat_from_name or "—")
                mat_val = _pretty_label(mat_val_raw)
                row.append(P(translations.get(mat_val, mat_val), "CellSmall"))
            if show_mode:
                mode_val_raw = str(it.get("construction_mode") or it.get("mode") or "—")
                mode_val = _pretty_label(mode_val_raw)
                row.append(P(translations.get(mode_val, mode_val), "CellSmall"))

            row.extend([
                P(_fmt_m2(it.get("area_m2", 0))), 
                P(_money(it.get("unit_price", 0)), "CellSmall"), 
                P(_money(it.get("cost", 0)), "CellBold"),
            ])
            data.append(row)
    
    total_row = [P(translations["SUMME"], "CellBold")]
    if show_material:
        total_row.append("")
    if show_mode:
        total_row.append("")
    total_row.extend(["", "", P(_money(data_dict.get("total_cost", 0)), "CellBold")])
    data.append(total_row)
    
    # Auto-fit column widths tightly to text (keeps numeric cols narrow)
    head_txt = ["Bauteil"]
    if show_material: head_txt.append("Material")
    if show_mode: head_txt.append("Mod")
    head_txt += ["Fläche", "Preis/m²", "Gesamt"]

    rows_txt = [head_txt]
    for it in items:
        name_original = it.get("name", "—")
        name_clean, mat_from_name = _split_label_and_material(name_original)
        row_txt = [translations.get(name_clean, name_clean)]
        if show_material:
            mat_val_raw = str(it.get("material") or mat_from_name or "—")
            row_txt.append(translations.get(_pretty_label(mat_val_raw), _pretty_label(mat_val_raw)))
        if show_mode:
            mode_val_raw = str(it.get("construction_mode") or it.get("mode") or "—")
            row_txt.append(translations.get(_pretty_label(mode_val_raw), _pretty_label(mode_val_raw)))
        row_txt += [_fmt_m2(it.get("area_m2", 0)), _money(it.get("unit_price", 0)), _money(it.get("cost", 0))]
        rows_txt.append(row_txt)
    rows_txt.append([translations.get("SUMME", "SUMME")] + ([""] * (len(head_txt) - 2)) + [_money(data_dict.get("total_cost", 0))])

    usable_width = A4[0] - 36*mm
    col_widths = _auto_col_widths(rows_txt, usable_width, BASE_FONT, 9.5, pad_pt=10, min_col_pt=26)
    align_from_col = len(head_txt) - 3

    # Improve width balance: cap numeric columns and let Bauteil take the remaining space.
    usable_width = A4[0] - 36*mm
    ncols = len(head_txt)
    # indices from end: Fläche, Preis/m², Gesamt
    idx_flaeche = ncols - 3
    idx_preis = ncols - 2
    idx_gesamt = ncols - 1
    caps = [None] * ncols
    caps[idx_flaeche] = 92.0
    caps[idx_preis] = 92.0
    caps[idx_gesamt] = 98.0
    # cap Material/Mod a bit too
    if show_material:
        caps[1] = 120.0
    if show_mode:
        caps[2 if show_material else 1] = 90.0

    capped = []
    for i, w in enumerate(col_widths):
        cw = min(w, caps[i]) if caps[i] is not None else w
        capped.append(cw)
    # give remaining space to first column (Bauteil)
    rest = usable_width - sum(capped[1:])
    capped[0] = max(120.0, rest)
    col_widths = capped

    tbl = Table([head] + data, colWidths=col_widths)
    
    # Construim stilurile - folosim culori din paleta (coffee/sand/caramel)
    # Header: sand (#F1E6D3) - mai deschis
    # Alternating rows: coffee-800 (#3E2C22) cu opacitate redusă sau sand cu opacitate
    style_commands = [
        ("GRID", (0,0), (-1,-1), 0.3, colors.black), 
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F1E6D3")),  # sand pentru header
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), 
        ("ALIGN", (align_from_col,1), (-1,-1), "RIGHT")
    ]
    
    # Pentru finisaje, adăugăm background pentru header-urile de etaj - folosim caramel (#D8A25E) cu opacitate
    if is_finishes and floor_header_rows:
        for row_idx in floor_header_rows:
            # row_idx este index în data, trebuie +1 pentru că head este la 0
            style_commands.append(("BACKGROUND", (0, row_idx + 1), (-1, row_idx + 1), colors.HexColor("#D8A25E")))  # caramel pentru floor headers
    
    tbl.setStyle(TableStyle(style_commands))
    
    # Keep title + table together (no title on one page and table on next)
    story.append(KeepTogether([title_para, tbl, Spacer(1, 4*mm)]))
    story.append(Spacer(1, 2*mm))

def _table_roof_quantities(
    story,
    styles,
    pricing_data: dict,
    enforcer: GermanEnforcer,
    inclusions: dict | None = None,
):
    roof = pricing_data.get("breakdown", {}).get("roof", {})
    items = roof.get("items", []) or roof.get("detailed_items", [])
    display_items = _roof_items_for_pdf_table(items, inclusions)

    if not display_items:
        return
    
    # ✅ COLECTEAZĂ
    all_texts = ["Dachkonstruktion – Detail", "Komponente", "Material", "Bemerkung", "Menge", "Preis", "SUMME DACH"]
    for it in display_items:
        name_raw = it.get("name", "")
        name_clean = re.sub(r'\s*\([^)]*\)\s*', '', name_raw).strip()
        all_texts.append(name_clean)
        det = it.get("details", "") or ""
        mats = re.findall(r"\(([^)]+)\)", det) if isinstance(det, str) else []
        cat = str(it.get("category", "")).lower()
        is_cover = ("roof_cover" in cat) or ("eindeck" in str(it.get("name","")).lower())
        # Material rules:
        # - Dachstruktur / Spenglerarbeiten / Dämmung: blank in Material column
        # - Dacheindeckung: show selected material
        mat = (it.get("material") or (mats[-1].strip() if mats else "—")) if is_cover else ""
        # Dacheindeckung must have a remark
        if is_cover and (not det or not str(det).strip()):
            det = "Eindeckungsmaterial"
        if isinstance(det, str) and det:
            det = re.sub(r"\s*Material[^()]*\([^)]*\)\s*", " ", det, flags=re.IGNORECASE).strip()
        all_texts.append(_pretty_label(str(mat)) if mat else "")
        all_texts.append(det)
        unit_raw = str(it.get("unit", "") or "")
        all_texts.append(unit_raw)
    
    # ✅ TRADUCE
    translations = enforcer.translate_table_batch(all_texts)
    
    # ✅ CONSTRUIEȘTE
    title_para = Paragraph(translations["Dachkonstruktion – Detail"], styles["H2"])
    
    head = [
        P(translations["Komponente"], "CellBold"), 
        P(translations.get("Material", "Material"), "CellBold"),
        P(translations["Bemerkung"], "CellBold"), 
        P(translations["Menge"], "CellBold"), 
        P(translations["Preis"], "CellBold")
    ]
    
    data = []
    for it in display_items:
        name_raw = it.get("name", "")
        name_clean = re.sub(r'\s*\([^)]*\)\s*', '', name_raw).strip()
        det = it.get("details", "") or ""
        mats = re.findall(r"\(([^)]+)\)", det) if isinstance(det, str) else []
        cat = str(it.get("category", "")).lower()
        is_cover = ("roof_cover" in cat) or ("eindeck" in str(it.get("name","")).lower())
        mat = (it.get("material") or (mats[-1].strip() if mats else "—")) if is_cover else ""
        if is_cover and (not det or not str(det).strip()):
            det = "Eindeckungsmaterial"
        if isinstance(det, str) and det:
            det = re.sub(r"\s*Material[^()]*\([^)]*\)\s*", " ", det, flags=re.IGNORECASE).strip()
        
        unit_raw = str(it.get("unit", "") or "")
        
        data.append([
            P(translations.get(name_clean, name_clean)), 
            P(translations.get(_pretty_label(str(mat)), _pretty_label(str(mat))), "CellSmall") if mat else P("", "CellSmall"),
            P(translations.get(det, det), "CellSmall"), 
            P(_fmt_qty(it.get("quantity", 0), translations.get(unit_raw, unit_raw)), "CellSmall"), 
            P(_money(it.get("cost", 0)), "CellBold")
        ])
    
    data.append([
        P(translations["SUMME DACH"], "CellBold"), 
        "", 
        "", 
        "",
        P(_money(sum(it.get("cost", 0) for it in display_items)), "CellBold"),
    ])
    
    # Auto-fit widths for this roof table
    usable_width = A4[0] - 36*mm
    roof_rows_txt = [[
        translations.get("Komponente","Komponente"),
        translations.get("Material","Material"),
        translations.get("Bemerkung","Bemerkung"),
        translations.get("Menge","Menge"),
        translations.get("Preis","Preis"),
    ]]
    for it in display_items:
        name_raw = it.get("name", "")
        name_clean = re.sub(r'\s*\([^)]*\)\s*', '', name_raw).strip()
        det = it.get("details", "") or ""
        mats = re.findall(r"\(([^)]+)\)", det) if isinstance(det, str) else []
        cat = str(it.get("category", "")).lower()
        is_cover = ("roof_cover" in cat) or ("eindeck" in str(it.get("name","")).lower())
        mat = (it.get("material") or (mats[-1].strip() if mats else "—")) if is_cover else ""
        if is_cover and (not det or not str(det).strip()):
            det = "Eindeckungsmaterial"
        if isinstance(det, str) and det:
            det = re.sub(r"\s*Material[^()]*\([^)]*\)\s*", " ", det, flags=re.IGNORECASE).strip()
        unit_u = str(it.get("unit","") or "")
        roof_rows_txt.append([
            translations.get(name_clean, name_clean),
            _pretty_label(str(mat)) if mat else "",
            translations.get(det, det),
            _fmt_qty(it.get("quantity", 0), unit_u),
            _money(it.get("cost", 0)),
        ])
    roof_rows_txt.append([translations.get("SUMME DACH","SUMME DACH"), "", "", "", _money(sum(it.get("cost", 0) for it in display_items))])
    col_widths = _auto_col_widths(roof_rows_txt, usable_width, BASE_FONT, 9.5, pad_pt=14, min_col_pt=30)
    # Balance: keep Menge/Preis relatively narrow, give Bemerkung space
    # columns: Komponente, Material, Bemerkung, Menge, Preis
    col_widths[3] = min(col_widths[3], 95.0)   # Menge
    col_widths[4] = min(col_widths[4], 105.0)  # Preis
    # recompute Komponente to fill remaining
    rest = usable_width - (col_widths[1] + col_widths[2] + col_widths[3] + col_widths[4])
    col_widths[0] = max(110.0, rest)

    tbl = Table([head] + data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.black), 
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F1E6D3")),  # sand pentru header 
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), 
        ("ALIGN", (3,1), (4,-1), "RIGHT")
    ]))
    
    story.append(KeepTogether([title_para, tbl, Spacer(1, 4*mm)]))
    story.append(Spacer(1, 2*mm))

def _table_stairs(story, styles, stairs_dict: dict, enforcer: GermanEnforcer):
    items = stairs_dict.get("detailed_items", [])
    if not items: 
        return
    
    # ✅ COLECTEAZĂ
    all_texts = ["Treppenanlagen", "Bauteil", "Beschreibung", "Menge", "Gesamt", "SUMME TREPPEN"]
    for it in items:
        all_texts.append(it.get("name", ""))
        all_texts.append(it.get("details", ""))
        all_texts.append(it.get("unit", ""))
    
    # ✅ TRADUCE
    translations = enforcer.translate_table_batch(all_texts)
    
    # ✅ CONSTRUIEȘTE
    title_para = Paragraph(translations["Treppenanlagen"], styles["H2"])
    
    head = [
        P(translations["Bauteil"], "CellBold"), 
        P(translations["Beschreibung"], "CellBold"), 
        P(translations["Menge"], "CellBold"), 
        P(translations["Gesamt"], "CellBold")
    ]
    
    data = []
    for it in items:
        unit_raw = str(it.get("unit", "Stk.") or "Stk.")
        # normalize unit via _fmt_qty (handles buc->Stk. and prevents wrapping)
        qty_text = _fmt_qty(it.get("quantity", 0), translations.get(unit_raw, unit_raw))
        
        data.append([
            P(translations.get(it.get("name", "—"), it.get("name", "—"))), 
            P(translations.get(it.get("details", "—"), it.get("details", "—")), "CellSmall"), 
            P(qty_text, "CellSmall"), 
            P(_money(it.get("cost", 0)), "CellBold")
        ])
    
    data.append([
        P(translations["SUMME TREPPEN"], "CellBold"), 
        "", 
        "", 
        P(_money(stairs_dict.get("total_cost", 0)), "CellBold")
    ])
    
    # Auto-fit widths
    usable_width = A4[0] - 36*mm
    rows_txt = [[translations["Bauteil"], translations["Beschreibung"], translations["Menge"], translations["Gesamt"]]]
    for it in items:
        unit_raw = str(it.get("unit", "Stk.") or "Stk.")
        qty_text = _fmt_qty(it.get("quantity", 0), translations.get(unit_raw, unit_raw))
        rows_txt.append([
            translations.get(it.get("name", "—"), it.get("name", "—")),
            translations.get(it.get("details", "—"), it.get("details", "—")),
            qty_text,
            _money(it.get("cost", 0)),
        ])
    rows_txt.append([translations["SUMME TREPPEN"], "", "", _money(stairs_dict.get("total_cost", 0))])
    col_widths = _auto_col_widths(rows_txt, usable_width, BASE_FONT, 9.0, pad_pt=14, min_col_pt=30)

    tbl = Table([head] + data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.black), 
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F1E6D3")),  # sand pentru header 
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), 
        ("ALIGN", (2,1), (3,-1), "RIGHT")
    ]))
    
    # Keep title + table together
    story.append(KeepTogether([title_para, tbl, Spacer(1, 4*mm)]))
    story.append(Spacer(1, 4*mm))

def _table_global_openings(story, styles, all_openings: list, enforcer: GermanEnforcer):
    if not all_openings: 
        return
    
    story.append(PageBreak())
    
    # ✅ COLECTEAZĂ texte pentru clasificare
    all_texts = [
        "Zusammenfassung Fenster & Türen", 
        "Kategorie", 
        "Stückzahl", 
        "Ø Preis/Stk.", 
        "Gesamt",
        "SUMME ÖFFNUNGEN",
        "Außentür",
        "Innentür",
        "Fenster",
        "Zweiflügelig",
        "Einflügelig"
    ]
    
    # Adaugă și textele din items pentru clasificare
    for it in all_openings:
        all_texts.append(it.get("name", ""))
        all_texts.append(it.get("type", ""))
        all_texts.append(it.get("category", ""))
        all_texts.append(it.get("location", ""))
        all_texts.append(it.get("details", ""))
    
    # ✅ TRADUCE
    translations = enforcer.translate_table_batch(all_texts)
    
    title_para = Paragraph(translations["Zusammenfassung Fenster & Türen"], styles["H1"])
    
    groups = {}
    
    kw_window = ["fenster", "window", "glass", "fereastra", "schiebetür", "schiebetur", "sliding_door", "sliding door"]
    kw_door = ["tür", "door", "usa", "ușă"]
    kw_sliding = ["schiebetür", "schiebetur", "sliding_door", "sliding door"]
    kw_ext = ["aussen", "außen", "exterior", "entrance", "haustür", "main"]
    kw_double = ["doppel", "double", "dublu", "zwei", "2-"]
    
    for it in all_openings:
        full_text = (
            str(it.get("name", "")) + " " + 
            str(it.get("type", "")) + " " + 
            str(it.get("category", "")) + " " + 
            str(it.get("location", "")) + " " +
            str(it.get("details", ""))
        ).lower()
        
        cost = float(it.get("total_cost", 0))
        
        is_window = any(x in full_text for x in kw_window)
        is_door = any(x in full_text for x in kw_door)
        is_sliding = any(x in full_text for x in kw_sliding)
        
        if not (is_window or is_door): 
            continue
            
        is_ext = any(x in full_text for x in kw_ext)
        loc_str = "Fenster" if is_sliding else ("Außentür" if is_door and is_ext else "Innentür" if is_door else "Fenster")
        
        is_double = any(x in full_text for x in kw_double)
        type_str = translations["Zweiflügelig"] if is_double else translations["Einflügelig"]
        
        category_label = translations[loc_str]
        label = f"{category_label} ({type_str})"
            
        if label not in groups:
            groups[label] = {"n": 0, "eur": 0.0}
        groups[label]["n"] += 1
        groups[label]["eur"] += cost

    def avg(total, n): 
        return total / n if n > 0 else 0.0
    
    # Determine selected material (usually global for all openings in current pricing model)
    selected_material = None
    for it in all_openings:
        m = it.get("material")
        if m:
            selected_material = _pretty_label(str(m))
            break
    if not selected_material:
        selected_material = "—"
    else:
        # Translate material (e.g., "Lemn - Aluminiu" -> "Holz-Aluminium")
        selected_material = translations.get(selected_material, enforcer.get(selected_material))
    
    head = [
        P(translations["Kategorie"], "CellBold"), 
        P(enforcer.get("Material"), "CellBold"),
        P(translations["Stückzahl"], "CellBold"), 
        P(translations["Ø Preis/Stk."], "CellBold"), 
        P(translations["Gesamt"], "CellBold"),
    ]
    
    data = []
    total_eur = 0.0
    
    for label in sorted(groups.keys()):
        g = groups[label]
        count = g["n"]
        cost = g["eur"]
        total_eur += cost
        
        data.append([
            P(label),
            P(enforcer.get(selected_material)),
            P(str(count)),
            P(_money(avg(cost, count))),
            P(_money(cost)),
        ])

    if total_eur > 0:
        data.append([
            P(translations["SUMME ÖFFNUNGEN"], "CellBold"), 
            "", 
            "", 
            "",
            P(_money(total_eur), "CellBold"),
        ])
        
        # Auto-fit widths for openings summary (keep numeric cols narrow)
        usable_width = A4[0] - 36*mm
        rows_txt = [["Kategorie", "Material", "Stückzahl", "Ø Preis/Stk.", "Gesamt"]]
        for label in sorted(groups.keys()):
            g = groups[label]
            count = g["n"]
            cost = g["eur"]
            rows_txt.append([label, selected_material, str(count), _money(avg(cost, count)), _money(cost)])
        rows_txt.append(["SUMME ÖFFNUNGEN", "", "", "", _money(total_eur)])
        col_widths = _auto_col_widths(rows_txt, usable_width, BASE_FONT, 9.5, pad_pt=10, min_col_pt=26)

        tbl = Table([head] + data, colWidths=col_widths)
        tbl.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.3, colors.black),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F1E6D3")),  # sand pentru header
            ("ALIGN", (2,1), (-1,-1), "RIGHT"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE")
        ]))
        
        story.append(KeepTogether([title_para, tbl, Spacer(1, 8*mm)]))

def _table_global_utilities(story, styles, all_utilities: list, enforcer: GermanEnforcer):
    if not all_utilities: 
        return
    
    # ✅ COLECTEAZĂ
    all_texts = [
        "Zusammenfassung Haustechnik & Installationen",
        "Elektroinstallation",
        "Heizungstechnik",
        "Abwasser",
        "Ventilation",
        "Gewerk / Kategorie",
        "Gesamtpreis",
        "SUMME HAUSTECHNIK"
    ]
    
    for it in all_utilities:
        all_texts.append(it.get("category", ""))
    
    # ✅ TRADUCE
    translations = enforcer.translate_table_batch(all_texts)
    
    title_para = Paragraph(translations["Zusammenfassung Haustechnik & Installationen"], styles["H1"])
    
    # Aggregate by category (engine emits: electricity/heating/ventilation/sewage)
    agg = {}
    total_util = 0.0
    
    for it in all_utilities:
        cat_original = it.get("category", "Sonstiges")
        cat_label = {
            "electricity": translations.get("Elektroinstallation", "Elektroinstallation"),
            "heating": translations.get("Heizungstechnik", "Heizungstechnik"),
            "ventilation": translations.get("Ventilation", "Ventilation"),
            "sewage": translations.get("Abwasser", "Abwasser"),
            "fireplace": translations.get("Kamin", "Kamin"),
            "chimney": translations.get("Kaminabzug", "Kaminabzug"),
        }.get(cat_original, translations.get(cat_original, cat_original))
        cat_translated = cat_label
        cost = it.get("total_cost", 0.0)
        total_util += cost
        agg[cat_translated] = agg.get(cat_translated, 0.0) + cost
    
    head = [
        P(translations["Gewerk / Kategorie"], "CellBold"), 
        P(translations["Gesamtpreis"], "CellBold")
    ]
    
    data = []
    for k, v in agg.items(): 
        data.append([P(k), P(_money(v))])
        
    data.append([
        P(translations["SUMME HAUSTECHNIK"], "CellBold"), 
        P(_money(total_util), "CellBold")
    ])
    
    # Auto-fit utilities table widths
    usable_width = A4[0] - 36*mm
    rows_txt = [[translations["Gewerk / Kategorie"], translations["Gesamtpreis"]]]
    for k, v in agg.items():
        rows_txt.append([k, _money(v)])
    rows_txt.append([translations["SUMME HAUSTECHNIK"], _money(total_util)])
    col_widths = _auto_col_widths(rows_txt, usable_width, BASE_FONT, 9.5, pad_pt=10, min_col_pt=26)

    tbl = Table([head] + data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.black), 
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F1E6D3")),  # sand pentru header 
        ("ALIGN", (1,1), (1,-1), "RIGHT"), 
        ("VALIGN", (0,0), (-1,-1), "MIDDLE")
    ]))
    
    story.append(KeepTogether([title_para, tbl, Spacer(1, 8*mm)]))

def _closing_blocks(story, styles, enforcer: GermanEnforcer):
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(enforcer.get("Annahmen & Vorbehalte"), styles["H2"]))
    story.append(Paragraph(enforcer.get("Die vorliegende Kalkulation basiert auf den übermittelten Planunterlagen."), styles["Body"]))

def _scope_clarification(story, styles, enforcer: GermanEnforcer):
    """Clarificare scop (setarea corectă a așteptărilor)"""
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(enforcer.get("Klärung des Zwecks (richtige Erwartungen setzen)"), styles["H2"]))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        enforcer.get("Diese Schätzung dient der Orientierung und ist für die erste Diskussion mit dem Auftraggeber bestimmt."),
        styles["Body"]
    ))
    story.append(Paragraph(
        enforcer.get("Die Schätzung basiert auf den bereitgestellten Informationen und auf der automatischen Analyse der Pläne."),
        styles["Body"]
    ))
    story.append(Paragraph(
        enforcer.get("Das Dokument stellt kein verbindliches Angebot dar, sondern hilft bei:"),
        styles["Body"]
    ))
    story.append(Paragraph("• " + enforcer.get("schnelles Erhalten eines realistischen Budgetüberblicks"), styles["Body"]))
    story.append(Paragraph("• " + enforcer.get("Vermeidung von Zeitverlust bei Projekten, die finanziell nicht machbar sind"), styles["Body"]))
    story.append(Spacer(1, 3*mm))

def _project_overview(
    story,
    styles,
    frontend_data: dict,
    enforcer: GermanEnforcer,
    plans_data: list = None,
    *,
    job_root: Path | None = None,
):
    """Prezentare generală proiect - DOAR informații comune (nu per etaj)"""
    story.append(Paragraph(enforcer.get("Projektübersicht"), styles["H2"]))
    story.append(Spacer(1, 2*mm))
    
    # Calculează suprafața totală, pereți și acoperiș
    total_area = 0.0
    total_interior_walls = 0.0
    total_exterior_walls = 0.0
    total_roof_area = 0.0
    num_floors = len(plans_data) if plans_data else 1
    
    # Calculează ferestre și uși (număr și suprafață totală)
    total_windows = 0
    total_doors = 0
    total_windows_area = 0.0
    total_doors_area = 0.0
    
    if plans_data:
        for entry in plans_data:
            pricing = entry.get("pricing", {})
            total_area += pricing.get("total_area_m2", 0.0)
            
            # Numără ferestre și uși și calculează suprafața
            breakdown = pricing.get("breakdown", {})
            openings_bd = breakdown.get("openings", {})
            openings_items = openings_bd.get("items", []) or openings_bd.get("detailed_items", [])
            
            for op in openings_items:
                obj_type = str(op.get("type", "")).lower()
                area_m2 = float(op.get("area_m2", 0.0))
                
                if "window" in obj_type or obj_type == "sliding_door":
                    total_windows += 1
                    total_windows_area += area_m2
                elif "door" in obj_type and obj_type != "sliding_door":
                    total_doors += 1
                    total_doors_area += area_m2
            
            # Încarcă area_data pentru a obține pereți și acoperiș
            plan = entry.get("info")
            if plan:
                measurements_plan_path = plan.stage_work_dir / "measurements_plan.json"
                area_json_path = plan.stage_work_dir.parent.parent / "area" / plan.plan_id / "areas_calculated.json"
                area_data = None
                if measurements_plan_path.exists():
                    try:
                        with open(measurements_plan_path, "r", encoding="utf-8") as f:
                            mp = json.load(f)
                        area_data = (mp.get("areas") or {}) if isinstance(mp, dict) else None
                    except Exception:
                        area_data = None
                if area_data is None and area_json_path.exists():
                    try:
                        with open(area_json_path, "r", encoding="utf-8") as f:
                            area_data = json.load(f)
                    except Exception as e:
                        print(f"⚠️ [PDF] Eroare la încărcarea area_data pentru {plan.plan_id}: {e}")
                if area_data:
                    try:
                        # Pereți interiori și exteriori
                        walls_data = area_data.get("walls", {})
                        interior_walls = walls_data.get("interior", {}).get("net_area_m2", 0.0)
                        exterior_walls = walls_data.get("exterior", {}).get("net_area_m2", 0.0)
                        total_interior_walls += float(interior_walls) if interior_walls else 0.0
                        total_exterior_walls += float(exterior_walls) if exterior_walls else 0.0
                        
                        # Acoperiș
                        surfaces = area_data.get("surfaces", {})
                        roof_m2 = surfaces.get("roof_m2")
                        if roof_m2:
                            total_roof_area += float(roof_m2)
                    except Exception as e:
                        print(f"⚠️ [PDF] Eroare la încărcarea area_data pentru {plan.plan_id}: {e}")
    
    # Extrage date din formular (doar cele comune, nu per etaj)
    sistem_constructiv = frontend_data.get("sistemConstructiv", {})
    materiale_finisaj = merge_plan0_form_keys_into_ground_display(dict(frontend_data.get("materialeFinisaj") or {}))
    performanta = frontend_data.get("performanta", {})
    nivel_oferta_ov = _normalize_nivel_oferta(frontend_data)
    inclusions_ov = _get_offer_inclusions(nivel_oferta_ov, frontend_data)
    
    tip_sistem = sistem_constructiv.get("tipSistem", "HOLZRAHMEN")
    acces_santier = sistem_constructiv.get("accesSantier") or frontend_data.get("logistica", {}).get("accesSantier", "—")
    tip_fundatie = sistem_constructiv.get("tipFundatie", "—")
    tip_acoperis = sistem_constructiv.get("tipAcoperis", "Satteldach")
    material_acoperis = materiale_finisaj.get("materialAcoperis", "—")
    # Citim tipul de încălzire din performanta (mutat acolo) sau din incalzire (fallback)
    incalzire_data = frontend_data.get("incalzire", {})
    performanta_energetica = frontend_data.get("performantaEnergetica", {})
    incalzire = performanta.get("tipIncalzire") or performanta_energetica.get("tipIncalzire") or incalzire_data.get("tipIncalzire") or performanta.get("incalzire", "—")
    nivel_energetic = performanta.get("nivelEnergetic") or performanta_energetica.get("nivelEnergetic", "—")
    # Citim tipul de semineu din performantaEnergetica (prioritate), apoi performanta, apoi incalzire
    # Conform structurii din compute.utils.ts, tipSemineu poate fi în toate cele 3 locuri
    tip_semineu = (
        performanta_energetica.get("tipSemineu") or 
        performanta.get("tipSemineu") or 
        incalzire_data.get("tipSemineu") or 
        "Kein Kamin"
    )
    print(f"🔍 [PDF] tip_semineu sources: performantaEnergetica={performanta_energetica.get('tipSemineu')}, performanta={performanta.get('tipSemineu')}, incalzire={incalzire_data.get('tipSemineu')}")
    print(f"🔍 [PDF] Final tip_semineu: '{tip_semineu}'")
    tamplarie = materiale_finisaj.get("tamplarie", "—")
    nivel_finisare = _normalize_nivel_oferta(frontend_data)
    
    # Finisaje per etaj: mereu din formular când există chei; completări din pricing (plans_data) chiar dacă pachetul nu include „finishes” în sumar.
    finishes_per_floor = _finishes_per_floor_from_form(materiale_finisaj)
    aufstockung_omit = _aufstockung_form_overview_omit_floor_labels(
        str(frontend_data.get("run_id") or "").strip(),
        frontend_data,
        job_root,
    )
    if aufstockung_omit and finishes_per_floor:
        finishes_per_floor = {
            k: v
            for k, v in finishes_per_floor.items()
            if not _overview_floor_key_matches_omit(k, aufstockung_omit)
        }
    if plans_data:
        import re
        for entry in plans_data:
            pricing = entry.get("pricing", {})
            breakdown = pricing.get("breakdown", {})
            finishes_bd = breakdown.get("finishes", {})
            if finishes_bd and finishes_bd.get("total_cost", 0) > 0:
                items = finishes_bd.get("detailed_items", []) or finishes_bd.get("items", [])
                for item in items:
                    floor_label = item.get("floor_label", "")
                    if not floor_label:
                        name = item.get("name", "")
                        if "Erdgeschoss" in name or "ground" in name.lower():
                            floor_label = "Erdgeschoss"
                        elif "Mansardă" in name or "Mansarda" in name:
                            floor_label = "Mansardă"
                        elif re.search(r"\d+\.\s*Obergeschoss", name, re.I):
                            md = re.search(r"(\d+)\.\s*Obergeschoss", name, re.I)
                            floor_label = f"Obergeschoss {md.group(1)}" if md else "Obergeschoss"
                        elif "Obergeschoss" in name:
                            match = re.search(r"Obergeschoss\s*(\d+)", name, re.I)
                            floor_label = f"Obergeschoss {match.group(1)}" if match else "Obergeschoss"
                        elif "Dachgeschoss" in name:
                            floor_label = "Dachgeschoss"
                        else:
                            plan = entry.get("info")
                            entry_type = (entry.get("type") or "").strip().lower()
                            if entry_type == "ground_floor":
                                floor_label = "Erdgeschoss"
                            elif "top" in entry_type or "mansard" in entry_type:
                                floor_label = "Dachgeschoss"
                            else:
                                floor_label = "Obergeschoss"
                    if not floor_label:
                        continue
                    cfl = _canonical_finishes_floor_key(str(floor_label).strip())
                    if cfl:
                        floor_label = cfl
                    if floor_label not in finishes_per_floor:
                        finishes_per_floor[floor_label] = {"interior_inner": None, "interior_outer": None, "exterior": None}
                    category = item.get("category", "")
                    material = item.get("material", "")
                    # Completați doar dacă formularul nu a furnizat deja valoarea
                    if category == "finish_interior_inner" and finishes_per_floor[floor_label]["interior_inner"] is None:
                        finishes_per_floor[floor_label]["interior_inner"] = material
                    elif category == "finish_interior_outer" and finishes_per_floor[floor_label]["interior_outer"] is None:
                        finishes_per_floor[floor_label]["interior_outer"] = material
                    elif "interior" in category and finishes_per_floor[floor_label]["interior_inner"] is None:
                        finishes_per_floor[floor_label]["interior_inner"] = material
                    elif "exterior" in category and finishes_per_floor[floor_label]["exterior"] is None:
                        finishes_per_floor[floor_label]["exterior"] = material
    if aufstockung_omit and finishes_per_floor:
        finishes_per_floor = {
            k: v
            for k, v in finishes_per_floor.items()
            if not _overview_floor_key_matches_omit(k, aufstockung_omit)
        }
    if finishes_per_floor:
        finishes_per_floor = _merge_finishes_per_floor_by_canonical(finishes_per_floor)
    finishes_per_floor = merge_neues_geschoss_plan1_finishes_into_erdgeschoss(finishes_per_floor or {})

    # Traduce valorile
    tip_sistem_de = enforcer.get(tip_sistem) if tip_sistem else "Holzrahmenbau"
    acces_santier_de = enforcer.get(acces_santier) if acces_santier and acces_santier != "—" else "—"
    tip_fundatie_de = enforcer.get(tip_fundatie) if tip_fundatie and tip_fundatie != "—" else "—"
    tip_acoperis_de = enforcer.get(tip_acoperis) if tip_acoperis else "Satteldach"
    material_acoperis_de = enforcer.get(material_acoperis) if material_acoperis and material_acoperis != "—" else "—"
    incalzire_de = enforcer.get(incalzire) if incalzire and incalzire != "—" else "—"
    nivel_energetic_de = enforcer.get(nivel_energetic) if nivel_energetic and nivel_energetic != "—" else "—"
    tamplarie_de = enforcer.get(tamplarie) if tamplarie else "—"
    nivel_finisare_de = enforcer.get(nivel_finisare) if nivel_finisare else "Schlüsselfertig"
    
    def _append_overview_value(label: str, value, *, translate: bool = True, force_bool: bool = False):
        if force_bool:
            if value is None or value == "":
                return
            rendered = "Ja" if bool(value) else "Nein"
        else:
            if value is None:
                return
            rendered = str(value).strip()
            if rendered == "" or rendered == "—":
                return
            if translate:
                rendered = enforcer.get(rendered) or rendered
        overview_items.append(f"<b>{label}:</b> <b>{rendered}</b>")

    def _append_floor_values(prefix: str, mapping: list[tuple[str, str]], values: dict, *, translate: bool = True):
        for key, label in mapping:
            value = values.get(key)
            if value is None or str(value).strip() == "":
                continue
            rendered = enforcer.get(str(value)) if translate else str(value)
            overview_items.append(f"<b>{prefix} ({label}):</b> <b>{rendered}</b>")

    # DOAR informații comune (nu per etaj)
    overview_items = []
    # Suprafața brută totală a casei (suma tuturor etajelor)
    total_house_area = 0.0
    if plans_data:
        for idx, entry in enumerate(plans_data):
            pricing = entry.get("pricing", {})
            area = pricing.get("total_area_m2", 0.0)
            total_house_area += area
            print(f"🔍 [PDF] Plan {idx}: area={area} m² (total acum: {total_house_area} m²)")
    
    if total_house_area > 0:
        _wp_pkg = str((frontend_data or {}).get("wizard_package") or "").strip().lower()
        _house_area_label = (
            enforcer.get("Bruttogeschossfläche") if _wp_pkg in ("aufstockung", "zubau", "zubau_aufstockung") else enforcer.get("Hausfläche")
        )
        overview_items.append(f"<b>{_house_area_label}:</b> ca. <b>{total_house_area:.0f} m²</b>")
        print(f"✅ [PDF] Suprafața brută totală a casei afișată: {total_house_area:.0f} m²")
    else:
        print(f"⚠️ [PDF] total_house_area is 0 - nu se afișează suprafața casei")
    
    # Wintergarten-Glas: afișăm explicit înainte de secțiunea Dach.
    wg_glass_rows_ov: list[str] = []
    wg_glass_total_ov = 0.0
    if plans_data and inclusions_ov.get("finishes", False):
        for idx_wg, entry in enumerate(plans_data, 1):
            pricing = entry.get("pricing", {}) if isinstance(entry, dict) else {}
            breakdown = pricing.get("breakdown", {}) if isinstance(pricing, dict) else {}
            wb = breakdown.get("wintergaerten_balkone", {}) if isinstance(breakdown, dict) else {}
            items_wb = wb.get("detailed_items", []) if isinstance(wb, dict) else []
            for it in items_wb:
                if not isinstance(it, dict):
                    continue
                if str(it.get("category") or "").strip().lower() != "wintergarten_glass":
                    continue
                c = float(it.get("total_cost") or it.get("cost") or 0.0)
                if c <= 0:
                    continue
                fl = str(entry.get("floor_label") or f"Stockwerk {idx_wg}")
                wg_glass_rows_ov.append(f"{fl}: {_money(c)}")
                wg_glass_total_ov += c
    if wg_glass_total_ov > 0:
        overview_items.append(f"<b>Wintergarten – Glasflächen:</b> <b>{_money(wg_glass_total_ov)}</b>")
        for row in wg_glass_rows_ov:
            overview_items.append(f"• {row}")

    # Suprafețe acoperiș: afișăm atât cu Überstand cât și Dämmzone (fără Überstand) dacă avem roof_pricing.json.
    run_id = str(frontend_data.get("run_id") or "").strip()
    roof_pricing_path = None
    roof_pricing = {}
    if run_id:
        for base in (OUTPUT_ROOT / run_id, JOBS_ROOT / run_id / "output", RUNNER_ROOT / "output" / run_id):
            p = base / "roof" / "roof_3d" / "entire" / "mixed" / "roof_pricing.json"
            if p.exists():
                roof_pricing_path = p
                break
        if roof_pricing_path and roof_pricing_path.exists():
            try:
                with open(roof_pricing_path, encoding="utf-8") as f:
                    roof_pricing = json.load(f)
            except Exception:
                roof_pricing = {}
    roof_area_with_overhang = None
    roof_area_without_overhang = None
    if roof_pricing_path and roof_pricing_path.exists() and inclusions_ov.get("roof", False):
        try:
            rm = (roof_pricing or {}).get("roof_measurements") or {}
            roof_area_with_overhang = rm.get("roof_area_with_overhang_m2")
            roof_area_without_overhang = rm.get("roof_area_without_overhang_m2")
        except Exception:
            roof_area_with_overhang = None
            roof_area_without_overhang = None

    if inclusions_ov.get("roof", False):
        if roof_area_with_overhang is not None and float(roof_area_with_overhang) > 0:
            overview_items.append(f"<b>Dachfläche (mit Überstand):</b> <b>{float(roof_area_with_overhang):.1f} m²</b>")
        elif total_roof_area > 0:
            overview_items.append(f"<b>{enforcer.get('Dachfläche')}:</b> <b>{total_roof_area:.1f} m²</b>")

        if roof_area_without_overhang is not None and float(roof_area_without_overhang) > 0:
            overview_items.append(f"<b>Dachfläche (ohne Überstand):</b> <b>{float(roof_area_without_overhang):.1f} m²</b>")
        try:
            rm_ins = (roof_pricing or {}).get("roof_measurements") or {}
            roof_area_insulated = rm_ins.get("roof_area_insulated_m2")
            if roof_area_insulated is not None and float(roof_area_insulated) >= 0:
                overview_items.append(f"<b>Dachfläche gedämmt:</b> <b>{float(roof_area_insulated):.1f} m²</b>")
        except Exception:
            pass

    # Selections done in DetectionsReview popup (Aufstockung/Zubau) must be visible in final offer.
    overview_items.extend(
        _build_editor_popup_overview_items(frontend_data, plans_data or [], enforcer)
    )

    # Finishes in Projektübersicht are already merged from pricing breakdown (floor_label on each line).
    # Wandaufbau/Boden in the form dict may only have *_ground while pricing applied those values to OG too —
    # mirror the same breakdown into empty per-floor keys so Wandaufbau/Boden lines match Innenausbau.
    wa_ov, bd_ov = _wand_boden_overlay_from_plans_pricing(frontend_data, plans_data or [])
    fd_overview = dict(frontend_data) if isinstance(frontend_data, dict) else {}
    fd_overview["wandaufbau"] = wa_ov
    fd_overview["bodenDeckeBelag"] = bd_ov

    overview_items.extend(
        build_selected_form_overview_items(
            fd_overview,
            enforcer,
            inclusions_ov,
            finishes_per_floor,
            num_floors,
            acces_santier_de=acces_santier_de,
            tip_fundatie_de=tip_fundatie_de,
            tip_sistem_de=tip_sistem_de,
            tip_acoperis_de=tip_acoperis_de,
            material_acoperis_de=material_acoperis_de,
            incalzire_de=incalzire_de,
            nivel_energetic_de=nivel_energetic_de,
            nivel_finisare_de=nivel_finisare_de,
            tip_semineu=tip_semineu,
            aufstockung_omit_floor_labels=aufstockung_omit,
        )
    )
    
    for item in overview_items:
        story.append(Paragraph(item, styles["Body"]))
    
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(f"({enforcer.get('gemäß verfügbaren Plänen und Informationen')})", styles["Small"]))
    story.append(Spacer(1, 3*mm))

def _floor_specific_info(story, styles, entry: dict, enforcer: GermanEnforcer):
    """Informații specifice pentru fiecare etaj"""
    plan = entry["info"]
    pricing = entry["pricing"]
    
    # Calculează suprafața pentru acest etaj
    floor_area = pricing.get("total_area_m2", 0.0)
    
    # Cost total pentru acest etaj
    floor_cost = pricing.get("total_cost_eur", 0.0)
    
    # Creează tabel cu informații specifice etajului (fără ferestre și uși - sunt la total)
    floor_info_items = []
    if floor_area > 0:
        floor_info_items.append(f"<b>{enforcer.get('Nutzfläche')}:</b> <b>{floor_area:.1f} m²</b>")
    if floor_cost > 0:
        floor_info_items.append(f"<b>{enforcer.get('Geschätzte Kosten')}:</b> <b>{_money(floor_cost)}</b>")
    
    if floor_info_items:
        story.append(Spacer(1, 2*mm))
        for item in floor_info_items:
            story.append(Paragraph(item, styles["Body"]))

def _simplified_cost_structure(story, styles, plans_data: list, inclusions: dict, enforcer: GermanEnforcer):
    """Structură de cost simplificată (eliminată complet)"""
    # Secțiunea Kostenstruktur a fost eliminată complet conform cerințelor
    pass
    
    # Calculează sumele totale pentru fiecare categorie
    foundation_total = 0.0
    structure_total = 0.0
    floors_total = 0.0
    roof_total = 0.0
    openings_total = 0.0
    finishes_total = 0.0
    utilities_total = 0.0
    fireplace_total = 0.0
    stairs_total = 0.0
    aufstockung_phase1_total = 0.0
    
    for entry in plans_data:
        pricing = entry["pricing"]
        bd = pricing.get("breakdown", {})
        
        # Folosim același logic ca în filtrarea breakdown-ului: verificăm inclusions pentru fiecare categorie
        if inclusions.get("foundation", False):
            foundation_total += bd.get("foundation", {}).get("total_cost", 0.0)
        # structure_walls este inclus în "structure" sau "structure_walls"
        if inclusions.get("structure_walls", False) or inclusions.get("structure", False):
            structure_total += bd.get("structure_walls", {}).get("total_cost", 0.0)
        # Verificăm și dacă există o categorie "structure" directă
        if inclusions.get("structure", False) and bd.get("structure"):
            structure_total += bd.get("structure", {}).get("total_cost", 0.0)
        if inclusions.get("floors_ceilings", False):
            floors_total += bd.get("floors_ceilings", {}).get("total_cost", 0.0)
            # Stairs sunt incluse în floors_ceilings dacă este inclus
            stairs_total += bd.get("stairs", {}).get("total_cost", 0.0)
            aufstockung_phase1_total += bd.get("aufstockung_phase1", {}).get("total_cost", 0.0)
        if inclusions.get("roof", False):
            roof_total += bd.get("roof", {}).get("total_cost", 0.0)
        if inclusions.get("openings", False):
            openings_total += bd.get("openings", {}).get("total_cost", 0.0)
        if inclusions.get("finishes", False) and bd.get("finishes"):
            finishes_total += bd.get("finishes", {}).get("total_cost", 0.0)
        if inclusions.get("utilities", False):
            utilities_total += bd.get("utilities", {}).get("total_cost", 0.0)
            # Fireplace este inclus în utilities dacă este inclus
            fireplace_total += bd.get("fireplace", {}).get("total_cost", 0.0)
        # Basement este inclus dacă foundation este inclus (basement face parte din fundație/structură)
        if inclusions.get("foundation", False):
            basement_total = bd.get("basement", {}).get("total_cost", 0.0)
            foundation_total += basement_total  # Adăugăm basement la foundation pentru consistență
    
    # Include și roof_total și floors_total în structure_total (pentru că sunt parte din structură)
    structure_total += roof_total + floors_total
    
    # Calculează totalul final (fără tabel, doar pentru verificare)
    total_cost = foundation_total + structure_total + openings_total + finishes_total + utilities_total + fireplace_total + stairs_total + aufstockung_phase1_total
    
    story.append(Spacer(1, 3*mm))

def _exclusions_section(story, styles, enforcer: GermanEnforcer):
    """Ce NU este inclus"""
    story.append(Paragraph(enforcer.get("Was NICHT enthalten ist (sehr wichtig)"), styles["H2"]))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(enforcer.get("Nicht in dieser Schätzung enthalten:"), styles["Body"]))
    story.append(Paragraph("• " + enforcer.get("Grundstückskosten"), styles["Body"]))
    story.append(Paragraph("• " + enforcer.get("Außenanlagen (Zaun, Hof, Wege)"), styles["Body"]))
    story.append(Paragraph("• " + enforcer.get("Küche und Möbel"), styles["Body"]))
    story.append(Paragraph("• " + enforcer.get("Sonderausstattung oder individuelle Anforderungen"), styles["Body"]))
    story.append(Paragraph("• " + enforcer.get("Steuern, Genehmigungen, Anschlüsse"), styles["Body"]))
    story.append(Paragraph(enforcer.get("Diese werden in den nächsten Phasen besprochen."), styles["Body"]))
    story.append(Spacer(1, 3*mm))

def _precision_section(story, styles, enforcer: GermanEnforcer):
    """Precizia estimării"""
    story.append(Paragraph(enforcer.get("Genauigkeit der Schätzung (rechtliche Sicherheit)"), styles["H2"]))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        enforcer.get("Die Schätzung liegt erfahrungsgemäß in einem Bereich von ±10-15 %, abhängig von den finalen Ausführungs- und Planungsdetails."),
        styles["Body"],
    ))
    story.append(Spacer(1, 3*mm))

def _legal_disclaimer(story, styles, enforcer: GermanEnforcer, pdf_texts: dict | None = None):
    """Legal disclaimer complet"""
    pdf_texts = pdf_texts or {}
    legal_title = str(pdf_texts.get("legal_title") or enforcer.get("Rechtlicher Hinweis / Haftungsausschluss"))
    legal_content_raw = str(pdf_texts.get("legal_content_raw") or "").strip()
    legal_paragraph_1 = str(
        pdf_texts.get("legal_paragraph_1")
        or enforcer.get("Dieses Dokument ist eine unverbindliche Kostenschätzung zur ersten Budgetorientierung und ersetzt kein verbindliches Angebot.")
    )
    legal_paragraph_2 = str(
        pdf_texts.get("legal_paragraph_2")
        or enforcer.get("Die dargestellten Werte basieren auf den vom Nutzer bereitgestellten Informationen und typischen Erfahrungswerten der jeweiligen Holzbaufirma.")
    )
    legal_paragraph_3 = str(
        pdf_texts.get("legal_paragraph_3")
        or enforcer.get("Abweichungen durch Planänderungen, Ausführungsdetails, Grundstücksgegebenheiten, behördliche Auflagen oder individuelle Wünsche sind möglich.")
    )
    legal_exclusions_intro = str(
        pdf_texts.get("legal_exclusions_intro")
        or enforcer.get("Nicht Bestandteil dieser Schätzung sind insbesondere:")
    )
    legal_exclusions = pdf_texts.get("legal_exclusions")
    if isinstance(legal_exclusions, str):
        legal_exclusions = [ln.strip() for ln in legal_exclusions.splitlines() if ln.strip()]
    if not isinstance(legal_exclusions, list) or not legal_exclusions:
        legal_exclusions = [
            enforcer.get("Außenanlagen (z. B. Einfriedungen, Einfahrten, Garten- und Landschaftsgestaltung)"),
            enforcer.get("statische Berechnungen"),
            enforcer.get("bauphysikalische Nachweise"),
            enforcer.get("Grundstücks- und Bodenbeschaffenheit"),
            enforcer.get("Förderungen, Gebühren und Abgaben"),
            enforcer.get("behördliche oder rechtliche Prüfungen"),
        ]
    legal_final_paragraph = str(
        pdf_texts.get("legal_final_paragraph")
        or enforcer.get("Die endgültige Preisfestlegung erfolgt ausschließlich im Rahmen eines individuellen Angebots nach detaillierter Planung und Prüfung durch die ausführende Holzbaufirma.")
    )

    # Adaugă PageBreak doar dacă e necesar (nu automat)
    story.append(Spacer(1, 5*mm))
    legal_block = [Paragraph(legal_title, styles["H1"]), Spacer(1, 2*mm)]

    if legal_content_raw:
        blocks = [block.strip() for block in legal_content_raw.split("\n\n") if block.strip()]
        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if lines and all(line.startswith(("-", "*", "•")) for line in lines):
                for line in lines:
                    legal_block.append(Paragraph("• " + line.lstrip("-*• ").strip(), styles["Body"]))
                legal_block.append(Spacer(1, 2*mm))
            else:
                legal_block.append(Paragraph("<br/>".join(lines), styles["Body"]))
                legal_block.append(Spacer(1, 2*mm))
        story.append(KeepTogether(legal_block))
        return

    legal_block.append(Paragraph(legal_paragraph_1, styles["Body"]))
    legal_block.append(Paragraph(legal_paragraph_2, styles["Body"]))
    legal_block.append(Spacer(1, 2*mm))
    legal_block.append(Paragraph(legal_paragraph_3, styles["Body"]))
    legal_block.append(Spacer(1, 2*mm))
    legal_block.append(Paragraph(legal_exclusions_intro, styles["Body"]))
    for item in legal_exclusions:
        legal_block.append(Paragraph("• " + str(item), styles["Body"]))
    legal_block.append(Spacer(1, 2*mm))
    legal_block.append(Paragraph(legal_final_paragraph, styles["Body"]))
    story.append(KeepTogether(legal_block))


def _roof_only_intro(story, styles, client: dict, enforcer: GermanEnforcer):
    story.append(Paragraph("Schätzungsangebot Dachstuhl", styles["H2"]))
    story.append(Paragraph(enforcer.get("Sehr geehrte Damen und Herren,"), styles["Body"]))
    story.append(Paragraph(
        "Nachfolgend erhalten Sie unsere Kostenschätzung ausschließlich für die Dachkonstruktion (Dachstuhl), "
        "basierend auf den übermittelten Planunterlagen und Ihren Angaben.",
        styles["Body"],
    ))
    story.append(Spacer(1, 6 * mm))


def _roof_only_summary(story, styles, frontend_data: dict, enforcer: GermanEnforcer):
    story.append(Paragraph("Projektübersicht – Dach", styles["H2"]))
    story.append(Spacer(1, 2 * mm))
    pd = frontend_data.get("projektdaten") or {}
    dd = frontend_data.get("daemmungDachdeckung") or {}
    lines: list[str] = []
    if pd:
        if pd.get("projektumfang"):
            lines.append(f"<b>Projektumfang:</b> {pd.get('projektumfang')}")
        if pd.get("nutzungDachraum"):
            lines.append(f"<b>Nutzung Dachraum:</b> {pd.get('nutzungDachraum')}")
        if pd.get("deckenInnenausbau"):
            lines.append(f"<b>Decken-Innenausbau:</b> {pd.get('deckenInnenausbau')}")
    if dd:
        if dd.get("daemmung"):
            lines.append(f"<b>Dämmung:</b> {dd.get('daemmung')}")
        if dd.get("unterdach"):
            lines.append(f"<b>Unterdach:</b> {dd.get('unterdach')}")
        if dd.get("dachstuhlTyp"):
            lines.append(f"<b>Dachstuhl:</b> {dd.get('dachstuhlTyp')}")
        if dd.get("dachdeckung"):
            lines.append(f"<b>Dachdeckung:</b> {dd.get('dachdeckung')}")
        if dd.get("sichtdachstuhl") is not None:
            lines.append(f"<b>Sichtdachstuhl:</b> {'Ja' if dd.get('sichtdachstuhl') else 'Nein'}")
    if not lines:
        lines.append("Angaben gemäß Formular (Dachstuhl).")
    story.append(Paragraph("<br/>".join(lines), styles["Body"]))
    story.append(Spacer(1, 4 * mm))


def _legal_disclaimer_roof_only(story, styles, enforcer: GermanEnforcer):
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(enforcer.get("Rechtlicher Hinweis / Haftungsausschluss"), styles["H1"]))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "Dieses Dokument ist eine unverbindliche Kostenschätzung zur Budgetorientierung "
        "<b>nur für die Dachkonstruktion (Dachstuhl)</b> und ersetzt kein verbindliches Angebot.",
        styles["Body"],
    ))
    story.append(Paragraph(
        enforcer.get(
            "Die dargestellten Werte basieren auf den vom Nutzer bereitgestellten Informationen und typischen Erfahrungswerten der jeweiligen Holzbaufirma."
        ),
        styles["Body"],
    ))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "Abweichungen durch Planänderungen, Ausführungsdetails oder individuelle Wünsche sind möglich.",
        styles["Body"],
    ))

# ---------- MAIN GENERATOR ----------
def generate_complete_offer_pdf(run_id: str, output_path: Path | None = None, job_root: Path | None = None) -> Path:
    print(f"🚀 [PDF] START: {run_id}")
    
    jobs_root_base = PROJECT_ROOT / "jobs"
    output_root = jobs_root_base / run_id
    
    if not output_root.exists(): 
        output_root = RUNNER_ROOT / "output" / run_id
        if not output_root.exists():
            print(f"❌ EROARE: Nu găsesc directorul de output pentru run_id='{run_id}'")
            raise FileNotFoundError(f"Output nu există: {output_root}")
        print(f"⚠️ Folosind directorul standard: {output_root.resolve()}")
    else:
        print(f"✅ Folosind JOBS_ROOT: {output_root.resolve()}")

    if job_root is None:
        job_root = JOBS_ROOT / run_id
        if not job_root.exists():
            job_root = None
    if job_root:
        print(f"🔹 [PDF] Loading frontend_data from job_root: {job_root}")

    if output_path is None:
        final_output_root = RUNNER_ROOT / "output" / run_id
        pdf_dir = final_output_root / "offer_pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        output_path = pdf_dir / f"oferta_{run_id}.pdf"

    frontend_data = load_frontend_data_for_run(run_id, job_root)
    fd_ctx = frontend_data if isinstance(frontend_data, dict) else {}
    _pdf_money_tok = _pdf_currency_suffix.set("\u00a0CHF" if _parse_display_currency_code(fd_ctx) == "CHF" else "\u00a0€")
    pdf_vat_rate = _parse_vat_rate_decimal(fd_ctx)
    try:
        tenant_slug = frontend_data.get("tenant_slug") if isinstance(frontend_data, dict) else None
        try:
            from pricing.db_loader import fetch_pricing_parameters
            calc_mode = frontend_data.get("calc_mode") or "default"
            pricing_coeffs = fetch_pricing_parameters(tenant_slug or "", calc_mode=calc_mode)
            summary_params = (pricing_coeffs or {}).get("_raw_params", {}) if isinstance(pricing_coeffs, dict) else {}
        except Exception as e:
            print(f"⚠️ [PDF] Could not load pricing coefficients for summary margins: {e}")
            summary_params = {}
        
        branding = _apply_branding(tenant_slug)
        assets = (branding.get("assets") or {}) if isinstance(branding, dict) else {}
        company_overrides = (branding.get("company") or {}) if isinstance(branding, dict) else {}
        pdf_text_overrides = (branding.get("texts") or {}) if isinstance(branding, dict) else {}
        # Prefer logo and company from job (Preisdatenbank) – signed logo URL and company data sent by API
        if isinstance(frontend_data, dict):
            job_pdf_assets = frontend_data.get("pdf_assets")
            if isinstance(job_pdf_assets, dict) and job_pdf_assets:
                assets = {**assets, **job_pdf_assets}
            job_pdf_company = frontend_data.get("pdf_company")
            if isinstance(job_pdf_company, dict) and job_pdf_company:
                company_overrides = {**company_overrides, **job_pdf_company}
                print(f"🔍 [PDF] pdf_company from job: name={job_pdf_company.get('name')!r}, addr_lines={job_pdf_company.get('addr_lines')!r}, phone={job_pdf_company.get('phone')!r}, email={job_pdf_company.get('email')!r}, keys={list(job_pdf_company.keys())}", flush=True)
            job_pdf_texts = frontend_data.get("pdf_texts")
            if isinstance(job_pdf_texts, dict) and job_pdf_texts:
                pdf_text_overrides = {**pdf_text_overrides, **job_pdf_texts}
        offer_prefix = (branding.get("offer_prefix") or "CHH") if isinstance(branding, dict) else "CHH"
        client_id = (branding.get("client_id") or offer_prefix) if isinstance(branding, dict) else offer_prefix
        handler = (branding.get("handler_name") or "Florian Siemer") if isinstance(branding, dict) else "Florian Siemer"
        # Prefer handler from job (Preisdatenbank "Reprezentant firmă")
        if isinstance(company_overrides, dict) and company_overrides.get("handler_name"):
            handler = str(company_overrides.get("handler_name", "")).strip() or handler
        offer_title = branding.get("offer_title") if isinstance(branding, dict) else None
        if tenant_slug and str(tenant_slug).lower() == "betonbau":
            offer_title = "Angebot für Ihr Massivhaus"
        # Titlu intro: job / preview (frontend_data + pdf_texts) bate branding-ul din DB
        offer_title_effective = offer_title
        if isinstance(frontend_data, dict):
            ot = frontend_data.get("offer_title")
            if isinstance(ot, str) and ot.strip():
                offer_title_effective = ot.strip()
        _pot = pdf_text_overrides.get("offer_title") if isinstance(pdf_text_overrides, dict) else None
        if isinstance(_pot, str) and _pot.strip():
            offer_title_effective = _pot.strip()
        
        # Apply company overrides (header/footer content)
        if isinstance(company_overrides, dict) and company_overrides:
            for k, v in company_overrides.items():
                if v is not None:
                    if k in {"footer_left", "footer_mid", "footer_right"}:
                        COMPANY[k] = _clean_multiline_text(v, max_lines=6, max_line_len=72)
                    elif k == "legal":
                        COMPANY[k] = _clean_multiline_text(v, max_lines=2, max_line_len=60).replace("\n", " ")
                    elif k == "name":
                        COMPANY[k] = str(v).strip()
                    elif k == "email":
                        COMPANY[k] = str(v).strip()
                    elif k == "web":
                        COMPANY[k] = str(v).strip() if v else ""
                    elif k == "phone":
                        COMPANY[k] = str(v).strip()
                    elif k == "fax":
                        COMPANY[k] = str(v).strip()
                    elif k == "addr_lines" and isinstance(v, list):
                        COMPANY[k] = [str(x).strip() for x in v if str(x).strip()][:3]
                    elif k == "address" and isinstance(v, str):
                        # Fallback: single string -> addr_lines (e.g. from Preisdatenbank)
                        lines = [ln.strip() for ln in v.replace("\r\n", "\n").split("\n") if ln.strip()][:3]
                        if lines:
                            COMPANY["addr_lines"] = lines
                    elif k != "handler_name":
                        COMPANY[k] = v
            # Dacă tenantul a setat doar Firmenname (name), folosim și pentru legal ca să apară în header/footer
            if "name" in company_overrides and company_overrides.get("name") and "legal" not in company_overrides:
                COMPANY["legal"] = str(company_overrides["name"]).strip()
        print(f"🔍 [PDF] COMPANY after overrides: name={COMPANY.get('name')!r}, legal={COMPANY.get('legal')!r}, addr_lines={COMPANY.get('addr_lines')!r}, phone={COMPANY.get('phone')!r}, email={COMPANY.get('email')!r}", flush=True)
        client_data_untranslated = frontend_data.get("client", frontend_data)
        nivel_oferta = _normalize_nivel_oferta(frontend_data)
        roof_only = bool(frontend_data.get("roof_only_offer")) if isinstance(frontend_data, dict) else False
        inclusions = _get_offer_inclusions(nivel_oferta, frontend_data)
        
        try: 
            plan_infos = load_plan_infos(run_id, stage_name="pricing")
        except PlansListError: 
            plan_infos = []
        plan_infos_ordered = list(plan_infos)
        pid_to_kind, pid_to_label = _plan_id_to_aufstockung_kind_and_label(
            run_id, plan_infos_ordered, job_root, frontend_data
        )

        enriched_plans = []
        for plan in plan_infos:
            plan_name_parts = plan.plan_id.split('_')
            
            if len(plan_name_parts) >= 2 and plan_name_parts[-2] == 'cluster':
                meta_filename = f"{plan_name_parts[-2]}_{plan_name_parts[-1]}.json"
            else:
                meta_filename = f"{plan.plan_id}.json"
        
            meta_path = output_root / "plan_metadata" / meta_filename 
            
            floor_type = "unknown"
            
            if meta_path.exists():
                try: 
                    raw_type = json.load(open(meta_path)).get("floor_classification", {}).get("floor_type", "unknown")
                    floor_type = raw_type.strip().lower()
                except Exception as e: 
                    floor_type = "unknown"
            
            if floor_type == "unknown":
                if "parter" in plan.plan_id.lower() or "ground" in plan.plan_id.lower(): 
                    floor_type = "ground_floor"
                elif "etaj" in plan.plan_id.lower() or "top" in plan.plan_id.lower(): 
                    floor_type = "top_floor"
            
            sort_key = 0 if floor_type == "ground_floor" else 1
            enriched_plans.append({"plan": plan, "floor_type": floor_type, "sort": sort_key})
            
        enriched_plans.sort(key=lambda x: x["sort"])
        
        plans_data = []
        global_openings = []
        global_utilities = []
        aufstockung_floor_kinds = _load_aufstockung_floor_kinds(run_id, frontend_data, job_root)

        for raw_idx, p_data in enumerate(enriched_plans):
            plan = p_data["plan"]
            pricing_path = plan.stage_work_dir / "pricing_raw.json"
            
            if pricing_path.exists():
                with open(pricing_path, encoding="utf-8") as f: 
                    p_json = json.load(f)
                
                breakdown = p_json.get("breakdown", {})
                filtered_breakdown = {}
                filtered_total = 0.0
                
                for category_key, category_data in breakdown.items():
                    # Stairs este inclus dacă floors_ceilings este inclus
                    if category_key == "stairs" and inclusions.get("floors_ceilings", False):
                        filtered_breakdown[category_key] = category_data
                        filtered_total += category_data.get("total_cost", 0.0)
                        continue
                    if category_key == "aufstockung_phase1" and inclusions.get("floors_ceilings", False):
                        filtered_breakdown[category_key] = category_data
                        filtered_total += category_data.get("total_cost", 0.0)
                        continue
                    
                    # Fireplace este inclus dacă utilities este inclus (fireplace face parte din utilities)
                    if category_key == "fireplace" and inclusions.get("utilities", False):
                        filtered_breakdown[category_key] = category_data
                        filtered_total += category_data.get("total_cost", 0.0)
                        continue
                    
                    # Basement este inclus dacă foundation este inclus (basement face parte din fundație/structură)
                    if category_key == "basement" and inclusions.get("foundation", False):
                        filtered_breakdown[category_key] = category_data
                        filtered_total += category_data.get("total_cost", 0.0)
                        continue
                    
                    # Verificăm dacă categoria este în inclusions
                    if inclusions.get(category_key, False):
                        filtered_breakdown[category_key] = category_data
                        filtered_total += category_data.get("total_cost", 0.0)
                
                p_json["breakdown"] = filtered_breakdown
                p_json["total_cost_eur"] = filtered_total
                
                if inclusions.get("utilities", False): 
                    util_bd = breakdown.get("utilities", {}) or {}
                    global_utilities.extend(util_bd.get("items", []) or util_bd.get("detailed_items", []) or [])
                # Adăugăm și fireplace în global_utilities pentru afișare (doar dacă utilities este inclus)
                if inclusions.get("utilities", False) and breakdown.get("fireplace"):
                    fireplace_bd = breakdown.get("fireplace", {})
                    global_utilities.extend(fireplace_bd.get("items", []) or fireplace_bd.get("detailed_items", []) or [])
                if inclusions.get("openings", False): 
                    global_openings.extend(breakdown.get("openings", {}).get("items", []))
                if inclusions.get("roof", False):
                    roof_items = breakdown.get("roof", {}).get("items", [])
                    extra_wall = next((it for it in roof_items if "extra_walls" in it.get("category", "")), None)
                    if extra_wall and inclusions.get("structure_walls", False):
                        cost = extra_wall.get("cost", 0.0)
                        ws = filtered_breakdown.get("structure_walls", {})
                        target = next((it for it in ws.get("items", []) if "Außenverkleidung" in it.get("name","") or "Fassadenverkleidung" in it.get("name","") or "Außenwände" in it.get("name","") or "Exterior" in it.get("name","")), None)
                        if target: 
                            target["cost"] += cost
                            ws["total_cost"] += cost
                            filtered_breakdown.get("roof", {})["total_cost"] -= cost
                            filtered_total += cost
        
                floor_kind = str(pid_to_kind.get(plan.plan_id, "existing")).strip().lower()
                floor_label = str(pid_to_label.get(plan.plan_id, "") or "").strip() or f"Stockwerk {raw_idx + 1}"
                manifest_plan_index = next(
                    (i for i, pinfo in enumerate(plan_infos_ordered) if pinfo.plan_id == plan.plan_id),
                    raw_idx,
                )
                plans_data.append(
                    {
                        "info": plan,
                        "type": p_data["floor_type"],
                        "pricing": p_json,
                        "floor_kind": floor_kind,
                        "floor_label": floor_label,
                        "manifest_plan_index": int(manifest_plan_index),
                        # Keep unfiltered costs for cross-sections (e.g. Wintergarten glass in Fenster/Verglasung).
                        "pricing_breakdown_full": breakdown,
                    }
                )
        
        # ✅ Înlocuiește roof breakdown cu roof_pricing.json dacă există (pentru top floor, când roof e inclus)
        # Căutăm roof_pricing în toate locațiile posibile (OUTPUT_ROOT, JOBS_ROOT/output, RUNNER_ROOT/output)
        def _find_roof_pricing_path() -> Path | None:
            for base in (OUTPUT_ROOT / run_id, JOBS_ROOT / run_id / "output", RUNNER_ROOT / "output" / run_id):
                p = base / "roof" / "roof_3d" / "entire" / "mixed" / "roof_pricing.json"
                if p.exists():
                    return p
            return output_root / "roof" / "roof_3d" / "entire" / "mixed" / "roof_pricing.json"
        roof_pricing_path = _find_roof_pricing_path()
        roof_pricing_total_eur: float | None = None  # Preț acoperiș din workflow nou (roof_metrics + formular)
        roof_skylight_count = 0
        roof_skylight_total_cost = 0.0
        roof_pricing: dict = {}
        if roof_pricing_path and roof_pricing_path.exists() and inclusions.get("roof", False):
            try:
                with open(roof_pricing_path, encoding="utf-8") as f:
                    roof_pricing = json.load(f)
                items = roof_pricing.get("detailed_items") or roof_pricing.get("items", [])
                form_data = roof_pricing.get("form_data") or {}
                roof_skylight_count = int(form_data.get("dachfenster_count") or 0)
                roof_skylight_total_cost = float(
                    sum(float(it.get("cost", 0) or 0) for it in items if str(it.get("category", "")).lower() == "roof_skylights")
                )
                total_roof = roof_pricing.get("total_cost", 0.0)
                if items and total_roof > 0:
                    roof_pricing_total_eur = total_roof
                    roof_breakdown = {"total_cost": total_roof, "items": items, "detailed_items": items}
                    # Adaugă la planul TOP FLOOR (ultimul după sortare) sau singurul plan (clădire cu 1 etaj)
                    top_entry = next((e for e in reversed(plans_data) if e.get("type") != "ground_floor"), None)
                    if top_entry is None and len(plans_data) == 1:
                        top_entry = plans_data[0]  # Casă cu 1 etaj – acel etaj are acoperișul
                    if top_entry:
                        p_json = top_entry["pricing"]
                        old_roof = p_json.get("breakdown", {}).get("roof", {})
                        old_cost = old_roof.get("total_cost", 0.0)
                        p_json["breakdown"]["roof"] = roof_breakdown
                        p_json["total_cost_eur"] = p_json.get("total_cost_eur", 0.0) - old_cost + total_roof
                        print(f"✅ [PDF] Roof pricing din roof_pricing.json aplicat la top floor: {total_roof:,.2f} EUR")
            except Exception as e:
                print(f"⚠️ [PDF] Eroare la încărcarea roof_pricing.json: {e}")
        
        # ✅ INIȚIALIZARE ENFORCER (fără collect & process_translation_queue)
        print("🇩🇪 [PDF] Initializing GermanEnforcer (Direct Table Translation Mode)...")
        enforcer = GermanEnforcer()
        
        # --- BUILD PDF ---
        # Prefer offer_no from job (API sets client_id-YYYY-NNN per tenant counter)
        if isinstance(frontend_data, dict) and frontend_data.get("offer_no"):
            offer_no = str(frontend_data["offer_no"]).strip()
        else:
            offer_no = f"{offer_prefix}-{datetime.now().strftime('%Y')}-{random.randint(1000,9999)}"
        # PDF-Metadaten für Browser-Tab; leerer Author → oft „(anonymous)“.
        _pdf_author = (
            str(COMPANY.get("name") or "").strip()
            or str(COMPANY.get("legal") or "").strip()
            or "Holzbot"
        )
        doc = SimpleDocTemplate(
            str(output_path), 
            pagesize=A4, 
            leftMargin=18*mm, 
            rightMargin=18*mm, 
            topMargin=42*mm, 
            bottomMargin=22*mm, 
            title=f"Angebot {offer_no}", 
            author=_pdf_author,
        )
        
        styles = _styles()
        story = []
        
        _header_block(story, styles, offer_no, client_data_untranslated, enforcer, assets=assets, tenant_slug=tenant_slug)
        if roof_only:
            # Același intro configurabil ca la Vollangebot (Angebotsanpassung / preview)
            _intro(
                story,
                styles,
                client_data_untranslated,
                enforcer,
                offer_title_effective,
                str(pdf_text_overrides.get("intro_content") or ""),
            )
            _roof_only_summary(story, styles, frontend_data, enforcer)
            # Roof-only offer: show chosen price positions (aggregated total), without measurements.
            if roof_pricing_path and roof_pricing_path.exists():
                try:
                    rp = roof_pricing if isinstance(roof_pricing, dict) else {}
                    rp_items = rp.get("detailed_items") or rp.get("items") or []
                    rp_items = [it for it in rp_items if float(it.get("cost", 0) or 0) > 0]
                    if rp_items:
                        story.append(Paragraph("Preisaufstellung Dach (gesamt)", styles["H2"]))
                        story.append(Spacer(1, 2 * mm))
                        head = [P("Position", "CellBold"), P("Preis", "CellBold")]
                        rows = []
                        for it in rp_items:
                            rows.append([
                                P(str(it.get("name") or "Position"), "Cell"),
                                P(_money(float(it.get("cost", 0) or 0)), "Cell"),
                            ])
                        rows.append([P("<b>Gesamtsumme Dach</b>", "CellBold"), P(_money(float(rp.get("total_cost", 0) or 0)), "CellBold")])
                        tbl = Table([head] + rows, colWidths=[125 * mm, 45 * mm])
                        tbl.setStyle(TableStyle([
                            ("GRID", (0, 0), (-1, -1), 0.3, colors.black),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1E6D3")),
                            ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ]))
                        story.append(tbl)
                        story.append(Spacer(1, 3 * mm))
                except Exception as _e:
                    pass
        else:
            _intro(
                story,
                styles,
                client_data_untranslated,
                enforcer,
                offer_title_effective,
                str(pdf_text_overrides.get("intro_content") or ""),
            )
            # Secțiunea 2: Prezentare generală proiect (doar informații comune)
            frontend_data_with_run_id = {**frontend_data, "run_id": run_id}
            _project_overview(
                story,
                styles,
                frontend_data_with_run_id,
                enforcer,
                plans_data,
                job_root=job_root,
            )
        
        # Structură de cost simplificată (eliminată complet conform cerințelor)
        # _simplified_cost_structure(story, styles, plans_data, inclusions, enforcer)

        # Calculează ferestre și uși la total pentru afișare la planuri.
        # Dachfenster (din roof editor) se includ explicit la "Fenster".
        total_windows_global = 0
        total_doors_global = 0
        total_doors_interior = 0
        total_doors_exterior = 0
        total_windows_area_global = 0.0
        total_doors_area_global = 0.0
        
        if plans_data and not roof_only:
            for entry in plans_data:
                pricing = entry.get("pricing", {})
                breakdown = pricing.get("breakdown", {})
                openings_bd = breakdown.get("openings", {})
                openings_items = openings_bd.get("items", []) or openings_bd.get("detailed_items", [])
                
                for op in openings_items:
                    obj_type = str(op.get("type", "")).lower()
                    area_m2 = float(op.get("area_m2", 0.0))
                    
                    if "window" in obj_type or obj_type == "sliding_door":
                        total_windows_global += 1
                        total_windows_area_global += area_m2
                    elif "door" in obj_type and obj_type != "sliding_door":
                        total_doors_global += 1
                        total_doors_area_global += area_m2
                        # Status poate fi în status sau location (pentru compatibilitate)
                        # Status este păstrat în items din calculate_openings_details
                        status_raw = op.get("status", op.get("location", ""))
                        status = str(status_raw).lower()
                        
                        # Debug pentru a vedea ce se întâmplă
                        print(f"🔍 [PDF] Door item: type={obj_type}, status_raw={status_raw}, status={status}, keys={list(op.keys())}")
                        
                        # Verificăm dacă este exterior (status="exterior" sau location="Exterior")
                        if status == "exterior":
                            total_doors_exterior += 1
                            print(f"✅ [PDF] Exterior door counted: total_exterior={total_doors_exterior}")
                        else:
                            total_doors_interior += 1
                            print(f"✅ [PDF] Interior door counted: total_interior={total_doors_interior}")
            
            # Debug final
            print(f"🔍 [PDF] Final door counts: total={total_doors_global}, interior={total_doors_interior}, exterior={total_doors_exterior}")
            # Dachfenster aus roof_pricing nur zur Fenster-Sektion zählen, wenn Angebotsumfang Fenster enthält
            if inclusions.get("openings") and roof_skylight_count > 0:
                total_windows_global += roof_skylight_count
        
        # Planuri (side by side) și ferestre/uși; planurile apar și la ofertă Dachstuhl-only
        if plans_data:
            # Wintergarten-Glas: im Angebot direkt vor Dach ausweisen.
            wg_glass_rows: list[tuple[str, float]] = []
            wg_glass_total = 0.0
            for idx_wg, entry in enumerate(plans_data, 1):
                bd_wg = entry.get("pricing_breakdown_full") or (entry.get("pricing") or {}).get("breakdown", {}) or {}
                wb = bd_wg.get("wintergaerten_balkone", {}) or {}
                for it in (wb.get("detailed_items", []) or []):
                    if not isinstance(it, dict):
                        continue
                    if str(it.get("category") or "").strip().lower() != "wintergarten_glass":
                        continue
                    cost_wg = float(it.get("total_cost") or it.get("cost") or 0.0)
                    if cost_wg <= 0:
                        continue
                    plan_lbl = str(entry.get("floor_label") or f"Plan {idx_wg}")
                    wg_glass_rows.append((plan_lbl, cost_wg))
                    wg_glass_total += cost_wg
            if wg_glass_total > 0:
                wg_content = [Paragraph("<b>Wintergarten – Glasflächen</b>", styles["H3"])]
                for plan_lbl, cost_wg in wg_glass_rows:
                    wg_content.append(Paragraph(f"{plan_lbl}: {_money(cost_wg)}", styles["Body"]))
                wg_content.append(Paragraph(f"Gesamtpreis Wintergarten Glas: {_money(wg_glass_total)}", styles["Body"]))
                wg_content.append(Spacer(1, 2 * mm))
                story.append(KeepTogether(wg_content))

            # Dach / Dämmung & Dachdeckung – preț acoperiș din roof_pricing.json sau fallback la plans_data
            roof_total_price = roof_pricing_total_eur if roof_pricing_total_eur is not None and roof_pricing_total_eur > 0 else 0.0
            if roof_total_price == 0:
                for entry in plans_data:
                    bd = entry.get("pricing", {}).get("breakdown", {})
                    roof_total_price += bd.get("roof", {}).get("total_cost", 0.0)
            # Dach / Dämmung & Dachdeckung – afișat doar în tabelul de prețuri, nu separat

            # Aufstockung: show editor selections for existing floors explicitly,
            # even when they are excluded from standard room/structure blocks.
            if (
                str(frontend_data.get("wizard_package") or "").strip().lower() in ("aufstockung", "zubau", "zubau_aufstockung")
                and not roof_only
            ):
                for entry in plans_data:
                    fk_ph = str(entry.get("floor_kind") or "").strip().lower()
                    if fk_ph in ("new", "zubau", "aufstockung"):
                        continue
                    breakdown = (entry.get("pricing") or {}).get("breakdown", {}) or {}
                    phase1 = breakdown.get("aufstockung_phase1", {}) or {}
                    phase1_rows: list[str] = []
                    subtotal = 0.0
                    for item in (phase1.get("detailed_items", []) or []):
                        if not isinstance(item, dict):
                            continue
                        name = str(item.get("name") or "Position")
                        unit_price = float(item.get("unit_price") or 0.0)
                        total_cost = float(item.get("total_cost") or item.get("cost") or 0.0)
                        cat = str(item.get("category") or "")
                        if cat == "aufstockung_stair_opening":
                            qty = float(item.get("quantity") or 1.0)
                            oa = item.get("opening_area_m2")
                            ow, ol = item.get("opening_width_m"), item.get("opening_length_m")
                            extra = ""
                            if ow is not None and ol is not None and float(ow) > 0 and float(ol) > 0:
                                le = max(float(ow), float(ol))
                                br = min(float(ow), float(ol))
                                extra = f" (Öffnung ca. {le:.2f} × {br:.2f} m"
                                if oa is not None and float(oa) > 0:
                                    extra += f", {float(oa):.2f} m²"
                                extra += ")"
                            elif oa is not None and float(oa) > 0:
                                extra = f" (Öffnung ca. {float(oa):.2f} m²)"
                            phase1_rows.append(
                                f"{name}{extra}: {qty:.0f} Stk. × {unit_price:.2f} EUR = {total_cost:.2f} EUR"
                            )
                            subtotal += total_cost
                            continue
                        qty = float(item.get("area_m2") or item.get("quantity") or 0.0)
                        unit_suffix = (
                            "m²"
                            if ("area_m2" in item and item.get("area_m2") is not None)
                            or cat == "aufstockung_demolition"
                            or cat == "aufstockung_statik"
                            else "Stk."
                        )
                        phase1_rows.append(
                            f"{name}: {qty:.2f} {unit_suffix} × {unit_price:.2f} EUR = {total_cost:.2f} EUR"
                        )
                        subtotal += total_cost
                    if not phase1_rows:
                        continue
                    fl_ph = str(entry.get("floor_label") or "").strip() or "Geschoss"
                    # Phase-1 positions apply to Bestand / bearbeitete Geschosse (Abbruch, Statik, Treppenöffnung).
                    title_ph = "Aufstockung"
                    phase1_content = [Paragraph(f"<b>{title_ph} – {fl_ph}</b>", styles["H3"])]
                    for line in phase1_rows:
                        phase1_content.append(Paragraph(f"• {line}", styles["Body"]))
                    phase1_content.append(Paragraph(f"Summe ({title_ph}): {_money(subtotal)}", styles["Body"]))
                    phase1_content.append(Spacer(1, 2 * mm))
                    story.append(KeepTogether(phase1_content))
        
            # Fenster & Verglasung / Türen / Dachfenster – afișate doar în tabelul de prețuri, nu separat
        
            plan_images = []
            plan_labels = []
            plan_info_texts = []

            for entry in plans_data:
                plan = entry["info"]
                pricing = entry["pricing"]
                plan_labels.append("")
                info_text = []
                plan_info_texts.append("<br/>".join(info_text) if info_text else "")
                plan_img_path = resolve_plan_image_for_pdf(plan.plan_image, plan.plan_id, output_root, job_root)
                if plan_img_path and plan_img_path.exists():
                    try:
                        im = PILImage.open(plan_img_path).convert("L")
                        im = ImageEnhance.Brightness(im).enhance(0.9)
                        im = ImageOps.autocontrast(im)
                        width, height = im.size
                        aspect = width / height
                        target_width = (A4[0]-36*mm-10*mm) / 2
                        if aspect < 1:
                            target_width = target_width * 0.9
                        img_byte_arr = io.BytesIO()
                        im.save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)
                        rl_img = Image(img_byte_arr)
                        rl_img._restrictSize(target_width, 100*mm)
                        plan_images.append(rl_img)
                    except Exception:
                        plan_images.append(None)
                else:
                    plan_images.append(None)

            num_plans = len(plans_data)
            for i in range(0, num_plans, 2):
                row_images = []
                row_labels = []
                row_infos = []

                if i < num_plans:
                    row_labels.append(plan_labels[i])
                    row_infos.append(plan_info_texts[i])
                    row_images.append(plan_images[i] if plan_images[i] else "")

                if i + 1 < num_plans:
                    row_labels.append(plan_labels[i + 1])
                    row_infos.append(plan_info_texts[i + 1])
                    row_images.append(plan_images[i + 1] if plan_images[i + 1] else "")
                else:
                    row_labels.append("")
                    row_infos.append("")
                    row_images.append("")

                col_width = (A4[0]-36*mm-10*mm) / 2

                img_row = [
                    row_images[0] if row_images[0] else P("", "Body"),
                    row_images[1] if row_images[1] else P("", "Body")
                ]

                table_data = [img_row]
                tbl = Table(table_data, colWidths=[col_width, col_width])
                tbl.setStyle(TableStyle([
                    ("VALIGN", (0,0), (-1,-1), "TOP"),
                    ("ALIGN", (0,0), (-1,-1), "CENTER"),
                    ("LEFTPADDING", (0,0), (-1,-1), 0),
                    ("RIGHTPADDING", (0,0), (-1,-1), 5*mm),
                ]))

                story.append(tbl)
                if i + 2 < num_plans:
                    story.append(Spacer(1, 6*mm))
        
        # Nu mai afișăm tabele detaliate pentru openings și utilities - sunt în structura simplificată
        
        story.append(Spacer(1, 6*mm))
        story.append(Paragraph(enforcer.get("Gesamtkostenzusammenstellung"), styles["H1"]))
        story.append(Spacer(1, 3*mm))
        
        # Recalculăm filtered_total din breakdown-ul filtrat pentru a ne asigura că este corect
        # (plans_data are deja breakdown-ul filtrat și total_cost_eur recalculat)
        filtered_total = sum(e["pricing"].get("total_cost_eur", 0.0) for e in plans_data)
        
        # Verificare: dacă filtered_total este 0, încercăm să calculăm din breakdown direct
        if filtered_total == 0:
            filtered_total = 0.0
            for entry in plans_data:
                pricing = entry["pricing"]
                bd = pricing.get("breakdown", {})
                for cat_key, cat_data in bd.items():
                    if isinstance(cat_data, dict):
                        filtered_total += cat_data.get("total_cost", 0.0)
        
        baustelle_key, baustelle_percent = _resolve_selected_baustelleneinrichtung_percent(fd_ctx, summary_params)
        try:
            profit_percent = float((summary_params or {}).get("profit_margin_percent", 0.0) or 0.0)
        except Exception:
            profit_percent = 0.0
        total_percent = baustelle_percent + profit_percent
        if total_percent <= 0:
            total_percent = 10.0
        baustelle_amount = filtered_total * (baustelle_percent / 100.0)
        profit_amount = filtered_total * (profit_percent / 100.0)
        baukosten_with_profit = filtered_total + profit_amount
        print(
            f"🔍 [PDF] Summary percent parts | Baustelleneinrichtung=[({baustelle_key!r}, {baustelle_percent:.4f})] "
            f"| Profit=[('profit_margin_percent', {profit_percent:.4f})] | "
            f"TOTAL={total_percent:.4f}",
            flush=True,
        )
        print(
            f"🔍 [PDF] Summary amount | filtered_total={filtered_total:.2f} | "
            f"profit_amount={profit_amount:.2f} | baustelleneinrichtung_amount={baustelle_amount:.2f}",
            flush=True,
        )
        
        net = baukosten_with_profit + baustelle_amount
        vat = net * pdf_vat_rate
        gross = net + vat
        
        head = [
            P(enforcer.get("Position"), "CellBold"), 
            P(enforcer.get("Betrag"), "CellBold")
        ]
        
        pos0_label = (
            "Dachkosten (Dachstuhl)"
            if roof_only
            else _baukosten_position_label_de(inclusions)
        )
        data = [
            [P(pos0_label), P(_money(baukosten_with_profit))],
            [P(enforcer.get("Baustelleneinrichtung, Logistik & Planung")), P(_money(baustelle_amount))],
            [P(f"<b>{enforcer.get('Nettosumme (ohne MwSt.)')}</b>"), P(_money(net), "CellBold")],
            [P(_vat_line_label_for_offer(fd_ctx, pdf_vat_rate)), P(_money(vat))],
            [P(f"<b>{enforcer.get('GESAMTSUMME BRUTTO')}</b>"), P(_money(gross), "H2")],
        ]
        
        tbl = Table([head] + data, colWidths=[120*mm, 50*mm])
        tbl.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.3, colors.black), 
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F1E6D3")),  # sand pentru header 
            ("ALIGN", (1,1), (1,-1), "RIGHT"), 
            ("VALIGN", (0,0), (-1,-1), "MIDDLE")
        ]))
        
        story.append(tbl)
        story.append(Spacer(1, 5*mm))
        
        # Rechtlicher Block: aceleași texte configurabile (Angebotsanpassung) ca la Vollangebot
        _legal_disclaimer(story, styles, enforcer, pdf_text_overrides)
        
        doc.build(
            story, 
            onFirstPage=_first_page_canvas(offer_no, handler, assets=assets, tenant_slug=tenant_slug), 
            onLaterPages=_later_pages_canvas
        )
        
        print(f"✅ [PDF] Generat Final (DE): {output_path}")
        return output_path
    finally:
        _pdf_currency_suffix.reset(_pdf_money_tok)

# ---------- ADMIN PDF GENERATOR (STRICT, NO BRANDING) ----------
def generate_admin_offer_pdf(run_id: str, output_path: Path | None = None, job_root: Path | None = None) -> Path:
    """
    Generează un PDF strict pentru admin, fără branding fancy, fără "povesti",
    doar date brute și tabele cu prețuri.
    """
    print(f"🚀 [PDF ADMIN] START: {run_id}")
    
    jobs_root_base = PROJECT_ROOT / "jobs"
    output_root = jobs_root_base / run_id
    
    if not output_root.exists(): 
        output_root = RUNNER_ROOT / "output" / run_id
        if not output_root.exists():
            print(f"❌ EROARE: Nu găsesc directorul de output pentru run_id='{run_id}'")
            raise FileNotFoundError(f"Output nu există: {output_root}")
        print(f"⚠️ Folosind directorul standard: {output_root.resolve()}")
    else:
        print(f"✅ Folosind JOBS_ROOT: {output_root.resolve()}")

    if job_root is None:
        job_root = JOBS_ROOT / run_id
        if not job_root.exists():
            job_root = None

    if output_path is None:
        final_output_root = RUNNER_ROOT / "output" / run_id
        pdf_dir = final_output_root / "offer_pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        output_path = pdf_dir / f"oferta_admin_{run_id}.pdf"

    frontend_data = load_frontend_data_for_run(run_id, job_root)
    fd_ctx = frontend_data if isinstance(frontend_data, dict) else {}
    _pdf_money_tok = _pdf_currency_suffix.set("\u00a0CHF" if _parse_display_currency_code(fd_ctx) == "CHF" else "\u00a0€")
    admin_pdf_vat_rate = _parse_vat_rate_decimal(fd_ctx)
    try:
        tenant_slug = frontend_data.get("tenant_slug") if isinstance(frontend_data, dict) else None
        roof_only = bool(frontend_data.get("roof_only_offer")) if isinstance(frontend_data, dict) else False
        
        # Preisdatenbank (coefficienți preț) pentru tabelul Formular + Preise
        try:
            from pricing.db_loader import fetch_pricing_parameters
            calc_mode = frontend_data.get("calc_mode") or "default"
            pricing_coeffs = fetch_pricing_parameters(tenant_slug or "", calc_mode=calc_mode)
        except Exception as e:
            print(f"⚠️ [PDF ADMIN] Could not load pricing coefficients: {e}")
            pricing_coeffs = {}
        
        try:
            plan_infos = load_plan_infos(run_id, stage_name="pricing")
        except PlansListError: 
            plan_infos = []
        
        enriched_plans = []
        for plan in plan_infos:
            plan_name_parts = plan.plan_id.split('_')
            
            if len(plan_name_parts) >= 2 and plan_name_parts[-2] == 'cluster':
                meta_filename = f"{plan_name_parts[-2]}_{plan_name_parts[-1]}.json"
            else:
                meta_filename = f"{plan.plan_id}.json"
        
            meta_path = output_root / "plan_metadata" / meta_filename 
            
            floor_type = "unknown"
            
            if meta_path.exists():
                try: 
                    raw_type = json.load(open(meta_path)).get("floor_classification", {}).get("floor_type", "unknown")
                    floor_type = raw_type.strip().lower()
                except Exception as e: 
                    floor_type = "unknown"
            
            if floor_type == "unknown":
                if "parter" in plan.plan_id.lower() or "ground" in plan.plan_id.lower(): 
                    floor_type = "ground_floor"
                elif "etaj" in plan.plan_id.lower() or "top" in plan.plan_id.lower(): 
                    floor_type = "top_floor"
            
            sort_key = 0 if floor_type == "ground_floor" else 1
            enriched_plans.append({"plan": plan, "floor_type": floor_type, "sort": sort_key, "index": len(enriched_plans)})
            
        enriched_plans.sort(key=lambda x: x["sort"])
        
        # Încărcăm ordinea etajelor (de jos în sus) și indexul beciului pentru denumiri corecte în PDF
        order_from_bottom = None
        basement_plan_index = None
        for run_dir_candidate in (output_root, RUNNER_ROOT / "output" / run_id, RUNS_ROOT / run_id):
            if not Path(run_dir_candidate).exists():
                continue
            if order_from_bottom is None:
                fo_path = Path(run_dir_candidate) / "floor_order.json"
                if fo_path.exists():
                    try:
                        data = json.loads(fo_path.read_text(encoding="utf-8"))
                        ob = data.get("order_from_bottom")
                        if ob and len(ob) == len(plan_infos) and set(ob) == set(range(len(plan_infos))):
                            order_from_bottom = ob
                    except Exception:
                        pass
            if basement_plan_index is None:
                bp_path = Path(run_dir_candidate) / "basement_plan_id.json"
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

        if order_from_bottom is None:
            mp = JOBS_ROOT / run_id / "detections_review_manifest.json"
            if mp.exists():
                try:
                    mdata = json.loads(mp.read_text(encoding="utf-8"))
                    fpo = mdata.get("floorPlanOrder")
                    n_pi = len(plan_infos)
                    if (
                        isinstance(fpo, list)
                        and len(fpo) == n_pi
                        and n_pi > 0
                        and {int(x) for x in fpo} == set(range(n_pi))
                    ):
                        order_from_bottom = [int(x) for x in fpo]
                except Exception:
                    pass
        
        # Ordine finală: folosim order_from_bottom dacă există, altfel păstrăm sortarea ground_floor apoi restul
        enriched_by_idx = {p["index"]: p for p in enriched_plans}
        if order_from_bottom is not None:
            enriched_ordered = [enriched_by_idx[i] for i in order_from_bottom if i in enriched_by_idx]
            if len(enriched_ordered) != len(enriched_plans):
                enriched_ordered = enriched_plans
        else:
            enriched_ordered = enriched_plans
        
        # Atribuim etichete etaj: Keller, Erdgeschoss, 1. Obergeschoss, 2. Obergeschoss, ..., Mansardă/Dachgeschoss
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
        
        plans_data = []
        global_openings = []
        global_utilities = []
        
        # ADMIN: Folosim aceeași filtrare ca în user PDF pentru același preț
        nivel_oferta = _normalize_nivel_oferta(frontend_data)
        inclusions = _get_offer_inclusions(nivel_oferta, frontend_data)
        _sec_openings = "section 3" if inclusions.get("openings") else "area calculation"
        _note_ext_openings = (
            "<i>Exterior openings = sum of all window areas + exterior door areas (see section 3).</i>"
            if inclusions.get("openings")
            else "<i>Exterior openings = sum of window and exterior door areas from the plan (openings not itemized in this offer scope).</i>"
        )
        _note_int_openings = (
            "<i>Interior openings = sum of all interior door areas (see section 3).</i>"
            if inclusions.get("openings")
            else "<i>Interior openings = sum of interior door areas from the plan (openings not itemized in this offer scope).</i>"
        )
        
        for pos, p_data in enumerate(enriched_ordered):
            plan = p_data["plan"]
            pricing_path = plan.stage_work_dir / "pricing_raw.json"
            floor_label = _floor_display_label(pos, p_data, basement_plan_index, enriched_ordered)
            
            if pricing_path.exists():
                with open(pricing_path, encoding="utf-8") as f: 
                    p_json = json.load(f)
                
                breakdown = p_json.get("breakdown", {})
                filtered_breakdown = {}
                filtered_total = 0.0
                
                for category_key, category_data in breakdown.items():
                    # Stairs este inclus dacă floors_ceilings este inclus
                    if category_key == "stairs" and inclusions.get("floors_ceilings", False):
                        filtered_breakdown[category_key] = category_data
                        filtered_total += category_data.get("total_cost", 0.0)
                        continue
                    if category_key == "aufstockung_phase1" and inclusions.get("floors_ceilings", False):
                        filtered_breakdown[category_key] = category_data
                        filtered_total += category_data.get("total_cost", 0.0)
                        continue
                    
                    # Fireplace este inclus dacă utilities este inclus (fireplace face parte din utilities)
                    if category_key == "fireplace" and inclusions.get("utilities", False):
                        filtered_breakdown[category_key] = category_data
                        filtered_total += category_data.get("total_cost", 0.0)
                        continue
                    
                    # Basement este inclus dacă foundation este inclus (basement face parte din fundație/structură)
                    if category_key == "basement" and inclusions.get("foundation", False):
                        filtered_breakdown[category_key] = category_data
                        filtered_total += category_data.get("total_cost", 0.0)
                        continue
                    
                    # Wintergärten & Balkone: doar când oferta include finisaje (Altbau-Logik); nu pentru Rohbau-only
                    if (
                        category_key == "wintergaerten_balkone"
                        and not roof_only
                        and inclusions.get("finishes", False)
                    ):
                        filtered_breakdown[category_key] = category_data
                        filtered_total += category_data.get("total_cost", 0.0)
                        continue
                    
                    # Verificăm dacă categoria este în inclusions
                    if inclusions.get(category_key, False):
                        filtered_breakdown[category_key] = category_data
                        filtered_total += category_data.get("total_cost", 0.0)
                
                p_json["breakdown"] = filtered_breakdown
                p_json["total_cost_eur"] = filtered_total
                
                if inclusions.get("utilities", False): 
                    util_bd = breakdown.get("utilities", {}) or {}
                    global_utilities.extend(util_bd.get("items", []) or util_bd.get("detailed_items", []) or [])
                # Adăugăm și fireplace în global_utilities pentru afișare (doar dacă utilities este inclus)
                if inclusions.get("utilities", False) and breakdown.get("fireplace"):
                    fireplace_bd = breakdown.get("fireplace", {})
                    global_utilities.extend(fireplace_bd.get("items", []) or fireplace_bd.get("detailed_items", []) or [])
                if inclusions.get("openings", False): 
                    global_openings.extend(breakdown.get("openings", {}).get("items", []))
                if inclusions.get("roof", False):
                    roof_items = breakdown.get("roof", {}).get("items", [])
                    extra_wall = next((it for it in roof_items if "extra_walls" in it.get("category", "")), None)
                    if extra_wall and inclusions.get("structure_walls", False):
                        cost = extra_wall.get("cost", 0.0)
                        ws = filtered_breakdown.get("structure_walls", {})
                        target = next((it for it in ws.get("items", []) if "Außenverkleidung" in it.get("name","") or "Fassadenverkleidung" in it.get("name","") or "Außenwände" in it.get("name","") or "Exterior" in it.get("name","")), None)
                        if target: 
                            target["cost"] += cost
                            ws["total_cost"] += cost
                            filtered_breakdown.get("roof", {})["total_cost"] -= cost
                            filtered_total += cost
                
                plans_data.append({"info": plan, "type": p_data["floor_type"], "floor_label": floor_label, "pricing": p_json})
        
        # ADMIN: Enforcer simplu, fără branding
        enforcer = GermanEnforcer()
        
        # --- BUILD PDF (MINIMAL) ---
        offer_no = f"ADMIN-{datetime.now().strftime('%Y')}-{random.randint(1000,9999)}"
        doc = SimpleDocTemplate(
            str(output_path), 
            pagesize=A4, 
            leftMargin=18*mm, 
            rightMargin=18*mm, 
            topMargin=20*mm, 
            bottomMargin=15*mm, 
            title=f"Admin Offer {offer_no}", 
            author="Admin"
        )
        
        styles = _styles()
        story = []
        
        # ADMIN: Header minimal, fără date client
        story.append(Paragraph(f"ADMIN OFFER - {offer_no}", styles["H1"]))
        story.append(Paragraph(f"<b>Mandant:</b> {tenant_slug or '—'}", styles["Cell"]))
        story.append(Spacer(1, 6*mm))
        if roof_only:
            story.append(Paragraph("<b>Dachstuhl-Schätzung</b> – nur Dachkosten (kein Formularpreis-Überblick für Vollhaus).", styles["Body"]))
            story.append(Spacer(1, 4*mm))
        
        # ADMIN: Tabel Variablen & Preise (Formularauswahl + Preisdatenbank)
        if not roof_only:
            story.append(Spacer(1, 6*mm))
            story.append(Paragraph("VARIABLEN & PREISE (FORMULARAUSWAHL UND PREISDATENBANK)", styles["H2"]))
            story.append(Spacer(1, 2*mm))
            try:
                form_preis_rows = _build_form_preisdatenbank_rows(frontend_data, pricing_coeffs, enforcer)
                if form_preis_rows:
                    head = [
                        P(enforcer.get("Kategorie") or "Kategorie", "CellBold"),
                        P(enforcer.get("Parameter") or "Parameter", "CellBold"),
                        P("Im Formular gewählt", "CellBold"),
                        P("Preis (Preisdatenbank)", "CellBold"),
                    ]
                    data = []
                    for cat, param, form_val, price_str in form_preis_rows:
                        data.append([P(cat, "CellSmall"), P(param, "CellSmall"), P(form_val or "—", "Cell"), P(price_str or "—", "Cell")])
                    tbl = Table([head] + data, colWidths=[35*mm, 48*mm, 52*mm, 42*mm])
                    tbl.setStyle(TableStyle([
                        ("GRID", (0,0), (-1,-1), 0.3, colors.black),
                        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F1E6D3")),
                        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                        ("ALIGN", (3,0), (3,-1), "RIGHT"),
                        ("FONTSIZE", (0,0), (-1,-1), 8),
                        ("LEFTPADDING", (0,0), (-1,-1), 3),
                        ("RIGHTPADDING", (0,0), (-1,-1), 3),
                    ]))
                    story.append(tbl)
                else:
                    story.append(Paragraph("Keine Preisdatenbank geladen oder keine Variablen.", styles["Small"]))
            except Exception as e:
                story.append(Paragraph(f"Tabelle konnte nicht erstellt werden: {e}", styles["Small"]))
        
        # ADMIN: Planuri cu imagini și date detaliate
        story.append(Paragraph("DACH – KOSTEN (DETAIL)" if roof_only else "DETALII PLANURI & KOSTEN", styles["H2"]))
        story.append(Spacer(1, 3*mm))
        
        for entry in plans_data:
            plan = entry["info"]
            pricing = entry["pricing"]
            
            # Nu mai adăugăm PageBreak automat, doar spacing
            if entry != plans_data[0]:
                story.append(Spacer(1, 6*mm))
            
            entry_type_clean = entry["type"].strip().lower() 
            floor_label = entry.get("floor_label") or ("Erdgeschoss" if entry_type_clean == "ground_floor" else "Obergeschoss / Dachgeschoss")
            
            story.append(Paragraph(f"Plan: {floor_label} ({plan.plan_id})", styles["H2"]))
            story.append(Spacer(1, 2*mm))
            
            plan_img_path = resolve_plan_image_for_pdf(plan.plan_image, plan.plan_id, output_root, job_root)
            if plan_img_path and plan_img_path.exists():
                try:
                    im = PILImage.open(plan_img_path).convert("L")
                    im = ImageEnhance.Brightness(im).enhance(0.9)
                    im = ImageOps.autocontrast(im)
                    width, height = im.size
                    aspect = width / height
                    target_width = A4[0]-36*mm
                    if aspect < 1:
                        target_width = (A4[0]-36*mm) * 0.65
                    img_byte_arr = io.BytesIO()
                    im.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    rl_img = Image(img_byte_arr)
                    rl_img._restrictSize(target_width, 75*mm)
                    rl_img.hAlign = 'CENTER'
                    story.append(Spacer(1, 2*mm))
                    story.append(rl_img)
                    story.append(Spacer(1, 3*mm))
                except Exception as e:
                    print(f"⚠️ [PDF ADMIN] Nu pot încărca imaginea planului: {e}")
            
            bd = pricing.get("breakdown", {})
            
            # ADMIN: Afișăm TOATE categoriile, fără filtru
            # ADMIN: Nu afișăm coloana Mod (nu se mai folosește)
            if bd.get("foundation"): 
                _table_standard(story, styles, "Fundament / Bodenplatte", bd.get("foundation", {}), enforcer, show_mod_column=False)
            if bd.get("basement"): 
                _table_standard(story, styles, "Keller / Untergeschoss", bd.get("basement", {}), enforcer, show_mod_column=False)
            if bd.get("structure_walls"): 
                _table_standard(story, styles, "Tragwerkskonstruktion – Wände", bd.get("structure_walls", {}), enforcer, show_mod_column=False)
                
                # Adaugă tabelul cu măsurătorile pereților pentru acest plan
                try:
                    area_json_path = plan.stage_work_dir.parent.parent / "area" / plan.plan_id / "areas_calculated.json"
                    if area_json_path.exists():
                        with open(area_json_path, "r", encoding="utf-8") as f:
                            area_data = json.load(f)
                        
                        from .tables import create_wall_measurements_table
                        story.append(Spacer(1, 2*mm))
                        story.append(Paragraph(f"<b>{enforcer.get('Măsurători Pereți')} - {plan.plan_id}:</b>", styles["Body"]))
                        measurements_table = create_wall_measurements_table(area_data, enforcer, inclusions)
                        story.append(measurements_table)
                        story.append(Spacer(1, 3*mm))
                except Exception as e:
                    print(f"⚠️ [PDF ADMIN] Could not load area_data for {plan.plan_id}: {e}")
            
            if bd.get("floors_ceilings"): 
                _table_standard(story, styles, "Geschossdecken & Balken", bd.get("floors_ceilings", {}), enforcer, show_mod_column=False)
            if bd.get("stairs"): 
                _table_stairs(story, styles, bd.get("stairs", {}), enforcer)
            if bd.get("roof"): 
                _table_roof_quantities(story, styles, pricing, enforcer, inclusions=inclusions)
            if bd.get("finishes"): 
                _table_standard(story, styles, "Oberflächen & Ausbau", bd.get("finishes", {}), enforcer, show_mod_column=False)
            if bd.get("wintergaerten_balkone") and (bd.get("wintergaerten_balkone", {}).get("total_cost", 0) or 0) > 0:
                _table_standard(story, styles, "Wintergärten & Balkone", bd.get("wintergaerten_balkone", {}), enforcer, show_mod_column=False)
        
        # ADMIN: Toate deschiderile și utilitățile
        if global_openings: 
            _table_global_openings(story, styles, global_openings, enforcer)
        if global_utilities: 
            _table_global_utilities(story, styles, global_utilities, enforcer)
        
        # ADMIN: Summary strict
        story.append(PageBreak())
        story.append(Paragraph("Gesamtkostenzusammenstellung (ADMIN)", styles["H1"]))
        
        filtered_total = sum(e["pricing"].get("total_cost_eur", 0.0) for e in plans_data)
        
        summary_params = (pricing_coeffs or {}).get("_raw_params", {}) if isinstance(pricing_coeffs, dict) else {}
        baustelle_key, baustelle_percent = _resolve_selected_baustelleneinrichtung_percent(fd_ctx, summary_params)
        try:
            profit_percent = float((summary_params or {}).get("profit_margin_percent", 0.0) or 0.0)
        except Exception:
            profit_percent = 0.0
        total_percent = baustelle_percent + profit_percent
        if total_percent <= 0:
            total_percent = 10.0
        baustelle_amount = filtered_total * (baustelle_percent / 100.0)
        profit_amount = filtered_total * (profit_percent / 100.0)
        baukosten_with_profit = filtered_total + profit_amount
        print(
            f"🔍 [PDF ADMIN] Summary percent parts | Baustelleneinrichtung=[({baustelle_key!r}, {baustelle_percent:.4f})] "
            f"| Profit=[('profit_margin_percent', {profit_percent:.4f})] | "
            f"TOTAL={total_percent:.4f}",
            flush=True,
        )
        print(
            f"🔍 [PDF ADMIN] Summary amount | filtered_total={filtered_total:.2f} | "
            f"profit_amount={profit_amount:.2f} | baustelleneinrichtung_amount={baustelle_amount:.2f}",
            flush=True,
        )
        
        net = baukosten_with_profit + baustelle_amount
        vat = net * admin_pdf_vat_rate
        gross = net + vat
        
        head = [
            P("Position", "CellBold"), 
            P("Betrag", "CellBold")
        ]
        
        admin_pos0 = "Dachkosten (Dachstuhl)" if roof_only else _baukosten_position_label_de(inclusions)
        data = [
            [P(admin_pos0), P(_money(baukosten_with_profit))],
            [P("Baustelleneinrichtung, Logistik & Planung"), P(_money(baustelle_amount))],
            [P("<b>Nettosumme (ohne MwSt.)</b>"), P(_money(net), "CellBold")],
            [P(_vat_line_label_for_offer(fd_ctx, admin_pdf_vat_rate)), P(_money(vat))],
            [P("<b>GESAMTSUMME BRUTTO</b>"), P(_money(gross), "H2")],
        ]
        
        tbl = Table([head] + data, colWidths=[120*mm, 50*mm])
        tbl.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.3, colors.black), 
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F1E6D3")),  # sand pentru header 
            ("ALIGN", (1,1), (1,-1), "RIGHT"), 
            ("VALIGN", (0,0), (-1,-1), "MIDDLE")
        ]))
        
        story.append(tbl)
        
        # ADMIN: Fără closing blocks verbose, doar o notă minimală
        story.append(Spacer(1, 8*mm))
        story.append(Paragraph(
            "ADMIN VERSION – nur Dachkosten (Dachstuhl)" if roof_only else "ADMIN VERSION - Raw data, no branding",
            styles["Small"],
        ))
        
        # ADMIN: Fără canvas fancy, doar PDF simplu
        doc.build(story)
        
        print(f"✅ [PDF ADMIN] Generat: {output_path}")
        return output_path
    finally:
        _pdf_currency_suffix.reset(_pdf_money_tok)

# ---------- ADMIN CALCULATION METHOD PDF GENERATOR (ENGLISH) ----------
def generate_admin_calculation_method_pdf(run_id: str, output_path: Path | None = None, job_root: Path | None = None) -> Path:
    """
    Generates a detailed calculation method PDF for admin users, explaining
    all formulas, coefficients, and numerical examples in English.
    """
    print(f"🚀 [PDF CALC METHOD] START: {run_id}")
    
    jobs_root_base = PROJECT_ROOT / "jobs"
    output_root = jobs_root_base / run_id
    
    if not output_root.exists(): 
        output_root = RUNNER_ROOT / "output" / run_id
        if not output_root.exists():
            print(f"❌ EROARE: Nu găsesc directorul de output pentru run_id='{run_id}'")
            raise FileNotFoundError(f"Output nu există: {output_root}")
        print(f"⚠️ Folosind directorul standard: {output_root.resolve()}")
    else:
        print(f"✅ Folosind JOBS_ROOT: {output_root.resolve()}")

    if job_root is None:
        job_root = JOBS_ROOT / run_id
        if not job_root.exists():
            job_root = None

    if output_path is None:
        final_output_root = RUNNER_ROOT / "output" / run_id
        pdf_dir = final_output_root / "offer_pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        output_path = pdf_dir / f"calculation_method_{run_id}.pdf"

    frontend_data = load_frontend_data_for_run(run_id, job_root)
    fd_cm = frontend_data if isinstance(frontend_data, dict) else {}
    calc_method_vat = _parse_vat_rate_decimal(fd_cm)
    calc_method_currency = _parse_display_currency_code(fd_cm)
    calc_cur_label = "CHF" if calc_method_currency == "CHF" else "EUR"
    tenant_slug = frontend_data.get("tenant_slug") if isinstance(frontend_data, dict) else None
    
    try: 
        plan_infos = load_plan_infos(run_id, stage_name="pricing")
    except PlansListError: 
        plan_infos = []

    # Load pricing coefficients
    from pricing.db_loader import fetch_pricing_parameters
    calc_mode = frontend_data.get("calc_mode") or "default"
    try:
        pricing_coeffs = fetch_pricing_parameters(tenant_slug, calc_mode=calc_mode)
    except Exception as e:
        print(f"⚠️ Could not load pricing coefficients: {e}")
        pricing_coeffs = {}

    # Load actual pricing data for each plan - folosim aceeași filtrare ca în user PDF
    nivel_oferta = _normalize_nivel_oferta(frontend_data)
    inclusions = _get_offer_inclusions(nivel_oferta, frontend_data)
    auf_kinds_admin = _load_aufstockung_floor_kinds(run_id, frontend_data, job_root)

    plans_data = []
    for pi_admin, plan in enumerate(plan_infos):
        pricing_path = plan.stage_work_dir / "pricing_raw.json"
        if pricing_path.exists():
            with open(pricing_path, encoding="utf-8") as f: 
                p_json = json.load(f)
            
            breakdown = p_json.get("breakdown", {})
            filtered_breakdown = {}
            filtered_total = 0.0
            
            for category_key, category_data in breakdown.items():
                # Stairs este inclus dacă floors_ceilings este inclus
                if category_key == "stairs" and inclusions.get("floors_ceilings", False):
                    filtered_breakdown[category_key] = category_data
                    filtered_total += category_data.get("total_cost", 0.0)
                    continue
                if category_key == "aufstockung_phase1" and inclusions.get("floors_ceilings", False):
                    filtered_breakdown[category_key] = category_data
                    filtered_total += category_data.get("total_cost", 0.0)
                    continue
                
                # Fireplace este inclus dacă utilities este inclus (fireplace face parte din utilities)
                if category_key == "fireplace" and inclusions.get("utilities", False):
                    filtered_breakdown[category_key] = category_data
                    filtered_total += category_data.get("total_cost", 0.0)
                    continue
                
                # Verificăm dacă categoria este în inclusions
                if inclusions.get(category_key, False):
                    filtered_breakdown[category_key] = category_data
                    filtered_total += category_data.get("total_cost", 0.0)
            
            p_json["breakdown"] = filtered_breakdown
            p_json["total_cost_eur"] = filtered_total
            fk_adm = (
                str(auf_kinds_admin[pi_admin]).strip().lower()
                if pi_admin < len(auf_kinds_admin)
                else "existing"
            )
            plans_data.append({"info": plan, "pricing": p_json, "floor_kind": fk_adm})

    # Load basement (Keller) info: which plan index is treated as basement
    basement_plan_index = None
    for run_dir_candidate in (RUNS_ROOT / run_id, output_root):
        bp_file = Path(run_dir_candidate) / "basement_plan_id.json"
        if bp_file.exists():
            try:
                bp_data = json.loads(bp_file.read_text(encoding="utf-8"))
                raw = bp_data.get("basement_plan_index")
                if raw is not None:
                    idx = int(raw) if isinstance(raw, (int, float)) else int(raw)
                    if 0 <= idx < len(plan_infos):
                        basement_plan_index = idx
                        break
            except Exception:
                pass

    # Build PDF
    doc = SimpleDocTemplate(
        str(output_path), 
        pagesize=A4, 
        leftMargin=18*mm, 
        rightMargin=18*mm, 
        topMargin=20*mm, 
        bottomMargin=15*mm, 
        title=f"Calculation Method - {run_id}", 
        author="Admin"
    )
    
    styles = _styles()
    story = []
    
    # Title
    story.append(Paragraph("CALCULATION METHOD DOCUMENTATION", styles["H1"]))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(f"<b>Run ID:</b> {run_id}", styles["Body"]))
    story.append(Paragraph(f"<b>Tenant:</b> {tenant_slug or '—'}", styles["Body"]))
    story.append(Spacer(1, 4*mm))
    
    # Introduction
    story.append(Paragraph("Introduction", styles["H2"]))
    story.append(Paragraph(
        "This document provides a detailed explanation of the pricing calculation method used for this specific project. "
        "For each plan, you will find the exact materials selected, the formulas used, the coefficients applied, "
        "and the complete numerical calculations with actual values.",
        styles["Body"]
    ))
    story.append(Spacer(1, 4*mm))

    # ---------- Basement (Keller): which floors are treated as basement, what is and isn't calculated ----------
    story.append(Paragraph("Basement (Keller) – Floors and calculation scope", styles["H2"]))
    structura_cladirii_calc = frontend_data.get("structuraCladirii", {}) if isinstance(frontend_data, dict) else {}
    tip_fundatie_beci_calc = structura_cladirii_calc.get("tipFundatieBeci", "")
    has_basement_form = bool(frontend_data.get("basement", False)) or (
        tip_fundatie_beci_calc and "Keller" in str(tip_fundatie_beci_calc) and "Kein Keller" not in str(tip_fundatie_beci_calc)
    )
    _tfbc = str(tip_fundatie_beci_calc)
    basement_livable = bool(frontend_data.get("basementUse", False)) or (
        "mit einfachem Ausbau" in _tfbc or "Keller (mit Ausbau)" in _tfbc
    )

    if not has_basement_form:
        story.append(Paragraph(
            "No basement was selected for this project (form: no Keller / only floor slab). "
            "All plans are priced as normal floors (foundation, structure, roof, openings, etc. as applicable).",
            styles["Body"]
        ))
    elif basement_plan_index is None:
        story.append(Paragraph(
            "A basement was selected in the form, but no dedicated basement plan was identified for this run. "
            "If the project has a basement, it may have been included as part of the ground floor cost (basement block added to ground floor).",
            styles["Body"]
        ))
    else:
        basement_plan_id = plan_infos[basement_plan_index].plan_id if basement_plan_index < len(plan_infos) else f"index {basement_plan_index}"
        story.append(Paragraph(
            f"<b>Floor treated as basement (Keller):</b> Plan {basement_plan_index + 1} ({basement_plan_id}) is the floor identified and priced as the basement.",
            styles["Body"]
        ))
        story.append(Paragraph(
            "<b>Calculated for the basement:</b> Interior walls (structure + finish), floors and ceiling, utilities. "
            + ("Utilities include electricity, heating, ventilation, and sewage (livable basement)." if basement_livable else "Utilities include electricity and sewage only (unheated/utility basement)."),
            styles["Body"]
        ))
        story.append(Paragraph(
            "<b>Not calculated for the basement:</b> Foundation (0), exterior walls (0), roof (0), openings (windows/doors) (0), stairs (0), fireplace (0). "
            "The basement is assumed to be below ground; its cost is limited to interior construction and utilities.",
            styles["Body"]
        ))
    story.append(Spacer(1, 4*mm))

    # ---------- Form value to English (all options chosen in the form) ----------
    FORM_VALUE_TO_ENGLISH = {
        "Placă": "Slab", "Piloți": "Piles", "Soclu": "Plinth",
        "Blockbau": "Blockbau", "Holzrahmen": "Timber frame", "Massivholz": "Solid wood",
        "PANOURI": "Panels", "BALLOON": "Balloon", "PLACA": "Slab",
        "Structură": "Structure only", "Structură + ferestre": "Structure + windows", "Casă completă": "Turnkey house",
        "Ușor (camion 40t)": "Easy (40t truck)", "Mediu": "Medium", "Dificil": "Difficult",
        "Plan": "Flat", "Pantă ușoară": "Gentle slope", "Pantă mare": "Steep slope",
        "Kein Keller (nur Bodenplatte)": "No basement (only floor slab)",
        "Keller (ohne Ausbau)": "Basement (without finish)",
        "Keller (unbeheizt / Nutzkeller) (ohne Ausbau)": "Basement (unheated / utility, without finish)",
        "Keller (mit Ausbau)": "Basement (with finish)",
        "Keller (unbeheizt / Nutzkeller)": "Basement (unheated / utility)",
        "Keller (mit einfachem Ausbau)": "Basement (with simple finish)",
        "Standard (2,50 m)": "Standard (2.50 m)", "Komfort (2,70 m)": "Comfort (2.70 m)", "Hoch (2,85+ m)": "High (2.85+ m)",
        "Flachdach": "Flat roof", "Pultdach": "Shed roof", "Gründach": "Green roof", "Satteldach": "Gable roof",
        "Krüppelwalmdach": "Hip and valley roof", "Mansardendach": "Mansard roof",
        "Mansardendach mit Fußwalm": "Mansard roof with foot hip", "Mansardendach mit Schlepp": "Mansard roof with extension",
        "Mansardenwalmdach": "Mansard hip roof", "Walmdach": "Hip roof", "Paralleldach": "Parallel roof",
        "Nein": "No", "Ja – einzelne": "Yes – individual", "Ja – mehrere / große Glasflächen": "Yes – multiple / large glass surfaces",
        "3-fach verglast": "Triple glazed", "3-fach verglast, Passiv": "Triple glazed, Passive",
        "Standard (2m)": "Standard (2 m)", "Erhöht / Sondermaß (2,2+ m)": "Raised / custom size (2.2+ m)",
        "Tencuială": "Plaster", "Lemn": "Wood", "Fibrociment": "Fiber cement", "Mix": "Mix",
        "Țiglă": "Roof tile", "Tablă": "Sheet metal", "Membrană": "Roof membrane",
        "Standard": "Standard", "KfW 55": "KfW 55", "KfW 40": "KfW 40", "KfW 40+": "KfW 40+",
        "Gaz": "Gas", "Pompa de căldură": "Heat pump", "Electric": "Electric",
        "Kein Kamin": "No fireplace", "Klassischer Holzofen": "Classic wood stove",
        "Moderner Design-Kaminofen": "Modern design fireplace", "Pelletofen (automatisch)": "Pellet stove (automatic)",
        "Einbaukamin": "Built-in fireplace", "Kachel-/wassergeführter Kamin": "Tiled/water-bearing fireplace",
        "Pereți Interiori": "Interior Walls", "Pereți Exteriori": "Exterior Walls",
        "Structură Planșeu/Podea": "Floor structure", "Structură Tavan": "Ceiling structure",
        "Lemn-Aluminiu": "Wood-Aluminum",
        "Scară interioară completă (Structură + Finisaj)": "Complete interior stair (structure + finish)",
        "Balustradă scară": "Stair railing",
    }
    def _en(v):
        if v is None or v == "" or v == "—": return "—"
        return FORM_VALUE_TO_ENGLISH.get(str(v).strip(), str(v))
    
    # Extract user selections from frontend_data – structured per form steps, all in English
    sistem_constructiv = frontend_data.get("sistemConstructiv", {})
    structura_cladirii = frontend_data.get("structuraCladirii", {})
    materiale_finisaj = frontend_data.get("materialeFinisaj", {})
    performanta = frontend_data.get("performanta", {})
    performanta_energetica = frontend_data.get("performantaEnergetica", {})
    ferestre_usi = frontend_data.get("ferestreUsi", {})
    incalzire = frontend_data.get("incalzire", {})
    
    tip_sistem = sistem_constructiv.get("tipSistem")
    acces_santier = sistem_constructiv.get("accesSantier")
    tip_fundatie = sistem_constructiv.get("tipFundatie")
    tip_acoperis = sistem_constructiv.get("tipAcoperis")
    nivel_oferta = _normalize_nivel_oferta(frontend_data)
    teren = sistem_constructiv.get("teren")
    utilitati = sistem_constructiv.get("utilitati")
    tip_fundatie_beci = structura_cladirii.get("tipFundatieBeci")
    pilons = structura_cladirii.get("pilons")
    inaltime_etaje = structura_cladirii.get("inaltimeEtaje")
    window_quality = ferestre_usi.get("windowQuality")
    door_mi_pdf = (ferestre_usi.get("doorMaterialInterior") or "Standard").strip()
    door_me_pdf = (ferestre_usi.get("doorMaterialExterior") or "Standard").strip()
    nivel_energetic = performanta.get("nivelEnergetic") or performanta_energetica.get("nivelEnergetic")
    tip_incalzire = performanta.get("tipIncalzire") or performanta_energetica.get("tipIncalzire")
    ventilatie = performanta.get("ventilatie", False) or performanta_energetica.get("ventilatie", False)
    tip_semineu = performanta_energetica.get("tipSemineu") or performanta.get("tipSemineu") or incalzire.get("tipSemineu")
    semineu = tip_semineu and str(tip_semineu) != "Kein Kamin"
    material_acoperis = materiale_finisaj.get("materialAcoperis")
    
    story.append(Paragraph("User Selections for This Project (by form step)", styles["H2"]))
    
    story.append(Paragraph("<b>Step – General project information / Building structure</b>", styles["H3"]))
    story.append(Paragraph(f"Construction system: {_en(tip_sistem)}", styles["Body"]))
    story.append(Paragraph(f"Offer level: {_en(nivel_oferta)}", styles["Body"]))
    story.append(Paragraph(f"Site access: {_en(acces_santier)}", styles["Body"]))
    story.append(Paragraph(f"Terrain: {_en(teren)}", styles["Body"]))
    story.append(Paragraph(f"Electricity/water connection available: {'Yes' if utilitati else 'No'}", styles["Body"]))
    
    story.append(Paragraph("<b>Step – Basement / Foundation</b>", styles["H3"]))
    story.append(Paragraph(f"Basement / foundation: {_en(tip_fundatie_beci)}", styles["Body"]))
    story.append(Paragraph(f"Foundation type (slab/piles/plinth): {_en(tip_fundatie)}", styles["Body"]))
    story.append(Paragraph(f"Pile foundation required: {'Yes' if pilons else 'No'}", styles["Body"]))
    story.append(Paragraph(f"Floor height: {_en(inaltime_etaje)}", styles["Body"]))
    
    story.append(Paragraph("<b>Step – Roof type</b>", styles["H3"]))
    story.append(Paragraph(f"Roof type: {_en(tip_acoperis)}", styles["Body"]))
    
    if inclusions.get("openings"):
        story.append(Paragraph("<b>Step – Windows &amp; Doors</b>", styles["H3"]))
        story.append(Paragraph(f"Window quality: {_en(window_quality)}", styles["Body"]))
        story.append(Paragraph(f"Door types (Innen / Außen): {_en(door_mi_pdf)} / {_en(door_me_pdf)}", styles["Body"]))
    
    if inclusions.get("finishes"):
        story.append(Paragraph("<b>Step – Materials &amp; finish level</b>", styles["H3"]))
        story.append(Paragraph(f"Interior finish (basement): {_en(materiale_finisaj.get('finisajInteriorBeci'))}", styles["Body"]))
        story.append(Paragraph(f"Interior finish inner walls (ground floor): {_en(materiale_finisaj.get('finisajInteriorInnen_ground') or materiale_finisaj.get('finisajInterior_ground'))}", styles["Body"]))
        story.append(Paragraph(f"Interior finish exterior walls (ground floor): {_en(materiale_finisaj.get('finisajInteriorAussen_ground') or materiale_finisaj.get('finisajInterior_ground'))}", styles["Body"]))
        story.append(Paragraph(f"Facade (ground floor): {_en(materiale_finisaj.get('fatada_ground'))}", styles["Body"]))
        story.append(Paragraph(f"Interior finish inner walls (attic): {_en(materiale_finisaj.get('finisajInteriorInnenMansarda') or materiale_finisaj.get('finisajInteriorMansarda'))}", styles["Body"]))
        story.append(Paragraph(f"Interior finish exterior walls (attic): {_en(materiale_finisaj.get('finisajInteriorAussenMansarda') or materiale_finisaj.get('finisajInteriorMansarda'))}", styles["Body"]))
        story.append(Paragraph(f"Facade (attic): {_en(materiale_finisaj.get('fatadaMansarda'))}", styles["Body"]))
    if inclusions.get("roof"):
        story.append(Paragraph("<b>Step – Roof covering (material)</b>", styles["H3"]))
        story.append(Paragraph(f"Roof material: {_en(material_acoperis)}", styles["Body"]))
    
    if inclusions.get("utilities"):
        story.append(Paragraph("<b>Step – Energy efficiency &amp; Heating</b>", styles["H3"]))
        story.append(Paragraph(f"Energy level: {_en(nivel_energetic)}", styles["Body"]))
        story.append(Paragraph(f"Heating type: {_en(tip_incalzire)}", styles["Body"]))
        story.append(Paragraph(f"Ventilation / heat recovery: {'Yes' if ventilatie else 'No'}", styles["Body"]))
        story.append(Paragraph(f"Fireplace: {_en(tip_semineu) if tip_semineu else 'No'}", styles["Body"]))
    
    story.append(Spacer(1, 6*mm))
    
    # Iterate through each plan and show specific calculations
    for plan_idx, entry in enumerate(plans_data, 1):
        plan = entry["info"]
        pricing = entry["pricing"]
        breakdown = pricing.get("breakdown", {})
        
        # Plan header
        story.append(PageBreak() if plan_idx > 1 else Spacer(1, 0))
        story.append(Paragraph(f"PLAN {plan_idx}: {plan.plan_id}", styles["H1"]))
        is_basement_plan = (basement_plan_index is not None and (plan_idx - 1) == basement_plan_index)
        if is_basement_plan:
            story.append(Paragraph(
                "<i>This floor is treated as the basement (Keller). Only interior walls, finishes, floors, and utilities are calculated; no foundation, exterior walls, roof, or openings.</i>",
                styles["Small"]
            ))
        story.append(Spacer(1, 3*mm))

        # 1. FOUNDATION CALCULATION FOR THIS PLAN – always show (area + rooms + cost when applicable)
        foundation = breakdown.get("foundation", {})
        story.append(Paragraph("1. Foundation Calculation", styles["H2"]))
        story.append(Paragraph(
            "<b>Formula:</b> Cost = Foundation Area (m²) × Unit Price per m². "
            "Foundation area = total house surface (sum of room areas or gross area for this plan).",
            styles["Body"]
        ))
        scale_dir = plan.stage_work_dir.parent.parent / "scale" / plan.plan_id
        measurements_plan_path = plan.stage_work_dir / "measurements_plan.json"
        room_scales_path = scale_dir / "cubicasa_steps" / "raster_processing" / "walls_from_coords" / "room_scales.json"
        area_json_path_foundation = plan.stage_work_dir.parent.parent / "area" / plan.plan_id / "areas_calculated.json"
        room_areas_list = []
        total_from_rooms = 0.0
        if room_scales_path.exists():
            try:
                with open(room_scales_path, "r", encoding="utf-8") as f:
                    room_data = json.load(f)
                rooms_dict = room_data.get("room_scales") or room_data.get("rooms") or {}
                if isinstance(rooms_dict, dict):
                    for room_id, room_info in rooms_dict.items():
                        if isinstance(room_info, dict):
                            a = float(room_info.get("area_m2", 0))
                        else:
                            a = float(room_info) if room_info else 0
                        if a > 0:
                            room_areas_list.append((room_id, a))
                            total_from_rooms += a
            except Exception:
                pass
        foundation_area = 0.0
        surface_area_source = ""
        surface_area_px = None
        surface_area_mpp = None
        if measurements_plan_path.exists():
            try:
                with open(measurements_plan_path, "r", encoding="utf-8") as f:
                    mp = json.load(f)
                areas_mp = (mp.get("areas") or {}) if isinstance(mp, dict) else {}
                surfaces_mp = (areas_mp.get("surfaces") or {}) if isinstance(areas_mp, dict) else {}
                foundation_area = float(surfaces_mp.get("foundation_m2") or 0.0) or foundation_area
                surface_area_source = str(areas_mp.get("surface_area_source") or "")
                surface_area_px = areas_mp.get("surface_area_from_09_interior_px")
                surface_area_mpp = areas_mp.get("surface_area_from_09_interior_mpp")
            except Exception:
                pass
        items = foundation.get("detailed_items", [])
        for item in items:
            foundation_area = item.get("area_m2", 0) or foundation_area
            break
        if not foundation_area and total_from_rooms > 0:
            foundation_area = total_from_rooms
        if foundation_area <= 0 and area_json_path_foundation.exists():
            try:
                with open(area_json_path_foundation, "r", encoding="utf-8") as f:
                    area_f = json.load(f)
                surfaces = area_f.get("surfaces", {})
                foundation_area = float(surfaces.get("foundation_m2") or 0.0)
                if foundation_area <= 0:
                    foundation_area = float(area_f.get("input_gross_area_m2", 0.0))
            except Exception:
                pass
        story.append(Paragraph("<b>House surface (foundation area for this plan):</b>", styles["Body"]))
        if room_areas_list:
            story.append(Paragraph("Per-room OCR areas (informative only):", styles["Body"]))
            for room_id, area_m2 in room_areas_list:
                story.append(Paragraph(f"  • Room «{room_id}»: {area_m2:.2f} m²", styles["Small"]))
            story.append(Paragraph(f"  Sum of OCR rooms: {total_from_rooms:.2f} m²", styles["Small"]))
            story.append(Paragraph(f"<b>Total house/foundation area used: {foundation_area:.2f} m²</b>", styles["Body"]))
        else:
            story.append(Paragraph(f"Total foundation area: <b>{foundation_area:.2f} m²</b>", styles["Body"]))
        if surface_area_source == "09_interior_mask" and surface_area_px is not None and surface_area_mpp:
            story.append(Paragraph(
                f"Surface source: 09_interior mask, computed as {int(surface_area_px)} px × ({float(surface_area_mpp):.9f} m/px)^2.",
                styles["Small"]
            ))
        if foundation and foundation.get("total_cost", 0) > 0 and items:
            for item in items:
                area = item.get("area_m2", 0) or foundation_area
                unit_price = item.get("unit_price", 0)
                cost = item.get("cost", 0)
                name = item.get("name", "")
                foundation_type_display = "Slab"
                if "(" in name and ")" in name:
                    foundation_type_display = _en(name.split("(")[1].split(")")[0].strip())
                story.append(Paragraph(
                    f"<b>Foundation type selected:</b> {foundation_type_display}",
                    styles["Body"]
                ))
                story.append(Paragraph(
                    f"<b>Calculation:</b> {area:.2f} m² × {unit_price:.2f} EUR/m² = <b>{cost:.2f} EUR</b>",
                    styles["Body"]
                ))
        else:
            story.append(Paragraph(
                "<i>No foundation cost applied for this plan (e.g. upper floor; foundation only on ground floor).</i>",
                styles["Body"]
            ))
        story.append(Spacer(1, 4*mm))
    
        # 2. STRUCTURAL WALLS – always show full calculation data from areas_calculated.json
        walls = breakdown.get("structure_walls", {})
        story.append(Paragraph("2. Structural Walls Calculation", styles["H2"]))
        story.append(Paragraph(
            "<b>Formula:</b> Cost = Interior Wall Area (m²) × Interior Unit Price × Prefabrication Modifier + "
            "Exterior Wall Area (m²) × Exterior Unit Price × Prefabrication Modifier",
            styles["Body"]
        ))
        
        area_data = None
        walls_measurements_raw = None
        cubicasa_data = None
        try:
            scale_dir = plan.stage_work_dir.parent.parent / "scale" / plan.plan_id
            area_json_path = plan.stage_work_dir.parent.parent / "area" / plan.plan_id / "areas_calculated.json"
            raster_walls_path = scale_dir / "cubicasa_steps" / "raster_processing" / "walls_from_coords" / "walls_measurements.json"
            cubicasa_json_path_scale = scale_dir / "cubicasa_result.json"
            cubicasa_json_path = cubicasa_json_path_scale
            
            if area_json_path.exists():
                with open(area_json_path, "r", encoding="utf-8") as f:
                    area_data = json.load(f)
            if raster_walls_path.exists():
                try:
                    with open(raster_walls_path, "r", encoding="utf-8") as f:
                        walls_measurements_raw = json.load(f)
                except Exception:
                    pass
            wall_lengths_path = scale_dir / "cubicasa_steps" / "raster_processing" / "walls_from_coords" / "wall_lengths.json"
            wall_lengths_raw = None
            if wall_lengths_path.exists():
                try:
                    with open(wall_lengths_path, "r", encoding="utf-8") as f:
                        wall_lengths_raw = json.load(f)
                except Exception:
                    pass
            if cubicasa_json_path_scale.exists():
                try:
                    with open(cubicasa_json_path_scale, "r", encoding="utf-8") as f:
                        cubicasa_data = json.load(f)
                except Exception:
                    pass
            
            ext_length = 0.0
            int_length_finish = 0.0
            int_length_structure = 0.0
            wall_height = 2.7
            if area_data:
                walls_data = area_data.get("walls", {})
                interior_data = area_data.get("walls", {}).get("interior", {})
                exterior_data = area_data.get("walls", {}).get("exterior", {})
                ext_length = float(exterior_data.get("length_m", 0.0))
                int_length_finish = float(interior_data.get("length_m", 0.0))
                int_length_structure = float(interior_data.get("length_m_structure", 0.0))
                if area_data.get("wall_height_m") is not None:
                    wall_height = float(area_data["wall_height_m"])
                elif interior_data.get("length_m") and interior_data.get("gross_area_m2"):
                    L = float(interior_data.get("length_m", 1))
                    wall_height = float(interior_data.get("gross_area_m2", 0)) / max(L, 0.001) if L else 2.7
            if (ext_length <= 0 and int_length_finish <= 0) and walls_measurements_raw:
                avg = walls_measurements_raw.get("estimations", {}).get("average_result", {})
                ext_length = float(avg.get("exterior_meters", 0.0))
                int_length_finish = float(avg.get("interior_meters", 0.0))
                int_length_structure = float(avg.get("interior_meters_structure", 0.0))
            if (ext_length <= 0 and int_length_finish <= 0) and cubicasa_data:
                metrics = (cubicasa_data.get("measurements") or {}).get("metrics", {})
                ext_length = float(metrics.get("walls_ext_m", 0.0))
                int_length_finish = float(metrics.get("walls_int_m", 0.0))
                int_length_structure = float(metrics.get("walls_skeleton_structure_int_m", metrics.get("walls_int_m", 0.0)))
            
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph("<b>Wall lengths (interior and exterior):</b>", styles["H3"]))
            story.append(Paragraph(
                f"  • Exterior walls: <b>{ext_length:.2f} m</b>  |  "
                f"Interior (finishes / outline): <b>{int_length_finish:.2f} m</b>  |  "
                f"Interior (structure / skeleton): <b>{int_length_structure:.2f} m</b>",
                styles["Body"]
            ))
            story.append(Spacer(1, 2*mm))
            
            if area_data:
                story.append(Spacer(1, 2*mm))
                story.append(Paragraph("<b>Detailed wall data (from area calculation):</b>", styles["H3"]))
                walls_data = area_data.get("walls", {})
                interior_data = walls_data.get("interior", {})
                exterior_data = walls_data.get("exterior", {})
                wall_height = area_data.get("wall_height_m")
                if wall_height is None and interior_data.get("length_m"):
                    wall_height = interior_data.get("gross_area_m2", 0) / max(interior_data.get("length_m", 1), 0.001) if interior_data.get("length_m") else 2.7
                if wall_height is None:
                    wall_height = 2.7
                wall_height = float(wall_height)
                ext_length = float(exterior_data.get("length_m", 0.0))
                ext_gross = float(exterior_data.get("gross_area_m2", 0.0))
                ext_openings = float(exterior_data.get("openings_area_m2", 0.0))
                ext_net = float(exterior_data.get("net_area_m2", 0.0))
                int_length_finish = float(interior_data.get("length_m", 0.0))
                int_gross_finish = float(interior_data.get("gross_area_m2", 0.0))
                int_openings = float(interior_data.get("openings_area_m2", 0.0))
                int_net_finish = float(interior_data.get("net_area_m2", 0.0))
                int_length_structure = float(interior_data.get("length_m_structure", 0.0))
                int_gross_structure = float(interior_data.get("gross_area_m2_structure", 0.0))
                int_net_structure = float(interior_data.get("net_area_m2_structure", 0.0))
                story.append(Paragraph(f"<b>Wall height used:</b> {wall_height:.2f} m (from floor height setting).", styles["Body"]))
                story.append(Spacer(1, 2*mm))
                story.append(Paragraph("<b>Exterior walls (structure and finishes use same length):</b>", styles["Body"]))
                story.append(Paragraph(f"  • Length (exterior): <b>{ext_length:.2f} m</b>", styles["Body"]))
                story.append(Paragraph(
                    f"  • Surface: Length × Height = {ext_length:.2f} m × {wall_height:.2f} m = <b>Gross area: {ext_gross:.2f} m²</b>",
                    styles["Body"]
                ))
                story.append(Paragraph(
                    f"  • Openings deducted (windows + exterior doors, from {_sec_openings}): <b>{ext_openings:.2f} m²</b>",
                    styles["Body"]
                ))
                story.append(Paragraph(
                    f"  • Net area (exterior): Gross − Openings = {ext_gross:.2f} − {ext_openings:.2f} = <b>{ext_net:.2f} m²</b>",
                    styles["Body"]
                ))
                story.append(Paragraph(
                    _note_ext_openings,
                    styles["Small"]
                ))
                story.append(Spacer(1, 2*mm))
                story.append(Paragraph("<b>Interior walls – finishes (green outline):</b>", styles["Body"]))
                story.append(Paragraph(f"  • Length (interior, for finishes): <b>{int_length_finish:.2f} m</b>", styles["Body"]))
                story.append(Paragraph(
                    f"  • Surface: Length × Height = {int_length_finish:.2f} m × {wall_height:.2f} m = <b>Gross area: {int_gross_finish:.2f} m²</b>",
                    styles["Body"]
                ))
                story.append(Paragraph(
                    f"  • Openings deducted (interior doors, from {_sec_openings}): <b>{int_openings:.2f} m²</b>",
                    styles["Body"]
                ))
                story.append(Paragraph(
                    f"  • Net area (interior finishes): Gross − Openings = {int_gross_finish:.2f} − {int_openings:.2f} = <b>{int_net_finish:.2f} m²</b>",
                    styles["Body"]
                ))
                story.append(Paragraph(
                    _note_int_openings,
                    styles["Small"]
                ))
                story.append(Spacer(1, 2*mm))
                story.append(Paragraph("<b>Interior walls – structure (skeleton):</b>", styles["Body"]))
                story.append(Paragraph(f"  • Length (interior, for structure): <b>{int_length_structure:.2f} m</b>", styles["Body"]))
                story.append(Paragraph(
                    f"  • Surface: Length × Height = {int_length_structure:.2f} m × {wall_height:.2f} m = <b>Gross area: {int_gross_structure:.2f} m²</b>",
                    styles["Body"]
                ))
                story.append(Paragraph(f"  • Openings deducted (interior doors, from {_sec_openings}): <b>{int_openings:.2f} m²</b>", styles["Body"]))
                story.append(Paragraph(
                    f"  • Net area (interior structure): Gross − Openings = {int_gross_structure:.2f} − {int_openings:.2f} = <b>{int_net_structure:.2f} m²</b>",
                    styles["Body"]
                ))
                story.append(Spacer(1, 2*mm))
                if walls_measurements_raw:
                    avg = walls_measurements_raw.get("estimations", {}).get("average_result", {})
                    story.append(Paragraph("<b>Raw lengths from walls_measurements.json (raster):</b>", styles["Body"]))
                    story.append(Paragraph(
                        f"  Interior (finishes): {avg.get('interior_meters', 0):.2f} m  |  "
                        f"Exterior: {avg.get('exterior_meters', 0):.2f} m  |  "
                        f"Interior (structure): {avg.get('interior_meters_structure', 0):.2f} m",
                        styles["Small"]
                    ))
                    story.append(Spacer(1, 2*mm))
                if wall_lengths_raw:
                    scale_m_px = wall_lengths_raw.get("scale_m_per_px", 0)
                    story.append(Paragraph("<b>Lengths from wall_lengths.json (pixel × scale):</b>", styles["Body"]))
                    story.append(Paragraph(
                        f"  Scale: {scale_m_px:.9f} m/px  |  "
                        f"Interior finish: {wall_lengths_raw.get('interior_finish_length_m', 0):.2f} m  |  "
                        f"Exterior: {wall_lengths_raw.get('exterior_length_m', 0):.2f} m  |  "
                        f"Interior structure: {wall_lengths_raw.get('interior_structure_length_m', 0):.2f} m  |  "
                        f"Exterior structure: {wall_lengths_raw.get('exterior_structure_length_m', 0):.2f} m",
                        styles["Small"]
                    ))
                    story.append(Spacer(1, 2*mm))
                from .tables import create_wall_measurements_table_english
                measurements_table = create_wall_measurements_table_english(area_data, inclusions)
                story.append(Paragraph("<b>Summary table – wall measurements:</b>", styles["Body"]))
                story.append(measurements_table)
                story.append(Spacer(1, 2*mm))
            # Optional: CubiCasa pixel/scale detail (cubicasa_data already loaded from scale_dir above)
            if cubicasa_data and cubicasa_data.get("measurements"):
                story.append(Paragraph("<b>Scale and pixel lengths (CubiCasa source):</b>", styles["H3"]))
                measurements = cubicasa_data["measurements"]
                scale_m_px = measurements.get("metrics", {}).get("scale_m_per_px", 0.0)
                pixels_data = measurements.get("pixels", {})
                metrics_data = measurements.get("metrics", {})
                story.append(Paragraph(f"Scale: <b>{scale_m_px:.9f} m/pixel</b>", styles["Body"]))
                px_ext = pixels_data.get("walls_len_ext", 0)
                px_int = pixels_data.get("walls_len_int", 0)
                walls_ext_m = metrics_data.get("walls_ext_m", 0.0)
                walls_int_m = metrics_data.get("walls_int_m", 0.0)
                story.append(Paragraph(
                    f"Exterior: {px_ext:,} px × scale = <b>{walls_ext_m:.2f} m</b>  |  "
                    f"Interior (outline): {px_int:,} px × scale = <b>{walls_int_m:.2f} m</b>",
                    styles["Small"]
                ))
                story.append(Spacer(1, 2*mm))
        except Exception as e:
            print(f"⚠️ [PDF] Could not load detailed measurements for {plan.plan_id}: {e}")
            import traceback
            traceback.print_exc()
        
        if not area_data:
            story.append(Paragraph("<i>No area data (areas_calculated.json) available for this plan.</i>", styles["Body"]))
        
        if walls and walls.get("total_cost", 0) > 0:
            story.append(Paragraph("<b>Cost calculation:</b>", styles["Body"]))
            for item in walls.get("detailed_items", []):
                area = item.get("area_m2", 0)
                unit_price = item.get("unit_price", 0)
                cost = item.get("cost", 0)
                name = item.get("name", "")
                material = _en(item.get("material", "—"))
                construction_mode = _en(item.get("construction_mode", "—"))
                story.append(Paragraph(
                    f"  {material} | {construction_mode}: {area:.2f} m² × {unit_price:.2f} EUR/m² = <b>{cost:.2f} EUR</b>",
                    styles["Body"]
                ))
        story.append(Spacer(1, 4*mm))
    
        if inclusions.get("openings"):
            # 3. OPENINGS (WINDOWS & DOORS) CALCULATION FOR THIS PLAN – always show (formula, settings, list; cost when > 0)
            openings = breakdown.get("openings", {})
            story.append(Paragraph("3. Openings (Windows & Doors) Calculation", styles["H2"]))
            story.append(Paragraph(
                "<b>Formula:</b> Windows: area (m²) × €/m² by Fensterart. "
                "Normal doors: €/Stück by selected Türtyp (Innen/Außen). Garage doors: €/Stück by type if „Garagentor gewünscht“. "
                "Areas for deduction still use width × height from editor when available.",
                styles["Body"]
            ))
            story.append(Paragraph(
                f"<b>Door types:</b> Innen {_en(door_mi_pdf)}, Außen {_en(door_me_pdf)}.",
                styles["Body"]
            ))
            story.append(Paragraph(
                f"<b>Fensterart:</b> {_en(window_quality)} → €/m² glass area.",
                styles["Body"]
            ))
            opening_items = openings.get("detailed_items", openings.get("items", [])) if openings else []
            num_windows = sum(1 for it in opening_items if "window" in str(it.get("type", "")).lower())
            num_doors = sum(1 for it in opening_items if "door" in str(it.get("type", "")).lower())
            # Total areas by category (for explicit deduction in wall sections)
            total_windows_area = sum(float(it.get("area_m2", 0)) for it in opening_items if "window" in str(it.get("type", "")).lower())
            total_exterior_doors_area = sum(float(it.get("area_m2", 0)) for it in opening_items if "door" in str(it.get("type", "")).lower() and str(it.get("status", "")).lower() == "exterior")
            total_interior_doors_area = sum(float(it.get("area_m2", 0)) for it in opening_items if "door" in str(it.get("type", "")).lower() and str(it.get("status", "")).lower() != "exterior")
            total_openings_area = total_windows_area + total_exterior_doors_area + total_interior_doors_area
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph(
                f"<b>All openings for this plan:</b> {num_windows} window(s), {num_doors} door(s). Detailed list:", styles["Body"]
            ))
            if opening_items:
                for item in opening_items:
                    name = item.get("name", "")
                    area = item.get("area_m2", 0)
                    unit_price = item.get("unit_price", 0)
                    cost = item.get("total_cost", 0)
                    w_m = item.get("width_m")
                    h_m = item.get("height_m")
                    if w_m is not None and h_m is not None:
                        dim_str = f"{float(w_m):.2f} × {float(h_m):.2f}"
                    else:
                        dim_str = item.get("dimensions_m", "—")
                    material = _en(item.get("material", "—"))
                    if openings.get("total_cost", 0) > 0 and (unit_price or cost):
                        story.append(Paragraph(
                            f"  • {name}: dimensions {dim_str} m → area {area:.2f} m² × {unit_price:.2f} EUR/m² = <b>{cost:.2f} EUR</b>",
                            styles["Body"]
                        ))
                    else:
                        story.append(Paragraph(
                            f"  • {name}: dimensions {dim_str} m → area <b>{area:.2f} m²</b>",
                            styles["Body"]
                        ))
            else:
                story.append(Paragraph("<i>No openings detected for this plan.</i>", styles["Body"]))
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph("<b>Total area of openings (used for wall surface deductions):</b>", styles["Body"]))
            story.append(Paragraph(
                f"  • Windows: <b>{total_windows_area:.2f} m²</b>",
                styles["Body"]
            ))
            story.append(Paragraph(
                f"  • Exterior doors: <b>{total_exterior_doors_area:.2f} m²</b> (deducted from exterior wall finishes and exterior wall structure)",
                styles["Body"]
            ))
            story.append(Paragraph(
                f"  • Interior doors: <b>{total_interior_doors_area:.2f} m²</b> (deducted from interior wall finishes and interior wall structure)",
                styles["Body"]
            ))
            story.append(Paragraph(
                f"  • Total openings area: <b>{total_openings_area:.2f} m²</b>",
                styles["Body"]
            ))
            story.append(Paragraph(
                "In sections 2 and 4, gross wall surfaces are reduced by these opening areas to obtain net wall area for pricing.",
                styles["Small"]
            ))
            if openings and openings.get("total_cost", 0) > 0:
                story.append(Paragraph(
                    f"<b>Openings total for this plan: {openings.get('total_cost', 0):.2f} EUR</b>",
                    styles["Body"]
                ))
            story.append(Spacer(1, 4*mm))
        
        if inclusions.get("finishes"):
            # 4. FINISHES CALCULATION (wall finishes) – always show section with full data from areas_calculated.json
            finishes = breakdown.get("finishes", {})
            story.append(Paragraph("4. Finishes Calculation (Wall Finishes)", styles["H2"]))
            story.append(Paragraph(
                "<b>Formula:</b> Cost = Net wall area (m²) × Finish unit price per m². "
                "Net area = (Length × Height) − Openings (doors/windows) area.",
                styles["Body"]
            ))
            area_data_fin = None
            try:
                area_json_path = plan.stage_work_dir.parent.parent / "area" / plan.plan_id / "areas_calculated.json"
                if area_json_path.exists():
                    with open(area_json_path, "r", encoding="utf-8") as f:
                        area_data_fin = json.load(f)
            except Exception:
                pass
            if area_data_fin:
                walls_fin = area_data_fin.get("walls", {})
                int_fin = walls_fin.get("interior", {})
                ext_fin = walls_fin.get("exterior", {})
                wall_height_f = area_data_fin.get("wall_height_m")
                if wall_height_f is None and int_fin.get("length_m"):
                    g = float(int_fin.get("gross_area_m2", 0))
                    L = float(int_fin.get("length_m", 1))
                    wall_height_f = (g / L) if L > 0 else 2.7
                if wall_height_f is None:
                    wall_height_f = 2.7
                wall_height_f = float(wall_height_f)
                len_int = float(int_fin.get("length_m", 0.0))
                len_ext = float(ext_fin.get("length_m", 0.0))
                gross_int = float(int_fin.get("gross_area_m2", 0.0))
                gross_ext = float(ext_fin.get("gross_area_m2", 0.0))
                open_int = float(int_fin.get("openings_area_m2", 0.0))
                open_ext = float(ext_fin.get("openings_area_m2", 0.0))
                net_int = float(int_fin.get("net_area_m2", 0.0))
                net_ext = float(ext_fin.get("net_area_m2", 0.0))
                story.append(Paragraph("<b>Wall height used for finishes:</b>", styles["Body"]))
                story.append(Paragraph(f"  {wall_height_f:.2f} m (from floor height setting).", styles["Body"]))
                story.append(Spacer(1, 2*mm))
                story.append(Paragraph("<b>Interior wall finishes:</b>", styles["Body"]))
                story.append(Paragraph(
                    f"  • Length (interior walls, outline): <b>{len_int:.2f} m</b>",
                    styles["Body"]
                ))
                story.append(Paragraph(
                    f"  • Surface: Length × Height = {len_int:.2f} m × {wall_height_f:.2f} m = <b>Gross area: {gross_int:.2f} m²</b>",
                    styles["Body"]
                ))
                story.append(Paragraph(
                    f"  • Openings deducted (interior doors only, from section 3): <b>{open_int:.2f} m²</b>",
                    styles["Body"]
                ))
                story.append(Paragraph(
                    f"  • Net area for interior finishes: Gross − Openings = {gross_int:.2f} − {open_int:.2f} = <b>{net_int:.2f} m²</b>",
                    styles["Body"]
                ))
                story.append(Paragraph(
                    "<i>Interior openings = sum of all interior door areas (section 3).</i>",
                    styles["Small"]
                ))
                story.append(Spacer(1, 2*mm))
                story.append(Paragraph("<b>Exterior wall finishes (facade):</b>", styles["Body"]))
                story.append(Paragraph(
                    f"  • Length (exterior walls): <b>{len_ext:.2f} m</b>",
                    styles["Body"]
                ))
                story.append(Paragraph(
                    f"  • Surface: Length × Height = {len_ext:.2f} m × {wall_height_f:.2f} m = <b>Gross area: {gross_ext:.2f} m²</b>",
                    styles["Body"]
                ))
                story.append(Paragraph(
                    f"  • Openings deducted (windows + exterior doors, from section 3): <b>{open_ext:.2f} m²</b>",
                    styles["Body"]
                ))
                story.append(Paragraph(
                    f"  • Net area for exterior finishes: Gross − Openings = {gross_ext:.2f} − {open_ext:.2f} = <b>{net_ext:.2f} m²</b>",
                    styles["Body"]
                ))
                story.append(Paragraph(
                    "<i>Exterior openings = sum of all window areas + exterior door areas (section 3).</i>",
                    styles["Small"]
                ))
                story.append(Spacer(1, 4*mm))
            else:
                story.append(Paragraph("<i>No area data (areas_calculated.json) available for this plan.</i>", styles["Body"]))
                story.append(Spacer(1, 2*mm))
            
            if finishes and finishes.get("total_cost", 0) > 0:
                story.append(Paragraph("<b>Cost calculation (net area × unit price):</b>", styles["Body"]))
                for item in finishes.get("detailed_items", []):
                    area = item.get("area_m2", 0)
                    unit_price = item.get("unit_price", 0)
                    cost = item.get("cost", 0)
                    material = _en(item.get("material", "—"))
                    story.append(Paragraph(
                        f"  • {material}: Net area <b>{area:.2f} m²</b> × {unit_price:.2f} EUR/m² = <b>{cost:.2f} EUR</b>",
                        styles["Body"]
                    ))
            else:
                story.append(Paragraph("<i>Finishes not included in this offer level or cost is zero.</i>", styles["Body"]))
            story.append(Spacer(1, 4*mm))
        
        # 5. FLOORS & CEILINGS CALCULATION FOR THIS PLAN
        floors = breakdown.get("floors_ceilings", {})
        if floors and floors.get("total_cost", 0) > 0:
            story.append(Paragraph("5. Floors & Ceilings Calculation", styles["H2"]))
            story.append(Paragraph(
                "<b>Formula:</b> Cost = Floor Area (m²) × Floor Coefficient + Ceiling Area (m²) × Ceiling Coefficient",
                styles["Body"]
            ))
            
            area_coeffs = pricing_coeffs.get("area", {})
            floor_coeff = area_coeffs.get("floor_coefficient_per_m2", 0)
            ceiling_coeff = area_coeffs.get("ceiling_coefficient_per_m2", 0)
            
            story.append(Paragraph(f"<b>Floor Coefficient Used:</b> {floor_coeff:.2f} EUR/m²", styles["Body"]))
            story.append(Paragraph(f"<b>Ceiling Coefficient Used:</b> {ceiling_coeff:.2f} EUR/m²", styles["Body"]))
            
            items = floors.get("detailed_items", [])
            for item in items:
                name = _en(item.get("name", "")) or item.get("name", "")
                area = item.get("area_m2", 0)
                unit_price = item.get("unit_price", 0)
                cost = item.get("cost", 0)
                story.append(Paragraph(
                    f"<b>Calculation:</b> {name}: {area:.2f} m² × {unit_price:.2f} EUR/m² = <b>{cost:.2f} EUR</b>",
                    styles["Body"]
                ))
            story.append(Spacer(1, 4*mm))
    
        # 6. UTILITIES CALCULATION FOR THIS PLAN (only for ground floor)
        utilities = breakdown.get("utilities", {})
        if utilities and utilities.get("total_cost", 0) > 0:
            story.append(Paragraph("6. Utilities & Installations Calculation", styles["H2"]))
            story.append(Paragraph(
                "<b>Formula:</b> Cost = Total Floor Area (m²) × Base Coefficient × Energy Modifier × Type Modifier",
                styles["Body"]
            ))
            
            # Show user selections
            if nivel_energetic and nivel_energetic != "—":
                story.append(Paragraph(
                    f"<b>Energy Level Selected:</b> {nivel_energetic}",
                    styles["Body"]
                ))
            if tip_incalzire and tip_incalzire != "—":
                story.append(Paragraph(
                    f"<b>Heating Type Selected:</b> {tip_incalzire}",
                    styles["Body"]
                ))
            story.append(Paragraph(
                f"<b>Ventilation Selected:</b> {'Yes' if ventilatie else 'No'}",
                styles["Body"]
            ))
            
            items = utilities.get("detailed_items", [])
            for item in items:
                name = item.get("name", "")
                area = item.get("area_m2", 0)
                base_price = item.get("base_price_per_m2", 0)
                final_price = item.get("final_price_per_m2", 0)
                cost = item.get("total_cost", 0)
                energy_mod = item.get("energy_modifier", 1.0)
                type_mod = item.get("type_modifier", 1.0)
                
                story.append(Paragraph(
                    f"<b>Calculation:</b> {name}",
                    styles["Body"]
                ))
                story.append(Paragraph(
                    f"  Base: {base_price:.2f} EUR/m² × Energy Modifier: {energy_mod:.2f}x × Type Modifier: {type_mod:.2f}x = {final_price:.2f} EUR/m²",
                    styles["Small"]
                ))
                story.append(Paragraph(
                    f"  {area:.2f} m² × {final_price:.2f} EUR/m² = <b>{cost:.2f} EUR</b>",
                    styles["Body"]
                ))
            
            story.append(Spacer(1, 4*mm))
    
        # 7. FIREPLACE & CHIMNEY CALCULATION FOR THIS PLAN (only for ground floor)
        fireplace = breakdown.get("fireplace", {})
        if fireplace and fireplace.get("total_cost", 0) > 0:
            story.append(Paragraph("7. Fireplace & Chimney Calculation", styles["H2"]))
            story.append(Paragraph(
                "<b>Formula:</b> Fireplace cost = price from Preisdatenbank for selected type (Kamin/Ofen). "
                "Chimney (Kaminabzug/Horn) = horn_price_per_floor × number of floors (no fixed base).",
                styles["Body"]
            ))
            
            story.append(Paragraph(
                f"<b>Fireplace Selected:</b> {'Yes' if semineu else 'No'}",
                styles["Body"]
            ))
            
            items = fireplace.get("detailed_items", [])
            for item in items:
                name = item.get("name", "")
                unit_price = item.get("unit_price", 0)
                quantity = item.get("quantity", 1)
                cost = item.get("total_cost", 0)
                
                story.append(Paragraph(
                    f"<b>Calculation:</b> {name}: {quantity} × {unit_price:.2f} EUR = <b>{cost:.2f} EUR</b>",
                    styles["Body"]
                ))
            
            story.append(Spacer(1, 4*mm))
    
        # 8. STAIRS CALCULATION FOR THIS PLAN
        stairs = breakdown.get("stairs", {})
        if stairs and stairs.get("total_cost", 0) > 0:
            story.append(Paragraph("8. Stairs Calculation", styles["H2"]))
            story.append(Paragraph(
                "<b>Formula:</b> Cost = Number of Stair Units × Price per Unit + Railing Cost",
                styles["Body"]
            ))
            
            stairs_coeffs = pricing_coeffs.get("stairs", {})
            stair_unit_price = stairs_coeffs.get("price_per_stair_unit", 0)
            railing_price = stairs_coeffs.get("railing_price_per_stair", 0)
            
            story.append(Paragraph(f"<b>Stair unit price used:</b> {stair_unit_price:.2f} EUR/unit", styles["Body"]))
            story.append(Paragraph(f"<b>Railing price used:</b> {railing_price:.2f} EUR/stair", styles["Body"]))
            items = stairs.get("detailed_items", [])
            for item in items:
                name = _en(item.get("name", "")) or item.get("name", "")
                if not name or name == "—":
                    name = "Complete interior stair (structure + finish)" if "Scară" in str(item.get("name", "")) else "Stair railing"
                quantity = item.get("quantity", 0)
                unit_price = item.get("unit_price", 0)
                cost = item.get("cost", 0)
                story.append(Paragraph(
                    f"<b>Calculation:</b> {name}: {quantity} × {unit_price:.2f} EUR = <b>{cost:.2f} EUR</b>",
                    styles["Body"]
                ))
            
            story.append(Spacer(1, 4*mm))
    
        # 9. ROOF CALCULATION FOR THIS PLAN
        roof = breakdown.get("roof", {})
        if roof and roof.get("total_cost", 0) > 0:
            story.append(Paragraph("9. Roof Calculation", styles["H2"]))
            story.append(Paragraph(
                "<b>Formula:</b> Cost = Roof Area (m²) × Material Unit Price per m²",
                styles["Body"]
            ))
            
            story.append(Paragraph(
                f"<b>Roof type selected:</b> {_en(tip_acoperis)}",
                styles["Body"]
            ))
            story.append(Paragraph(
                f"<b>Roof material selected:</b> {_en(material_acoperis)}",
                styles["Body"]
            ))
            
            roof_items = roof.get("detailed_items", [])
            for item in roof_items:
                name = item.get("name", "")
                cost = item.get("cost", 0)
                area_m2 = item.get("area_m2", 0)
                unit_price = item.get("unit_price", 0)
                quantity = item.get("quantity", 0)
                unit = item.get("unit", "m²")
                # Roof module now provides area_m2 and unit_price; for ml items use quantity (length) × unit_price
                if unit == "ml" and quantity and quantity > 0:
                    story.append(Paragraph(
                        f"<b>Calculation:</b> {name}: {quantity:.2f} m × {unit_price:.2f} EUR/m = <b>{cost:.2f} EUR</b>",
                        styles["Body"]
                    ))
                else:
                    if not area_m2 and quantity and quantity > 0:
                        area_m2 = quantity
                    if not unit_price and quantity and quantity > 0:
                        unit_price = cost / quantity
                    story.append(Paragraph(
                        f"<b>Calculation:</b> {name}: {area_m2:.2f} m² × {unit_price:.2f} EUR/m² = <b>{cost:.2f} EUR</b>",
                        styles["Body"]
                    ))
            story.append(Spacer(1, 4*mm))

        # 9.2. AUFSTOCKUNG BESTAND (editor selections with applied prices)
        phase1 = breakdown.get("aufstockung_phase1", {})
        phase_items = phase1.get("detailed_items", []) if isinstance(phase1, dict) else []
        if phase_items:
            fk_92 = str(entry.get("floor_kind") or "existing").strip().lower()
            if fk_92 == "zubau":
                title_92 = "9.2. Zubau (Editor-Auswahl)"
            else:
                title_92 = "9.2. Aufstockung (Editor-Auswahl)"
            story.append(Paragraph(title_92, styles["H2"]))
            for item in phase_items:
                name = item.get("name", "Position")
                total_cost = float(item.get("total_cost") or item.get("cost") or 0.0)
                cat = str(item.get("category") or "")
                if cat == "aufstockung_stair_opening":
                    qty = float(item.get("quantity") or 1.0)
                    unit_price = float(item.get("unit_price") or 0.0)
                    oa = item.get("opening_area_m2")
                    ow, ol = item.get("opening_width_m"), item.get("opening_length_m")
                    dim_txt = ""
                    if ow is not None and ol is not None and float(ow) > 0 and float(ol) > 0:
                        le = max(float(ow), float(ol))
                        br = min(float(ow), float(ol))
                        dim_txt = f" Öffnung ca. {le:.2f} × {br:.2f} m"
                    if oa is not None and float(oa) > 0:
                        dim_txt += (", " if dim_txt else " ") + f"{float(oa):.2f} m² Fläche"
                    story.append(Paragraph(
                        f"<b>Calculation:</b> {name}:{dim_txt} — {qty:.0f} Stk. × {unit_price:.2f} {calc_cur_label}/Stk. = <b>{total_cost:.2f} {calc_cur_label}</b>",
                        styles["Body"],
                    ))
                elif "area_m2" in item:
                    qty = float(item.get("area_m2") or 0.0)
                    unit_price = float(item.get("unit_price") or 0.0)
                    story.append(Paragraph(
                        f"<b>Calculation:</b> {name}: {qty:.2f} m² × {unit_price:.2f} {calc_cur_label}/m² = <b>{total_cost:.2f} {calc_cur_label}</b>",
                        styles["Body"],
                    ))
                else:
                    qty = float(item.get("quantity") or 1.0)
                    unit_price = float(item.get("unit_price") or 0.0)
                    story.append(Paragraph(
                        f"<b>Calculation:</b> {name}: {qty:.2f} × {unit_price:.2f} {calc_cur_label} = <b>{total_cost:.2f} {calc_cur_label}</b>",
                        styles["Body"],
                    ))
            story.append(Paragraph(
                f"<b>Summe (dieses Geschoss):</b> {float(phase1.get('total_cost') or 0.0):.2f} {calc_cur_label}",
                styles["Body"],
            ))
            story.append(Spacer(1, 4*mm))
        
        # 9.5. BASEMENT CALCULATION FOR THIS PLAN (if exists)
        basement = breakdown.get("basement", {})
        if basement and basement.get("total_cost", 0) > 0:
            story.append(Paragraph("9.5. Basement Calculation", styles["H2"]))
            story.append(Paragraph(
                "<b>Formula:</b> Basement Cost = Interior Walls + Floors + Finishes + Utilities (if livable)",
                styles["Body"]
            ))
            story.append(Paragraph(
                "Basement is calculated using the same areas as ground floor, but only interior walls (no exterior walls).",
                styles["Small"]
            ))
            
            items = basement.get("detailed_items", [])
            for item in items:
                name = item.get("name", "")
                quantity = item.get("quantity", 0) or item.get("area_m2", 0)
                unit_price = item.get("unit_price", 0)
                cost = item.get("cost", 0) or item.get("total_cost", 0)
                
                story.append(Paragraph(
                    f"<b>Calculation:</b> {name}: {quantity:.2f} × {unit_price:.2f} {calc_cur_label} = <b>{cost:.2f} {calc_cur_label}</b>",
                    styles["Body"]
                ))
            
            story.append(Spacer(1, 4*mm))
        
        # Plan total
        plan_total = pricing.get("total_cost_eur", 0.0)
        story.append(Paragraph(
            f"<b>Total Cost for This Plan:</b> {plan_total:,.2f} {calc_cur_label}",
            styles["H2"]
        ))
        story.append(Spacer(1, 6*mm))
    
    story.append(PageBreak())
    
    # 10. TOTAL COST CALCULATION
    story.append(Paragraph("10. Total Cost Calculation", styles["H2"]))
    story.append(Paragraph(
        "<b>Formula:</b> Gross Total = (Construction Costs + Logistics Margin + Oversight Margin) × (1 + VAT Rate)",
        styles["Body"]
    ))
    story.append(Paragraph(
        "• Construction Costs: Sum of all component costs (foundation, walls, openings, finishes, etc.)",
        styles["Small"]
    ))
    story.append(Paragraph(
        "• Logistics Margin: 10% of construction costs",
        styles["Small"]
    ))
    story.append(Paragraph(
        "• Oversight Margin: 10% of construction costs",
        styles["Small"]
    ))
    vat_pct_display = calc_method_vat * 100.0
    vat_pct_str = f"{vat_pct_display:.2f}".rstrip("0").rstrip(".")
    story.append(Paragraph(
        f"• VAT Rate: {vat_pct_str}% (applied to net total; tenant setting)",
        styles["Small"]
    ))
    
    # Calculate totals from actual data
    total_construction = sum(e["pricing"].get("total_cost_eur", 0.0) for e in plans_data)
    logistics_margin = total_construction * 0.10
    oversight_margin = total_construction * 0.10
    net_total = total_construction + logistics_margin + oversight_margin
    vat = net_total * calc_method_vat
    gross_total = net_total + vat
    
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("<b>Actual Calculation for This Project:</b>", styles["Body"]))
    story.append(Paragraph(
        f"Construction Costs: {total_construction:,.2f} {calc_cur_label}",
        styles["Body"]
    ))
    story.append(Paragraph(
        f"Logistics Margin (10%): {logistics_margin:,.2f} {calc_cur_label}",
        styles["Body"]
    ))
    story.append(Paragraph(
        f"Oversight Margin (10%): {oversight_margin:,.2f} {calc_cur_label}",
        styles["Body"]
    ))
    story.append(Paragraph(
        f"<b>Net Total (excl. VAT):</b> {net_total:,.2f} {calc_cur_label}",
        styles["Body"]
    ))
    story.append(Paragraph(
        f"VAT ({vat_pct_str}%): {vat:,.2f} {calc_cur_label}",
        styles["Body"]
    ))
    story.append(Paragraph(
        f"<b>Gross Total (incl. VAT):</b> {gross_total:,.2f} {calc_cur_label}",
        styles["H2"]
    ))
    
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph("END OF CALCULATION METHOD DOCUMENTATION", styles["Small"]))
    
    doc.build(story)
    
    print(f"✅ [PDF CALC METHOD] Generat: {output_path}")
    return output_path