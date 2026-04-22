from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


def _overview_floor_label_matches_omit(key: str, omit: frozenset[str]) -> bool:
    if not key or not omit:
        return False
    k = key.strip()
    if k in omit:
        return True
    m = re.match(r"^Obergeschoss\s+(\d+)$", k, re.I)
    if m and f"{m.group(1)}. Obergeschoss" in omit:
        return True
    if k == "Mansardă":
        return bool(omit & {"Mansardă", "Dachgeschoss"})
    if k == "Dachgeschoss":
        return bool(omit & {"Dachgeschoss", "Mansardă"})
    return False


def _de_floor_display_label(label: str) -> str:
    """German standard: «1. Obergeschoss» instead of «Obergeschoss 1» (same physical floor, clearer in PDF)."""
    s = (label or "").strip()
    if not s:
        return s
    m = re.match(r"^Obergeschoss\s+(\d+)$", s, re.I)
    if m:
        return f"{int(m.group(1))}. Obergeschoss"
    return s


def _stockwerk_label_from_form_key_suffix(suffix: str) -> str:
    """Map form field suffix (after außenwande_/bodenaufbau_/…) to a Projektübersicht floor label."""
    s = (suffix or "").strip()
    if not s:
        return s
    sl = s.lower()
    if sl in ("ground", "parter"):
        return "Erdgeschoss"
    m_plan = re.match(r"^plan_(\d+)$", s, re.I)
    if m_plan:
        return f"Neues Geschoss (Plan {int(m_plan.group(1)) + 1})"
    if sl == "beci" or "beci" in sl:
        return "Keller"
    m = re.match(r"^floor_(\d+)$", s, re.I)
    if m:
        return f"Obergeschoss {int(m.group(1))}"
    if sl == "mansarda":
        return "Dachgeschoss"
    if sl == "pod":
        return "Dachboden"
    if sl == "intermediar":
        return "Zwischengeschoss"
    return s.replace("_", " ").strip() or s


def merge_plan0_form_keys_into_ground_display(values: dict | None) -> dict:
    """
    Zubau (primul plan): editorul poate salva *_plan_0; în Projektübersicht rândurile fixe folosesc *_ground
    pentru «Erdgeschoss». Copiem plan_0 → ground doar dacă ground e gol (nu suprascriem wizard EG).
    """
    out = dict(values or {})

    def _empty(v) -> bool:
        return v is None or str(v).strip() == ""

    pairs = (
        ("außenwande_plan_0", "außenwande_ground"),
        ("innenwande_plan_0", "innenwande_ground"),
        ("bodenaufbau_plan_0", "bodenaufbau_ground"),
        ("deckenaufbau_plan_0", "deckenaufbau_ground"),
        ("bodenbelag_plan_0", "bodenbelag_ground"),
        ("finisajInteriorInnen_plan_0", "finisajInteriorInnen_ground"),
        ("finisajInteriorAussen_plan_0", "finisajInteriorAussen_ground"),
        ("finisajInterior_plan_0", "finisajInterior_ground"),
        ("fatada_plan_0", "fatada_ground"),
    )
    for src, dst in pairs:
        if not _empty(out.get(src)) and _empty(out.get(dst)):
            out[dst] = out[src]
            try:
                del out[src]
            except KeyError:
                pass
    return out


def merge_neues_geschoss_plan1_finishes_into_erdgeschoss(finishes_per_floor: dict | None) -> dict:
    """După _finishes_per_floor_from_form: tier «Neues Geschoss (Plan 1)» completează «Erdgeschoss» unde lipsește."""
    out: dict = {}
    for k, v in (finishes_per_floor or {}).items():
        out[k] = dict(v) if isinstance(v, dict) else v
    nsp = "Neues Geschoss (Plan 1)"
    eg = "Erdgeschoss"
    if nsp not in out:
        return out
    src = out[nsp]
    if not isinstance(src, dict):
        return out
    if eg not in out:
        out[eg] = {"interior_inner": None, "interior_outer": None, "exterior": None}
    egd = out[eg]
    if not isinstance(egd, dict):
        out[eg] = {"interior_inner": None, "interior_outer": None, "exterior": None}
        egd = out[eg]
    for sk in ("interior_inner", "interior_outer", "exterior"):
        cur = egd.get(sk)
        if (cur is None or str(cur).strip() == "") and src.get(sk) not in (None, ""):
            egd[sk] = src[sk]
    return out


def _append_extra_per_floor_form_keys(
    overview_items: list[str],
    *,
    prefix: str,
    title_de: str,
    field_tag: str,
    values: dict,
    known_keys: set[str],
    omit: frozenset[str],
    _resolve_display_value,
    translate: bool,
) -> None:
    """Emit lines for e.g. außenwande_floor_4 or außenwande_intermediar saved by the detections editor."""
    rx = re.compile(rf"^{re.escape(prefix)}_(.+)$", re.I)
    for key in sorted(values.keys()):
        if not isinstance(key, str):
            continue
        m = rx.match(key)
        if not m:
            continue
        if key in known_keys:
            continue
        raw_val = values.get(key)
        if raw_val is None or str(raw_val).strip() == "":
            continue
        label_raw = _stockwerk_label_from_form_key_suffix(m.group(1))
        if omit and _overview_floor_label_matches_omit(label_raw, omit):
            continue
        rendered = (
            _resolve_display_value(str(raw_val), field_tag, translate=translate)
            if translate
            else str(raw_val)
        )
        label_disp = _de_floor_display_label(label_raw)
        overview_items.append(f"<b>{title_de} ({label_disp}):</b> <b>{rendered}</b>")


def _strip_label_for_option(label: str) -> str:
    if not label or not isinstance(label, str):
        return ""
    stripped = (
        label
        .replace(" (€/m²)", "")
        .replace(" (€)", "")
        .replace(" (€/Stück)", "")
        .replace(" (Faktor)", "")
        .replace(" (€/m)", "")
        .strip()
    )
    # Keep behavior close to frontend for labels with spacing variants.
    stripped = re.sub(r"\s*\(\s*€\/m²\s*\)\s*$", "", stripped, flags=re.I)
    stripped = re.sub(r"\s*\(\s*€\s*\)\s*$", "", stripped, flags=re.I)
    stripped = re.sub(r"\s*\(\s*€\/Stück\s*\)\s*$", "", stripped, flags=re.I)
    stripped = re.sub(r"\s*\(\s*Faktor\s*\)\s*$", "", stripped, flags=re.I)
    stripped = re.sub(r"\s*\(\s*€\/m\s*\)\s*$", "", stripped, flags=re.I)
    return stripped or label


@lru_cache(maxsize=1)
def _base_option_value_to_price_key() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}

    def _collect_price_sections(container: dict) -> None:
        tag = container.get("fieldTag")
        if tag:
            bucket = out.setdefault(str(tag), {})
            for variable in container.get("variables", []) or []:
                if not isinstance(variable, dict):
                    continue
                key = str(variable.get("key") or "").strip()
                label = _strip_label_for_option(str(variable.get("label") or "").strip())
                if key and label:
                    bucket[label] = key

        for section in container.get("priceSections", []) or []:
            if not isinstance(section, dict):
                continue
            section_tag = section.get("fieldTag")
            if not section_tag:
                continue
            bucket = out.setdefault(str(section_tag), {})
            for variable in section.get("variables", []) or []:
                if not isinstance(variable, dict):
                    continue
                key = str(variable.get("key") or "").strip()
                label = _strip_label_for_option(str(variable.get("label") or "").strip())
                if key and label:
                    bucket[label] = key

    try:
        schema_path = Path(__file__).resolve().parents[2] / "holzbot-web" / "data" / "form-schema" / "holzbau-form-steps.json"
        data = json.loads(schema_path.read_text(encoding="utf-8"))
        steps = data.get("steps") if isinstance(data, dict) else data
        if not isinstance(steps, list):
            return out
        for step in steps:
            if not isinstance(step, dict):
                continue
            _collect_price_sections(step)
            for subsection in step.get("subsections", []) or []:
                if not isinstance(subsection, dict):
                    continue
                _collect_price_sections(subsection)

        preisdatenbank_sections = ((data.get("preisdatenbank") or {}).get("sections") or []) if isinstance(data, dict) else []
        for step in preisdatenbank_sections:
            if not isinstance(step, dict):
                continue
            _collect_price_sections(step)
            for subsection in step.get("subsections", []) or []:
                if not isinstance(subsection, dict):
                    continue
                _collect_price_sections(subsection)
    except Exception:
        return {}
    return out


def build_selected_form_overview_items(
    frontend_data: dict,
    enforcer,
    inclusions_ov: dict,
    finishes_per_floor: dict,
    num_floors: int,
    *,
    acces_santier_de: str = "—",
    tip_fundatie_de: str = "—",
    tip_sistem_de: str = "—",
    tip_acoperis_de: str = "—",
    material_acoperis_de: str = "—",
    incalzire_de: str = "—",
    nivel_energetic_de: str = "—",
    nivel_finisare_de: str = "—",
    tip_semineu: str = "",
    aufstockung_omit_floor_labels: frozenset[str] | None = None,
) -> list[str]:
    overview_items: list[str] = []
    pdf_display = frontend_data.get("pdf_display") or {}
    custom_options = pdf_display.get("customOptions") or {}
    param_label_overrides = pdf_display.get("paramLabelOverrides") or {}
    base_option_value_to_price_key = _base_option_value_to_price_key()

    def _resolve_display_value(value, field_tag: str | None = None, *, translate: bool = True) -> str:
        rendered = str(value).strip()
        if not rendered:
            return rendered
        if field_tag:
            price_key = base_option_value_to_price_key.get(field_tag, {}).get(rendered)
            if price_key:
                override = str(param_label_overrides.get(price_key) or "").strip()
                if override:
                    return override
        if field_tag:
            for option in custom_options.get(field_tag, []) or []:
                option_value = str(option.get("value") or "").strip()
                option_label = str(option.get("label") or "").strip()
                price_key = str(option.get("price_key") or "").strip()
                if rendered == option_value or rendered == option_label:
                    override = str(param_label_overrides.get(price_key) or "").strip()
                    return override or option_label or rendered
        return (enforcer.get(rendered) or rendered) if translate else rendered

    def _append_overview_value(label: str, value, *, field_tag: str | None = None, translate: bool = True, force_bool: bool = False):
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
                rendered = _resolve_display_value(rendered, field_tag)
        overview_items.append(f"<b>{label}:</b> <b>{rendered}</b>")

    omit = aufstockung_omit_floor_labels or frozenset()
    finishes_per_floor = merge_neues_geschoss_plan1_finishes_into_erdgeschoss(
        dict(finishes_per_floor) if isinstance(finishes_per_floor, dict) else {}
    )

    def _append_floor_values(prefix: str, mapping: list[tuple[str, str, str | None]], values: dict, *, translate: bool = True):
        for key, label, field_tag in mapping:
            if omit and _overview_floor_label_matches_omit(label, omit):
                continue
            value = values.get(key)
            if value is None or str(value).strip() == "":
                continue
            rendered = _resolve_display_value(str(value), field_tag, translate=translate) if translate else str(value)
            label_disp = _de_floor_display_label(label)
            overview_items.append(f"<b>{prefix} ({label_disp}):</b> <b>{rendered}</b>")

    sistem_constructiv = frontend_data.get("sistemConstructiv", {}) or {}
    materiale_finisaj = merge_plan0_form_keys_into_ground_display(frontend_data.get("materialeFinisaj") or {})
    performanta = frontend_data.get("performanta", {}) or {}
    # Popup values from detections editor must remain visible in final offer even
    # when the package level would normally hide "finishes" sections.
    if finishes_per_floor:
        floor_order = ["Keller", "Erdgeschoss"] + [f"Obergeschoss {i}" for i in range(1, 10)] + ["Mansardă", "Dachgeschoss"]
        seen_fin = set()

        def _emit_finishes_for_floor(floor_label: str) -> None:
            if omit and _overview_floor_label_matches_omit(floor_label, omit):
                return
            if floor_label not in finishes_per_floor:
                return
            floor_finishes = finishes_per_floor[floor_label]
            fin_int_inner = floor_finishes.get("interior_inner")
            fin_int_outer = floor_finishes.get("interior_outer")
            fin_ext = floor_finishes.get("exterior")
            fl_disp = _de_floor_display_label(floor_label)
            if fin_int_inner:
                overview_items.append(f"<b>Innenausbau Innenwände ({fl_disp}):</b> <b>{_resolve_display_value(fin_int_inner, 'interior_finish_interior_walls')}</b>")
            if fin_int_outer:
                overview_items.append(f"<b>Innenausbau Außenwände ({fl_disp}):</b> <b>{_resolve_display_value(fin_int_outer, 'interior_finish_exterior_walls')}</b>")
            if fin_ext:
                overview_items.append(f"<b>{enforcer.get('Fațadă')} ({fl_disp}):</b> <b>{_resolve_display_value(fin_ext, 'exterior_facade')}</b>")

        for floor_label in floor_order:
            seen_fin.add(floor_label)
            _emit_finishes_for_floor(floor_label)
        for floor_label in sorted(finishes_per_floor.keys(), key=lambda s: str(s).lower()):
            if floor_label in seen_fin:
                continue
            _emit_finishes_for_floor(floor_label)

    overview_items.append(f"<b>{enforcer.get('Anzahl der Stockwerke')}:</b> <b>{num_floors}</b>")
    overview_items.append(f"<b>{enforcer.get('Bausystem')}:</b> <b>{tip_sistem_de}</b>")
    _append_overview_value("Untergeschoss / Fundament", sistem_constructiv.get("tipFundatieBeci") or frontend_data.get("structuraCladirii", {}).get("tipFundatieBeci"))
    if acces_santier_de != "—":
        overview_items.append(f"<b>{enforcer.get('Baustellenzufahrt')}:</b> <b>{acces_santier_de}</b>")
    _append_overview_value("Gelände", sistem_constructiv.get("teren"))
    if tip_fundatie_de != "—":
        overview_items.append(f"<b>{enforcer.get('Tip fundație')}:</b> <b>{tip_fundatie_de}</b>")
    sc_ov = frontend_data.get("structuraCladirii") or {}
    rh_raw = frontend_data.get("raumhoeheCm", sc_ov.get("raumhoeheCm"))
    rh_display = None
    if rh_raw not in (None, ""):
        try:
            rh_display = f"{float(str(rh_raw).replace(',', '.')):g} cm"
        except (TypeError, ValueError):
            rh_display = str(rh_raw)
    if rh_display:
        overview_items.append(f"<b>Raumhöhe:</b> <b>{rh_display}</b>")
    else:
        _append_overview_value(
            "Raumhöhe",
            frontend_data.get("inaltimeEtaje") or sc_ov.get("inaltimeEtaje"),
            field_tag="floor_height",
        )
    _append_overview_value("Treppentyp", frontend_data.get("structuraCladirii", {}).get("treppeTyp"), field_tag="stairs_type")
    overview_items.append(f"<b>{enforcer.get('Dachtyp')}:</b> <b>{tip_acoperis_de}</b>")
    if inclusions_ov.get("finishes", False) and material_acoperis_de != "—":
        overview_items.append(f"<b>{enforcer.get('Dachmaterial')}:</b> <b>{material_acoperis_de}</b>")

    dd = frontend_data.get("daemmungDachdeckung") or {}
    if inclusions_ov.get("finishes") or inclusions_ov.get("utilities"):
        for key, label in [
            ("daemmung", "Dämmung"),
            ("dachdeckung", "Dachdeckung"),
            ("unterdach", "Unterdach"),
            ("dachstuhlTyp", "Dachstuhl-Typ"),
        ]:
            value = dd.get(key)
            if value:
                field_tag = {
                    "daemmung": "roof_insulation",
                    "dachdeckung": "roof_covering",
                    "dachstuhlTyp": "roof_structure_type",
                }.get(key)
                overview_items.append(f"<b>{enforcer.get(label)}:</b> <b>{_resolve_display_value(value, field_tag)}</b>")
        _append_overview_value("Sichtdachstuhl", dd.get("sichtdachstuhl"), force_bool=True, translate=False)
        if dd.get("dachfensterImDach") is not None:
            roof_window_value = "Ja"
            roof_window_type = (dd.get("dachfensterTyp") or "").strip()
            if not dd.get("dachfensterImDach"):
                roof_window_value = "Nein"
            elif roof_window_type:
                roof_window_value = f"Ja ({enforcer.get(roof_window_type) or roof_window_type})"
            overview_items.append(f"<b>Dachfenster:</b> <b>{roof_window_value}</b>")

    tip_semineu_str = str(tip_semineu).strip() if tip_semineu else ""
    if inclusions_ov.get("utilities"):
        if tip_semineu_str:
            semineu_de = _resolve_display_value(tip_semineu_str, "fireplace_type")
            overview_items.append(f"<b>{enforcer.get('Kamin')}:</b> <b>{semineu_de}</b>")
            if tip_semineu_str.lower() != "kein kamin":
                overview_items.append(f"<b>{enforcer.get('Kaminabzug')}:</b> <b>für {num_floors} Geschosse</b>")
        if incalzire_de != "—":
            overview_items.append(f"<b>{enforcer.get('Heizsystem')}:</b> <b>{_resolve_display_value(incalzire_de, 'heating_type')}</b>")
    if inclusions_ov.get("finishes"):
        overview_items.append(f"<b>{enforcer.get('Fertigstellungsgrad')}:</b> <b>{nivel_finisare_de}</b>")

    ferestre_usi = frontend_data.get("ferestreUsi", {}) or {}
    if inclusions_ov.get("openings"):
        _append_overview_value("Glasflächen / Fensterart", ferestre_usi.get("windowQuality") or materiale_finisaj.get("tamplarie"), field_tag="window_quality")
        th_raw = frontend_data.get("tuerhoeheCm", ferestre_usi.get("tuerhoeheCm"))
        if th_raw not in (None, ""):
            try:
                overview_items.append(f"<b>Türhöhe:</b> <b>{float(str(th_raw).replace(',', '.')):g} cm</b>")
            except (TypeError, ValueError):
                overview_items.append(f"<b>Türhöhe:</b> <b>{th_raw}</b>")
        _append_overview_value("Innentüren", ferestre_usi.get("doorMaterialInterior"), field_tag="door_material_interior")
        _append_overview_value("Außentüren", ferestre_usi.get("doorMaterialExterior"), field_tag="door_material_exterior")
        garage_desired = ferestre_usi.get("garagentorGewuenscht")
        if garage_desired is not None:
            garage_type = (ferestre_usi.get("garageDoorType") or "").strip()
            garage_text = "Ja"
            if not garage_desired:
                garage_text = "Nein"
            elif garage_type:
                garage_text = f"Ja ({_resolve_display_value(garage_type, 'garage_door_type')})"
            overview_items.append(f"<b>Garagentor:</b> <b>{garage_text}</b>")

    wandaufbau = merge_plan0_form_keys_into_ground_display(frontend_data.get("wandaufbau") or {})
    _wa_aussen_static = [
        ("außenwandeBeci", "Keller", "wandaufbau_aussen"),
        ("außenwande_ground", "Erdgeschoss", "wandaufbau_aussen"),
        ("außenwande_floor_1", "Obergeschoss 1", "wandaufbau_aussen"),
        ("außenwande_floor_2", "Obergeschoss 2", "wandaufbau_aussen"),
        ("außenwande_floor_3", "Obergeschoss 3", "wandaufbau_aussen"),
        ("außenwandeMansarda", "Dachgeschoss", "wandaufbau_aussen"),
    ]
    _wa_innen_static = [
        ("innenwandeBeci", "Keller", "wandaufbau_innen"),
        ("innenwande_ground", "Erdgeschoss", "wandaufbau_innen"),
        ("innenwande_floor_1", "Obergeschoss 1", "wandaufbau_innen"),
        ("innenwande_floor_2", "Obergeschoss 2", "wandaufbau_innen"),
        ("innenwande_floor_3", "Obergeschoss 3", "wandaufbau_innen"),
        ("innenwandeMansarda", "Dachgeschoss", "wandaufbau_innen"),
    ]
    _append_floor_values("Wandaufbau Außenwände", _wa_aussen_static, wandaufbau)
    _append_extra_per_floor_form_keys(
        overview_items,
        prefix="außenwande",
        title_de="Wandaufbau Außenwände",
        field_tag="wandaufbau_aussen",
        values=wandaufbau,
        known_keys={t[0] for t in _wa_aussen_static},
        omit=omit,
        _resolve_display_value=_resolve_display_value,
        translate=True,
    )
    _append_floor_values("Wandaufbau Innenwände", _wa_innen_static, wandaufbau)
    _append_extra_per_floor_form_keys(
        overview_items,
        prefix="innenwande",
        title_de="Wandaufbau Innenwände",
        field_tag="wandaufbau_innen",
        values=wandaufbau,
        known_keys={t[0] for t in _wa_innen_static},
        omit=omit,
        _resolve_display_value=_resolve_display_value,
        translate=True,
    )

    boden_decke = merge_plan0_form_keys_into_ground_display(frontend_data.get("bodenDeckeBelag") or {})
    _bd_static = [
        ("bodenaufbau_ground", "Erdgeschoss", "bodenaufbau"),
        ("bodenaufbau_floor_1", "Obergeschoss 1", "bodenaufbau"),
        ("bodenaufbau_floor_2", "Obergeschoss 2", "bodenaufbau"),
        ("bodenaufbau_floor_3", "Obergeschoss 3", "bodenaufbau"),
        ("bodenaufbauMansarda", "Dachgeschoss", "bodenaufbau"),
        ("bodenaufbauPod", "Dachboden", "bodenaufbau"),
    ]
    _dc_static = [
        ("deckenaufbauBeci", "Keller", "deckenaufbau"),
        ("deckenaufbau_ground", "Erdgeschoss", "deckenaufbau"),
        ("deckenaufbau_floor_1", "Obergeschoss 1", "deckenaufbau"),
        ("deckenaufbau_floor_2", "Obergeschoss 2", "deckenaufbau"),
        ("deckenaufbau_floor_3", "Obergeschoss 3", "deckenaufbau"),
        ("deckenaufbauMansarda", "Dachgeschoss", "deckenaufbau"),
        ("deckenaufbauPod", "Dachboden", "deckenaufbau"),
    ]
    _bb_static = [
        ("bodenbelagBeci", "Keller", "bodenbelag"),
        ("bodenbelag_ground", "Erdgeschoss", "bodenbelag"),
        ("bodenbelag_floor_1", "Obergeschoss 1", "bodenbelag"),
        ("bodenbelag_floor_2", "Obergeschoss 2", "bodenbelag"),
        ("bodenbelag_floor_3", "Obergeschoss 3", "bodenbelag"),
        ("bodenbelagMansarda", "Dachgeschoss", "bodenbelag"),
        ("bodenbelagPod", "Dachboden", "bodenbelag"),
    ]
    _append_floor_values("Bodenaufbau", _bd_static, boden_decke)
    _append_extra_per_floor_form_keys(
        overview_items,
        prefix="bodenaufbau",
        title_de="Bodenaufbau",
        field_tag="bodenaufbau",
        values=boden_decke,
        known_keys={t[0] for t in _bd_static},
        omit=omit,
        _resolve_display_value=_resolve_display_value,
        translate=True,
    )
    _append_floor_values("Deckenaufbau", _dc_static, boden_decke)
    _append_extra_per_floor_form_keys(
        overview_items,
        prefix="deckenaufbau",
        title_de="Deckenaufbau",
        field_tag="deckenaufbau",
        values=boden_decke,
        known_keys={t[0] for t in _dc_static},
        omit=omit,
        _resolve_display_value=_resolve_display_value,
        translate=True,
    )
    _append_floor_values("Bodenbelag", _bb_static, boden_decke)
    _append_extra_per_floor_form_keys(
        overview_items,
        prefix="bodenbelag",
        title_de="Bodenbelag",
        field_tag="bodenbelag",
        values=boden_decke,
        known_keys={t[0] for t in _bb_static},
        omit=omit,
        _resolve_display_value=_resolve_display_value,
        translate=True,
    )

    return overview_items
