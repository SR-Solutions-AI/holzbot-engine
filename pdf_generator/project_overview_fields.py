from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


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

    def _append_floor_values(prefix: str, mapping: list[tuple[str, str, str | None]], values: dict, *, translate: bool = True):
        for key, label, field_tag in mapping:
            value = values.get(key)
            if value is None or str(value).strip() == "":
                continue
            rendered = _resolve_display_value(str(value), field_tag, translate=translate) if translate else str(value)
            overview_items.append(f"<b>{prefix} ({label}):</b> <b>{rendered}</b>")

    sistem_constructiv = frontend_data.get("sistemConstructiv", {}) or {}
    materiale_finisaj = frontend_data.get("materialeFinisaj", {}) or {}
    performanta = frontend_data.get("performanta", {}) or {}
    performanta_energetica = frontend_data.get("performantaEnergetica", {}) or {}

    if inclusions_ov.get("finishes") and finishes_per_floor:
        floor_order = ["Keller", "Erdgeschoss"] + [f"Obergeschoss {i}" for i in range(1, 10)] + ["Mansardă", "Dachgeschoss"]
        for floor_label in floor_order:
            if floor_label not in finishes_per_floor:
                continue
            floor_finishes = finishes_per_floor[floor_label]
            fin_int_inner = floor_finishes.get("interior_inner")
            fin_int_outer = floor_finishes.get("interior_outer")
            fin_ext = floor_finishes.get("exterior")
            if fin_int_inner:
                overview_items.append(f"<b>Innenausbau Innenwände ({floor_label}):</b> <b>{_resolve_display_value(fin_int_inner, 'interior_finish_interior_walls')}</b>")
            if fin_int_outer:
                overview_items.append(f"<b>Innenausbau Außenwände ({floor_label}):</b> <b>{_resolve_display_value(fin_int_outer, 'interior_finish_exterior_walls')}</b>")
            if fin_ext:
                overview_items.append(f"<b>{enforcer.get('Fațadă')} ({floor_label}):</b> <b>{_resolve_display_value(fin_ext, 'exterior_facade')}</b>")

    overview_items.append(f"<b>{enforcer.get('Anzahl der Stockwerke')}:</b> <b>{num_floors}</b>")
    overview_items.append(f"<b>{enforcer.get('Bausystem')}:</b> <b>{tip_sistem_de}</b>")
    _append_overview_value("Untergeschoss / Fundament", sistem_constructiv.get("tipFundatieBeci") or frontend_data.get("structuraCladirii", {}).get("tipFundatieBeci"))
    if acces_santier_de != "—":
        overview_items.append(f"<b>{enforcer.get('Baustellenzufahrt')}:</b> <b>{acces_santier_de}</b>")
    _append_overview_value("Gelände", sistem_constructiv.get("teren"))
    if tip_fundatie_de != "—":
        overview_items.append(f"<b>{enforcer.get('Tip fundație')}:</b> <b>{tip_fundatie_de}</b>")
    _append_overview_value("Raumhöhe", frontend_data.get("structuraCladirii", {}).get("inaltimeEtaje"), field_tag="floor_height")
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
        if nivel_energetic_de != "—":
            overview_items.append(f"<b>{enforcer.get('Nivel energetic')}:</b> <b>{_resolve_display_value(nivel_energetic_de, 'energy_level')}</b>")
        _append_overview_value(
            "Lüftung / Wärmerückgewinnung",
            performanta_energetica.get("ventilatie") if "ventilatie" in performanta_energetica else performanta.get("ventilatie"),
            force_bool=True,
            translate=False,
        )
    if inclusions_ov.get("finishes"):
        overview_items.append(f"<b>{enforcer.get('Fertigstellungsgrad')}:</b> <b>{nivel_finisare_de}</b>")

    ferestre_usi = frontend_data.get("ferestreUsi", {}) or {}
    if inclusions_ov.get("openings"):
        _append_overview_value("Fensterart", ferestre_usi.get("windowQuality") or materiale_finisaj.get("tamplarie"), field_tag="window_quality")
        _append_overview_value("Innentüren", ferestre_usi.get("doorMaterialInterior"), field_tag="door_material_interior")
        _append_overview_value("Außentüren", ferestre_usi.get("doorMaterialExterior"), field_tag="door_material_exterior")
        _append_overview_value("Schiebetür", ferestre_usi.get("slidingDoorType"), field_tag="sliding_door_type")
        garage_desired = ferestre_usi.get("garagentorGewuenscht")
        if garage_desired is not None:
            garage_type = (ferestre_usi.get("garageDoorType") or "").strip()
            garage_text = "Ja"
            if not garage_desired:
                garage_text = "Nein"
            elif garage_type:
                garage_text = f"Ja ({_resolve_display_value(garage_type, 'garage_door_type')})"
            overview_items.append(f"<b>Garagentor:</b> <b>{garage_text}</b>")

    extras = frontend_data.get("wintergaertenBalkone", {}) or {}
    if inclusions_ov.get("finishes"):
        _append_overview_value("Wintergarten", extras.get("wintergartenTyp"), field_tag="wintergarten_type")
        _append_overview_value("Balkon", extras.get("balkonTyp"), field_tag="balkon_type")

        wandaufbau = frontend_data.get("wandaufbau", {}) or {}
        _append_floor_values(
            "Wandaufbau Außenwände",
            [
                ("außenwandeBeci", "Keller", "wandaufbau_aussen"),
                ("außenwande_ground", "Erdgeschoss", "wandaufbau_aussen"),
                ("außenwande_floor_1", "Obergeschoss 1", "wandaufbau_aussen"),
                ("außenwande_floor_2", "Obergeschoss 2", "wandaufbau_aussen"),
                ("außenwande_floor_3", "Obergeschoss 3", "wandaufbau_aussen"),
                ("außenwandeMansarda", "Dachgeschoss", "wandaufbau_aussen"),
            ],
            wandaufbau,
        )
        _append_floor_values(
            "Wandaufbau Innenwände",
            [
                ("innenwandeBeci", "Keller", "wandaufbau_innen"),
                ("innenwande_ground", "Erdgeschoss", "wandaufbau_innen"),
                ("innenwande_floor_1", "Obergeschoss 1", "wandaufbau_innen"),
                ("innenwande_floor_2", "Obergeschoss 2", "wandaufbau_innen"),
                ("innenwande_floor_3", "Obergeschoss 3", "wandaufbau_innen"),
                ("innenwandeMansarda", "Dachgeschoss", "wandaufbau_innen"),
            ],
            wandaufbau,
        )

        boden_decke = frontend_data.get("bodenDeckeBelag", {}) or {}
        _append_floor_values(
            "Bodenaufbau",
            [
                ("bodenaufbau_ground", "Erdgeschoss", "bodenaufbau"),
                ("bodenaufbau_floor_1", "Obergeschoss 1", "bodenaufbau"),
                ("bodenaufbau_floor_2", "Obergeschoss 2", "bodenaufbau"),
                ("bodenaufbau_floor_3", "Obergeschoss 3", "bodenaufbau"),
                ("bodenaufbauMansarda", "Dachgeschoss", "bodenaufbau"),
                ("bodenaufbauPod", "Dachboden", "bodenaufbau"),
            ],
            boden_decke,
        )
        _append_floor_values(
            "Deckenaufbau",
            [
                ("deckenaufbauBeci", "Keller", "deckenaufbau"),
                ("deckenaufbau_ground", "Erdgeschoss", "deckenaufbau"),
                ("deckenaufbau_floor_1", "Obergeschoss 1", "deckenaufbau"),
                ("deckenaufbau_floor_2", "Obergeschoss 2", "deckenaufbau"),
                ("deckenaufbau_floor_3", "Obergeschoss 3", "deckenaufbau"),
                ("deckenaufbauMansarda", "Dachgeschoss", "deckenaufbau"),
                ("deckenaufbauPod", "Dachboden", "deckenaufbau"),
            ],
            boden_decke,
        )
        _append_floor_values(
            "Bodenbelag",
            [
                ("bodenbelagBeci", "Keller", "bodenbelag"),
                ("bodenbelag_ground", "Erdgeschoss", "bodenbelag"),
                ("bodenbelag_floor_1", "Obergeschoss 1", "bodenbelag"),
                ("bodenbelag_floor_2", "Obergeschoss 2", "bodenbelag"),
                ("bodenbelag_floor_3", "Obergeschoss 3", "bodenbelag"),
                ("bodenbelagMansarda", "Dachgeschoss", "bodenbelag"),
                ("bodenbelagPod", "Dachboden", "bodenbelag"),
            ],
            boden_decke,
        )

    return overview_items
