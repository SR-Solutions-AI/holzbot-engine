# pricing/form_tags.py
"""
Construiește values_by_tag din frontend_data pentru formula de preț.
Tag-urile sunt identificatori stabili: indiferent de numele câmpului sau opțiunile din formular
(per client), formula de calcul folosește tag-ul (ex: system_type, site_access).
"""
from __future__ import annotations

from typing import Any

# Mapare (step_key, field_name) → tag pentru a construi values_by_tag fără schema.
# Permite rulări cu frontend_data vechi care nu trimit _valuesByTag.
# Poate fi extins per tenant sau încărcat din schema la runtime.
BUILTIN_FIELD_TAG_MAP: dict[tuple[str, str], str] = {
    ("sistemConstructiv", "tipSistem"): "system_type",
    ("sistemConstructiv", "nivelOferta"): "offer_scope",
    ("sistemConstructiv", "accesSantier"): "site_access",
    ("sistemConstructiv", "teren"): "terrain",
    ("sistemConstructiv", "utilitati"): "utilities_connection",
    ("structuraCladirii", "tipFundatieBeci"): "foundation_type",
    ("structuraCladirii", "pilons"): "pilings_required",
    ("structuraCladirii", "inaltimeEtaje"): "floor_height",
    ("daemmungDachdeckung", "daemmung"): "roof_insulation",
    ("daemmungDachdeckung", "unterdach"): "under_roof",
    ("daemmungDachdeckung", "dachstuhlTyp"): "roof_structure_type",
    ("daemmungDachdeckung", "sichtdachstuhl"): "visible_roof_structure",
    ("daemmungDachdeckung", "dachdeckung"): "roof_covering",
    ("daemmungDachdeckung", "pantaAcoperis"): "roof_pitch",
    ("ferestreUsi", "bodentiefeFenster"): "floor_level_windows",
    ("ferestreUsi", "windowQuality"): "window_quality",
    ("ferestreUsi", "turhohe"): "door_height",
    ("performantaEnergetica", "nivelEnergetic"): "energy_level",
    ("performantaEnergetica", "tipIncalzire"): "heating_type",
    ("performantaEnergetica", "ventilatie"): "ventilation",
    ("performantaEnergetica", "tipSemineu"): "fireplace_type",
    # Alias-uri pentru payload-uri care folosesc step keys diferite
    ("performanta", "nivelEnergetic"): "energy_level",
    ("performanta", "tipIncalzire"): "heating_type",
    ("performanta", "ventilatie"): "ventilation",
    ("performanta", "tipSemineu"): "fireplace_type",
}


def build_values_by_tag(
    frontend_data: dict[str, Any],
    steps_schema: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Construiește un dicționar { tag: value } din datele formularului.

    - Dacă frontend_data conține "_valuesByTag", îl returnează (frontend/API l-a construit din schema).
    - Altfel, dacă steps_schema e dat, parcurge steps[].fields[].tag și step key/field name pentru value.
    - Altfel, folosește BUILTIN_FIELD_TAG_MAP pe frontend_data (per step key → dict de fields).

    Returns:
        Dict tag → value (ex: {"system_type": "Blockbau", "site_access": "Mittel", ...}).
    """
    if isinstance(frontend_data.get("_valuesByTag"), dict):
        return dict(frontend_data["_valuesByTag"])

    out: dict[str, Any] = {}

    if steps_schema:
        for step in steps_schema:
            step_key = step.get("key")
            if not step_key:
                continue
            step_data = frontend_data.get(step_key)
            if not isinstance(step_data, dict):
                continue
            for field in step.get("fields") or []:
                tag = field.get("tag")
                name = field.get("name")
                if not tag or not name:
                    continue
                if name in step_data:
                    out[tag] = step_data[name]
        return out

    # Fallback: built-in map
    for (step_key, field_name), tag in BUILTIN_FIELD_TAG_MAP.items():
        step_data = frontend_data.get(step_key)
        if isinstance(step_data, dict) and field_name in step_data:
            out[tag] = step_data[field_name]
    return out
