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
    ("structuraCladirii", "raumhoeheCm"): "room_height_cm",
    ("structuraCladirii", "balkonBoden"): "balkon_boden",
    ("daemmungDachdeckung", "daemmung"): "roof_insulation",
    ("daemmungDachdeckung", "unterdach"): "under_roof",
    ("daemmungDachdeckung", "dachstuhlTyp"): "roof_structure_type",
    ("daemmungDachdeckung", "sichtdachstuhl"): "visible_roof_structure",
    ("daemmungDachdeckung", "dachdeckung"): "roof_covering",
    ("daemmungDachdeckung", "pantaAcoperis"): "roof_pitch",
    ("daemmungDachdeckung", "dachfensterImDach"): "roof_skylights",
    ("daemmungDachdeckung", "dachfensterTyp"): "roof_skylight_type",
    ("ferestreUsi", "windowQuality"): "window_quality",
    ("ferestreUsi", "doorMaterialInterior"): "door_material_interior",
    ("ferestreUsi", "doorMaterialExterior"): "door_material_exterior",
    ("ferestreUsi", "tuerhoeheCm"): "door_height_cm",
    ("ferestreUsi", "garageDoorType"): "garage_door_type",
    ("ferestreUsi", "garagentorGewuenscht"): "garage_door_desired",
    ("ferestreUsi", "treppeTyp"): "stairs_type",
    ("performantaEnergetica", "tipIncalzire"): "heating_type",
    ("performantaEnergetica", "tipSemineu"): "fireplace_type",
    ("performantaEnergetica", "includeElectricity"): "include_electricity",
    ("performantaEnergetica", "includeSewage"): "include_sewage",
    # Alias-uri pentru payload-uri care folosesc step keys diferite
    ("performanta", "tipIncalzire"): "heating_type",
    ("performanta", "tipSemineu"): "fireplace_type",
    ("performanta", "includeElectricity"): "include_electricity",
    ("performanta", "includeSewage"): "include_sewage",
}


def _collect_tag_values_from_steps(
    frontend_data: dict[str, Any],
    steps_schema: list[dict] | None,
) -> dict[str, Any]:
    """Values read from live step blobs (ferestreUsi, sistemConstructiv, …)."""
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
                    val = step_data[name]
                    if val is not None and val != "":
                        out[tag] = val
        return out

    for (step_key, field_name), tag in BUILTIN_FIELD_TAG_MAP.items():
        step_data = frontend_data.get(step_key)
        if isinstance(step_data, dict) and field_name in step_data:
            val = step_data[field_name]
            if val is not None and val != "":
                out[tag] = val
    return out


def build_values_by_tag(
    frontend_data: dict[str, Any],
    steps_schema: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Construiește un dicționar { tag: value } din datele formularului.

    - Valorile din pașii curenți (ferestreUsi.windowQuality etc.) au **prioritate** față de
      `_valuesByTag`, astfel încât un snapshot vechi din job JSON nu poate suprascrie
      modificările salvate în offer_steps (PDF vede pașii; pricing trebuie la fel).
    - `_valuesByTag` completează doar tag-urile lipsă din pași.

    Returns:
        Dict tag → value (ex: {"system_type": "Blockbau", "site_access": "Mittel", ...}).
    """
    from_steps = _collect_tag_values_from_steps(frontend_data, steps_schema)
    snap = frontend_data.get("_valuesByTag")
    if isinstance(snap, dict) and snap:
        return {**snap, **from_steps}
    return from_steps
