"""
Pure offer-scope logic (nivel ofertă → categories included). No ReportLab / heavy deps.
Used by generator.py and unit-tested without pulling the full PDF stack.
"""

from __future__ import annotations


def draft_step(frontend_data: dict | None, step: str) -> dict:
    """Wizard drafts: nivelOferta sometimes lives under drafts.<step>."""
    if not isinstance(frontend_data, dict):
        return {}
    d = frontend_data.get("drafts")
    if not isinstance(d, dict):
        return {}
    s = d.get(step)
    return s if isinstance(s, dict) else {}


def normalize_nivel_oferta(frontend_data: dict) -> str:
    """Map form / calc_mode values to: Structură | Structură + ferestre | Casă completă."""
    materiale = frontend_data.get("materialeFinisaj") or {}
    sist = frontend_data.get("sistemConstructiv") or {}
    dm = draft_step(frontend_data, "materialeFinisaj")
    ds = draft_step(frontend_data, "sistemConstructiv")
    # sistemConstructiv before materialeFinisaj (Rohbau often only on structure step).
    raw = (
        sist.get("nivelOferta")
        or materiale.get("nivelOferta")
        or ds.get("nivelOferta")
        or dm.get("nivelOferta")
        or ""
    )
    raw = str(raw).strip()
    calc_mode = (frontend_data.get("calc_mode") or "").strip().lower()
    if not raw and calc_mode:
        if calc_mode in ("full_house", "full_house_premium", "casa_completa", "casacompleta"):
            return "Casă completă"
        if calc_mode in ("structure_windows", "structura_ferestre", "structura+ferestre"):
            return "Structură + ferestre"
        if calc_mode in ("structure", "structura"):
            return "Structură"
    if not raw:
        return "Casă completă"
    lower = raw.lower()
    if lower in (
        "casă completă",
        "casa completa",
        "full_house",
        "full_house_premium",
        "casa_completa",
        "casacompleta",
        "schlüsselfertig",
        "schlüsselfertiges haus",
        "turnkey house",
    ):
        return "Casă completă"
    if ("fenster" in lower or "ferestre" in lower) and (
        "tragwerk" in lower or "rohbau" in lower or "structură" in lower or "structura" in lower
    ):
        return "Structură + ferestre"
    if lower in (
        "structură + ferestre",
        "structure_windows",
        "structura+ferestre",
        "structura + ferestre",
        "structure + windows",
        "rohbau + fenster",
    ):
        return "Structură + ferestre"
    if "rohbau" in lower and "fenster" not in lower and "ferestre" not in lower:
        return "Structură"
    if "tragwerk" in lower and "fenster" not in lower and "ferestre" not in lower:
        return "Structură"
    if lower in ("structură", "structura", "structure", "structure only", "rohbau", "rohbau / konstruktion"):
        return "Structură"
    if raw in ("Casă completă", "Structură + ferestre", "Structură"):
        return raw
    return "Casă completă"


def get_offer_inclusions(nivel_oferta: str, frontend_data: dict | None = None) -> dict:
    if frontend_data and bool(frontend_data.get("roof_only_offer")):
        return {
            "foundation": False,
            "structure_walls": False,
            "roof": True,
            "floors_ceilings": False,
            "openings": False,
            "finishes": False,
            "utilities": False,
            "openings_doors": False,
        }
    INCLUSIONS = {
        "Structură": {
            "foundation": True,
            "structure_walls": True,
            "roof": True,
            "floors_ceilings": True,
            "openings": False,
            "finishes": False,
            "utilities": False,
        },
        "Structură + ferestre": {
            "foundation": True,
            "structure_walls": True,
            "roof": True,
            "floors_ceilings": True,
            "openings": True,
            "finishes": False,
            "utilities": False,
        },
        "Casă completă": {
            "foundation": True,
            "structure_walls": True,
            "roof": True,
            "floors_ceilings": True,
            "openings": True,
            "finishes": True,
            "utilities": True,
        },
    }
    base = INCLUSIONS.get(nivel_oferta, INCLUSIONS["Casă completă"]).copy()
    # PDF Maßnahmen / Türen-Block: nur bei Schlüsselfertig; bei „Rohbau + Fenster“ nur Fenster.
    base["openings_doors"] = nivel_oferta == "Casă completă"
    return base


def baukosten_position_label_de(inclusions: dict) -> str:
    """
    Hauptzeile Gesamtkosten: kein Wort „Ausbau“ bei Rohbau / Tragwerk ohne Innenausbau & Haustechnik.
    """
    if inclusions.get("finishes") or inclusions.get("utilities"):
        return "Baukosten (Konstruktion, Ausbau, Technik)"
    return "Baukosten (Konstruktion, Technik)"


def roof_items_for_pdf_table(items: list, inclusions: dict | None) -> list:
    """
    Roof breakdown rows for admin PDF. If openings are out of scope, hide roof_skylights
    rows so Dachfenster do not appear as separate Fenster-like lines.
    """
    display_items = [it for it in items if "extra_walls" not in str(it.get("category", ""))]
    if inclusions is not None and not inclusions.get("openings"):
        display_items = [
            it
            for it in display_items
            if str(it.get("category", "")).lower() != "roof_skylights"
        ]
    return display_items
