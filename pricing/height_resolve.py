"""Resolve wall / door heights (m) from wizard cm fields or legacy selects."""
from __future__ import annotations

import re

from building_dimensions import STANDARD_DOOR_HEIGHT_M, STANDARD_WALL_HEIGHT_M


def _parse_height_from_label(label: str | None) -> float | None:
    if not label:
        return None
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*m", str(label).lower())
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _cm_to_m(raw: object) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip().replace(",", ".")
    if not s:
        return None
    try:
        cm = float(s)
    except (TypeError, ValueError):
        return None
    if cm <= 0:
        return None
    return cm / 100.0


def resolve_room_height_m(
    frontend_data: dict | None,
    floor_height_m_by_option: dict | None = None,
) -> float:
    """Raumhöhe in m: prefer `raumhoeheCm` (flat or structuraCladirii), then legacy select + optional DB map."""
    if not isinstance(frontend_data, dict):
        return float(STANDARD_WALL_HEIGHT_M)
    sc = frontend_data.get("structuraCladirii")
    if not isinstance(sc, dict):
        sc = {}
    for key in ("raumhoeheCm", "roomHeightCm"):
        m = _cm_to_m(frontend_data.get(key) or sc.get(key))
        if m is not None:
            return float(m)
    label = str(frontend_data.get("inaltimeEtaje") or sc.get("inaltimeEtaje") or "").strip()
    fhmap = floor_height_m_by_option if isinstance(floor_height_m_by_option, dict) else None
    if fhmap and label and label in fhmap:
        try:
            return float(fhmap[label])
        except (TypeError, ValueError):
            pass
    if "Komfort" in label or "2,70" in label:
        return 2.70
    if "Hoch" in label or "2,85" in label:
        return 2.85
    if label:
        return 2.50
    return float(STANDARD_WALL_HEIGHT_M)


def resolve_door_height_m(
    frontend_data: dict | None,
    door_height_m_by_option: dict | None = None,
) -> float:
    """Türhöhe in m: prefer `tuerhoeheCm` (flat or ferestreUsi), then legacy option label + map."""
    if not isinstance(frontend_data, dict):
        return float(STANDARD_DOOR_HEIGHT_M)
    fu = frontend_data.get("ferestreUsi")
    if not isinstance(fu, dict):
        fu = {}
    m = _cm_to_m(frontend_data.get("tuerhoeheCm") or fu.get("tuerhoeheCm"))
    if m is not None:
        return float(m)
    label = str(
        fu.get("doorHeightOption")
        or fu.get("turhohe")
        or frontend_data.get("doorHeightOption")
        or "",
    ).strip()
    dhmap = door_height_m_by_option if isinstance(door_height_m_by_option, dict) else None
    if dhmap and label and label in dhmap:
        try:
            return float(dhmap[label])
        except (TypeError, ValueError):
            pass
    parsed = _parse_height_from_label(label)
    if parsed and parsed > 0:
        return float(parsed)
    return float(STANDARD_DOOR_HEIGHT_M)
