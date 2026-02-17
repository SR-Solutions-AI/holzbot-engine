# roof/roof_pricing.py
"""
Generează roof_pricing.json cu prețuri pentru acoperiș, bazate pe roof_metrics.json
și datele din formular (daemmungDachdeckung). Prețuri orientative pentru Germania (EUR).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.settings import OUTPUT_ROOT, JOBS_ROOT, RUNNER_ROOT


# Prețuri orientative Germania 2024 (EUR) – ajustabile per tenant
PRICES = {
    # Dämmung (EUR/m²) – per tip din formular
    "daemmung": {
        "Keine": 0,
        "Zwischensparren": 85,
        "Aufsparren": 120,
        "Kombination": 105,
    },
    # Dachdeckung (EUR/m²)
    "dachdeckung": {
        "Ziegel": 95,
        "Betonstein": 78,
        "Blech": 65,
        "Schindel": 110,
        "Sonstiges": 85,
    },
    # Unterdach / Folie (EUR/m²)
    "unterdach": {
        "Folie": 18,
        "Schalung + Folie": 35,
    },
    # Klempnerarbeiten (EUR/m) – perimetrul acoperișului
    "klempnerarbeiten": 95,
    # Finisaj interior mansardă (EUR/m²)
    "finisaj_interior": 45,
}

DEFAULT_DAEMMUNG = "Zwischensparren"
DEFAULT_DACHDECKUNG = "Ziegel"
DEFAULT_UNTERDACH = "Folie"


def _get_out_root(run_id: str) -> Path | None:
    out = OUTPUT_ROOT / run_id
    if out.exists():
        return out
    out = JOBS_ROOT / run_id / "output"
    if out.exists():
        return out
    out = RUNNER_ROOT / "output" / run_id
    return out if out.exists() else None


def generate_roof_pricing(run_id: str, frontend_data: dict | None = None) -> Path | None:
    """
    Generează roof_pricing.json în roof/roof_3d/entire/mixed/.
    Returnează path-ul fișierului sau None dacă nu s-a putut genera.
    """
    out_root = _get_out_root(run_id)
    if not out_root:
        print(f"⚠️ [ROOF PRICING] Output nu există: {run_id}")
        return None

    metrics_path = out_root / "roof" / "roof_3d" / "entire" / "mixed" / "roof_metrics.json"
    if not metrics_path.exists():
        print(f"⚠️ [ROOF PRICING] roof_metrics.json lipsește: {metrics_path}")
        return None

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    total = metrics.get("total") or {}
    area_m2 = total.get("area_m2")
    contour_m = total.get("contour_m")

    if area_m2 is None or area_m2 <= 0:
        print(f"⚠️ [ROOF PRICING] area_m2 invalid în roof_metrics: {area_m2}")
        return None

    dd = (frontend_data or {}).get("daemmungDachdeckung") or {}
    daemmung = dd.get("daemmung") or DEFAULT_DAEMMUNG
    dachdeckung = dd.get("dachdeckung") or DEFAULT_DACHDECKUNG
    unterdach = dd.get("unterdach") or DEFAULT_UNTERDACH

    price_daemmung = PRICES["daemmung"].get(daemmung, PRICES["daemmung"][DEFAULT_DAEMMUNG])
    price_dachdeckung = PRICES["dachdeckung"].get(dachdeckung, PRICES["dachdeckung"][DEFAULT_DACHDECKUNG])
    price_unterdach = PRICES["unterdach"].get(unterdach, PRICES["unterdach"][DEFAULT_UNTERDACH])
    price_klempner = PRICES["klempnerarbeiten"]
    price_finisaj = PRICES["finisaj_interior"]

    area = float(area_m2)
    contour = float(contour_m) if contour_m is not None else 0.0

    items: list[dict[str, Any]] = []
    total_cost = 0.0

    # 1. Dämmung (suprafață m²)
    if price_daemmung > 0:
        cost = round(area * price_daemmung, 2)
        items.append({
            "category": "roof_insulation",
            "name": "Dämmung",
            "details": f"Material: {daemmung}",
            "quantity": round(area, 2),
            "unit": "m²",
            "unit_price": price_daemmung,
            "cost": cost,
            "material": daemmung,
        })
        total_cost += cost

    # 2. Unterdach / Folie (suprafață m²)
    if price_unterdach > 0:
        cost = round(area * price_unterdach, 2)
        items.append({
            "category": "roof_unterdach",
            "name": "Unterspannbahn / Folie",
            "details": f"Material: {unterdach}",
            "quantity": round(area, 2),
            "unit": "m²",
            "unit_price": price_unterdach,
            "cost": cost,
            "material": unterdach,
        })
        total_cost += cost

    # 3. Dachdeckung (suprafață m²)
    if price_dachdeckung > 0:
        cost = round(area * price_dachdeckung, 2)
        items.append({
            "category": "roof_cover",
            "name": "Dachdeckung (Eindeckung)",
            "details": f"Material: {dachdeckung}",
            "quantity": round(area, 2),
            "unit": "m²",
            "unit_price": price_dachdeckung,
            "cost": cost,
            "material": dachdeckung,
        })
        total_cost += cost

    # 4. Klempnerarbeiten (perimetru m)
    if contour > 0 and price_klempner > 0:
        cost = round(contour * price_klempner, 2)
        items.append({
            "category": "roof_sheet_metal",
            "name": "Klempnerarbeiten",
            "details": "Dachumfang",
            "quantity": round(contour, 2),
            "unit": "m",
            "unit_price": price_klempner,
            "cost": cost,
        })
        total_cost += cost

    # 5. Finisaj interior (suprafață m² – pentru mansardă)
    if price_finisaj > 0:
        cost = round(area * price_finisaj, 2)
        items.append({
            "category": "roof_finish_interior",
            "name": "Innenverkleidung (Finisaj)",
            "details": "Unterdachfläche",
            "quantity": round(area, 2),
            "unit": "m²",
            "unit_price": price_finisaj,
            "cost": cost,
        })
        total_cost += cost

    result = {
        "form_data": {
            "daemmung": daemmung,
            "dachdeckung": dachdeckung,
            "unterdach": unterdach,
            "dachstuhlTyp": dd.get("dachstuhlTyp"),
            "sichtdachstuhl": dd.get("sichtdachstuhl"),
        },
        "metrics": {
            "area_m2": area,
            "contour_m": contour,
        },
        "items": items,
        "detailed_items": items,
        "total_cost": round(total_cost, 2),
    }

    out_dir = metrics_path.parent
    out_path = out_dir / "roof_pricing.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ [ROOF PRICING] Generat: {out_path} (total: {total_cost:,.2f} EUR)")
    return out_path
