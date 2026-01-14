# new/runner/roof/calculator.py
from __future__ import annotations

import json
import math
from pathlib import Path
from datetime import datetime
from typing import Dict

from .config import (
    ROOF_TYPES_FILE,
    ROOF_COEFFICIENTS_FILE,
    DEFAULT_COEFFICIENTS,
    MATERIAL_PRICE_KEY
)
from .mapper import normalize_roof_type, normalize_material


def _load_roof_types() -> list[dict]:
    """Load toate tipurile de acoperiș din JSON."""
    if not ROOF_TYPES_FILE.exists():
        raise FileNotFoundError(f"❌ Lipsă fișier: {ROOF_TYPES_FILE}")
    
    with open(ROOF_TYPES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data.get("dachformen", [])


def _load_coefficients() -> dict:
    """Load coeficienți din JSON cu fallback la default."""
    if ROOF_COEFFICIENTS_FILE.exists():
        try:
            with open(ROOF_COEFFICIENTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Merge cu default pentru siguranță
            return {**DEFAULT_COEFFICIENTS, **data}
        except Exception:
            pass
    
    return DEFAULT_COEFFICIENTS.copy()


def _perimeter_from_area(area_m2: float) -> float:
    """
    Estimare perimetru din arie (casă ~pătrată): P ≈ 4 × √A
    """
    if area_m2 <= 0:
        return 0.0
    return 4.0 * math.sqrt(area_m2)


def calculate_roof_price(
    house_area_m2: float,
    ceiling_area_m2: float,
    perimeter_m: float | None,
    roof_type_user: str,
    material_user: str | None = None,
    frontend_data: dict | None = None,
    total_floors: int = 1
) -> dict:
    """
    Calculează prețul acoperișului cu toate componentele.
    
    Args:
        house_area_m2: Aria acoperișului extins (m²) - din area module (roof_m2)
        ceiling_area_m2: Aria tavanului util (m²) - pentru calcul izolație
        perimeter_m: Perimetrul casei (m) - din perimeter module sau None
        roof_type_user: Tipul ales de utilizator (RO/EN/DE)
        material_user: Materialul ales (RO) - "Țiglă"/"Tablă"/"Membrană"
        frontend_data: Date suplimentare din frontend (opțional)
        total_floors: Numărul total de etaje (pentru context)
    
    Returns:
        Dict cu breakdown complet costuri
    """
    
    # ==========================================
    # LOAD DATE
    # ==========================================
    roof_types = _load_roof_types()
    coeffs = _load_coefficients()
    
    # Override coeficienți cu date din frontend (dacă există)
    if frontend_data:
        coeffs.update(frontend_data)
    
    # ==========================================
    # NORMALIZARE INPUT
    # ==========================================
    roof_key = normalize_roof_type(roof_type_user)
    material_key = normalize_material(material_user)
    
    # ==========================================
    # GĂSEȘTE TIPUL DE ACOPERIȘ
    # ==========================================
    roof = next((
        r for r in roof_types
        if str(r.get("name_de", "")).strip() == roof_key
    ), None)
    
    if not roof:
        raise ValueError(
            f"❌ Tipul de acoperiș '{roof_key}' nu a fost găsit în roof_types_germany.json.\n"
            f"Tipuri disponibile: {', '.join(r['name_de'] for r in roof_types)}"
        )
    
    # ==========================================
    # VALIDARE COST RANGE
    # ==========================================
    cost_range = roof.get("cost_estimate_eur_per_m2")
    if not isinstance(cost_range, list) or len(cost_range) != 2:
        raise ValueError(f"❌ Tipul '{roof['name_de']}' nu are date valide pentru cost/m²")
    
    cmin, cmax = float(cost_range[0]), float(cost_range[1])
    
    # ==========================================
    # PERIMETRU (din input sau estimat)
    # ==========================================
    if perimeter_m is None or perimeter_m <= 0:
        perimeter_m = _perimeter_from_area(house_area_m2)
        perimeter_source = "estimated (4×√area)"
    else:
        perimeter_source = "from perimeter module"
    
    # ==========================================
    # COEFICIENȚI
    # ==========================================
    currency = str(coeffs.get("currency", "EUR"))
    roof_overhang_m = float(coeffs.get("roof_overhang_m", 0.4))
    sheet_metal_price_per_m = float(coeffs.get("sheet_metal_price_per_m", 28.0))
    insulation_price_per_m2 = float(coeffs.get("insulation_price_per_m2", 22.0))
    extra_walls_price_per_m = float(roof.get("extra_walls_price_eur_per_m", 0.0))
    
    # Preț material selectat
    material_price_key = MATERIAL_PRICE_KEY.get(material_key, "tile_price_per_m2")
    material_unit_price = float(coeffs.get(material_price_key, 35.0))
    
    # ==========================================
    # CALCULE
    # ==========================================
    
    # 1) COST BAZĂ ACOPERIȘ (șarpantă + montaj)
    min_roof = house_area_m2 * cmin
    max_roof = house_area_m2 * cmax
    avg_roof = (min_roof + max_roof) / 2.0
    
    # 2) TINICHIGERIE (CORECTATĂ)
    # Perimetru cu streașină = Perimetru casă + (4 × streașină)
    perimeter_with_overhang = perimeter_m + (4.0 * roof_overhang_m)
    sheet_metal_total = perimeter_with_overhang * sheet_metal_price_per_m
    
    # 3) PEREȚI EXTRA (specifici tipului de acoperiș)
    extra_walls_total = perimeter_m * extra_walls_price_per_m
    
    # 4) IZOLAȚIE (CORECTAT - pe suprafața tavanului util, nu pe acoperiș)
    # ceiling_area_m2 este suprafața netă (fără pereți) calculată în area.py
    # Aceasta reprezintă exact zona unde se montează izolația (lână minerală)
    insulation_total = ceiling_area_m2 * insulation_price_per_m2
    
    # 5) MATERIAL ÎNVELITOARE
    material_total = house_area_m2 * material_unit_price
    
    # TOTAL FINAL
    final_total = round(
        avg_roof + sheet_metal_total + extra_walls_total + insulation_total + material_total,
        2
    )
    
    # ==========================================
    # STRUCTURĂ REZULTAT
    # ==========================================
    result = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "currency": currency,
            "perimeter_source": perimeter_source,
            "total_floors": total_floors
        },
        
        "inputs": {
            "house_area_m2": round(house_area_m2, 2),
            "ceiling_area_m2": round(ceiling_area_m2, 2),
            "perimeter_m": round(perimeter_m, 2),
            
            "roof_type": {
                "user_input": roof_type_user,
                "matched_name_de": roof["name_de"],
                "matched_name_en": roof.get("name_en", ""),
                "description": roof.get("description", ""),
                "cost_range_eur_per_m2": [cmin, cmax],
                "extra_walls_price_eur_per_m": extra_walls_price_per_m
            },
            
            "material": {
                "user_input": material_user or "default",
                "normalized_key": material_key,
                "unit_price_eur_per_m2": material_unit_price
            },
            
            "coefficients": {
                "roof_overhang_m": roof_overhang_m,
                "sheet_metal_price_per_m": sheet_metal_price_per_m,
                "insulation_price_per_m2": insulation_price_per_m2
            }
        },
        
        "components": {
            "roof_base": {
                "description": "Șarpantă + montaj (interval min-max)",
                "formula": "house_area_m2 × cost_per_m2",
                "min_total_eur": round(min_roof, 2),
                "max_total_eur": round(max_roof, 2),
                "average_total_eur": round(avg_roof, 2)
            },
            
            "sheet_metal": {
                "description": "Tinichigerie (jgheaburi, burlane, coperiș streașină)",
                "formula": "(perimeter_m + 4×overhang_m) × sheet_metal_price_per_m",
                "perimeter_with_overhang_m": round(perimeter_with_overhang, 2),
                "total_eur": round(sheet_metal_total, 2)
            },
            
            "extra_walls": {
                "description": "Pereți suplimentari (specifici tipului de acoperiș)",
                "formula": "perimeter_m × extra_walls_price_eur_per_m",
                "total_eur": round(extra_walls_total, 2)
            },
            
            "insulation": {
                "description": "Izolație tavan (sub acoperiș)",
                "formula": "ceiling_area_m2 × insulation_price_per_m2",
                "total_eur": round(insulation_total, 2),
                "note": "Calculată pe suprafața netă a tavanului (fără pereți)"
            },
            
            "material": {
                "description": f"Material învelitoare ({material_key})",
                "formula": "house_area_m2 × material_unit_price_eur_per_m2",
                "total_eur": round(material_total, 2)
            }
        },
        
        "roof_final_total_eur": final_total
    }
    
    return result