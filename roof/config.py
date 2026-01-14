# new/runner/roof/config.py
from __future__ import annotations
from pathlib import Path

# Paths la fișiere de date
DATA_DIR = Path(__file__).parent / "data"
ROOF_TYPES_FILE = DATA_DIR / "roof_types_germany.json"
ROOF_COEFFICIENTS_FILE = DATA_DIR / "roof_coefficients.json"

# Coeficienți default (fallback dacă frontend_data.json lipsește)
DEFAULT_COEFFICIENTS = {
    "currency": "EUR",
    "roof_overhang_m": 0.4,              # streașină (metri)
    "sheet_metal_price_per_m": 28.0,    # tinichigerie (€/m liniar)
    "insulation_price_per_m2": 22.0,    # izolație (€/m²)
    "tile_price_per_m2": 35.0,          # țiglă (€/m²)
    "metal_price_per_m2": 22.0,         # tablă (€/m²)
    "membrane_price_per_m2": 12.0       # membrană (€/m²)
}

# Mapare materiale RO → cheie internă
MATERIAL_PRICE_KEY = {
    "tigla": "tile_price_per_m2",
    "țiglă": "tile_price_per_m2",
    "tabla": "metal_price_per_m2",
    "tablă": "metal_price_per_m2",
    "membrana": "membrane_price_per_m2",
    "membrană": "membrane_price_per_m2",
    # Sadiki new internal material keys (from roof/mapper.py)
    "shingle": "roof_shingle_price_per_m2",
    "metal_tile": "roof_metal_tile_price_per_m2",
    "ceramic_tile": "roof_ceramic_tile_price_per_m2",
    "tpo_pvc": "roof_tpo_pvc_price_per_m2",
    "green_extensive": "roof_green_extensive_price_per_m2",
}