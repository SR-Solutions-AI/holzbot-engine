# new/runner/roof/mapper.py
from __future__ import annotations

# Mapare FRONTEND (RO) → BACKEND (DE)
ROOF_TYPE_MAPPING = {
    # Variante simple
    "drept": "Flachdach",
    "plat": "Flachdach",
    
    # Două ape
    "doua ape": "Satteldach",
    "două ape": "Satteldach",
    "satteldach": "Satteldach",
    
    # Patru ape
    "patru ape": "Walmdach",
    "walmdach": "Walmdach",
    
    # Mansardat
    "mansardat": "Mansardendach",
    "mansardendach": "Mansardendach",
    
    # Complex (fallback generic)
    "sarpanta complexa": "Kreuzdach",
    "șarpantă complexă": "Kreuzdach",
    "complex": "Kreuzdach",
}


def normalize_roof_type(user_input: str | None) -> str:
    """
    Convertește input utilizator (RO/EN/DE) în name_de standard.
    
    Args:
        user_input: Input de la frontend (ex: "Două ape", "Walmdach")
    
    Returns:
        Tipul standard DE (ex: "Satteldach")
        Default: "Satteldach" dacă nu găsește match
    """
    if not user_input:
        return "Satteldach"  # default
    
    normalized = user_input.lower().strip()
    
    # Încearcă match din mapare
    matched = ROOF_TYPE_MAPPING.get(normalized)
    if matched:
        return matched
    
    # Dacă nu găsește, presupunem că e deja în DE și returnăm capitalizat
    # (ex: utilizatorul a scris direct "Walmdach")
    return user_input.strip()


def normalize_material(user_input: str | None) -> str:
    """
    Convertește input material (RO) în cheie internă.
    
    Args:
        user_input: "Țiglă" / "Tablă" / "Membrană"
    
    Returns:
        "tile" / "metal" / "membrane"
    """
    if not user_input:
        return "tile"  # default
    
    normalized = user_input.lower().strip()
    
    material_map = {
        "tigla": "tile",
        "țiglă": "tile",
        "tabla": "metal",
        "tablă": "metal",
        "membrana": "membrane",
        "membrană": "membrane",
    }
    
    return material_map.get(normalized, "tile")