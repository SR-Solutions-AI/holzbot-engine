# new/runner/pricing/modules/utilities.py
from __future__ import annotations


def calculate_utilities_details(
    coeffs_electricity: dict,
    coeffs_heating: dict,
    coeffs_ventilation: dict,
    coeffs_sewage: dict,
    total_floor_area_m2: float,  # Suma tuturor etajelor
    energy_level: str,             # "Standard" | "KfW 55" | "KfW 40" | "KfW 40+"
    heating_type: str,             # "Gaz" | "Pompa de căldură" | "Electric"
    has_ventilation: bool,         # True/False
    has_sewage: bool = True        # Implicit True (mereu inclus)
) -> dict:
    """
    Calculează costurile pentru utilități & instalații:
    - Electricitate (cu modifier performanță energetică)
    - Încălzire (cu modifier tip + performanță)
    - Ventilație (opțional)
    - Canalizare (implicit inclus)
    
    Formula:
      Cost = suprafață_totală × coeficient_bază × modifier_performanță × modifier_tip
    
    Args:
        coeffs_*: Dicționare cu coeficienți din JSON
        total_floor_area_m2: Suma ariilor tuturor etajelor (floor_m2 totalizat)
        energy_level: Nivelul energetic ales
        heating_type: Tipul de încălzire ales
        has_ventilation: Dacă utilizatorul a bifat ventilație
        has_sewage: Dacă se include canalizarea (implicit True)
    
    Returns:
        Dict cu breakdown detaliat pentru fiecare utilitate
    """
    
    items = []
    total_cost = 0.0
    
    # ==========================================
    # 1. ELECTRICITATE
    # ==========================================
    elec_base = float(coeffs_electricity.get("coefficient_electricity_per_m2", 60.0))
    elec_modifiers = coeffs_electricity.get("energy_performance_modifiers", {})
    elec_modifier = float(elec_modifiers.get(energy_level, 1.0))
    
    elec_cost = total_floor_area_m2 * elec_base * elec_modifier
    total_cost += elec_cost
    
    items.append({
        "category": "electricity",
        "name": f"Instalație electrică ({energy_level})",
        "area_m2": round(total_floor_area_m2, 2),
        "base_price_per_m2": elec_base,
        "energy_modifier": elec_modifier,
        "final_price_per_m2": round(elec_base * elec_modifier, 2),
        "total_cost": round(elec_cost, 2)
    })
    
    # ==========================================
    # 2. ÎNCĂLZIRE
    # ==========================================
    heat_base = float(coeffs_heating.get("coefficient_heating_per_m2", 70.0))
    heat_type_modifiers = coeffs_heating.get("type_coefficients", {})
    heat_energy_modifiers = coeffs_heating.get("energy_performance_modifiers", {})
    
    heat_type_modifier = float(heat_type_modifiers.get(heating_type, 1.0))
    heat_energy_modifier = float(heat_energy_modifiers.get(energy_level, 1.0))
    
    heat_cost = total_floor_area_m2 * heat_base * heat_type_modifier * heat_energy_modifier
    total_cost += heat_cost
    
    items.append({
        "category": "heating",
        "name": f"Sistem încălzire ({heating_type}, {energy_level})",
        "area_m2": round(total_floor_area_m2, 2),
        "base_price_per_m2": heat_base,
        "type_modifier": heat_type_modifier,
        "energy_modifier": heat_energy_modifier,
        "final_price_per_m2": round(heat_base * heat_type_modifier * heat_energy_modifier, 2),
        "total_cost": round(heat_cost, 2)
    })
    
    # ==========================================
    # 3. VENTILAȚIE (OPȚIONAL)
    # ==========================================
    if has_ventilation:
        vent_base = float(coeffs_ventilation.get("coefficient_ventilation_per_m2", 55.0))
        vent_cost = total_floor_area_m2 * vent_base
        total_cost += vent_cost
        
        items.append({
            "category": "ventilation",
            "name": "Ventilație mecanică cu recuperare căldură",
            "area_m2": round(total_floor_area_m2, 2),
            "base_price_per_m2": vent_base,
            "total_cost": round(vent_cost, 2)
        })
    
    # ==========================================
    # 4. CANALIZARE (IMPLICIT INCLUS)
    # ==========================================
    if has_sewage:
        sewage_base = float(coeffs_sewage.get("coefficient_sewage_per_m2", 45.0))
        sewage_cost = total_floor_area_m2 * sewage_base
        total_cost += sewage_cost
        
        items.append({
            "category": "sewage",
            "name": "Canalizare",
            "area_m2": round(total_floor_area_m2, 2),
            "base_price_per_m2": sewage_base,
            "total_cost": round(sewage_cost, 2)
        })
    
    # ==========================================
    # RETURNARE
    # ==========================================
    return {
        "total_cost": round(total_cost, 2),
        "detailed_items": items,
        "summary": {
            "electricity_cost": round(elec_cost, 2),
            "heating_cost": round(heat_cost, 2),
            "ventilation_cost": round(vent_cost if has_ventilation else 0.0, 2),
            "sewage_cost": round(sewage_cost if has_sewage else 0.0, 2)
        }
    }


def calculate_fireplace_details(
    fireplace_type: str | None,
    total_floors: int,
    fireplace_coeffs: dict | None = None,
) -> dict:
    """
    Calculează costurile pentru semineu și horn (coș de fum).
    Prețurile vin din pricing_coeffs["fireplace"]["prices"] dacă sunt furnizate.

    Args:
        fireplace_type: Tipul de semineu ales (sau None/"Kein Kamin" dacă nu e selectat)
        total_floors: Numărul total de etaje (pentru calcul horn)
        fireplace_coeffs: Dict cu "prices": { "Kein Kamin": 0, "Klassischer Holzofen": ..., } din DB

    Returns:
        Dict cu breakdown pentru semineu și horn
    """
    items = []
    total_cost = 0.0

    prices_map = (fireplace_coeffs or {}).get("prices", {})
    fireplace_prices = {
        "Kein Kamin": prices_map.get("Kein Kamin", 0),
        "Klassischer Holzofen": prices_map.get("Klassischer Holzofen", 4200),
        "Moderner Design-Kaminofen": prices_map.get("Moderner Design-Kaminofen", 6500),
        "Pelletofen (automatisch)": prices_map.get("Pelletofen (automatisch)", 8500),
        "Einbaukamin": prices_map.get("Einbaukamin", 7200),
        "Kachel-/wassergeführter Kamin": prices_map.get("Kachel-/wassergeführter Kamin", 9500),
    }
    if "Kachel-/wassergeführter Kamin" in prices_map:
        fireplace_prices["Kachel-/wassergeführter Kamin"] = prices_map["Kachel-/wassergeführter Kamin"]

    fireplace_names = {
        "Kein Kamin": "Kein Kamin",
        "Klassischer Holzofen": "Klassischer Holzofen",
        "Moderner Design-Kaminofen": "Moderner Design-Kaminofen",
        "Pelletofen (automatisch)": "Pelletofen (automatisch)",
        "Einbaukamin": "Einbaukamin",
        "Kachel-/wassergeführter Kamin": "Kachel-/wassergeführter Kamin",
    }

    if fireplace_type and fireplace_type != "Kein Kamin" and fireplace_type in fireplace_prices:
        fireplace_cost = float(fireplace_prices[fireplace_type])
        total_cost += fireplace_cost

        fireplace_name = fireplace_names.get(fireplace_type, "Kamin")
        items.append({
            "category": "fireplace",
            "name": fireplace_name,
            "unit_price": fireplace_cost,
            "quantity": 1,
            "total_cost": round(fireplace_cost, 2)
        })

        horn_base_cost = 4500.0
        horn_cost_per_floor = 1500.0
        horn_total_cost = horn_base_cost + (horn_cost_per_floor * total_floors)
        total_cost += horn_total_cost

        items.append({
            "category": "chimney",
            "name": f"Schornstein (4500€ Standardpreis + 1500€ pro Geschoss für {total_floors} Geschosse)",
            "unit_price": horn_base_cost + (horn_cost_per_floor * total_floors),
            "quantity": 1,
            "total_cost": round(horn_total_cost, 2)
        })

    return {
        "total_cost": round(total_cost, 2),
        "detailed_items": items
    }