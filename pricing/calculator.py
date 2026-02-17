# pricing/calculator.py
from __future__ import annotations

from .form_tags import build_values_by_tag
from .modules.finishes import calculate_finishes_details
from .modules.foundation import calculate_foundation_details
from .modules.openings import calculate_openings_details
from .modules.walls import calculate_walls_details
from .modules.floors import calculate_floors_details
from .modules.roof import calculate_roof_details
from .modules.utilities import calculate_utilities_details, calculate_fireplace_details
from .modules.stairs import calculate_stairs_details

def calculate_pricing_for_plan(
    area_data: dict,
    openings_data: list,
    frontend_input: dict,
    pricing_coeffs: dict,  # <--- AICI ESTE SCHIMBAREA MAJORĂ: Primim coeficienții direct
    roof_data: dict | None = None,
    total_floors: int = 1,
    is_ground_floor: bool = False,
    plan_index: int = 0,
    intermediate_floor_index: int = 0,  # Indexul etajului intermediar (1, 2, 3, etc.)
    is_basement_plan: bool = False,  # Acest plan este beciul ales (plan dedicat beci)
    has_dedicated_basement_plan: bool = False,  # Există un plan dedicat beci → nu mai adăugăm cost_basement la parter
) -> dict:
    """
    Calculează toate costurile pentru un plan folosind coeficienții din DB.
    """
    
    # 1. EXTRAGEM COEFICIENȚII DIN DICȚIONARUL UNIFICAT
    # (Acest dicționar vine din db_loader.py)
    finish_coeffs = pricing_coeffs["finishes"]
    foundation_coeffs = pricing_coeffs["foundation"]
    openings_coeffs = pricing_coeffs["openings"]
    system_coeffs = pricing_coeffs["system"]
    area_coeffs = pricing_coeffs["area"]
    stairs_coeffs = pricing_coeffs["stairs"]
    
    # Utilități
    electricity_coeffs = pricing_coeffs["utilities"]["electricity"]
    heating_coeffs = pricing_coeffs["utilities"]["heating"]
    ventilation_coeffs = pricing_coeffs["utilities"]["ventilation"]
    sewage_coeffs = pricing_coeffs["utilities"]["sewage"]

    # 2. EXTRAGE ARII
    walls_data = area_data.get("walls", {})
    # Pentru structură: folosim net_area_m2_structure pentru interior
    w_int_net_structure = float(walls_data.get("interior", {}).get("net_area_m2_structure", 
                                                                    walls_data.get("interior", {}).get("net_area_m2", 0.0)))
    # Pentru finisaje: folosim net_area_m2 pentru interior
    w_int_net_finish = float(walls_data.get("interior", {}).get("net_area_m2", 0.0))
    # Exterior: același pentru structură și finisaje
    w_ext_net = float(walls_data.get("exterior", {}).get("net_area_m2", 0.0))
    
    surfaces = area_data.get("surfaces", {})
    foundation_area = float(surfaces.get("foundation_m2") or 0.0)
    floor_area = float(surfaces.get("floor_m2") or 0.0)
    ceiling_area = float(surfaces.get("ceiling_m2") or 0.0)

    # 3. PREFERINȚE UTILIZATOR (citite după tag când e cazul, pentru formulare diferite per client)
    values_by_tag = build_values_by_tag(frontend_input)
    sist_constr = frontend_input.get("sistemConstructiv", {})
    mat_finisaj = frontend_input.get("materialeFinisaj", {})
    performanta = frontend_input.get("performantaEnergetica", {}) or frontend_input.get("performanta", {})

    def _fv(tag: str, fallback: any = None):
        """Valoare din formular după tag, cu fallback."""
        v = values_by_tag.get(tag)
        return v if v is not None and v != "" else fallback

    # Debug: afișăm toate cheile de finisaje disponibile
    print(f"🔍 [PRICING] Available finish keys in mat_finisaj: {list(mat_finisaj.keys())}")
    for key in ["finisajInterior_ground", "fatada_ground", "finisajInterior_floor_1", "fatada_floor_1", 
                "finisajInterior_floor_2", "fatada_floor_2", "finisajInteriorMansarda", "fatadaMansarda"]:
        if key in mat_finisaj:
            print(f"   {key} = {mat_finisaj[key]}")
    
    # Normalize user-provided strings (sursă: tag system_type sau câmp tipSistem)
    raw_system = str(_fv("system_type") or sist_constr.get("tipSistem", "HOLZRAHMEN") or "HOLZRAHMEN").strip()

    # SYSTEM normalization to match coeff keys
    sys_u = raw_system.upper()
    if "HOLZRAHMEN" in sys_u:
        system_constructie = "HOLZRAHMEN"
    elif "CLT" in sys_u and "PREMIUM" in sys_u:
        system_constructie = "CLT Premium"
    elif "CLT" in sys_u:
        system_constructie = "CLT"
    elif "MASSIV" in sys_u:
        system_constructie = "MASSIVHOLZ"
    else:
        system_constructie = raw_system

    # Fundament (tag: foundation_type)
    structura_cladirii = frontend_input.get("structuraCladirii", {})
    foundation_type = _fv("foundation_type") or structura_cladirii.get("tipFundatieBeci") or sist_constr.get("tipFundatie", "Placă")
    if foundation_type == "Placă":
        foundation_type = "Kein Keller (nur Bodenplatte)"
    
    # Finishes/materials normalization (tolerate tenant-specific labels)
    def _norm_finish(v: str, default: str) -> str:
        s = str(v or "").strip()
        if not s:
            return default
        u = s.upper()
        if "TENC" in u or "PUTZ" in u:
            return "Tencuială"
        if "FIBRO" in u:
            return "Fibrociment"
        if "MIX" in u:
            return "Mix"
        if "ARS" in u or "SHOU" in u:
            return "Lemn Ars (Shou Sugi Ban)"
        if "LEMN" in u or "HOLZ" in u:
            return "Lemn"
        return s

    # Determinăm finisajele bazat pe etaj
    floor_type = area_data.get("floor_type", "").lower()
    is_top_floor_plan = ("top" in floor_type or "mansard" in floor_type) or (total_floors == 1)
    
    # Verificăm dacă avem beci locuibil din frontend_data (și din câmpul basementUse)
    structura_cladirii = frontend_input.get("structuraCladirii", {})
    tip_fundatie_beci = _fv("foundation_type") or structura_cladirii.get("tipFundatieBeci", "")
    has_basement_livable = frontend_input.get("basementUse", False) or ("mit einfachem Ausbau" in str(tip_fundatie_beci))
    
    # ---------- Plan dedicat beci: doar pereți interiori, finisaje interior, podele, utilități (fără exterior/fundație/acoperiș) ----------
    if is_basement_plan:
        finish_int_beci = _norm_finish(
            mat_finisaj.get("finisajInteriorBeci") or mat_finisaj.get("finisajInterior", "Tencuială"),
            "Tencuială"
        )
        cost_walls_b = calculate_walls_details(
            system_coeffs, w_int_net_structure, 0.0,
            system=system_constructie
        )
        cost_finishes_b = calculate_finishes_details(
            finish_coeffs, w_int_net_finish, 0.0,
            type_int=finish_int_beci, type_ext="Tencuială",
            floor_label="Beci"
        )
        cost_floors_b = calculate_floors_details(area_coeffs, floor_area, ceiling_area)
        electricity_coeffs = pricing_coeffs["utilities"]["electricity"]
        heating_coeffs = pricing_coeffs["utilities"]["heating"]
        ventilation_coeffs = pricing_coeffs["utilities"]["ventilation"]
        sewage_coeffs = pricing_coeffs["utilities"]["sewage"]
        energy_level = _fv("energy_level") or performanta.get("nivelEnergetic", "Standard")
        heating_type = _fv("heating_type") or performanta.get("tipIncalzire") or frontend_input.get("incalzire", {}).get("tipIncalzire") or "Gaz"
        has_ventilation = _fv("ventilation") if _fv("ventilation") is not None and _fv("ventilation") != "" else performanta.get("ventilatie", False)
        if has_basement_livable:
            cost_utilities_b = calculate_utilities_details(
                electricity_coeffs, heating_coeffs, ventilation_coeffs, sewage_coeffs,
                total_floor_area_m2=floor_area, energy_level=energy_level,
                heating_type=heating_type, has_ventilation=has_ventilation, has_sewage=True
            )
        else:
            elec_base = float(electricity_coeffs.get("coefficient_electricity_per_m2", 60.0))
            elec_modifiers = electricity_coeffs.get("energy_performance_modifiers", {})
            elec_modifier = float(elec_modifiers.get(energy_level, 1.0))
            sewage_base = float(sewage_coeffs.get("coefficient_sewage_per_m2", 45.0))
            cost_utilities_b = {
                "total_cost": round(floor_area * (elec_base * elec_modifier + sewage_base), 2),
                "detailed_items": [
                    {"category": "electricity", "name": f"Instalație electrică beci ({energy_level})", "area_m2": round(floor_area, 2), "total_cost": round(floor_area * elec_base * elec_modifier, 2)},
                    {"category": "sewage", "name": "Canalizare beci", "area_m2": round(floor_area, 2), "total_cost": round(floor_area * sewage_base, 2)}
                ]
            }
        total_b = cost_walls_b["total_cost"] + cost_finishes_b["total_cost"] + cost_floors_b["total_cost"] + cost_utilities_b["total_cost"]
        cost_basement_only = {
            "total_cost": round(total_b, 2),
            "detailed_items": (
                cost_walls_b.get("detailed_items", []) +
                cost_finishes_b.get("detailed_items", []) +
                cost_floors_b.get("detailed_items", []) +
                cost_utilities_b.get("detailed_items", [])
            )
        }
        print(f"✅ [PRICING] Plan beci (dedicat): {cost_basement_only['total_cost']:,.0f} EUR (fără finisaje exterioare)")
        return {
            "total_cost_eur": round(total_b, 2),
            "total_area_m2": floor_area,
            "currency": "EUR",
            "breakdown": {
                "foundation": {"total_cost": 0.0, "detailed_items": []},
                "structure_walls": cost_walls_b,
                "floors_ceilings": cost_floors_b,
                "roof": {"total_cost": 0.0, "detailed_items": []},
                "openings": {"total_cost": 0.0, "detailed_items": []},
                "finishes": cost_finishes_b,
                "utilities": cost_utilities_b,
                "stairs": {"total_cost": 0.0, "detailed_items": []},
                "fireplace": {"total_cost": 0.0, "detailed_items": []},
                "basement": cost_basement_only
            }
        }
    
    # Verificăm dacă top floor este mansardă sau pod
    # În frontend, listaEtaje conține tipurile de etaje: 'intermediar', 'pod', 'mansarda_ohne', 'mansarda_mit'
    lista_etaje = structura_cladirii.get("listaEtaje", [])
    is_mansarda = False
    if lista_etaje and isinstance(lista_etaje, list):
        ultimul_etaj = lista_etaje[-1]
        if ultimul_etaj and isinstance(ultimul_etaj, str) and ultimul_etaj.startswith("mansarda"):
            is_mansarda = True
    
    # Calculăm totalFloors din frontend (ground + etaje intermediare, fără mansardă/pod)
    # Numărăm câte etaje intermediare sunt în listaEtaje
    etaje_intermediare_count = 0
    if lista_etaje and isinstance(lista_etaje, list):
        etaje_intermediare_count = sum(1 for e in lista_etaje if e == "intermediar")
    total_floors_frontend = 1 + etaje_intermediare_count  # Ground (1) + etaje intermediare
    
    # IMPORTANT: În frontend, etajele sunt numerotate astfel:
    # - idx=0 → 'Erdgeschoss' → finisajInterior_ground / fatada_ground
    # - idx=1 → 'Obergeschoss 1' → finisajInterior_floor_1 / fatada_floor_1
    # - idx=2 → 'Obergeschoss 2' → finisajInterior_floor_2 / fatada_floor_2
    # - Mansardă → finisajInteriorMansarda / fatadaMansarda (separat, dacă ultimul etaj este mansardă)
    # - Pod → folosește ultimul index din totalFloors (dacă ultimul etaj este pod)
    # 
    # În frontend, totalFloors = 1 + etajeIntermediare (ground + etaje intermediare, fără mansardă/pod)
    # Mansardă/Pod este afișat separat dacă există.
    
    # Determinăm cheile de finisaje bazat pe tipul etajului
    if is_ground_floor:
        # Ground floor
        finish_int_key = "finisajInterior_ground"
        finish_ext_key = "fatada_ground"
    elif is_top_floor_plan:
        # Top floor - verificăm dacă este mansardă sau pod
        if is_mansarda:
            # Mansardă - folosim finisajele pentru mansardă
            finish_int_key = "finisajInteriorMansarda"
            finish_ext_key = "fatadaMansarda"
        else:
            # Pod - folosim ultimul index din totalFloors (din frontend)
            # totalFloors = 1 + etajeIntermediare, deci ultimul index este totalFloors
            # De exemplu: ground + pod → totalFloors = 1 → floor_1
            # ground + 1 intermediar + pod → totalFloors = 2 → floor_2
            # ground + 2 intermediare + pod → totalFloors = 3 → floor_3
            floor_idx = total_floors_frontend  # Ultimul index (ground=0, primul intermediar=1, pod=totalFloors)
            finish_int_key = f"finisajInterior_floor_{floor_idx}"
            finish_ext_key = f"fatada_floor_{floor_idx}"
    else:
        # Etaj intermediar - folosim intermediate_floor_index + 1 pentru floor_X
        floor_idx = intermediate_floor_index + 1  # floor_1, floor_2, etc.
        finish_int_key = f"finisajInterior_floor_{floor_idx}"
        finish_ext_key = f"fatada_floor_{floor_idx}"
    
    # Fallback la valorile vechi dacă nu există valorile noi per etaj
    finish_int = _norm_finish(
        mat_finisaj.get(finish_int_key) or mat_finisaj.get("finisajInterior", "Tencuială"), 
        "Tencuială"
    )
    finish_ext = _norm_finish(
        mat_finisaj.get(finish_ext_key) or mat_finisaj.get("fatada", "Tencuială"), 
        "Tencuială"
    )
    
    # Determinăm eticheta etajului pentru afișare în PDF
    if is_ground_floor:
        floor_label = "Erdgeschoss"
    elif is_top_floor_plan:
        if is_mansarda:
            floor_label = "Mansardă"
        else:
            floor_label = f"Obergeschoss {total_floors_frontend}" if total_floors_frontend > 1 else "Dachgeschoss"
    else:
        # Etaj intermediar
        floor_label = f"Obergeschoss {intermediate_floor_index + 1}"
    
    # Debug: afișăm valorile finale folosite
    floor_type_str = "ground" if is_ground_floor else ("top" if is_top_floor_plan else f"intermediate_{intermediate_floor_index + 1}")
    print(f"✅ [PRICING] Plan {plan_index} ({floor_type_str}): Final finishes - Interior: {finish_int}, Exterior: {finish_ext}")
    print(f"   Used keys: {finish_int_key}, {finish_ext_key}")
    print(f"   Floor label: {floor_label}")
    
    # Pentru beci locuibil, calculăm separat finisajele interioare
    # (beciul nu are fațadă exterioară, doar pereți interiori)
    if has_basement_livable and is_ground_floor:
        # Dacă suntem la ground floor și avem beci locuibil, folosim finisajele beciului pentru pereții interiori ai beciului
        # Notă: Acest lucru ar trebui să fie calculat separat pentru beci, dar pentru moment folosim aceeași logică
        finish_int_beci = _norm_finish(
            mat_finisaj.get("finisajInteriorBeci") or finish_int,
            "Tencuială"
        )
        # Pentru beci, folosim finisajul interior beci pentru pereții interiori
        # (dar pentru moment, folosim finish_int normal pentru calculul general)
    
    energy_level = _fv("energy_level") or performanta.get("nivelEnergetic", "Standard")
    has_ventilation = _fv("ventilation") if _fv("ventilation") is not None and _fv("ventilation") != "" else performanta.get("ventilatie", False)
    
    # Citim tipul de încălzire din pasul "performanta" (mutat acolo) sau din pasul "incalzire" (fallback)
    heating_type = _fv("heating_type") or performanta.get("tipIncalzire") or frontend_input.get("incalzire", {}).get("tipIncalzire") or performanta.get("incalzire", "Gaz")
    
    # Citim tipul de semineu din pasul "performantaEnergetica" (nou) sau "incalzire" (fallback)
    tip_semineu = _fv("fireplace_type") or performanta.get("tipSemineu") or frontend_input.get("incalzire", {}).get("tipSemineu") or frontend_input.get("performantaEnergetica", {}).get("tipSemineu")
    # Fallback pentru vechiul câmp boolean "semineu"
    if not tip_semineu:
        incalzire_data = frontend_input.get("incalzire", {})
        has_semineu_old = incalzire_data.get("semineu", False) or False
        if has_semineu_old:
            tip_semineu = "Klassischer Holzofen"  # Default pentru vechiul câmp boolean
    
    # 4. CALCULE COMPONENTE (NESCHIMBAT - DOAR INPUTURILE SUNT NOI)
    
    cost_walls = calculate_walls_details(
        system_coeffs, w_int_net_structure, w_ext_net,
        system=system_constructie
    )
    
    cost_finishes = calculate_finishes_details(
        finish_coeffs, w_int_net_finish, w_ext_net,
        type_int=finish_int, type_ext=finish_ext,
        floor_label=floor_label
    )
    
    cost_foundation = calculate_foundation_details(
        foundation_coeffs, foundation_area,
        type_foundation=foundation_type
    )
    
    cost_openings = calculate_openings_details(
        openings_coeffs, openings_data,
        frontend_data=frontend_input
    )
    
    cost_floors_ceilings = calculate_floors_details(
        area_coeffs, floor_area, ceiling_area
    )

    if roof_data:
        # Aici e un mic truc: roof_data are nevoie de coeficienți integrați
        # În versiunea V1, roof_coefficients erau citiți în roof/calculator.py
        # Pentru V2 rapid, putem injecta prețurile în roof_data înainte de calcul
        roof_coeffs = pricing_coeffs["roof"]
        # Suprascriem valorile din roof_data cu cele din DB
        roof_data["price_coeffs"] = roof_coeffs 
        cost_roof = calculate_roof_details(roof_data)
    else:
        cost_roof = {"total_cost": 0.0, "detailed_items": []}
    
    cost_utilities = calculate_utilities_details(
        electricity_coeffs,
        heating_coeffs,
        ventilation_coeffs,
        sewage_coeffs,
        total_floor_area_m2=floor_area,
        energy_level=energy_level,
        heating_type=heating_type,
        has_ventilation=has_ventilation,
        has_sewage=True 
    )

    if is_ground_floor:
        cost_stairs = calculate_stairs_details(stairs_coeffs, total_floors)
    else:
        cost_stairs = {"total_cost": 0.0, "detailed_items": []}
    
    # Calculăm costurile pentru semineu și horn (doar o dată, la ground floor); prețuri din DB
    fireplace_coeffs = pricing_coeffs.get("fireplace", {})
    if is_ground_floor:
        cost_fireplace = calculate_fireplace_details(tip_semineu, total_floors, fireplace_coeffs)
    else:
        cost_fireplace = {"total_cost": 0.0, "detailed_items": []}

    # 5. TOTAL – Nivel ofertă (tag: offer_scope) decide CE includem, nu coeficient
    nivel_oferta = (str(_fv("offer_scope") or sist_constr.get("nivelOferta") or "").strip())
    # Rohbau = doar structură; Tragwerk+Fenster = + deschideri; Schlüsselfertig sau necunoscut = tot
    include_openings = nivel_oferta != "Rohbau/Tragwerk"
    include_finishes = nivel_oferta not in ("Rohbau/Tragwerk", "Tragwerk + Fenster")
    include_utilities = nivel_oferta not in ("Rohbau/Tragwerk", "Tragwerk + Fenster")
    include_stairs = nivel_oferta not in ("Rohbau/Tragwerk", "Tragwerk + Fenster")
    include_fireplace = nivel_oferta not in ("Rohbau/Tragwerk", "Tragwerk + Fenster")

    # 5. Structură totală (fundație + pereți + planșeu + acoperiș) × acces șantier × teren
    sist_coeffs = pricing_coeffs.get("sistem_constructiv", {})
    structure_total = (
        cost_foundation["total_cost"] +
        cost_walls["total_cost"] +
        cost_floors_ceilings["total_cost"] +
        cost_roof["total_cost"]
    )
    # Mapare valori formular (RO/altă limbă) la cheile din DB (DE)
    _acces_to_key = {
        "Leicht (LKW 40t)": "Leicht (LKW 40t)", "Mittel": "Mittel", "Schwierig": "Schwierig",
        "Ușor (camion 40t)": "Leicht (LKW 40t)", "Mediu": "Mittel", "Dificil": "Schwierig",
    }
    _teren_to_key = {
        "Eben": "Eben", "Leichte Hanglage": "Leichte Hanglage", "Starke Hanglage": "Starke Hanglage",
        "Plan": "Eben", "Pantă ușoară": "Leichte Hanglage", "Pantă mare": "Starke Hanglage",
    }
    acces_santier_raw = _fv("site_access") or sist_constr.get("accesSantier") or (frontend_input.get("logistica") or {}).get("accesSantier")
    acces_santier = _acces_to_key.get(str(acces_santier_raw or "").strip(), (acces_santier_raw or "").strip())
    acces_factor = 1.0
    if acces_santier and sist_coeffs.get("acces_santier_factor"):
        acces_factor = float(sist_coeffs["acces_santier_factor"].get(acces_santier, 1.0))
    teren_raw = _fv("terrain") or sist_constr.get("teren") or (frontend_input.get("logistica") or {}).get("teren")
    teren = _teren_to_key.get(str(teren_raw or "").strip(), (teren_raw or "").strip())
    teren_factor = 1.0
    if teren and sist_coeffs.get("teren_factor"):
        teren_factor = float(sist_coeffs["teren_factor"].get(teren, 1.0))
    structure_total = structure_total * acces_factor * teren_factor
    if plan_index == 0:
        print(f"✅ [PRICING] Acces șantier: {acces_santier_raw!r} → factor {acces_factor}; Teren: {teren_raw!r} → factor {teren_factor}")

    total_plan_cost = (
        structure_total +
        (cost_openings["total_cost"] if include_openings else 0.0) +
        (cost_finishes["total_cost"] if include_finishes else 0.0) +
        (cost_utilities["total_cost"] if include_utilities else 0.0) +
        (cost_stairs["total_cost"] if include_stairs else 0.0) +
        (cost_fireplace["total_cost"] if include_fireplace else 0.0)
    )

    # Strom-/Wasseranschluss: pauschal o singură dată per proiect (doar la parter)
    has_utilitati = _fv("utilities_connection")
    has_utilitati = has_utilitati if (has_utilitati is not None and has_utilitati != "") else sist_constr.get("utilitati", True)
    if is_ground_floor and not has_utilitati:
        total_plan_cost = total_plan_cost + float(sist_coeffs.get("utilitati_anschluss_price", 0))

    # 6. CALCUL BASEMENT (dacă există și NU e deja un plan dedicat beci)
    # Când has_dedicated_basement_plan = True, beciul e calculat pe planul dedicat; nu mai adăugăm bloc aici.
    cost_basement = {"total_cost": 0.0, "detailed_items": []}
    has_basement = tip_fundatie_beci and "Keller" in str(tip_fundatie_beci) and "Kein Keller" not in str(tip_fundatie_beci)
    
    if has_basement and is_ground_floor and not has_dedicated_basement_plan:
        # Coeficienți pentru beci
        if has_basement_livable:
            # Beci locuibil: coeficienți mai mari (include finisaje elaborate)
            coeff_walls = 0.85  # 85% din pereții parterului
            coeff_floors = 0.90  # 90% din suprafața parterului
        else:
            # Beci nelocuibil: coeficienți mai mici (finisaje simple/minime)
            coeff_walls = 0.60  # 60% din pereții parterului
            coeff_floors = 0.70  # 70% din suprafața parterului
        
        # Finisaje pentru basement (folosim finisajul specificat pentru beci sau fallback la interior)
        finish_int_beci = _norm_finish(
            mat_finisaj.get("finisajInteriorBeci") or finish_int,
            "Tencuială"
        )
        
        # Calculăm ariile pentru beci folosind coeficienții
        w_int_net_structure_basement = w_int_net_structure * coeff_walls
        w_int_net_finish_basement = w_int_net_finish * coeff_walls
        floor_area_basement = floor_area * coeff_floors
        ceiling_area_basement = ceiling_area * coeff_floors
        
        print(f"   📐 [BASEMENT] Coeficienți aplicați: pereți={coeff_walls:.0%}, podele={coeff_floors:.0%}")
        print(f"      - Pereți structură: {w_int_net_structure:.2f} m² × {coeff_walls:.0%} = {w_int_net_structure_basement:.2f} m²")
        print(f"      - Pereți finisaje: {w_int_net_finish:.2f} m² × {coeff_walls:.0%} = {w_int_net_finish_basement:.2f} m²")
        print(f"      - Podele: {floor_area:.2f} m² × {coeff_floors:.0%} = {floor_area_basement:.2f} m²")
        
        # Pereți interiori pentru basement (folosim ariile calculate cu coeficienții)
        cost_walls_basement = calculate_walls_details(
            system_coeffs, w_int_net_structure_basement, 0.0,
            system=system_constructie
        )
        
        # Finisaje interioare pentru basement (folosim ariile calculate cu coeficienții)
        cost_finishes_basement = calculate_finishes_details(
            finish_coeffs, w_int_net_finish_basement, 0.0,  # Nu avem fațadă pentru basement
            type_int=finish_int_beci, type_ext="Tencuială",  # Nu contează type_ext pentru basement
            floor_label="Beci"
        )
        
        # Podele și tavan pentru basement (folosim ariile calculate cu coeficienții)
        cost_floors_basement = calculate_floors_details(
            area_coeffs, floor_area_basement, ceiling_area_basement
        )
        
        # Utilități pentru basement
        cost_utilities_basement = {"total_cost": 0.0, "detailed_items": []}
        
        if has_basement_livable:
            # Beci locuibil: toate utilitățile (curent + încălzire + ventilație + canalizare)
            # Folosim suprafața calculată cu coeficientul pentru beci
            cost_utilities_basement = calculate_utilities_details(
                electricity_coeffs,
                heating_coeffs,
                ventilation_coeffs,
                sewage_coeffs,
                total_floor_area_m2=floor_area_basement,  # Folosim suprafața calculată pentru beci
                energy_level=energy_level,
                heating_type=heating_type,
                has_ventilation=has_ventilation,
                has_sewage=True
            )
            print(f"      ✅ Utilități beci locuibil: {cost_utilities_basement['total_cost']:,.0f} EUR (arie: {floor_area_basement:.2f} m²)")
        else:
            # Beci nelocuibil: doar curent + canalizare (fără încălzire și ventilație)
            elec_base = float(electricity_coeffs.get("coefficient_electricity_per_m2", 60.0))
            elec_modifiers = electricity_coeffs.get("energy_performance_modifiers", {})
            elec_modifier = float(elec_modifiers.get(energy_level, 1.0))
            elec_cost = floor_area_basement * elec_base * elec_modifier
            
            sewage_base = float(sewage_coeffs.get("coefficient_sewage_per_m2", 45.0))
            sewage_cost = floor_area_basement * sewage_base
            
            cost_utilities_basement = {
                "total_cost": round(elec_cost + sewage_cost, 2),
                "detailed_items": [
                    {
                        "category": "electricity",
                        "name": f"Instalație electrică beci ({energy_level})",
                        "area_m2": round(floor_area_basement, 2),
                        "base_price_per_m2": elec_base,
                        "energy_modifier": elec_modifier,
                        "final_price_per_m2": round(elec_base * elec_modifier, 2),
                        "total_cost": round(elec_cost, 2)
                    },
                    {
                        "category": "sewage",
                        "name": "Canalizare beci",
                        "area_m2": round(floor_area_basement, 2),
                        "base_price_per_m2": sewage_base,
                        "total_cost": round(sewage_cost, 2)
                    }
                ]
            }
            print(f"      ✅ Utilități beci nelocuibil: {cost_utilities_basement['total_cost']:,.0f} EUR (arie: {floor_area_basement:.2f} m², doar curent + canalizare)")
        
        # Structura beciului (pereți + planșeu) se înmulțește cu acces șantier și teren
        structure_basement = cost_walls_basement["total_cost"] + cost_floors_basement["total_cost"]
        structure_basement = structure_basement * acces_factor * teren_factor

        total_basement_cost = (
            structure_basement +
            cost_finishes_basement["total_cost"] +
            cost_utilities_basement["total_cost"]
        )
        
        cost_basement = {
            "total_cost": round(total_basement_cost, 2),
            "detailed_items": (
                cost_walls_basement.get("detailed_items", []) +
                cost_finishes_basement.get("detailed_items", []) +
                cost_floors_basement.get("detailed_items", []) +
                cost_utilities_basement.get("detailed_items", [])
            )
        }
        
        # Adăugăm costurile basement-ului la total
        total_plan_cost += cost_basement["total_cost"]
        basement_type = "locuibil" if has_basement_livable else "nelocuibil"
        print(f"✅ [PRICING] Basement calculat ({basement_type}): {cost_basement['total_cost']:,.0f} EUR")

    # Breakdown: componente excluse prin nivel ofertă apar cu cost 0; structura e afișată cu factorii acces × teren aplicați
    def _zero_if_excluded(cost_dict: dict, include: bool) -> dict:
        if include:
            return cost_dict
        return {"total_cost": 0.0, "detailed_items": []}

    def _scale_cost_dict(d: dict, scale: float) -> dict:
        if scale == 1.0:
            return d
        out = {"total_cost": round(d["total_cost"] * scale, 2), "detailed_items": []}
        for it in d.get("detailed_items", []):
            item = dict(it)
            if "cost" in item:
                item["cost"] = round(item["cost"] * scale, 2)
            if "total_cost" in item:
                item["total_cost"] = round(item["total_cost"] * scale, 2)
            out["detailed_items"].append(item)
        return out

    structure_scale = acces_factor * teren_factor
    return {
        "total_cost_eur": round(total_plan_cost, 2),
        "total_area_m2": floor_area,
        "currency": "EUR",
        "breakdown": {
            "foundation": _scale_cost_dict(cost_foundation, structure_scale),
            "structure_walls": _scale_cost_dict(cost_walls, structure_scale),
            "floors_ceilings": _scale_cost_dict(cost_floors_ceilings, structure_scale),
            "roof": _scale_cost_dict(cost_roof, structure_scale),
            "openings": _zero_if_excluded(cost_openings, include_openings),
            "finishes": _zero_if_excluded(cost_finishes, include_finishes),
            "utilities": _zero_if_excluded(cost_utilities, include_utilities),
            "stairs": _zero_if_excluded(cost_stairs, include_stairs),
            "fireplace": _zero_if_excluded(cost_fireplace, include_fireplace),
            "basement": cost_basement
        }
    }