# pricing/calculator.py
from __future__ import annotations

# Importăm modulele de calcul logic (care rămân neschimbate)
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
    intermediate_floor_index: int = 0  # Indexul etajului intermediar (1, 2, 3, etc.)
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

    # 2. EXTRAGE ARII (NESCHIMBAT)
    walls_data = area_data.get("walls", {})
    w_int_net = float(walls_data.get("interior", {}).get("net_area_m2", 0.0))
    w_ext_net = float(walls_data.get("exterior", {}).get("net_area_m2", 0.0))
    
    surfaces = area_data.get("surfaces", {})
    foundation_area = float(surfaces.get("foundation_m2") or 0.0)
    floor_area = float(surfaces.get("floor_m2") or 0.0)
    ceiling_area = float(surfaces.get("ceiling_m2") or 0.0)

    # 3. PREFERINȚE UTILIZATOR (NESCHIMBAT)
    sist_constr = frontend_input.get("sistemConstructiv", {})
    mat_finisaj = frontend_input.get("materialeFinisaj", {})
    performanta = frontend_input.get("performanta", {})
    
    # Debug: afișăm toate cheile de finisaje disponibile
    print(f"🔍 [PRICING] Available finish keys in mat_finisaj: {list(mat_finisaj.keys())}")
    for key in ["finisajInterior_ground", "fatada_ground", "finisajInterior_floor_1", "fatada_floor_1", 
                "finisajInterior_floor_2", "fatada_floor_2", "finisajInteriorMansarda", "fatadaMansarda"]:
        if key in mat_finisaj:
            print(f"   {key} = {mat_finisaj[key]}")
    
    # Normalize user-provided strings from dynamic forms (DE/RO variants).
    raw_system = str(sist_constr.get("tipSistem", "HOLZRAHMEN") or "HOLZRAHMEN").strip()
    raw_prefab = str(sist_constr.get("gradPrefabricare", "PANOURI") or "PANOURI").strip()

    # SYSTEM normalization to match coeff keys (case-insensitive, tolerate labels like "Holzrahmen Standard")
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

    # PREFAB normalization to match coeff keys (MODULE/PANOURI/SANTIER)
    pf_u = raw_prefab.upper()
    if "MODUL" in pf_u:
        prefab_type = "MODULE"
    elif "PANOU" in pf_u:
        prefab_type = "PANOURI"
    elif "SANTIER" in pf_u or "ȘANTIER" in pf_u:
        prefab_type = "SANTIER"
    else:
        prefab_type = pf_u
    foundation_type = sist_constr.get("tipFundatie", "Placă")
    
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

    def _norm_tamplarie(v: str, default: str = "PVC") -> str:
        s = str(v or "").strip()
        if not s:
            return default
        u = s.upper()
        if "PVC" in u:
            return "PVC"
        if ("LEMN" in u or "HOLZ" in u) and ("ALU" in u or "ALUM" in u):
            # use the key we have in coeffs
            return "Lemn-Aluminiu" if "PREMIUM" not in u else "Premium Holz-Alu"
        if "ALU" in u or "ALUM" in u:
            return "Aluminiu"
        if "LEMN" in u or "HOLZ" in u:
            return "Lemn"
        return s

    # Determinăm finisajele bazat pe etaj
    floor_type = area_data.get("floor_type", "").lower()
    is_top_floor_plan = ("top" in floor_type or "mansard" in floor_type) or (total_floors == 1)
    
    # Verificăm dacă avem beci locuibil din frontend_data
    structura_cladirii = frontend_input.get("structuraCladirii", {})
    tip_fundatie_beci = structura_cladirii.get("tipFundatieBeci", "")
    has_basement_livable = "mit einfachem Ausbau" in str(tip_fundatie_beci)
    
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
    
    # Folosim preț standard "Lemn-Aluminiu" pentru ferestre/uși (nu mai citim din formular)
    material_tamplarie = "Lemn-Aluminiu"
    
    energy_level = performanta.get("nivelEnergetic", "Standard")
    has_ventilation = performanta.get("ventilatie", False)
    
    # Citim tipul de încălzire din pasul "performanta" (mutat acolo) sau din pasul "incalzire" (fallback)
    heating_type = performanta.get("tipIncalzire") or frontend_input.get("incalzire", {}).get("tipIncalzire") or performanta.get("incalzire", "Gaz")
    
    # Citim tipul de semineu din pasul "performantaEnergetica" (nou) sau "incalzire" (fallback)
    tip_semineu = performanta.get("tipSemineu") or frontend_input.get("incalzire", {}).get("tipSemineu") or frontend_input.get("performantaEnergetica", {}).get("tipSemineu")
    # Fallback pentru vechiul câmp boolean "semineu"
    if not tip_semineu:
        incalzire_data = frontend_input.get("incalzire", {})
        has_semineu_old = incalzire_data.get("semineu", False) or False
        if has_semineu_old:
            tip_semineu = "Klassischer Holzofen"  # Default pentru vechiul câmp boolean
    
    # 4. CALCULE COMPONENTE (NESCHIMBAT - DOAR INPUTURILE SUNT NOI)
    
    cost_walls = calculate_walls_details(
        system_coeffs, w_int_net, w_ext_net,
        system=system_constructie, prefab_type=prefab_type
    )
    
    cost_finishes = calculate_finishes_details(
        finish_coeffs, w_int_net, w_ext_net,
        type_int=finish_int, type_ext=finish_ext,
        floor_label=floor_label
    )
    
    cost_foundation = calculate_foundation_details(
        foundation_coeffs, foundation_area,
        type_foundation=foundation_type
    )
    
    cost_openings = calculate_openings_details(
        openings_coeffs, openings_data,
        material=material_tamplarie,
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
    
    # Calculăm costurile pentru semineu și horn (doar o dată, la ground floor)
    if is_ground_floor:
        cost_fireplace = calculate_fireplace_details(tip_semineu, total_floors)
    else:
        cost_fireplace = {"total_cost": 0.0, "detailed_items": []}

    # 5. TOTAL
    total_plan_cost = (
        cost_walls["total_cost"] +
        cost_finishes["total_cost"] +
        cost_foundation["total_cost"] +
        cost_openings["total_cost"] +
        cost_floors_ceilings["total_cost"] +
        cost_roof["total_cost"] +
        cost_utilities["total_cost"] +
        cost_stairs["total_cost"] +
        cost_fireplace["total_cost"]
    )

    # 6. CALCUL BASEMENT (dacă există)
    # Dacă avem basement, calculăm costurile folosind datele de la ground floor
    cost_basement = {"total_cost": 0.0, "detailed_items": []}
    structura_cladirii = frontend_input.get("structuraCladirii", {})
    tip_fundatie_beci = structura_cladirii.get("tipFundatieBeci", "")
    has_basement = tip_fundatie_beci and "Keller" in str(tip_fundatie_beci) and "Kein Keller" not in str(tip_fundatie_beci)
    has_basement_livable = has_basement and "mit einfachem Ausbau" in str(tip_fundatie_beci)
    
    if has_basement and is_ground_floor:
        # Pentru basement, folosim aceleași arii ca la ground floor
        # Basement-ul are doar pereți interiori (nu exteriori, pentru că e subteran)
        # Folosim finisajele pentru basement
        finish_int_beci = _norm_finish(
            mat_finisaj.get("finisajInteriorBeci") or finish_int,
            "Tencuială"
        )
        
        # Pereți interiori pentru basement (folosim aceeași arie ca la ground floor)
        cost_walls_basement = calculate_walls_details(
            system_coeffs, w_int_net, 0.0,  # Nu avem pereți exteriori pentru basement
            system=system_constructie, prefab_type=prefab_type
        )
        
        # Finisaje interioare pentru basement (doar pereți interiori)
        cost_finishes_basement = calculate_finishes_details(
            finish_coeffs, w_int_net, 0.0,  # Nu avem fațadă pentru basement
            type_int=finish_int_beci, type_ext="Tencuială"  # Nu contează type_ext pentru basement
        )
        
        # Podele și tavan pentru basement (folosim aceeași arie ca la ground floor)
        cost_floors_basement = calculate_floors_details(
            area_coeffs, floor_area, ceiling_area
        )
        
        # Utilități pentru basement (doar dacă este locuibil)
        cost_utilities_basement = {"total_cost": 0.0, "detailed_items": []}
        if has_basement_livable:
            # Dacă beciul este locuibil, adăugăm utilități: curent, ventilație, încălzire, canalizare
            cost_utilities_basement = calculate_utilities_details(
                electricity_coeffs,
                heating_coeffs,
                ventilation_coeffs,
                sewage_coeffs,
                total_floor_area_m2=floor_area,  # Folosim aceeași arie ca la ground floor
                energy_level=energy_level,
                heating_type=heating_type,
                has_ventilation=has_ventilation,
                has_sewage=True
            )
            print(f"✅ [PRICING] Basement utilities calculated: {cost_utilities_basement['total_cost']:,.0f} EUR (livable basement)")
        
        # Nu avem deschideri pentru basement (nu avem ferestre/uși exterioare)
        # Nu avem acoperiș pentru basement
        # Nu avem scări pentru basement (sunt calculate la ground floor)
        
        total_basement_cost = (
            cost_walls_basement["total_cost"] +
            cost_finishes_basement["total_cost"] +
            cost_floors_basement["total_cost"] +
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
        basement_type = "livable" if has_basement_livable else "unheated"
        print(f"✅ [PRICING] Basement calculated ({basement_type}): {cost_basement['total_cost']:,.0f} EUR (using ground floor data)")

    return {
        "total_cost_eur": round(total_plan_cost, 2),
        "total_area_m2": floor_area,  # Suprafața utilă a etajului (din area_data.surfaces.floor_m2)
        "currency": "EUR",
        "breakdown": {
            "foundation": cost_foundation,
            "structure_walls": cost_walls,
            "floors_ceilings": cost_floors_ceilings,
            "roof": cost_roof,
            "openings": cost_openings,
            "finishes": cost_finishes,
            "utilities": cost_utilities,
            "stairs": cost_stairs,
            "fireplace": cost_fireplace,
            "basement": cost_basement
        }
    }