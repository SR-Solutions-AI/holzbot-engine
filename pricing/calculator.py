# pricing/calculator.py
from __future__ import annotations

from .form_tags import build_values_by_tag
from .modules.finishes import calculate_finishes_details
from .modules.foundation import calculate_foundation_details
from .modules.openings import calculate_openings_details
from .modules.walls import calculate_walls_details
from .modules.floors import calculate_floors_details
from .modules.roof import calculate_roof_details
from .modules.utilities import calculate_utilities_details
from .modules.stairs import calculate_stairs_details
from area.config import WALL_HEIGHT_EXTRA_STRUCTURE_AND_EXT_FINISH_M
from pricing.height_resolve import resolve_room_height_m


def _stair_opening_dims_m_from_bbox_and_area(bbox: object, area_m2: float) -> tuple[float | None, float | None]:
    """Kürzere × längere Seite (m) aus Pixel-BBox und realer Öffnungsfläche (m²)."""
    if area_m2 <= 0 or not isinstance(bbox, list) or len(bbox) != 4:
        return None, None
    try:
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        w_px = max(0.0, x2 - x1)
        h_px = max(0.0, y2 - y1)
        apx = w_px * h_px
        if apx <= 0:
            return None, None
        mpp = (float(area_m2) / apx) ** 0.5
        w_m = w_px * mpp
        h_m = h_px * mpp
        return (min(w_m, h_m), max(w_m, h_m))
    except (TypeError, ValueError):
        return None, None


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
    w_int_net_finish_inner = float(
        walls_data.get("interior", {}).get("net_area_m2_finish_interior_walls", w_int_net_finish)
    )
    w_int_net_finish_outer = float(
        walls_data.get("interior", {}).get("net_area_m2_finish_exterior_walls", 0.0)
    )
    # Exterior: separăm structură vs finisaje
    w_ext_net_structure = float(walls_data.get("exterior", {}).get("net_area_m2_structure",
                                                                    walls_data.get("exterior", {}).get("net_area_m2", 0.0)))
    w_ext_net_finish = float(walls_data.get("exterior", {}).get("net_area_m2", 0.0))
    
    surfaces = area_data.get("surfaces", {})
    foundation_area = float(surfaces.get("foundation_m2") or 0.0)
    floor_area = float(surfaces.get("floor_m2") or 0.0)  # pardoseală (fără pereți)
    ceiling_area = float(surfaces.get("ceiling_m2") or 0.0)  # tavan (fără pereți)
    floor_structure_area = float(surfaces.get("floor_structure_m2") or floor_area)  # podea structură (cu pereți)

    # 3. PREFERINȚE UTILIZATOR (citite după tag când e cazul, pentru formulare diferite per client)
    values_by_tag = build_values_by_tag(frontend_input)
    sist_constr = frontend_input.get("sistemConstructiv", {})
    mat_finisaj = frontend_input.get("materialeFinisaj", {})
    performanta = frontend_input.get("performantaEnergetica", {}) or frontend_input.get("performanta", {})

    def _fv(tag: str, fallback: any = None):
        """Valoare din formular după tag, cu fallback."""
        v = values_by_tag.get(tag)
        return v if v is not None and v != "" else fallback

    def _as_bool(v: any, default: bool = False) -> bool:
        if isinstance(v, bool):
            return v
        if v is None:
            return default
        s = str(v).strip().lower()
        if s in {"true", "1", "yes", "ja", "on"}:
            return True
        if s in {"false", "0", "no", "nein", "off"}:
            return False
        return default

    # Debug: afișăm toate cheile de finisaje disponibile
    print(f"🔍 [PRICING] Available finish keys in mat_finisaj: {list(mat_finisaj.keys())}")
    for key in ["finisajInterior_ground", "fatada_ground", "finisajInterior_floor_1", "fatada_floor_1", 
                "finisajInterior_floor_2", "fatada_floor_2", "finisajInteriorMansarda", "fatadaMansarda"]:
        if key in mat_finisaj:
            print(f"   {key} = {mat_finisaj[key]}")
    
    # Wall structure: prefer per-floor Wandaufbau (Außenwände/Innenwände); fallback to legacy system_type/tipSistem if no Wandaufbau
    wandaufbau_data = frontend_input.get("wandaufbau", {})
    raw_system = str(_fv("system_type") or sist_constr.get("tipSistem", "HOLZRAHMEN") or "HOLZRAHMEN").strip()
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
    _tfb = str(tip_fundatie_beci)
    has_basement_livable = frontend_input.get("basementUse", False) or (
        "mit einfachem Ausbau" in _tfb or "Keller (mit Ausbau)" in _tfb
    )

    # Zubau (neues Geschoss): Wandabbruch als Streifen L×(Raumhöhe+18cm) → Abzug von Außenwand-Flächenanteilen; Positionskosten wie Dachabbruch m².
    cost_zubau_wall_demolition_extra = 0.0
    cost_zubau_wall_demolition_items: list = []
    wizard_pkg_z = str(frontend_input.get("wizard_package") or "").strip().lower() if isinstance(frontend_input, dict) else ""
    phase1_z = frontend_input.get("aufstockungPhase1", {}) if isinstance(frontend_input, dict) else {}
    auf_fk_z = frontend_input.get("aufstockungFloorKinds") if isinstance(frontend_input.get("aufstockungFloorKinds"), list) else []
    phase_floor_z = None
    if isinstance(phase1_z, dict):
        for entry in (phase1_z.get("existingFloors") or []) + (phase1_z.get("newFloors") or []):
            if isinstance(entry, dict):
                try:
                    if int(entry.get("plan_index", -1)) == int(plan_index):
                        phase_floor_z = entry
                        break
                except Exception:
                    continue
    fk_z = str((phase_floor_z or {}).get("floorKind") or (auf_fk_z[plan_index] if plan_index < len(auf_fk_z) else "")).strip().lower()
    tip_fb_z = str(structura_cladirii.get("tipFundatieBeci") or "")
    has_keller_z = "Keller" in tip_fb_z and "Kein Keller" not in tip_fb_z
    if wizard_pkg_z in ("aufstockung", "zubau", "zubau_aufstockung") and is_basement_plan and has_keller_z:
        fk_z = "bestand"
    _fkz = str(fk_z).strip().lower()
    is_zubau_new_for_walls = (
        wizard_pkg_z in ("zubau", "zubau_aufstockung")
        and _fkz in ("new", "zubau")
        and not is_basement_plan
    )

    if is_zubau_new_for_walls and isinstance(phase_floor_z, dict):
        lines_z = phase_floor_z.get("zubauWallDemolitionLines") or []
        custom_demolition_price_z = phase_floor_z.get("customDemolitionPrice")
        custom_demolition_total_z = (
            float(custom_demolition_price_z)
            if isinstance(custom_demolition_price_z, (int, float)) and float(custom_demolition_price_z) >= 0
            else None
        )
        fhmap = pricing_coeffs.get("area", {}).get("floor_height_m") if isinstance(pricing_coeffs.get("area"), dict) else None
        wall_h_m = float(resolve_room_height_m(frontend_input, fhmap))
        if is_top_floor_plan:
            try:
                ipm = frontend_input.get("inaltimePeretiMansarda")
                if ipm is not None and float(ipm) > 0:
                    wall_h_m = float(ipm)
            except Exception:
                pass
        strip_h = float(wall_h_m) + float(WALL_HEIGHT_EXTRA_STRUCTURE_AND_EXT_FINISH_M)
        w1, w2, w3 = w_int_net_finish_outer, w_ext_net_finish, w_ext_net_structure
        tot_w = w1 + w2 + w3
        d_total = 0.0
        strip_rows: list[tuple[int, float]] = []
        for iz, ln in enumerate(lines_z):
            if not isinstance(ln, dict):
                continue
            lm = float(ln.get("length_m") or 0.0)
            if lm <= 0:
                continue
            strip_m2 = lm * strip_h
            d_total += strip_m2
            strip_rows.append((iz, strip_m2))
        if custom_demolition_total_z is not None and strip_rows:
            sum_area = sum(a for _, a in strip_rows)
            for iz, strip_m2 in strip_rows:
                alloc = custom_demolition_total_z * (strip_m2 / sum_area) if sum_area > 0 else 0.0
                eff_unit = (alloc / strip_m2) if strip_m2 > 0 else 0.0
                cost_zubau_wall_demolition_items.append(
                    {
                        "category": "zubau_wall_demolition",
                        "name": f"Wandabbruch (Streifen) #{iz + 1}",
                        "area_m2": round(strip_m2, 2),
                        "unit_price": round(eff_unit, 2),
                        "total_cost": round(alloc, 2),
                    }
                )
                cost_zubau_wall_demolition_extra += alloc
        if d_total > 0 and tot_w > 0:
            w_int_net_finish_outer = max(0.0, w1 - d_total * (w1 / tot_w))
            w_ext_net_finish = max(0.0, w2 - d_total * (w2 / tot_w))
            w_ext_net_structure = max(0.0, w3 - d_total * (w3 / tot_w))
    
    # ---------- Plan dedicat beci: structură pereți interiori + exteriori; finisaje doar interior; podele, utilități (fără fundație/acoperiș) ----------
    if is_basement_plan:
        finish_int_beci = _norm_finish(
            mat_finisaj.get("finisajInteriorBeci") or mat_finisaj.get("finisajInterior", "Tencuială"),
            "Tencuială"
        )
        _wb_coeffs = pricing_coeffs.get("wandaufbau", {})
        _wb_beci_innen = (wandaufbau_data.get("innenwandeBeci") or "").strip()
        _wb_beci_aussen = (wandaufbau_data.get("außenwandeBeci") or "").strip()
        items_walls_b = []
        _cost_b_int = 0.0
        _cost_b_ext = 0.0
        if _wb_coeffs and _wb_beci_innen:
            _p_beci_innen = float(_wb_coeffs.get("innen", {}).get(_wb_beci_innen, 280))
            _cost_b_int = w_int_net_structure * _p_beci_innen
            items_walls_b.append({"category": "walls_structure_int", "name": f"Pereți Interiori Beci ({_wb_beci_innen})", "area_m2": round(w_int_net_structure, 2), "unit_price": round(_p_beci_innen, 2), "cost": round(_cost_b_int, 2)})
        else:
            _p_int = float(system_coeffs.get("base_unit_prices", {}).get(system_constructie, {}).get("interior", 280))
            _cost_b_int = w_int_net_structure * _p_int
            items_walls_b.append({"category": "walls_structure_int", "name": f"Pereți Interiori Beci ({system_constructie})", "area_m2": round(w_int_net_structure, 2), "unit_price": round(_p_int, 2), "cost": round(_cost_b_int, 2)})
        # Structură pereți exteriori beci (fără finisaje exterioare)
        if _wb_coeffs and _wb_beci_aussen:
            _p_beci_aussen = float(_wb_coeffs.get("aussen", {}).get(_wb_beci_aussen, 280))
            _cost_b_ext = w_ext_net_structure * _p_beci_aussen
            items_walls_b.append({"category": "walls_structure_ext", "name": f"Pereți Exteriori Beci ({_wb_beci_aussen})", "area_m2": round(w_ext_net_structure, 2), "unit_price": round(_p_beci_aussen, 2), "cost": round(_cost_b_ext, 2)})
        elif w_ext_net_structure > 0:
            _p_ext = float(system_coeffs.get("base_unit_prices", {}).get(system_constructie, {}).get("exterior", 280))
            _cost_b_ext = w_ext_net_structure * _p_ext
            items_walls_b.append({"category": "walls_structure_ext", "name": f"Pereți Exteriori Beci ({system_constructie})", "area_m2": round(w_ext_net_structure, 2), "unit_price": round(_p_ext, 2), "cost": round(_cost_b_ext, 2)})
        cost_walls_b = {"total_cost": round(_cost_b_int + _cost_b_ext, 2), "detailed_items": items_walls_b}
        # Keller: keine Fassade (area_ext=0), aber Innenausbau Außenwände = Außenseite der Innenwände
        # (w_int_net_finish_outer), nicht mit „Außenfassade“ verwechseln — daher getrennte Flächen wie im Area-JSON.
        cost_finishes_b = calculate_finishes_details(
            finish_coeffs, w_int_net_finish_inner, w_int_net_finish_outer, 0.0,
            type_int_inner=finish_int_beci, type_int_outer=finish_int_beci, type_ext="Tencuială",
            floor_label="Beci"
        )
        cost_floors_b = calculate_floors_details(area_coeffs, floor_structure_area, ceiling_area)
        electricity_coeffs = pricing_coeffs["utilities"]["electricity"]
        heating_coeffs = pricing_coeffs["utilities"]["heating"]
        ventilation_coeffs = pricing_coeffs["utilities"]["ventilation"]
        sewage_coeffs = pricing_coeffs["utilities"]["sewage"]
        energy_level = _fv("energy_level") or performanta.get("nivelEnergetic", "Standard")
        heating_type = _fv("heating_type") or performanta.get("tipIncalzire") or frontend_input.get("incalzire", {}).get("tipIncalzire") or "Gaz"
        has_ventilation = _fv("ventilation") if _fv("ventilation") is not None and _fv("ventilation") != "" else performanta.get("ventilatie", False)
        include_electricity = _as_bool(_fv("include_electricity"), True)
        include_sewage = _as_bool(_fv("include_sewage"), True)
        electricity_coeffs_eff = dict(electricity_coeffs)
        if not include_electricity:
            electricity_coeffs_eff["coefficient_electricity_per_m2"] = 0.0
        if has_basement_livable:
            cost_utilities_b = calculate_utilities_details(
                electricity_coeffs_eff, heating_coeffs, ventilation_coeffs, sewage_coeffs,
                total_floor_area_m2=floor_area, energy_level=energy_level,
                heating_type=heating_type, has_ventilation=has_ventilation, has_sewage=include_sewage
            )
        else:
            elec_base = float(electricity_coeffs_eff.get("coefficient_electricity_per_m2", 0.0))
            elec_modifiers = electricity_coeffs_eff.get("energy_performance_modifiers", {})
            elec_modifier = float(elec_modifiers.get(energy_level, 1.0))
            sewage_base = float(sewage_coeffs.get("coefficient_sewage_per_m2", 45.0)) if include_sewage else 0.0
            cost_utilities_b = {
                "total_cost": round(floor_area * (elec_base * elec_modifier + sewage_base), 2),
                "detailed_items": [
                    {"category": "electricity", "name": f"Instalație electrică beci ({energy_level})", "area_m2": round(floor_area, 2), "total_cost": round(floor_area * elec_base * elec_modifier, 2)},
                    {"category": "sewage", "name": "Canalizare beci", "area_m2": round(floor_area, 2), "total_cost": round(floor_area * sewage_base, 2)}
                ]
            }
        # Uși/ferestre beci: aceeași listă ca la celelate planuri + același „nivel ofertă” (Rohbau fără deschideri)
        nivel_oferta_raw_b = (str(_fv("offer_scope") or sist_constr.get("nivelOferta") or mat_finisaj.get("nivelOferta") or "").strip())
        _nivel_map_b = {
            "Structură": "Rohbau/Tragwerk",
            "Structură + ferestre": "Tragwerk + Fenster",
            "Casă completă": "Schlüsselfertiges Haus",
        }
        nivel_oferta_b = _nivel_map_b.get(nivel_oferta_raw_b, nivel_oferta_raw_b)
        if nivel_oferta_b == nivel_oferta_raw_b:
            nl_b = nivel_oferta_b.lower()
            if ("rohbau" in nl_b or "tragwerk" in nl_b) and "fenster" not in nl_b:
                nivel_oferta_b = "Rohbau/Tragwerk"
            elif "tragwerk" in nl_b and "fenster" in nl_b:
                nivel_oferta_b = "Tragwerk + Fenster"
            elif "schlüsselfertig" in nl_b or "schlusselfertig" in nl_b:
                nivel_oferta_b = "Schlüsselfertiges Haus"
        include_openings_b = nivel_oferta_b != "Rohbau/Tragwerk"
        cost_openings_b = calculate_openings_details(
            openings_coeffs, openings_data,
            frontend_data=frontend_input,
        )
        if include_openings_b:
            openings_breakdown_b = cost_openings_b
        else:
            openings_breakdown_b = {"total_cost": 0.0, "detailed_items": [], "items": []}

        total_b = (
            cost_walls_b["total_cost"]
            + cost_finishes_b["total_cost"]
            + cost_floors_b["total_cost"]
            + cost_utilities_b["total_cost"]
            + (cost_openings_b["total_cost"] if include_openings_b else 0.0)
        )
        cost_basement_only = {
            "total_cost": round(total_b, 2),
            "detailed_items": (
                cost_walls_b.get("detailed_items", []) +
                cost_finishes_b.get("detailed_items", []) +
                cost_floors_b.get("detailed_items", []) +
                cost_utilities_b.get("detailed_items", [])
            )
        }
        print(f"✅ [PRICING] Plan beci (dedicat): {cost_basement_only['total_cost']:,.0f} EUR (structură + finisaje interior + utilități + deschideri după nivel ofertă)")
        return {
            "total_cost_eur": round(total_b, 2),
            "total_area_m2": floor_area,
            "currency": "EUR",
            "breakdown": {
                "foundation": {"total_cost": 0.0, "detailed_items": []},
                "structure_walls": cost_walls_b,
                "floors_ceilings": cost_floors_b,
                "roof": {"total_cost": 0.0, "detailed_items": []},
                "openings": openings_breakdown_b,
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

    # Primul plan Zubau: editorul salvează *_plan_0, nu *_ground (evită coliziunea cu EG din wizard).
    use_plan_slug_keys = (
        wizard_pkg_z in ("zubau", "zubau_aufstockung")
        and int(plan_index) == 0
        and _fkz in ("zubau", "new")
        and not is_basement_plan
    )
    # Oferte vechi: datele pot fi încă pe *_ground; folosim plan_0 doar dacă există cel puțin o valoare acolo.
    if use_plan_slug_keys:
        wb = wandaufbau_data if isinstance(wandaufbau_data, dict) else {}
        bdb = frontend_input.get("bodenDeckeBelag", {}) if isinstance(frontend_input, dict) else {}
        if not isinstance(bdb, dict):
            bdb = {}
        plan0_any = (
            any(str(wb.get(x) or "").strip() for x in ("außenwande_plan_0", "innenwande_plan_0"))
            or any(
                str(mat_finisaj.get(x) or "").strip()
                for x in (
                    "finisajInterior_plan_0",
                    "finisajInteriorInnen_plan_0",
                    "finisajInteriorAussen_plan_0",
                    "fatada_plan_0",
                )
            )
            or any(
                str(bdb.get(x) or "").strip()
                for x in ("bodenaufbau_plan_0", "deckenaufbau_plan_0", "bodenbelag_plan_0")
            )
        )
        if not plan0_any:
            use_plan_slug_keys = False

    # Determinăm cheile de finisaje bazat pe tipul etajului
    if use_plan_slug_keys:
        ps = "plan_0"
        finish_int_key = f"finisajInterior_{ps}"
        finish_int_inner_key = f"finisajInteriorInnen_{ps}"
        finish_int_outer_key = f"finisajInteriorAussen_{ps}"
        finish_ext_key = f"fatada_{ps}"
    elif is_ground_floor:
        # Ground floor
        finish_int_key = "finisajInterior_ground"
        finish_int_inner_key = "finisajInteriorInnen_ground"
        finish_int_outer_key = "finisajInteriorAussen_ground"
        finish_ext_key = "fatada_ground"
    elif is_top_floor_plan:
        # Top floor - verificăm dacă este mansardă sau pod
        if is_mansarda:
            # Mansardă - folosim finisajele pentru mansardă
            finish_int_key = "finisajInteriorMansarda"
            finish_int_inner_key = "finisajInteriorInnenMansarda"
            finish_int_outer_key = "finisajInteriorAussenMansarda"
            finish_ext_key = "fatadaMansarda"
        else:
            # Pod - folosim ultimul index din totalFloors (din frontend)
            # totalFloors = 1 + etajeIntermediare, deci ultimul index este totalFloors
            # De exemplu: ground + pod → totalFloors = 1 → floor_1
            # ground + 1 intermediar + pod → totalFloors = 2 → floor_2
            # ground + 2 intermediare + pod → totalFloors = 3 → floor_3
            floor_idx = total_floors_frontend  # Ultimul index (ground=0, primul intermediar=1, pod=totalFloors)
            finish_int_key = f"finisajInterior_floor_{floor_idx}"
            finish_int_inner_key = f"finisajInteriorInnen_floor_{floor_idx}"
            finish_int_outer_key = f"finisajInteriorAussen_floor_{floor_idx}"
            finish_ext_key = f"fatada_floor_{floor_idx}"
    else:
        # Etaj intermediar - folosim intermediate_floor_index + 1 pentru floor_X
        floor_idx = intermediate_floor_index + 1  # floor_1, floor_2, etc.
        finish_int_key = f"finisajInterior_floor_{floor_idx}"
        finish_int_inner_key = f"finisajInteriorInnen_floor_{floor_idx}"
        finish_int_outer_key = f"finisajInteriorAussen_floor_{floor_idx}"
        finish_ext_key = f"fatada_floor_{floor_idx}"
    
    # Fallback la valorile vechi dacă nu există valorile noi per etaj
    finish_int = _norm_finish(
        mat_finisaj.get(finish_int_key) or mat_finisaj.get("finisajInterior", "Tencuială"),
        "Tencuială"
    )
    finish_int_inner = _norm_finish(
        mat_finisaj.get(finish_int_inner_key) or mat_finisaj.get(finish_int_key) or mat_finisaj.get("finisajInterior", "Tencuială"),
        "Tencuială"
    )
    finish_int_outer = _norm_finish(
        mat_finisaj.get(finish_int_outer_key) or mat_finisaj.get(finish_int_key) or mat_finisaj.get("finisajInterior", "Tencuială"),
        "Tencuială"
    )
    finish_ext = _norm_finish(
        mat_finisaj.get(finish_ext_key) or mat_finisaj.get("fatada", "Tencuială"), 
        "Tencuială"
    )
    
    # Determinăm eticheta etajului pentru afișare în PDF
    if use_plan_slug_keys:
        floor_label = "Zubau (Plan 1)"
    elif is_ground_floor:
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
    print(f"✅ [PRICING] Plan {plan_index} ({floor_type_str}): Final finishes - Innenwände: {finish_int_inner}, Außenwände: {finish_int_outer}, Fassade: {finish_ext}")
    print(f"   Used keys: {finish_int_inner_key}, {finish_int_outer_key}, {finish_ext_key}")
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
    
    tip_semineu = None
    
    # 4. CALCULE COMPONENTE – pereți: Wandaufbau per etaj dacă există, altfel system_constructie (legacy)
    wandaufbau_coeffs = pricing_coeffs.get("wandaufbau", {})
    _wau = (finish_ext_key or "").replace("fatada_", "außenwande_").replace("fatadaMansarda", "außenwandeMansarda")
    _win = (finish_int_key or "").replace("finisajInterior_", "innenwande_").replace("finisajInteriorMansarda", "innenwandeMansarda").replace("finisajInteriorBeci", "innenwandeBeci")
    sel_aussen = (wandaufbau_data.get(_wau) or "").strip() if _wau else ""
    sel_innen = (wandaufbau_data.get(_win) or "").strip() if _win else ""
    use_wandaufbau = wandaufbau_coeffs and (
        sel_aussen
        or sel_innen
        or wandaufbau_data.get("außenwande_ground")
        or wandaufbau_data.get("innenwande_ground")
        or wandaufbau_data.get("außenwande_plan_0")
        or wandaufbau_data.get("innenwande_plan_0")
    )
    if use_wandaufbau:
        aussen_map = wandaufbau_coeffs.get("aussen", {})
        innen_map = wandaufbau_coeffs.get("innen", {})
        if not sel_aussen:
            sel_aussen = wandaufbau_data.get("außenwande_ground") or wandaufbau_data.get("außenwandeMansarda") or ""
        if not sel_innen:
            sel_innen = wandaufbau_data.get("innenwande_ground") or wandaufbau_data.get("innenwandeMansarda") or ""
        price_aussen = float(aussen_map.get(sel_aussen, 280))
        price_innen = float(innen_map.get(sel_innen, 280))
        cost_int = w_int_net_structure * price_innen
        cost_ext = w_ext_net_structure * price_aussen
        cost_walls = {
            "total_cost": round(cost_int + cost_ext, 2),
            "detailed_items": [
                {"category": "walls_structure_int", "name": f"Pereți Interiori ({sel_innen or 'Wandaufbau'})", "material": sel_innen or "Wandaufbau", "construction_mode": "Wandaufbau", "area_m2": round(w_int_net_structure, 2), "unit_price": round(price_innen, 2), "cost": round(cost_int, 2)},
                {"category": "walls_structure_ext", "name": f"Pereți Exteriori ({sel_aussen or 'Wandaufbau'})", "material": sel_aussen or "Wandaufbau", "construction_mode": "Wandaufbau", "area_m2": round(w_ext_net_structure, 2), "unit_price": round(price_aussen, 2), "cost": round(cost_ext, 2)},
            ]
        }
    else:
        cost_walls = calculate_walls_details(
            system_coeffs, w_int_net_structure, w_ext_net_structure,
            system=system_constructie
        )
    
    cost_finishes = calculate_finishes_details(
        finish_coeffs, w_int_net_finish_inner, w_int_net_finish_outer, w_ext_net_finish,
        type_int_inner=finish_int_inner, type_int_outer=finish_int_outer, type_ext=finish_ext,
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

    # Podea / tavan: dacă există alegere Bodenaufbau, Deckenaufbau, Bodenbelag per etaj, folosim prețurile din aceste opțiuni
    bdb_data = frontend_input.get("bodenDeckeBelag", {})
    bdb_coeffs = pricing_coeffs.get("boden_decke_belag", {})
    use_boden_decke_belag = bool(
        bdb_coeffs
        and (
            bdb_data.get("bodenaufbau_ground")
            or bdb_data.get("deckenaufbau_ground")
            or bdb_data.get("bodenbelagBeci")
            or bdb_data.get("deckenaufbauBeci")
            or bdb_data.get("bodenaufbau_plan_0")
            or bdb_data.get("deckenaufbau_plan_0")
            or bdb_data.get("bodenbelag_plan_0")
        )
    )
    if use_boden_decke_belag:
        bodenaufbau_map = bdb_coeffs.get("bodenaufbau", {})
        deckenaufbau_map = bdb_coeffs.get("deckenaufbau", {})
        bodenbelag_map = bdb_coeffs.get("bodenbelag", {})
        cost_floor_str, cost_ceil_str, cost_belag = 0.0, 0.0, 0.0
        items: list = []
        if is_basement_plan:
            opt_belag = (bdb_data.get("bodenbelagBeci") or "").strip()
            if opt_belag:
                price_b = float(bodenbelag_map.get(opt_belag, 0))
                cost_belag = floor_area * price_b
                items.append({"category": "bodenbelag", "name": f"Bodenbelag Keller ({opt_belag})", "area_m2": round(floor_area, 2), "unit_price": round(price_b, 2), "cost": round(cost_belag, 2)})
            opt_decken_beci = (bdb_data.get("deckenaufbauBeci") or "").strip()
            if opt_decken_beci:
                price_d_beci = float(deckenaufbau_map.get(opt_decken_beci, 0))
                cost_ceil_str = ceiling_area * price_d_beci
                items.append({"category": "deckenaufbau", "name": f"Deckenaufbau Keller ({opt_decken_beci})", "area_m2": round(ceiling_area, 2), "unit_price": round(price_d_beci, 2), "cost": round(cost_ceil_str, 2)})
        else:
            if use_plan_slug_keys:
                suffix = "plan_0"
            elif is_ground_floor:
                suffix = "ground"
            elif is_mansarda:
                suffix = "Mansarda"
            else:
                suffix = f"floor_{intermediate_floor_index + 1}"
            # Fără beci: la parter nu cerem/calculăm structura podei (Bodenaufbau), doar pardoseală și tavan
            has_basement = tip_fundatie_beci and "Keller" in str(tip_fundatie_beci) and "Kein Keller" not in str(tip_fundatie_beci)
            skip_bodenaufbau = (suffix == "ground" and not has_basement) and not use_plan_slug_keys
            bodenaufbau_key = "bodenaufbauMansarda" if suffix == "Mansarda" else f"bodenaufbau_{suffix}"
            bodenbelag_key = "bodenbelagMansarda" if suffix == "Mansarda" else f"bodenbelag_{suffix}"
            if not skip_bodenaufbau:
                opt_boden = (bdb_data.get(bodenaufbau_key) or "").strip()
                if opt_boden:
                    price_b = float(bodenaufbau_map.get(opt_boden, 0))
                    cost_floor_str = floor_structure_area * price_b
                    items.append({"category": "bodenaufbau", "name": f"Bodenaufbau ({opt_boden})", "area_m2": round(floor_structure_area, 2), "unit_price": round(price_b, 2), "cost": round(cost_floor_str, 2)})
            opt_belag_sel = (bdb_data.get(bodenbelag_key) or "").strip()
            if opt_belag_sel:
                price_bl = float(bodenbelag_map.get(opt_belag_sel, 0))
                cost_belag = floor_area * price_bl
                items.append({"category": "bodenbelag", "name": f"Bodenbelag ({opt_belag_sel})", "area_m2": round(floor_area, 2), "unit_price": round(price_bl, 2), "cost": round(cost_belag, 2)})
            decken_key = "deckenaufbauMansarda" if suffix == "Mansarda" else f"deckenaufbau_{suffix}"
            opt_decken = (bdb_data.get(decken_key) or "").strip()
            if opt_decken:
                price_d = float(deckenaufbau_map.get(opt_decken, 0))
                cost_ceil_str = ceiling_area * price_d
                items.append({"category": "deckenaufbau", "name": f"Deckenaufbau ({opt_decken})", "area_m2": round(ceiling_area, 2), "unit_price": round(price_d, 2), "cost": round(cost_ceil_str, 2)})
        total_bdb = cost_floor_str + cost_ceil_str + cost_belag
        cost_floors_ceilings = {"total_cost": round(total_bdb, 2), "detailed_items": items} if items else calculate_floors_details(area_coeffs, floor_structure_area, ceiling_area)
    else:
        cost_floors_ceilings = calculate_floors_details(
            area_coeffs, floor_structure_area, ceiling_area
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
    
    include_electricity = _as_bool(_fv("include_electricity"), True)
    include_sewage = _as_bool(_fv("include_sewage"), True)
    electricity_coeffs_eff = dict(electricity_coeffs)
    if not include_electricity:
        electricity_coeffs_eff["coefficient_electricity_per_m2"] = 0.0

    cost_utilities = calculate_utilities_details(
        electricity_coeffs_eff,
        heating_coeffs,
        ventilation_coeffs,
        sewage_coeffs,
        total_floor_area_m2=floor_area,
        energy_level=energy_level,
        heating_type=heating_type,
        has_ventilation=has_ventilation,
        has_sewage=include_sewage
    )

    stairs_type_selected = str(_fv("stairs_type") or (frontend_input.get("ferestreUsi", {}) or {}).get("treppeTyp") or "Standard").strip()
    stairs_floors_count = frontend_input.get("_stairs_floors_count")
    stairs_total_count = frontend_input.get("_stairs_total_count")
    if is_ground_floor:
        cost_stairs = calculate_stairs_details(
            stairs_coeffs,
            total_floors,
            stairs_floors_count=stairs_floors_count if isinstance(stairs_floors_count, int) else None,
            stair_type=stairs_type_selected,
            stairs_total_count=stairs_total_count if isinstance(stairs_total_count, int) else None,
        )
    else:
        cost_stairs = {"total_cost": 0.0, "detailed_items": []}

    sc_lift = (frontend_input.get("structuraCladirii", {}) or {})
    editor_lift = bool(frontend_input.get("_lift_present"))
    form_lift = bool(
        _as_bool(_fv("lift_present"), _as_bool(sc_lift.get("aufzugVorhanden"), False))
    )
    lift_present = editor_lift or form_lift
    lift_type_selected = str(_fv("lift_type") or (frontend_input.get("structuraCladirii", {}) or {}).get("aufzugTyp") or "Hydraulikaufzug").strip()
    lift_price_map = stairs_coeffs.get("lift_type_price_map") or {}
    if is_ground_floor and lift_present:
        lift_unit = float(lift_price_map.get(lift_type_selected, 0) or 0)
        cost_lift = {
            "total_cost": round(lift_unit, 2),
            "detailed_items": [{
                "category": "lift",
                "name": f"Aufzug ({lift_type_selected})",
                "quantity": 1,
                "unit": "Stück",
                "unit_price": round(lift_unit, 2),
                "cost": round(lift_unit, 2),
            }],
        }
    else:
        cost_lift = {"total_cost": 0.0, "detailed_items": []}

    pillar_volume_m3 = float(frontend_input.get("_pillar_volume_m3") or 0.0)
    pillar_type = str(_fv("pillar_type") or (frontend_input.get("structuraCladirii", {}) or {}).get("pilonType") or "Stahlbeton").strip()
    pillar_price_map = area_coeffs.get("pillar_type_price_per_m3") or {}
    if is_ground_floor and pillar_volume_m3 > 0:
        pillar_unit = float(pillar_price_map.get(pillar_type, 0) or 0)
        cost_pillars = {
            "total_cost": round(pillar_volume_m3 * pillar_unit, 2),
            "detailed_items": [{
                "category": "pillars",
                "name": f"Pilonen ({pillar_type})",
                "quantity": round(pillar_volume_m3, 3),
                "unit": "m3",
                "unit_price": round(pillar_unit, 2),
                "cost": round(pillar_volume_m3 * pillar_unit, 2),
            }],
        }
    else:
        cost_pillars = {"total_cost": 0.0, "detailed_items": []}
    
    # Fireplace/Kamin is deprecated and excluded from pricing.
    cost_fireplace = {"total_cost": 0.0, "detailed_items": []}

    # 5. TOTAL – Nivel ofertă (tag: offer_scope) decide CE includem, nu coeficient
    nivel_oferta_raw = (str(_fv("offer_scope") or sist_constr.get("nivelOferta") or mat_finisaj.get("nivelOferta") or "").strip())
    # Normalizare: RO/EN/altă limbă → chei DE pentru comparație
    _nivel_map = {
        "Structură": "Rohbau/Tragwerk",
        "Structură + ferestre": "Tragwerk + Fenster",
        "Casă completă": "Schlüsselfertiges Haus",
        "Schlüsselfertig Haus": "Schlüsselfertiges Haus",
    }
    nivel_oferta = _nivel_map.get(nivel_oferta_raw, nivel_oferta_raw)
    if nivel_oferta == nivel_oferta_raw:
        nl = nivel_oferta.lower()
        if ("rohbau" in nl or "tragwerk" in nl) and "fenster" not in nl:
            nivel_oferta = "Rohbau/Tragwerk"
        elif "tragwerk" in nl and "fenster" in nl:
            nivel_oferta = "Tragwerk + Fenster"
        elif "schlüsselfertig" in nl or "schlusselfertig" in nl or "schlüsselfertig haus" in nl:
            nivel_oferta = "Schlüsselfertiges Haus"
    # Rohbau = doar structură; Tragwerk+Fenster = + deschideri; Schlüsselfertig sau necunoscut = tot
    include_openings = nivel_oferta != "Rohbau/Tragwerk"
    include_finishes = nivel_oferta not in ("Rohbau/Tragwerk", "Tragwerk + Fenster")
    include_utilities = nivel_oferta not in ("Rohbau/Tragwerk", "Tragwerk + Fenster")
    include_stairs = nivel_oferta not in ("Rohbau/Tragwerk", "Tragwerk + Fenster")
    include_pillars = True
    # Lift is structural (like pillars); include in Rohbau/Tragwerk when detection/form says so.
    include_lift = include_pillars
    include_fireplace = False

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
        (cost_lift["total_cost"] if include_lift else 0.0) +
        (cost_pillars["total_cost"] if include_pillars else 0.0) +
        (cost_fireplace["total_cost"] if include_fireplace else 0.0)
    )
    total_plan_cost = round(total_plan_cost + float(cost_zubau_wall_demolition_extra or 0.0), 2)

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
        floor_area_basement = floor_structure_area * coeff_floors
        ceiling_area_basement = ceiling_area * coeff_floors
        
        print(f"   📐 [BASEMENT] Coeficienți aplicați: pereți={coeff_walls:.0%}, podele={coeff_floors:.0%}")
        print(f"      - Pereți structură: {w_int_net_structure:.2f} m² × {coeff_walls:.0%} = {w_int_net_structure_basement:.2f} m²")
        print(f"      - Pereți finisaje: {w_int_net_finish:.2f} m² × {coeff_walls:.0%} = {w_int_net_finish_basement:.2f} m²")
        print(f"      - Podele: {floor_area:.2f} m² × {coeff_floors:.0%} = {floor_area_basement:.2f} m²")
        
        # Pereți interiori + exteriori pentru basement (structură); finisaje doar interior
        _wb_innen = (wandaufbau_data.get("innenwandeBeci") or "").strip()
        _wb_aussen = (wandaufbau_data.get("außenwandeBeci") or "").strip()
        w_ext_net_basement = w_ext_net_structure * coeff_walls
        if use_wandaufbau and _wb_innen:
            _p_innen = float(wandaufbau_coeffs.get("innen", {}).get(_wb_innen, 280))
            _c_int_b = w_int_net_structure_basement * _p_innen
            _c_ext_b = 0.0
            items_b = [{"category": "walls_structure_int", "name": f"Pereți Interiori Beci ({_wb_innen})", "area_m2": round(w_int_net_structure_basement, 2), "unit_price": round(_p_innen, 2), "cost": round(_c_int_b, 2)}]
            if w_ext_net_basement > 0:
                if _wb_aussen:
                    _p_aussen = float(wandaufbau_coeffs.get("aussen", {}).get(_wb_aussen, 280))
                    _c_ext_b = w_ext_net_basement * _p_aussen
                    items_b.append({"category": "walls_structure_ext", "name": f"Pereți Exteriori Beci ({_wb_aussen})", "area_m2": round(w_ext_net_basement, 2), "unit_price": round(_p_aussen, 2), "cost": round(_c_ext_b, 2)})
                else:
                    _p_ext_sys = float(system_coeffs.get("base_unit_prices", {}).get(system_constructie, {}).get("exterior", 280))
                    _c_ext_b = w_ext_net_basement * _p_ext_sys
                    items_b.append({"category": "walls_structure_ext", "name": f"Pereți Exteriori Beci ({system_constructie})", "area_m2": round(w_ext_net_basement, 2), "unit_price": round(_p_ext_sys, 2), "cost": round(_c_ext_b, 2)})
            cost_walls_basement = {"total_cost": round(_c_int_b + _c_ext_b, 2), "detailed_items": items_b}
        elif use_wandaufbau and _wb_aussen and w_ext_net_basement > 0:
            _p_aussen = float(wandaufbau_coeffs.get("aussen", {}).get(_wb_aussen, 280))
            _p_innen = float(system_coeffs.get("base_unit_prices", {}).get(system_constructie, {}).get("interior", 280))
            _c_int_b = w_int_net_structure_basement * _p_innen
            _c_ext_b = w_ext_net_basement * _p_aussen
            cost_walls_basement = {"total_cost": round(_c_int_b + _c_ext_b, 2), "detailed_items": [
                {"category": "walls_structure_int", "name": f"Pereți Interiori Beci ({system_constructie})", "area_m2": round(w_int_net_structure_basement, 2), "unit_price": round(_p_innen, 2), "cost": round(_c_int_b, 2)},
                {"category": "walls_structure_ext", "name": f"Pereți Exteriori Beci ({_wb_aussen})", "area_m2": round(w_ext_net_basement, 2), "unit_price": round(_p_aussen, 2), "cost": round(_c_ext_b, 2)},
            ]}
        else:
            cost_walls_basement = calculate_walls_details(
                system_coeffs, w_int_net_structure_basement, w_ext_net_basement,
                system=system_constructie
            )
        
        # Finisaje interioare pentru basement (folosim ariile calculate cu coeficienții)
        cost_finishes_basement = calculate_finishes_details(
            finish_coeffs, w_int_net_finish_basement, 0.0, 0.0,  # Nu avem fațadă pentru basement
            type_int_inner=finish_int_beci, type_int_outer=finish_int_beci, type_ext="Tencuială",  # Nu contează type_ext pentru basement
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
            electricity_coeffs_eff_basement = dict(electricity_coeffs)
            if not include_electricity:
                electricity_coeffs_eff_basement["coefficient_electricity_per_m2"] = 0.0
            cost_utilities_basement = calculate_utilities_details(
                electricity_coeffs_eff_basement,
                heating_coeffs,
                ventilation_coeffs,
                sewage_coeffs,
                total_floor_area_m2=floor_area_basement,  # Folosim suprafața calculată pentru beci
                energy_level=energy_level,
                heating_type=heating_type,
                has_ventilation=has_ventilation,
                has_sewage=include_sewage
            )
            print(f"      ✅ Utilități beci locuibil: {cost_utilities_basement['total_cost']:,.0f} EUR (arie: {floor_area_basement:.2f} m²)")
        else:
            # Beci nelocuibil: doar curent + canalizare (fără încălzire și ventilație)
            elec_base = float(electricity_coeffs.get("coefficient_electricity_per_m2", 60.0)) if include_electricity else 0.0
            elec_modifiers = electricity_coeffs.get("energy_performance_modifiers", {})
            elec_modifier = float(elec_modifiers.get(energy_level, 1.0))
            elec_cost = floor_area_basement * elec_base * elec_modifier
            
            sewage_base = float(sewage_coeffs.get("coefficient_sewage_per_m2", 45.0)) if include_sewage else 0.0
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

    # Balkon: Raster extensions_measurements (Boden). Wintergarten = cameră normală (fără poziții separate).
    cost_wintergaerten_balkone = {"total_cost": 0.0, "detailed_items": []}
    extm = area_data.get("extensions_measurements") if isinstance(area_data, dict) else None
    if isinstance(extm, dict) and extm:
        try:
            bfloor = float(extm.get("balkon_floor_m2") or 0.0)
            ext_cfg = pricing_coeffs.get("extensions") or {}
            bf_map = ext_cfg.get("balkon_floor_price_per_m2") or {}
            sc = frontend_input.get("structuraCladirii") if isinstance(frontend_input.get("structuraCladirii"), dict) else {}
            bdeck = str(
                _fv("balkon_boden") or sc.get("balkonBoden") or frontend_input.get("balkonBoden") or "Holz"
            ).strip()
            bprice = float(bf_map.get(bdeck, next(iter(bf_map.values()), 95.0)) if bf_map else 95.0)
            items_bw: list[dict] = []
            tot_bw = 0.0
            if bfloor > 0 and bprice > 0:
                c = round(bfloor * bprice, 2)
                tot_bw += c
                items_bw.append(
                    {
                        "category": "balkon_floor",
                        "name": f"Balkonboden ({bdeck})",
                        "area_m2": round(bfloor, 2),
                        "unit_price": bprice,
                        "total_cost": c,
                    }
                )
            if items_bw:
                cost_wintergaerten_balkone = {"total_cost": round(tot_bw, 2), "detailed_items": items_bw}
        except Exception as _ex:
            print(f"      ⚠️ [PRICING] extensions Balkon: {_ex}")

    if include_finishes and float(cost_wintergaerten_balkone.get("total_cost") or 0.0) > 0:
        total_plan_cost = round(total_plan_cost + float(cost_wintergaerten_balkone["total_cost"]), 2)

    # Aufstockung Phase 1 (Bestand): per-floor Rückbau / Treppenöffnung / Statik.
    cost_aufstockung_phase1 = {"total_cost": 0.0, "detailed_items": []}
    phase1_data = frontend_input.get("aufstockungPhase1", {}) if isinstance(frontend_input, dict) else {}
    wizard_package = str(frontend_input.get("wizard_package") or "").strip().lower() if isinstance(frontend_input, dict) else ""
    aufstockung_floor_kinds = frontend_input.get("aufstockungFloorKinds") if isinstance(frontend_input.get("aufstockungFloorKinds"), list) else []
    phase_floor = None
    def _as_int(v, default=-1):
        try:
            return int(v)
        except Exception:
            return default

    if isinstance(phase1_data, dict):
        for entry in (phase1_data.get("existingFloors") or []):
            if isinstance(entry, dict) and _as_int(entry.get("plan_index", -1)) == _as_int(plan_index):
                phase_floor = entry
                break
        if phase_floor is None:
            for entry in (phase1_data.get("newFloors") or []):
                if isinstance(entry, dict) and _as_int(entry.get("plan_index", -1)) == _as_int(plan_index):
                    phase_floor = entry
                    break
    floor_kind = str((phase_floor or {}).get("floorKind") or (aufstockung_floor_kinds[plan_index] if plan_index < len(aufstockung_floor_kinds) else "")).strip().lower()
    # Aufstockung: dacă există orice tip de Keller în structura clădirii, beciul este mereu tratat ca Bestand.
    tip_fundatie_beci_for_kind = str((frontend_input.get("structuraCladirii", {}) or {}).get("tipFundatieBeci") or "")
    has_keller_in_structure = "Keller" in tip_fundatie_beci_for_kind and "Kein Keller" not in tip_fundatie_beci_for_kind
    if wizard_package in ("aufstockung", "zubau", "zubau_aufstockung") and is_basement_plan and has_keller_in_structure:
        floor_kind = "bestand"
    floor_kind_norm = str(floor_kind).strip().lower()
    is_unified_offer = wizard_package in ("aufstockung", "zubau", "zubau_aufstockung")
    is_existing_aufstockung_floor = is_unified_offer and floor_kind_norm not in ("new", "zubau", "aufstockung")
    is_zubau_floor = is_unified_offer and floor_kind_norm in ("new", "zubau")
    global_combined_price = (
        float(phase1_data.get("globalCombinedPrice"))
        if isinstance(phase1_data, dict) and isinstance(phase1_data.get("globalCombinedPrice"), (int, float))
        else None
    )
    use_global_combined_price = global_combined_price is not None and global_combined_price >= 0

    if is_existing_aufstockung_floor and isinstance(phase_floor, dict):
        raw_params = pricing_coeffs.get("_raw_params", {}) or {}
        items_phase1 = []
        demolition_items = phase_floor.get("demolitionSelections", []) or []
        custom_demolition_price = phase_floor.get("customDemolitionPrice")
        custom_demolition_total = (
            float(custom_demolition_price)
            if isinstance(custom_demolition_price, (int, float)) and float(custom_demolition_price) >= 0
            else None
        )
        if use_global_combined_price:
            custom_demolition_total = None
        # Rows with positive area from editor; demolition price is entered directly by user.
        demo_rows: list[tuple[int, float]] = []
        for idx, sel in enumerate(demolition_items):
            if not isinstance(sel, dict):
                continue
            area_m2 = float(sel.get("area_m2") or 0.0)
            if area_m2 <= 0:
                continue
            demo_rows.append((idx, area_m2))

        # Abbruch-Eigenpreis: editor value is a lump-sum EUR for all Aufstandsflächen on this floor (split by area).
        if custom_demolition_total is not None and demo_rows:
            sum_area = sum(a for _, a in demo_rows)
            for idx, area_m2 in demo_rows:
                alloc = custom_demolition_total * (area_m2 / sum_area) if sum_area > 0 else 0.0
                eff_unit = (alloc / area_m2) if area_m2 > 0 else 0.0
                items_phase1.append({
                    "category": "aufstockung_demolition",
                    "name": f"Aufstandsfläche / Abreißung #{idx + 1}",
                    "area_m2": round(area_m2, 2),
                    "unit_price": round(eff_unit, 4),
                    "total_cost": round(alloc, 2),
                })
        stair_cfg = pricing_coeffs.get("stairs", {}) if isinstance(pricing_coeffs.get("stairs"), dict) else {}
        stair_map = stair_cfg.get("stair_type_price_map", {}) if isinstance(stair_cfg.get("stair_type_price_map"), dict) else {}
        stair_default = float(stair_cfg.get("price_per_stair_unit", raw_params.get("price_per_stair_unit", 0.0)) or 0.0)
        treppe_typ = str(
            (frontend_input.get("structuraCladirii", {}) or {}).get("treppeTyp")
            or frontend_input.get("treppeTyp")
            or "Standard"
        ).strip()
        unit_piece = float(stair_map.get(treppe_typ, stair_default) or stair_default)

        stair_items = phase_floor.get("stairOpenings", []) or []
        for idx, sel in enumerate(stair_items):
            if not isinstance(sel, dict):
                continue
            area_m2_stair = float(sel.get("area_m2") or 0.0)
            if area_m2_stair <= 0:
                continue
            ow = sel.get("opening_width_m")
            ol = sel.get("opening_length_m")
            w_m = float(ow) if isinstance(ow, (int, float)) else None
            l_m = float(ol) if isinstance(ol, (int, float)) else None
            if w_m is None or l_m is None or w_m <= 0 or l_m <= 0:
                w_m, l_m = _stair_opening_dims_m_from_bbox_and_area(sel.get("bbox"), area_m2_stair)
            cost = unit_piece
            items_phase1.append({
                "category": "aufstockung_stair_opening",
                "name": f"Treppenöffnung #{idx + 1}",
                "quantity": 1.0,
                "unit_price": round(unit_piece, 2),
                "total_cost": round(cost, 2),
                "opening_area_m2": round(area_m2_stair, 2),
                "opening_width_m": round(float(w_m), 3) if w_m is not None and w_m > 0 else None,
                "opening_length_m": round(float(l_m), 3) if l_m is not None and l_m > 0 else None,
            })

        statik = phase_floor.get("statikChoice", {}) or {}
        if isinstance(statik, dict) and not use_global_combined_price:
            custom_price = float(statik.get("customPiecePrice") or 0.0)
            if custom_price > 0:
                items_phase1.append({
                    "category": "aufstockung_statik",
                    "name": "Statik (Editor)",
                    "quantity": 1,
                    "unit_price": round(custom_price, 2),
                    "total_cost": round(custom_price, 2),
                })

        if items_phase1:
            cost_aufstockung_phase1 = {
                "total_cost": round(sum(float(it.get("total_cost") or 0.0) for it in items_phase1), 2),
                "detailed_items": items_phase1,
            }
            total_plan_cost += cost_aufstockung_phase1["total_cost"]

    if (
        is_zubau_floor
        and not is_existing_aufstockung_floor
        and cost_zubau_wall_demolition_items
    ):
        prev_items = list(cost_aufstockung_phase1.get("detailed_items") or [])
        prev_total = float(cost_aufstockung_phase1.get("total_cost") or 0.0)
        add_t = round(sum(float(x.get("total_cost") or 0.0) for x in cost_zubau_wall_demolition_items), 2)
        cost_aufstockung_phase1 = {
            "total_cost": round(prev_total + add_t, 2),
            "detailed_items": prev_items + cost_zubau_wall_demolition_items,
        }

    if (
        is_zubau_floor
        and not is_existing_aufstockung_floor
        and isinstance(phase_floor, dict)
        and not cost_zubau_wall_demolition_items
        and not use_global_combined_price
    ):
        cdp = phase_floor.get("customDemolitionPrice")
        if cdp is not None:
            try:
                cdp_f = float(cdp)
                if cdp_f > 0:
                    prev_items = list(cost_aufstockung_phase1.get("detailed_items") or [])
                    prev_total = float(cost_aufstockung_phase1.get("total_cost") or 0.0)
                    prev_items.append({
                        "category": "zubau_demolition_pauschal",
                        "name": "Abbruch (Pauschal)",
                        "quantity": 1.0,
                        "unit_price": round(cdp_f, 2),
                        "total_cost": round(cdp_f, 2),
                    })
                    cost_aufstockung_phase1 = {
                        "total_cost": round(prev_total + cdp_f, 2),
                        "detailed_items": prev_items,
                    }
                    total_plan_cost = round(total_plan_cost + cdp_f, 2)
            except (TypeError, ValueError):
                pass

    if use_global_combined_price and plan_index == 0:
        prev_items = list(cost_aufstockung_phase1.get("detailed_items") or [])
        prev_total = float(cost_aufstockung_phase1.get("total_cost") or 0.0)
        prev_items.append({
            "category": "aufstockung_global_combined",
            "name": "Abbruch + Statik (global)",
            "quantity": 1.0,
            "unit_price": round(float(global_combined_price or 0.0), 2),
            "total_cost": round(float(global_combined_price or 0.0), 2),
        })
        cost_aufstockung_phase1 = {
            "total_cost": round(prev_total + float(global_combined_price or 0.0), 2),
            "detailed_items": prev_items,
        }
        total_plan_cost = round(total_plan_cost + float(global_combined_price or 0.0), 2)

    if is_existing_aufstockung_floor:
        # Existing-floor Aufstockung pricing: keep only roof + phase1 editor-driven items.
        cost_foundation = {"total_cost": 0.0, "detailed_items": []}
        cost_walls = {"total_cost": 0.0, "detailed_items": []}
        cost_floors_ceilings = {"total_cost": 0.0, "detailed_items": []}
        cost_openings = {"total_cost": 0.0, "detailed_items": []}
        cost_finishes = {"total_cost": 0.0, "detailed_items": []}
        cost_utilities = {"total_cost": 0.0, "detailed_items": []}
        cost_stairs = {"total_cost": 0.0, "detailed_items": []}
        cost_fireplace = {"total_cost": 0.0, "detailed_items": []}
        cost_basement = {"total_cost": 0.0, "detailed_items": []}
        cost_wintergaerten_balkone = {"total_cost": 0.0, "detailed_items": []}
        total_plan_cost = round(
            float(cost_roof.get("total_cost") or 0.0) + float(cost_aufstockung_phase1.get("total_cost") or 0.0),
            2,
        )

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
            "lift": _zero_if_excluded(cost_lift, include_lift),
            "pillars": _zero_if_excluded(cost_pillars, include_pillars),
            "fireplace": _zero_if_excluded(cost_fireplace, include_fireplace),
            "basement": cost_basement,
            "wintergaerten_balkone": _zero_if_excluded(cost_wintergaerten_balkone, include_finishes),
            "aufstockung_phase1": cost_aufstockung_phase1,
        }
    }