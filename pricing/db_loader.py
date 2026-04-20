# pricing/db_loader.py
from common.supabase_client import get_supabase_client

def fetch_pricing_parameters(tenant_slug: str, calc_mode: str | None = None) -> dict:
    client = get_supabase_client()
    
    tenant_res = client.table("tenants").select("id").eq("slug", tenant_slug).single().execute()
    if not tenant_res.data:
        raise ValueError(f"Tenant '{tenant_slug}' not found")
    tenant_id = tenant_res.data["id"]

    # Prefer mode-specific parameters when the schema supports it.
    # If the column doesn't exist, fall back to tenant-only parameters.
    params_query = client.table("pricing_parameters").select("*").eq("tenant_id", tenant_id)
    if calc_mode:
        try:
            params_query = params_query.eq("calc_mode", calc_mode)
        except Exception:
            # Supabase client may not error until execute(); keep fallback below.
            pass

    try:
        params_res = params_query.execute()
        # If mode filter returns no rows, fall back to tenant-only.
        if calc_mode and not params_res.data:
            params_res = client.table("pricing_parameters").select("*").eq("tenant_id", tenant_id).execute()
    except Exception:
        params_res = client.table("pricing_parameters").select("*").eq("tenant_id", tenant_id).execute()

    data_map = {row["key"]: float(row["value"]) for row in params_res.data}

    # Aufstockung Phase 1: chei noi pot lipsi din `pricing_parameters` pentru tenanți vechi (niciodată
    # „Speichern” pe cardul Preisdatenbank după adăugarea variabilelor în schema JSON). Aliniat la
    # `holzbau-form-steps.json` → preisdatenbank.sections (fieldTag aufstockung_phase1).
    _aufstockung_raw_defaults: dict[str, float] = {
        "aufstockung_demolition_roof_basic_m2": 85.0,
        "aufstockung_demolition_roof_complex_m2": 120.0,
        "aufstockung_demolition_roof_special_m2": 160.0,
        "aufstockung_stair_opening_piece": 2800.0,
        "aufstockung_stair_opening_m2": 320.0,
        "aufstockung_statik_stahlbetonverbunddecke_m2": 145.0,
    }
    for _k, _v in _aufstockung_raw_defaults.items():
        if _k not in data_map:
            data_map[_k] = _v
    _summary_percent_defaults: dict[str, float] = {
        "baustelleneinrichtung_standard_percent": 3.0,
        "baustelleneinrichtung_erschwert_percent": 4.0,
        "baustelleneinrichtung_sondertransport_percent": 3.0,
        "profit_margin_betrieb_percent": 4.0,
        "profit_margin_risiko_percent": 2.0,
        "profit_margin_unternehmer_percent": 3.0,
    }
    for _k, _v in _summary_percent_defaults.items():
        if _k not in data_map:
            data_map[_k] = _v

    # Baustellenzufahrt (accesSantier): toate cele 3 opțiuni din form influențează prețul
    _elec_base = float(data_map.get("electricity_base_price", 60.0))
    _heat_base = float(data_map.get("heating_base_price", 70.0))

    _nutz = data_map.get("unit_price_keller_nutzkeller", 145)
    _k_ausbau = data_map.get("unit_price_keller_ausbau", 185)
    out = {
        "_raw_params": data_map,
        "foundation": {
            "unit_price_per_m2": {
                # Form: Untergeschoss / Fundament (neu + Legacy gespeicherte Angebote)
                "Kein Keller (nur Bodenplatte)": data_map.get("unit_price_placa", 120),
                "Keller (ohne Ausbau)": _nutz,
                "Keller (unbeheizt / Nutzkeller) (ohne Ausbau)": _nutz,
                "Keller (mit Ausbau)": _k_ausbau,
                "Keller (unbeheizt / Nutzkeller)": _nutz,
                "Keller (mit einfachem Ausbau)": _k_ausbau,
                # Legacy / Pfahlgründung (când pilons=True se aplică în plus)
                "Placă": data_map.get("unit_price_placa", 120),
                "Piloți": data_map.get("unit_price_piloti", 180),
                "Soclu": data_map.get("unit_price_soclu", 95),
            }
        },
        "system": {
            "base_unit_prices": {
                "Blockbau": { "interior": data_map.get("clt_interior_price", 0), "exterior": data_map.get("clt_exterior_price", 0) },
                "CLT": { "interior": data_map.get("clt_interior_price", 0), "exterior": data_map.get("clt_exterior_price", 0) },
                "CLT Premium": { "interior": data_map.get("clt_interior_price", 0), "exterior": data_map.get("clt_exterior_price", 0) },
                "Holzrahmen": { "interior": data_map.get("holzrahmen_interior_price", 0), "exterior": data_map.get("holzrahmen_exterior_price", 0) },
                "HOLZRAHMEN": { "interior": data_map.get("holzrahmen_interior_price", 0), "exterior": data_map.get("holzrahmen_exterior_price", 0) },
                "Holzrahmen Standard": { "interior": data_map.get("holzrahmen_interior_price", 0), "exterior": data_map.get("holzrahmen_exterior_price", 0) },
                "Massivholz": { "interior": data_map.get("massivholz_interior_price", 0), "exterior": data_map.get("massivholz_exterior_price", 0) },
                "MASSIVHOLZ": { "interior": data_map.get("massivholz_interior_price", 0), "exterior": data_map.get("massivholz_exterior_price", 0) }
            },
        },
        # Acces șantier și teren: factori cu care se înmulțește prețul întreg al structurii (fundație + pereți + planșeu + acoperiș)
        "sistem_constructiv": {
            "acces_santier_factor": {
                "Leicht (LKW 40t)": data_map.get("acces_santier_leicht_factor", 1.0),
                "Mittel": data_map.get("acces_santier_mittel_factor", 1.1),
                "Schwierig": data_map.get("prefab_modifier_santier", 1.25),
            },
            "teren_factor": {
                "Eben": data_map.get("teren_eben_factor", 1.0),
                "Leichte Hanglage": data_map.get("teren_leichte_hanglage_factor", 1.05),
                "Starke Hanglage": data_map.get("teren_starke_hanglage_factor", 1.15),
            },
            "utilitati_anschluss_price": data_map.get("utilitati_anschluss_price", 2500),
        },
        "roof": {
            "overhang_m": data_map.get("overhang_m", 0.4),
            "sheet_metal_price_per_m": data_map.get("sheet_metal_price_per_m", 0),
            "insulation_price_per_m2": data_map.get("insulation_price_per_m2", 0),
            "tile_price_per_m2": data_map.get("tile_price_per_m2", 0),
            "metal_price_per_m2": data_map.get("metal_price_per_m2", 0),
            "membrane_price_per_m2": data_map.get("membrane_price_per_m2", 0),
            # Dämmung (form: daemmung)
            "daemmung_keine_price": data_map.get("roofonly_daemmung_keine_price", data_map.get("daemmung_keine_price", 0)),
            "daemmung_zwischensparren_price": data_map.get("roofonly_daemmung_zwischensparren_price", data_map.get("daemmung_zwischensparren_price", 55)),
            "daemmung_aufsparren_price": data_map.get("roofonly_daemmung_aufsparren_price", data_map.get("daemmung_aufsparren_price", 75)),
            # Unterdach (form: unterdach)
            "unterdach_folie_price": data_map.get("roofonly_unterdach_folie_price", data_map.get("unterdach_folie_price", 12)),
            "unterdach_schalung_folie_price": data_map.get("roofonly_unterdach_schalung_folie_price", data_map.get("unterdach_schalung_folie_price", 28)),
            # Dachstuhl-Typ (form: dachstuhlTyp)
            "dachstuhl_sparrendach_price": data_map.get("roofonly_dachstuhl_sparrendach_price", data_map.get("dachstuhl_sparrendach_price", 95)),
            "dachstuhl_pfettendach_price": data_map.get("roofonly_dachstuhl_pfettendach_price", data_map.get("dachstuhl_pfettendach_price", 110)),
            "dachstuhl_kehlbalkendach_price": data_map.get("roofonly_dachstuhl_kehlbalkendach_price", data_map.get("dachstuhl_kehlbalkendach_price", 105)),
            "dachstuhl_sonderkonstruktion_price": data_map.get("roofonly_dachstuhl_sonderkonstruktion_price", data_map.get("dachstuhl_sonderkonstruktion_price", 130)),
            "sichtdachstuhl_zuschlag_price": data_map.get("roofonly_sichtdachstuhl_price", data_map.get("sichtdachstuhl_zuschlag_price", 25)),
            "panta_acoperis_zuschlag_per_grad": data_map.get("panta_acoperis_zuschlag_per_grad", 0.5),
            # Sadiki-specific roof cover prices (if present in DB)
            "roof_shingle_price_per_m2": data_map.get("roof_shingle_price_per_m2", 0),
            "roof_metal_tile_price_per_m2": data_map.get("roof_metal_tile_price_per_m2", 0),
            "roof_ceramic_tile_price_per_m2": data_map.get("roof_ceramic_tile_price_per_m2", 0),
            "roof_tpo_pvc_price_per_m2": data_map.get("roof_tpo_pvc_price_per_m2", 0),
            "roof_green_extensive_price_per_m2": data_map.get("roof_green_extensive_price_per_m2", 0),
            # Roof-only (Dachstuhl package) pricing variables
            "roofonly_leistung_abbund_percent": data_map.get("roofonly_leistung_abbund_percent", 8),
            "roofonly_leistung_lieferung_percent": data_map.get("roofonly_leistung_lieferung_percent", 4),
            "roofonly_leistung_montage_percent": data_map.get("roofonly_leistung_montage_percent", 16),
            "roofonly_leistung_kranarbeiten_percent": data_map.get("roofonly_leistung_kranarbeiten_percent", 6),
            "roofonly_leistung_geruest_percent": data_map.get("roofonly_leistung_geruest_percent", 5),
            "roofonly_leistung_entsorgung_percent": data_map.get("roofonly_leistung_entsorgung_percent", 3),
            "roofonly_daemmung_keine_price": data_map.get("roofonly_daemmung_keine_price", 0),
            "roofonly_daemmung_zwischensparren_price": data_map.get("roofonly_daemmung_zwischensparren_price", 62),
            "roofonly_daemmung_aufsparren_price": data_map.get("roofonly_daemmung_aufsparren_price", 92),
            "roofonly_daemmung_kombination_price": data_map.get("roofonly_daemmung_kombination_price", 108),
            "roofonly_unterdach_folie_price": data_map.get("roofonly_unterdach_folie_price", 14),
            "roofonly_unterdach_schalung_folie_price": data_map.get("roofonly_unterdach_schalung_folie_price", 32),
            "roofonly_dachstuhl_sparrendach_price": data_map.get("roofonly_dachstuhl_sparrendach_price", 96),
            "roofonly_dachstuhl_pfettendach_price": data_map.get("roofonly_dachstuhl_pfettendach_price", 114),
            "roofonly_dachstuhl_kehlbalkendach_price": data_map.get("roofonly_dachstuhl_kehlbalkendach_price", 108),
            "roofonly_dachstuhl_sonderkonstruktion_price": data_map.get("roofonly_dachstuhl_sonderkonstruktion_price", 138),
            "roofonly_sichtdachstuhl_price": data_map.get("roofonly_sichtdachstuhl_price", 36),
            "roofonly_dachdeckung_ziegel_price": data_map.get("roofonly_dachdeckung_ziegel_price", 84),
            "roofonly_dachdeckung_betonstein_price": data_map.get("roofonly_dachdeckung_betonstein_price", 74),
            "roofonly_dachdeckung_blech_price": data_map.get("roofonly_dachdeckung_blech_price", 69),
            "roofonly_dachdeckung_schindel_price": data_map.get("roofonly_dachdeckung_schindel_price", 89),
            "roofonly_dachdeckung_sonstiges_price": data_map.get("roofonly_dachdeckung_sonstiges_price", 78),
            "roofonly_decken_innenausbau_standard_price": data_map.get("roofonly_decken_innenausbau_standard_price", 38),
            "roofonly_decken_innenausbau_premium_price": data_map.get("roofonly_decken_innenausbau_premium_price", 56),
            "roofonly_decken_innenausbau_exklusiv_price": data_map.get("roofonly_decken_innenausbau_exklusiv_price", 74),
            "roofonly_tinichigerie_percent": data_map.get("roofonly_tinichigerie_percent", 5),
            "tinichigerie_percent": data_map.get("roofonly_tinichigerie_percent", data_map.get("tinichigerie_percent", 5)),
            # Dachfenster (Stück) – Vollhaus
            "dachfenster_stueck_standard": data_map.get("dachfenster_stueck_standard", 650),
            "dachfenster_stueck_velux": data_map.get("dachfenster_stueck_velux", 890),
            "dachfenster_stueck_roto": data_map.get("dachfenster_stueck_roto", 820),
            "dachfenster_stueck_fakro": data_map.get("dachfenster_stueck_fakro", 850),
            "dachfenster_stueck_sonstiges": data_map.get("dachfenster_stueck_sonstiges", 750),
            # Dachfenster (Stück) – roof-only
            "roofonly_dachfenster_stueck_standard": data_map.get("roofonly_dachfenster_stueck_standard", 650),
            "roofonly_dachfenster_stueck_velux": data_map.get("roofonly_dachfenster_stueck_velux", 890),
            "roofonly_dachfenster_stueck_roto": data_map.get("roofonly_dachfenster_stueck_roto", 820),
            "roofonly_dachfenster_stueck_fakro": data_map.get("roofonly_dachfenster_stueck_fakro", 850),
            "roofonly_dachfenster_stueck_sonstiges": data_map.get("roofonly_dachfenster_stueck_sonstiges", 750),
            # Dachfenster €/m² (Fläche aus Editor: Breite×Länge); wenn 0 → Fallback Stückpreis
            "dachfenster_m2_price": float(data_map.get("dachfenster", 0) or 0),
        },
        "finishes": {
            "interior_inner": {
                "Tencuială": data_map.get("interior_tencuiala", 0),
                "Lemn": data_map.get("interior_lemn", 0),
                "Fibrociment": data_map.get("interior_fibrociment", 0),
                "Mix": data_map.get("interior_mix", 0),
                # Sadiki-specific options (if present in DB)
                "Rigips + glet + lavabil": data_map.get("interior_rigips_glet_lavabil", 0),
                "Plăci gips-fibră (Fermacell)": data_map.get("interior_fermacell", 0),
                "Placare OSB aparent": data_map.get("interior_osb_aparent", 0),
                "Lambriu (molid/larice)": data_map.get("interior_lambriu", 0),
                "Placare cu panouri acustice": data_map.get("interior_panouri_acustice", 0),
            },
            "interior_outer": {
                "Tencuială": data_map.get("interior_outer_tencuiala", data_map.get("interior_tencuiala", 0)),
                "Lemn": data_map.get("interior_outer_lemn", data_map.get("interior_lemn", 0)),
                "Fibrociment": data_map.get("interior_outer_fibrociment", data_map.get("interior_fibrociment", 0)),
                "Mix": data_map.get("interior_outer_mix", data_map.get("interior_mix", 0)),
                "Rigips + glet + lavabil": data_map.get("interior_outer_rigips_glet_lavabil", data_map.get("interior_rigips_glet_lavabil", 0)),
                "Plăci gips-fibră (Fermacell)": data_map.get("interior_outer_fermacell", data_map.get("interior_fermacell", 0)),
                "Placare OSB aparent": data_map.get("interior_outer_osb_aparent", data_map.get("interior_osb_aparent", 0)),
                "Lambriu (molid/larice)": data_map.get("interior_outer_lambriu", data_map.get("interior_lambriu", 0)),
                "Placare cu panouri acustice": data_map.get("interior_outer_panouri_acustice", data_map.get("interior_panouri_acustice", 0)),
            },
            "exterior": { 
                "Tencuială": data_map.get("exterior_tencuiala", 0), 
                "Lemn": data_map.get("exterior_lemn", 0),
                "Fibrociment": data_map.get("exterior_fibrociment", 0),
                "Mix": data_map.get("exterior_mix", 0),
                "Lemn Ars (Shou Sugi Ban)": data_map.get("exterior_lemn_ars", 0),  # optional
                # Sadiki-specific facade options
                "Fațadă ventilată HPL": data_map.get("exterior_hpl_ventilat", 0),
                "Fațadă ventilată cu ceramică": data_map.get("exterior_ceramica_ventilat", 0),
                "Cărămidă aparentă (placaj)": data_map.get("exterior_caramida_aparenta_placaj", 0),
                "Piatra naturală (placaj)": data_map.get("exterior_piatra_naturala_placaj", 0),
                "Siding compozit (WPC)": data_map.get("exterior_wpc", 0),
            }
        },
        "openings": {
            # Uși normale: preț €/Stück per tip (Innen/Außen) – același câmp în formular ca „Türtyp“
            "door_interior_prices": {
                "Standard": float(data_map.get("door_interior_standard", data_map.get("door_interior_price", 320))),
                "Holz": float(data_map.get("door_interior_holz", 580)),
                "Glas": float(data_map.get("door_interior_glas", 890)),
                "Weiß lackiert": float(data_map.get("door_interior_weiss_lackiert", 420)),
            },
            "door_exterior_prices": {
                "Standard": float(data_map.get("door_exterior_standard", data_map.get("door_exterior_price", 1450))),
                "Holz": float(data_map.get("door_exterior_holz", 2200)),
                "Aluminium": float(data_map.get("door_exterior_aluminium", 2800)),
                "Kunststoff": float(data_map.get("door_exterior_kunststoff", 1600)),
            },
            # Ferestre: €/m² (Fensterart)
            "windows_price_per_m2": {
                "2-fach verglast": data_map.get("window_2_fach_price", data_map.get("window_3fach_verglast_price", 320)),
                "3-fach verglast": data_map.get("window_3_fach_price", data_map.get("window_3fach_verglast_price", 420)),
                "3-fach verglast, Passiv": data_map.get("window_3fach_passiv_price", 580),
            },
            "sliding_door_prices_per_m2": {
                "Standard": float(data_map.get("sliding_door_standard_price", 690)),
                "Hebeschiebetür": float(data_map.get("sliding_door_hebeschiebetuer_price", 880)),
                "Panorama": float(data_map.get("sliding_door_panorama_price", 1040)),
                "Aluminium Premium": float(data_map.get("sliding_door_aluminium_premium_price", 980)),
            },
            # Garagentor: €/Stück (doar dacă formular: Garagentor gewünscht)
            "garage_door_prices": {
                "Sektionaltor Standard": float(
                    data_map.get("garage_door_sektional_standard_stueck", data_map.get("garage_door_standard_price", 2400))
                ),
                "Sektionaltor Premium": float(
                    data_map.get("garage_door_sektional_premium_stueck", data_map.get("garage_door_premium_price", 3200))
                ),
                "Rolltor": float(data_map.get("garage_door_rolltor_stueck", data_map.get("garage_door_rolltor_price", 2100))),
                "Schwingtor": float(data_map.get("garage_door_schwingtor_stueck", 2600)),
                "Seiten-Sektionaltor": float(data_map.get("garage_door_seiten_sektional_stueck", 3800)),
            },
            "door_height_m": {
                "Standard (2,01 m)": float(data_map.get("door_height_standard_m", 2.01)),
                "Komfort (2,10 m)": float(data_map.get("door_height_komfort_m", 2.10)),
                "Hoch (2,20 m)": float(data_map.get("door_height_hoch_m", 2.20)),
            },
        },
        "area": {
            "floor_coefficient_per_m2": data_map.get("floor_coeff_per_m2", 0),
            "ceiling_coefficient_per_m2": data_map.get("ceiling_coeff_per_m2", 0),
            # Raumhöhe (floor_height): înălțimi (m) per opțiune – folosite la calculul ariilor pereți, nu la preț
            "floor_height_m": {
                "Standard (2,50 m)": float(data_map.get("inaltime_etaje_standard_m", 2.5)),
                "Komfort (2,70 m)": float(data_map.get("inaltime_etaje_komfort_m", 2.7)),
                "Hoch (2,85+ m)": float(data_map.get("inaltime_etaje_hoch_m", 2.85)),
            },
        },
        "stairs": {
            "price_per_stair_unit": data_map.get("price_per_stair_unit", 0),
            "railing_price_per_stair": data_map.get("railing_price_per_stair", 0),
            "stair_type_price_map": {
                "Standard": data_map.get("stairs_type_standard_piece_price", data_map.get("price_per_stair_unit", 0)),
                "Holz": data_map.get("stairs_type_holz_piece_price", data_map.get("price_per_stair_unit", 0)),
                "Beton": data_map.get("stairs_type_beton_piece_price", data_map.get("price_per_stair_unit", 0)),
                "Metall": data_map.get("stairs_type_metall_piece_price", data_map.get("price_per_stair_unit", 0)),
                "Sonder": data_map.get("stairs_type_sonder_piece_price", data_map.get("price_per_stair_unit", 0)),
            },
        },
        "aufstockung_phase1": {
            "demolition_price_keys": [
                "aufstockung_demolition_roof_basic_m2",
                "aufstockung_demolition_roof_complex_m2",
                "aufstockung_demolition_roof_special_m2",
            ],
            "stair_opening_price_key": "aufstockung_stair_opening_piece",
            "statik_stahlbetonverbunddecke_key": "aufstockung_statik_stahlbetonverbunddecke_m2",
        },
        "utilities": {
            "electricity": {
                "coefficient_electricity_per_m2": data_map.get("electricity_base_price", 60.0),
                "energy_performance_modifiers": {
                    "Standard": 1.0 + (data_map.get("nivel_energetic_standard_price", 0) / max(_elec_base, 1)),
                    "KfW 55": 1.0 + (data_map.get("nivel_energetic_kfw55_price", 25) / max(_elec_base, 1)),
                    "KfW 40": 1.0 + (data_map.get("nivel_energetic_kfw40_price", 45) / max(_elec_base, 1)),
                    "KfW 40+": 1.0 + (data_map.get("nivel_energetic_kfw40plus_price", 65) / max(_elec_base, 1)),
                }
            },
            "heating": {
                "coefficient_heating_per_m2": data_map.get("heating_base_price", 70.0),
                "type_coefficients": {
                    "Gas": data_map.get("tip_incalzire_gas_price", 55) / max(_heat_base, 1),
                    "Wärmepumpe": data_map.get("tip_incalzire_waermepumpe_price", 95) / max(_heat_base, 1),
                    "Elektrisch": data_map.get("tip_incalzire_elektrisch_price", 45) / max(_heat_base, 1),
                    "Gaz": data_map.get("tip_incalzire_gas_price", 55) / max(_heat_base, 1),
                },
                "energy_performance_modifiers": {
                    "Standard": 1.0,
                    "KfW 55": 1.0 + (data_map.get("nivel_energetic_kfw55_price", 25) / max(_heat_base, 1)),
                    "KfW 40": 1.0 + (data_map.get("nivel_energetic_kfw40_price", 45) / max(_heat_base, 1)),
                    "KfW 40+": 1.0 + (data_map.get("nivel_energetic_kfw40plus_price", 65) / max(_heat_base, 1)),
                }
            },
            "sewage": { "coefficient_sewage_per_m2": data_map.get("sewage_base_price", 45.0) },
            "ventilation": { "coefficient_ventilation_per_m2": data_map.get("ventilation_base_price", 55.0) }
        },
        "fireplace": {
            "prices": {
                "Kein Kamin": data_map.get("tip_semineu_kein_price", 0),
                "Klassischer Holzofen": data_map.get("tip_semineu_holzofen_price", 4200),
                "Moderner Design-Kaminofen": data_map.get("tip_semineu_design_price", 6500),
                "Pelletofen (automatisch)": data_map.get("tip_semineu_pelletofen_price", 8500),
                "Einbaukamin": data_map.get("tip_semineu_einbaukamin_price", 7200),
                "Kachel-/wassergeführter Kamin": data_map.get("tip_semineu_kachel_price", 9500),
            },
            "horn_per_floor": data_map.get("horn_price_per_floor", 1500.0),
        },
        "wandaufbau": {
            "aussen": {
                "CLT 35cm": data_map.get("wandaufbau_aussen_clt_35", 400),
                "CLT 32cm": data_map.get("wandaufbau_aussen_clt_32", 380),
                "CLT 30cm": data_map.get("wandaufbau_aussen_clt_30", 360),
                "Holzriegel 35cm": data_map.get("wandaufbau_aussen_holzriegel_35", 300),
                "Holzriegel 32cm": data_map.get("wandaufbau_aussen_holzriegel_32", 280),
                "Holzriegel 30cm": data_map.get("wandaufbau_aussen_holzriegel_30", 260),
                "Beton 35cm": data_map.get("wandaufbau_aussen_beton_35", 200),
                "Beton 32cm": data_map.get("wandaufbau_aussen_beton_32", 180),
                "Beton 30cm": data_map.get("wandaufbau_aussen_beton_30", 160),
            },
            "innen": {
                "CLT 35cm": data_map.get("wandaufbau_innen_clt_35", 400),
                "CLT 32cm": data_map.get("wandaufbau_innen_clt_32", 380),
                "CLT 30cm": data_map.get("wandaufbau_innen_clt_30", 360),
                "Holzriegel 35cm": data_map.get("wandaufbau_innen_holzriegel_35", 300),
                "Holzriegel 32cm": data_map.get("wandaufbau_innen_holzriegel_32", 280),
                "Holzriegel 30cm": data_map.get("wandaufbau_innen_holzriegel_30", 260),
                "Beton 35cm": data_map.get("wandaufbau_innen_beton_35", 200),
                "Beton 32cm": data_map.get("wandaufbau_innen_beton_32", 180),
                "Beton 30cm": data_map.get("wandaufbau_innen_beton_30", 160),
            },
        },
        "wintergaerten_balkone": {
            "wintergarten": {
                "Glaswand": data_map.get("wintergarten_glaswand_price", 560),
                "Plexiglaswand": data_map.get("wintergarten_plexiglaswand_price", 150),
            },
            "balkon": {
                "Holzgeländer": data_map.get("balkon_holzgelander_price", 560),
                "Stahlgeländer": data_map.get("balkon_stahlgelander_price", 150),
                "Glasgeländer": data_map.get("balkon_glasgelander_price", 700),
            },
        },
        "boden_decke_belag": {
            "bodenaufbau": {
                "Geschossdecke Holz Standard": data_map.get("bodenaufbau_holz_standard_price", 560),
                "Holzbalkendecke": data_map.get("bodenaufbau_holz_balken_price", 280),
                "Massivdecke Stahlbeton": data_map.get("bodenaufbau_beton_price", 320),
                "Bodenplatte Beton": data_map.get("bodenaufbau_bodenplatte_price", 150),
            },
            "deckenaufbau": {
                "Gipskarton Standard": data_map.get("deckenaufbau_gipskarton_price", 45),
                "Gipskarton Akustik": data_map.get("deckenaufbau_gipskarton_akustik_price", 55),
                "Sichtschalung Holz": data_map.get("deckenaufbau_sichtschalung_price", 65),
                "Unterdecke abgehängt": data_map.get("deckenaufbau_abgehaengt_price", 75),
            },
            "bodenbelag": {
                "Estrich + Fliesen": data_map.get("bodenbelag_estrich_fliesen_price", 35),
                "Parkett Eiche": data_map.get("bodenbelag_parkett_price", 42),
                "Laminat": data_map.get("bodenbelag_laminat_price", 38),
                "Teppichboden": data_map.get("bodenbelag_teppich_price", 28),
            },
        },
    }

    # Opțiuni custom (din Preisdatenbank): același tag, apar în formular și aici în pricing
    try:
        opt_res = client.table("tenant_form_options").select("field_tag, option_label, option_value, price_key").eq("tenant_id", tenant_id).execute()
        if opt_res.data:
            for row in opt_res.data:
                tag = (row.get("field_tag") or "").strip()
                label = (row.get("option_label") or row.get("option_value") or "").strip()
                pk = (row.get("price_key") or "").strip()
                if not tag or not label or not pk:
                    continue
                val = float(data_map.get(pk, 0))
                if tag == "system_type":
                    out.setdefault("system", {}).setdefault("base_unit_prices", {})[label] = {"interior": val, "exterior": val}
                elif tag == "site_access":
                    out.setdefault("sistem_constructiv", {}).setdefault("acces_santier_factor", {})[label] = val
                elif tag == "terrain":
                    out.setdefault("sistem_constructiv", {}).setdefault("teren_factor", {})[label] = val
                elif tag == "foundation_type":
                    out.setdefault("foundation", {}).setdefault("unit_price_per_m2", {})[label] = val
                elif tag == "interior_finish_interior_walls":
                    out.setdefault("finishes", {}).setdefault("interior_inner", {})[label] = val
                elif tag == "interior_finish_exterior_walls":
                    out.setdefault("finishes", {}).setdefault("interior_outer", {})[label] = val
                elif tag == "sliding_door_type":
                    out.setdefault("openings", {}).setdefault("sliding_door_prices_per_m2", {})[label] = val
                elif tag == "garage_door_type":
                    out.setdefault("openings", {}).setdefault("garage_door_prices", {})[label] = val
                elif tag == "door_height":
                    out.setdefault("openings", {}).setdefault("door_height_m", {})[label] = val
    except Exception:
        pass

    return out