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

    # Baustellenzufahrt (accesSantier): toate cele 3 opțiuni din form influențează prețul
    _elec_base = float(data_map.get("electricity_base_price", 60.0))
    _heat_base = float(data_map.get("heating_base_price", 70.0))

    out = {
        "foundation": {
            "unit_price_per_m2": {
                # Form: Untergeschoss / Fundament
                "Kein Keller (nur Bodenplatte)": data_map.get("unit_price_placa", 120),
                "Keller (unbeheizt / Nutzkeller)": data_map.get("unit_price_keller_nutzkeller", 145),
                "Keller (mit einfachem Ausbau)": data_map.get("unit_price_keller_ausbau", 185),
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
            "daemmung_keine_price": data_map.get("daemmung_keine_price", 0),
            "daemmung_zwischensparren_price": data_map.get("daemmung_zwischensparren_price", 55),
            "daemmung_aufsparren_price": data_map.get("daemmung_aufsparren_price", 75),
            # Unterdach (form: unterdach)
            "unterdach_folie_price": data_map.get("unterdach_folie_price", 12),
            "unterdach_schalung_folie_price": data_map.get("unterdach_schalung_folie_price", 28),
            # Dachstuhl-Typ (form: dachstuhlTyp)
            "dachstuhl_sparrendach_price": data_map.get("dachstuhl_sparrendach_price", 95),
            "dachstuhl_pfettendach_price": data_map.get("dachstuhl_pfettendach_price", 110),
            "dachstuhl_kehlbalkendach_price": data_map.get("dachstuhl_kehlbalkendach_price", 105),
            "dachstuhl_sonderkonstruktion_price": data_map.get("dachstuhl_sonderkonstruktion_price", 130),
            "sichtdachstuhl_zuschlag_price": data_map.get("sichtdachstuhl_zuschlag_price", 25),
            "panta_acoperis_zuschlag_per_grad": data_map.get("panta_acoperis_zuschlag_per_grad", 0.5),
            # Sadiki-specific roof cover prices (if present in DB)
            "roof_shingle_price_per_m2": data_map.get("roof_shingle_price_per_m2", 0),
            "roof_metal_tile_price_per_m2": data_map.get("roof_metal_tile_price_per_m2", 0),
            "roof_ceramic_tile_price_per_m2": data_map.get("roof_ceramic_tile_price_per_m2", 0),
            "roof_tpo_pvc_price_per_m2": data_map.get("roof_tpo_pvc_price_per_m2", 0),
            "roof_green_extensive_price_per_m2": data_map.get("roof_green_extensive_price_per_m2", 0),
        },
        "finishes": {
            "interior": {
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
            # Uși: un preț per m² pentru interior, unul pentru exterior (folosit doar pentru calcul arie × preț)
            "door_interior_price_per_m2": data_map.get("door_interior_price", data_map.get("door_standard_2m_price", 0)),
            "door_exterior_price_per_m2": data_map.get("door_exterior_price", data_map.get("door_standard_2m_price", 0)),
            # Ferestre: doar 2 straturi sau 3 straturi (alegere din formular Fensterart)
            "windows_price_per_m2": {
                "2-fach verglast": data_map.get("window_2_fach_price", data_map.get("window_3fach_verglast_price", 320)),
                "3-fach verglast": data_map.get("window_3_fach_price", data_map.get("window_3fach_verglast_price", 420)),
                "3-fach verglast, Passiv": data_map.get("window_3fach_passiv_price", 580),
            }
        },
        "area": {
            "floor_coefficient_per_m2": data_map.get("floor_coeff_per_m2", 0),
            "ceiling_coefficient_per_m2": data_map.get("ceiling_coeff_per_m2", 0),
            # Geschosshöhe: înălțimi (m) per opțiune – folosite la calculul ariilor pereți, nu la preț
            "floor_height_m": {
                "Standard (2,50 m)": float(data_map.get("inaltime_etaje_standard_m", 2.5)),
                "Komfort (2,70 m)": float(data_map.get("inaltime_etaje_komfort_m", 2.7)),
                "Hoch (2,85+ m)": float(data_map.get("inaltime_etaje_hoch_m", 2.85)),
            },
        },
        "stairs": { "price_per_stair_unit": data_map.get("price_per_stair_unit", 0), "railing_price_per_stair": data_map.get("railing_price_per_stair", 0) },
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
            }
        }
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
    except Exception:
        pass

    return out