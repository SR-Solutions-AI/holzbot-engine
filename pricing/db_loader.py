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

    return {
        "foundation": {
            "unit_price_per_m2": {
                "Placă": data_map.get("unit_price_placa", 0),
                "Piloți": data_map.get("unit_price_piloti", 0),
                "Soclu": data_map.get("unit_price_soclu", 0)
            }
        },
        "system": {
            "base_unit_prices": {
                "CLT": { "interior": data_map.get("clt_interior_price", 0), "exterior": data_map.get("clt_exterior_price", 0) },
                "CLT Premium": { "interior": data_map.get("clt_interior_price", 0), "exterior": data_map.get("clt_exterior_price", 0) },
                "HOLZRAHMEN": { "interior": data_map.get("holzrahmen_interior_price", 0), "exterior": data_map.get("holzrahmen_exterior_price", 0) },
                "Holzrahmen Standard": { "interior": data_map.get("holzrahmen_interior_price", 0), "exterior": data_map.get("holzrahmen_exterior_price", 0) },
                "MASSIVHOLZ": { "interior": data_map.get("massivholz_interior_price", 0), "exterior": data_map.get("massivholz_exterior_price", 0) },
                "Massivholz": { "interior": data_map.get("massivholz_interior_price", 0), "exterior": data_map.get("massivholz_exterior_price", 0) }
            },
            "prefabrication_modifiers": {
                "MODULE": data_map.get("prefab_modifier_module", 1.0),
                "PANOURI": data_map.get("prefab_modifier_panouri", 1.0),
                "SANTIER": data_map.get("prefab_modifier_santier", 1.0)
            }
        },
        "roof": {
            "overhang_m": data_map.get("overhang_m", 0.4),
            "sheet_metal_price_per_m": data_map.get("sheet_metal_price_per_m", 0),
            "insulation_price_per_m2": data_map.get("insulation_price_per_m2", 0),
            "tile_price_per_m2": data_map.get("tile_price_per_m2", 0),
            "metal_price_per_m2": data_map.get("metal_price_per_m2", 0),
            "membrane_price_per_m2": data_map.get("membrane_price_per_m2", 0),
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
            "windows_unit_prices_per_m2": {
                "PVC": data_map.get("window_pvc_price", 0),
                "Lemn": data_map.get("window_lemn_price", 0),
                "Aluminiu": data_map.get("window_aluminiu_price", 0),
                "Lemn-Aluminiu": data_map.get("window_lemn_aluminiu_price", 0),
                "Premium Holz-Alu": data_map.get("window_premium_wood", 0),  # optional
                # Sadiki-specific options
                "Aluminiu cu barieră termică": data_map.get("window_aluminiu_termo", 0),
                "PVC 7 camere": data_map.get("window_pvc_7_camere", 0),
                "Oțel (profil subțire, tip loft)": data_map.get("window_otel_loft", 0),
                "Lemn stratificat (triplustrat)": data_map.get("window_lemn_triplustrat", 0),
                "Ferestre fixe panoramice (curtain wall)": data_map.get("window_curtain_wall", 0),
            },
            "doors_interior_unit_prices_per_m2": {
                "PVC": data_map.get("door_int_pvc_price", 0),
                "Lemn": data_map.get("door_int_lemn_price", 0),
                "Aluminiu": data_map.get("door_int_aluminiu_price", 0),
                "Lemn-Aluminiu": data_map.get("door_int_lemn_aluminiu_price", 0),
                "Premium Holz-Alu": data_map.get("door_int_lemn_aluminiu_price", 0),
                # Sadiki-specific options
                "Aluminiu cu barieră termică": data_map.get("door_int_aluminiu_termo", 0),
                "PVC 7 camere": data_map.get("door_int_pvc_7_camere", 0),
                "Oțel (profil subțire, tip loft)": data_map.get("door_int_otel_loft", 0),
                "Lemn stratificat (triplustrat)": data_map.get("door_int_lemn_triplustrat", 0),
                "Ferestre fixe panoramice (curtain wall)": data_map.get("door_int_curtain_wall", 0),
            },
            "doors_exterior_unit_prices_per_m2": {
                "PVC": data_map.get("door_ext_pvc_price", 0),
                "Lemn": data_map.get("door_ext_lemn_price", 0),
                "Aluminiu": data_map.get("door_ext_aluminiu_price", 0),
                "Lemn-Aluminiu": data_map.get("door_ext_lemn_aluminiu_price", 0),
                "Premium Holz-Alu": data_map.get("door_ext_lemn_aluminiu_price", 0),
                # Sadiki-specific options
                "Aluminiu cu barieră termică": data_map.get("door_ext_aluminiu_termo", 0),
                "PVC 7 camere": data_map.get("door_ext_pvc_7_camere", 0),
                "Oțel (profil subțire, tip loft)": data_map.get("door_ext_otel_loft", 0),
                "Lemn stratificat (triplustrat)": data_map.get("door_ext_lemn_triplustrat", 0),
                "Ferestre fixe panoramice (curtain wall)": data_map.get("door_ext_curtain_wall", 0),
            }
        },
        "area": { "floor_coefficient_per_m2": data_map.get("floor_coeff_per_m2", 0), "ceiling_coefficient_per_m2": data_map.get("ceiling_coeff_per_m2", 0) },
        "stairs": { "price_per_stair_unit": data_map.get("price_per_stair_unit", 0), "railing_price_per_stair": data_map.get("railing_price_per_stair", 0) },
        "utilities": {
            "electricity": { "coefficient_electricity_per_m2": data_map.get("electricity_base_price", 60.0), "energy_performance_modifiers": { "Standard": 1.0 } },
            "heating": { "coefficient_heating_per_m2": data_map.get("heating_base_price", 70.0), "type_coefficients": { "Gaz": 1.0 } },
            "sewage": { "coefficient_sewage_per_m2": data_map.get("sewage_base_price", 45.0) },
            "ventilation": { "coefficient_ventilation_per_m2": data_map.get("ventilation_base_price", 55.0) }
        }
    }