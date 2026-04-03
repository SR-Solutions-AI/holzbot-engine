# tests/test_preisdatenbank.py
"""
Tests that Preisdatenbank (pricing parameters) are loaded and mapped correctly.
- TestPreisdatenbankMapping: keys from pricing_parameters map to correct structure and values.
- TestPreisdatenbankPricesApplied: the same prices are actually used in the calculator modules
  (foundation, walls, stairs, fireplace) so that cost = area × unit_price or count × price.
"""
import pytest
from unittest.mock import MagicMock, patch


def _build_coeffs_from_data_map(data_map: dict) -> dict:
    """
    Same logic as db_loader: build the pricing coeffs structure from a flat key->value map.
    Used to test mapping without Supabase.
    """
    _elec_base = float(data_map.get("electricity_base_price", 60.0))
    _heat_base = float(data_map.get("heating_base_price", 70.0))

    out = {
        "foundation": {
            "unit_price_per_m2": {
                "Kein Keller (nur Bodenplatte)": data_map.get("unit_price_placa", 120),
                "Keller (ohne Ausbau)": data_map.get("unit_price_keller_nutzkeller", 145),
                "Keller (unbeheizt / Nutzkeller) (ohne Ausbau)": data_map.get("unit_price_keller_nutzkeller", 145),
                "Keller (mit Ausbau)": data_map.get("unit_price_keller_ausbau", 185),
                "Keller (unbeheizt / Nutzkeller)": data_map.get("unit_price_keller_nutzkeller", 145),
                "Keller (mit einfachem Ausbau)": data_map.get("unit_price_keller_ausbau", 185),
                "Placă": data_map.get("unit_price_placa", 120),
                "Piloți": data_map.get("unit_price_piloti", 180),
                "Soclu": data_map.get("unit_price_soclu", 95),
            }
        },
        "system": {
            "base_unit_prices": {
                "Holzrahmen": {
                    "interior": data_map.get("holzrahmen_interior_price", 0),
                    "exterior": data_map.get("holzrahmen_exterior_price", 0),
                },
                "CLT": {
                    "interior": data_map.get("clt_interior_price", 0),
                    "exterior": data_map.get("clt_exterior_price", 0),
                },
            }
        },
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
        "stairs": {
            "price_per_stair_unit": data_map.get("price_per_stair_unit", 0),
            "railing_price_per_stair": data_map.get("railing_price_per_stair", 0),
        },
        "fireplace": {
            "prices": {
                "Kein Kamin": data_map.get("tip_semineu_kein_price", 0),
                "Klassischer Holzofen": data_map.get("tip_semineu_holzofen_price", 4200),
            },
            "horn_per_floor": data_map.get("horn_price_per_floor", 1500.0),
        },
        "openings": {
            "door_interior_prices": {
                "Standard": data_map.get("door_interior_standard", data_map.get("door_interior_price", 320)),
                "Holz": data_map.get("door_interior_holz", 580),
            },
            "door_exterior_prices": {
                "Standard": data_map.get("door_exterior_standard", data_map.get("door_exterior_price", 1450)),
                "Holz": data_map.get("door_exterior_holz", 2200),
            },
            "windows_price_per_m2": {
                "2-fach verglast": data_map.get("window_2_fach_price", 320),
                "3-fach verglast": data_map.get("window_3_fach_price", 420),
                "3-fach verglast, Passiv": data_map.get("window_3fach_passiv_price", 580),
            },
            "sliding_door_prices_per_m2": {
                "Standard": data_map.get("sliding_door_standard_price", 690),
                "Hebeschiebetür": data_map.get("sliding_door_hebeschiebetuer_price", 880),
                "Panorama": data_map.get("sliding_door_panorama_price", 1040),
                "Aluminium Premium": data_map.get("sliding_door_aluminium_premium_price", 980),
            },
        },
        "finishes": {
            "interior_inner": {
                "Tencuială": data_map.get("interior_tencuiala", 0),
                "Lemn": data_map.get("interior_lemn", 0),
                "Fibrociment": data_map.get("interior_fibrociment", 0),
                "Mix": data_map.get("interior_mix", 0),
            },
            "interior_outer": {
                "Tencuială": data_map.get("interior_outer_tencuiala", data_map.get("interior_tencuiala", 0)),
                "Lemn": data_map.get("interior_outer_lemn", data_map.get("interior_lemn", 0)),
                "Fibrociment": data_map.get("interior_outer_fibrociment", data_map.get("interior_fibrociment", 0)),
                "Mix": data_map.get("interior_outer_mix", data_map.get("interior_mix", 0)),
            },
            "exterior": {
                "Tencuială": data_map.get("exterior_tencuiala", 0),
                "Lemn": data_map.get("exterior_lemn", 0),
                "Fibrociment": data_map.get("exterior_fibrociment", 0),
                "Mix": data_map.get("exterior_mix", 0),
            },
        },
        "utilities": {
            "electricity": {
                "coefficient_electricity_per_m2": data_map.get("electricity_base_price", 60.0),
            },
            "heating": {
                "coefficient_heating_per_m2": data_map.get("heating_base_price", 70.0),
            },
            "sewage": {"coefficient_sewage_per_m2": data_map.get("sewage_base_price", 45.0)},
            "ventilation": {"coefficient_ventilation_per_m2": data_map.get("ventilation_base_price", 55.0)},
        },
    }
    return out


class TestPreisdatenbankMapping:
    """Preisdatenbank keys must map to correct structure and preserve numeric values."""

    def test_foundation_prices_from_data_map(self):
        data_map = {
            "unit_price_placa": 125.0,
            "unit_price_keller_nutzkeller": 150.0,
            "unit_price_keller_ausbau": 190.0,
            "unit_price_piloti": 185.0,
            "unit_price_soclu": 98.0,
        }
        out = _build_coeffs_from_data_map(data_map)
        f = out["foundation"]["unit_price_per_m2"]
        assert f["Kein Keller (nur Bodenplatte)"] == 125.0
        assert f["Keller (ohne Ausbau)"] == 150.0
        assert f["Keller (unbeheizt / Nutzkeller) (ohne Ausbau)"] == 150.0
        assert f["Keller (mit Ausbau)"] == 190.0
        assert f["Keller (unbeheizt / Nutzkeller)"] == 150.0
        assert f["Keller (mit einfachem Ausbau)"] == 190.0
        assert f["Placă"] == 125.0
        assert f["Piloți"] == 185.0
        assert f["Soclu"] == 98.0

    def test_foundation_defaults_when_keys_missing(self):
        out = _build_coeffs_from_data_map({})
        f = out["foundation"]["unit_price_per_m2"]
        assert f["Kein Keller (nur Bodenplatte)"] == 120
        assert f["Piloți"] == 180
        assert f["Soclu"] == 95

    def test_stairs_and_fireplace_prices(self):
        data_map = {
            "price_per_stair_unit": 4500.0,
            "railing_price_per_stair": 800.0,
            "tip_semineu_holzofen_price": 4200.0,
            "horn_price_per_floor": 1500.0,
        }
        out = _build_coeffs_from_data_map(data_map)
        assert out["stairs"]["price_per_stair_unit"] == 4500.0
        assert out["stairs"]["railing_price_per_stair"] == 800.0
        assert out["fireplace"]["prices"]["Klassischer Holzofen"] == 4200.0
        assert out["fireplace"]["horn_per_floor"] == 1500.0

    def test_stairs_fireplace_not_zero_when_set(self):
        data_map = {
            "price_per_stair_unit": 5000.0,
            "railing_price_per_stair": 900.0,
            "tip_semineu_kein_price": 0,
            "tip_semineu_holzofen_price": 4300.0,
            "horn_price_per_floor": 1600.0,
        }
        out = _build_coeffs_from_data_map(data_map)
        assert out["stairs"]["price_per_stair_unit"] == 5000.0
        assert out["stairs"]["railing_price_per_stair"] == 900.0
        assert out["fireplace"]["prices"]["Kein Kamin"] == 0
        assert out["fireplace"]["prices"]["Klassischer Holzofen"] == 4300.0
        assert out["fireplace"]["horn_per_floor"] == 1600.0

    def test_system_and_acces_teren_factors(self):
        data_map = {
            "holzrahmen_interior_price": 280.0,
            "holzrahmen_exterior_price": 320.0,
            "acces_santier_leicht_factor": 1.0,
            "acces_santier_mittel_factor": 1.12,
            "prefab_modifier_santier": 1.28,
            "teren_eben_factor": 1.0,
            "teren_leichte_hanglage_factor": 1.06,
            "teren_starke_hanglage_factor": 1.18,
            "utilitati_anschluss_price": 3000.0,
        }
        out = _build_coeffs_from_data_map(data_map)
        assert out["system"]["base_unit_prices"]["Holzrahmen"]["interior"] == 280.0
        assert out["system"]["base_unit_prices"]["Holzrahmen"]["exterior"] == 320.0
        sc = out["sistem_constructiv"]
        assert sc["acces_santier_factor"]["Leicht (LKW 40t)"] == 1.0
        assert sc["acces_santier_factor"]["Mittel"] == 1.12
        assert sc["acces_santier_factor"]["Schwierig"] == 1.28
        assert sc["teren_factor"]["Eben"] == 1.0
        assert sc["teren_factor"]["Leichte Hanglage"] == 1.06
        assert sc["utilitati_anschluss_price"] == 3000.0

    def test_openings_and_utilities(self):
        data_map = {
            "door_interior_standard": 180.0,
            "door_exterior_standard": 220.0,
            "window_2_fach_price": 320.0,
            "window_3_fach_price": 420.0,
            "window_3fach_passiv_price": 580.0,
            "sliding_door_standard_price": 690.0,
            "sliding_door_panorama_price": 1040.0,
            "electricity_base_price": 62.0,
            "heating_base_price": 72.0,
            "sewage_base_price": 46.0,
            "ventilation_base_price": 56.0,
        }
        out = _build_coeffs_from_data_map(data_map)
        assert out["openings"]["door_interior_prices"]["Standard"] == 180.0
        assert out["openings"]["door_exterior_prices"]["Standard"] == 220.0
        assert out["openings"]["windows_price_per_m2"]["2-fach verglast"] == 320.0
        assert out["openings"]["windows_price_per_m2"]["3-fach verglast"] == 420.0
        assert out["openings"]["windows_price_per_m2"]["3-fach verglast, Passiv"] == 580.0
        assert out["openings"]["sliding_door_prices_per_m2"]["Standard"] == 690.0
        assert out["openings"]["sliding_door_prices_per_m2"]["Panorama"] == 1040.0
        assert out["utilities"]["electricity"]["coefficient_electricity_per_m2"] == 62.0
        assert out["utilities"]["heating"]["coefficient_heating_per_m2"] == 72.0
        assert out["utilities"]["sewage"]["coefficient_sewage_per_m2"] == 46.0
        assert out["utilities"]["ventilation"]["coefficient_ventilation_per_m2"] == 56.0


class TestPreisdatenbankDbLoader:
    """Integration-style: fetch_pricing_parameters with mocked Supabase returns valid structure."""

    @pytest.mark.skip(reason="Requires Supabase; run manually or in CI with env")
    def test_fetch_returns_structure(self):
        from pricing.db_loader import fetch_pricing_parameters
        out = fetch_pricing_parameters("holzbot", calc_mode="default")
        assert "foundation" in out
        assert "system" in out
        assert "stairs" in out
        assert "fireplace" in out
        assert "openings" in out
        assert "utilities" in out
        assert "sistem_constructiv" in out
        # At least one price should be non-zero if DB is populated
        stairs = out.get("stairs", {})
        assert isinstance(stairs.get("price_per_stair_unit"), (int, float))
        assert isinstance(stairs.get("railing_price_per_stair"), (int, float))


class TestPreisdatenbankPricesApplied:
    """Verifică că prețurile din Preisdatenbank se aplică efectiv în calcul (cost = arie × preț / cantitate × preț)."""

    def test_foundation_price_applied(self):
        from pricing.modules.foundation import calculate_foundation_details
        coeffs = {"unit_price_per_m2": {"Kein Keller (nur Bodenplatte)": 125.0}}
        result = calculate_foundation_details(coeffs, 100.0, "Kein Keller (nur Bodenplatte)")
        assert result["total_cost"] == 12500.0
        assert result["detailed_items"][0]["unit_price"] == 125.0 and result["detailed_items"][0]["cost"] == 12500.0

    def test_stairs_prices_applied(self):
        from pricing.modules.stairs import calculate_stairs_details
        coeffs = {"price_per_stair_unit": 4500.0, "railing_price_per_stair": 800.0}
        result = calculate_stairs_details(coeffs, total_floors=2)
        assert result["total_cost"] == 5300.0
        struct = next(i for i in result["detailed_items"] if i["category"] == "stairs_structure")
        assert struct["unit_price"] == 4500.0 and struct["cost"] == 4500.0

    def test_fireplace_and_horn_prices_applied(self):
        from pricing.modules.utilities import calculate_fireplace_details
        coeffs = {"prices": {"Klassischer Holzofen": 4200.0}, "horn_per_floor": 1500.0}
        result = calculate_fireplace_details("Klassischer Holzofen", 2, coeffs)
        assert result["total_cost"] == 7200.0  # 4200 + 2*1500
        kamin = next(i for i in result["detailed_items"] if i["category"] == "fireplace")
        assert kamin["unit_price"] == 4200.0

    def test_walls_system_prices_applied(self):
        from pricing.modules.walls import calculate_walls_details
        coeffs = {"base_unit_prices": {"Holzrahmen": {"interior": 280.0, "exterior": 320.0}}}
        result = calculate_walls_details(coeffs, 50.0, 30.0, "Holzrahmen")
        assert result["total_cost"] == 23600.0
        int_item = next(i for i in result["detailed_items"] if i["category"] == "walls_structure_int")
        assert int_item["unit_price"] == 280.0 and int_item["cost"] == 14000.0

    def test_coeffs_from_data_map_used_in_foundation(self):
        from pricing.modules.foundation import calculate_foundation_details
        data_map = {"unit_price_placa": 118.50}
        coeffs = _build_coeffs_from_data_map(data_map)
        result = calculate_foundation_details(
            coeffs["foundation"], 95.0, "Kein Keller (nur Bodenplatte)"
        )
        assert result["total_cost"] == round(95.0 * 118.50, 2)
        assert result["detailed_items"][0]["unit_price"] == 118.50

    def test_coeffs_from_data_map_used_in_stairs(self):
        from pricing.modules.stairs import calculate_stairs_details
        coeffs = _build_coeffs_from_data_map({"price_per_stair_unit": 4800.0, "railing_price_per_stair": 750.0})
        result = calculate_stairs_details(coeffs["stairs"], total_floors=3)
        assert result["total_cost"] == 2 * (4800.0 + 750.0)
        struct = next(i for i in result["detailed_items"] if i["category"] == "stairs_structure")
        assert struct["unit_price"] == 4800.0

    def test_sliding_door_priced_per_m2_and_kept_in_openings(self):
        from pricing.modules.openings import calculate_openings_details

        coeffs = {
            "door_interior_prices": {"Standard": 320.0},
            "door_exterior_prices": {"Standard": 1450.0},
            "windows_price_per_m2": {"3-fach verglast": 420.0},
            "sliding_door_prices_per_m2": {"Standard": 690.0, "Panorama": 1040.0},
        }
        frontend = {
            "ferestreUsi": {
                "windowQuality": "3-fach verglast",
                "doorMaterialInterior": "Standard",
                "doorMaterialExterior": "Standard",
                "slidingDoorType": "Panorama",
            }
        }

        result = calculate_openings_details(
            coeffs,
            [{"id": 1, "type": "sliding_door", "width_m": 2.4, "height_m": 2.1, "status": "exterior"}],
            frontend,
        )

        assert result["total_cost"] == round(2.4 * 2.1 * 1040.0, 2)
        item = result["items"][0]
        assert item["type"] == "sliding_door"
        assert item["material"] == "Panorama"
        assert item["price_unit"] == "€/m²"
