import unittest

from pdf_generator.project_overview_fields import build_selected_form_overview_items


class _Enforcer:
    def get(self, text):
        return text


class ProjectOverviewFieldsTest(unittest.TestCase):
    def test_selected_form_values_are_included_in_overview(self):
        frontend_data = {
            "sistemConstructiv": {
                "tipSistem": "Holzrahmen",
                "accesSantier": "Mittel",
                "teren": "Leichte Hanglage",
                "tipFundatie": "Piloți",
                "tipAcoperis": "Satteldach",
            },
            "structuraCladirii": {
                "tipFundatieBeci": "Keller (mit Ausbau)",
                "inaltimeEtaje": "Komfort (2,70 m)",
                "treppeTyp": "Holz",
            },
            "daemmungDachdeckung": {
                "daemmung": "Zwischensparren",
                "unterdach": "Schalung + Folie",
                "dachstuhlTyp": "Pfettendach",
                "sichtdachstuhl": True,
                "dachfensterImDach": True,
                "dachfensterTyp": "Velux",
                "dachdeckung": "Ziegel",
            },
            "ferestreUsi": {
                "windowQuality": "3-fach verglast",
                "doorMaterialInterior": "Holz",
                "doorMaterialExterior": "Aluminium",
                "slidingDoorType": "Panorama",
                "garagentorGewuenscht": True,
                "garageDoorType": "Rolltor",
            },
            "materialeFinisaj": {"materialAcoperis": "Țiglă"},
            "performantaEnergetica": {
                "tipIncalzire": "Wärmepumpe",
                "nivelEnergetic": "KfW 40",
                "tipSemineu": "Klassischer Holzofen",
                "ventilatie": True,
            },
            "wintergaertenBalkone": {
                "wintergartenTyp": "Glaswand",
                "balkonTyp": "Glasgeländer",
            },
            "wandaufbau": {
                "außenwande_ground": "CLT 35cm",
                "innenwande_ground": "Beton 30cm",
            },
            "bodenDeckeBelag": {
                "bodenaufbau_ground": "Holzbalkendecke",
                "deckenaufbau_ground": "Gipskarton Akustik",
                "bodenbelag_ground": "Parkett Eiche",
            },
            "pdf_display": {
                "customOptions": {
                    "window_quality": [
                        {"label": "Premium Glas Plus", "value": "3-fach verglast", "price_key": "window_quality_3fach_price"}
                    ],
                    "sliding_door_type": [
                        {"label": "Panorama XL", "value": "Panorama", "price_key": "sliding_door_panorama_price"}
                    ],
                    "interior_finish_interior_walls": [
                        {"label": "Innenputz Fine", "value": "Tencuială", "price_key": "interior_tencuiala"}
                    ],
                    "garage_door_type": [
                        {"label": "Sektionaltor Premium", "value": "Rolltor", "price_key": "garage_door_sectional_price"}
                    ],
                },
                "paramLabelOverrides": {
                    "window_quality_3fach_price": "Premium Glas Plus",
                    "sliding_door_panorama_price": "Panorama XL",
                    "interior_tencuiala": "Innenputz Fine",
                    "garage_door_sectional_price": "Sektionaltor Premium",
                    "tip_semineu_holzofen_price": "Holzofen",
                    "tip_incalzire_waermepumpe_price": "Pelletheizung",
                    "nivel_energetic_kfw40_price": "Effizienzhaus 40",
                    "bodenbelag_parkett_price": "Vinyl / Designboden",
                    "deckenaufbau_gipskarton_akustik_price": "Holz-Beton-Verbunddecke",
                    "bodenbelag_2_price": "Vinyl / Designboden",
                    "deckenaufbau_2_price": "Brettsperrholzdecke massiv (CLT)",
                },
            },
        }
        finishes_per_floor = {
            "Erdgeschoss": {
                "interior_inner": "Tencuială",
                "interior_outer": "Lemn",
                "exterior": "Fibrociment",
            }
        }

        items = build_selected_form_overview_items(
            frontend_data,
            _Enforcer(),
            {"finishes": True, "openings": True, "utilities": True},
            finishes_per_floor,
            2,
            acces_santier_de="Mittel",
            tip_fundatie_de="Piloți",
            tip_sistem_de="Holzrahmen",
            tip_acoperis_de="Satteldach",
            material_acoperis_de="Țiglă",
            incalzire_de="Wärmepumpe",
            nivel_energetic_de="KfW 40",
            nivel_finisare_de="Schlüsselfertig",
            tip_semineu="Klassischer Holzofen",
        )
        joined = "\n".join(items)

        for expected in [
            "Fensterart",
            "Innentüren",
            "Außentüren",
            "Schiebetür",
            "Garagentor",
            "Wintergarten",
            "Balkon",
            "Wandaufbau Außenwände",
            "Wandaufbau Innenwände",
            "Bodenaufbau",
            "Deckenaufbau",
            "Bodenbelag",
            "Treppentyp",
            "Raumhöhe",
            "Sichtdachstuhl",
            "Dachfenster",
            "Lüftung / Wärmerückgewinnung",
        ]:
            self.assertIn(expected, joined)

        self.assertIn("Premium Glas Plus", joined)
        self.assertIn("Panorama XL", joined)
        self.assertIn("Innenputz Fine", joined)
        self.assertIn("Sektionaltor Premium", joined)
        self.assertIn("Holzofen", joined)
        self.assertIn("Pelletheizung", joined)
        self.assertIn("Effizienzhaus 40", joined)
        self.assertIn("Vinyl / Designboden", joined)
        self.assertTrue(
            "Holz-Beton-Verbunddecke" in joined or "Brettsperrholzdecke massiv (CLT)" in joined
        )


if __name__ == "__main__":
    unittest.main()
