"""
Offer-scope gating: nivel ofertă → what appears in PDF tables and pricing breakdown.

Run: python -m unittest tests.test_offer_inclusions -v
(from holzbot-engine directory)
"""

import unittest

from pdf_generator.offer_scope import (
    baukosten_position_label_de,
    get_offer_inclusions,
    normalize_nivel_oferta,
    roof_items_for_pdf_table,
)


class TestNormalizeNivelOferta(unittest.TestCase):
    def test_sistem_constructiv_wins_over_stale_materiale(self):
        fd = {
            "sistemConstructiv": {"nivelOferta": "Structură"},
            "materialeFinisaj": {"nivelOferta": "Casă completă"},
        }
        self.assertEqual(normalize_nivel_oferta(fd), "Structură")

    def test_materiale_used_when_sistem_empty(self):
        fd = {
            "sistemConstructiv": {},
            "materialeFinisaj": {"nivelOferta": "Structură + ferestre"},
        }
        self.assertEqual(normalize_nivel_oferta(fd), "Structură + ferestre")

    def test_german_rohbau_slash(self):
        fd = {"sistemConstructiv": {"nivelOferta": "Rohbau / Konstruktion"}}
        self.assertEqual(normalize_nivel_oferta(fd), "Structură")

    def test_calc_mode_structure_when_no_raw(self):
        fd = {"calc_mode": "structure"}
        self.assertEqual(normalize_nivel_oferta(fd), "Structură")


class TestGetOfferInclusions(unittest.TestCase):
    def test_matrix(self):
        cases = [
            ("Structură", False, False, False, False),
            ("Structură + ferestre", True, False, False, False),
            ("Casă completă", True, True, True, True),
        ]
        for nivel, op, fin, util, doors in cases:
            with self.subTest(nivel=nivel):
                inc = get_offer_inclusions(nivel, {})
                self.assertEqual(inc["openings"], op)
                self.assertEqual(inc["finishes"], fin)
                self.assertEqual(inc["utilities"], util)
                self.assertEqual(inc.get("openings_doors"), doors)

    def test_roof_only_forces_minimal(self):
        inc = get_offer_inclusions("Casă completă", {"roof_only_offer": True})
        self.assertTrue(inc["roof"])
        self.assertFalse(inc["openings"])
        self.assertFalse(inc["finishes"])
        self.assertFalse(inc.get("openings_doors"))


class TestBaukostenLabel(unittest.TestCase):
    def test_no_ausbau_for_rohbau(self):
        self.assertEqual(
            baukosten_position_label_de({"finishes": False, "utilities": False}),
            "Baukosten (Konstruktion, Technik)",
        )

    def test_ausbau_when_finishes_or_utilities(self):
        self.assertEqual(
            baukosten_position_label_de({"finishes": True, "utilities": False}),
            "Baukosten (Konstruktion, Ausbau, Technik)",
        )
        self.assertEqual(
            baukosten_position_label_de({"finishes": False, "utilities": True}),
            "Baukosten (Konstruktion, Ausbau, Technik)",
        )


class TestRoofItemsForPdfTable(unittest.TestCase):
    def test_keeps_skylights_when_openings_included(self):
        items = [
            {"category": "roof_structure", "name": "A", "cost": 100},
            {"category": "roof_skylights", "name": "Dachfenster", "cost": 50},
        ]
        out = roof_items_for_pdf_table(items, {"openings": True})
        self.assertEqual(len(out), 2)

    def test_drops_skylights_when_rohbau(self):
        items = [
            {"category": "roof_structure", "name": "A", "cost": 100},
            {"category": "roof_skylights", "name": "Dachfenster", "cost": 50},
        ]
        out = roof_items_for_pdf_table(items, {"openings": False})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["category"], "roof_structure")

    def test_inclusions_none_keeps_skylights(self):
        items = [{"category": "roof_skylights", "name": "X", "cost": 1}]
        out = roof_items_for_pdf_table(items, None)
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
