"""Clamp gedämmte Dachfläche auf [0, ohne Überstand] (ohne frühere 5 %-Kollaps-Regel)."""

import importlib.util
import unittest
from pathlib import Path

_engine_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_insulated_area_for_tests",
    _engine_root / "roof" / "insulated_area.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
collapse_negligible_roof_insulated_share = _mod.collapse_negligible_roof_insulated_share


class TestRoofInsulatedThreshold(unittest.TestCase):
    def test_small_insulated_share_preserved(self):
        self.assertEqual(collapse_negligible_roof_insulated_share(100.0, 4.0), 4.0)
        self.assertEqual(collapse_negligible_roof_insulated_share(100.0, 4.99), 4.99)

    def test_five_percent_insulated_unchanged(self):
        self.assertEqual(collapse_negligible_roof_insulated_share(100.0, 5.0), 5.0)

    def test_high_insulated_no_snap_to_full_roof(self):
        self.assertEqual(collapse_negligible_roof_insulated_share(100.0, 97.0), 97.0)
        self.assertEqual(collapse_negligible_roof_insulated_share(100.0, 96.0), 96.0)

    def test_mid_range_unchanged(self):
        self.assertEqual(collapse_negligible_roof_insulated_share(100.0, 50.0), 50.0)

    def test_zero_roof_area_returns_raw_clamped(self):
        self.assertEqual(collapse_negligible_roof_insulated_share(0.0, 3.0), 3.0)

    def test_insulated_clamped_above_roof_total(self):
        self.assertEqual(collapse_negligible_roof_insulated_share(100.0, 120.0), 100.0)


if __name__ == "__main__":
    unittest.main()
