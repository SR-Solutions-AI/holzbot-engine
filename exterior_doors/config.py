# new/runner/exterior_doors/config.py
from __future__ import annotations

# Dilatare ușoară a măștii flood pentru contact mai robust
BLUE_MASK_DILATE_RATIO = 0.002   # ~0.2% din min(H,W), min 1 px

# CRITERIU NOU: distanța maximă permisă = jumătate din diagonala ușii
MAX_DISTANCE_RATIO = 0.5  # dist ≤ diagonal/2 → EXTERIOR

# Fallback: dacă distanța = 0 (contact direct) → automat EXTERIOR
MIN_TOUCH_PIXELS = 1  # Orice contact direct = exterior