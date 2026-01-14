# new/runner/exterior_doors/config.py
from __future__ import annotations

# Culori (Format BGR pentru OpenCV)
COLOR_DARK_BLUE = (139, 0, 0)    # Outdoor Overlay (Albastru Închis)
COLOR_YELLOW    = (0, 255, 255)  # Indoor Overlay (Galben)
COLOR_RED       = (0, 0, 255)    # Exterior Door Box (Roșu)
COLOR_GREEN     = (0, 255, 0)    # Interior Door Box (Verde)

# Parametri clasificare
# O ușă e exterioară dacă distanța până la masca de outdoor este <= jumătate din diagonala ei
MAX_DISTANCE_RATIO = 0.4