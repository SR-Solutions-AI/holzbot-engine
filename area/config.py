# new/runner/area/config.py
from __future__ import annotations

# Înălțimi standard (DIN 277 - Germania)
STANDARD_WALL_HEIGHT_M = 2.5
# Adaos la înălțimea din formular pentru suprafață verticală: structură int/ext + finisaj exterior (nu finisaj interior).
WALL_HEIGHT_EXTRA_STRUCTURE_AND_EXT_FINISH_M = 0.18
STANDARD_DOOR_HEIGHT_M = 2.05
STANDARD_WINDOW_HEIGHT_M = 1.25

# Grosimi standard pereți (pentru calcul amprentă)
WALL_THICKNESS_EXTERIOR_M = 0.30  # 30 cm
WALL_THICKNESS_INTERIOR_M = 0.15  # 15 cm