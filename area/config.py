# new/runner/area/config.py
from __future__ import annotations

from building_dimensions import STANDARD_DOOR_HEIGHT_M, STANDARD_WALL_HEIGHT_M

# Adaos la înălțimea din formular pentru suprafață verticală: structură int/ext + finisaj exterior (nu finisaj interior).
WALL_HEIGHT_EXTRA_STRUCTURE_AND_EXT_FINISH_M = 0.18
STANDARD_WINDOW_HEIGHT_M = 1.25

# Grosimi standard pereți (pentru calcul amprentă)
WALL_THICKNESS_EXTERIOR_M = 0.30  # 30 cm
WALL_THICKNESS_INTERIOR_M = 0.15  # 15 cm