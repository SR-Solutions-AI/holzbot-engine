# new/runner/count_objects/config.py
from __future__ import annotations

# Thresholds
CONF_THRESHOLD = 0.3
OVERLAP = 30
TEMPLATE_SIMILARITY = 0.45
GEMINI_THRESHOLD_MIN = 0.35
GEMINI_THRESHOLD_MAX = 0.50

# Overlap thresholds
STAIRS_OVERLAP_THRESHOLD = 0.3  # 30% overlap cu scara = ignoră
DETECTION_OVERLAP_THRESHOLD = 0.4  # 40% overlap între detecții = ignoră

# Roboflow projects
ROBOFLOW_MAIN_PROJECT = "house-plan-uwkew"
ROBOFLOW_MAIN_VERSION = 5

# Scări - MODEL STANDARD
ROBOFLOW_STAIRS_PROJECT = "stairs-czdvt"
ROBOFLOW_STAIRS_VERSION = 2
ROBOFLOW_STAIRS_WORKSPACE = "blueprint-recognition"

# Culori pentru vizualizare (BGR)
COLORS = {
    "stairs": (0, 255, 0),  # VERDE - scări
    
    "door": {
        "template": (0, 165, 255),    # portocaliu - confirmat template
        "gemini": (0, 255, 255),      # galben - confirmat Gemini
        "rejected": (0, 0, 255)       # ROȘU - respins
    },
    "double-door": {
        "template": (255, 0, 200),
        "gemini": (255, 150, 255),
        "rejected": (0, 0, 255)
    },
    "window": {
        "template": (255, 0, 0),      # albastru
        "gemini": (255, 150, 150),
        "rejected": (0, 0, 255)
    },
    "double-window": {
        "template": (0, 255, 0),
        "gemini": (144, 238, 144),
        "rejected": (0, 0, 255)
    }
}

# Template matching
SCALES = [0.9, 1.0, 1.1]
ROTATION_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

# Paralelizare
MAX_GEMINI_WORKERS = 5
MAX_TEMPLATE_WORKERS = 16
MAX_DETECTION_WORKERS = 8
MAX_TYPE_WORKERS = 4