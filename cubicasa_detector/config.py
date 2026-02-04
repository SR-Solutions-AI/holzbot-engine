# new/runner/cubicasa_detector/config.py
from pathlib import Path

# Path-uri relative
MODULE_DIR = Path(__file__).parent
WEIGHTS_FILE = MODULE_DIR / "model_weights.pth"

# Configurări Gemini
GEMINI_MODEL = "gemini-3-flash-preview"

# Parametri detectare camere
MIN_ROOM_AREA_M2 = 1.0
MAX_ROOM_AREA_M2 = 300.0

# Parametri segmentare
MIN_PIXEL_AREA_RATIO = 0.0005  # 0.05% din arie totală
SAMPLE_STRIDE_RATIO = 0.005    # 0.5% din lățime