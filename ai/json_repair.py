# ai/json_repair.py
# Re-export: reparare JSON prin Gemini (implementare în common/json_repair.py).

from __future__ import annotations

from common.json_repair import repair_json_with_gemini, repair_json_with_gpt

__all__ = ["repair_json_with_gemini", "repair_json_with_gpt"]
