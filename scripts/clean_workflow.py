#!/usr/bin/env python3
"""
Wrapper care rulează holzbot-roof clean_workflow.
Workflow curat: rectangles, roof_types (1_w, 2_w, 4_w, 4.5_w), entire, unfold.
"""
import shutil
import subprocess
import sys
from pathlib import Path

_ENGINE_ROOT = Path(__file__).resolve().parents[1]
_ROOF_ROOT = _ENGINE_ROOT.parent / "holzbot-roof"

if __name__ == "__main__":
    if not _ROOF_ROOT.exists():
        print(f"⚠ holzbot-roof nu există la {_ROOF_ROOT}", file=sys.stderr)
        sys.exit(1)
    script = _ROOF_ROOT / "scripts" / "clean_workflow.py"
    if not script.exists():
        print(f"⚠ {script} nu există", file=sys.stderr)
        sys.exit(1)
    if len(sys.argv) < 2:
        print("Usage: python clean_workflow.py <wall_mask_path_or_folder> [output_dir]")
        sys.exit(1)
    args = [sys.executable, str(script), sys.argv[1]]
    if len(sys.argv) > 2:
        args.append(sys.argv[2])
    _py = _ROOF_ROOT / ".venv314" / "bin" / "python"
    if _py.exists():
        args[0] = str(_py)
    result = subprocess.run(args, cwd=str(_ROOF_ROOT))
    sys.exit(result.returncode)
