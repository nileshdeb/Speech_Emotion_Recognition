from __future__ import annotations

from pathlib import Path
import runpy
import sys

project_dir = Path(__file__).resolve().parent / "speech-emotion-recognition"
sys.path.insert(0, str(project_dir))

runpy.run_path(project_dir / "app.py", run_name="__main__")