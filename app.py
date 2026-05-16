from __future__ import annotations

import inspect
import os
import pathlib
import runpy
import sys
from pathlib import Path

# ── Patch: fix AttributeError: 'WindowsPath' object has no attribute 'endswith'
# Triggered by torch internal inspection on Python 3.10 + Windows (runpy frames
# pass WindowsPath objects where inspect expects plain str filenames).
# We wrap every inspect helper that can receive or return a path object.

def _to_str(p):
    """Convert any path-like to str; return None or non-paths unchanged."""
    if isinstance(p, pathlib.PurePath):
        return str(p)
    return p

_orig_getsourcefile = inspect.getsourcefile
def _getsourcefile(obj):
    try:
        return _to_str(_orig_getsourcefile(obj))
    except Exception:
        return None
inspect.getsourcefile = _getsourcefile

_orig_getfile = inspect.getfile
def _getfile(obj):
    return _to_str(_orig_getfile(obj))
inspect.getfile = _getfile

_orig_getabsfile = inspect.getabsfile
def _getabsfile(obj, _filename=None):
    _filename = _to_str(_filename)
    return _to_str(_orig_getabsfile(obj, _filename))
inspect.getabsfile = _getabsfile

_orig_getmodule = inspect.getmodule
def _getmodule(obj, _filename=None):
    _filename = _to_str(_filename)
    return _orig_getmodule(obj, _filename)
inspect.getmodule = _getmodule

_orig_findsource = inspect.findsource
def _findsource(obj):
    try:
        return _orig_findsource(obj)
    except AttributeError:
        # WindowsPath leaked somewhere — return empty so callers get None safely
        return [], 0
inspect.findsource = _findsource

_orig_getframeinfo = inspect.getframeinfo
def _getframeinfo(frame, context=1):
    try:
        return _orig_getframeinfo(frame, context)
    except AttributeError:
        return inspect.Traceback("<unknown>", 0, "<unknown>", [], 0)
inspect.getframeinfo = _getframeinfo
# ─────────────────────────────────────────────────────────────────────────────

project_dir = Path(__file__).resolve().parent / "speech-emotion-recognition"
sys.path.insert(0, str(project_dir))

runpy.run_path(str(project_dir / "app.py"), run_name="__main__")
