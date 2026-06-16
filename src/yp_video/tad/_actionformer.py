"""Shared helpers for locating and importing the vendored ActionFormer submodule.

Both train.py and infer.py need to put ActionFormer on sys.path before importing
its ``libs.*`` packages; keep that in one place so the path logic can't drift.
"""

from __future__ import annotations

import sys

from yp_video.config import ACTIONFORMER_DIR


def setup_actionformer() -> None:
    """Add ActionFormer (and its libs/utils) to sys.path for imports."""
    af_dir = str(ACTIONFORMER_DIR)
    af_utils = str(ACTIONFORMER_DIR / "libs" / "utils")
    if af_dir not in sys.path:
        sys.path.insert(0, af_dir)
    if af_utils not in sys.path:
        sys.path.insert(0, af_utils)


def check_actionformer_installed() -> bool:
    """Return True if the ActionFormer submodule is present."""
    return (ACTIONFORMER_DIR / "libs" / "modeling" / "meta_archs.py").exists()
