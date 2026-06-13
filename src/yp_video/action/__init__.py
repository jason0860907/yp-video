"""Action (SPOT) pipeline: the yp-video side of per-frame action detection.

This package is the orchestration home for action/SPOT work, symmetric to the
``tad`` package. It does NOT contain the model — SPOT lives in the separate
``~/yp-spot`` repo/venv and is invoked across a process boundary (subprocess +
JSON on disk). Here we only build the command, resolve checkpoints, and parse /
convert SPOT output into annotations.

Modules:
  - ``prelabel``: SPOT checkpoint discovery, command building, and turning raw
    SPOT predictions into action annotation records.
  - ``frames``: action-frame cache management used by SPOT training.

Web routers (``web/routers/action_annotate.py`` / ``action_train.py``) own the
job/progress/GPU-lock orchestration and call into this package; the reusable,
non-HTTP logic lives here so it can be used outside the web layer.
"""

from .prelabel import (
    ACTION_LABELS,
    build_command,
    checkpoint_ref,
    default_checkpoint,
    list_checkpoints,
    load_predictions,
    predictions_to_annotation,
    resolve_checkpoint,
    spot_available,
)
from .frames import (
    ActionFrameCacheError,
    action_frame_dir,
    ensure_action_frame_cache,
    ensure_action_frame_caches,
    inspect_action_frame_cache,
)

__all__ = [
    "ACTION_LABELS",
    "build_command",
    "checkpoint_ref",
    "default_checkpoint",
    "list_checkpoints",
    "load_predictions",
    "predictions_to_annotation",
    "resolve_checkpoint",
    "spot_available",
    "ActionFrameCacheError",
    "action_frame_dir",
    "ensure_action_frame_cache",
    "ensure_action_frame_caches",
    "inspect_action_frame_cache",
]
