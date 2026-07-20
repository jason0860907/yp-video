"""Centralized project configuration.

Single source of truth for all project paths. Eliminates hardcoded
Path(__file__).parent.parent chains throughout the codebase (DIP).
"""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple


def _env_path(name: str, default: Path) -> Path:
    """Return ``$name`` as an expanded Path, falling back to *default*."""
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def _find_project_root() -> Path:
    """Find project root by walking up from this file to find pyproject.toml."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


# ── Project root ──────────────────────────────────────────────────
PROJECT_ROOT = _find_project_root()

# ── External dependencies (at project root) ──────────────────────
# yp-spot lives in its own repo + venv, reached across a subprocess boundary.
# Defaults to a sibling of yp-video (same parent dir); both paths are
# overridable so the integration isn't pinned to a fixed location.
SPOT_DIR = _env_path("YP_SPOT_DIR", PROJECT_ROOT.parent / "yp-spot")
SPOT_PYTHON = _env_path("YP_SPOT_PYTHON", SPOT_DIR / ".venv" / "bin" / "python")
SPOT_PACKAGE_DIR = SPOT_DIR / "yp_spot"
# Invoked as ``python -m <module>`` (no script-path coupling).
SPOT_INFERENCE_MODULE = "yp_spot.inference"
SPOT_TRAIN_MODULE = "yp_spot.train"
SPOT_AUDIO_PRECOMPUTE_MODULE = "yp_spot.audio.precompute"
# yp-reid: same pattern as yp-spot — sibling repo + venv, subprocess boundary,
# contract handshake (yp_video/contracts/reid.py ⇄ yp_reid/contract.py).
REID_PKG_DIR = _env_path("YP_REID_DIR", PROJECT_ROOT.parent / "yp-reid")
REID_PYTHON = _env_path("YP_REID_PYTHON", REID_PKG_DIR / ".venv" / "bin" / "python")
REID_EMBED_MODULE = "yp_reid.embed"
REID_TRAIN_MODULE = "yp_reid.train"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
VLLM_ENV_PATH = PROJECT_ROOT / "vllm.env"
R2_ENV_PATH = PROJECT_ROOT / "r2.env"
# Shared across the volleyiq projects, so it lives at the workspace root
# (PROJECT_ROOT.parent), not inside yp-video.
TOKENS_ENV_PATH = PROJECT_ROOT.parent / "tokens.env"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
LOGS_DIR = PROJECT_ROOT / "logs"
APP_LOG_PATH = LOGS_DIR / "yp-app.log"

# ── User data directories (~/videos) ─────────────────────────────
VIDEOS_DIR = _env_path("YP_VIDEOS_DIR", PROJECT_ROOT.parent / "videos")
RAW_VIDEOS_DIR = VIDEOS_DIR / "raw-videos"
# Cuts split by capture style — picked in the UI when exporting from Cut page.
# "broadcast" = TV-style with replays / overlays / cuts (e.g. VNL, U19, TPVL)
# "sideline"  = amateur side-court / tripod recording (e.g. 臨打 practice)
CUTS_BROADCAST_DIR = VIDEOS_DIR / "cuts-broadcast"
CUTS_SIDELINE_DIR = VIDEOS_DIR / "cuts-sideline"
CUTS_DIRS = (CUTS_BROADCAST_DIR, CUTS_SIDELINE_DIR)
# Layout rule: everything belonging to a model family — human annotations
# and machine-generated data alike — lives under that family's dir (action/,
# rally/, rally-spot/, reid/). Only sources (cuts) and the hand-edited
# val-set stay at the top level. The annotations/ subdir of each family is
# the hand-made, irreplaceable part; rally spans land under rally-spot/
# because they are its training labels. R2 category names mirror this layout
# 1:1 (the category IS the bucket key prefix AND the dir relative to videos/).
#
# seg-annotations: the VLM detect flow's raw per-clip verdicts (in_rally /
# shot_type), which vlm_to_rally consolidates into rally/pre-annotations.
SEG_ANNOTATIONS_DIR = VIDEOS_DIR / "rally" / "seg-annotations"
RALLY_PRE_ANNOTATIONS_DIR = VIDEOS_DIR / "rally" / "pre-annotations"
RALLY_ANNOTATIONS_DIR = VIDEOS_DIR / "rally-spot" / "annotations"
ACTION_ANNOTATIONS_DIR = VIDEOS_DIR / "action" / "annotations"
ACTION_PRE_ANNOTATIONS_DIR = VIDEOS_DIR / "action" / "pre-annotations"
ACTION_FRAMES_DIR = VIDEOS_DIR / "action" / "frames"
# Precomputed per-frame audio features for SPOT late-fusion training, keyed by
# backend name (e.g. action/audio/logmel/<video>.npy). Visual-only training
# ("none" backend) needs nothing here.
ACTION_AUDIO_DIR = VIDEOS_DIR / "action" / "audio"
ACTION_WAVEFORMS_DIR = VIDEOS_DIR / "action" / "waveforms"
ACTION_CHECKPOINTS_DIR = VIDEOS_DIR / "action" / "checkpoints"
# Validation set for "holdout" Action Train mode: one video filename per line
# (stem, <stem>.mp4, or <stem>_actions.jsonl all accepted; blank lines and lines
# starting with "#" are ignored). Edit this file to pick the val videos by hand.
ACTION_VAL_SET_FILE = VIDEOS_DIR / "action-val-set.txt"
# SPOT rally (segment) training. Frame caches are extracted at a reduced fps —
# native-fps caches for 800+ full matches would need ~1 TB — and keyed per rate:
# rally-spot/frames/fps2/<stem>/000000.jpg. Labels are written in the same
# reduced-fps frame space, so yp-spot trains on them unchanged.
RALLY_SPOT_FRAMES_DIR = VIDEOS_DIR / "rally-spot" / "frames"
RALLY_SPOT_CHECKPOINTS_DIR = VIDEOS_DIR / "rally-spot" / "checkpoints"
# SPOT rally predictions live apart from the VLM pre-annotations so the two
# model families never overwrite each other; Rally Label can load either.
RALLY_SPOT_PRE_ANNOTATIONS_DIR = VIDEOS_DIR / "rally-spot" / "pre-annotations"
# Player ReID. reid/annotations holds the human labels (player assignments +
# actor fixes); everything else under reid/ is recomputable derived data.
REID_DIR = VIDEOS_DIR / "reid"
REID_ANNOTATIONS_DIR = REID_DIR / "annotations"
# Exported training datasets (yp-reid Contract A: manifest.json +
# samples.jsonl referencing crops/ by relative path). Derived data:
# rebuildable from annotations/ + crops/, deliberately absent from
# R2_CATEGORIES.
REID_DATASETS_DIR = REID_DIR / "datasets"
# yp-reid checkpoint packages (Contract B: manifest.json + state dict +
# metrics), written by yp-reid training / import_weights.
REID_CHECKPOINTS_DIR = REID_DIR / "checkpoints"

# third_party checkout; weights are gated on Hugging Face.
SAM3D_DIR = Path(
    os.environ.get("SAM3D_DIR")
    or Path(__file__).resolve().parents[2].parent / "third_party" / "sam-3d-body"
)

# R2 category → local directory + glob pattern + display label. The category
# key doubles as the bucket key prefix and matches the local dir relative to
# VIDEOS_DIR; the Storage page renders this table verbatim (order included),
# so a new category never needs a frontend edit.
class R2Category(NamedTuple):
    local_dir: Path
    glob_pattern: str
    label: str
    local_only: bool = False  # listed locally, never synced to R2

R2_CATEGORIES: dict[str, R2Category] = {
    "videos": R2Category(RAW_VIDEOS_DIR, "*.mp4", "Raw Videos", local_only=True),
    "cuts-broadcast": R2Category(CUTS_BROADCAST_DIR, "*.mp4", "Cuts (Broadcast)"),
    "cuts-sideline": R2Category(CUTS_SIDELINE_DIR, "*.mp4", "Cuts (Sideline)"),
    "rally/seg-annotations": R2Category(SEG_ANNOTATIONS_DIR, "*.jsonl", "Rally Clip Verdicts (VLM)"),
    "rally/pre-annotations": R2Category(RALLY_PRE_ANNOTATIONS_DIR, "*.jsonl", "Rally Predictions (VLM)"),
    "rally-spot/annotations": R2Category(RALLY_ANNOTATIONS_DIR, "*.jsonl", "Rally Annotations"),
    "rally-spot/pre-annotations": R2Category(RALLY_SPOT_PRE_ANNOTATIONS_DIR, "*.jsonl", "Rally Predictions (SPOT)"),
    "rally-spot/checkpoints": R2Category(RALLY_SPOT_CHECKPOINTS_DIR, "**/*", "Rally Checkpoints"),
    "action/annotations": R2Category(ACTION_ANNOTATIONS_DIR, "*.jsonl", "Action Annotations"),
    "action/pre-annotations": R2Category(ACTION_PRE_ANNOTATIONS_DIR, "*.jsonl", "Action Pre-Annotations"),
    "action/checkpoints": R2Category(ACTION_CHECKPOINTS_DIR, "**/*", "Action Checkpoints"),
    "reid/annotations": R2Category(REID_ANNOTATIONS_DIR, "*.json", "ReID Annotations"),
    "reid/checkpoints": R2Category(REID_CHECKPOINTS_DIR, "**/*", "ReID Checkpoints"),
}

# Registry: kind → (local cuts dir, R2 category name). Single source of truth
# so adding a new cut kind (e.g. indoor/outdoor) is a one-line change here.
class CutKind(NamedTuple):
    local_dir: Path
    r2_category: str

CUT_KINDS: dict[str, CutKind] = {
    "broadcast": CutKind(CUTS_BROADCAST_DIR, "cuts-broadcast"),
    "sideline":  CutKind(CUTS_SIDELINE_DIR, "cuts-sideline"),
}
DEFAULT_CUT_KIND = "broadcast"

# Tuple form for callers that just need the R2 category list.
CUT_R2_CATEGORIES = tuple(k.r2_category for k in CUT_KINDS.values())


def iter_all_cuts() -> Iterator[Path]:
    """Yield every cut video across both split dirs."""
    seen: set[str] = set()
    for d in CUTS_DIRS:
        if not d.exists():
            continue
        for p in d.glob("*.mp4"):
            if p.name in seen:
                continue
            seen.add(p.name)
            yield p


def find_cut(name: str) -> Path | None:
    """Return the first matching cut video across both split dirs."""
    for d in CUTS_DIRS:
        p = d / name
        if p.exists():
            return p
    return None


def cut_kind_of(path: Path) -> str:
    """Return the cut kind ('broadcast' / 'sideline') based on parent dir."""
    parent = path.parent.resolve()
    for kind, info in CUT_KINDS.items():
        if parent == info.local_dir.resolve():
            return kind
    return DEFAULT_CUT_KIND

# ── Web frontend (built React SPA) ───────────────────────────────
FRONTEND_DIST_DIR = Path(__file__).resolve().parent / "web" / "frontend" / "dist"


def _load_env_file(path: Path) -> dict[str, str]:
    """Load key=value pairs from an env file."""
    config: dict[str, str] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    config[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return config


def load_vllm_env() -> dict[str, str]:
    """Load key=value pairs from vllm.env."""
    return _load_env_file(VLLM_ENV_PATH)


def load_r2_env() -> dict[str, str]:
    """Load key=value pairs from r2.env."""
    return _load_env_file(R2_ENV_PATH)


def load_tokens_env() -> dict[str, str]:
    """Load key=value pairs from tokens.env (iOS upload worker URL +
    auth token + library user id, used by the app-export flow)."""
    return _load_env_file(TOKENS_ENV_PATH)


def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    path = PROMPTS_DIR / filename
    with open(path) as f:
        return f.read()


def count_files(directory: Path, pattern: str = "*") -> int:
    """Count files matching *pattern* in *directory* without building a list."""
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.glob(pattern))
