"""Centralized project configuration.

Single source of truth for all project paths. Eliminates hardcoded
Path(__file__).parent.parent chains throughout the codebase (DIP).
"""

from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple


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
ACTIONFORMER_DIR = PROJECT_ROOT / "actionformer"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
VLLM_ENV_PATH = PROJECT_ROOT / "vllm.env"
R2_ENV_PATH = PROJECT_ROOT / "r2.env"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

# ── TAD paths ─────────────────────────────────────────────────────
TAD_PKG_DIR = Path(__file__).resolve().parent / "tad"
TAD_CONFIGS_DIR = TAD_PKG_DIR / "configs"

# ── User data directories (~/videos) ─────────────────────────────
VIDEOS_DIR = Path.home() / "videos"
RAW_VIDEOS_DIR = VIDEOS_DIR / "raw-videos"
# Cuts split by capture style — picked in the UI when exporting from Cut page.
# "broadcast" = TV-style with replays / overlays / cuts (e.g. VNL, U19, TPVL)
# "sideline"  = amateur side-court / tripod recording (e.g. 臨打 practice)
CUTS_BROADCAST_DIR = VIDEOS_DIR / "cuts-broadcast"
CUTS_SIDELINE_DIR = VIDEOS_DIR / "cuts-sideline"
CUTS_DIRS = (CUTS_BROADCAST_DIR, CUTS_SIDELINE_DIR)
SEG_ANNOTATIONS_DIR = VIDEOS_DIR / "seg-annotations"
PRE_ANNOTATIONS_DIR = VIDEOS_DIR / "rally-pre-annotations"
ANNOTATIONS_DIR = VIDEOS_DIR / "rally-annotations"
PREDICTIONS_DIR = VIDEOS_DIR / "tad-predictions"
RALLY_CLIPS_DIR = VIDEOS_DIR / "rally_clips"
FEATURES_DIR = VIDEOS_DIR / "tad-features"
TAD_FEATURES_DIR = FEATURES_DIR / "vjepa-b"  # default (ViT-B); use tad_features_dir() for other sizes
TAD_CHECKPOINTS_DIR = VIDEOS_DIR / "tad-checkpoints"
TAD_ANNOTATIONS_DIR = VIDEOS_DIR / "tad-annotations"
TAD_ANNOTATIONS_FILE = TAD_ANNOTATIONS_DIR / "volleyball_anno.json"

# ── VLM (Qwen3.5-VL fine-tune) paths ─────────────────────────────
VLM_DIR = VIDEOS_DIR / "vlm"
VLM_MANIFEST_FILE = VLM_DIR / "rally_windows.jsonl"
VLM_CHECKPOINTS_DIR = VIDEOS_DIR / "vlm-checkpoints"

# R2 category → local directory + glob pattern mapping
class R2Category(NamedTuple):
    local_dir: Path
    glob_pattern: str

R2_CATEGORIES: dict[str, R2Category] = {
    "videos": R2Category(RAW_VIDEOS_DIR, "*.mp4"),
    "cuts-broadcast": R2Category(CUTS_BROADCAST_DIR, "*.mp4"),
    "cuts-sideline": R2Category(CUTS_SIDELINE_DIR, "*.mp4"),
    "seg-annotations": R2Category(SEG_ANNOTATIONS_DIR, "*.jsonl"),
    "rally-pre-annotations": R2Category(PRE_ANNOTATIONS_DIR, "*.jsonl"),
    "rally-annotations": R2Category(ANNOTATIONS_DIR, "*.jsonl"),
    "tad-predictions": R2Category(PREDICTIONS_DIR, "*.jsonl"),
    "tad-features": R2Category(FEATURES_DIR, "**/*.npy"),
    "rally_clips": R2Category(RALLY_CLIPS_DIR, "*.mp4"),
    "tad-checkpoints": R2Category(TAD_CHECKPOINTS_DIR, "**/*"),
    "vlm-checkpoints": R2Category(VLM_CHECKPOINTS_DIR, "**/*"),
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

# ── Web static assets ────────────────────────────────────────────
STATIC_DIR = Path(__file__).resolve().parent / "web" / "static"


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
