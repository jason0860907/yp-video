"""Centralized project configuration.

Single source of truth for all project paths. Eliminates hardcoded
Path(__file__).parent.parent chains throughout the codebase (DIP).
"""

from pathlib import Path


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
OPENTAD_DIR = PROJECT_ROOT / "OpenTAD"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
VLLM_ENV_PATH = PROJECT_ROOT / "vllm.env"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

# ── TAD paths ─────────────────────────────────────────────────────
TAD_PKG_DIR = Path(__file__).resolve().parent / "tad"
TAD_CONFIGS_DIR = TAD_PKG_DIR / "configs"
TAD_DATA_DIR = TAD_PKG_DIR / "data"
TAD_FEATURES_DIR = TAD_DATA_DIR / "features"
TAD_ANNOTATIONS_DIR = TAD_DATA_DIR / "annotations"
TAD_ANNOTATIONS_FILE = TAD_ANNOTATIONS_DIR / "volleyball_anno.json"
TAD_CHECKPOINTS_DIR = TAD_PKG_DIR / "checkpoints"

# ── User data directories (~/videos) ─────────────────────────────
VIDEOS_DIR = Path.home() / "videos"
CUTS_DIR = VIDEOS_DIR / "cuts"
SEG_ANNOTATIONS_DIR = VIDEOS_DIR / "seg-annotations"
PRE_ANNOTATIONS_DIR = VIDEOS_DIR / "rally-pre-annotations"
ANNOTATIONS_DIR = VIDEOS_DIR / "rally-annotations"
PREDICTIONS_DIR = VIDEOS_DIR / "tad-predictions"

# ── Web static assets ────────────────────────────────────────────
STATIC_DIR = Path(__file__).resolve().parent / "web" / "static"


def load_vllm_env() -> dict[str, str]:
    """Load key=value pairs from vllm.env."""
    config: dict[str, str] = {}
    try:
        with open(VLLM_ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    config[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return config


def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    path = PROMPTS_DIR / filename
    with open(path) as f:
        return f.read()
