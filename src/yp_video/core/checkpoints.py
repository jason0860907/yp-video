"""The path grammar every checkpoint store speaks.

Checkpoint roots all live under ``VIDEOS_DIR`` (action/checkpoints,
rally-spot/checkpoints, reid/checkpoints, ...), and each store accepts the
same three spellings: an absolute path, a ``VIDEOS_DIR``-relative display ref
(what ``checkpoint_ref`` produces), or a name/path relative to its own root.
That grammar lives here once. What a checkpoint *is* — a ``.pt`` file, a
manifest package, a ``model.json`` directory — stays with its store; two of
them used to carry word-for-word copies of these functions instead.
"""

from __future__ import annotations

from pathlib import Path

from yp_video.config import VIDEOS_DIR


def checkpoint_ref(path: Path) -> str:
    """Display/API ref: the path relative to ``VIDEOS_DIR`` when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(VIDEOS_DIR.resolve()))
    except ValueError:
        return str(resolved)


def resolve_ref(value: str | Path, root: Path) -> Path:
    """Resolve a checkpoint ref to an absolute path, without validating it.

    Absolute paths pass through unchanged. A relative path that starts with
    ``root``'s own path relative to ``VIDEOS_DIR`` (the ``checkpoint_ref``
    format, e.g. ``rally-spot/checkpoints/<run>/...``) is taken relative to
    ``VIDEOS_DIR``; any other relative path is taken relative to ``root``.
    Existence and containment are the caller's checks — see ``is_under``.
    """
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    root_rel = root.relative_to(VIDEOS_DIR).parts
    if path.parts[: len(root_rel)] == root_rel:
        return VIDEOS_DIR / path
    return root / path


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
