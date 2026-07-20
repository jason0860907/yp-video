"""Export labeled crops as a yp-reid training dataset (Contract A).

A dataset is two small files under ``reid/datasets/<name>/`` — manifest.json
and samples.jsonl — referencing crops by path relative to ``crops_root``
(the reid/ data dir). No symlinks, no copies: the crop store stays the single
source of image bytes, and stems with spaces or CJK never travel through a
format string or a shell. See contracts/reid.py for the authoritative schema.

The trainer imposes invariants that are easy to violate and expensive to
discover mid-training, so plan_export enforces them up front and reports
every excluded player with a reason:

- a TRAIN player needs >= MIN_TRAIN_CROPS_PER_PLAYER crops (positive-pair
  sampling draws a random *other* crop of the same player)
- a TEST player needs >= 1 query and >= 1 gallery row
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from yp_video.config import REID_DATASETS_DIR, REID_DIR
from yp_video.contracts.reid import (
    DATASET_MANIFEST_NAME,
    DATASET_SAMPLES_NAME,
    MIN_TRAIN_CROPS_PER_PLAYER,
    ROLE_GALLERY,
    ROLE_QUERY,
    SPLIT_TEST,
    SPLIT_TRAIN,
    DatasetManifest,
)
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.reid.identity import load_assignments
from yp_video.reid.sessions import SessionGroup
from yp_video.reid.store import crop_dir, masked_crop_dir, reid_path

DATASETS_DIR = REID_DATASETS_DIR

SPLIT_MODES = ("auto", "session", "crops", "all_train")

#: A player split across train and test needs >= 2 train crops and >= 1 of
#: each test role.
MIN_SPLIT_CROPS = 4


@dataclass(frozen=True)
class Sample:
    """One samples.jsonl line (contracts.reid.DatasetSample)."""

    id: str
    path: str  # relative to crops_root
    pid: int
    split: str
    role: str | None
    group: str
    fold: int


@dataclass(frozen=True)
class ExportPlan:
    samples: tuple[Sample, ...]
    #: pid -> {group, name, n_train, n_test}
    players: dict[int, dict]
    #: reason -> the players or ids it excluded
    dropped: dict[str, list[str]] = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    #: resolved group membership at plan time — group ids are not stable
    #: across relabeling, so an export records what they meant.
    groups: dict[str, list[str]] = field(default_factory=dict)


def _candidates(groups: Sequence[SessionGroup], masked: bool):
    """(crops-by-player, dropped) — one entry per labeled crop."""
    by_player: dict[tuple[str, str], list[dict]] = {}
    dropped: dict[str, list[str]] = {}
    seen: set[str] = set()

    for group in groups:
        for stem in group.stems:
            assignments = load_assignments(stem)
            _meta, records = read_jsonl_cached(reid_path(stem))
            source_dir = masked_crop_dir(stem) if masked else crop_dir(stem)
            for record in records:
                name = assignments.get(record["id"])
                if name is None:
                    continue
                if not record.get("crop"):
                    dropped.setdefault("no_crop", []).append(record["id"])
                    continue
                src = source_dir / record["crop"]
                if not src.exists():
                    dropped.setdefault("crop_missing", []).append(record["id"])
                    continue
                if record["id"] in seen:
                    dropped.setdefault("duplicate_id", []).append(record["id"])
                    continue
                seen.add(record["id"])
                by_player.setdefault((group.id, name), []).append({
                    "id": record["id"],
                    "path": str(src.relative_to(REID_DIR)),
                    "frame": record.get("frame") or 0,
                })
    return by_player, dropped


def plan_export(
    groups: Sequence[SessionGroup],
    *,
    split_mode: str = "auto",
    test_ratio: float = 0.25,
    seed: int = 42,
    masked: bool = False,
) -> ExportPlan:
    """Decide every sample without writing anything.

    ``split_mode``:
      - ``session`` — hold out whole sessions. The textbook ReID protocol
        (train and test share no identities), needs >= 2 groups.
      - ``crops`` — per-player stratified split. The only thing that works
        with a single session, but train and test share identities, so its
        numbers flatter the model.
      - ``all_train`` — everything trains, test stays empty (the trainer
        skips evaluation and calls the last epoch best).
      - ``auto`` — ``session`` when there are >= 2 groups, else ``crops``.
    """
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"Unknown split_mode {split_mode!r} (have: {', '.join(SPLIT_MODES)})")
    by_player, dropped = _candidates(groups, masked)
    resolved = "session" if (split_mode == "auto" and len(groups) >= 2) else (
        "crops" if split_mode == "auto" else split_mode
    )
    if resolved == "session" and len(groups) < 2:
        raise ValueError("split_mode='session' needs at least 2 session groups")

    import random

    rng = random.Random(seed)
    group_order = [g.id for g in groups]
    test_groups: set[str] = set()
    if resolved == "session":
        # Hold out whole sessions, smallest first, until test_ratio is met.
        sizes = {g.id: sum(len(v) for (gid, _n), v in by_player.items() if gid == g.id) for g in groups}
        total = sum(sizes.values())
        for gid in sorted(sizes, key=lambda g: sizes[g]):
            if len(test_groups) + 1 >= len(groups):
                break  # never hold out every group
            if sum(sizes[t] for t in test_groups) >= total * test_ratio:
                break
            test_groups.add(gid)
        if not test_groups:
            test_groups.add(min(sizes, key=lambda g: sizes[g]))

    samples: list[Sample] = []
    players: dict[int, dict] = {}
    for pid, ((gid, name), crops) in enumerate(sorted(by_player.items())):
        crops = sorted(crops, key=lambda c: (c["frame"], c["id"]))
        fold = group_order.index(gid)

        if resolved == "session":
            train, test = ([], crops) if gid in test_groups else (crops, [])
        elif resolved == "all_train":
            train, test = crops, []
        else:  # crops
            if len(crops) < MIN_SPLIT_CROPS:
                train, test = crops, []  # too small to split; still useful to train on
            else:
                shuffled = crops[:]
                rng.shuffle(shuffled)
                n_test = max(2, round(len(shuffled) * test_ratio))
                test, train = shuffled[:n_test], shuffled[n_test:]

        if train and len(train) < MIN_TRAIN_CROPS_PER_PLAYER:
            dropped.setdefault("train_singleton", []).append(f"{gid}/{name}")
            train = []
        if test and len(test) < 2:
            dropped.setdefault("test_unpaired", []).append(f"{gid}/{name}")
            test = []
        if not train and not test:
            dropped.setdefault("no_usable_split", []).append(f"{gid}/{name}")
            continue

        for c in train:
            samples.append(Sample(c["id"], c["path"], pid, SPLIT_TRAIN, None, gid, fold))
        for i, c in enumerate(test):
            # First crop per player is the query, the rest gallery.
            samples.append(Sample(c["id"], c["path"], pid, SPLIT_TEST, ROLE_QUERY if i == 0 else ROLE_GALLERY, gid, fold))
        players[pid] = {"group": gid, "name": name, "n_train": len(train), "n_test": len(test)}

    counts = {
        "n_samples": len(samples),
        "n_players": len(players),
        "n_train": sum(1 for s in samples if s.split == SPLIT_TRAIN),
        "n_test": sum(1 for s in samples if s.split == SPLIT_TEST),
        "n_query": sum(1 for s in samples if s.role == ROLE_QUERY),
        "n_gallery": sum(1 for s in samples if s.role == ROLE_GALLERY),
        "n_dropped": sum(len(v) for v in dropped.values()),
    }
    return ExportPlan(
        samples=tuple(samples),
        players=players,
        dropped={k: sorted(v) for k, v in sorted(dropped.items())},
        counts=counts,
        config={"split_mode": resolved, "requested_mode": split_mode, "test_ratio": test_ratio,
                "seed": seed, "masked": masked, "test_groups": sorted(test_groups)},
        groups={g.id: list(g.stems) for g in groups},
    )


def _manifest(plan: ExportPlan, name: str) -> dict:
    """The manifest.json payload, validated against the contract model."""
    return DatasetManifest.model_validate({
        "created_at": time.time(),
        "name": name,
        "crops_root": str(REID_DIR),
        "config": plan.config,
        "players": {str(k): v for k, v in plan.players.items()},
        "groups": plan.groups,
        "counts": plan.counts,
        "dropped": plan.dropped,
    }).model_dump()


def write_export(plan: ExportPlan, root: Path) -> dict:
    """Materialise manifest.json + samples.jsonl, atomically via a staging dir."""
    staging = root.parent / f".{root.name}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    lines = [json.dumps(asdict(s), ensure_ascii=False) for s in plan.samples]
    (staging / DATASET_SAMPLES_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (staging / DATASET_MANIFEST_NAME).write_text(
        json.dumps(_manifest(plan, root.name), ensure_ascii=False, indent=1), encoding="utf-8"
    )

    if root.exists():
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, root)
    return {"root": str(root), **plan.counts}


def export_manifest_jsonl(plan: ExportPlan) -> str:
    """The plan as ndjson with a _meta header — the project's download shape."""
    head = {
        "_meta": True,
        "type": "reid_dataset_plan",
        "crops_root": str(REID_DIR),
        "config": plan.config,
        "counts": plan.counts,
        "groups": plan.groups,
        "players": {str(k): v for k, v in plan.players.items()},
        "dropped": plan.dropped,
    }
    lines = [json.dumps(head, ensure_ascii=False)]
    lines += [json.dumps(asdict(s), ensure_ascii=False) for s in plan.samples]
    return "\n".join(lines) + "\n"


def list_datasets() -> list[dict]:
    """Exported datasets on disk, newest first."""
    if not DATASETS_DIR.exists():
        return []
    out = []
    for d in DATASETS_DIR.iterdir():
        manifest_path = d / DATASET_MANIFEST_NAME
        if not d.is_dir() or not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({
            "name": d.name,
            "created_at": manifest.get("created_at", 0),
            "counts": manifest.get("counts", {}),
            "config": manifest.get("config", {}),
        })
    return sorted(out, key=lambda d: -d["created_at"])
