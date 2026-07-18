"""Export labeled crops as a CLIP-ReIdent training dataset.

Their loaders build image paths as ``"{img_path}/{folder}/{img_id}.jpeg"``
with the extension hardcoded (clipreid/dataset.py:50), and train.py passes
``img_path=config.data_dir`` while reading ``"{data_dir}/train_df.csv"`` — so
the CSV and the image folders are siblings under one root.

Our crops are ``.jpg``, so the folders are filled with SYMLINKS to the real
files under reid/crops/. Copying thousands of files would duplicate hundreds
of MB of recomputable data and go stale the moment a crop is re-extracted.

Folder names are opaque tokens (``v000``) rather than video stems: stems carry
spaces and CJK ("0104排島臨打 1") and flow through an unquoted str.format
straight into cv2.imread. The token -> stem mapping lives in meta.json.

The loaders impose invariants that are easy to violate and expensive to
discover mid-training, so plan_export enforces them up front and reports
every excluded player with a reason:

- a TRAIN player needs >= 2 crops (__getitem__ draws a random *other* crop of
  the same player; np.random.choice on an empty array raises)
- a TEST player needs >= 1 query and >= 1 gallery row
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from yp_video.config import REID_DATASETS_DIR
from yp_video.core.jsonl import read_jsonl_cached
from yp_video.reid.identity import load_assignments
from yp_video.reid.sessions import SessionGroup
from yp_video.reid.store import crop_dir, masked_crop_dir, reid_path

DATASETS_DIR = REID_DATASETS_DIR

#: Bumped when the CSV/meta layout changes in a way a reader must notice.
CONTRACT_VERSION = 1

SPLIT_MODES = ("auto", "session", "crops", "all_train")

#: A player split across train and test needs >= 2 train crops and >= 1 of
#: each test type.
MIN_SPLIT_CROPS = 4


@dataclass(frozen=True)
class ExportRow:
    img_id: str
    folder: str
    player: int
    game: str
    split: str
    img_type: str
    fold: int


@dataclass(frozen=True)
class ExportPlan:
    rows: tuple[ExportRow, ...]
    #: pid -> {group, name, n_train, n_test}
    players: dict[int, dict]
    #: "v000" -> stem
    folders: dict[str, str]
    #: img_id -> source crop path (absolute), for write_export
    sources: dict[str, str] = field(default_factory=dict)
    #: reason -> the players or ids it excluded
    dropped: dict[str, list[str]] = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    #: resolved group membership at plan time — group ids are not stable
    #: across relabeling, so an export records what they meant.
    groups: dict[str, list[str]] = field(default_factory=dict)


def _candidates(groups: Sequence[SessionGroup], masked: bool):
    """(rows-in-waiting, folders, dropped) — one entry per labeled crop."""
    folders: dict[str, str] = {}
    by_player: dict[tuple[str, str], list[dict]] = {}
    dropped: dict[str, list[str]] = {}
    seen: set[str] = set()

    for group in groups:
        for stem in group.stems:
            token = f"v{len(folders):03d}"
            folders[token] = stem
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
                by_player.setdefault((group.id, name), []).append(
                    {"img_id": record["id"], "folder": token, "src": str(src.resolve()),
                     "frame": record.get("frame") or 0}
                )
    return by_player, folders, dropped


def plan_export(
    groups: Sequence[SessionGroup],
    *,
    split_mode: str = "auto",
    test_ratio: float = 0.25,
    seed: int = 42,
    masked: bool = False,
) -> ExportPlan:
    """Decide every CSV row without writing anything.

    ``split_mode``:
      - ``session`` — hold out whole sessions. The textbook ReID protocol
        (train and test share no identities), needs >= 2 groups.
      - ``crops`` — per-player stratified split. The only thing that works
        with a single session, but train and test share identities, so its
        numbers flatter the model.
      - ``all_train`` — everything trains; a minimal mirrored test split is
        still emitted because train.py evaluates on ``split == "test"``
        even when train_on_all is set, and an empty one crashes it.
      - ``auto`` — ``session`` when there are >= 2 groups, else ``crops``.
    """
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"Unknown split_mode {split_mode!r} (have: {', '.join(SPLIT_MODES)})")
    by_player, folders, dropped = _candidates(groups, masked)
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

    rows: list[ExportRow] = []
    players: dict[int, dict] = {}
    sources: dict[str, str] = {}
    for pid, ((gid, name), crops) in enumerate(sorted(by_player.items())):
        crops = sorted(crops, key=lambda c: (c["frame"], c["img_id"]))
        fold = group_order.index(gid)

        if resolved == "session":
            split_of = "test" if gid in test_groups else "train"
            train, test = ([], crops) if split_of == "test" else (crops, [])
        elif resolved == "all_train":
            train, test = crops, (crops[:2] if len(crops) >= 2 else [])
        else:  # crops
            if len(crops) < MIN_SPLIT_CROPS:
                train, test = crops, []  # too small to split; still useful to train on
            else:
                shuffled = crops[:]
                rng.shuffle(shuffled)
                n_test = max(2, round(len(shuffled) * test_ratio))
                test, train = shuffled[:n_test], shuffled[n_test:]

        if train and len(train) < 2:
            dropped.setdefault("train_singleton", []).append(f"{gid}/{name}")
            train = []
        if test and len(test) < 2:
            dropped.setdefault("test_unpaired", []).append(f"{gid}/{name}")
            test = []
        if not train and not test:
            dropped.setdefault("no_usable_split", []).append(f"{gid}/{name}")
            continue

        for c in train:
            rows.append(ExportRow(c["img_id"], c["folder"], pid, gid, "train", "g", fold))
            sources[c["img_id"]] = c["src"]
        for i, c in enumerate(test):
            # First crop per player is the query, the rest gallery — the same
            # convention preprocess_data.py uses.
            rows.append(ExportRow(c["img_id"], c["folder"], pid, gid, "test", "q" if i == 0 else "g", fold))
            sources[c["img_id"]] = c["src"]
        players[pid] = {"group": gid, "name": name, "n_train": len(train), "n_test": len(test)}

    counts = {
        "n_rows": len(rows),
        "n_players": len(players),
        "n_train": sum(1 for r in rows if r.split == "train"),
        "n_test": sum(1 for r in rows if r.split == "test"),
        "n_query": sum(1 for r in rows if r.split == "test" and r.img_type == "q"),
        "n_gallery": sum(1 for r in rows if r.split == "test" and r.img_type == "g"),
        "n_dropped": sum(len(v) for v in dropped.values()),
    }
    return ExportPlan(
        rows=tuple(rows),
        players=players,
        folders=folders,
        sources=sources,
        dropped={k: sorted(v) for k, v in sorted(dropped.items())},
        counts=counts,
        config={"split_mode": resolved, "requested_mode": split_mode, "test_ratio": test_ratio,
                "seed": seed, "masked": masked, "test_groups": sorted(test_groups)},
        groups={g.id: list(g.stems) for g in groups},
    )


COLUMNS = ("img_id", "folder", "player", "game", "split", "img_type", "fold")


def write_export(
    plan: ExportPlan,
    root: Path,
    *,
    link: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Materialise train_df.csv + meta.json + the image folders.

    Symlinks are absolute so the tree survives being referenced from
    elsewhere; it is NOT movable or tarrable without --dereference, which is
    the accepted trade for not duplicating the crops. A single OSError flips
    the whole run to copying rather than leaving a half-linked tree.
    """
    staging = root.parent / f".{root.name}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for token in plan.folders:
        (staging / token).mkdir(exist_ok=True)

    total = len(plan.rows)
    # Probe once: a filesystem that refuses one symlink refuses them all, and
    # deciding up front is what keeps the tree from ending up half-linked.
    mode = "copy"
    if link and plan.rows:
        probe = staging / ".link-probe"
        try:
            probe.symlink_to(Path(plan.sources[plan.rows[0].img_id]))
            probe.unlink()
            mode = "symlink"
        except OSError:
            mode = "copy"

    for i, row in enumerate(plan.rows):
        src = Path(plan.sources[row.img_id])
        dst = staging / row.folder / f"{row.img_id}.jpeg"
        if mode == "symlink":
            dst.symlink_to(src)
        else:
            shutil.copy2(src, dst)
        if on_progress and (i % 200 == 0 or i + 1 == total):
            on_progress(i + 1, total)

    with open(staging / "train_df.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows([getattr(row, c) for c in COLUMNS] for row in plan.rows)

    meta = {
        "contract_version": CONTRACT_VERSION,
        "created_at": time.time(),
        "link_mode": mode,
        "folders": plan.folders,
        "players": {str(k): v for k, v in plan.players.items()},
        "groups": plan.groups,
        "config": plan.config,
        "counts": plan.counts,
        "dropped": plan.dropped,
    }
    (staging / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    if root.exists():
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, root)
    return {"root": str(root), "n_images": total, "link_mode": mode, **plan.counts}


def export_manifest_jsonl(plan: ExportPlan) -> str:
    """The plan as ndjson with a _meta header — the project's download shape."""
    head = {
        "_meta": True,
        "type": "reid_dataset_plan",
        "contract_version": CONTRACT_VERSION,
        "config": plan.config,
        "counts": plan.counts,
        "folders": plan.folders,
        "groups": plan.groups,
        "players": {str(k): v for k, v in plan.players.items()},
        "dropped": plan.dropped,
    }
    lines = [json.dumps(head, ensure_ascii=False)]
    lines += [json.dumps(asdict(r), ensure_ascii=False) for r in plan.rows]
    return "\n".join(lines) + "\n"


def list_datasets() -> list[dict]:
    """Exported datasets on disk, newest first."""
    if not DATASETS_DIR.exists():
        return []
    out = []
    for d in DATASETS_DIR.iterdir():
        meta_path = d / "meta.json"
        if not d.is_dir() or not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({
            "name": d.name,
            "created_at": meta.get("created_at", 0),
            "counts": meta.get("counts", {}),
            "config": meta.get("config", {}),
        })
    return sorted(out, key=lambda d: -d["created_at"])
