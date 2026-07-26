"""Stamp the rally fingerprint onto tracklets cut before it existed.

A tracklet's identity is ``"{rally_id}:{track_id}"`` and ``rally_id`` is
positional, so re-labelling rallies renumbers every key and each stored one
quietly points at a different player. The tracks header records a fingerprint
of the spans it was cut from precisely so a reader can notice (see
extraction/prerequisites._tracks_stale).

Videos tracked before that existed have no fingerprint, and the freshness
check reads them as "unknown, not stale" — which is honest but means the
warning can never fire for them, on exactly the labels most worth protecting.

Writing today's fingerprint would ASSERT the spans have not moved since, and
that is only true if the rally annotation is older than the tracks file. This
checks that, per video, and refuses the ones it cannot prove. Nothing is
guessed: an unprovable video keeps its silence rather than gaining a claim.

    uv run python scripts/backfill_rally_fingerprint.py
    uv run python scripts/backfill_rally_fingerprint.py --apply
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yp_video.config import VIDEOS_DIR  # noqa: E402
from yp_video.core.jsonl import read_jsonl, read_jsonl_header, write_jsonl  # noqa: E402
from yp_video.core.rallies import (  # noqa: E402
    load_rallies,
    rally_annotation_path,
    rally_fingerprint,
)
from yp_video.tracklets.store import tracks_path  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    stem: str
    fingerprint: str | None
    #: Why this video cannot be stamped, or None when it can.
    refused: str | None


def survey() -> list[Candidate]:
    out: list[Candidate] = []
    for path in sorted(VIDEOS_DIR.glob("tracks/*_tracks.jsonl")):
        stem = path.name[: -len("_tracks.jsonl")]
        header = read_jsonl_header(path)
        if (header.get("rallies") or {}).get("fingerprint"):
            continue
        rally = rally_annotation_path(stem)
        if rally is None:
            out.append(Candidate(stem, None, "no rally source"))
            continue
        if rally.stat().st_mtime > path.stat().st_mtime:
            # The spans were edited after tracking ran, so what these
            # tracklets were cut from is genuinely unknown — which is the one
            # state a fingerprint must never claim to know.
            out.append(Candidate(stem, None, "rallies edited after tracking"))
            continue
        out.append(Candidate(stem, rally_fingerprint(stem), None))
    return out


def stamp(stem: str, fingerprint: str) -> None:
    path = tracks_path(stem)
    meta, records = read_jsonl(path)
    meta["rallies"] = {"count": len(load_rallies(stem)), "fingerprint": fingerprint}
    write_jsonl(path, meta, records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the headers")
    args = parser.parse_args()

    candidates = survey()
    provable = [c for c in candidates if c.refused is None]
    refused = [c for c in candidates if c.refused is not None]

    if not candidates:
        print("every tracks file already records its rally fingerprint")
        return 0

    print(f"{len(candidates)} video(s) without a fingerprint\n")
    for c in provable:
        print(f"  ✓ {c.stem[:40]:42} {c.fingerprint}")
    for c in refused:
        print(f"  ✗ {c.stem[:40]:42} {c.refused}")

    if not args.apply:
        print(f"\nDry run — {len(provable)} stampable. Re-run with --apply.")
        return 0

    for c in provable:
        assert c.fingerprint is not None
        stamp(c.stem, c.fingerprint)
    print(f"\nStamped {len(provable)}; left {len(refused)} unproven.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
