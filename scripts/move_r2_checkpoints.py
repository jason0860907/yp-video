"""One-off: mirror the checkpoint-package move on R2.

Local packages moved to ``videos/spot/checkpoints`` (one directory for every
SPOT recipe); this copies the R2 objects from the two old prefixes to
``spot/checkpoints/``, re-uploads the restamped ``manifest.json`` /
``config.json`` over the copied (pre-registry) ones, and — only with
``--delete`` — removes the old prefixes once every object is verified under
the new one::

    cd yp-video && uv run python scripts/move_r2_checkpoints.py           # copy + restamp
    cd yp-video && uv run python scripts/move_r2_checkpoints.py --delete  # then drop old prefixes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yp_video.config import SPOT_CHECKPOINTS_DIR  # noqa: E402
from yp_video.web.r2_client import r2_client  # noqa: E402

OLD_PREFIXES = ("action/checkpoints/", "rally-spot/checkpoints/")
NEW_PREFIX = "spot/checkpoints/"


def new_key(old_key: str) -> str:
    return NEW_PREFIX + old_key.split("/checkpoints/", 1)[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--delete", action="store_true", help="remove the old prefixes after verifying the copy")
    args = parser.parse_args()
    if not r2_client.configured:
        raise SystemExit("R2 is not configured")
    client, bucket = r2_client._get_client(), r2_client.bucket

    old_keys = [o["key"] for prefix in OLD_PREFIXES for o in r2_client.list_objects(prefix)]
    present = {o["key"] for o in r2_client.list_objects(NEW_PREFIX)}
    copied = 0
    for key in old_keys:
        if new_key(key) in present:
            continue
        client.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": key}, Key=new_key(key))
        copied += 1
    present = {o["key"] for o in r2_client.list_objects(NEW_PREFIX)}
    missing = [k for k in old_keys if new_key(k) not in present]
    print(f"old objects {len(old_keys)}, copied now {copied}, under {NEW_PREFIX}: {len(present)}, missing {len(missing)}")
    if missing:
        raise SystemExit("copy incomplete; not deleting anything")

    uploaded = 0
    for package in sorted(p for p in SPOT_CHECKPOINTS_DIR.iterdir() if p.is_dir()):
        for name in ("manifest.json", "config.json"):
            local = package / name
            if local.is_file():
                r2_client.upload_file(local, f"{NEW_PREFIX}{package.name}/{name}")
                uploaded += 1
    print(f"re-uploaded {uploaded} restamped metadata files")

    if args.delete:
        deleted = r2_client.delete_objects(old_keys)
        print(f"deleted {deleted} objects under {OLD_PREFIXES}")
    else:
        print("old prefixes kept; rerun with --delete to remove them")


if __name__ == "__main__":
    main()
