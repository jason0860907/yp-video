"""The dependency rule that keeps actor association and ReID separable.

Before the split, ``detector`` imported the association policy inside a
function to dodge a circular import, and the identity layer imported an actor
verdict type because one JSON file carried both kinds of label. Neither was
visible from any single file, so both survived review. This test makes the
layering a thing that fails loudly instead.

    person       perception primitives; depends on nobody
    tracklets    who is on court over time; reads person, nothing above
    extraction.store   the on-disk shape of extraction output; a leaf
    actor        who acted;  may read person/tracklets/extraction.store
    reid         who they are; same, and never actor (nor actor them)
    extraction.* the roof; the only place allowed to combine them all
"""

from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "yp_video"

#: package prefix → the yp_video prefixes it may NOT import.
#: `extraction` is forbidden wholesale rather than module by module: a new
#: module in the roof package should be off-limits by DEFAULT, or the rule
#: only holds until somebody forgets to add one here.
FORBIDDEN: dict[str, tuple[str, ...]] = {
    "person": ("yp_video.actor", "yp_video.reid", "yp_video.tracklets",
               "yp_video.extraction", "yp_video.web"),
    # Tracking reads rallies and frames. Reading the action annotation is what
    # made it wait for a stage it does not depend on; reading records would
    # make it wait for one that depends on IT.
    "tracklets": ("yp_video.actor", "yp_video.reid", "yp_video.extraction",
                  "yp_video.web"),
    "actor": ("yp_video.reid", "yp_video.web", "yp_video.extraction"),
    "reid": ("yp_video.actor", "yp_video.web", "yp_video.extraction"),
}

#: The one thing the roof publishes downward: where extraction output lives.
ALLOWED = ("yp_video.extraction.store",)

#: extraction/store.py is the leaf its consumers read; it may not reach back.
STORE_FORBIDDEN = ("yp_video.person", "yp_video.tracklets", "yp_video.actor",
                   "yp_video.reid", "yp_video.web")


def _imported_modules(path: Path) -> set[str]:
    """Every yp_video module a file imports, including inside functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(
                alias.name for alias in node.names
                if alias.name.startswith("yp_video")
            )
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            if module.startswith("yp_video"):
                found.add(module)
                found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


def _covers(prefix: str, imported: str) -> bool:
    return imported == prefix or imported.startswith(prefix + ".")


def _violations(path: Path, forbidden: tuple[str, ...]) -> list[str]:
    return sorted(
        f"{path.relative_to(SRC)} → {imported}"
        for imported in _imported_modules(path)
        if any(_covers(p, imported) for p in forbidden)
        and not any(_covers(a, imported) for a in ALLOWED)
    )


class LayeringTests(unittest.TestCase):
    def test_packages_only_depend_downward(self) -> None:
        offenders: list[str] = []
        for package, forbidden in FORBIDDEN.items():
            for path in sorted((SRC / package).rglob("*.py")):
                offenders.extend(_violations(path, forbidden))
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_extraction_store_stays_a_leaf(self) -> None:
        self.assertEqual(
            _violations(SRC / "extraction" / "store.py", STORE_FORBIDDEN), []
        )

    def test_the_rule_would_actually_catch_a_violation(self) -> None:
        """A layering test that can't fail is decoration.

        Deferring an import into a function body is exactly how the old
        detector→association cycle hid, so the scan must see those too.
        """
        source = (
            "from yp_video.reid.store import players_path\n"
            "import yp_video.reid.identity\n"
            "def f():\n"
            "    from yp_video.reid import identity\n"
        )
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "sneaky.py"
            path.write_text(source, encoding="utf-8")
            found = _imported_modules(path)

        self.assertEqual(
            found,
            {
                "yp_video.reid.store",
                "yp_video.reid.store.players_path",
                "yp_video.reid.identity",
                "yp_video.reid",
            },
        )


if __name__ == "__main__":
    unittest.main()
