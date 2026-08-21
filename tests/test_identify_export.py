"""The identify export: per-unit crops, and a tree a client can re-cut.

The export ships units and a linkage tree instead of ready-made clusters, so
a viewer can pick any granularity without another GPU pass. That only works
if the client's cut agrees with scipy's — the contract test below pins the
algorithm the iOS side has to implement.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster

from yp_video.reid.identity import Unit, linkage_tree, unit_centroids
from yp_video.extraction.identify import _with_crops


def _unit(key, rows, events=None):
    return Unit(key=key, event_ids=tuple(events or [f"e{r}" for r in rows]), rows=tuple(rows))


def _records(count, crop=lambda i: f"{i}.jpg"):
    return [{"id": f"e{i}", "crop": crop(i)} for i in range(count)]


def _write_crops(tmp_path, names):
    for name in names:
        (tmp_path / name).write_bytes(b"x")


def _patch_crop_dir(monkeypatch, tmp_path):
    import yp_video.extraction.store as store
    monkeypatch.setattr(store, "crop_dir", lambda stem: tmp_path)


def test_representatives_are_nearest_the_units_own_centroid(monkeypatch, tmp_path):
    _patch_crop_dir(monkeypatch, tmp_path)
    _write_crops(tmp_path, ["0.jpg", "1.jpg", "2.jpg"])
    # Three crops spread over a quarter turn. Row 1 sits between the other
    # two, so it is closest to their mean and must lead; row 0 is the most
    # extreme and drops out when only two representatives are kept.
    matrix = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    exported, kept = _with_crops("m", _records(3), matrix, [_unit("t:1:1", [0, 1, 2])], 2)

    assert [u.key for u in exported] == ["t:1:1"]
    assert [p.name for p in exported[0].crop_paths] == ["1.jpg", "2.jpg"]
    assert len(kept) == 1


def test_units_with_no_crop_on_disk_are_dropped_before_the_tree(monkeypatch, tmp_path):
    _patch_crop_dir(monkeypatch, tmp_path)
    _write_crops(tmp_path, ["0.jpg", "2.jpg"])  # unit 1's file never landed
    matrix = np.eye(3)
    units = [_unit("a", [0]), _unit("b", [1]), _unit("c", [2])]

    exported, kept = _with_crops("m", _records(3), matrix, units, 3)

    # The survivors keep their relative order — they become linkage leaves,
    # so index i in `exported` must be index i in `kept`.
    assert [u.key for u in exported] == ["a", "c"]
    assert [u.key for u in kept] == ["a", "c"]


def test_a_unit_whose_record_has_no_crop_field_is_dropped(monkeypatch, tmp_path):
    _patch_crop_dir(monkeypatch, tmp_path)
    _write_crops(tmp_path, ["0.jpg"])
    records = [{"id": "e0", "crop": "0.jpg"}, {"id": "e1", "crop": None}]

    exported, _kept = _with_crops("m", records, np.eye(2), [_unit("a", [0]), _unit("b", [1])], 3)

    assert [u.key for u in exported] == ["a"]


def test_linkage_has_one_row_per_merge():
    matrix = unit_centroids(
        [_unit(f"u{i}", [i]) for i in range(5)],
        np.random.default_rng(0).normal(size=(5, 8)),
    )
    tree = linkage_tree(matrix)
    assert tree.shape == (4, 4)
    assert linkage_tree(matrix[:1]) is None


def _cut_like_a_client(n, merges, threshold):
    """The algorithm the iOS side implements, in Python.

    A node becomes one flat cluster when the largest merge distance anywhere
    in its subtree is within the cutoff — scipy's 'distance' criterion, stated
    without assuming average linkage happens to be monotonic.
    """
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    max_dist = [0.0] * (2 * n - 1)
    rep = list(range(n)) + [0] * (n - 1)
    for i, (a, b, distance, _count) in enumerate(merges):
        a, b = int(a), int(b)
        node = n + i
        max_dist[node] = max(distance, max_dist[a], max_dist[b])
        rep[node] = rep[a]
        if max_dist[node] <= threshold:
            ra, rb = find(rep[a]), find(rep[b])
            if ra != rb:
                parent[rb] = ra
    return [find(leaf) for leaf in range(n)]


def _partition(labels):
    groups = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, []).append(i)
    return sorted(tuple(v) for v in groups.values())


def test_a_client_side_cut_matches_scipy_at_every_threshold():
    rng = np.random.default_rng(20260821)
    for _ in range(20):
        n = int(rng.integers(2, 40))
        matrix = rng.normal(size=(n, 16))
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        tree = linkage_tree(matrix)
        for threshold in np.linspace(0.0, float(tree[:, 2].max()) * 1.1, 25):
            assert _partition(_cut_like_a_client(n, tree, threshold)) == _partition(
                fcluster(tree, t=threshold, criterion="distance")
            ), f"mismatch at n={n} t={threshold}"
