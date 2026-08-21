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


def test_context_cut_places_the_person_inside_the_photo():
    import numpy as np

    from yp_video.extraction.identify import _context_cut

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cut, box = _context_cut(frame, [900, 400, 1000, 700])

    # Wider than the person, and the person sits in the middle of it.
    assert cut.shape[0] <= 448 and cut.shape[1] <= 448
    x0, y0, x1, y1 = box
    assert 0.0 <= x0 < x1 <= 1.0
    assert 0.0 <= y0 < y1 <= 1.0
    assert abs(((x0 + x1) / 2) - 0.5) < 0.02
    assert abs(((y0 + y1) / 2) - 0.5) < 0.02
    # The person takes about a third of each axis (the context scale).
    assert 0.25 < (x1 - x0) < 0.4


def test_context_cut_clamps_at_the_frame_edge():
    import numpy as np

    from yp_video.extraction.identify import _context_cut

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # A person hard against the left edge: the cut cannot extend past it, so
    # they end up off-centre and the box must say so rather than claiming the
    # middle.
    cut, box = _context_cut(frame, [0, 400, 80, 700])
    assert cut is not None
    assert box[0] == 0.0
    # Off-centre towards the edge they are pinned against.
    assert ((box[0] + box[2]) / 2) < 0.4


def test_context_cut_rejects_a_degenerate_box():
    import numpy as np

    from yp_video.extraction.identify import _context_cut

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert _context_cut(frame, [100, 100, 100, 300]) is None


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
