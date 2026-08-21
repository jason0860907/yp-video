"""Contracts for complete frames and the client-cuttable suggestion tree."""

import numpy as np
from scipy.cluster.hierarchy import fcluster

from yp_video.extraction.identify import _full_frame
from yp_video.reid.identity import linkage_tree


def test_landscape_export_keeps_the_complete_frame_and_ratio():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    image, box = _full_frame(frame, [900, 400, 1000, 700])

    assert image.shape == (252, 448, 3)
    assert image.shape[1] / image.shape[0] == 448 / 252
    assert box == (900 / 1920, 400 / 1080, 1000 / 1920, 700 / 1080)


def test_portrait_export_keeps_the_complete_frame_and_ratio():
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)

    image, box = _full_frame(frame, [100, 300, 500, 1500])

    assert image.shape == (448, 252, 3)
    assert box == (100 / 1080, 300 / 1920, 500 / 1080, 1500 / 1920)


def test_box_is_clamped_to_the_complete_frame_edges():
    frame = np.zeros((400, 800, 3), dtype=np.uint8)

    image, box = _full_frame(frame, [-20, 100, 900, 450])

    assert image.shape == (224, 448, 3)
    assert box == (0.0, 0.25, 1.0, 1.0)


def test_small_source_returns_the_exact_complete_frame():
    frame = np.arange(300 * 400 * 3, dtype=np.uint8).reshape((300, 400, 3))

    image, _box = _full_frame(frame, [100, 80, 200, 240])

    assert image is frame
    assert np.array_equal(image, frame)


def test_invalid_boxes_are_rejected():
    frame = np.zeros((400, 800, 3), dtype=np.uint8)

    assert _full_frame(frame, [100, 100, 100, 300]) is None
    assert _full_frame(frame, [-100, 100, -10, 300]) is None
    assert _full_frame(frame, [100, 100, float("nan"), 300]) is None
    assert _full_frame(frame, [100, 100, 300]) is None


def _cut_like_the_app(unit_count, merges, threshold):
    parent = list(range(unit_count))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    subtree_max = [0.0] * (2 * unit_count - 1)
    leaf = list(range(unit_count)) + [0] * (unit_count - 1)
    for index, (left, right, distance, _size) in enumerate(merges):
        left, right = int(left), int(right)
        node = unit_count + index
        subtree_max[node] = max(distance, subtree_max[left], subtree_max[right])
        leaf[node] = leaf[left]
        if subtree_max[node] <= threshold:
            root_left, root_right = find(leaf[left]), find(leaf[right])
            if root_left != root_right:
                parent[root_right] = root_left
    return [find(index) for index in range(unit_count)]


def _partition(labels):
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    return sorted(tuple(group) for group in groups.values())


def test_client_side_slider_cut_matches_scipy():
    rng = np.random.default_rng(20260821)
    for _ in range(20):
        unit_count = int(rng.integers(2, 40))
        matrix = rng.normal(size=(unit_count, 16))
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        tree = linkage_tree(matrix)
        for threshold in np.linspace(0.0, float(tree[:, 2].max()) * 1.1, 25):
            assert _partition(_cut_like_the_app(unit_count, tree, threshold)) == _partition(
                fcluster(tree, t=threshold, criterion="distance")
            )
