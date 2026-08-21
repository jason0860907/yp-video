"""Contract tests for complete representative frames exported by identify."""

import numpy as np

from yp_video.extraction.identify import _full_frame


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
