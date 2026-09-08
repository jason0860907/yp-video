"""Per-player action clips: the geometry and the sample rules.

The crop must stay on the player through a tracking gap and past the frame
edge, the mosaic must round-trip, and a re-export must pick the same
negatives — each of these fails silently otherwise, as a slightly wrong
dataset rather than an error.
"""

from __future__ import annotations

import unittest

import numpy as np

from yp_video.actor import clips
from yp_video.actor.clips import (
    CROP_SIZE,
    GRID,
    NEGATIVES_PER_EVENT,
    crop_square,
    cut_square,
    fill_boxes,
    pack,
    pick_negatives,
    unpack,
)


class GeometryTests(unittest.TestCase):
    def test_square_is_centred_on_the_box_and_scaled(self) -> None:
        # A 100×200 box at (100, 50)–(200, 250) in a 1000×500 frame.
        x, y, side = crop_square((0.1, 0.1, 0.2, 0.5), 1000, 500)
        self.assertEqual(side, round(200 * clips.CROP_SCALE))
        self.assertEqual((x + side / 2, y + side / 2), (150, 150))

    def test_cut_pads_black_past_the_frame_edge(self) -> None:
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        crop = cut_square(frame, -20, -20, 40)
        self.assertEqual(crop.shape, (CROP_SIZE, CROP_SIZE, 3))
        # Top-left quadrant is off-frame (black); bottom-right is the frame.
        self.assertEqual(int(crop[0, 0, 0]), 0)
        self.assertEqual(int(crop[-1, -1, 0]), 255)

    def test_cut_fully_outside_is_black_not_an_error(self) -> None:
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        crop = cut_square(frame, 500, 500, 40)
        self.assertEqual(int(crop.max()), 0)

    def test_fill_boxes_follows_the_nearest_present_offset(self) -> None:
        a, b = (0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)
        filled, present = fill_boxes([None, a, None, None, b, None])
        self.assertEqual(present, [False, True, False, False, True, False])
        self.assertEqual(filled, [a, a, a, b, b, b])

    def test_fill_boxes_rejects_a_boxless_candidate(self) -> None:
        with self.assertRaises(ValueError):
            fill_boxes([None, None])


class MosaicTests(unittest.TestCase):
    def test_pack_unpack_round_trip(self) -> None:
        crops = [
            np.full((CROP_SIZE, CROP_SIZE, 3), i, dtype=np.uint8)
            for i in range(GRID * GRID)
        ]
        mosaic = pack(crops)
        self.assertEqual(mosaic.shape, (GRID * CROP_SIZE, GRID * CROP_SIZE, 3))
        for original, back in zip(crops, unpack(mosaic)):
            np.testing.assert_array_equal(original, back)

    def test_pack_needs_exactly_nine(self) -> None:
        with self.assertRaises(ValueError):
            pack([np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)] * 4)


class NegativeSamplingTests(unittest.TestCase):
    def test_same_event_id_picks_the_same_negatives(self) -> None:
        others = [{"track": f"1:{i}"} for i in range(12)]
        first = pick_negatives("act_abc", others)
        # Only the event id decides: the same picks whatever order the
        # candidates arrive in, and different picks for another event.
        second = pick_negatives("act_abc", list(reversed(others)))
        self.assertEqual(len(first), NEGATIVES_PER_EVENT)
        self.assertEqual([o["track"] for o in first], [o["track"] for o in second])
        self.assertNotEqual(
            [o["track"] for o in first],
            [o["track"] for o in pick_negatives("act_xyz", others)],
        )

    def test_fewer_others_than_wanted(self) -> None:
        self.assertEqual(len(pick_negatives("x", [{"track": "1:1"}])), 1)
        self.assertEqual(pick_negatives("x", []), [])


if __name__ == "__main__":
    unittest.main()
