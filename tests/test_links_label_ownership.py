"""A label's tracklet link stands down when the tracklet is somebody else's."""

from __future__ import annotations

import unittest

from yp_video.extraction.links import _claimed_by_other

# Measured on 0104排島臨打 5, frame 21491: the receiver behind a teammate has
# no tracklet; the teammate's tracklet overlaps the label box at IoU 0.56.
LABEL = [1008.0, 526.0, 1203.0, 814.0]
OCCLUDER_TRACK = [1004.0, 453.0, 1166.0, 891.0]
OCCLUDER_2XL = [1003.0, 452.0, 1159.0, 892.0]


class ClaimedByOtherTests(unittest.TestCase):
    def test_the_occluders_own_box_takes_her_tracklet_from_the_label(self):
        self.assertTrue(_claimed_by_other(OCCLUDER_TRACK, LABEL, [OCCLUDER_2XL, LABEL]))

    def test_a_re_detection_of_the_labelled_player_never_claims_it(self):
        near = [1010.0, 528.0, 1200.0, 816.0]
        self.assertFalse(_claimed_by_other(LABEL, LABEL, [near]))

    def test_without_another_person_on_the_frame_the_link_stands(self):
        self.assertFalse(_claimed_by_other(OCCLUDER_TRACK, LABEL, [LABEL]))


if __name__ == "__main__":
    unittest.main()
