"""Rally clip export reads a cut that lives only in R2 over its presigned URL."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from yp_video.web.routers import annotate


class ClipSourceTests(unittest.TestCase):
    def test_an_r2_only_cut_is_read_over_its_url(self) -> None:
        cut = Path("/videos/cuts-sideline/G103 #A vs. #B_set2.mp4")
        with (
            patch.object(annotate, "resolve_cut", return_value=cut),
            patch.object(annotate, "cut_media_source", return_value="https://r2/signed") as source,
        ):
            stem, url = annotate._resolve_clip_source(cut.name)
        source.assert_called_once_with(cut)
        self.assertEqual((stem, url), ("G103 #A vs. #B_set2", "https://r2/signed"))

    def test_a_cut_nowhere_is_a_404(self) -> None:
        with patch.object(annotate, "resolve_cut", return_value=None):
            with self.assertRaises(HTTPException) as ctx:
                annotate._resolve_clip_source("missing.mp4")
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
