from __future__ import annotations

import signal
import sys
import tempfile
import unittest
from pathlib import Path

from yp_video.web.job_helpers import (
    stream_subprocess,
    subprocess_exit_status,
    subprocess_failure,
)


class SubprocessStatusTests(unittest.TestCase):
    def test_success_status_is_explicit(self) -> None:
        self.assertEqual(subprocess_exit_status(0), "exited successfully (code 0)")

    def test_exit_code_precedes_last_output(self) -> None:
        self.assertEqual(
            subprocess_failure("SPOT training", 3, "Segment mAP: 40.87%"),
            "SPOT training exited with code 3; last output: Segment mAP: 40.87%",
        )

    def test_sigkill_is_not_declared_to_be_oom(self) -> None:
        status = subprocess_exit_status(-signal.SIGKILL)
        self.assertIn("killed by SIGKILL (9)", status)
        self.assertIn("possible host OOM", status)
        self.assertIn("verify", status)
        self.assertTrue(
            subprocess_failure("SPOT training", -signal.SIGKILL, "mAP: 40.87%").startswith(
                "SPOT training killed by SIGKILL (9)"
            )
        )


class StreamSubprocessLogTests(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_log_ends_with_positive_exit_status(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            log_path = Path(raw_dir) / "terminal.log"
            rc, last = await stream_subprocess(
                "no-job",
                [sys.executable, "-c", "print('ordinary final output'); raise SystemExit(7)"],
                raw_dir,
                log_path=log_path,
                update_job=False,
            )
            self.assertEqual((rc, last), (7, "ordinary final output"))
            self.assertTrue(
                log_path.read_text(encoding="utf-8").endswith(
                    "# Process exited with code 7\n"
                )
            )

    async def test_terminal_log_records_signal(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            log_path = Path(raw_dir) / "terminal.log"
            rc, last = await stream_subprocess(
                "no-job",
                [
                    sys.executable,
                    "-c",
                    (
                        "import os, signal; "
                        "print('metric before death', flush=True); "
                        "os.kill(os.getpid(), signal.SIGTERM)"
                    ),
                ],
                raw_dir,
                log_path=log_path,
                update_job=False,
            )
            self.assertEqual((rc, last), (-signal.SIGTERM, "metric before death"))
            self.assertTrue(
                log_path.read_text(encoding="utf-8").endswith(
                    "# Process killed by SIGTERM (15)\n"
                )
            )


if __name__ == "__main__":
    unittest.main()
