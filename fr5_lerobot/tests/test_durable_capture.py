"""Durable recorder failures with synthetic images and local temporary files."""

import tempfile
import threading
import time
from pathlib import Path
import unittest
from unittest.mock import patch

from lerobot_robot_fairino.continuous_capture import ContinuousCapture
from lerobot_robot_fairino.demo_writer import DemoWriter
from fr5_lerobot.tests.test_continuous_capture import FakeRobot
from fr5_lerobot.tests.test_demo_writer import synthetic_sample, synthetic_command


class DurableCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fr5-durable-capture-", dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_frame_fsync_failure_blocks_capture_start(self):
        capture = ContinuousCapture(FakeRobot(), self.root / "continuous", task="SYNTHETIC",
                                    fps=6, sync_writes=True)
        with patch("lerobot_robot_fairino.continuous_capture.os.fsync",
                   side_effect=OSError("synthetic remote sync failure")):
            with self.assertRaisesRegex(RuntimeError, "remote sync failure"):
                capture.start()
        self.assertFalse(capture._report["capture_complete"])
        self.assertEqual(capture._report["outcome"], "interrupted")
        self.assertFalse(capture._thread.is_alive())

    def test_unresponsive_capture_lock_has_a_bounded_health_check(self):
        capture = ContinuousCapture(FakeRobot(), self.root / "continuous", task="SYNTHETIC")
        held, release = threading.Event(), threading.Event()

        def writer():
            with capture._lock:
                held.set()
                release.wait(2)

        worker = threading.Thread(target=writer)
        worker.start()
        try:
            self.assertTrue(held.wait(1))
            started = time.monotonic()
            with self.assertRaisesRegex(TimeoutError, "storage is unresponsive"):
                capture.assert_healthy()
            self.assertLess(time.monotonic() - started, 1)
        finally:
            release.set()
            worker.join(2)

    def test_journal_fsync_failure_is_sticky_and_not_counted(self):
        writer = DemoWriter(self.root / "journal", journal_only=True, sync_writes=True,
                            external_capture_root=self.root / "continuous",
                            metadata={"image_width": 16, "image_height": 12}).start()
        sample = synthetic_sample()
        with patch("lerobot_robot_fairino.demo_writer.os.fsync",
                   side_effect=OSError("synthetic sync failure")):
            with self.assertRaisesRegex(OSError, "sync failure"):
                writer.add(sample, synthetic_command(sample))
        with self.assertRaisesRegex(RuntimeError, "Journal storage failed"):
            writer.add(sample, synthetic_command(sample))
        report = writer.finish("interrupted")
        self.assertEqual(report["recorded_frames"], 0)
        self.assertFalse(report["storage_complete"])


if __name__ == "__main__":
    unittest.main()
