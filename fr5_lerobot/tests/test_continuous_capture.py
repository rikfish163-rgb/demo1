"""Synthetic images and mocked calls only; no hardware transport is available."""
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from lerobot_robot_fairino.continuous_capture import ContinuousCapture
from lerobot_robot_fairino.feedback import gripper_feedback_valid


class FakeRobot:
    is_connected = True

    def __init__(self):
        self.config = SimpleNamespace(
            max_sync_skew_s=.1,
            camera=SimpleNamespace(timeout_s=.1, max_age_s=1., expected_serial="SYNTHETIC"))
        self.first_time = None
        self.repeat = False
        self.serial = "SYNTHETIC"

    def get_sample(self):
        captured = self.first_time if self.repeat and self.first_time else time.time()
        self.first_time = self.first_time or captured
        return {
            "camera": {"color_rgb": np.zeros((12, 16, 3), np.uint8),
                       "depth_mm": np.full((12, 16), 500., np.float32),
                       "metadata": {"source_timestamp_unix": captured, "source_camera_serial": self.serial}},
            "robot_state": {"timestamp": captured + .001, "gripper_position": 0,
                            "gripper_active": 1, "gripper_motiondone": 0, "gripper_speed": 0,
                            "gripper_fault": 0, "gripper_fault_id": 0},
            "sync_skew_s": .001,
        }


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.robot = FakeRobot()
        self.root = Path(self.tmp.name) / "synthetic_only"
        self.capture = ContinuousCapture(self.robot, self.root, fps=30, task="SYNTHETIC ONLY")

    def tearDown(self):
        if self.capture._started and self.capture._report is None:
            self.capture.finish("interrupted")
        self.tmp.cleanup()

    def wait_for(self, predicate):
        deadline = time.monotonic() + 1
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertTrue(predicate())

    def test_continuous_unique_frames_events_and_invalid_feedback_remain_explicit(self):
        c = self.capture.start()
        phase = c.phase("close", teacher={"policy": "SYNTHETIC"})
        call = Mock(return_value=0)
        self.assertEqual(c.call_once("FixtureOnly", call, 12), 0)
        call.assert_called_once_with(12)
        self.wait_for(lambda: len(c._times) >= 4)
        report = c.finish("success", evidence=["SYNTHETIC fixture label"])
        rows = [json.loads(s) for s in (self.root / "capture.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), report["frames"])
        self.assertEqual(report["frames_with_invalid_gripper_feedback"], len(rows))
        self.assertFalse(report["training_ready"])
        self.assertFalse(report["has_target_action_labels"])
        self.assertEqual(rows[-1]["phase_event_id"], phase["event_id"])
        self.assertIsNotNone(rows[-1]["last_command_attempt_id"])
        for row in rows:
            self.assertTrue((self.root / row["rgb_file"]).exists())
            with np.load(self.root / row["depth_file"], allow_pickle=False) as packet:
                self.assertEqual(packet["depth_mm"].shape, (12, 16))
        self.assertFalse(c._thread.is_alive())

    def test_duplicate_causes_sticky_failure_and_blocks_next_command(self):
        c = self.capture.start()
        self.robot.repeat = True
        self.wait_for(lambda: c._error is not None)
        call = Mock()
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            c.call_once("FixtureOnly", call)
        call.assert_not_called()
        self.assertEqual(c.finish("success")["outcome"], "interrupted")

    def test_uncertain_rpc_is_not_retried_or_reported_as_completion(self):
        c = self.capture.start()
        call = Mock(side_effect=TimeoutError("uncertain fixture"))
        with self.assertRaises(TimeoutError):
            c.call_once("FixtureOnly", call)
        call.assert_called_once()
        report = c.finish("interrupted")
        self.assertEqual(report["commands_without_rpc_return"], 1)
        events = [json.loads(s) for s in (self.root / "events.jsonl").read_text().splitlines()]
        self.assertEqual([e["kind"] for e in events], ["command_attempt", "command_uncertain"])

    def test_identity_change_and_existing_output_refused(self):
        c = self.capture.start()
        self.robot.serial = "OTHER"
        self.wait_for(lambda: c._error is not None)
        self.assertFalse(c.finish()["capture_complete"])
        with self.assertRaises(FileExistsError):
            ContinuousCapture(self.robot, self.root, task="SYNTHETIC ONLY").start()

    def test_failed_target_writer_blocks_the_next_command(self):
        self.capture.sample_callback = Mock(side_effect=OSError("target writer fixture"))
        with self.assertRaisesRegex(RuntimeError, "target writer fixture"):
            self.capture.start()
        self.assertFalse(self.capture._thread.is_alive())
        self.assertFalse(self.capture._report["capture_complete"])

    def test_feedback_zero_is_valid_only_with_valid_status_and_no_fault(self):
        s = self.robot.get_sample()["robot_state"]
        self.assertFalse(gripper_feedback_valid(s))
        s["gripper_motiondone"] = 1
        self.assertTrue(gripper_feedback_valid(s))
        s["gripper_motiondone"] = 3  # Valid measurement is not successful holding.
        self.assertTrue(gripper_feedback_valid(s))
        s["gripper_fault_id"] = 1
        self.assertFalse(gripper_feedback_valid(s))


if __name__ == "__main__":
    unittest.main()
