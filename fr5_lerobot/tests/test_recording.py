"""Synthetic integration fixtures are temporary, never presented as real demos."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_camera_gemini import GeminiBridgeConfig
from lerobot_robot_fairino import FairinoFR5Config
from lerobot_robot_fairino.record import record
from lerobot_robot_fairino.robot_fairino import STATE_NAMES, STATUS_NAMES
from lerobot_robot_fairino.verify import verify_dataset


def synthetic_samples(n=3):
    now = time.time() - 10
    result = []
    for i in range(n):
        rgb = np.zeros((36, 64, 3), np.uint8)
        rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2] = 20 + i, 100, 200
        state = dict.fromkeys(STATUS_NAMES, 0)
        state.update(joints=[1 + i, 2, 3, 4, 5, 6], tcp=[1, 2, 3, 4, 5, 6],
                     flange=[1, 2, 3, 4, 5, 6], joint_speeds_deg_s=[0] * 6,
                     gripper_position=50, timestamp=now + i / 2 + .01, frame_counter=i)
        meta = {"source_timestamp_unix": now + i / 2, "source_camera_serial": "TEST_SYNTHETIC"}
        result.append({
            "robot_state": state, "sync_skew_s": .01,
            "observation": dict(zip(STATE_NAMES, [*state["joints"], 50])),
            "camera": {"color_rgb": rgb, "depth_mm": np.full((36, 64), 500, np.float32),
                       "metadata": meta},
        })
    return result


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "synthetic_only"
        self.config = FairinoFR5Config(camera=GeminiBridgeConfig(
            expected_serial="TEST_SYNTHETIC", width=64, height=36))

    def tearDown(self):
        self.tmp.cleanup()

    def run_record(self, samples, disconnect_error=None):
        with patch("lerobot_robot_fairino.record.FairinoFR5") as cls:
            robot = cls.return_value
            robot.name = "fairino_fr5_readonly"
            robot.get_sample.side_effect = deepcopy(samples)
            robot.disconnect.side_effect = disconnect_error
            with patch("lerobot_robot_fairino.record.time.sleep"):
                return record(self.config, self.root, frames=3, fps=2, task="SYNTHETIC TEST ONLY")

    def test_real_lerobot_roundtrip_with_depth_and_precise_timestamps(self):
        report = self.run_record(synthetic_samples())
        self.assertTrue(report["capture_complete"])
        verified = verify_dataset(self.root)
        self.assertTrue(verified["passed"])
        self.assertEqual(verified["frames"], 3)
        dataset = LeRobotDataset(repo_id=report["repo_id"], root=self.root)
        self.assertNotIn("action", dataset.features)
        np.testing.assert_allclose(dataset[0]["observation.images.front"][:, 0, 0],
                                   np.array([20, 100, 200]) / 255, atol=1e-7)
        self.assertTrue((self.root / "preview.png").exists())
        # Corrupt source capture metadata; verification must reject it.
        path = self.root / "meta/capture.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        rows[1]["camera"]["source_timestamp_unix"] += .02
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        with self.assertRaisesRegex(ValueError, "timestamps"):
            verify_dataset(self.root, write_report=False)

    def test_capture_error_preserves_partial_episode_and_marks_incomplete(self):
        samples = synthetic_samples()
        with self.assertRaisesRegex(TimeoutError, "stale fixture"):
            self.run_record([samples[0], samples[1], TimeoutError("stale fixture")])
        report = json.loads((self.root / "meta/recording.json").read_text())
        self.assertFalse(report["capture_complete"])
        self.assertEqual(report["recorded_frames"], 2)
        dataset = LeRobotDataset(repo_id=report["repo_id"], root=self.root)
        self.assertEqual(len(dataset), 2)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            verify_dataset(self.root)

    def test_disconnect_error_still_finalizes_saved_frames(self):
        with self.assertRaisesRegex(RuntimeError, "disconnect fixture"):
            self.run_record(synthetic_samples(), RuntimeError("disconnect fixture"))
        report = json.loads((self.root / "meta/recording.json").read_text())
        self.assertFalse(report["capture_complete"])
        self.assertEqual(len(LeRobotDataset(repo_id=report["repo_id"], root=self.root)), 3)


if __name__ == "__main__":
    unittest.main()
