"""Synthetic demonstrations only: no robot or camera connections in these tests."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_robot_fairino.demo_writer import DemoWriter
from lerobot_robot_fairino.robot_fairino import STATE_NAMES, STATUS_NAMES


def synthetic_sample(index=0):
    timestamp = 1_791_630_000.123456 + index * 0.100003
    state = dict.fromkeys(STATUS_NAMES, 0)
    state.update(joints=[1 + index, 2, 3, 4, 5, 6], gripper_position=50,
                 tcp=[10, 20, 30, 40, 50, 60], flange=[0] * 6,
                 joint_speeds_deg_s=[0] * 6, timestamp=timestamp + 0.004321,
                 frame_counter=index)
    rgb = np.zeros((12, 16, 3), np.uint8)
    rgb[:] = [20 + index, 100, 200]
    return {
        "robot_state": state, "sync_skew_s": state["timestamp"] - timestamp,
        "observation": dict(zip(STATE_NAMES, [*state["joints"], 50])),
        "camera": {"color_rgb": rgb, "depth_mm": np.full((12, 16), 500 + index, np.float32),
                   "metadata": {"source_timestamp_unix": timestamp,
                                "source_camera_serial": "SYNTHETIC_ONLY",
                                "color_intrinsic": {"width": 16, "height": 12}}},
    }


def synthetic_command(sample, sequence=0, kind="hold", executed=False):
    return {"target": [11, 22, 33, 44, 55, 66, 80], "sequence": sequence,
            "issued_unix": sample["camera"]["metadata"]["source_timestamp_unix"] - 0.001234,
            "source": "keyboard", "kind": kind, "executed": executed}


class DemoWriterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "synthetic_only"
        self.writer = DemoWriter(self.root, fps=10, task="SYNTHETIC TEST ONLY",
                                 metadata={"image_width": 16, "image_height": 12})

    def tearDown(self):
        if self.writer._started and not self.writer._finished:
            self.writer.finish("interrupted")
        self.tmp.cleanup()

    def test_lerobot_roundtrip_keeps_target_labels_and_microsecond_precision(self):
        self.writer.start()
        samples = [synthetic_sample(i) for i in range(3)]
        commands = [synthetic_command(s, i, "hold" if i == 0 else "arm", i != 0)
                    for i, s in enumerate(samples)]
        for sample, command in zip(samples, commands):
            self.writer.add(sample, command)
        report = self.writer.finish("success")
        self.assertTrue(report["data_valid"])
        self.assertTrue(report["storage_complete"])
        self.assertFalse(report["training_ready"])
        self.assertFalse(report["grasp_success_verified"])
        self.assertFalse(report["gripper_feedback_verified"])
        self.assertEqual(report["outcome_source"], "operator_label")
        self.assertEqual(report["frames_with_command_executed_false"], 1)
        self.assertEqual(report["robot_commands_sent_by_writer"], 0)
        self.assertIsNone(self.writer.dataset.image_writer)
        self.assertTrue(self.writer._log.closed)
        dataset = LeRobotDataset(repo_id=report["repo_id"], root=self.root)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.num_episodes, 1)
        rows = [json.loads(line) for line in (self.root / "meta/capture.jsonl").read_text().splitlines()]
        for i, (sample, command) in enumerate(zip(samples, commands)):
            frame = dataset[i]
            np.testing.assert_array_equal(frame["action"], np.asarray(command["target"], np.float32))
            self.assertFalse(np.array_equal(frame["action"], frame["observation.state"]))
            times = np.asarray(frame["observation.source_time_us"])
            self.assertEqual(times.dtype, np.int64)
            self.assertEqual(times[0], round(sample["camera"]["metadata"]["source_timestamp_unix"] * 1_000_000))
            action_time = np.asarray(frame["action.source_time_us"])
            self.assertEqual(action_time.dtype, np.int64)
            # LeRobot maps shape-(1,) scalar features to scalar tensors on read.
            self.assertEqual(action_time.item(), round(command["issued_unix"] * 1_000_000))
            self.assertEqual(frame["action.sequence"].item(), i)
            self.assertAlmostEqual(frame["timestamp"].item(), i / 10, places=6)
            self.assertEqual(rows[i]["command"], command)
            with np.load(self.root / rows[i]["depth_file"], allow_pickle=False) as packet:
                np.testing.assert_array_equal(packet["depth_mm"], sample["camera"]["depth_mm"])
        self.assertEqual(self.writer.finish("success"), report)
        with self.assertRaises(ValueError):
            self.writer.finish("failure")

    def test_future_command_rejected_and_episode_marked_invalid(self):
        self.writer.start()
        sample = synthetic_sample()
        command = synthetic_command(sample)
        command["issued_unix"] = sample["camera"]["metadata"]["source_timestamp_unix"] + 0.000010
        with self.assertRaisesRegex(ValueError, "Future action"):
            self.writer.add(sample, command)
        for i in range(2):
            sample = synthetic_sample(i)
            self.writer.add(sample, synthetic_command(sample))
        report = self.writer.finish("failure")
        self.assertEqual(report["recorded_frames"], 2)
        self.assertFalse(report["data_valid"])
        self.assertTrue(report["storage_complete"])
        self.assertFalse(report["training_ready"])
        self.assertIn("Future action", report["errors"][0])

    def test_agent_teacher_keeps_actual_targets_and_requires_provenance(self):
        sample = synthetic_sample()
        command = synthetic_command(sample, kind="gripper", executed=True)
        command["source"] = "agent_teacher"
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.writer._prepare(sample, command)
        command["teacher"] = {"policy": "SYNTHETIC", "decision_id": "fixture-1", "phase": "close"}
        self.writer.start()
        self.writer.add(sample, command)
        report = self.writer.finish("success")
        self.assertEqual(report["dataset_kind"], "agent_teacher_demonstration")
        self.assertEqual(report["command_sources"], ["agent_teacher"])
        self.assertEqual(report["frames_with_invalid_gripper_feedback"], 1)
        self.assertFalse(report["training_ready"])
        dataset = LeRobotDataset(repo_id=report["repo_id"], root=self.root)
        np.testing.assert_array_equal(dataset[0]["action"], command["target"])
        self.assertEqual(dataset[0]["observation.gripper_feedback_valid"].item(), 0)

    def test_duplicate_or_backwards_source_time_rejected(self):
        self.writer.start()
        sample = synthetic_sample()
        self.writer.add(sample, synthetic_command(sample))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.writer.add(sample, synthetic_command(sample))
        backwards = synthetic_sample(-1)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.writer.add(backwards, synthetic_command(backwards))
        report = self.writer.finish("interrupted")
        self.assertFalse(report["data_valid"])
        self.assertFalse(report["training_ready"])

    def test_invalid_targets_and_shape_do_not_become_labels(self):
        self.writer.start()
        sample = synthetic_sample()
        invalid_commands = []
        for key, value in (("target", [0] * 6), ("target", [0] * 6 + [101]),
                           ("target", [float("nan")] + [0] * 6),
                           ("executed", 1), ("sequence", -1), ("source", "invented")):
            command = synthetic_command(sample)
            command[key] = value
            invalid_commands.append(command)
        for command in invalid_commands:
            with self.subTest(command=command), self.assertRaises(ValueError):
                self.writer.add(sample, command)
        broken = deepcopy(sample)
        broken["camera"]["depth_mm"] = np.ones((1, 1), np.float32)
        with self.assertRaisesRegex(ValueError, "Depth"):
            self.writer.add(broken, synthetic_command(sample))
        report = self.writer.finish("discarded")
        self.assertEqual(report["recorded_frames"], 0)
        self.assertFalse(report["storage_complete"])
        self.assertFalse(report["training_ready"])

    def test_save_failure_still_closes_all_writers_and_reports_error(self):
        self.writer.start()
        for i in range(2):
            sample = synthetic_sample(i)
            self.writer.add(sample, synthetic_command(sample))
        with patch.object(self.writer.dataset, "save_episode", side_effect=OSError("disk fixture")), \
                patch.object(self.writer.dataset, "finalize", wraps=self.writer.dataset.finalize) as finalize:
            report = self.writer.finish("interrupted")
        finalize.assert_called_once()
        self.assertIsNone(self.writer.dataset.image_writer)
        self.assertTrue(self.writer._log.closed)
        self.assertFalse(report["storage_complete"])
        self.assertFalse(report["training_ready"])
        self.assertIn("disk fixture", report["cleanup_errors"][0])

    def test_root_and_lifecycle_are_explicit(self):
        with self.assertRaises(FileExistsError):
            DemoWriter(Path(self.tmp.name))
        with self.assertRaises(RuntimeError):
            self.writer.add(synthetic_sample(), synthetic_command(synthetic_sample()))
        self.writer.start()
        with self.assertRaises(RuntimeError):
            self.writer.start()
        sample = synthetic_sample()
        self.writer.add(sample, synthetic_command(sample))
        report = self.writer.finish("discarded")
        self.assertEqual(report["recorded_frames"], 1)
        self.assertFalse(report["data_valid"])
        self.assertFalse(report["training_ready"])
        with self.assertRaises(RuntimeError):
            self.writer.add(sample, synthetic_command(sample))


if __name__ == "__main__":
    unittest.main()
