"""Startup synchronization tests with synthetic camera and robot data only."""

import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from lerobot_robot_fairino import FairinoFR5, FairinoFR5Config


class InitialSampleSyncTests(unittest.TestCase):
    def setUp(self):
        self.stream = Mock()
        self.camera = Mock()
        patches = (
            patch('lerobot_robot_fairino.robot_fairino.FR5StateStream', return_value=self.stream),
            patch('lerobot_robot_fairino.robot_fairino.GeminiBridgeCamera', return_value=self.camera),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.robot = FairinoFR5(FairinoFR5Config())
        self.addCleanup(self.robot.disconnect)
        self.now = time.monotonic()
        self.state = {'joints': [0.] * 6, 'gripper_position': 100,
                      'received_monotonic': self.now + .003, 'timestamp': time.time()}
        self.stream.latest.return_value = {**self.state, 'received_monotonic': self.now - .01}
        self.robot.connect()

    def bundle(self, stamp):
        return {'metadata': {'source_timestamp_monotonic': stamp},
                'color_rgb': np.zeros((2, 3, 3), dtype=np.uint8)}

    def test_old_initial_frame_is_discarded_and_next_frame_is_paired(self):
        self.camera.read_bundle.side_effect = [self.bundle(self.now - .4), self.bundle(self.now)]
        self.stream.nearest.return_value = self.state

        sample = self.robot.get_sample()

        self.assertEqual(sample['camera']['metadata']['source_timestamp_monotonic'], self.now)
        self.assertAlmostEqual(sample['sync_skew_s'], .003)
        self.assertEqual(self.camera.read_bundle.call_count, 2)
        self.stream.nearest.assert_called_once()
        self.assertTrue(all(c.kwargs['require_new'] for c in self.camera.read_bundle.call_args_list))
        self.assertTrue(all(c.kwargs['max_skew_s'] == self.robot.config.max_sync_skew_s
                            for c in self.stream.nearest.call_args_list))

    def test_initial_unmatched_frame_can_be_followed_by_a_real_match(self):
        later = self.now + .1
        self.camera.read_bundle.side_effect = [self.bundle(self.now), self.bundle(later)]
        self.stream.nearest.side_effect = [
            TimeoutError('No FR5 state frame within requested timestamp skew'),
            {**self.state, 'received_monotonic': later + .003}]

        sample = self.robot.get_sample()

        self.assertEqual(sample['camera']['metadata']['source_timestamp_monotonic'], later)
        self.assertAlmostEqual(sample['sync_skew_s'], .003)
        self.assertEqual(self.stream.nearest.call_count, 2)

    def test_sync_failure_after_start_is_not_retried_or_hidden(self):
        self.camera.read_bundle.return_value = self.bundle(self.now)
        self.stream.nearest.return_value = self.state
        self.robot.get_sample()
        self.camera.read_bundle.reset_mock()
        self.stream.nearest.side_effect = TimeoutError('lost time synchronization')

        with self.assertRaisesRegex(TimeoutError, 'lost time synchronization'):
            self.robot.get_sample()
        self.assertEqual(self.camera.read_bundle.call_count, 1)

    def test_startup_cannot_fabricate_a_match_when_no_new_frame_arrives(self):
        self.camera.read_bundle.side_effect = [self.bundle(self.now - .4),
                                                TimeoutError('No new Gemini frame before timeout')]
        self.stream.nearest.side_effect = TimeoutError('No matching state')

        with self.assertRaisesRegex(TimeoutError, 'No new Gemini frame'):
            self.robot.get_sample()
        self.assertIsNone(self.robot.last_sample)


if __name__ == '__main__':
    unittest.main()
