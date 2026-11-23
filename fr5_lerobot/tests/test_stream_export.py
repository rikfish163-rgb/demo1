"""Offline tests for measured-feedback trajectory retargeting and provenance."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lerobot_robot_fairino import export_stream_policy as stream


def fixture_module():
    path = Path(__file__).with_name('stream_fixtures.py')
    spec = importlib.util.spec_from_file_location('stream_data_fixtures', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tree_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def mocked_capture(times=None):
    times = times or [10., 10.14, 10.32, 10.50, 10.68, 10.86]
    captures = []
    for index, t in enumerate(times):
        captures.append({'frame_index': index, 'camera': {'source_timestamp_monotonic': t},
                         'robot': {'received_monotonic': t + .02, 'joints': [10. * index] * 6,
                                   'gripper_position': 85., 'gripper_active': 1,
                                   'gripper_fault': 0, 'gripper_fault_id': 0,
                                   'gripper_motiondone': 1, 'gripper_speed': 0},
                         'rgb_file': f'{index}.jpg'})
    journal = [
        {'sequence': 1, 'issued_monotonic': 9.9, 'kind': 'hold', 'executed': False,
         'target': [0.] * 6 + [100.]},
        {'sequence': 2, 'issued_monotonic': 10.1, 'kind': 'arm', 'executed': True,
         'target': [50.] * 6 + [100.]},
        {'sequence': 3, 'issued_monotonic': 10.25, 'kind': 'gripper', 'executed': True,
         'target': [50.] * 6 + [76.]},
        {'sequence': 4, 'issued_monotonic': 10.6, 'kind': 'gripper', 'executed': True,
         'target': [50.] * 6 + [100.]},
    ]
    media = [{'rgb_jpeg_sha256': f'fake-{i}'} for i in range(len(captures))]
    return captures, journal, media


class RetargetTest(unittest.TestCase):
    def setUp(self):
        self.source = {'root': Path('/not-opened')}

    def build(self, data, **options):
        with patch.object(stream, '_source_data', return_value=data):
            return stream._build_clips(self.source, **options)

    def test_strict_mode_never_treats_initial_hold_as_an_issued_command(self):
        with self.assertRaisesRegex(ValueError, 'initial hold is not an RPC'):
            self.build(mocked_capture())

    def test_causal_pair_future_feedback_and_last_issued_gripper(self):
        clips, coverage = self.build(mocked_capture(), allow_initial_hold=True)
        rows = clips[0]
        first = rows[0]
        provenance = first['mapping']
        self.assertLessEqual(provenance['observation']['camera_time_monotonic'], provenance['grid_time_monotonic'])
        self.assertLessEqual(provenance['observation']['robot_time_monotonic'], provenance['grid_time_monotonic'])
        self.assertAlmostEqual(provenance['target_time_monotonic'] - provenance['grid_time_monotonic'], 1 / 6)
        interp = provenance['joint_retarget']
        self.assertEqual((interp['lower']['frame_index'], interp['upper']['frame_index']), (1, 2))
        expected_deg = 10. * (1. - interp['upper_weight']) + 20. * interp['upper_weight']
        self.assertAlmostEqual(first['action'][0], math.radians(expected_deg), places=6)
        self.assertNotAlmostEqual(first['action'][0], first['state'][0], places=5)
        self.assertEqual(provenance['active_arm_wire_target']['target_degrees_percent'][0], 50.)
        self.assertNotAlmostEqual(first['action'][0], math.radians(50), places=4)
        self.assertFalse(provenance['gripper_target']['executed'])
        self.assertEqual(provenance['gripper_target']['source'], 'initial_hold_not_rpc')
        self.assertEqual(first['action'][6], 1.)
        self.assertGreater(coverage['initial_hold_action_frames'], 0)
        # At the next target tick the real close command has been issued.
        self.assertEqual(rows[1]['mapping']['gripper_target']['sequence'], 3)
        self.assertTrue(rows[1]['mapping']['gripper_target']['executed'])
        self.assertAlmostEqual(rows[1]['action'][6], .76, places=6)
        # It stays at the same discrete target until the later open command.
        self.assertTrue(all(round(float(r['action'][6]), 2) in (.76, 1.) for r in rows))
        self.assertTrue(any(r['mapping']['gripper_target']['sequence'] == 4 for r in rows))

    def test_invalid_gripper_observation_uses_validity_bit(self):
        data = mocked_capture()
        data[0][0]['robot']['gripper_fault'] = 1
        clips, coverage = self.build(data, allow_initial_hold=True)
        self.assertEqual(clips[0][0]['state'][6:].tolist(), [0., 0.])
        self.assertFalse(clips[0][0]['mapping']['observation']['gripper_valid'])
        self.assertGreaterEqual(coverage['invalid_gripper_observations'], 1)

    def test_selected_pair_never_uses_state_or_command_from_the_future(self):
        data = mocked_capture()
        # At t=10.02+1/6, frame 1's camera has arrived, but its robot
        # feedback has not; frame 0 is the last usable paired observation.
        data[0][1]['robot']['received_monotonic'] = 10.19
        # A gripper command shortly after a target tick must not be latched.
        data[1][2]['issued_monotonic'] = 10.02 + 2 / 6 + 1e-9
        clips, _ = self.build(data, allow_initial_hold=True)
        self.assertEqual(clips[0][1]['mapping']['observation']['frame_index'], 0)
        self.assertEqual(clips[0][1]['mapping']['gripper_target']['source'], 'initial_hold_not_rpc')
        self.assertEqual(clips[0][2]['mapping']['gripper_target']['sequence'], 3)

    def test_gap_splits_episodes_and_never_interpolates_across_it(self):
        data = mocked_capture([10., 10.16, 10.32, 11.4, 11.56, 11.72, 11.88])
        clips, coverage = self.build(data, allow_initial_hold=True)
        self.assertEqual(len(clips), 2)
        self.assertEqual(coverage['source_spans'], 2)
        for clip in clips:
            for row in clip:
                interp = row['mapping']['joint_retarget']
                left = interp['lower']['robot_time_monotonic']
                right = interp['upper']['robot_time_monotonic']
                self.assertLessEqual(right - left, stream.MAX_FRAME_GAP_S)
                self.assertTrue((right < 11.) or (left > 11.))

    def test_splits_are_whole_source_and_default_to_last(self):
        self.assertEqual(stream._splits(3, None), ['train', 'train', 'validation'])
        self.assertEqual(stream._splits(3, [0, 2]), ['validation', 'train', 'validation'])
        for invalid in ([0, 0], [3], [-1], [0, 1, 2]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                stream._splits(3, invalid)


class ExportIntegrationTest(unittest.TestCase):
    def test_reviewed_sources_export_verify_train_only_stats_and_immutable_inputs(self):
        make_source = fixture_module().make_source
        with tempfile.TemporaryDirectory(prefix='fr5-stream-export-test-') as directory:
            root = Path(directory)
            a = make_source(root / 'a', trial=0)
            b = make_source(root / 'b', trial=1)
            original_a, original_b = tree_hashes(a), tree_hashes(b)
            output = root / 'datasets' / 'new'
            with patch.object(stream, 'DATA_ROOT', root):
                result = stream.export([a, b], output, allow_initial_hold=True)
                self.assertEqual(result['status'], 'passed')
                self.assertEqual(result['source_count'], 2)
                self.assertEqual(result['train_episode_indices'], [0])
                self.assertEqual(result['validation_episode_indices'], [1])
                self.assertEqual(stream.verify(output)['status'], 'passed')
                with self.assertRaises(FileExistsError):
                    stream.export([a, b], output, allow_initial_hold=True)
            self.assertEqual(tree_hashes(a), original_a)
            self.assertEqual(tree_hashes(b), original_b)
            proof = json.loads((output / 'meta/stream_provenance.json').read_text())
            stats = json.loads((output / 'meta/stream_stats.json').read_text())
            self.assertEqual(stats['source_indices'], [0])
            self.assertEqual(stats['stats'], json.loads((output / 'meta/stats.json').read_text()))
            self.assertEqual(stats['stats']['observation.state']['count'], [result['train_frames']])
            import pyarrow.parquet as pq
            rows = pq.read_table(sorted((output / 'data').rglob('*.parquet'))).to_pylist()
            train_values = np.asarray([r['observation.state'] for r in rows if r['episode_index'] == 0])
            all_values = np.asarray([r['observation.state'] for r in rows])
            np.testing.assert_allclose(stats['stats']['observation.state']['mean'], train_values.mean(axis=0))
            self.assertFalse(np.allclose(stats['stats']['observation.state']['mean'], all_values.mean(axis=0)))
            for source in proof['sources']:
                copy = output / source['user_review_file']
                self.assertEqual(hashlib.sha256(copy.read_bytes()).hexdigest(), source['user_review_sha256'])
                self.assertGreater(source['coverage']['initial_hold_action_frames'], 0)
            mapping = output / 'meta/stream_source_frames.jsonl'
            lines = mapping.read_text().splitlines()
            tampered = json.loads(lines[0])
            tampered['joint_retarget']['upper_weight'] = .123456
            lines[0] = json.dumps(tampered)
            mapping.write_text('\n'.join(lines) + '\n')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                stream.verify(output)

    def test_duplicate_or_unreviewed_source_is_rejected_before_dataset_creation(self):
        make_source = fixture_module().make_source
        with tempfile.TemporaryDirectory(prefix='fr5-stream-review-test-') as directory:
            root = Path(directory)
            a = make_source(root / 'a', trial=0)
            b = make_source(root / 'b', trial=1)
            with patch.object(stream, 'DATA_ROOT', root):
                out = root / 'output'
                with self.assertRaisesRegex(ValueError, 'Duplicate reviewed source'):
                    stream.export([a, a], out, allow_initial_hold=True)
                self.assertFalse(out.exists())
                review = json.loads((b / 'user-review.json').read_text())
                review['verdict'] = 'failure'
                (b / 'user-review.json').write_text(json.dumps(review))
                with self.assertRaisesRegex(ValueError, 'User success review'):
                    stream.export([a, b], out, allow_initial_hold=True)
                self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
