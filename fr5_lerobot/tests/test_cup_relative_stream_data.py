"""Offline synthetic fixtures for the cup-relative source binding."""

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from fr5_lerobot.training import cup_relative_stream_data as cup_data


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _reference(capture):
    return {'frame_index': capture['frame_index'],
            'camera_time_monotonic': capture['camera']['source_timestamp_monotonic'],
            'robot_time_monotonic': capture['robot']['received_monotonic']}


def _joint_state(capture, valid=True):
    robot = capture['robot']
    return ([math.radians(value) for value in robot['joints']]
            + [robot['gripper_position'] / 100. if valid else 0., float(valid)])


class PickupWaitMaskTests(unittest.TestCase):
    def test_only_the_leading_open_wait_at_stable_pickup_is_masked(self):
        state = [.1597, .0713, .0958, 0., 0., 0., 1., 1.]
        request = {'cup_relative_state': state, 'observation_history': [{'age_s': .5, 'state': state}],
                   'cup_tracking': {'native_contact': 0, 'rim_minus_jaw_mm': [-13.5, -3.4, 13.9]},
                   'cup_timing': {'phase': 'at_pickup'}}
        targets = [state[:6] + [opening] for opening in (1., 1., .76, .76, 1.)]
        original = copy.deepcopy(targets)
        self.assertEqual(cup_data.pickup_wait_mask(targets, request), [True, True, False, False, False])
        self.assertEqual(targets, original)
        far = copy.deepcopy(request)
        far['cup_relative_state'][0] += .026
        self.assertEqual(cup_data.pickup_wait_mask(targets, far), [False] * 5)
        invalid = copy.deepcopy(request)
        invalid['cup_relative_state'][7] = 0.
        self.assertEqual(cup_data.pickup_wait_mask(targets, invalid), [False] * 5)


class CupRelativeStreamDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.dataset = root / 'dataset'
        (self.dataset / 'meta').mkdir(parents=True)
        self.rows, self.episodes, self.mappings, sources = [], {}, [], []
        for source_index, name in enumerate(cup_data.SOURCE_EPISODES):
            trial = root / name / 'materialized'
            (trial / 'continuous').mkdir(parents=True)
            split = 'validation' if source_index == 7 else 'train'
            start = 1000. + 100. * source_index
            base = [100. + 10. * source_index, -800. + source_index, 5.]
            observation = {'camera': {'camera': {'serial': 'CP1E54200015'},
                                      'capture': {'received_monotonic': start - 2.}},
                           'calibration': {'robot': 'J1', 'direction': cup_data._CALIBRATION_DIRECTION}}
            observation_raw = _json_bytes(observation)
            geometry = {'source_observation': 'recorded-offline-fixture',
                        'source_observation_sha256': _sha(observation_raw),
                        'cup_uv_selected_by_llm': [1200. + source_index, 700.],
                        'cup_model': {'base_center_mm': base, 'method': 'synthetic fixture'}}
            geometry_raw = _json_bytes(geometry)
            scope = f'synthetic-scope-{source_index}'
            control = {'scope_sha256': scope,
                       'files': {'geometry.json': {'sha256': _sha(geometry_raw)},
                                 'source-observation.json': {'sha256': _sha(observation_raw)}}}
            (trial / 'source-observation.json').write_bytes(observation_raw)
            (trial / 'geometry.json').write_bytes(geometry_raw)
            (trial / 'control-manifest.json').write_bytes(_json_bytes(control))
            captures = []
            for frame in range(10):
                stamp = start + .25 * frame
                captures.append({'frame_index': frame, 'camera': {'source_timestamp_monotonic': stamp},
                    'robot': {'received_monotonic': stamp, 'tool': 0, 'user': 0,
                              'joints': [10. + source_index + frame * .1 + j for j in range(6)],
                              'tcp': [base[0] + 100. + 10. * frame,
                                      base[1] + 50. + 5. * frame, base[2] + 20. + 2. * frame,
                                      90. + frame, -2. - .5 * frame, -70. + .2 * frame],
                              'gripper_position': 100. - .5 * frame}})
            journal = b''.join(_json_bytes(capture) + b'\n' for capture in captures)
            (trial / 'continuous/capture.jsonl').write_bytes(journal)
            sources.append({'source_index': source_index, 'split': split, 'trial': str(trial),
                            'source_scope_sha256': scope, 'capture_journal_sha256': _sha(journal)})
            self.episodes[source_index] = []
            for step, (current_frame, lo_frame, hi_frame) in enumerate(((0, 0, 1), (4, 8, 9))):
                current, lower, upper = captures[current_frame], captures[lo_frame], captures[hi_frame]
                grip_target = .76 if step == 0 else .2
                weight = .25 if step == 0 else .5
                action = ([math.radians((1. - weight) * lo + weight * hi) for lo, hi in
                           zip(lower['robot']['joints'], upper['robot']['joints'])] + [grip_target])
                history = ([None] * 6 if step == 0 else
                           [{'age_s': .5, 'state': _joint_state(captures[2], valid=False)},
                            {'age_s': 1., 'state': _joint_state(captures[0])}, None, None, None, None])
                index = len(self.rows)
                self.rows.append({'episode_index': source_index, 'frame_index': step,
                    '_physical_source': str(source_index), '_split': split,
                    '_camera_time': current['camera']['source_timestamp_monotonic'],
                    '_source_capture_frame': current_frame,
                    'observation.state': _joint_state(current), 'action': action,
                    '_physical_observation_history': history})
                self.episodes[source_index].append(index)
                self.mappings.append({'frame': index, 'episode_index': source_index,
                    'step': step, 'source_index': source_index, 'split': split,
                    'observation': {**_reference(current), 'gripper_valid': True},
                    'target_time_monotonic': ((1. - weight) * lower['robot']['received_monotonic']
                                              + weight * upper['robot']['received_monotonic']),
                    'joint_retarget': {'lower': _reference(lower), 'upper': _reference(upper),
                        'upper_weight': weight, 'lower_joint_degrees': lower['robot']['joints'],
                        'upper_joint_degrees': upper['robot']['joints']},
                    'derived_action_radians_fraction': action})
        self.proof = {'schema': 'fr5-stream-retarget-compact-v2', 'training_data_kind': 'captured',
                      'source_count': 8, 'sources': sources, 'frames': len(self.rows)}
        self._write_mappings()

    def _write_mappings(self):
        (self.dataset / 'meta/stream_source_frames.jsonl').write_bytes(
            b''.join(_json_bytes(mapping) + b'\n' for mapping in self.mappings))

    def _attach(self):
        return cup_data.attach_cup_relative(self.rows, self.episodes, self.proof, self.dataset)

    def assert_vector_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=9)

    def assert_no_partial_rows(self):
        self.assertTrue(all('_cup_relative_state' not in row and '_cup_relative_action' not in row
                            and '_cup_relative_history' not in row for row in self.rows))

    def test_interpolated_action_and_causal_measured_history(self):
        result = self._attach()
        first, second = self.rows[:2]
        self.assert_vector_close(first['_cup_relative_state'],
                                 [.1, .05, .02, math.radians(90), math.radians(-2),
                                  math.radians(-70), 1., 1.])
        self.assert_vector_close(first['_cup_relative_action'],
                                 [.1025, .05125, .0205, math.radians(90.25), math.radians(-2.125),
                                  math.radians(-69.95), .76])
        self.assert_vector_close(second['_cup_relative_action'],
                                 [.185, .0925, .037, math.radians(98.5), math.radians(-6.25),
                                  math.radians(-68.3), .2])
        self.assertEqual(first['_cup_relative_history'], [None] * 6)
        self.assertEqual(second['_cup_relative_history'][0]['age_s'], .5)
        self.assert_vector_close(second['_cup_relative_history'][0]['state'],
                                 [.12, .06, .024, math.radians(92), math.radians(-3),
                                  math.radians(-69.6), 0., 0.])
        self.assert_vector_close(second['_cup_relative_history'][1]['state'], first['_cup_relative_state'])
        self.assertEqual(second['_cup_relative_history'][2:], [None] * 4)
        self.assertEqual(len(result['cup_sources']), 8)
        self.assertEqual(result['cup_sources'][7]['cup_base_xyz_mm'], [170., -793., 5.])
        self.assertEqual(result['cup_sources'][7]['initial_cup_uv'], [1207., 700.])
        self.assertEqual(result['cup_sources'][7]['split'], 'validation')
        self.assertEqual(result['cup_sources'][7]['capture_journal_sha256'],
                         self.proof['sources'][7]['capture_journal_sha256'])
        coverage = result['coverage']
        self.assertEqual((coverage['source_count'], coverage['rows'], coverage['state_rows'],
                          coverage['action_rows'], coverage['train_rows'], coverage['validation_rows']),
                         (8, 16, 16, 16, 14, 2))
        self.assertEqual((coverage['history_entries'], coverage['missing_history_entries'],
                          coverage['history_entries_not_observation_frames']), (16, 80, 8))
        json.dumps(result, allow_nan=False)

    def test_changed_capture_rejected_by_existing_receipt(self):
        path = Path(self.proof['sources'][0]['trial']) / 'continuous/capture.jsonl'
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'capture journal differs'):
            self._attach()
        self.assert_no_partial_rows()

    def test_changed_geometry_rejected_by_existing_receipt(self):
        path = Path(self.proof['sources'][0]['trial']) / 'geometry.json'
        geometry = json.loads(path.read_bytes())
        geometry['cup_model']['base_center_mm'][0] += 10.
        path.write_bytes(_json_bytes(geometry))
        with self.assertRaisesRegex(ValueError, 'geometry.json differs'):
            self._attach()
        self.assert_no_partial_rows()

    def test_eight_distinct_original_sources_required(self):
        self.proof['sources'][7]['trial'] = self.proof['sources'][6]['trial']
        with self.assertRaisesRegex(ValueError, 'expected original demonstration'):
            self._attach()
        self.assert_no_partial_rows()

    def test_physical_history_must_match_causal_source(self):
        self.rows[-1]['_physical_observation_history'][0]['state'] = copy.deepcopy(
            self.rows[-1]['observation.state'])
        with self.assertRaisesRegex(ValueError, 'joint state and selected capture frame differ'):
            self._attach()
        self.assert_no_partial_rows()

    def test_retarget_weight_must_match_the_original_target_time(self):
        self.mappings[-1]['joint_retarget']['upper_weight'] = .25
        self._write_mappings()
        with self.assertRaisesRegex(ValueError, 'do not bracket the mapped target time'):
            self._attach()
        self.assert_no_partial_rows()

    def test_cup_geometry_must_reference_the_recorded_observation(self):
        trial = Path(self.proof['sources'][0]['trial'])
        observation = json.loads((trial / 'source-observation.json').read_bytes())
        observation['camera']['capture']['received_monotonic'] -= 1.
        raw = _json_bytes(observation)
        control = json.loads((trial / 'control-manifest.json').read_bytes())
        control['files']['source-observation.json']['sha256'] = _sha(raw)
        (trial / 'source-observation.json').write_bytes(raw)
        (trial / 'control-manifest.json').write_bytes(_json_bytes(control))
        with self.assertRaisesRegex(ValueError, 'cup and initial observation differ'):
            self._attach()
        self.assert_no_partial_rows()

    def test_cup_observation_must_precede_episode(self):
        trial = Path(self.proof['sources'][0]['trial'])
        observation = json.loads((trial / 'source-observation.json').read_bytes())
        observation['camera']['capture']['received_monotonic'] = self.rows[0]['_camera_time'] + 1.
        observation_raw = _json_bytes(observation)
        geometry = json.loads((trial / 'geometry.json').read_bytes())
        geometry['source_observation_sha256'] = _sha(observation_raw)
        geometry_raw = _json_bytes(geometry)
        control = json.loads((trial / 'control-manifest.json').read_bytes())
        control['files']['source-observation.json']['sha256'] = _sha(observation_raw)
        control['files']['geometry.json']['sha256'] = _sha(geometry_raw)
        (trial / 'source-observation.json').write_bytes(observation_raw)
        (trial / 'geometry.json').write_bytes(geometry_raw)
        (trial / 'control-manifest.json').write_bytes(_json_bytes(control))
        with self.assertRaisesRegex(ValueError, 'cup observation follows'):
            self._attach()
        self.assert_no_partial_rows()

    def test_mapped_capture_requires_matching_base_frame(self):
        path = Path(self.proof['sources'][0]['trial']) / 'continuous/capture.jsonl'
        captures = [json.loads(line) for line in path.read_bytes().splitlines()]
        captures[0]['robot']['user'] = 1
        raw = b''.join(_json_bytes(capture) + b'\n' for capture in captures)
        path.write_bytes(raw)
        self.proof['sources'][0]['capture_journal_sha256'] = _sha(raw)
        with self.assertRaisesRegex(ValueError, 'tool 0 and user 0'):
            self._attach()
        self.assert_no_partial_rows()


if __name__ == '__main__':
    unittest.main()
