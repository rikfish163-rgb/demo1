"""CPU geometry and timing checks for saved native prediction chunks."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from fr5_runtime.kinematics import transform
from fr5_lerobot.training import summarize_jev_pi05 as analysis


JOINTS_DEG = [65.72217559814453, -30.337493896484375, 52.30057907104492,
              -18.186397552490234, 132.82252502441406, 3.894381284713745]
TCP = [-272.2250061035156, -754.3921508789062, 112.32106018066406,
       92.76986694335938, -1.3237062692642212, -67.22650146484375]
OFFSET = [5.208, 8.072, 162.]


def record(*, valid_steps=50, split='train'):
    joints = np.radians(JOINTS_DEG).tolist()
    targets = [[*joints, 1.] for _ in range(50)]
    return {'absolute_actions': deepcopy(targets), 'target_absolute': targets,
            'state': [*joints, 1., 1.], 'valid_mask': [True] * valid_steps + [False] * (50-valid_steps),
            'source_index': 0 if split == 'train' else 7, 'row_index': 3,
            'split': split, 'draw': 0, 'tags': ['uniform'], 'events': [], 'holds': [],
            'source_grid_time_monotonic': 10., 'source_camera_time_monotonic': 9.9}


def close_event():
    return {'old_target': 1., 'new_target': .76, 'kind': 'close',
            'where': 'before', 'event_row_index': 4}


class GraspPositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kinematics = analysis.GraspKinematics(analysis.PROJECT / 'hardware/fr5.urdf', OFFSET)

    def test_grasp_point_matches_the_measured_controller_pose(self):
        expected = transform(TCP)[:3, :3] @ OFFSET + np.asarray(TCP[:3])
        point = self.kinematics.point(np.radians(JOINTS_DEG))
        np.testing.assert_allclose(point, expected, atol=.05, rtol=0)

    def test_one_degree_base_rotation_is_measured_in_millimeters(self):
        case = record(valid_steps=2)
        case['absolute_actions'][0][0] += math.radians(1.)
        for action in case['absolute_actions'][2:]:
            action[0] += 1.
        summary = analysis.Summary(self.kinematics, 10)
        summary.add(case)
        result = summary.result()['train']['position']['uniform']
        anchor = self.kinematics.point(np.radians(JOINTS_DEG))
        expected = 2 * np.linalg.norm(anchor[:2]) * math.sin(math.radians(.5))
        self.assertAlmostEqual(result['first']['grasp_position_error_mm']['mean'], expected, places=6)
        self.assertAlmostEqual(result['full']['grasp_position_error_mm']['mean'], expected / 2, places=6)
        self.assertEqual(result['full']['grasp_position_error_mm']['count'], 2)
        self.assertAlmostEqual(result['first']['absolute_vertical_error_mm']['max'], 0., places=9)

    def test_hold_metrics_exclude_predictions_after_the_recorded_hold(self):
        case = record(valid_steps=10)
        case['tags'].append('hold_start')
        case['holds'] = [{'start_monotonic': 10., 'end_monotonic': 10.7}]
        for action in case['absolute_actions'][4:]:
            action[1] += math.radians(3.)
        summary = analysis.Summary(self.kinematics, 10)
        summary.add(case)
        metrics = summary.result()['train']['position']['hold_start']['prefix']
        self.assertEqual(metrics['recorded_hold']['grasp_position_error_mm']['count'], 4)
        self.assertEqual(metrics['recorded_hold']['grasp_position_error_mm']['max'], 0.)
        self.assertGreater(metrics['grasp_position_error_mm']['max'], 1.)

    def test_repeated_draws_keep_physical_anchor_and_split_counts(self):
        summary = analysis.Summary(self.kinematics, 10)
        for draw in range(3):
            case = record(valid_steps=1)
            case['draw'] = draw
            summary.add(case)
        summary.add(record(valid_steps=1, split='validation'))
        result = summary.result()
        self.assertEqual(result['train']['unique_anchors'], 1)
        self.assertEqual(result['train']['prediction_draws'], 3)
        self.assertEqual(result['validation']['unique_anchors'], 1)
        self.assertEqual(result['validation']['prediction_draws'], 1)


class SwitchTimingTests(unittest.TestCase):
    def test_before_switch_delay_and_camera_age_are_separate(self):
        case = record()
        for action in case['target_absolute'][1:]:
            action[6] = .76
        for action in case['absolute_actions'][3:]:
            action[6] = .76
        timing = analysis.switch_timing(case, close_event(), 10)
        self.assertEqual(timing['target_index'], 1)
        self.assertEqual(timing['predicted_index'], 3)
        self.assertAlmostEqual(timing['timing_error_s'], 2 / 6)
        self.assertAlmostEqual(timing['target_time_from_camera_s'], .1 + 2 / 6)

    def test_early_prediction_is_not_counted_as_same_step(self):
        case = record()
        for action in case['target_absolute'][1:]:
            action[6] = .76
        for action in case['absolute_actions']:
            action[6] = .76
        metrics = analysis.TimingMetrics()
        metrics.add(analysis.switch_timing(case, close_event(), 10))
        result = metrics.result()
        self.assertEqual(result['early'], 1)
        self.assertEqual(result['same_label_step'], 0)
        self.assertAlmostEqual(result['timing_error_when_switched_s']['mean'], -1 / 6)

    def test_switch_outside_prefix_is_censored_in_prefix_only(self):
        case = record()
        for action in case['target_absolute'][1:]:
            action[6] = .76
        for action in case['absolute_actions'][12:]:
            action[6] = .76
        prefix = analysis.switch_timing(case, close_event(), 10)
        full = analysis.switch_timing(case, close_event(), 50)
        self.assertTrue(prefix['never_switched_in_horizon'])
        self.assertIsNone(prefix['timing_error_s'])
        self.assertAlmostEqual(prefix['late_lower_bound_s'], 8 / 6)
        self.assertFalse(full['never_switched_in_horizon'])
        self.assertAlmostEqual(full['timing_error_s'], 11 / 6)

    def test_padding_cannot_supply_a_predicted_switch(self):
        case = record(valid_steps=2)
        case['target_absolute'][1][6] = .76
        for action in case['absolute_actions'][2:]:
            action[6] = .76
        timing = analysis.switch_timing(case, close_event(), 50)
        self.assertTrue(timing['never_switched_in_horizon'])
        self.assertEqual(timing['late_lower_bound_s'], 0.)


class OutputTests(unittest.TestCase):
    def test_complete_output_is_saved_without_overwriting_or_reading_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            evaluation = Path(temporary) / 'evaluation'
            evaluation.mkdir()
            output = evaluation / 'cartesian-timing.json'
            (evaluation / 'summary.json').write_text(json.dumps({
                'evaluation': 'open_loop_recorded_label_fit_not_task_success',
                'dataset_matches_checkpoint': True, 'prefix_steps': 10,
                'checkpoint': '/checkpoint', 'models': {'fine_tuned': {}}}))
            case = record(valid_steps=1)
            (evaluation / 'fine_tuned.jsonl').write_text(json.dumps(case) + '\n')
            with patch.object(analysis, 'EVALUATIONS', evaluation):
                result = analysis.summarize(evaluation, output, OFFSET)
                self.assertEqual(json.loads(output.read_text()), result)
                with self.assertRaises(FileExistsError):
                    analysis.summarize(evaluation, output, OFFSET)
            self.assertEqual(result['models']['fine_tuned']['train']['position']['all']['full']
                             ['grasp_position_error_mm']['count'], 1)
            self.assertIn('not_task_success', result['evaluation'])


if __name__ == '__main__':
    unittest.main()
