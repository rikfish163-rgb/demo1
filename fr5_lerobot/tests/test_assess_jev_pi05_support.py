"""Offline time alignment and support-reference checks for saved forecasts."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from fr5_lerobot.diagnostics import assess_jev_pi05_support as support


class FakeKinematics:
    def point(self, radians):
        return np.rad2deg(np.asarray(radians[:3], dtype=float))


def frame(index, stamp, *, xyz=(0., 0., 0.), valid=True, contact=2, motion=1, speed=.2):
    return {'frame_index': index, 'camera': {'source_timestamp_monotonic': stamp - .002},
            'gripper_feedback_valid': valid,
            'robot': {'received_monotonic': stamp, 'joints': [*xyz, 0., 0., 0.],
                      'gripper_position': 78 if valid else 0, 'gripper_motiondone': contact,
                      'motion_done': motion, 'tcp_speed': [speed, 0.]}}


def record(grid=11.6):
    return {'source_index': 0, 'split': 'validation', 'episode': 0, 'row_index': 17,
            'case_index': 3, 'source_capture_frame': 4, 'draw': 1, 'seed': 123,
            'source_camera_time_monotonic': grid - .05, 'source_grid_time_monotonic': grid,
            'state': [0.] * 6 + [.78, 1.], 'tags': ['open_at'], 'events': [],
            'absolute_actions': [[0.] * 6 + [.76] for _ in range(50)],
            'target_absolute': [[0.] * 6 + [1.] for _ in range(50)],
            'valid_mask': [True] * 50}


class SupportAnalysisTests(unittest.TestCase):
    def setUp(self):
        for name in ('socket.socket.connect', 'socket.socket.connect_ex'):
            guard = patch(name, side_effect=AssertionError('Offline test cannot connect to a device'))
            guard.start()
            self.addCleanup(guard.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trial = self.root / 'trial'
        (self.trial / 'continuous').mkdir(parents=True)
        (self.trial / 'geometry.json').write_text(json.dumps({'table': {'normal': [0., 0., 1.],
                                                                       'offset_mm': 0.}}))
        journal = []
        for sequence, phase, stamp, jaw in ((1, 'close', 1., 76.), (2, 'lift_probe', 2., 76.),
                                             (3, 'lift_full', 3., 76.), (4, 'return', 10., 76.),
                                             (5, 'release', 12., 100.)):
            journal.append({'sequence': sequence, 'issued_monotonic': stamp, 'executed': True,
                            'kind': 'gripper' if phase in ('close', 'release') else 'arm',
                            'target': [0.] * 6 + [jaw], 'teacher': {'cycle': 1, 'phase': phase}})
            if phase in ('close', 'release'):
                (self.trial / f'command-{sequence:04d}.json').write_text(json.dumps(
                    {'method': 'MoveGripper', 'args': [1, jaw], 'status': 'completed'}))
        (self.trial / 'target-command-journal.json').write_text(json.dumps(journal))
        frames = [frame(0, 1.1), frame(1, 5.1, xyz=(0., 0., 50.)),
                  frame(2, 10.5, xyz=(0., 0., 25.), motion=0),
                  frame(3, 11.7), frame(4, 11.85), frame(5, 11.95),
                  frame(6, 12.1, valid=False, contact=0)]
        (self.trial / 'continuous/capture.jsonl').write_text(
            ''.join(json.dumps(item) + '\n' for item in frames))
        self.source = {'source_index': 0, 'split': 'validation', 'trial': str(self.trial),
                       'hold_records': [{'repetition': 1, 'start_monotonic': 5., 'end_monotonic': 10.,
                                         'start_source_frame': 1, 'end_source_frame': 2,
                                         'lift_command_sequence': 3, 'return_command_sequence': 4}]}
        self.kin = FakeKinematics()
        self.evidence = support.SourceEvidence(self.source, self.kin)
        self.cycle = self.evidence.cycles[0]

    def test_recorded_reference_uses_stable_feedback_and_explicit_plane(self):
        reference = self.cycle['reference']
        self.assertEqual(reference['status'], 'stable_jaw_contact_proxy')
        self.assertEqual(reference['frame_index'], 5)
        self.assertEqual(reference['pre_release_candidate_frames'], [3, 4, 5])
        self.assertEqual(reference['point_base_mm'], [0., 0., 0.])
        self.assertEqual(support.phase_at(1.5, self.cycle), 'after_close_before_lift_probe')
        self.assertEqual(support.phase_at(2.5, self.cycle), 'lift_probe_before_full')
        self.assertEqual(support.phase_at(4., self.cycle), 'lift_full_before_hold')
        self.assertEqual(support.phase_at(5., self.cycle), 'recorded_hold')
        self.assertEqual(support.phase_at(11., self.cycle), 'return_before_release')
        self.assertEqual(support.phase_at(12., self.cycle), 'after_release')

    def test_unknown_gripper_feedback_is_not_reported_as_no_contact(self):
        value = self.evidence.feedback_at(12.1, self.cycle)
        self.assertTrue(value['available'])
        self.assertFalse(value['gripper_feedback_valid'])
        self.assertIsNone(value['jaw_contact_reported'])
        previous = self.evidence.feedback_at(12.01, self.cycle, preceding=True)
        self.assertEqual(previous['frame_index'], 5)
        self.assertTrue(previous['jaw_contact_reported'])
        missing = self.evidence.feedback_at(15., self.cycle, preceding=True)
        self.assertFalse(missing['available'])

    def test_first_raw_opening_uses_the_same_action_position_and_valid_mask(self):
        r = record(grid=4.)
        r['absolute_actions'][0] = [0., 0., math.radians(20.), 0., 0., 0., .99]
        r['valid_mask'][0] = False
        r['absolute_actions'][1] = [0., 0., math.radians(25.), 0., 0., 0., 1.02]
        self.assertEqual(support.first_open(r, 90), 1)
        detail = support.prediction_detail(r, 1, self.evidence, self.cycle, self.kin, 10)
        self.assertAlmostEqual(detail['raw_gripper_percent'], 102.)
        self.assertAlmostEqual(detail['predicted_grasp_vs_release_reference']['height_along_table_normal_mm'], 25.)
        self.assertAlmostEqual(detail['nominal_time_monotonic'], 4. + 2. / 6.)
        self.assertTrue(detail['within_execution_prefix'])

    def test_hold_uses_provenance_interval_for_untagged_anchors(self):
        r = record(grid=4.5)
        r['tags'] = ['uniform']
        r['source_capture_frame'] = 0
        r['absolute_actions'][3][6] = .92
        original = json.loads(json.dumps(r))
        value = support.hold_detail(r, self.evidence, self.cycle, self.kin, 10, 10)
        self.assertEqual(value['anchor_relation_to_recorded_hold'], 'before_hold')
        self.assertEqual(value['hold_ticks'], list(range(3, 11)))
        self.assertEqual(value['first_hold_at_least_90_percent']['tick'], 4)
        self.assertIsNone(value['first_hold_at_least_98_percent'])
        self.assertEqual(value['anchor_measured_gripper_percent'], 78.)
        self.assertEqual(value['anchor_last_move_gripper']['target_percent'], 76.)
        point = next(p for p in value['hold_points'] if p['tick'] == 4)
        self.assertEqual(point['projected_integer_percent'], 92)
        self.assertEqual(point['increase_over_anchor_measured_pp'], 14.)
        self.assertEqual(point['increase_over_anchor_last_command_pp'], 16.)
        self.assertEqual(point['increase_over_nominal_last_command_pp'], 16.)
        summary = support.summarize_hold([value])
        self.assertEqual(summary['projected_vs_anchor_measured']['draws_at_least_2pp'], 1)
        self.assertEqual(summary['projected_vs_anchor_measured']['points_at_least_2pp'], 1)
        self.assertEqual(summary['projected_vs_anchor_measured']['maximum_increase_pp'], 14.)
        self.assertEqual(summary['projected_vs_anchor_last_command']['maximum_increase_pp'], 16.)
        self.assertEqual(r, original)

    def test_hold_opening_beyond_prefix_remains_in_full_forecast(self):
        r = record(grid=3.5)
        r['source_capture_frame'] = 0
        r['absolute_actions'][15][6] = 1.
        prefix = support.hold_detail(r, self.evidence, self.cycle, self.kin, 10, 10)
        full = support.hold_detail(r, self.evidence, self.cycle, self.kin, 50, 10)
        self.assertIsNone(prefix['first_hold_at_least_90_percent'])
        opening = full['first_hold_at_least_90_percent']
        self.assertEqual(opening['tick'], 16)
        self.assertFalse(opening['within_execution_prefix'])
        point = next(p for p in full['hold_points'] if p['tick'] == 16)
        self.assertEqual(point['projected_integer_percent'], 100)
        self.assertEqual(point['increase_over_anchor_measured_pp'], 22.)
        self.assertEqual(point['increase_over_anchor_last_command_pp'], 24.)
        self.assertEqual(support.summarize_hold([prefix])['projected_vs_anchor_measured']
                         ['points_at_least_2pp'], 0)
        self.assertEqual(support.summarize_hold([full])['projected_vs_anchor_measured']
                         ['points_at_least_2pp'], 1)

    def test_native_projection_and_missing_last_command(self):
        from lerobot_robot_fairino.stream_contract import decode_action

        for raw in (-.2, 0., .754, .785, .8, .85259, 1., 1.2):
            with self.subTest(raw=raw):
                self.assertEqual(support.projected_gripper_percent(raw),
                                 decode_action([0.] * 6 + [raw])['gripper_integer_percent'])
        (self.trial / 'command-0001.json').unlink()
        evidence = support.SourceEvidence(self.source, self.kin)
        last = evidence.last_gripper_command(1.1)
        self.assertFalse(last['available'])
        self.assertIsNone(last['target_percent'])
        r = record(grid=4.5)
        r['source_capture_frame'] = 0
        value = support.hold_detail(r, evidence, evidence.cycles[0], self.kin, 10, 10)
        self.assertIsNone(value['hold_points'][0]['increase_over_anchor_last_command_pp'])
        summary = support.summarize_hold([value])
        self.assertEqual(summary['projected_vs_anchor_last_command']['eligible_points'], 0)
        self.assertEqual(summary['projected_vs_anchor_last_command']['missing_points'], 8)

    def test_two_point_opening_counts_position_and_last_target_separately(self):
        r = record(grid=4.5)
        r['source_capture_frame'] = 0
        r['absolute_actions'][2][6] = .79
        r['absolute_actions'][3][6] = .80
        result = support.hold_detail(r, self.evidence, self.cycle, self.kin, 10, 10)
        summary = support.summarize_hold([result])
        measured = summary['projected_vs_anchor_measured']
        command = summary['projected_vs_anchor_last_command']
        self.assertEqual((measured['draws_at_least_2pp'], measured['points_at_least_2pp']), (1, 1))
        self.assertEqual(measured['maximum_increase_pp'], 2.)
        self.assertEqual((command['draws_at_least_2pp'], command['points_at_least_2pp']), (1, 2))
        self.assertEqual(command['maximum_increase_pp'], 4.)
        r['state'][7] = 0.
        invalid = support.hold_detail(r, self.evidence, self.cycle, self.kin, 10, 10)
        invalid_summary = support.summarize_hold([invalid])
        self.assertIsNone(invalid['anchor_measured_gripper_percent'])
        self.assertEqual(invalid_summary['projected_vs_anchor_measured']['eligible_points'], 0)
        self.assertEqual(invalid_summary['projected_vs_anchor_measured']['missing_points'], 8)
        self.assertEqual(invalid_summary['projected_vs_anchor_last_command']['points_at_least_2pp'], 2)

    def test_release_lag_keeps_labels_raw_targets_and_reference_separate(self):
        r = record()
        for i in range(10):
            r['absolute_actions'][i][2] = math.radians(.2 * i)
        r['absolute_actions'][4][6] = .90
        r['absolute_actions'][5][6] = .99
        event = {'kind': 'open', 'where': 'at', 'command_sequence': 5,
                 'event_row_index': 17, 'old_target': .76, 'new_target': 1.}
        value = support.release_detail(r, event, self.evidence, self.cycle, self.kin, 10)
        timing = value['label_switch_timing_prefix']
        self.assertEqual(timing['target_index'], 0)
        self.assertEqual(timing['predicted_index'], 4)
        self.assertAlmostEqual(timing['timing_error_s'], 4. / 6.)
        self.assertEqual([v['tick'] for v in value['waiting_for_label_switch']], [1, 2, 3, 4])
        self.assertEqual(value['at_first_open_label']['raw_gripper_percent'], 76.)
        self.assertEqual(value['first_at_least_90_percent']['tick'], 5)
        self.assertEqual(value['first_at_least_98_percent']['tick'], 6)
        self.assertAlmostEqual(value['first_at_least_90_percent']['predicted_grasp_vs_release_reference']
                               ['height_along_table_normal_mm'], .8)

    def test_censored_opening_and_unavailable_reference_remain_explicit(self):
        r = record()
        r['absolute_actions'][20][6] = .99
        event = {'kind': 'open', 'where': 'at', 'command_sequence': 5,
                 'event_row_index': 17, 'old_target': .76, 'new_target': 1.}
        cycle = {**self.cycle, 'reference': {'status': 'unavailable', 'reason': 'no_feedback'}}
        value = support.release_detail(r, event, self.evidence, cycle, self.kin, 10)
        self.assertTrue(value['label_switch_timing_prefix']['never_switched_in_horizon'])
        self.assertEqual(len(value['waiting_for_label_switch']), 10)
        self.assertFalse(value['first_at_least_98_percent']['within_execution_prefix'])
        self.assertIsNone(value['first_at_least_98_percent']['predicted_grasp_vs_release_reference'])
        summary = support.summarize_release([value])
        self.assertEqual(summary['stable_reference_draws'], 0)
        self.assertEqual(summary['nearest_label_switch']['at_label_distance_from_reference_mm']['count'], 0)
        cycle['reference'] = {'status': 'unstable_or_insufficient_feedback', 'point_base_mm': [0., 0., 0.]}
        weak = support.release_detail(r, event, self.evidence, cycle, self.kin, 10)
        self.assertIsNotNone(weak['at_first_open_label']['predicted_grasp_vs_release_reference'])
        weak_summary = support.summarize_release([weak])
        self.assertEqual(weak_summary['nearest_label_switch']['at_label_distance_from_reference_mm']['count'], 0)
        self.assertEqual(weak_summary['nearest_label_switch']['at_label_raw_gripper_percent']['count'], 1)

    def test_completed_evaluation_integration_and_non_overwriting_cli(self):
        dataset = self.root / 'dataset'
        (dataset / 'meta').mkdir(parents=True)
        (dataset / 'meta/stream_provenance.json').write_text(json.dumps(
            {'fps': 6, 'chunk_size': 50, 'sources': [self.source]}))
        evaluation = self.root / 'evaluation'
        evaluation.mkdir()
        metadata = {'evaluation': 'open_loop_recorded_label_fit_not_task_success',
                    'dataset': str(dataset), 'dataset_matches_checkpoint': True, 'checkpoint': 'checkpoint-4000',
                    'prefix_steps': 10, 'models': {'fine_tuned': {}}}
        (evaluation / 'summary.json').write_text(json.dumps(metadata))
        r = record()
        r['events'] = [{'kind': 'open', 'where': 'at', 'command_sequence': 5,
                        'event_row_index': 17, 'old_target': .76, 'new_target': 1.}]
        r['absolute_actions'][4][6] = .90
        (evaluation / 'fine_tuned.jsonl').write_text(json.dumps(r) + '\n')
        report = support.assess([evaluation], kin=self.kin)
        result = report['evaluations']['evaluation']['models']['fine_tuned']
        self.assertEqual(len(result['release_details']), 1)
        self.assertAlmostEqual(result['summary']['validation']['release_anchors']['at']
                               ['nearest_label_switch']['timing_error_s']['mean'], 4. / 6.)
        output = self.root / 'report.json'
        output.write_text('existing')
        with patch.object(support, 'assess') as assess:
            with self.assertRaises(FileExistsError):
                support.main(['--evaluation', str(evaluation), '--output', str(output)])
            assess.assert_not_called()
        self.assertEqual(output.read_text(), 'existing')


if __name__ == '__main__':
    unittest.main()
