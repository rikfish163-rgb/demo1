"""CPU evidence joins and immutable inputs for shared terminal evaluation."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'fr5_lerobot/diagnostics'))

import prepare_cup_terminal_evaluation as prepare
from fr5_jev.cup_timing import CupTiming
from evaluate_cup_timing_policy import SETDOWN_CLEAN_PROFILE, SETDOWN_FIXED_PROFILE, model_request
from replay_true_five_second_pi05 import model_request as fit_request


ROOT = '/recorded/source-7'
CUP = {'camera_serial': 'gemini', 'observed_monotonic': 99., 'base_xyz_mm': [0., 0., 0.],
       'rim_xyz_mm': [0., 0., 110.], 'table_normal': [0., 0., 1.], 'table_offset_mm': 0.}


def fixture(*, fit=False, corrected=False):
    timing = {'state': {'phase': 'at_pickup', 'hover_seconds': None, 'visual': 'missing',
                        'support_gap_mm': None, 'setdown_age_s': None},
              'camera_monotonic': 100.6, 'feedback_monotonic': 100.603,
              'contact_monotonic': None, 'hover_start_monotonic': None,
              'flange_rise_mm': None, 'rim_rise_mm': None,
              'pickup_reference_error_mm': [0., 0., 0.]}
    tracking = {'native_contact': 0, 'rim_xyz_mm': None, 'rim_minus_jaw_mm': None, 'reason': 'rim_not_observed'}
    request = {'schema': 'fr5-stream-request-v1', 'id': 'release', 'task': '抓杯，悬停4秒后放下并松开夹爪',
               'source_time': 100.6, 'source_clock': 'host_monotonic', 'rgb_jpeg_base64': 'YXJjaGl2ZWQ=',
               'state': [0.] * 6 + [.99, 1.], 'cup_relative_state': [.1597, .0713, .0958, 0., 0., 0., .99, 1.],
               'initial_cup_uv': [1400., 700.],
               'observation_history': [{'age_s': .5, 'state': [0.] * 8}] + [None] * 5,
               'cup_timing': deepcopy(timing['state'])}
    case = {'case_id': 'release', 'group': 'train_fit' if fit else 'heldout_source_7', 'kind': 'retreat',
            'source_index': 7, 'source_root': ROOT, 'source_frame_index': 3, 'tracker_start_frame_index': 0,
            'initial_cup': deepcopy(CUP), 'request': request,
            'targets': [{'target_time_monotonic': 100.766, 'cup_relative_action': [.16, .07, .096, 0., 0., 0., .99],
                         'cut_after': False}], 'targets_camera_aligned': [{'source_frame_index': 4}],
            'event_times': {'release_issued_monotonic': 100.5}}
    if fit:
        case['observers'] = {'setdown': {'tracking': deepcopy(tracking), 'timing': deepcopy(timing),
                                        'model_timing': deepcopy(timing['state']), 'support_gap_source': 'observer'},
                             'old': {'model_timing': {'phase': 'at_pickup'}}}
    else:
        case.update(tracking=deepcopy(tracking), timing_setdown=deepcopy(timing))
    after = deepcopy(timing)
    after['state']['phase'] = 'released'
    capture = {'frame_index': 3, 'camera': {'source_timestamp_monotonic': 100.6},
               'robot': {'received_monotonic': 100.603, 'gripper_position': 99}, 'gripper_feedback_valid': True}
    source = {'id': 'source-7', 'trial': ROOT, 'initial_cup': deepcopy(CUP), 'start_frame': 0,
              'evidence': '/audit/source-7.jsonl', 'method': 'verified_original_frame_audit',
              'records': {3: {'capture': capture, 'tracking': tracking, 'before': timing, 'after': after,
                               'was_released': True, 'trace_line_1based': 4}}}
    if corrected:
        prepare.set_phase(case, 'released')
    return case, source


def causal_rows():
    rows = []
    for frame, (contact, position, valid) in enumerate(((0, 99, True), (1, 78, True),
                                                       (None, 99, False), (0, 99, True))):
        stamp = 100. + frame * .2
        capture = {'frame_index': frame, 'camera': {'source_timestamp_monotonic': stamp,
                                                   'source_camera_serial': 'gemini'},
                   'robot': {'received_monotonic': stamp + .003, 'gripper_position': position,
                             'flange': [159.7, 71.3, 95.8, 0., 0., 0.], 'tcp_speed': [0., 0.], 'tool': 0, 'user': 0},
                   'gripper_feedback_valid': valid}
        rows.append({'capture': capture, 'tracking': {'new': {'native_contact': contact, 'rim_xyz_mm': None,
                                                              'rim_minus_jaw_mm': None}}})
    return rows


class InputJoinTests(unittest.TestCase):
    def test_release_changes_only_phase_and_keeps_the_original_case(self):
        case, source = fixture()
        saved = deepcopy(case)
        corrected, detail = prepare.prepare_case(case, source)
        self.assertEqual(case, saved)
        self.assertEqual(detail['status'], 'assessed')
        self.assertEqual(detail['changed_paths'], ['/timing_setdown/state/phase'])
        self.assertEqual(corrected['request'], saved['request'])
        self.assertEqual(corrected['targets'], saved['targets'])
        self.assertEqual(corrected['targets_camera_aligned'], saved['targets_camera_aligned'])

    def test_fit_updates_both_consumed_phase_copies(self):
        case, source = fixture(fit=True)
        saved = deepcopy(case)
        corrected, detail = prepare.prepare_case(case, source)
        self.assertEqual(case, saved)
        self.assertEqual(len(detail['changed_paths']), 2)
        self.assertEqual(corrected['observers']['setdown']['timing']['state']['phase'], 'released')
        self.assertEqual(corrected['observers']['setdown']['model_timing']['phase'], 'released')
        self.assertEqual(corrected['observers']['old'], saved['observers']['old'])

    def test_real_evaluation_builders_consume_new_phase_with_original_task_and_image(self):
        for fit in (False, True):
            with self.subTest(fit=fit):
                case, source = fixture(fit=fit)
                corrected, _ = prepare.prepare_case(case, source)
                for profile in (SETDOWN_CLEAN_PROFILE, SETDOWN_FIXED_PROFILE):
                    build = (lambda c: fit_request(c, 'setdown', profile)) if fit else (
                        lambda c: model_request(c, 'candidate', 'native', profile=profile))
                    old_request, new_request = build(case), build(corrected)
                    self.assertEqual(new_request['cup_timing']['phase'], 'released')
                    self.assertEqual(list(prepare.differences(old_request, new_request)), [('cup_timing', 'phase')])
                    self.assertEqual(new_request['task'], case['request']['task'])
                    self.assertEqual(new_request['rgb_jpeg_base64'], case['request']['rgb_jpeg_base64'])
                    self.assertEqual(corrected['request']['cup_timing']['phase'], 'at_pickup')

    def test_wrong_source_frame_time_or_start_remains_unassessed(self):
        edits = ((('source_root',), '/recorded/other-source'), (('source_frame_index',), 4),
                 (('request', 'source_time'), 100.8), (('request', 'source_clock'), 'device_time'),
                 (('source_index',), 6), (('tracker_start_frame_index',), 1))
        for path, value in edits:
            with self.subTest(path=path):
                case, source = fixture()
                item = case
                for key in path[:-1]:
                    item = item[key]
                item[path[-1]] = value
                corrected, detail = prepare.prepare_case(case, source)
                self.assertEqual(corrected, case)
                self.assertEqual(detail['status'], 'unassessed')
                self.assertTrue(detail['unassessed_reasons'])

    def test_missing_release_evidence_never_labels_an_open_gripper_as_released(self):
        for key, value in (('valid', False), ('contact', 1), ('released', False)):
            with self.subTest(key=key):
                case, source = fixture()
                record = source['records'][3]
                if key == 'valid':
                    record['capture']['gripper_feedback_valid'] = value
                elif key == 'contact':
                    record['tracking']['native_contact'] = value
                    case['tracking']['native_contact'] = value
                else:
                    record['was_released'] = value
                corrected, detail = prepare.prepare_case(case, source)
                self.assertEqual(corrected, case)
                self.assertIn('missing_confirmed_release', detail['unassessed_reasons'])

    def test_clock_tracking_and_fit_model_state_mismatches_are_not_patched(self):
        for field in ('hover_seconds', 'support_gap_mm', 'setdown_age_s'):
            with self.subTest(field=field):
                case, source = fixture()
                case['timing_setdown']['state'][field] = 1.
                corrected, detail = prepare.prepare_case(case, source)
                self.assertEqual(corrected, case)
                self.assertIn('nonphase_timing_mismatch', detail['unassessed_reasons'])
        case, source = fixture()
        case['tracking']['rim_minus_jaw_mm'] = [1., 2., 3.]
        corrected, detail = prepare.prepare_case(case, source)
        self.assertEqual(corrected, case)
        self.assertIn('tracking_mismatch', detail['unassessed_reasons'])
        case, source = fixture(fit=True)
        case['observers']['setdown']['model_timing']['phase'] = 'unknown'
        corrected, detail = prepare.prepare_case(case, source)
        self.assertEqual(corrected, case)
        self.assertIn('model_timing_mismatch', detail['unassessed_reasons'])

    def test_already_corrected_pgi_is_an_unchanged_assessed_case(self):
        case, source = fixture(corrected=True)
        corrected, detail = prepare.prepare_case(case, source)
        self.assertEqual(corrected, case)
        self.assertEqual(detail['status'], 'assessed')
        self.assertFalse(detail['phase_changed'])

    def test_missing_case_is_retained_and_marks_the_bundle_unready(self):
        case, _ = fixture()
        original = {'schema': 'fr5-cup-timing-replay-inputs-v1', 'timing_ready': True, 'cases': [case]}
        saved = deepcopy(original)
        outputs, details, counts = prepare.prepare_inputs({'heldout-inputs.json': original}, {}, {},
                                                          Path('/audit'), Path('/output'),
                                                          {'heldout-inputs.json': Path('/input/heldout.json')})
        self.assertEqual(original, saved)
        self.assertEqual(outputs['heldout-inputs.json']['cases'], original['cases'])
        self.assertFalse(outputs['heldout-inputs.json']['terminal_phase_evaluation']['ready'])
        self.assertEqual(counts['heldout-inputs.json']['unassessed_cases'], 1)
        self.assertEqual(details[0]['unassessed_reasons'], ['missing_source_evidence'])

    def test_failed_replay_review_and_role_are_preserved(self):
        case, source = fixture()
        case['group'] = 'failed_trial_evaluation_only'
        original = {'cases': [case], 'failed_trial_use': 'evaluation_only',
                    'user_review': {'verdict': 'failure'}, 'training_ready': False}
        outputs, _, counts = prepare.prepare_inputs({'trial01-inputs.json': original}, {ROOT: source}, {},
                                                     Path('/audit'), Path('/output'),
                                                     {'trial01-inputs.json': Path('/input/trial01.json')})
        result = outputs['trial01-inputs.json']
        self.assertEqual(result['user_review'], original['user_review'])
        self.assertEqual(result['failed_trial_use'], 'evaluation_only')
        self.assertFalse(result['training_ready'])
        self.assertEqual(result['cases'][0]['group'], 'failed_trial_evaluation_only')
        self.assertEqual(counts['trial01-inputs.json']['phase_changed_cases'], 1)

    def test_verification_rejects_original_observation_or_target_changes(self):
        edits = ((('request', 'rgb_jpeg_base64'), 'other'), (('request', 'source_time'), 101.),
                 (('request', 'state', 6), .78), (('request', 'observation_history', 0, 'age_s'), .6),
                 (('request', 'task'), '悬停5秒'), (('targets', 0, 'target_time_monotonic'), 101.),
                 (('tracking', 'native_contact'), 1), (('timing_setdown', 'contact_monotonic'), 99.),
                 (('timing_setdown', 'state', 'setdown_age_s'), .3))
        for path, value in edits:
            with self.subTest(path=path):
                original, source = fixture()
                corrected, _ = prepare.prepare_case(original, source)
                item = corrected
                for key in path[:-1]:
                    item = item[key]
                item[path[-1]] = value
                with self.assertRaises(ValueError):
                    prepare.verify_case(original, corrected)
        self.assertEqual(list(prepare.differences({'valid': True}, {'valid': 1})), [('valid',)])

    def test_audited_sources_skip_timer_and_rgbd_replay(self):
        case, source = fixture()
        saved = {ROOT: {'id': 'source-7', 'trial': ROOT}}
        with patch.object(prepare, 'load_audit', return_value=({}, saved, {})), \
                patch.object(prepare, 'audited_source', return_value=source) as lookup, \
                patch.object(prepare, 'timer_class', side_effect=AssertionError('unexpected timer replay')), \
                patch.object(prepare, 'capture_tracking', side_effect=AssertionError('unexpected RGB-D replay')), \
                redirect_stdout(io.StringIO()):
            _, evidence, errors = prepare.collect_evidence({'heldout': {'cases': [case]}}, Path('/audit'), PROJECT)
        lookup.assert_called_once()
        self.assertEqual(evidence[ROOT], source)
        self.assertFalse(errors)


class CausalReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = cls.after = CupTiming

    def test_dense_feedback_replay_preserves_release_phase_and_clocks(self):
        source = {'id': 'trial', 'trial': ROOT, 'initial_cup': CUP, 'evidence': '/capture.jsonl'}
        result = prepare.causal_source(source, {0, 1, 2, 3}, causal_rows(), self.before, self.after)
        records = result['records']
        self.assertEqual([records[i]['after']['state']['phase'] for i in range(4)],
                         ['at_pickup', 'grasp_contact', 'unknown', 'released'])
        self.assertEqual(result['changed_phase_frames'], [])
        self.assertEqual(result['nonphase_timing_changes'], 0)
        self.assertEqual(result['internal_clock_changes'], 0)
        self.assertFalse(records[0]['was_released'])
        self.assertEqual(list(prepare.differences(records[3]['before'], records[3]['after'])), [])

    def test_replay_stops_at_the_selected_observation_without_future_feedback(self):
        rows = causal_rows()
        rows[3]['capture']['camera']['source_timestamp_monotonic'] = -1.
        source = {'id': 'trial', 'trial': ROOT, 'initial_cup': CUP, 'evidence': '/capture.jsonl'}
        result = prepare.causal_source(source, {2}, rows, self.before, self.after)
        self.assertEqual(result['frames_replayed'], 3)
        self.assertEqual(result['records'][2]['after']['state']['phase'], 'unknown')

    def test_replay_rejects_saved_timing_that_does_not_match_the_baseline(self):
        rows = causal_rows()
        rows[0]['timing'] = {'new': {'state': {'phase': 'released'}}}
        source = {'id': 'trial', 'trial': ROOT, 'initial_cup': CUP, 'evidence': '/capture.jsonl'}
        with self.assertRaisesRegex(ValueError, 'Original timing replay differs'):
            prepare.causal_source(source, {3}, rows, self.before, self.after)


if __name__ == '__main__':
    unittest.main()
