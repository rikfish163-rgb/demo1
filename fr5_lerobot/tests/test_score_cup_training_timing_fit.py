"""Check training timing scores against recorded clocks and PGI feedback."""

from bisect import bisect_right
import copy
import unittest

from fr5_lerobot.diagnostics import score_cup_training_timing_fit as timing


PROFILE = 'state-history-physical-cup-setdown-fixed-v1'


def action(height=100., jaw=.78, x=0.):
    return [x / 1000., 0., height / 1000., 0., 0., 0., jaw]


def recorded_example(kind='release_at', stamp=111., *, visual='table_height'):
    events = {'close_issued_monotonic': 100.2, 'return_issued_monotonic': 109.2,
              'release_issued_monotonic': 111.2, 'contact_confirmed_monotonic': 102.,
              'contact_confirmed_source_frame': 40, 'close_reference_tcp_xyz_mm': [0., 0., 100.],
              'hold_start_monotonic': 104., 'hold_end_monotonic': 109.2,
              'hold_record_actual_seconds': 5.2, 'requested_hold_seconds': 5.,
              'five_second_deadline_monotonic': 109.}
    journal = [{'kind': 'hold', 'sequence': 1, 'executed': False, 'issued_monotonic': 98.,
                'target': [0.] * 6 + [100.]}]
    for sequence, phase, jaw in ((2, 'close', 76.), (3, 'return', 76.), (4, 'release', 100.)):
        journal.append({'kind': 'arm' if phase == 'return' else 'gripper', 'executed': True,
                        'sequence': sequence, 'teacher': {'cycle': 1, 'phase': phase},
                        'issued_monotonic': events[f'{phase}_issued_monotonic'],
                        'target': [0.] * 6 + [jaw]})
    frames = []
    for i in range(251):
        t = 98. + i / 10.
        held = 102. <= t < 111.4
        valid = not (100.2 <= t < 102. or 111.4 <= t < 111.6)
        jaw = 78 if held else 100 if valid else 0
        height = 150. if 104. <= t < 109.2 else 100.
        frames.append({'frame_index': i, 'camera': {'source_timestamp_monotonic': t,
                                                    'source_camera_serial': 'recorded-gemini'},
                       'gripper_feedback_valid': valid,
                       'robot': {'tcp': [0., 0., height, 0., 0., 0.], 'tcp_speed': [.2, 0.],
                                 'received_monotonic': t + .002,
                                 'gripper_motiondone': 2 if held else 1 if valid else 0,
                                 'gripper_position': jaw}})
    records = timing.source_records(frames, journal)
    index = records['times'].index(stamp)
    frame, robot = frames[index], frames[index]['robot']
    valid = frame['gripper_feedback_valid']
    state = [*action(robot['tcp'][2], robot['gripper_position'] / 100. if valid else 0.), float(valid)]
    phase = 'unknown' if not valid else 'hovering' if robot['tcp'][2] == 150. else (
        'at_table_height' if timing.contact(frame) else 'at_pickup')
    model_timing = {'phase': phase, 'hover_seconds': 5. if stamp >= 109.2 else max(0., stamp - 104.),
                    'visual': visual, 'support_gap_mm': robot['tcp'][2] - 100. if timing.contact(frame) else None,
                    'setdown_age_s': stamp - 110.4 if valid and timing.contact(frame) and stamp >= 110.4 else None}
    observer = {'tracking': {'native_contact': 1 if timing.contact(frame) else 0 if valid else None,
                             'rim_minus_jaw_mm': None},
                'timing': {'state': copy.deepcopy(model_timing), 'camera_monotonic': stamp,
                           'feedback_monotonic': robot['received_monotonic']},
                'model_timing': model_timing}
    case = {'case_id': f'source0-cycle1-{kind}-{index}', 'group': 'train_fit', 'split': 'train',
            'source_index': 0, 'cycle': 1, 'kind': kind, 'source_root': '/original/source0',
            'source_frame_index': index, 'event_times': events,
            'initial_cup': {'base_xyz_mm': [0., 0., 0.], 'table_normal': [0., 0., 1.],
                            'camera_serial': 'recorded-gemini'},
            'request': {'id': f'source0-cycle1-{kind}-{index}', 'source_time': stamp,
                        'task': '抓杯，悬停5秒，放稳再松爪', 'cup_relative_state': state},
            'observers': {'setdown': observer}, 'targets': []}
    for tick in range(1, 51):
        nominal = stamp + tick / 6.
        raw_index = bisect_right(records['times'], nominal) - 1
        raw, raw_time = frames[raw_index], records['times'][raw_index]
        command = timing.latched_command(records, raw_time)
        case['targets'].append({'tick': tick, 'nominal_time_monotonic': nominal,
                                'raw_camera_frame': raw_index, 'raw_camera_time_monotonic': raw_time,
                                'raw_camera_age_s': nominal - raw_time,
                                'raw_robot_time_monotonic': raw['robot']['received_monotonic'],
                                'raw_cup_relative_action': action(raw['robot']['tcp'][2], command['jaw_percent'] / 100.)})
    return case, records


def saved_draw(case, actions):
    request = timing.fit.model_request(case, 'setdown', PROFILE)
    return {'case_id': case['case_id'], 'seed': 20261105, 'observer': 'setdown',
            **{key: copy.deepcopy(request[key]) for key in ('task', 'source_time', 'cup_tracking', 'cup_timing')},
            'reply': {'actions': actions, 'wire_action_representation': 'cup-relative-tcp-v1'}}


def score_example(case, records, actions, horizon=3, support=None):
    support = support if support is not None else timing.support_reference(case, records)
    evidence = timing.case_evidence(case, records, support)
    return timing.score_draw(case, saved_draw(case, actions), evidence, support, horizon, PROFILE)


class TrainingTimingFitTests(unittest.TestCase):
    def test_normal_close_cannot_pass_by_closing_after_execution_prefix(self):
        case, records = recorded_example('close_at', 100.)
        late = [action(jaw=1.)] * 3 + [action(jaw=.76)] * 47
        short = score_example(case, records, late)
        long = score_example(case, records, late, 10)
        self.assertEqual('assessed', short['checks']['normal_close']['status'])
        self.assertEqual(3, short['checks']['normal_close']['deadline_tick'])
        self.assertTrue(short['checks']['normal_close']['late_or_missing'])
        self.assertTrue(long['checks']['normal_close']['late_or_missing'])
        self.assertIn('late_normal_close', [v['code'] for v in short['violations']])
        timely = score_example(case, records, [action(jaw=1.)] * 2 + [action(jaw=.76)] * 48)
        self.assertFalse(timely['checks']['normal_close']['late_or_missing'])
        self.assertEqual('pass', timely['gate'])

    def test_far_close_is_separate_from_normal_close_deadline(self):
        case, records = recorded_example('close_at', 100.)
        result = score_example(case, records, [action(jaw=.76, x=26.)] * 50)
        self.assertFalse(result['checks']['normal_close']['late_or_missing'])
        self.assertEqual(['far_close'], [v['code'] for v in result['violations']])

    def test_outside_prefix_deadline_is_unassessed_even_with_an_early_close(self):
        case, records = recorded_example('close_before', 99.8)
        result = score_example(case, records, [action(jaw=.76)] * 50)
        self.assertEqual('unassessed', result['checks']['normal_close']['status'])
        self.assertIsNone(result['checks']['normal_close']['late_or_missing'])
        self.assertEqual('inconclusive', result['gate'])
        self.assertEqual(0, timing.counts([result])['passed'])
        long = score_example(case, records, [action(jaw=.76)] * 50, 10)
        self.assertEqual('assessed', long['checks']['normal_close']['status'])

    def test_supported_release_requires_full_open_not_a_partial_increase(self):
        case, records = recorded_example()
        partial = score_example(case, records, [action(jaw=.81)] * 50)
        self.assertTrue(partial['checks']['late_open']['late_or_missing'])
        self.assertEqual(['late_supported_open'], [v['code'] for v in partial['violations']])
        timely = score_example(case, records, [action()] + [action(jaw=1.)] * 49)
        self.assertEqual('pass', timely['gate'])
        self.assertEqual(1, timing.counts([timely])['late_open_assessable'])

    def test_release_above_table_fails_for_partial_and_full_open(self):
        case, records = recorded_example('return_late', 110.)
        for jaw in (.81, 1.):
            with self.subTest(jaw=jaw):
                result = score_example(case, records, [action(height=103.7, jaw=jaw)] * 50)
                self.assertIn('open_above_support_proxy', [v['code'] for v in result['violations']])
                self.assertEqual('fail', result['gate'])

    def test_near_height_alone_does_not_allow_open_before_support_age(self):
        case, records = recorded_example('return_late', 110.5)
        premature = score_example(case, records, [action(jaw=1.)] * 50)
        self.assertFalse(premature['observations']['observed_ready_proxy'])
        self.assertIn('open_before_stable_support_proxy', [v['code'] for v in premature['violations']])
        settled = score_example(case, records, [action()] + [action(jaw=1.)] * 49)
        self.assertNotIn('open_before_stable_support_proxy', [v['code'] for v in settled['violations']])

    def test_invalid_feedback_never_counts_as_a_pass(self):
        case, records = recorded_example('release_after', 111.5)
        result = score_example(case, records, [action(jaw=1.)] * 50)
        self.assertEqual('unassessed', result['status'])
        self.assertEqual('unassessed', result['checks']['late_open']['status'])
        self.assertIsNone(result['checks']['late_open']['late_or_missing'])
        summary = timing.counts([result])
        self.assertEqual(0, summary['passed'])
        self.assertEqual(0, summary['late_open_assessable'])
        self.assertEqual(1, summary['unassessed'])

    def test_missing_stable_contact_pair_is_unassessed(self):
        case, records = recorded_example()
        retained = [f for f in records['frames'] if f['camera']['source_timestamp_monotonic'] in (102., 110.7, 111.1)]
        sparse = timing.source_records(retained, records['journal'])
        reference = timing.support_reference(case, sparse)
        self.assertEqual({'status': 'unassessed', 'reason': 'missing_stable_contact_pair'}, reference)
        result = score_example(case, records, [action(jaw=1.)] * 50, support=reference)
        self.assertEqual('unassessed', result['status'])
        self.assertEqual(0, timing.counts([result])['passed'])
        self.assertEqual(0, timing.counts([result])['late_open_assessable'])

    def test_contact_feedback_remains_held_after_release_command(self):
        case, records = recorded_example('release_at', 111.3, visual='missing')
        result = score_example(case, records, [action()] * 50)
        observed = result['observations']
        self.assertEqual(100., observed['latched_jaw_command_at_observation']['jaw_percent'])
        self.assertEqual(78, observed['feedback_jaw_percent'])
        self.assertFalse(observed['setdown_gate_recorded_contact'])
        self.assertTrue(observed['raw_feedback_contact'])
        self.assertTrue(observed['observed_ready_proxy'])
        self.assertTrue(observed['contact_excluded_by_release_clock'])
        self.assertEqual('assessed', result['checks']['late_open']['status'])
        self.assertIn('late_supported_open', [v['code'] for v in result['violations']])
        above = score_example(case, records, [action(height=103.7, jaw=1.)] * 50)
        self.assertIn('open_above_support_proxy', [v['code'] for v in above['violations']])

    def test_hover_uses_original_deadline_and_drop_from_peak(self):
        case, records = recorded_example('hover_late', 108.5, visual='raised')
        after = [action(150.), action(150.), action(147.)] + [action(147.)] * 47
        result = score_example(case, records, after)
        self.assertEqual([1, 2], result['checks']['hover']['ticks'])
        self.assertEqual(0., result['checks']['hover']['max_down_mm'])
        self.assertNotIn('early_true_five_second_descent', [v['code'] for v in result['violations']])
        before = [action(153.), action(150.)] + [action(150.)] * 48
        result = score_example(case, records, before)
        self.assertAlmostEqual(3., result['checks']['hover']['max_down_mm'])
        self.assertIn('early_true_five_second_descent', [v['code'] for v in result['violations']])
        self.assertNotIn('early_counterfactual_five_second_descent', [v['code'] for v in result['violations']])

    def test_camera_and_journal_timestamp_mismatches_are_rejected(self):
        case, records = recorded_example()
        wrong = copy.deepcopy(case)
        wrong['request']['source_time'] += .01
        with self.assertRaisesRegex(ValueError, 'raw Gemini timestamp differs'):
            timing.case_evidence(wrong, records, timing.support_reference(case, records))
        wrong = copy.deepcopy(records)
        next(r for r in wrong['journal'] if r.get('teacher', {}).get('phase') == 'release')['issued_monotonic'] += .01
        with self.assertRaisesRegex(ValueError, 'release: original timestamp differs'):
            timing.support_reference(case, wrong)
        wrong = copy.deepcopy(case)
        wrong['targets'][1]['nominal_time_monotonic'] += .01
        with self.assertRaisesRegex(ValueError, 'target 2: original timestamp differs'):
            timing.case_evidence(wrong, records, timing.support_reference(case, records))

    def test_original_label_and_command_deadlines_remain_separate(self):
        case, records = recorded_example('release_at', 111.1)
        record_at_command = next(f for f in records['frames'] if f['camera']['source_timestamp_monotonic'] == 111.2)
        remaining = [f for f in records['frames'] if f is not record_at_command]
        records = timing.source_records(remaining, records['journal'])
        first = case['targets'][0]
        first.update(raw_camera_frame=case['source_frame_index'], raw_camera_time_monotonic=111.1,
                     raw_camera_age_s=first['nominal_time_monotonic'] - 111.1,
                     raw_robot_time_monotonic=111.102, raw_cup_relative_action=action(jaw=.76))
        original = copy.deepcopy(case)
        result = score_example(case, records, [action()] + [action(jaw=1.)] * 49)
        self.assertEqual(1, result['checks']['late_open']['original_command_tick'])
        self.assertEqual(2, result['checks']['late_open']['original_label_tick'])
        self.assertEqual(2, result['checks']['late_open']['deadline_tick'])
        self.assertEqual(1, len(result['observations']['label_vs_nominal_command_mismatches_first_10']))
        self.assertEqual('train_fit', result['group'])
        self.assertEqual('train', result['split'])
        self.assertEqual(case, original)

    def test_saved_request_and_complete_fifty_point_output_are_required(self):
        case, _ = recorded_example()
        row = saved_draw(case, [action()] * 50)
        saved = {'schema': timing.fit.PREDICTION_SCHEMA, 'model': 'setdown', 'seeds': [row['seed']],
                 'checkpoint': '/checkpoint', 'observation_context_profile': PROFILE,
                 'observers': ['setdown'], 'predictions': [row]}
        timing.prediction_index({case['case_id']: case}, saved)
        for field in ('source_time', 'cup_timing'):
            wrong = copy.deepcopy(saved)
            if field == 'source_time':
                wrong['predictions'][0][field] += .1
            else:
                wrong['predictions'][0][field]['setdown_age_s'] += .1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, f'saved request differs in {field}'):
                timing.prediction_index({case['case_id']: case}, wrong)
        for actions in ([action()] * 49, [action()] * 49 + [[*action()[:6], float('nan')]]):
            wrong = copy.deepcopy(saved)
            wrong['predictions'][0]['reply']['actions'] = actions
            with self.assertRaisesRegex(ValueError, '50 x 7'):
                timing.prediction_index({case['case_id']: case}, wrong)

    def test_unselected_incomplete_proposal_and_missing_seed_cannot_pass(self):
        case, _ = recorded_example()
        row = saved_draw(case, [action()] * 50)
        saved = {'schema': timing.fit.PREDICTION_SCHEMA, 'model': 'setdown', 'seeds': [row['seed']],
                 'checkpoint': '/checkpoint', 'observation_context_profile': PROFILE,
                 'observers': ['setdown'], 'predictions': [row]}
        row['reply']['cup_proposal_selection'] = {
            'selected_index': 0, 'matched': True, 'draws': 2,
            'proposals': [{'actions': row['reply']['actions']}, {'actions': [action()] * 49}]}
        with self.assertRaisesRegex(ValueError, '50 x 7'):
            timing.prediction_index({case['case_id']: case}, saved)
        row['reply'].pop('cup_proposal_selection')
        saved['seeds'].append(20261113)
        with self.assertRaisesRegex(ValueError, 'exact original cases and seeds'):
            timing.prediction_index({case['case_id']: case}, saved)

    def test_validation_or_failed_cases_cannot_enter_training_fit(self):
        case, _ = recorded_example()
        inputs = {'schema': timing.fit.INPUT_SCHEMA, 'cases': [case]}
        for group, split, source in (('heldout_source_7', 'validation', 7),
                                     ('failed_trial_evaluation_only', 'evaluation_only', 0)):
            wrong = copy.deepcopy(inputs)
            wrong['cases'][0].update(group=group, split=split, source_index=source)
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, 'only original training sources'):
                timing.cases_index(wrong)


if __name__ == '__main__':
    unittest.main()
