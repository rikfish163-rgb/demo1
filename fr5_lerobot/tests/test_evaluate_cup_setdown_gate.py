import copy
import unittest

from fr5_lerobot.diagnostics import evaluate_cup_setdown_gate as gate


def action(z_mm=100., jaw=.76, x_mm=0.):
    return [x_mm / 1000., 0., z_mm / 1000., 0., 0., 0., jaw]


def example(kind='return_late', z_mm=105., jaw=.76, release_time=101.4):
    request = {'source_time': 100., 'cup_relative_state': [*action(z_mm, jaw), 1.]}
    case = {'case_id': 'example', 'group': 'heldout_source_7', 'kind': kind,
            'cycle': 1, 'request': request,
            'initial_cup': {'base_xyz_mm': [0., 0., 0.], 'table_normal': [0., 0., 1.]},
            'event_times': {'close_reference_tcp_xyz_mm': [0., 0., 100.],
                            'close_issued_monotonic': 100.1,
                            'release_issued_monotonic': release_time},
            'timing_aligned': {'state': {'visual': 'table_height'}},
            'targets_camera_aligned': [{'target_time_monotonic': 100. + i / 6,
                                        'cup_relative_action': action(100., .76)} for i in range(1, 11)]}
    frame = {'gripper_feedback_valid': True,
             'robot': {'gripper_motiondone': 2, 'tcp': [0., 0., z_mm], 'tcp_speed': [0., 0.]}}
    support = {'height_projection_mm': 100.}
    return case, frame, support


def draw(case, actions):
    return {'seed': 20261105, 'request': case['request'], 'reply': {'actions': actions}}


class SetdownGateTest(unittest.TestCase):
    def test_nominal_close_does_not_pass_when_complete_proposals_are_exhausted(self):
        case, frame, support = example(kind='close_at', z_mm=100., jaw=1.)
        row = draw(case, [action(100., .80)] * 10)
        row['reply']['cup_proposal_selection'] = {'matched': False, 'draws': 3}
        result = gate.score_draw(case, row, frame, support, 3)
        self.assertEqual(1, result['observations']['first_close_tick'])
        self.assertEqual([{'code': 'no_feasible_complete_proposal', 'draws': 3}],
                         result['violations'])
        self.assertEqual(1, gate.summarize([result])['close_at']['failure_draws'])

    def test_waiting_above_support_is_not_missing_open(self):
        case, frame, support = example()
        result = gate.score_draw(case, draw(case, [action(105.)] * 10), frame, support, 10)
        self.assertEqual([], result['violations'])
        self.assertIsNone(result['observations']['first_predicted_ready_proxy_tick'])
        self.assertIsNone(result['observations']['first_full_open_tick'])

    def test_open_checks_support_independent_of_teacher_label(self):
        case, frame, support = example()
        case['targets_camera_aligned'][0]['cup_relative_action'][6] = 1.
        result = gate.score_draw(case, draw(case, [action(105., 1.)] * 10), frame, support, 3)
        self.assertEqual('open_above_support_proxy', result['violations'][0]['code'])
        case, frame, support = example(z_mm=100.)
        result = gate.score_draw(case, draw(case, [action(100., 1.)] * 10), frame, support, 3)
        self.assertEqual([], result['violations'])
        self.assertTrue(result['observations']['observed_ready_proxy'])
        self.assertIsNone(result['observations']['teacher_open_label_tick'])

    def test_partial_open_above_support_is_not_ignored(self):
        case, frame, support = example()
        result = gate.score_draw(case, draw(case, [action(105., .79)] * 10), frame, support, 3)
        self.assertEqual('open_above_support_proxy', result['violations'][0]['code'])

    def test_support_reference_uses_raw_contact_and_stable_height(self):
        cases, frames, journal = {}, [], []
        for cycle in (1, 2):
            release = 100. * cycle + 1.
            contact = 10 * cycle
            events = {'close_reference_tcp_xyz_mm': [0., 0., 100.],
                      'close_issued_monotonic': release - 6., 'return_issued_monotonic': release - 4.,
                      'release_issued_monotonic': release, 'contact_confirmed_source_frame': contact,
                      'contact_confirmed_monotonic': release - 5.}
            case = {'case_id': str(cycle), 'group': 'heldout_source_7', 'kind': 'release_at',
                    'cycle': cycle, 'event_times': events, 'initial_cup': {'table_normal': [0., 0., 1.]}}
            cases[str(cycle)] = case
            for index, stamp, height in [(contact, release - 5., 100.),
                                         (contact + 1, release - .4, 100.),
                                         (contact + 2, release - .2, 100.1),
                                         (contact + 3, release - .05, 100.)]:
                frames.append({'frame_index': index, 'camera': {'source_timestamp_monotonic': stamp},
                               'gripper_feedback_valid': True,
                               'robot': {'gripper_motiondone': 2, 'tcp': [0., 0., height],
                                         'tcp_speed': [.2, 0.]}})
            for phase, t, jaw in [('close', release - 6., 76.), ('return', release - 4., 76.),
                                  ('release', release, 100.)]:
                journal.append({'teacher': {'cycle': cycle, 'phase': phase}, 'executed': True,
                                'issued_monotonic': t, 'target': [0.] * 6 + [jaw]})
        references, _ = gate.raw_support_references(cases, frames, journal)
        self.assertAlmostEqual(100., references[1]['height_projection_mm'])
        self.assertAlmostEqual(.1, references[1]['pre_release_height_spread_mm'])
        self.assertGreater(references[1]['seconds_ready_proxy_before_command'], 0.)
        frames[0]['gripper_feedback_valid'] = False
        with self.assertRaises(ValueError):
            gate.raw_support_references(cases, frames, journal)

    def test_supported_release_requires_timely_full_open(self):
        case, frame, support = example(kind='release_at', z_mm=100., release_time=100.1)
        late = [action(100., .76), action(100., .76), action(100., 1.)] + [action(100., 1.)] * 7
        result = gate.score_draw(case, draw(case, late), frame, support, 3)
        self.assertIn('late_supported_open', [v['code'] for v in result['violations']])
        timely = [action(100., .76), action(100., 1.)] + [action(100., 1.)] * 8
        result = gate.score_draw(case, draw(case, timely), frame, support, 3)
        self.assertEqual([], result['violations'])

    def test_delayed_normal_close_and_far_close_are_separate(self):
        case, frame, support = example(kind='close_at', z_mm=100., jaw=1.)
        frame['robot']['gripper_motiondone'] = 0
        late = [action(100., 1.)] * 3 + [action(100., .76)] * 7
        result = gate.score_draw(case, draw(case, late), frame, support, 3)
        self.assertEqual(['late_normal_close'], [v['code'] for v in result['violations']])
        self.assertIsNone(result['observations']['first_predicted_ready_proxy_tick'])
        far = [action(100., .76, x_mm=26.)] * 10
        result = gate.score_draw(case, draw(case, far), frame, support, 3)
        self.assertEqual(['far_close'], [v['code'] for v in result['violations']])

    def test_three_step_gate_does_not_hide_ten_step_unsafe_open(self):
        case, frame, support = example()
        actions = [action(105.)] * 4 + [action(105., 1.)] * 6
        short = gate.score_draw(case, draw(case, actions), frame, support, 3)
        full = gate.score_draw(case, draw(case, actions), frame, support, 10)
        self.assertEqual([], short['violations'])
        self.assertEqual('open_above_support_proxy', full['violations'][0]['code'])

    def test_heldout_five_second_counterfactual_is_separate_from_original_prompt(self):
        case, frame, support = example(kind='hover_middle', z_mm=150.)
        case['event_times']['hold_start_monotonic'] = 99.
        case['request']['task'] = '悬停5秒'
        actions = [action(150.), action(150.), action(147.)] + [action(147.)] * 7
        result = gate.score_draw(case, draw(case, actions), frame, support, 3)
        self.assertIn('early_counterfactual_five_second_descent', [v['code'] for v in result['violations']])
        case['request']['task'] = '悬停4秒'
        result = gate.score_draw(case, draw(case, actions), frame, support, 3)
        self.assertNotIn('early_counterfactual_five_second_descent', [v['code'] for v in result['violations']])

    def test_old_and_setdown_observers_can_use_same_updated_input(self):
        case, _, _ = example()
        case['request'].update(task='original', observation_context_profile='old')
        case['tracking'] = {'rim_minus_jaw_mm': None, 'native_contact': 1}
        case['timing_aligned']['state'].update(phase='returning')
        case['timing_setdown'] = {'state': {**case['timing_aligned']['state'], 'setdown_age_s': .5}}
        for profile, timing in [('state-history-physical-cup-aligned-v1', 'timing_aligned'),
                                ('state-history-physical-cup-setdown-v1', 'timing_setdown'),
                                ('state-history-physical-cup-setdown-clean-v1', 'timing_setdown'),
                                ('state-history-physical-cup-setdown-fixed-v1', 'timing_setdown')]:
            req = {**case['request'], 'task': 'five', 'observation_context_profile': profile,
                   'cup_tracking': case['tracking'], 'cup_timing': case[timing]['state']}
            gate.validate_request(case, req, 'five', profile)
            bad = copy.deepcopy(req)
            bad['cup_relative_state'][0] += .01
            with self.assertRaises(ValueError):
                gate.validate_request(case, bad, 'five', profile)
            bad = copy.deepcopy(req)
            bad['cup_timing']['phase'] = 'hovering'
            with self.assertRaises(ValueError):
                gate.validate_request(case, bad, 'five', profile)

    def test_five_second_deadline_and_prefix_are_independent(self):
        cycles = [(0, 1), (3, 1), (3, 2), (4, 1), (5, 1)]
        details = []
        for i in range(100):
            source, cycle = cycles[i % len(cycles)]
            for seed in gate.BASE_SEEDS:
                points = [{'tick': tick, 'nominal_time_monotonic': 104. + tick / 6,
                           'predicted_normal_delta_mm': -2.1 if i == 0 and seed == gate.BASE_SEEDS[0]
                           and tick == 4 else -3. if i == 1 and tick == 7 else 0.}
                          for tick in range(1, 11)]
                details.append({'case_id': f'case-{i}', 'source_index': source, 'cycle': cycle,
                                'kind': 'hover_middle', 'observer': 'setdown', 'seed': seed,
                                'source_time': 104.,
                                'event_times': {'requested_hold_seconds': 5., 'hold_start_monotonic': 100.,
                                                'five_second_deadline_monotonic': 105.},
                                'metrics': {'first_10': points}})
        report = {'schema': 'fr5-five-second-train-fit-report-v1', 'checkpoints': {'new': 'model'},
                  'seeds': list(gate.BASE_SEEDS), 'details': details}
        self.assertEqual(0, gate.fit_scores(report, 'model', 'setdown', 3)['failure_count'])
        self.assertEqual(1, gate.fit_scores(report, 'model', 'setdown', 10)['failure_count'])
        report['details'].pop()
        with self.assertRaises(ValueError):
            gate.fit_scores(report, 'model', 'setdown', 10)


if __name__ == '__main__':
    unittest.main()
