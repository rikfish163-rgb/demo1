import unittest

from fr5_lerobot.diagnostics import score_cup_boundary_replay as score


class CupBoundaryMetricsTest(unittest.TestCase):
    def test_gap_uses_measured_table_normal_and_contact_frame(self):
        case = {'initial_cup': {'base_xyz_mm': [0., 0., 0.], 'table_normal': [.6, 0., .8]},
                'event_times': {'close_reference_tcp_xyz_mm': [0., 0., 0.],
                                'contact_confirmed_source_frame': 7}}
        result = score.geometry(case, [.003, 0., .004], {7: [.6, 0., .8]})
        self.assertAlmostEqual(result['relative_to_close_command_tcp']['normal_gap_mm'], 5.)
        self.assertAlmostEqual(result['relative_to_confirmed_contact_tcp']['normal_gap_mm'], 4.)
        self.assertAlmostEqual(result['relative_to_close_command_tcp']['xy_distance_mm'], 3.)
        self.assertIsNone(score.geometry(case, [.003, 0., .004], {})['relative_to_confirmed_contact_tcp'])

    def test_effective_changes_respect_feedback_and_one_point_deadband(self):
        state = [0.] * 6 + [1., 1.]
        actions = [[0.] * 6 + [v] for v in (.99, .98, .977, .76, 1., 1.01)]
        self.assertEqual(score.effective_changes(state, actions), [
            {'tick': 2, 'direction': 'close', 'previous_percent': 100, 'target_percent': 98},
            {'tick': 4, 'direction': 'close', 'previous_percent': 98, 'target_percent': 76},
            {'tick': 5, 'direction': 'open', 'previous_percent': 76, 'target_percent': 100},
        ])
        state[7] = 0.
        self.assertIsNone(score.effective_changes(state, actions))

    def test_descent_checks_every_point_before_the_original_deadline(self):
        points = [{'tick': i, 'nominal_time_monotonic': t, 'predicted_normal_delta_mm': d,
                   'recorded_normal_delta_mm': 0.}
                  for i, (t, d) in enumerate(((101., -1.), (102., -4.), (103., -.2),
                                              (105., -8.)), start=1)]
        row = {'case_id': 'source0-cycle1-hover_middle-1', 'source_index': 0,
               'cycle': 1, 'kind': 'hover_middle', 'seed': 1, 'source_time': 100.,
               'event_times': {'hold_start_monotonic': 100.}, 'metrics': {'first_10': points}}
        value = score.fit_descent({('case', 1): row})['maximum_down_from_observation']
        self.assertEqual(value['maximum_down_from_observation_mm'], 4.)
        self.assertEqual(value['lowest_tick'], 2)
        self.assertEqual(value['points_before_deadline'], 3)

    def test_event_time_is_signed_against_original_command(self):
        case = {'case_id': 'example',
                'initial_cup': {'base_xyz_mm': [0., 0., 0.], 'table_normal': [0., 0., 1.]},
                'event_times': {'close_reference_tcp_xyz_mm': [0., 0., 0.],
                                'contact_confirmed_source_frame': 7,
                                'release_issued_monotonic': 10.5},
                'targets': [{'target_time_monotonic': 10. + (i + 1) / 6.,
                             'cup_relative_action': [0.] * 6 + [.76]} for i in range(10)]}
        draw = {'request': {'source_time': 10.},
                'reply': {'actions': [[0., 0., .003, 0., 0., 0., 1.]] * 10}}
        change = {'tick': 1, 'direction': 'open', 'previous_percent': 76, 'target_percent': 100}
        result = score.event_detail(case, draw, change, {7: [0., 0., 0.]})
        self.assertAlmostEqual(result['seconds_after_recorded_command'], -1. / 3.)
        self.assertEqual(result['relative_to_confirmed_contact_tcp']['normal_gap_mm'], 3.)

    def test_pair_rejects_different_observations(self):
        request = {'source_time': 100., 'cup_relative_state': [0.] * 8}
        case = {'case_id': 'source0', 'group': 'heldout_source_7', 'kind': 'hover_middle',
                'initial_cup': {}, 'event_times': {}, 'targets': [],
                'predictions': [{'seed': 1, 'candidate': {'request': request}}]}
        held = {'schema': 'fr5-cup-tracking-replay-comparison-v1', 'seeds': [1],
                'task_override': '5秒', 'cases': [case]}
        times = {'requested_hold_seconds': 5., 'hold_start_monotonic': 100.,
                 'five_second_deadline_monotonic': 105.}
        detail = {'case_id': 'source0', 'source_index': 0, 'cycle': 1, 'kind': 'hover_middle',
                  'seed': 1, 'observer': 'aligned', 'source_time': 100., 'task': '5秒',
                  'event_times': times, 'cup_timing': {}, 'metrics': {'first_10': []}}
        fit = {'schema': 'fr5-five-second-train-fit-report-v1', 'input': 'same-input.json', 'details': [detail]}
        score.validate_pair(held, 'candidate', held, 'candidate', fit, 'aligned', fit, 'aligned')
        changed_case = {**case, 'predictions': [{'seed': 1, 'candidate': {
            'request': {**request, 'source_time': 100.1}}}]}
        with self.assertRaisesRegex(ValueError, 'model requests differ'):
            score.validate_pair(held, 'candidate', {**held, 'cases': [changed_case]}, 'candidate',
                                fit, 'aligned', fit, 'aligned')


if __name__ == '__main__':
    unittest.main()
