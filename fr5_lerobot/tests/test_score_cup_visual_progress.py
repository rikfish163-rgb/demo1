"""Offline geometry and observation eligibility for archived cup progress."""

import copy
import math
import unittest

import numpy as np

from fr5_lerobot.diagnostics.score_cup_visual_progress import (
    RIM_REFERENCE_MM, jaw_tip, score, score_draw,
)


def example():
    state = [.2, .08, .13, 0., 0., 0., 1., 1.]
    request = {'cup_relative_state': state, 'cup_timing': {'phase': 'approaching'},
               'cup_tracking': {'rim_minus_jaw_mm': (RIM_REFERENCE_MM + [10., 0., 0.]).tolist(),
                                'native_contact': 0}}
    actions = np.tile(state[:7], (50, 1))
    actions[:3, 0] += [.001, .002, .003]
    return request, actions.tolist()


class CupVisualProgressTests(unittest.TestCase):
    def test_translation_units_sign_and_absolute_prefix_points(self):
        request, actions = example()
        result = score_draw(request, actions)
        self.assertEqual(result['assessment'], 'assessed')
        self.assertAlmostEqual(result['before_error_mm'], 10.)
        np.testing.assert_allclose(result['projected_error_mm'], [9., 8., 7.], atol=1e-10)
        self.assertAlmostEqual(result['prefix3_progress_mm'], 3.)
        for action in actions[:3]:
            action[0] = request['cup_relative_state'][0] - .002
        self.assertAlmostEqual(score_draw(request, actions)['prefix3_progress_mm'], -2.)

    def test_first_close_keeps_only_preclose_projection_even_after_reopening(self):
        request, actions = example()
        original = score_draw(request, actions)
        self.assertIsNone(original['first_close_tick'])
        self.assertEqual(original['assessed_prefix_ticks'], 3)
        for first_close in (1, 2, 3, 4, 50):
            with self.subTest(first_close=first_close):
                closing = copy.deepcopy(actions)
                closing[first_close - 1][6] = .76
                result = score_draw(request, closing)
                assessed = min(first_close - 1, 3)
                self.assertEqual(result['first_close_tick'], first_close)
                self.assertEqual(result['assessed_prefix_ticks'], assessed)
                self.assertEqual(result['assessment'], 'assessed' if assessed == 3 else 'unassessed')
                self.assertAlmostEqual(result['before_error_mm'], 10.)
                self.assertEqual(result['fixed_pickup_diagnostic'], original['fixed_pickup_diagnostic'])
                for index, tick in enumerate(result['tick_assessments']):
                    self.assertEqual(tick['tick'], index + 1)
                    if index < assessed:
                        self.assertEqual(tick['assessment'], 'assessed')
                        self.assertEqual(tick['unassessed_reasons'], [])
                        np.testing.assert_allclose(result['projected_rim_minus_jaw_mm'][index],
                                                   original['projected_rim_minus_jaw_mm'][index])
                        self.assertAlmostEqual(result['projected_error_mm'][index], 9. - index)
                    else:
                        self.assertEqual(tick['assessment'], 'unassessed')
                        self.assertEqual(tick['unassessed_reasons'], ['at_or_after_close'])
                        self.assertIsNone(result['projected_rim_minus_jaw_mm'][index])
                        self.assertIsNone(result['projected_error_mm'][index])
                if assessed == 3:
                    self.assertAlmostEqual(result['prefix3_progress_mm'], 3.)
                    self.assertEqual(result['unassessed_reasons'], [])
                else:
                    self.assertIsNone(result['prefix3_progress_mm'])
                    self.assertEqual(result['unassessed_reasons'], ['at_or_after_close'])

    def test_close_threshold_is_strictly_below_88_percent(self):
        request, actions = example()
        for action in actions:
            action[6] = .88
        self.assertEqual(score_draw(request, actions)['assessment'], 'assessed')
        actions[1][6] = float(np.nextafter(.88, 0.))
        result = score_draw(request, actions)
        self.assertEqual(result['first_close_tick'], 2)
        self.assertEqual(result['assessed_prefix_ticks'], 1)
        self.assertIsNone(result['prefix3_progress_mm'])

    def test_partial_prefixes_keep_tick_coverage_without_complete_prefix3_progress(self):
        request, actions = example()
        closed = {}
        for tick in (1, 2, 3):
            closed[tick] = copy.deepcopy(actions)
            closed[tick][tick - 1][6] = .76
        missing = copy.deepcopy(request)
        missing['cup_tracking']['rim_minus_jaw_mm'] = None
        predictions = []
        for index, (observation, proposals, selected) in enumerate([
                (request, [closed[2], actions], 1), (request, [closed[3]], 0),
                (request, [closed[1]], 0), (missing, [actions], 0)]):
            predictions.append({'case_id': f'chunk-{index:05d}', 'seed': 1, 'request': observation,
                                'reply': {'wire_action_representation': 'cup-relative-tcp-v1',
                                          'actions': proposals[selected],
                                          'cup_proposal_selection': {
                                              'draws': len(proposals), 'selected_index': selected,
                                              'proposals': [{'actions': p} for p in proposals]}}})
        summary = score({'predictions': predictions})['summary']
        self.assertEqual(summary['eligible'], 1)
        self.assertEqual(summary['unassessed'], 3)
        self.assertEqual(summary['partial_prefix3'], 1)
        self.assertEqual(summary['unassessed_reasons'], {'at_or_after_close': 2, 'missing_rim': 1})
        for label, counts, partial in (('selected', [2, 2, 1], 1), ('rawfirst', [2, 1, 0], 2)):
            branch = summary[label]
            self.assertEqual(branch['partial_prefix3'], partial)
            self.assertEqual(branch['prefix3']['draws'], 1 if label == 'selected' else 0)
            self.assertEqual(branch['prefix3']['positive'], 1 if label == 'selected' else 0)
            self.assertEqual(branch['fixed_pickup_diagnostic']['draws'], 4)
            self.assertEqual(sum(branch['paired_direction_on_eligible'].values()),
                             1 if label == 'selected' else 0)
            for index, tick in enumerate(branch['projected_error_by_tick']):
                self.assertEqual(tick['tick'], index + 1)
                self.assertEqual(tick['draws'], 4)
                self.assertEqual(tick['assessed'], counts[index])
                self.assertEqual(tick['unassessed'], 4 - counts[index])
                self.assertEqual(tick['unassessed_reasons'],
                                 {'at_or_after_close': 3 - counts[index], 'missing_rim': 1})
                self.assertEqual(tick['progress']['draws'], counts[index])
                if counts[index]:
                    self.assertAlmostEqual(tick['error']['mean_mm'], 9. - index)
                    self.assertAlmostEqual(tick['progress']['progress']['mean_mm'], index + 1.)
                else:
                    self.assertIsNone(tick['error'])
                    self.assertIsNone(tick['progress']['progress'])

    def test_base_axes_remain_base_axes_at_a_rotated_anchor(self):
        request, actions = example()
        request['cup_relative_state'][5] = math.pi / 2
        for action in actions:
            action[5] = math.pi / 2
        np.testing.assert_allclose(score_draw(request, actions)['projected_error_mm'], [9., 8., 7.], atol=1e-10)

    def test_cup_origin_cancels(self):
        request, actions = example()
        prior = score_draw(request, actions)
        shift = np.array([1.23, -2.35, 4.56])
        request['cup_relative_state'][:3] = (np.array(request['cup_relative_state'][:3]) + shift).tolist()
        for action in actions:
            action[:3] = (np.array(action[:3]) + shift).tolist()
        np.testing.assert_allclose(score_draw(request, actions)['projected_rim_minus_jaw_mm'],
                                   prior['projected_rim_minus_jaw_mm'], atol=1e-9)

    def test_tip_offset_radians_and_rotation_order(self):
        cases = [([0., 0., 0.], [5.208, 8.072, 162.]),
                 ([math.pi / 2, 0., 0.], [5.208, -162., 8.072]),
                 ([0., math.pi / 2, 0.], [162., 8.072, -5.208]),
                 ([0., 0., math.pi / 2], [-8.072, 5.208, 162.]),
                 ([0., math.pi / 2, math.pi / 2], [-8.072, 162., -5.208])]
        for angles, expected in cases:
            with self.subTest(angles=angles):
                pose = np.array([.1, .2, .3, *angles])
                np.testing.assert_allclose(jaw_tip(pose), np.array(expected) + [100., 200., 300.], atol=1e-10)

    def test_rotation_moves_the_tip_with_a_stationary_tcp(self):
        request, actions = example()
        for action in actions[:3]:
            action[:3] = request['cup_relative_state'][:3]
            action[5] = math.pi / 2
        expected = np.array(request['cup_tracking']['rim_minus_jaw_mm']) + [13.28, 2.864, 0.]
        result = score_draw(request, actions)
        np.testing.assert_allclose(result['projected_rim_minus_jaw_mm'], np.tile(expected, (3, 1)), atol=1e-10)
        self.assertLess(result['prefix3_progress_mm'], 0.)

    def test_angle_wrap_keeps_the_same_tip(self):
        request, actions = example()
        request['cup_relative_state'][3] = math.pi
        for action in actions[:3]:
            action[:3] = request['cup_relative_state'][:3]
            action[3] = -math.pi
        self.assertAlmostEqual(score_draw(request, actions)['prefix3_progress_mm'], 0., places=10)

    def test_missing_invalid_contact_and_other_phases_stay_unassessed(self):
        original, actions = example()
        changes = [('cup_tracking', 'rim_minus_jaw_mm', None, 'missing_rim'),
                   ('cup_tracking', 'rim_minus_jaw_mm', [0., float('nan'), 0.], 'invalid_rim'),
                   ('cup_tracking', 'rim_minus_jaw_mm', [0., 0.], 'invalid_rim'),
                   ('cup_tracking', 'native_contact', 1, 'contact'),
                   ('cup_tracking', 'native_contact', None, 'missing_contact'),
                   ('cup_tracking', 'native_contact', 2, 'invalid_contact'),
                   ('cup_timing', 'phase', 'returning', 'ineligible_phase')]
        for section, key, value, reason in changes:
            with self.subTest(reason=reason, value=value):
                request = copy.deepcopy(original)
                request[section][key] = value
                result = score_draw(request, actions)
                self.assertEqual(result['assessment'], 'unassessed')
                self.assertIn(reason, result['unassessed_reasons'])
                self.assertEqual(result['assessed_prefix_ticks'], 0)
                for tick in result['tick_assessments']:
                    self.assertEqual(tick['assessment'], 'unassessed')
                    self.assertIn(reason, tick['unassessed_reasons'])
                for field in ('before_error_mm', 'projected_error_mm', 'projected_rim_minus_jaw_mm',
                              'prefix3_progress_mm'):
                    self.assertIsNone(result[field])
        for index, value in ((7, 0.), (6, -1.), (3, float('nan'))):
            request = copy.deepcopy(original)
            request['cup_relative_state'][index] = value
            result = score_draw(request, actions)
            self.assertEqual(result['assessment'], 'unassessed')
            self.assertEqual(result['assessed_prefix_ticks'], 0)
            self.assertTrue(all(t['assessment'] == 'unassessed' for t in result['tick_assessments']))
        request = copy.deepcopy(original)
        request['cup_timing']['phase'] = 'at_pickup'
        self.assertEqual(score_draw(request, actions)['assessment'], 'assessed')

    def test_complete_wire_actions_and_selection_provenance(self):
        request, toward = example()
        away = copy.deepcopy(toward)
        for action in away[:3]:
            action[0] = request['cup_relative_state'][0] - .002
        saved = {'predictions': [{'case_id': 'chunk-00001', 'seed': 1, 'request': request,
                 'reply': {'wire_action_representation': 'cup-relative-tcp-v1', 'actions': toward,
                           'cup_proposal_selection': {'draws': 3, 'selected_index': 1, 'matched': True,
                                                     'proposals': [{'actions': away}, {'actions': toward},
                                                                   {'actions': away}]}}, 'draw_count': 3}]}
        unchanged = copy.deepcopy(saved)
        result = score(saved)
        self.assertEqual(saved, unchanged)
        self.assertEqual(result['summary']['proposal_draws'], 3)
        self.assertEqual(result['summary']['selected']['prefix3']['positive'], 1)
        self.assertEqual(result['summary']['rawfirst']['prefix3']['negative'], 1)
        self.assertEqual(result['summary']['partial_prefix3'], 0)
        for label, errors in (('selected', [9., 8., 7.]), ('rawfirst', [12., 12., 12.])):
            branch = result['summary'][label]
            self.assertEqual(branch['partial_prefix3'], 0)
            for tick, error in zip(branch['projected_error_by_tick'], errors):
                self.assertEqual(tick['assessed'], 1)
                self.assertEqual(tick['unassessed'], 0)
                self.assertEqual(tick['unassessed_reasons'], {})
                self.assertAlmostEqual(tick['error']['mean_mm'], error)
        self.assertEqual(len(result['rows'][0]['draws']), 3)
        saved['predictions'][0]['request']['cup_tracking']['rim_minus_jaw_mm'] = None
        result = score(saved)
        self.assertEqual(result['summary']['missing'], 1)
        self.assertEqual(result['summary']['eligible'], 0)
        self.assertEqual(result['summary']['selected']['prefix3']['draws'], 0)
        self.assertEqual(result['summary']['selected']['fixed_pickup_diagnostic']['draws'], 1)
        for truncated in (toward[:3], [a[:6] for a in toward]):
            with self.assertRaisesRegex(ValueError, '50 x 7'):
                score_draw(request, truncated)
        invalid_tail = copy.deepcopy(toward)
        invalid_tail[49][5] = float('nan')
        with self.assertRaisesRegex(ValueError, '50 x 7'):
            score_draw(request, invalid_tail)
        saved = copy.deepcopy(unchanged)
        saved['predictions'][0]['reply']['actions'] = away
        with self.assertRaisesRegex(ValueError, 'proposal trace'):
            score(saved)
        saved = copy.deepcopy(unchanged)
        saved['predictions'][0]['reply']['wire_action_representation'] = 'cup-relative-tcp-tick-delta-v1'
        with self.assertRaisesRegex(ValueError, 'absolute'):
            score(saved)


if __name__ == '__main__':
    unittest.main()
