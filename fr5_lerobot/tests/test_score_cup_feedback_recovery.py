"""Check PGI recovery scoring against recorded jaw targets and complete draws."""

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'diagnostics'))
from evaluate_cup_timing_policy import model_request
from score_cup_feedback_recovery import evaluate, timing_metrics, verify_recovery


def example(*, valid=True):
    state = [.1597, .0713, .0958, 1.62, -.022, -1.174, .99 if valid else 0., float(valid)]
    case = {'case_id': 'release-recovered-3', 'group': 'heldout_source_7',
            'kind': 'feedback_recovered', 'source_frame_index': 3,
            'initial_cup': {'table_normal': [0., 0., 1.], 'base_xyz_mm': [0., 0., 0.]},
            'request': {'id': 'release-recovered-3', 'task': '抓杯，悬停4秒再放下',
                        'source_time': 100., 'cup_relative_state': state,
                        'observation_history': [None] * 6},
            'tracking': {'native_contact': 0, 'rim_minus_jaw_mm': None},
            'timing_setdown': {'state': {'phase': 'at_pickup', 'support_gap_mm': None,
                                        'setdown_age_s': None, 'hover_seconds': None, 'visual': 'missing'}}}
    profile = 'state-history-physical-cup-setdown-fixed-v1'
    request = model_request(case, 'candidate', 'native', profile=profile)
    actions = [[*state[:6], v] for v in (.81, 1.01, *([1.] * 48))]
    reply = {'actions': actions, 'inference_seconds': .35,
             'cup_proposal_selection': {'draws': 1, 'selected_index': 0, 'matched': True,
                                        'proposals': [{'actions': actions, 'inference_seconds': .35,
                                                       'matched': True}]}}
    row = {'case_id': case['case_id'], 'group': case['group'], 'kind': case['kind'],
           'seed': 7, 'request': request, 'reply': reply, 'draw_count': 1,
           'selection_matched': True, 'predict_elapsed_seconds': .36}
    predictions = {'checkpoint': '/checkpoint', 'model': 'candidate', 'timing_mode': 'native',
                   'observation_context_profile': profile, 'task_override': None,
                   'seeds': [7], 'predictions': [row]}
    journal = [{'issued_monotonic': 99., 'executed': True, 'target': [0.] * 6 + [100.]}]
    return {'cases': [case]}, predictions, journal


class CupFeedbackRecoveryTests(unittest.TestCase):
    def test_recovered_reclose_reopen_can_be_matched_and_early(self):
        inputs, predictions, journal = example()
        saved = copy.deepcopy(predictions)
        report = evaluate(inputs, predictions, journal)
        self.assertEqual(report['summary']['total'], 1)
        self.assertEqual(report['summary']['trace_matched'], 1)
        prefix = report['summary']['selected']['prefix_3']
        self.assertEqual(prefix['early_close_draws'], 1)
        self.assertEqual(prefix['early_open_draws'], 0)
        self.assertEqual(prefix['late_open_assessed'], 0)
        self.assertTrue(report['rows'][0]['selected']['geometry']['close_then_reopen_in_three_ticks'])
        self.assertEqual(report['rows'][0]['selected_first_three'], saved['predictions'][0]['reply']['actions'][:3])
        self.assertEqual(predictions, saved)

    def test_invalid_feedback_is_unassessed_without_geometry_or_timing(self):
        inputs, predictions, journal = example(valid=False)
        with patch('score_cup_proposal_trace.geometry') as geometry:
            report = evaluate(inputs, predictions, journal)
        geometry.assert_not_called()
        self.assertEqual(report['summary']['unassessed'], 1)
        self.assertEqual(report['summary']['reported_matched'], 0)
        self.assertIsNone(report['rows'][0]['first'])
        self.assertIsNone(report['rows'][0]['selected'])

    def test_single_draw_without_selector_has_separate_matched_provenance(self):
        inputs, predictions, journal = example()
        predictions['predictions'][0]['reply'].pop('cup_proposal_selection')
        report = evaluate(inputs, predictions, journal)
        self.assertEqual(report['summary']['reported_matched'], 1)
        self.assertEqual(report['summary']['trace_matched'], 0)
        self.assertEqual(report['summary']['single_draw_without_selector'], 1)

    def test_three_unmatched_complete_draws_remain_unmatched(self):
        inputs, predictions, journal = example()
        row = predictions['predictions'][0]
        trace = row['reply']['cup_proposal_selection']
        trace.update(draws=3, matched=False,
                     proposals=[dict(trace['proposals'][0], matched=False) for _ in range(3)])
        row.update(draw_count=3, selection_matched=False)
        report = evaluate(inputs, predictions, journal)
        self.assertEqual(report['summary']['reported_matched'], 0)
        self.assertEqual(report['proposal_trace_summary']['heldout_source_7']['exhausted_without_match'], 1)

    def test_more_than_three_proposals_are_rejected(self):
        inputs, predictions, journal = example()
        row = predictions['predictions'][0]
        trace = row['reply']['cup_proposal_selection']
        trace.update(draws=4, proposals=trace['proposals'] * 4)
        row['draw_count'] = 4
        with self.assertRaisesRegex(ValueError, 'at most three'):
            evaluate(inputs, predictions, journal)

    def test_unselected_proposals_must_also_be_complete_50_by_7(self):
        inputs, predictions, journal = example()
        row = predictions['predictions'][0]
        trace = row['reply']['cup_proposal_selection']
        trace['proposals'].append(dict(trace['proposals'][0], actions=row['reply']['actions'][:49]))
        trace['draws'] = row['draw_count'] = 2
        with self.assertRaisesRegex(ValueError, '50 x 7'):
            evaluate(inputs, predictions, journal)

    def test_missing_seed_predictions_are_rejected(self):
        inputs, predictions, journal = example()
        predictions['seeds'].append(8)
        with self.assertRaisesRegex(ValueError, 'cover the recovery cases'):
            evaluate(inputs, predictions, journal)

    def test_changed_causal_observation_is_rejected(self):
        inputs, predictions, journal = example()
        predictions['predictions'][0]['request']['source_time'] += 1.
        with self.assertRaisesRegex(ValueError, 'saved causal request'):
            evaluate(inputs, predictions, journal)

    def test_effective_changes_use_deadband_and_clamp_without_rewriting_actions(self):
        request = {'source_time': 100., 'cup_relative_state': [0.] * 6 + [1., 1.]}
        actions = [[0.] * 6 + [v] for v in (.99, .98, 1.01)]
        journal = [{'issued_monotonic': 99., 'executed': True, 'target': [0.] * 6 + [100.]}]
        metrics = timing_metrics(request, actions, journal, 3)
        self.assertEqual([e['tick'] for e in metrics['early_close']], [2])
        self.assertEqual(actions[2][6], 1.01)

    def test_early_open_uses_the_original_closed_target(self):
        request = {'source_time': 100., 'cup_relative_state': [0.] * 6 + [.78, 1.]}
        actions = [[0.] * 6 + [1.]] * 3
        journal = [{'issued_monotonic': 99., 'executed': True, 'target': [0.] * 6 + [76.]}]
        metrics = timing_metrics(request, actions, journal, 3)
        self.assertEqual([e['tick'] for e in metrics['early_open']], [1])
        self.assertFalse(metrics['late_open']['assessed'])
        self.assertIsNone(metrics['late_open']['late_or_missing'])

    def test_late_open_requires_a_deadline_inside_the_prefix(self):
        request = {'source_time': 100., 'cup_relative_state': [0.] * 6 + [.78, 1.]}
        actions = [[0.] * 6 + [v] for v in (.76, .76, 1.)]
        journal = [{'issued_monotonic': 99., 'executed': True, 'target': [0.] * 6 + [76.]},
                   {'issued_monotonic': 100. + 1. / 6., 'executed': True, 'target': [0.] * 6 + [100.]}]
        late = timing_metrics(request, actions, journal, 3)['late_open']
        self.assertTrue(late['assessed'])
        self.assertTrue(late['late_or_missing'])
        self.assertEqual(late['deadline_tick'], 2)
        journal[-1]['issued_monotonic'] = 100. + 3. / 6.
        late = timing_metrics(request, actions, journal, 3)['late_open']
        self.assertFalse(late['assessed'])
        self.assertIsNone(late['late_or_missing'])

    def test_recovery_must_use_the_first_valid_frame(self):
        inputs, _, _ = example()
        captures = [{'frame_index': i, 'gripper_feedback_valid': valid,
                     'camera': {'source_timestamp_monotonic': stamp}}
                    for i, valid, stamp in ((1, False, 99.), (2, True, 99.5), (3, True, 100.))]
        recovery = {'observations': [{'case_id': 'release', 'invalid_source_frame': 1,
                                      'first_new_valid_frame': 3}]}
        with self.assertRaisesRegex(ValueError, 'first subsequent valid'):
            verify_recovery(inputs, captures, recovery)


if __name__ == '__main__':
    unittest.main()
