"""Preserve rejected cup draws and keep them out of controller pass counts."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'diagnostics'))
from evaluate_cup_timing_policy import model_request
from score_cup_proposal_trace import score


def example():
    state = [.202801, .086487, .136571, 1.61719, -.02211, -1.17431, 1., 1.]
    request = {'id': 'real-approach', 'task': '抓杯子，悬停5秒再放下',
               'cup_relative_state': state, 'observation_history': [None] * 6}
    case = {'case_id': 'trial-chunk-62', 'group': 'failed_trial_evaluation_only',
            'kind': 'failed_real_approach', 'request': request,
            'initial_cup': {'table_normal': [0., 0., 1.]},
            'tracking': {'native_contact': 0, 'rim_minus_jaw_mm': [-53.59, -19.03, -27.23]},
            'timing_setdown': {'state': {'phase': 'approaching', 'support_gap_mm': None,
                                        'setdown_age_s': None, 'hover_seconds': None, 'visual': 'table_height'}}}
    request = model_request(case, 'candidate', 'native', profile='state-history-physical-cup-setdown-v1')
    safe = [[*state[:6], 1.] for _ in range(50)]
    unsafe = copy.deepcopy(safe)
    unsafe[0][6] = .73998
    trace = {'draws': 2, 'selected_index': 1, 'matched': True,
             'proposals': [{'actions': unsafe, 'inference_seconds': .35, 'matched': False,
                            'sampling': {'cuda_rng_state': [1, 2]}, 'runtime_source': '/source/serve.py'},
                           {'actions': safe, 'inference_seconds': .35, 'matched': True,
                            'sampling': {'cuda_rng_state': [3, 4]}, 'runtime_source': '/source/serve.py'}]}
    reply = {'actions': safe, 'inference_seconds': .70, 'cup_proposal_selection': trace,
             'sampling': {'cuda_rng_state': [3, 4]}, 'runtime_source': '/source/serve.py'}
    row = {'case_id': case['case_id'], 'group': case['group'], 'kind': case['kind'],
           'seed': 7, 'request': request, 'reply': reply, 'draw_count': 2,
           'predict_elapsed_seconds': .72}
    predictions = {'checkpoint': '/checkpoint', 'observation_context_profile': 'state-history-physical-cup-setdown-v1',
                   'task_override': None, 'seeds': [7], 'predictions': [row]}
    return {'cases': [case]}, predictions


class CupProposalTraceTests(unittest.TestCase):
    def test_first_failed_draw_stays_failed_after_a_complete_replacement(self):
        inputs, predictions = example()
        report, raw = score(inputs, predictions)
        summary = report['summary']['failed_trial_evaluation_only']
        self.assertEqual(summary['first_geometry_rejected'], 1)
        self.assertEqual(summary['first_close_then_reopen'], 1)
        self.assertEqual(summary['selected_geometry_rejected'], 0)
        self.assertFalse(raw['predictions'][0]['selection_matched'])
        self.assertEqual(raw['predictions'][0]['reply']['actions'][0][6], .73998)
        self.assertEqual(raw['predictions'][0]['reply']['sampling']['cuda_rng_state'], [1, 2])
        self.assertEqual(predictions['predictions'][0]['reply']['actions'][0][6], 1.)
        self.assertEqual(predictions['predictions'][0]['reply']['sampling']['cuda_rng_state'], [3, 4])

    def test_three_unmatched_draws_cannot_count_as_a_controller_pass(self):
        inputs, predictions = example()
        row = predictions['predictions'][0]
        trace = row['reply']['cup_proposal_selection']
        safe = row['reply']['actions']
        trace.update(draws=3, selected_index=0, matched=False,
                     proposals=[{'actions': safe, 'inference_seconds': .35, 'matched': False}] * 3)
        row['draw_count'] = 3
        report, _ = score(inputs, predictions)
        self.assertEqual(report['summary']['failed_trial_evaluation_only']['exhausted_without_match'], 1)


if __name__ == '__main__':
    unittest.main()
