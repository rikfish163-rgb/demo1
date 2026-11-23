"""Check phase-probe adaptation, effective jaw timing and archived pairing evidence."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'diagnostics'))

from evaluate_cup_timing_policy import model_request
import probe_cup_release_phase as probe
import score_cup_release_phase_probe as phase_score


def example():
    cases = []
    for name, opening, contact, source_time in (('release', .99, 0, 100.), ('contact', .78, 1, 200.)):
        state = [.1597, .0713, .0958, 1.62, -.022, -1.174, opening, 1.]
        cases.append({'case_id': name, 'group': 'heldout_source_7', 'kind': 'feedback_recovered',
                      'source_frame_index': len(cases) + 1,
                      'initial_cup': {'table_normal': [0., 0., 1.], 'base_xyz_mm': [0., 0., 0.]},
                      'request': {'id': name, 'task': '抓杯，悬停4秒后放下', 'source_time': source_time,
                                  'state': [1.] * 6 + state[6:], 'cup_relative_state': state,
                                  'observation_history': [None] * 6},
                      'tracking': {'native_contact': contact, 'rim_minus_jaw_mm': None},
                      'timing_setdown': {'state': {'phase': 'grasp_contact' if contact else 'at_pickup',
                                                   'support_gap_mm': 0. if contact else None,
                                                   'hover_seconds': None, 'setdown_age_s': None, 'visual': 'missing'}}})
    original = {'schema': probe.INPUT_SCHEMA, 'timing_ready': True, 'cases': cases}
    corrected = deepcopy(original)
    corrected['cases'][0]['timing_setdown']['state']['phase'] = 'released'
    saved = {'schema': 'fr5-cup-release-phase-probe-v1', 'checkpoint': '/checkpoint', 'seeds': [7],
             'observation_context_profile': probe.PROFILE, 'action_shape': [50, 7], 'prefix_points': 3,
             'max_complete_candidates': 3, 'denoising_steps': 10, 'task_override': None, 'timing_mode': 'native',
             'thresholds': {'early_close_below_fraction': .88, 'full_open_at_least_fraction': .98},
             'robot_connections': 0, 'robot_commands': 0, 'results': [], 'summary': {}}
    for pair in probe.pair_inputs(original, corrected):
        result = {key: pair[key] for key in ('case_id', 'original_case_index', 'corrected_case_index',
                                             'recovery_role', 'phase_changed')}
        result.update({key: pair['case'][key] for key in ('group', 'kind', 'source_frame_index')})
        result.update(original_recorded_task=pair['case']['request']['task'], task_override=None)
        row = {'seed': 7, 'unpaired_draw_counts': {'original': 0, 'corrected': 0}}
        for phase in phase_score.PHASES:
            request = pair[phase + '_request']
            result[phase + '_request'] = request
            actions = [[*request['cup_relative_state'][:6], request['cup_relative_state'][6]] for _ in range(50)]
            traced = request['cup_timing']['phase'] == 'at_pickup'
            if traced:
                actions[0][6], actions[1][6] = .81, 1.01
            first = {'actions': actions, 'inference_seconds': .1,
                     'sampling': {'cuda_rng_state': [7, 0], 'denoising_steps': 10}}
            trace = {'draws': 1, 'selected_index': 0, 'matched': True,
                     'proposals': [dict(deepcopy(first), matched=True)]} if traced else None
            selected = deepcopy(first)
            if trace is not None:
                selected['cup_proposal_selection'] = trace
            row[phase] = {'raw_first_reply': first, 'selected_reply': selected,
                          'candidate_replies': [deepcopy(first)], 'selector_trace': trace,
                          'selector': probe.selector_classification(trace), 'draw_count': 1,
                          'new_draw_selected': False, 'raw_first_selected_actions_equal': True,
                          'predict_elapsed_seconds': .1}
        channels = {}
        for key, width in phase_score.CHANNEL_WIDTHS.items():
            before, after = [0] * width, [0] * width
            if key == probe.CLOCK:
                before[3 if pair['phase_changed'] else 4] = 1.
                after[9 if pair['phase_changed'] else 4] = 1.
            elif pair['phase_changed']:
                after[-1] = 1
            changed = [[0, i] for i, (a, b) in enumerate(zip(before, after)) if a != b]
            channels[key] = {'original': [before], 'corrected': [after], 'original_shape': [1, width],
                             'corrected_shape': [1, width], 'elementwise_equal': not changed,
                             'changed_indices': changed}
        row['paired_draw_audits'] = [{'draw_index': 0, 'pre_sampling_cuda_rng_equal': True,
                                     'preprocess': {'channels': channels, 'fixed_cup_action_clock': True,
                                                    'other_tensor_keys': ['observation.state', 'observation.images.base_0_rgb'],
                                                    'other_tensors_elementwise_equal': True}}]
        row['differences'] = {view: probe.action_difference(row['original'][view + '_reply']['actions'],
                                                           row['corrected'][view + '_reply']['actions'])
                              for view in ('raw_first', 'selected')}
        result['seeds'] = [row]
        saved['results'].append(result)
    journal = [{'issued_monotonic': t, 'executed': True, 'target': [0.] * 6 + [opening]}
               for t, opening in ((99., 100.), (199., 76.))]
    return original, corrected, saved, journal


class CupReleasePhaseScoreTests(unittest.TestCase):
    def test_not_applied_is_unassessed_for_selection_even_when_geometry_passes(self):
        original, corrected, saved, journal = example()
        archived = deepcopy((original, corrected, saved))
        report = phase_score.score_probe(original, corrected, saved, journal)
        before, after = (report['conditions'][phase] for phase in phase_score.PHASES)
        self.assertEqual(before['summary']['trace_matched'], 1)
        self.assertEqual(after['summary']['reported_matched'], 0)
        self.assertEqual(after['summary']['trace_matched'], 0)
        self.assertEqual(after['summary']['selector_status'], {'not_applied': 2})
        self.assertEqual(after['geometry_summary']['selected']['rejected'], 0)
        self.assertTrue(all(row['reported_matched'] is None for row in after['rows']))
        self.assertNotIn('proposal_trace_summary', after)
        self.assertEqual(after['summary']['selected']['prefix_3']['late_open_status'],
                         {'already_open_at_observation': 1, 'release_deadline_outside_prefix': 1})
        self.assertEqual((original, corrected, saved), archived)

    def test_integer_deadband_detects_reclose_above_float_close_threshold(self):
        original, corrected, saved, journal = example()
        prediction = saved['results'][0]['seeds'][0]['corrected']
        for reply in (prediction['raw_first_reply'], prediction['selected_reply'], *prediction['candidate_replies']):
            for tick, opening in enumerate((.979, .974, 1.01)):
                reply['actions'][tick][6] = opening
        report = phase_score.score_condition(corrected, saved, 'corrected', journal)
        events = report['rows'][0]['selected']['timing']['prefix_3']['early_close']
        self.assertEqual([(event['tick'], event['previous_percent'], event['target_percent']) for event in events],
                         [(2, 99, 97)])
        self.assertTrue(all(action[6] >= .88 for action in prediction['selected_reply']['actions'][:3]))
        self.assertEqual(report['summary']['selected']['prefix_3']['early_close_draws'], 1)
        self.assertEqual(prediction['selected_reply']['actions'][2][6], 1.01)

    def test_contact_control_tail_difference_is_not_hidden_by_prefix_equality(self):
        original, corrected, saved, journal = example()
        prediction = saved['results'][1]['seeds'][0]['corrected']
        for reply in (prediction['raw_first_reply'], prediction['selected_reply'], *prediction['candidate_replies']):
            reply['actions'][49][5] += .01
        report = phase_score.score_probe(original, corrected, saved, journal)['paired_evidence']
        self.assertFalse(report['contact_controls']['all_50x7_actions_elementwise_equal'])
        self.assertFalse(report['consistent_with_saved_probe_evidence'])
        self.assertTrue(any(issue.get('first_changed_index') == [49, 5] for issue in report['issues']))

    def test_rng_and_tensor_values_override_stale_equality_flags(self):
        _, _, saved, _ = example()
        row = saved['results'][1]['seeds'][0]
        row['corrected']['candidate_replies'][0]['sampling']['cuda_rng_state'] = [8, 0]
        row['paired_draw_audits'][0]['preprocess']['channels'][probe.TOKENS]['corrected'][0][20] = 1
        report = phase_score.audit_pairs(saved)
        self.assertFalse(report['consistent_with_saved_probe_evidence'])
        self.assertEqual(report['pre_sampling_cuda_rng_equal'], 1)
        self.assertTrue(any(issue['field'] == 'pre_sampling_cuda_rng_equal' for issue in report['issues']))
        self.assertEqual(report['channels'][probe.TOKENS]['saved_evidence_consistent_draws'], 1)

    def test_unknown_feedback_never_receives_geometry_or_timing_scores(self):
        original, _, saved, journal = example()
        for case, result in zip(original['cases'], saved['results']):
            case['request']['cup_relative_state'][7] = 0.
            case['request']['state'][7] = 0.
            result['original_request'] = model_request(case, 'candidate', 'native', profile=probe.PROFILE)
        with patch('score_cup_proposal_trace.geometry') as geometry:
            report = phase_score.score_condition(original, saved, 'original', journal)
        geometry.assert_not_called()
        self.assertEqual(report['summary']['unknown_feedback_predictions'], 2)
        self.assertEqual(report['summary']['reported_matched'], 0)
        self.assertTrue(all(row['raw_first'] is None and row['selected'] is None for row in report['rows']))

    def test_all_four_receipt_exit_codes_must_be_zero(self):
        receipt = {'schema': 'fr5-cup-release-phase-paired-inference-execution-v1',
                   'status': 'completed', 'robot_commands': 0,
                   'commands': [{'model': model, 'prompt': prompt} for model, prompt in phase_score.RUNS],
                   'stages': [{'model': model, 'prompt': prompt, 'exit_code': 0, 'finished_at_unix': 1.}
                              for model, prompt in phase_score.RUNS]}
        self.assertEqual(len(phase_score.completed_runs(receipt)[1]), 4)
        receipt['stages'][-1]['exit_code'] = 1
        with self.assertRaisesRegex(ValueError, 'exit_code 0'):
            phase_score.completed_runs(receipt)


if __name__ == '__main__':
    unittest.main()
