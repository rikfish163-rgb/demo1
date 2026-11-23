"""Check native fit draws, production selection and exact first-draw pairing."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import call, patch

import torch

from lerobot_robot_fairino import cup_action_selection
from fr5_lerobot.diagnostics import evaluate_cup_training_timing_fit as evaluation


PROFILE = evaluation.fit.SETDOWN_FIXED_PROFILE
SEEDS = [20261105, 20261113]


def fit_case(case_id='source0-cycle1-release_at', stamp=111.25):
    state = [.1597, .0713, .0958, 0., 0., 0., .76, 1.]
    return {'case_id': case_id, 'source_index': 0, 'cycle': 1,
            'group': 'train_fit', 'split': 'train', 'kind': 'release_at',
            'initial_cup': {'table_normal': [0., .6, .8]},
            'request': {'schema': 'fr5-stream-request-v1', 'id': case_id,
                        'task': '抓杯，悬停5秒，放稳再松爪', 'source_time': stamp,
                        'source_clock': 'host_monotonic', 'rgb_jpeg_base64': 'b3JpZ2luYWw=',
                        'state': [0.] * 6 + [.76, 1.], 'cup_relative_state': state,
                        'observation_history': [{'age_s': .5, 'state': list(state)}] + [None] * 5},
            'observers': {'setdown': {
                'tracking': {'native_contact': 1, 'rim_minus_jaw_mm': None},
                'model_timing': {'phase': 'returning', 'setdown_age_s': .5,
                                 'support_gap_mm': 0., 'hover_seconds': 5., 'visual': 'table_height'},
                'timing': {'camera_monotonic': stamp, 'feedback_monotonic': stamp + .002}}},
            'targets': [{'nominal_time_monotonic': stamp + 1. / 6.,
                         'raw_camera_time_monotonic': stamp + .1,
                         'raw_robot_time_monotonic': stamp + .102,
                         'raw_cup_relative_action': list(state[:7])}],
            'event_times': {'five_second_deadline_monotonic': stamp - 2.}}


def fit_inputs(*cases):
    return {'schema': evaluation.fit.INPUT_SCHEMA, 'cases': list(cases)}


class ToyNativeModel:
    def __init__(self, jaws=(.76, 1.), *, failure_on=None, short_on=None):
        self.info = {'observation_context_profile': PROFILE, 'n_action_steps': 3}
        self.root = Path('/checkpoint')
        self.torch = torch
        self.jaws, self.failure_on, self.short_on = jaws, failure_on, short_on
        self.draw_index = self.reset_count = 0
        self.requests, self.replies = [], []

    def reset(self):
        self.reset_count += 1
        self.draw_index = 0

    def predict(self, request):
        self.draw_index += 1
        self.requests.append(request)
        if self.draw_index == self.failure_on:
            raise RuntimeError('Native sampling failed')
        count = 49 if self.draw_index == self.short_on else 50
        actions = torch.rand((count, 7), dtype=torch.float64, device='cpu')
        actions[:, :3] = torch.tensor(request['cup_relative_state'][:3], dtype=torch.float64)
        actions[:, 6] = self.jaws[min(self.draw_index - 1, len(self.jaws) - 1)]
        reply = {'actions': actions.tolist(), 'wire_action_representation': 'cup-relative-tcp-v1',
                 'inference_seconds': .01 * self.draw_index,
                 'sampling': {'seed': torch.initial_seed(), 'native_draw': self.draw_index}}
        self.replies.append(reply)
        return reply


def predict(model, inputs, seeds=SEEDS):
    with redirect_stderr(io.StringIO()):
        return evaluation.predict_fit_cases(model, inputs, seeds)


class EvaluateCupTrainingTimingFitTests(unittest.TestCase):
    def test_second_candidate_preserves_both_native_complete_replies_and_original_inputs(self):
        model, case = ToyNativeModel(), fit_case()
        inputs = fit_inputs(case)
        original = deepcopy(inputs)
        with patch.object(evaluation, 'choose_proposal', wraps=cup_action_selection.choose_proposal) as choose:
            first, selected = predict(model, inputs, [SEEDS[0]])
        choose.assert_called_once_with(model, evaluation.fit.model_request(case, 'setdown', PROFILE),
                                       case['initial_cup']['table_normal'], check_boundaries=True)
        self.assertIs(first[0]['reply'], model.replies[0])
        self.assertIs(selected[0]['reply']['actions'], model.replies[1]['actions'])
        for row in (first[0], selected[0]):
            self.assertEqual([7] * 50, [len(action) for action in row['reply']['actions']])
            self.assertEqual(row['request'], evaluation.fit.model_request(case, 'setdown', PROFILE))
            self.assertEqual(case['request']['source_time'], row['source_time'])
        selection = selected[0]['reply']['cup_proposal_selection']
        self.assertEqual(1, selection['selected_index'])
        self.assertEqual(2, selection['draws'])
        self.assertIs(selection['proposals'][0]['actions'], first[0]['reply']['actions'])
        self.assertIs(selection['proposals'][1]['actions'], selected[0]['reply']['actions'])
        self.assertAlmostEqual(.01, first[0]['reply']['inference_seconds'])
        self.assertAlmostEqual(.03, selected[0]['reply']['inference_seconds'])
        self.assertEqual(original, inputs)
        self.assertNotIn('predict', model.__dict__)
        self.assertIs(ToyNativeModel.predict, model.predict.__func__)

    def test_every_case_seed_resets_before_native_sampling_and_remains_exactly_paired(self):
        model = ToyNativeModel()
        inputs = fit_inputs(fit_case(), fit_case('source3-cycle2-release_after', 222.75))
        with patch.object(torch, 'manual_seed', wraps=torch.manual_seed) as seed:
            first, selected = predict(model, inputs)
        self.assertEqual(4, model.reset_count)
        self.assertEqual([call(s) for s in SEEDS * 2], seed.call_args_list)
        self.assertEqual([1, 2] * 4, [reply['sampling']['native_draw'] for reply in model.replies])
        self.assertEqual(4, len(first))
        self.assertEqual(4, len(selected))
        self.assertEqual(first[0]['reply']['actions'], first[2]['reply']['actions'])
        self.assertEqual(first[1]['reply']['actions'], first[3]['reply']['actions'])
        self.assertNotEqual(first[0]['reply']['actions'], first[1]['reply']['actions'])
        torch.manual_seed(SEEDS[0])
        independent = ToyNativeModel()
        reply = independent.predict(evaluation.fit.model_request(inputs['cases'][0], 'setdown', PROFILE))
        self.assertEqual(reply['actions'], first[0]['reply']['actions'])

    def test_three_unmatched_draws_keep_first_and_do_not_count_as_matched(self):
        model = ToyNativeModel(jaws=(.76,))
        first, selected = predict(model, fit_inputs(fit_case()), [SEEDS[0]])
        self.assertEqual(3, len(model.requests))
        self.assertEqual(3, selected[0]['draw_count'])
        self.assertFalse(selected[0]['selection_matched'])
        self.assertFalse(selected[0]['new_draw_selected'])
        self.assertIs(first[0]['reply']['actions'], selected[0]['reply']['actions'])
        self.assertEqual(0, selected[0]['reply']['cup_proposal_selection']['selected_index'])
        self.assertEqual([7] * 50, [len(a) for a in selected[0]['reply']['actions']])

    def test_case_without_selection_uses_one_unchanged_native_reply(self):
        case = fit_case()
        case['observers']['setdown']['tracking']['native_contact'] = 0
        case['observers']['setdown']['model_timing'].update(phase='released', setdown_age_s=None)
        model = ToyNativeModel()
        first, selected = predict(model, fit_inputs(case), [SEEDS[0]])
        self.assertEqual(1, len(model.requests))
        self.assertIs(first[0]['reply'], selected[0]['reply'])
        self.assertNotIn('cup_proposal_selection', selected[0]['reply'])

    def test_existing_instance_predict_is_restored(self):
        model = ToyNativeModel()
        native = model.predict
        override = lambda request: native(request)
        model.predict = override
        predict(model, fit_inputs(fit_case()), [SEEDS[0]])
        self.assertIs(override, model.predict)

    def test_native_error_and_incomplete_second_draw_restore_predict(self):
        for kwargs, error in (({'failure_on': 2}, RuntimeError), ({'short_on': 2}, ValueError)):
            with self.subTest(kwargs=kwargs):
                model = ToyNativeModel(**kwargs)
                with self.assertRaises(error):
                    predict(model, fit_inputs(fit_case()), [SEEDS[0]])
                self.assertNotIn('predict', model.__dict__)
                self.assertIs(ToyNativeModel.predict, model.predict.__func__)
                self.assertEqual(2, len(model.requests))

    def test_invalid_seed_or_checkpoint_contract_does_not_sample(self):
        for seeds in ([], [SEEDS[0]] * 2):
            model = ToyNativeModel()
            with self.subTest(seeds=seeds), self.assertRaisesRegex(ValueError, 'Seeds'):
                predict(model, fit_inputs(fit_case()), seeds)
            self.assertEqual([], model.requests)
        for key, value in (('n_action_steps', 10), ('observation_context_profile', 'other')):
            model = ToyNativeModel()
            model.info[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'three-point'):
                predict(model, fit_inputs(fit_case()))
            self.assertEqual([], model.requests)

    def test_scorer_checks_embedded_rgb_state_and_time_in_both_reports(self):
        model, case = ToyNativeModel(), fit_case()
        first, selected = predict(model, fit_inputs(case), [SEEDS[0]])
        for rows in (first, selected):
            saved = evaluation.prediction_report(model, '/original/fit-inputs.json', [SEEDS[0]], rows)
            evaluation.timing.prediction_index({case['case_id']: case}, saved)
            for key, value in (('rgb_jpeg_base64', 'different'), ('state', [1.] * 8),
                               ('cup_relative_state', [1.] * 8), ('source_time', 112.)):
                changed = deepcopy(saved)
                changed['predictions'][0]['request'][key] = value
                with self.subTest(key=key, selected=rows is selected), self.assertRaisesRegex(
                        ValueError, 'saved full request differs'):
                    evaluation.timing.prediction_index({case['case_id']: case}, changed)

    def test_first_pairing_uses_case_seed_keys_and_exact_values_in_all_fifty_points(self):
        model = ToyNativeModel()
        first, _ = predict(model, fit_inputs(fit_case()), SEEDS)
        reference = deepcopy(first[::-1])
        paired = evaluation.compare_first_predictions(torch, first, reference)
        self.assertTrue(paired['exactly_paired'])
        self.assertEqual(2, paired['compared_draws'])
        last = reference[0]['reply']['actions'][49]
        last[5] = math.nextafter(last[5], math.inf)
        different = evaluation.compare_first_predictions(torch, first, reference)
        self.assertFalse(different['exactly_paired'])
        self.assertEqual(1, different['action_mismatch_draws'])
        self.assertEqual(1, different['json_action_mismatch_draws'])
        self.assertEqual(1, different['torch_action_mismatch_draws'])

    def test_cli_writes_only_selected_and_retains_exact_pairing_or_mismatch_report(self):
        case = fit_case()
        inputs = fit_inputs(case)
        baseline = ToyNativeModel()
        first, _ = predict(baseline, inputs)
        for mismatch in (False, True):
            with self.subTest(mismatch=mismatch):
                reference = evaluation.prediction_report(baseline, '/original/fit-inputs.json', SEEDS, deepcopy(first))
                if mismatch:
                    reference['predictions'][1]['reply']['actions'][49][5] += .01
                model = ToyNativeModel()
                output = evaluation.OUTPUT_ROOT / 'test-fit-selected-predictions.json'
                argv = ['evaluate-fit', '--checkpoint', '/checkpoint', '--inputs', '/original/fit-inputs.json',
                        '--raw-predictions', '/original/fit-predictions.json', '--output', str(output), '--gpu', '8']
                for seed in SEEDS:
                    argv += ['--seed', str(seed)]
                stdout = io.StringIO()
                with patch.object(sys, 'argv', argv), patch.object(sys, 'path', list(sys.path)), \
                        patch.object(evaluation.fit, 'read_json', side_effect=[inputs, reference]), \
                        patch.object(evaluation.timing, 'cases_index', return_value={case['case_id']: case}), \
                        patch.object(evaluation.fit, 'write_new') as write, \
                        patch.dict(sys.modules, {'serve_stream_pi05': SimpleNamespace(Inference=lambda *a, **k: model)}), \
                        redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                    status = evaluation.main()
                self.assertEqual(int(mismatch), status)
                write.assert_called_once()
                path, report = write.call_args.args
                self.assertEqual(output, path)
                self.assertEqual(evaluation.FIT_SCHEMA, report['schema'])
                self.assertEqual(2, len(report['predictions']))
                self.assertTrue(all(row['reply']['cup_proposal_selection']['selected_index'] == 1
                                    for row in report['predictions']))
                self.assertEqual(not mismatch, report['raw_first_pairing']['exactly_paired'])
                self.assertEqual(int(mismatch), report['raw_first_pairing']['action_mismatch_draws'])
                self.assertEqual(0, report['robot_commands'])
                summary = json.loads(stdout.getvalue())
                self.assertEqual(report['raw_first_pairing'], summary['raw_first_pairing'])
                self.assertNotIn('"actions"', stdout.getvalue())
                self.assertNotIn('predict', model.__dict__)

    def test_cli_rejects_local_prediction_output_before_loading_inference(self):
        argv = ['evaluate-fit', '--checkpoint', '/checkpoint', '--inputs', '/original/fit-inputs.json',
                '--raw-predictions', '/original/fit-predictions.json', '--output', '/tmp/fit-predictions.json',
                '--gpu', '8', '--seed', str(SEEDS[0])]
        with patch.object(sys, 'argv', argv), patch.object(evaluation.fit, 'read_json') as read:
            with self.assertRaisesRegex(ValueError, 'Output must be a JSON file under'):
                evaluation.main()
        read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
