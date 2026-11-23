"""CPU checks for phase-only pairing, complete draws and PGI recovery scores."""

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'diagnostics'))
import probe_cup_release_phase as probe
from fr5_lerobot.training.stream_observation_history import action_clock


def inputs():
    cases = []
    for frame, opening, contact in ((195, .78, 1), (309, .99, 0), (578, .78, 1), (692, .99, 0)):
        identifier = f'recovery-{frame}'
        state = [.1597, .0713, .0958, 1.62, -.022, -1.174, opening, 1.]
        cases.append({
            'case_id': identifier, 'group': 'heldout_source_7', 'kind': 'feedback_recovered',
            'source_frame_index': frame,
            'initial_cup': {'table_normal': [0., 0., 1.], 'base_xyz_mm': [0., 0., 0.]},
            'request': {'schema': 'fr5-stream-request-v1', 'id': identifier,
                        'task': '抓取桌面上的透明杯子，在空中维持4秒后放下并松开夹爪',
                        'rgb_jpeg_base64': 'YXJjaGl2ZWQtanBlZw==', 'source_time': 100. + frame,
                        'source_clock': 'host_monotonic', 'state': [1.] * 6 + state[6:],
                        'cup_relative_state': state, 'initial_cup_uv': [1351., 715.],
                        'observation_history': [{'age_s': .5, 'state': [0.] * 8}] + [None] * 5},
            'tracking': {'native_contact': contact, 'rim_minus_jaw_mm': None, 'reason': 'rim_not_observed'},
            'timing_setdown': {'state': {'phase': 'grasp_contact' if contact else 'at_pickup',
                                         'hover_seconds': None, 'visual': 'missing',
                                         'support_gap_mm': 0. if contact else None, 'setdown_age_s': None},
                               'camera_monotonic': 100. + frame, 'feedback_monotonic': 100. + frame + .003,
                               'contact_monotonic': 100. + frame if contact else None},
        })
    original = {'schema': probe.INPUT_SCHEMA, 'timing_ready': True, 'cases': cases,
                'timing_source_root': '/original-timer'}
    corrected = deepcopy(original)
    corrected['timing_source_root'] = '/corrected-timer'
    for case in corrected['cases']:
        if case['tracking']['native_contact'] == 0:
            case['timing_setdown']['state']['phase'] = 'released'
    return original, corrected


class CPUModel:
    """Exercise the real selector with seeded CPU draws and captured input tensors."""

    def __init__(self, *, replace_first=False, exhaust=False, malformed=False, change_image=False):
        self.torch = torch
        self.root = Path('/checkpoint')
        self.info = {'observation_context_profile': probe.PROFILE, 'n_action_steps': 3,
                     'action_representation': 'cup-relative-tcp-tick-delta-v1'}
        self.policy = SimpleNamespace(config=SimpleNamespace(num_inference_steps=10))
        self.training = SimpleNamespace(predict_action_chunk=lambda policy, batch: float(torch.rand(())))
        self.replace_first, self.exhaust = replace_first, exhaust
        self.malformed, self.change_image = malformed, change_image
        self.resets, self.draw = 0, 0
        self.requests, self.replies = [], []

    def reset(self):
        self.resets += 1
        self.draw = 0

    def predict(self, request):
        self.requests.append(deepcopy(request))
        self.draw += 1
        phase = request['cup_timing']['phase']
        tokens = torch.tensor([list(phase.encode('ascii'))])
        image = torch.zeros(1, 3, 2, 2)
        if self.change_image and phase == 'released':
            image[0, 0, 0, 0] = 1.
        batch = {probe.CLOCK: torch.tensor([action_clock(request['task'], request['cup_timing'],
                                                        cup_tracking=request['cup_tracking'], profile=probe.PROFILE)]),
                 probe.FIXED: True, probe.TOKENS: tokens, probe.MASK: torch.ones_like(tokens, dtype=torch.bool),
                 'observation.state': torch.tensor([request['cup_relative_state'] + [0.] * 24]),
                 'observation.images.front': image}
        sampling_state = torch.get_rng_state().tolist()
        noise = self.training.predict_action_chunk(self.policy, batch)
        state = request['cup_relative_state']
        actions = [[*state[:6], state[6]] for _ in range(50)]
        for tick, action in enumerate(actions[:3], 1):
            action[0] += noise * tick / 10000.
        if phase == 'at_pickup':
            actions[0][6], actions[1][6], actions[2][6] = .81, 1.01, .99
            if self.exhaust or (self.replace_first and self.draw == 1):
                actions[0][0] += .04
            elif self.replace_first:
                actions[0][6] = .99
        elif phase == 'released':
            actions[1][6] = 1.01
        if self.malformed:
            actions.pop()
        reply = {'schema': 'fr5-stream-reply-v1', 'id': request['id'], 'actions': actions,
                 'inference_seconds': .01, 'sampling': {'cuda_rng_state': sampling_state, 'denoising_steps': 10}}
        self.replies.append(deepcopy(reply))
        return reply


class PairingTests(unittest.TestCase):
    def test_task_condition_changes_both_phases_without_changing_source_inputs(self):
        original, corrected = inputs()
        saved = deepcopy((original, corrected))
        pairs = probe.pair_inputs(original, corrected)
        model = CPUModel()
        task = '抓取桌面上的透明杯子，在空中维持5秒后放下'
        with redirect_stderr(io.StringIO()):
            results = probe.predict_pairs(model, pairs, [20261202], task_override=task)
        self.assertEqual((original, corrected), saved)
        self.assertTrue(all(request['task'] == task for request in model.requests))
        for result in results:
            self.assertEqual(result['original_request']['task'], task)
            self.assertEqual(result['corrected_request']['task'], task)
            self.assertEqual(result['original_recorded_task'], original['cases'][0]['request']['task'])

    def test_pair_by_case_id_allows_only_the_exact_phase_change(self):
        original, corrected = inputs()
        saved = deepcopy(original)
        corrected['cases'].reverse()
        pairs = probe.pair_inputs(original, corrected)
        self.assertEqual([pair['case_id'] for pair in pairs], [case['case_id'] for case in original['cases']])
        self.assertEqual([pair['corrected_case_index'] for pair in pairs], [3, 2, 1, 0])
        self.assertEqual([pair['phase_changed'] for pair in pairs], [False, True, False, True])
        self.assertEqual([pair['recovery_role'] for pair in pairs],
                         ['contact_recovery_control', 'open_release_recovery'] * 2)
        for pair in pairs:
            expected = deepcopy(pair['original_request'])
            expected['cup_timing']['phase'] = pair['corrected_request']['cup_timing']['phase']
            self.assertEqual(expected, pair['corrected_request'])
            self.assertEqual(pair['original_request']['task'], pair['case']['request']['task'])
        self.assertEqual(original, saved)

    def test_rejects_concrete_nonphase_observation_and_timing_changes(self):
        edits = (
            (('request', 'rgb_jpeg_base64'), 'different-jpeg'),
            (('request', 'state', 0), 2.),
            (('request', 'cup_relative_state', 0), .16),
            (('request', 'observation_history', 0, 'age_s'), .7),
            (('request', 'observation_history', 0, 'state', 6), .99),
            (('request', 'source_time'), 410.),
            (('request', 'source_clock'), 'other-clock'),
            (('request', 'task'), '在空中维持5秒'),
            (('tracking', 'rim_minus_jaw_mm'), [1., 2., 3.]),
            (('tracking', 'native_contact'), 1),
            (('timing_setdown', 'state', 'support_gap_mm'), 0.),
            (('timing_setdown', 'state', 'hover_seconds'), 4.),
            (('timing_setdown', 'state', 'setdown_age_s'), .3),
            (('timing_setdown', 'state', 'visual'), 'table_height'),
            (('timing_setdown', 'camera_monotonic'), 410.),
            (('timing_setdown', 'feedback_monotonic'), 410.),
            (('timing_setdown', 'contact_monotonic'), 408.),
            (('initial_cup', 'base_xyz_mm', 0), 1.),
        )
        for path, value in edits:
            with self.subTest(path=path):
                original, corrected = inputs()
                target = corrected['cases'][1]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaisesRegex(ValueError, 'paired inputs differ at /'):
                    probe.pair_inputs(original, corrected)

    def test_labels_and_phase_outside_the_intervention_path_are_not_ignored(self):
        for field in ('targets_camera_aligned', 'labels', 'cup_timing'):
            with self.subTest(field=field):
                original, corrected = inputs()
                original['cases'][1]['request'][field] = {'phase': 'at_pickup', 'opening': .99}
                corrected['cases'][1]['request'][field] = {'phase': 'released', 'opening': .99}
                with self.assertRaisesRegex(ValueError, f'/request/{field}/phase'):
                    probe.pair_inputs(original, corrected)
        for path in (('cup_relative_action', 6), ('target_time_monotonic',), ('cut_after',)):
            original, corrected = inputs()
            label = {'cup_relative_action': [0.] * 6 + [.99], 'target_time_monotonic': 409.2, 'cut_after': False}
            for document in (original, corrected):
                document['cases'][1]['targets_camera_aligned'] = [deepcopy(label)]
            changed = corrected['cases'][1]['targets_camera_aligned'][0]
            if len(path) == 2:
                changed[path[0]][path[1]] = .78
            else:
                changed[path[0]] = True if path[0] == 'cut_after' else 409.3
            with self.assertRaisesRegex(ValueError, '/targets_camera_aligned/0/'):
                probe.pair_inputs(original, corrected)

    def test_rejects_missing_and_duplicate_cases(self):
        original, corrected = inputs()
        corrected['cases'].pop()
        with self.assertRaisesRegex(ValueError, 'case_id sets differ'):
            probe.pair_inputs(original, corrected)
        original, corrected = inputs()
        corrected['cases'].append(deepcopy(corrected['cases'][0]))
        with self.assertRaisesRegex(ValueError, 'unique case_id'):
            probe.pair_inputs(original, corrected)

    def test_release_requires_recovered_open_feedback_and_controls_stay_fixed(self):
        original, corrected = inputs()
        corrected['cases'][0]['timing_setdown']['state']['phase'] = 'released'
        with self.assertRaisesRegex(ValueError, 'controls must remain unchanged'):
            probe.pair_inputs(original, corrected)
        original, corrected = inputs()
        for document in (original, corrected):
            document['cases'][1]['request']['state'][7] = 0.
            document['cases'][1]['request']['cup_relative_state'][7] = 0.
        with self.assertRaisesRegex(ValueError, 'valid recovered'):
            probe.pair_inputs(original, corrected)

    def test_same_override_is_rejected_when_it_changes_the_original_task(self):
        original, corrected = inputs()
        for document in (original, corrected):
            document['cases'][1]['task_override'] = '在空中维持5秒'
        with self.assertRaisesRegex(ValueError, 'original recorded task'):
            probe.pair_inputs(original, corrected)


class PredictionTests(unittest.TestCase):
    def run_pairs(self, model, seeds=(7, 19)):
        with redirect_stderr(io.StringIO()):
            return probe.predict_pairs(model, probe.pair_inputs(*inputs()), seeds)

    def test_same_seeds_pair_raw_draws_and_capture_phase_input_channels(self):
        model = CPUModel()
        results = self.run_pairs(model)
        self.assertEqual(model.resets, 4 * 2 * 2)
        for case in results:
            for row in case['seeds']:
                audit = row['paired_draw_audits'][0]
                self.assertTrue(audit['pre_sampling_cuda_rng_equal'])
                self.assertTrue(audit['preprocess']['other_tensors_elementwise_equal'])
                clock = audit['preprocess']['channels'][probe.CLOCK]
                self.assertEqual(clock['elementwise_equal'], not case['phase_changed'])
                if case['phase_changed']:
                    self.assertEqual(clock['changed_indices'], [[0, 3], [0, 9]])
                    self.assertFalse(audit['preprocess']['channels'][probe.TOKENS]['elementwise_equal'])
                    self.assertFalse(audit['preprocess']['channels'][probe.MASK]['elementwise_equal'])
                    self.assertTrue(row['original']['scores']['raw_first']['early_close'])
                    self.assertTrue(row['corrected']['scores']['raw_first']['all_full_open'])
                else:
                    self.assertTrue(row['differences']['raw_first']['complete_actions_equal'])
                    self.assertTrue(row['differences']['selected']['complete_actions_equal'])

    def test_no_selector_is_unassessed_and_never_counts_as_a_selector_match(self):
        results = self.run_pairs(CPUModel(), seeds=(7,))
        for case in results:
            corrected = case['seeds'][0]['corrected']
            self.assertEqual(corrected['selector']['status'], 'not_applied')
            self.assertFalse(corrected['selector']['applied'])
            self.assertIsNone(corrected['selector']['matched'])
            self.assertIsNone(corrected['selector']['raw_first_matched'])
            self.assertIsNone(corrected['selector_trace'])
        summary = probe.summarize(results)
        self.assertEqual(summary['corrected']['selector_status'], {'not_applied': 4})
        self.assertEqual(summary['original']['selector_status'], {'not_applied': 2, 'matched': 2})

    def test_replacement_keeps_raw_and_selected_complete_predictions_separate(self):
        model = CPUModel(replace_first=True)
        results = self.run_pairs(model, seeds=(7,))
        row = results[1]['seeds'][0]
        original = row['original']
        self.assertEqual(original['draw_count'], 2)
        self.assertTrue(original['new_draw_selected'])
        self.assertFalse(original['selector']['raw_first_matched'])
        self.assertTrue(original['selector']['matched'])
        self.assertEqual(original['selector_trace']['selected_index'], 1)
        self.assertTrue(original['scores']['raw_first']['early_close'])
        self.assertFalse(original['scores']['selected']['early_close'])
        self.assertFalse(original['scores']['raw_first']['all_full_open'])
        self.assertTrue(original['scores']['selected']['all_full_open'])
        self.assertEqual(row['unpaired_draw_counts'], {'original': 1, 'corrected': 0})
        for candidate in original['candidate_replies']:
            self.assertEqual(len(candidate['actions']), 50)
            self.assertEqual({len(action) for action in candidate['actions']}, {7})
            self.assertEqual(candidate['actions'][1][6], 1.01)
        self.assertEqual(original['selected_reply']['actions'], original['candidate_replies'][1]['actions'])
        self.assertEqual(original['raw_first_reply']['actions'], original['candidate_replies'][0]['actions'])
        self.assertEqual(original['candidate_replies'], model.replies[2:4])

    def test_three_unmatched_draws_remain_unmatched(self):
        row = self.run_pairs(CPUModel(exhaust=True), seeds=(7,))[1]['seeds'][0]['original']
        self.assertEqual(row['draw_count'], 3)
        self.assertEqual(row['selector']['status'], 'unmatched')
        self.assertFalse(row['selector']['matched'])
        self.assertEqual(len(row['selector_trace']['proposals']), 3)
        self.assertEqual(row['raw_first_reply']['actions'], row['selected_reply']['actions'])

    def test_incomplete_chunk_is_rejected_and_capture_hook_is_restored(self):
        model = CPUModel(malformed=True)
        sample = model.training.predict_action_chunk
        pair = probe.pair_inputs(*inputs())[1]
        with self.assertRaisesRegex(ValueError, '50 x 7'):
            probe.predict_condition(model, pair['original_request'], pair['case']['initial_cup']['table_normal'])
        self.assertIs(model.training.predict_action_chunk, sample)

    def test_changed_preprocessed_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'state or image tensors differ'):
            self.run_pairs(CPUModel(change_image=True), seeds=(7,))

    def test_duplicate_seeds_and_wrong_prefix_are_rejected_before_prediction(self):
        model = CPUModel()
        with self.assertRaisesRegex(ValueError, 'distinct'):
            self.run_pairs(model, seeds=(7, 7))
        model.info['n_action_steps'] = 10
        with self.assertRaisesRegex(ValueError, 'prefix3'):
            self.run_pairs(model, seeds=(7,))
        self.assertFalse(model.requests)


class ScoringTests(unittest.TestCase):
    def test_thresholds_and_maximum_use_the_unmodified_first_three_points(self):
        pair = probe.pair_inputs(*inputs())[1]
        request, case = pair['original_request'], pair['case']
        actions = [[*request['cup_relative_state'][:6], 1.] for _ in range(50)]
        for action, opening, move in zip(actions, (.879999, .98, 1.01), (.005, .0001, .001)):
            action[6] = opening
            action[0] += move
        saved = deepcopy(actions)
        score = probe.score_prefix(case, request, actions)
        self.assertEqual(score['below_88_ticks'], [1])
        self.assertEqual(score['full_open_ge_98_ticks'], [2, 3])
        self.assertTrue(score['early_close'])
        self.assertFalse(score['all_full_open'])
        self.assertAlmostEqual(score['max_first_three_displacement_mm'], 5.)
        self.assertEqual(actions, saved)
        actions[0][6] = .88
        self.assertFalse(probe.score_prefix(case, request, actions)['early_close'])
        actions[0][6] = .98
        self.assertTrue(probe.score_prefix(case, request, actions)['all_full_open'])

    def test_missing_support_and_no_late_open_scope_do_not_pass(self):
        results = PredictionTests().run_pairs(CPUModel(), seeds=(7,))
        for condition in ('original', 'corrected'):
            release = results[1]['seeds'][0][condition]['scores']['raw_first']
            self.assertFalse(release['support']['assessed'])
            self.assertIsNone(release['support']['within_1_5mm'])
            self.assertEqual(release['support']['status'], 'missing_support')
            self.assertFalse(release['late_open']['assessed'])
            self.assertIsNone(release['late_open']['late_or_missing'])
            self.assertEqual(release['late_open']['status'], 'already_open_at_observation')
            control = results[0]['seeds'][0][condition]['scores']['raw_first']
            self.assertIsNone(control['early_close'])
            self.assertIsNone(control['all_full_open'])
            self.assertEqual(control['late_open']['status'], 'missing_release_labels')
        summary = probe.summarize(results)
        self.assertEqual(summary['corrected']['raw_first']['support_assessed'], 2)
        self.assertEqual(summary['corrected']['raw_first']['late_open_assessed'], 0)
        self.assertEqual(summary['corrected']['raw_first']['all_full_open'], 2)

    def test_late_open_requires_recorded_deadline_inside_available_prefix(self):
        pair = probe.pair_inputs(*inputs())[0]
        request, case = pair['original_request'], deepcopy(pair['case'])
        actions = [[*request['cup_relative_state'][:6], .78] for _ in range(50)]
        case['targets_camera_aligned'] = [{'cup_relative_action': [0.] * 6 + [.78]} for _ in range(3)]
        case['targets_camera_aligned'][2]['cup_relative_action'][6] = 1.
        late = probe.score_prefix(case, request, actions)['late_open']
        self.assertFalse(late['assessed'])
        self.assertIsNone(late['late_or_missing'])
        case['targets_camera_aligned'][0]['cup_relative_action'][6] = 1.
        late = probe.score_prefix(case, request, actions)['late_open']
        self.assertTrue(late['assessed'])
        self.assertTrue(late['late_or_missing'])
        self.assertEqual(late['deadline_tick'], 2)
        case['targets_camera_aligned'] = case['targets_camera_aligned'][:1]
        self.assertFalse(probe.score_prefix(case, request, actions)['late_open']['assessed'])


class OutputTests(unittest.TestCase):
    def test_output_requires_new_server_file_outside_checkpoint(self):
        checkpoint = probe.SERVER_ROOT / 'checkpoints/model'
        output = probe.SERVER_ROOT / 'audits/release-phase-new.json'
        with patch.object(Path, 'exists', return_value=False):
            self.assertEqual(probe.output_path(output, checkpoint), output.resolve())
            with self.assertRaisesRegex(ValueError, 'server learning directory'):
                probe.output_path(Path('/tmp/release-phase.json'), checkpoint)
            with self.assertRaisesRegex(ValueError, 'outside the checkpoint'):
                probe.output_path(checkpoint / 'probe.json', checkpoint)
        with patch.object(Path, 'exists', return_value=True):
            with self.assertRaises(FileExistsError):
                probe.output_path(output, checkpoint)

    def test_cli_saves_one_report_with_raw_selected_and_zero_motion(self):
        model = CPUModel()
        inference = Mock(return_value=model)
        output = probe.SERVER_ROOT / 'audits/release-phase-new.json'
        argv = ['probe', '--original-inputs', '/original.json', '--corrected-inputs', '/corrected.json',
                '--checkpoint', '/checkpoint', '--gpu', '9', '--seed', '7', '--seed', '19', '--output', str(output)]
        with (patch.object(probe.sys, 'argv', argv), patch.object(probe, 'read_json', side_effect=inputs()),
              patch.object(probe, 'output_path', return_value=output), patch.object(probe, 'write_new') as write,
              patch.dict(sys.modules, {'serve_stream_pi05': SimpleNamespace(Inference=inference)}),
              redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
            probe.main()
        inference.assert_called_once_with(Path('/checkpoint'), gpu=9)
        write.assert_called_once()
        path, report = write.call_args.args
        self.assertEqual(path, output)
        self.assertEqual(report['seeds'], [7, 19])
        self.assertEqual(report['action_shape'], [50, 7])
        self.assertEqual(report['robot_connections'], 0)
        self.assertEqual(report['robot_commands'], 0)
        self.assertFalse(report['task_success'])
        self.assertEqual(report['observation_context_profile'], probe.PROFILE)
        self.assertIn('raw_first_reply', report['results'][0]['seeds'][0]['original'])
        self.assertIn('selected_reply', report['results'][0]['seeds'][0]['original'])


if __name__ == '__main__':
    unittest.main()
