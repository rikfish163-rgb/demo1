"""CPU checks for paired terminal-sampler evidence and failure reporting."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'diagnostics'))
import probe_cup_terminal_gradients as probe
import relative_stream_actions as relative
import serve_stream_pi05 as service
import stream_terminal_loss as terminal_loss
import train_stream_pi05 as training


class AdaptedLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(1, 1)
        self.lora_A = nn.Module()
        self.lora_A.default = nn.Linear(1, 1, bias=False)
        self.lora_B = nn.Module()
        self.lora_B.default = nn.Linear(1, 1, bias=False)
        for parameter in self.parameters():
            nn.init.constant_(parameter, .1)

    def forward(self, value):
        return self.base_layer(value) + self.lora_B.default(self.lora_A.default(value))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.v_proj = AdaptedLinear(), AdaptedLinear()

    def forward(self, value):
        return self.q_proj(value) + self.v_proj(value)


def branch():
    result = nn.Module()
    layer = nn.Module()
    layer.self_attn = Attention()
    result.layers = nn.ModuleList([layer])
    return result


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_in_proj, self.action_out_proj = nn.Linear(32, 1), nn.Linear(1, 32)
        for parameter in (*self.action_in_proj.parameters(), *self.action_out_proj.parameters()):
            nn.init.constant_(parameter, .01)
        self.paligemma_with_expert = nn.Module()
        core = self.paligemma_with_expert
        core.gemma_expert = nn.Module()
        core.gemma_expert.model = branch()
        core.paligemma = nn.Module()
        core.paligemma.model = nn.Module()
        core.paligemma.model.language_model = branch()
        core.paligemma.model.vision_tower = nn.Module()
        core.paligemma.model.vision_tower.vision_model = nn.Module()
        core.paligemma.model.vision_tower.vision_model.encoder = branch()

    def embed_suffix(self, noise, timestep):
        valid = torch.ones(noise.shape[:2], dtype=torch.bool)
        return noise, valid, torch.zeros_like(valid), None

    def denoise_step(self, *, prefix_pad_masks, past_key_values, x_t, timestep):
        embedded, _, _, _ = self.embed_suffix(x_t, timestep)
        suffix = self.action_in_proj(embedded) + past_key_values
        suffix = self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn(suffix)
        return self.action_out_proj(suffix)

    @torch.no_grad()
    def sample_actions(self, images, image_masks, tokens, masks, *, noise, num_steps=10):
        core = self.paligemma_with_expert
        value = images[0].mean().reshape(1, 1, 1) + tokens.float().mean() / 100.
        value = core.paligemma.model.vision_tower.vision_model.encoder.layers[0].self_attn(value)
        prefix = core.paligemma.model.language_model.layers[0].self_attn(value)
        actions = noise
        for i in range(num_steps):
            velocity = self.denoise_step(prefix_pad_masks=None, past_key_values=prefix, x_t=actions,
                                         timestep=torch.full((1,), 1. - i / num_steps))
            actions = actions - velocity / num_steps
        return actions


class Policy(nn.Module):
    def __init__(self, difference=False):
        super().__init__()
        self.model = Model()
        self.config = SimpleNamespace(num_inference_steps=10, max_action_dim=32)
        self.difference = difference
        training.trainable_parameters(self, training.VISION_EXPERT_ADAPTATION)
        self.model.paligemma_with_expert.paligemma.model.vision_tower.eval()

    def _preprocess_images(self, batch):
        return [batch['observation.images.base_0_rgb']], [torch.ones(1, dtype=torch.bool)]

    @torch.no_grad()
    def predict_action_chunk(self, batch, *, noise):
        self.eval()
        images, masks = self._preprocess_images(batch)
        result = self.model.sample_actions(images, masks, batch['observation.language.tokens'],
            batch['observation.language.attention_mask'], noise=noise)[..., :7]
        if self.difference:
            result = result.clone()
            result[0, 49, 6] = torch.nextafter(result[0, 49, 6], torch.tensor(float('inf')))
        return result


def case(identifier, source, row=1):
    anchor = [.02, -.03, .04, .1, .2, .3, 1., 1.]
    return {'case_id': identifier, 'source_index': source, 'row_index': row, 'source_frame_index': row,
        'kind': 'first_close_offset', 'split': 'train', 'group': 'train_support',
        'request': {'id': identifier, 'task': '抓取桌面上的透明杯子，在空中维持5秒后放下',
                    'rgb_jpeg_base64': 'original-image', 'state': [0.] * 6 + [1., 1.],
                    'cup_relative_state': anchor, 'observation_history': [{'age_s': .5, 'state': anchor}],
                    'initial_cup_uv': [1238., 688.], 'cup_tracking': {'rim_minus_jaw_mm': None, 'native_contact': 0},
                    'cup_timing': {'phase': 'at_pickup', 'hover_seconds': None}},
        'targets_camera_aligned': [{'tick': tick,
            'cup_relative_action': [anchor[0] + .001 * tick, anchor[1] - .002 * tick,
                                    anchor[2] + .0005 * tick, *anchor[3:7]]} for tick in (1, 2, 3)]}


def inputs():
    return {'schema': probe.INPUT_SCHEMA, 'timing_ready': True,
            'cases': [case(identifier, source) for identifier, source in zip(probe.DEFAULT_CASE_IDS, (0, 2, 5))]}


def batch():
    return {probe.FIXED: True, probe.CLOCK: torch.tensor([[1.] * 25]),
            'observation.state': torch.zeros(1, 32), 'observation.images.base_0_rgb': torch.ones(1, 3, 2, 2),
            'observation.language.tokens': torch.tensor([[1, 2, 3]]),
            'observation.language.attention_mask': torch.tensor([[True, True, False]]), 'task': ['原始任务']}


def run_batch(policy=None):
    return probe.probe_batch(torch, training, policy or Policy(), batch(), torch.ones(1, 50, 32),
        torch.zeros(1, 3, 7), torch.full((7,), .0015), torch.ones(1, 3, 7), 21.)


class SelectionTests(unittest.TestCase):
    def test_default_cases_are_original_sources_0_2_5_and_indices_preserve_order(self):
        data = inputs()
        self.assertEqual([value['source_index'] for _, value in probe.select_cases(data)], [0, 2, 5])
        self.assertEqual([index for index, _ in probe.select_cases(data, [2, 0])], [2, 0])
        with self.assertRaisesRegex(ValueError, 'distinct'):
            probe.select_cases(data, [0, 0])
        with self.assertRaisesRegex(ValueError, 'outside'):
            probe.select_cases(data, [-1])

    def test_rejects_heldout_failed_and_task_override_observations(self):
        for change in ({'source_index': 7}, {'split': 'validation'}, {'group': 'failed_trial'},
                       {'case_id': 'trial02-failed'}, {'task_override': '换任务'}):
            data = inputs()
            data['cases'][0].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'original training-support'):
                probe.select_cases(data, [0])

    def test_cli_accepts_append_case_indices_without_loading_a_model(self):
        with patch.object(probe, 'run', return_value=0) as run:
            self.assertEqual(probe.main(['--checkpoint', '/checkpoint', '--inputs', '/inputs', '--gpu', '9',
                '--output', '/output', '--case-index', '7', '--case-index', '13']), 0)
        self.assertEqual(run.call_args.args[0].case_index, [7, 13])


class PreprocessingTests(unittest.TestCase):
    def test_capture_uses_service_request_path_and_restores_function_without_sampling(self):
        recorded = case(probe.DEFAULT_CASE_IDS[0], 0)['request']
        request = deepcopy(recorded)
        expected = batch()
        prepare = Mock(return_value=expected)
        model = SimpleNamespace(info={'action_representation': relative.CUP_TICK_RELATIVE,
            'image_profile': 'original-profile', 'observation_context_profile': probe.FIXED_PROFILE},
            torch=SimpleNamespace(), training=object(), pre=object())
        with patch.object(service, 'validate_request', return_value=(request['id'], request['task'], request['state'], object())), \
                patch.object(service, 'prepare_request_batch', prepare):
            actual = probe.capture_request_batch(service, model, request)
            self.assertIs(service.prepare_request_batch, prepare)
        self.assertIs(actual, expected)
        self.assertEqual(request, recorded)
        self.assertIs(prepare.call_args.args[2], request['cup_relative_state'])
        self.assertEqual(prepare.call_args.args[4], recorded['task'])
        self.assertIs(prepare.call_args.args[7], request['observation_history'])
        self.assertIs(prepare.call_args.kwargs['cup_tracking'], request['cup_tracking'])
        self.assertIs(prepare.call_args.kwargs['cup_timing'], request['cup_timing'])

    def test_capture_restores_prepare_function_on_preprocessing_error(self):
        request = case(probe.DEFAULT_CASE_IDS[0], 0)['request']
        model = SimpleNamespace(info={'action_representation': relative.CUP_TICK_RELATIVE,
            'image_profile': 'original-profile', 'observation_context_profile': probe.FIXED_PROFILE},
            torch=SimpleNamespace(), training=object(), pre=object())
        prepare = Mock(side_effect=ValueError('missing original image'))
        with patch.object(service, 'validate_request', return_value=(request['id'], request['task'], request['state'], object())), \
                patch.object(service, 'prepare_request_batch', prepare):
            with self.assertRaisesRegex(ValueError, 'missing original image'):
                probe.capture_request_batch(service, model, request)
            self.assertIs(service.prepare_request_batch, prepare)

    def test_targets_keep_original_anchor_tick_delta_and_half_quantile_span(self):
        recorded = case(probe.DEFAULT_CASE_IDS[0], 0)
        before = deepcopy(recorded)
        normalization = {'action': {'q01': [-.002] * 7, 'q99': [.002] * 7}}
        targets, scale = probe.terminal_targets(torch, relative, recorded, normalization, torch.device('cpu'))
        torch.testing.assert_close(targets[0, :, :3], torch.tensor([[.5, -1., .25]] * 3), atol=3e-6, rtol=0.)
        torch.testing.assert_close(scale, torch.full((7,), .002))
        self.assertEqual(recorded, before)


class SamplingTests(unittest.TestCase):
    def test_exact_full_draw_has_all_required_gradients_and_restores_modes_and_rng(self):
        policy = Policy()
        modes = probe.mode_snapshot(policy)
        flags = {name: parameter.requires_grad for name, parameter in policy.named_parameters()}
        rng = torch.get_rng_state().clone()
        result = run_batch(policy)
        self.assertEqual(result['status'], 'passed', result.get('error'))
        self.assertTrue(result['comparison']['elementwise_equal'])
        self.assertEqual(result['comparison']['compared_elements'], 350)
        self.assertTrue(result['terminal_requires_grad'])
        self.assertTrue(result['invariants_passed'])
        self.assertTrue(result['terminal_forward_restoration']['suffix_callable_object_restored'])
        self.assertTrue(result['restoration']['suffix_callable_object_restored'])
        self.assertEqual(probe.mode_snapshot(policy), modes)
        self.assertEqual({name: parameter.requires_grad for name, parameter in policy.named_parameters()}, flags)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name in probe.GRADIENT_GROUPS[:-1]:
            self.assertGreater(result['gradients']['groups'][name]['finite_nonzero_gradient_tensors'], 0)
            self.assertEqual(result['gradients']['groups'][name]['nonfinite_gradient_tensors'], 0)
        self.assertTrue(result['gradients']['expert_lora_frozen_without_gradients'])
        self.assertEqual(len(result['native_actions_normalized']), 50)
        self.assertEqual(len(result['terminal_actions_normalized']), 50)
        self.assertEqual(len(result['native_actions_normalized'][0]), 7)
        json.dumps(result, allow_nan=False)

    def test_one_ulp_difference_in_last_action_is_failed_with_gradient_evidence(self):
        result = run_batch(Policy(difference=True))
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['comparison']['elementwise_equal'])
        self.assertEqual(result['comparison']['unequal_elements'], 1)
        self.assertEqual(result['comparison']['first_differences'][0]['index'], [0, 49, 6])
        self.assertGreater(result['comparison']['max_abs_difference'], 0.)
        self.assertTrue(result['gradients']['passed'])

    def test_rng_consumption_is_failed_even_when_actions_match(self):
        original = terminal_loss.sample_terminal_actions

        def sampling(*args):
            torch.rand(())
            return original(*args)

        with patch.object(terminal_loss, 'sample_terminal_actions', sampling):
            result = run_batch()
        self.assertTrue(result['comparison']['elementwise_equal'])
        self.assertFalse(result['rng']['terminal_forward']['cpu_unchanged'])
        self.assertEqual(result['status'], 'failed')

    def test_oom_keeps_failed_evidence_and_restores_model_without_retry(self):
        policy = Policy()
        before = probe.mode_snapshot(policy)
        with patch.object(terminal_loss, 'sample_terminal_actions', side_effect=torch.cuda.OutOfMemoryError('24 GiB exhausted')) as sample:
            result = run_batch(policy)
        self.assertEqual(result['status'], 'failed_oom')
        self.assertEqual(sample.call_count, 1)
        self.assertEqual(len(result['native_actions_normalized']), 50)
        self.assertIsNone(result['terminal_actions_normalized'])
        self.assertIn('terminal_forward', result['measurements'])
        self.assertTrue(result['requires_grad_preserved'])
        self.assertTrue(result['explicit_noise_unchanged'])
        self.assertTrue(result['sampler_object_restored'])
        self.assertEqual(probe.mode_snapshot(policy), before)
        json.dumps(result, allow_nan=False)

    def test_nonfinite_and_zero_gradients_and_trainable_expert_lora_fail(self):
        policy = Policy()
        for _, parameter in policy.named_parameters():
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter)
        self.assertTrue(probe.gradient_statistics(torch, training, policy)['passed'])
        vision = next(parameter for name, parameter in policy.named_parameters() if training.is_vision_adapter(name))
        vision.grad.fill_(float('nan'))
        result = probe.gradient_statistics(torch, training, policy)
        self.assertFalse(result['passed'])
        self.assertEqual(result['groups']['vision_lora']['nonfinite_gradient_tensors'], 1)
        json.dumps(result, allow_nan=False)
        for name, parameter in policy.named_parameters():
            if training.is_vision_adapter(name):
                parameter.grad.zero_()
        self.assertFalse(probe.gradient_statistics(torch, training, policy)['required_groups_finite_nonzero'])
        for _, parameter in policy.named_parameters():
            if parameter.requires_grad:
                parameter.grad.fill_(1.)
        expert = next(parameter for name, parameter in policy.named_parameters()
                      if '.gemma_expert.model.' in name and '.lora_' in name)
        expert.requires_grad_(True)
        self.assertFalse(probe.gradient_statistics(torch, training, policy)['expert_lora_frozen_without_gradients'])


class ReportingTests(unittest.TestCase):
    def test_three_case_run_saves_full_actions_gradients_and_original_request_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = inputs()
            data['dataset'] = str(root / 'dataset')
            path = root / 'inputs.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            saved = {'dataset_path': data['dataset'], 'source_provenance': {'episodes': [
                {'source_index': source, 'split': 'train'} for source in (0, 2, 5)]},
                'first_action_weight': 12., 'executed_prefix_weight': 6., 'vertical_weight': 4.,
                'gripper_weight': 4., 'observation_context_profile': probe.FIXED_PROFILE}
            model = SimpleNamespace(torch=torch, policy=Policy(), training=training, info=saved)
            normalization = {'action': {'q01': [-.002] * 7, 'q99': [.002] * 7}}
            args = SimpleNamespace(checkpoint=root / 'checkpoint', inputs=path, output=root / 'evidence.json',
                                   gpu=9, seed=20261116, case_index=None)
            with patch.object(probe, 'AUDITS', root), patch.object(probe, 'load_model',
                    return_value=(model, service, relative, normalization)), \
                    patch.object(probe, 'capture_request_batch', side_effect=lambda *_: batch()) as prepare:
                self.assertEqual(probe.run(args), 0)
            report = json.loads(args.output.read_text(encoding='utf-8'))
            self.assertEqual([value['source_index'] for value in report['cases']], [0, 2, 5])
            self.assertEqual(prepare.call_count, 3)
            self.assertEqual(report['status'], 'passed')
            self.assertLess(args.output.stat().st_size, 200000)
            for value, recorded in zip(report['cases'], data['cases']):
                self.assertTrue(value['request_unchanged'])
                self.assertEqual(value['observation_history'], recorded['request']['observation_history'])
                self.assertEqual(value['task'], recorded['request']['task'])
                self.assertEqual(len(value['native_actions_normalized']), 50)
                self.assertEqual(value['native_actions_normalized'], value['terminal_actions_normalized'])
                self.assertTrue(value['gradients']['passed'])

    def test_memory_over_24_gib_is_measured_and_failed(self):
        cuda = SimpleNamespace(synchronize=Mock(), reset_peak_memory_stats=Mock(),
                               max_memory_allocated=Mock(return_value=probe.MEMORY_LIMIT_BYTES + 1),
                               max_memory_reserved=Mock(return_value=probe.MEMORY_LIMIT_BYTES + 1))
        evidence = {}
        with self.assertRaisesRegex(probe.MemoryLimitExceeded, '24 GiB'):
            with probe.measure(SimpleNamespace(cuda=cuda), SimpleNamespace(type='cuda'), evidence, 'terminal_backward'):
                pass
        self.assertEqual(evidence['terminal_backward']['peak_allocated_bytes'], probe.MEMORY_LIMIT_BYTES + 1)

    def test_load_oom_writes_small_failed_report_and_existing_output_prevents_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'inputs.json'
            path.write_text(json.dumps(inputs()), encoding='utf-8')
            output = root / 'evidence.json'
            args = SimpleNamespace(checkpoint=root / 'checkpoint', inputs=path, output=output,
                                   gpu=9, seed=20261116, case_index=None)
            with patch.object(probe, 'AUDITS', root), patch.object(probe, 'load_model',
                    side_effect=torch.cuda.OutOfMemoryError('24 GiB exhausted')) as load:
                self.assertEqual(probe.run(args), 1)
                self.assertEqual(load.call_count, 1)
                written = output.read_text(encoding='utf-8')
                report = json.loads(written)
                self.assertEqual(report['status'], 'failed_oom')
                self.assertEqual(report['optimizer_steps'], 0)
                self.assertEqual(report['robot_commands'], 0)
                self.assertNotIn('rgb_jpeg_base64', written)
                with self.assertRaisesRegex(ValueError, 'new file'):
                    probe.run(args)
                self.assertEqual(load.call_count, 1)
                self.assertEqual(output.read_text(encoding='utf-8'), written)


if __name__ == '__main__':
    unittest.main()
