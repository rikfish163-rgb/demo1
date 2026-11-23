"""Checks for sampled translation supervision and AdamW moment transfers."""

import sys
from copy import deepcopy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import stream_terminal_loss as terminal_loss
import train_stream_pi05 as training


class OptimizerMomentTests(unittest.TestCase):
    def test_uninitialized_optimizer_stays_empty(self):
        parameter = torch.nn.Parameter(torch.ones(3))
        optimizer = torch.optim.AdamW([parameter])
        for offload in (True, False):
            self.assertEqual(terminal_loss.move_optimizer_moments(optimizer, offload=offload),
                             {'moment_bytes': 0, 'devices': []})
            self.assertEqual(len(optimizer.state), 0)

    def test_cpu_noop_keeps_moments_step_parameters_and_gradients(self):
        parameter = torch.nn.Parameter(torch.tensor([.2, -.4, .6]))
        optimizer = torch.optim.AdamW([parameter])
        parameter.square().sum().backward()
        optimizer.step()
        state = optimizer.state[parameter]
        originals = dict(state)
        values = {name: value.clone() for name, value in state.items()}
        gradient = parameter.grad
        gradient_value = gradient.clone()
        parameter_value = parameter.detach().clone()
        expected_bytes = sum(state[name].numel() * state[name].element_size()
                             for name in ('exp_avg', 'exp_avg_sq'))
        with patch.object(torch.Tensor, 'to', side_effect=AssertionError('CPU state must not be copied')):
            for offload in (True, False, True, False):
                self.assertEqual(terminal_loss.move_optimizer_moments(optimizer, offload=offload),
                                 {'moment_bytes': expected_bytes, 'devices': ['cpu']})
        self.assertEqual(state.keys(), originals.keys())
        for name, original in originals.items():
            self.assertIs(state[name], original)
            torch.testing.assert_close(state[name], values[name], rtol=0, atol=0)
        self.assertIs(parameter.grad, gradient)
        torch.testing.assert_close(parameter.grad, gradient_value, rtol=0, atol=0)
        torch.testing.assert_close(parameter, parameter_value, rtol=0, atol=0)

    def _assert_multistep_adamw_equivalence(self, devices):
        initial = (torch.tensor([.2, -.4, .6], dtype=torch.float64),
                   torch.tensor([[.1, -.3], [.7, -.2]], dtype=torch.float32))
        parameters = [torch.nn.Parameter(value.to(device)) for value, device in zip(initial, devices)]
        resident_parameters = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
        options = dict(lr=.02, betas=(.8, .9), eps=1e-7, weight_decay=.03)
        optimizer = torch.optim.AdamW(parameters, **options)
        resident = torch.optim.AdamW(resident_parameters, **options)
        expected_bytes = 2 * sum(parameter.numel() * parameter.element_size() for parameter in parameters)
        for iteration in range(6):
            optimizer.zero_grad(set_to_none=True)
            resident.zero_grad(set_to_none=True)
            for parameter, baseline in zip(parameters, resident_parameters):
                (parameter.sin() + .1 * (iteration + 1) * parameter.square()).sum().backward()
                (baseline.sin() + .1 * (iteration + 1) * baseline.square()).sum().backward()
            gradients = [(parameter.grad, parameter.grad.clone()) for parameter in parameters]
            steps = [optimizer.state.get(parameter, {}).get('step') for parameter in parameters]
            step_values = [None if step is None else step.clone() for step in steps]
            offloaded = terminal_loss.move_optimizer_moments(optimizer, offload=True)
            self.assertEqual(offloaded, {'moment_bytes': expected_bytes if iteration else 0,
                                         'devices': ['cpu'] if iteration else []})
            for parameter, (gradient, value) in zip(parameters, gradients):
                self.assertIs(parameter.grad, gradient)
                torch.testing.assert_close(parameter.grad, value, rtol=0, atol=0)
            for parameter, baseline in zip(parameters, resident_parameters):
                (.03 * (iteration + 2) * parameter.square()).sum().backward()
                (.03 * (iteration + 2) * baseline.square()).sum().backward()
            gradients = [(parameter.grad, parameter.grad.clone()) for parameter in parameters]
            restored = terminal_loss.move_optimizer_moments(optimizer, offload=False)
            self.assertEqual(restored, {'moment_bytes': expected_bytes if iteration else 0,
                                       'devices': sorted({str(parameter.device) for parameter in parameters})
                                                  if iteration else []})
            for parameter, (gradient, value), step, step_value in zip(parameters, gradients, steps, step_values):
                self.assertIs(parameter.grad, gradient)
                torch.testing.assert_close(parameter.grad, value, rtol=0, atol=0)
                if step is not None:
                    self.assertIs(optimizer.state[parameter]['step'], step)
                    torch.testing.assert_close(step, step_value, rtol=0, atol=0)
                    for name in ('exp_avg', 'exp_avg_sq'):
                        self.assertEqual(optimizer.state[parameter][name].device, parameter.device)
            optimizer.step()
            resident.step()
            steps = [optimizer.state[parameter]['step'] for parameter in parameters]
            step_values = [step.clone() for step in steps]
            self.assertEqual(terminal_loss.move_optimizer_moments(optimizer, offload=True),
                             {'moment_bytes': expected_bytes, 'devices': ['cpu']})
            for parameter, baseline, step, step_value in zip(parameters, resident_parameters, steps, step_values):
                torch.testing.assert_close(parameter, baseline, rtol=0, atol=0)
                self.assertIs(optimizer.state[parameter]['step'], step)
                torch.testing.assert_close(step, step_value, rtol=0, atol=0)
                self.assertEqual(optimizer.state[parameter].keys(), resident.state[baseline].keys())
                for name, value in optimizer.state[parameter].items():
                    torch.testing.assert_close(value.cpu(), resident.state[baseline][name].cpu(), rtol=0, atol=0)

    def test_cpu_multistep_adamw_matches_resident_parameters_and_state(self):
        self._assert_multistep_adamw_equivalence((torch.device('cpu'), torch.device('cpu')))

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA for moment round trips')
    def test_cuda_multistep_adamw_matches_resident_parameters_and_state(self):
        self._assert_multistep_adamw_equivalence((torch.device('cpu'), torch.device('cuda:0')))


class SamplingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(.2))
        self.dropout = torch.nn.Dropout(.5)

    def embed_suffix(self, actions, timestep):
        return self.dropout(actions)

    def denoise_step(self, *, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix = self.embed_suffix(x_t, timestep)
        return self.gain * suffix + past_key_values

    @torch.no_grad()
    def sample_actions(self, images, image_masks, tokens, attention_mask, *, noise, num_steps):
        actions = noise
        for step in range(num_steps):
            velocity = self.denoise_step(prefix_pad_masks=None, past_key_values=images[0].mean(),
                                         x_t=actions, timestep=torch.tensor(1. - step / num_steps))
            actions = actions - velocity / num_steps
        return actions


class SamplingPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SamplingModel()
        self.prefix = torch.nn.Parameter(torch.tensor(.1))
        self.config = SimpleNamespace(num_inference_steps=10)

    def _preprocess_images(self, batch):
        return [batch['image'] * self.prefix], []


def sampling_batch():
    clock = torch.zeros(1, 25)
    clock[0, :2] = torch.tensor([.4, 1.])
    return {'image': torch.ones(1, 2, 2),
            'observation.language.tokens': torch.ones(1, 2, dtype=torch.long),
            'observation.language.attention_mask': torch.ones(1, 2, dtype=torch.bool),
            'observation.cup_action_clock': clock,
            'observation.fixed_cup_action_clock': True}


def inactive_boundary(prefix=3):
    return (torch.zeros(1, prefix), torch.ones(1, prefix), torch.tensor([[0., 0., 1.]]),
            torch.zeros(1), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool), torch.zeros(1))


class TerminalSamplingTests(unittest.TestCase):
    def test_step_checkpoint_keeps_native_parameter_gradients(self):
        policy = SamplingPolicy().train()
        baseline = deepcopy(policy)
        batch = sampling_batch()
        noise = torch.arange(50 * 32, dtype=torch.float32).reshape(1, 50, 32) / 100
        denoise = policy.model.denoise_step
        actual = terminal_loss.sample_terminal_actions(policy, batch, noise)
        with terminal_loss.sampling_mode(baseline), training.action_clock_condition(baseline, batch):
            images, masks = baseline._preprocess_images(batch)
            expected = baseline.model.sample_actions.__wrapped__(baseline.model, images, masks,
                batch['observation.language.tokens'], batch['observation.language.attention_mask'],
                noise=noise, num_steps=10)[..., :7]
        actual.square().mean().backward()
        expected.square().mean().backward()
        for left, right in zip(policy.parameters(), baseline.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-6, atol=1e-7)
        self.assertEqual(policy.model.denoise_step, denoise)

    def test_native_output_noise_and_modes_survive_backward_recomputation(self):
        policy = SamplingPolicy().train()
        policy.model.dropout.eval()
        states = [module.training for module in policy.modules()]
        batch = sampling_batch()
        noise = torch.arange(50 * 32, dtype=torch.float32).reshape(1, 50, 32) / 100
        noise.requires_grad_()
        original_noise = noise.detach().clone()
        rng = torch.get_rng_state().clone()
        suffix = policy.model.embed_suffix
        with terminal_loss.sampling_mode(policy), training.action_clock_condition(policy, batch):
            images, masks = policy._preprocess_images(batch)
            expected = policy.model.sample_actions(images, masks,
                batch['observation.language.tokens'], batch['observation.language.attention_mask'],
                noise=noise, num_steps=10)[..., :7]
        result = terminal_loss.sample_terminal_actions(policy, batch, noise)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertEqual(tuple(result.shape), (1, 50, 7))
        self.assertEqual([module.training for module in policy.modules()], states)
        result.square().mean().backward()
        for parameter in (policy.prefix, policy.model.gain):
            self.assertTrue(torch.isfinite(parameter.grad))
            self.assertGreater(parameter.grad.abs().item(), 0.)
        self.assertIsNone(noise.grad)
        self.assertTrue(torch.equal(noise.detach(), original_noise))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual([module.training for module in policy.modules()], states)
        self.assertEqual(policy.model.embed_suffix, suffix)

    def test_sampler_error_restores_mixed_module_modes_and_clock_hook(self):
        policy = SamplingPolicy().train()
        policy.model.dropout.eval()
        states = [module.training for module in policy.modules()]
        suffix = policy.model.embed_suffix
        def fail(*args, **kwargs):
            raise RuntimeError('sampler unavailable')
        sampler = SimpleNamespace(__wrapped__=fail)
        with patch.object(policy.model, 'sample_actions', sampler):
            with self.assertRaisesRegex(RuntimeError, 'sampler unavailable'):
                terminal_loss.sample_terminal_actions(policy, sampling_batch(), torch.zeros(1, 50, 32))
        self.assertEqual([module.training for module in policy.modules()], states)
        self.assertEqual(policy.model.embed_suffix, suffix)


class TerminalTranslationTests(unittest.TestCase):
    def test_separate_flow_backward_matches_combined_weighted_gradients(self):
        from fr5_lerobot.tests.test_stream_training import ToyFlow
        model = ToyFlow()
        model.action_out_proj = torch.nn.Linear(7, 7)
        model.sample_time = lambda size, device: torch.ones(size, device=device)
        def flow(_images, _masks, _tokens, _attention, actions, noise, times):
            velocity = model.action_out_proj(torch.ones_like(actions))
            return (velocity - (noise - actions)).square()
        model.forward = flow
        policy = SimpleNamespace(model=model, _preprocess_images=lambda batch: ([], []),
                                 prepare_action=lambda batch: batch['action'])
        valid = torch.tensor([[True, True, True] + [False] * 47])
        batch = {**sampling_batch(), 'action': torch.zeros(1, 50, 7),
                 'cup_boundary': inactive_boundary()}
        kwargs = dict(prefix_steps=3, first_action_weight=12., prefix_weight=6.,
                      vertical_weight=4., gripper_weight=4., cup_action_scale=torch.ones(7) * .001,
                      coupled_support=True, support_focus=True, xyz_loss_mode='terminal')
        def sample(*args):
            return model.action_out_proj(torch.ones(1, 50, 7))
        with patch.object(terminal_loss, 'sample_terminal_actions', side_effect=sample):
            combined = training.masked_flow_loss(policy, batch, valid, **kwargs)
            (combined * .25).backward()
            gradients = [p.grad.clone() for p in model.parameters()]
            model.zero_grad(set_to_none=True)
            separate = training.masked_flow_loss(policy, batch, valid, backward_scale=.25, **kwargs)
            (separate * .25).backward()
        torch.testing.assert_close(combined, separate)
        for parameter, expected in zip(model.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, expected, rtol=1e-6, atol=1e-7)

    def test_same_mm_huber_as_clean_xyz_with_padding_and_gripper_mask(self):
        scale = torch.tensor([.001, .002, .003, 1., 1., 1., .12])
        error = torch.zeros(1, 3, 7, requires_grad=True)
        with torch.no_grad():
            error[0, :, 0] = torch.tensor([2., -1., 100.])
            error[0, :, 6] = 10.
        weights = torch.ones_like(error)
        weights[0, :, 0] = torch.tensor([12., 6., 0.])
        weights[0, :, 6] = 0.
        clock = torch.zeros(1, 25)
        kwargs = dict(support_focus=True, coupled_support=True)
        clean = training.cup_boundary_penalty(error, inactive_boundary(), clock, scale, weights, **kwargs)
        boundary = training.cup_boundary_penalty(error, inactive_boundary(), clock, scale, weights,
                                                include_xyz=False, **kwargs)
        actual = terminal_loss.terminal_xyz_penalty(error, scale, weights)
        torch.testing.assert_close(actual, clean - boundary)
        actual.backward()
        self.assertGreater(error.grad[0, 0, 0], 0.)
        self.assertLess(error.grad[0, 1, 0], 0.)
        self.assertEqual(error.grad[0, 2].abs().sum(), 0.)
        self.assertEqual(error.grad[..., 3:].abs().sum(), 0.)

    def test_quarter_mm_tolerance_has_zero_penalty_and_gradient(self):
        error = torch.zeros(1, 3, 7, requires_grad=True)
        with torch.no_grad():
            error[0, :, 0] = torch.tensor([.2, .1, .06])
        value = terminal_loss.terminal_xyz_penalty(error, torch.ones(7) * .001, torch.ones_like(error))
        value.backward()
        self.assertEqual(value.item(), 0.)
        self.assertEqual(error.grad.abs().sum().item(), 0.)

    def test_terminal_replaces_only_xyz_and_preserves_flow_and_boundary(self):
        from fr5_lerobot.tests.test_stream_training import ToyFlow
        model = ToyFlow()
        model.action_out_proj = torch.nn.Identity()
        def flow(_images, _masks, _tokens, _attention, actions, noise, times):
            velocity = model.action_out_proj(torch.zeros_like(actions))
            return (velocity - (noise - actions)).square()
        model.forward = flow
        model.sample_time = lambda size, device: torch.ones(size, device=device)
        policy = SimpleNamespace(model=model, _preprocess_images=lambda batch: ([], []),
                                 prepare_action=lambda batch: batch['action'])
        valid = torch.tensor([[True, True, True] + [False] * 47])
        batch = {**sampling_batch(), 'action': torch.zeros(1, 50, 7),
                 'cup_boundary': inactive_boundary(),
                 'cup_pickup_wait_mask': torch.ones(1, 50, dtype=torch.bool)}
        batch['action'][0, :3, 0] = 2.
        scale = torch.ones(7) * .001
        kwargs = dict(prefix_steps=3, first_action_weight=12., prefix_weight=6.,
                      vertical_weight=4., gripper_weight=4., cup_action_scale=scale,
                      coupled_support=True, support_focus=True)
        clean_parts, terminal_parts = {}, {}
        clean = training.masked_flow_loss(policy, batch, valid, loss_parts=clean_parts, **kwargs)
        sampled = torch.ones(1, 50, 7, requires_grad=True)
        with patch.object(terminal_loss, 'sample_terminal_actions', return_value=sampled) as sampler:
            actual = training.masked_flow_loss(policy, batch, valid, xyz_loss_mode='terminal',
                                              loss_parts=terminal_parts, **kwargs)
        sampler.assert_called_once()
        self.assertEqual(terminal_parts['flow'], clean_parts['flow'])
        self.assertAlmostEqual(terminal_parts['boundary'], clean_parts['boundary'])
        self.assertGreater(clean_parts['xyz_clean'], 0.)
        self.assertGreater(terminal_parts['xyz_terminal'], 0.)
        self.assertAlmostEqual(clean.item(), sum(clean_parts.values()), places=6)
        self.assertAlmostEqual(actual.item(), sum(terminal_parts.values()), places=6)
        self.assertAlmostEqual(actual.item() - clean.item(),
                               terminal_parts['xyz_terminal'] - clean_parts['xyz_clean'], places=6)
        actual.backward()
        self.assertTrue(bool((sampled.grad[0, :3, 0] < 0).all()))
        self.assertTrue(bool((sampled.grad[0, :3, 1:3] > 0).all()))
        self.assertEqual(sampled.grad[0, 3:].abs().sum(), 0.)
        self.assertEqual(sampled.grad[..., 3:].abs().sum(), 0.)

    def test_resume_preserves_loss_mode_and_warm_start_can_change_it(self):
        self.assertEqual(training.resolve_xyz_loss_mode(None), 'clean')
        self.assertEqual(training.resolve_xyz_loss_mode(None, parent_report={'xyz_loss_mode': 'terminal'},
                                                        resume=True), 'terminal')
        self.assertEqual(training.resolve_xyz_loss_mode('terminal', parent_report={}), 'terminal')
        with self.assertRaisesRegex(ValueError, 'Resume'):
            training.resolve_xyz_loss_mode('terminal', parent_report={}, resume=True)


if __name__ == '__main__':
    unittest.main()
