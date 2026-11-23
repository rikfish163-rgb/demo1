"""CPU checks for FR5 joint-stream RTC guidance."""
import unittest

import torch
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import NormalizerProcessorStep

from fr5_lerobot.training import relative_stream_actions as relative
from fr5_lerobot.training.stream_rtc import FR5RTCProcessor, normalize_absolute_prefix


class StreamRTCTests(unittest.TestCase):
    def test_full_vjp_is_computed_inside_no_grad(self):
        processor = FR5RTCProcessor(RTCConfig(enabled=True, execution_horizon=2))
        latent = torch.zeros((1, 2, 32))
        prefix = torch.ones((1, 2, 7))
        seen = []

        def denoiser(value):
            seen.append(torch.is_grad_enabled() and value.requires_grad)
            return 2 * value

        with torch.no_grad():
            result = processor.denoise_step(latent, prefix, 2, .5, denoiser)

        self.assertEqual(seen, [True])
        torch.testing.assert_close(result, torch.zeros_like(latent))

    def test_seven_dimensional_error_keeps_cross_dimension_vjp(self):
        processor = FR5RTCProcessor(RTCConfig(enabled=True, execution_horizon=1))
        latent = torch.zeros((1, 2, 32))
        prefix = torch.zeros((1, 1, 7))
        prefix[0, 0, 0] = 1.

        def denoiser(value):
            return torch.cat((2 * value[..., 7:8], torch.zeros_like(value[..., 1:])), dim=-1)

        with torch.no_grad():
            result = processor.denoise_step(latent, prefix, 1, .5, denoiser)

        expected = torch.zeros_like(latent)
        expected[0, 0, 0] = -2.
        expected[0, 0, 7] = 2.
        torch.testing.assert_close(result, expected)

    def test_short_prefix_never_guides_padding_or_later_steps(self):
        processor = FR5RTCProcessor(RTCConfig(enabled=True, execution_horizon=10))
        latent = torch.ones((1, 4, 32))
        prefix = torch.zeros((1, 2, 7))

        with torch.no_grad():
            result = processor.denoise_step(latent, prefix, 1, .5, lambda value: value * 0.)

        expected = torch.zeros_like(latent)
        expected[0, 0, :7] = 2.
        expected[0, 1, :7] = 1.
        torch.testing.assert_close(result, expected)
        with torch.no_grad():
            delayed = processor.denoise_step(latent, prefix, 4, .5, lambda value: value * 0.)
        expected[0, 1, :7] = 2.
        torch.testing.assert_close(delayed, expected)

    def test_native_attention_schedules(self):
        schedules = {
            RTCAttentionSchedule.ZEROS: [2., 0., 0., 0.],
            RTCAttentionSchedule.ONES: [2., 2., 2., 2.],
            RTCAttentionSchedule.LINEAR: [2., 1.5, 1., .5],
            RTCAttentionSchedule.EXP: [2., .9751019869, .3775406688, .0826480883],
        }
        for schedule, values in schedules.items():
            with self.subTest(schedule=schedule):
                config = RTCConfig(enabled=True, execution_horizon=4, prefix_attention_schedule=schedule)
                processor = FR5RTCProcessor(config)
                latent = torch.ones((1, 6, 32))
                prefix = torch.zeros((1, 4, 7))
                with torch.no_grad():
                    result = processor.denoise_step(latent, prefix, 1, .5, lambda value: value * 0.)
                expected = torch.zeros_like(latent)
                expected[0, :4, :7] = torch.tensor(values)[:, None]
                torch.testing.assert_close(result, expected)

    def test_native_guidance_weight_and_cap(self):
        processor = FR5RTCProcessor(RTCConfig(enabled=True, execution_horizon=1,
                                              max_guidance_weight=3.))
        for time, expected in ((1., -3.), (.5, -2.), (.1, -3.)):
            with self.subTest(time=time), torch.no_grad():
                result = processor.denoise_step(torch.zeros((1, 1, 32)), torch.ones((1, 1, 7)),
                                                1, time, lambda value: value * 0.)
                torch.testing.assert_close(result[0, 0, :7], torch.full((7,), expected))
                torch.testing.assert_close(result[0, 0, 7:], torch.zeros(25))
        processor = FR5RTCProcessor(RTCConfig(enabled=True, execution_horizon=1,
                                              max_guidance_weight=10.))
        with torch.no_grad():
            uncapped = processor.denoise_step(torch.zeros((1, 1, 32)), torch.ones((1, 1, 7)),
                                              1, .25, lambda value: value * 0.)
        torch.testing.assert_close(uncapped[0, 0, :7], torch.full((7,), -10. / 3.))

    def test_prefix_batch_dtype_and_device_are_explicit(self):
        processor = FR5RTCProcessor(RTCConfig(enabled=True, execution_horizon=1))
        latent = torch.zeros((2, 1, 32))
        prefix = torch.stack((torch.ones((1, 7)), torch.full((1, 7), 2.)))
        with torch.no_grad():
            result = processor.denoise_step(latent, prefix, 1, .5, lambda value: value * 0.)
        torch.testing.assert_close(result[:, 0, :7], torch.tensor([[-2.] * 7, [-4.] * 7]))
        for invalid in (prefix[0], prefix.double(), torch.empty((2, 1, 7), device='meta')):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype, device=invalid.device):
                with self.assertRaises(ValueError):
                    processor.denoise_step(latent, invalid, 1, .5, lambda value: value * 0.)

    def test_unbatched_and_no_prefix_paths_preserve_shapes(self):
        processor = FR5RTCProcessor(RTCConfig(enabled=True, execution_horizon=1))
        latent = torch.zeros((2, 32))
        with torch.no_grad():
            result = processor.denoise_step(latent, torch.ones((1, 7)), 1, .5, lambda value: value * 0.)
        expected = torch.zeros_like(latent)
        expected[0, :7] = -2.
        torch.testing.assert_close(result, expected)
        with torch.inference_mode():
            torch.testing.assert_close(processor.denoise_step(latent, None, 0, .5, lambda value: value + 1.),
                                       torch.ones_like(latent))
            with self.assertRaisesRegex(RuntimeError, 'no_grad'):
                processor.denoise_step(latent, torch.ones((1, 7)), 1, .5, lambda value: value * 0.)

    def test_reanchor_uses_action_normalizer_and_preserves_absolute_gripper(self):
        normalizer = NormalizerProcessorStep(
            features={'action': {'type': 'ACTION', 'shape': [7]}},
            norm_map={'ACTION': 'QUANTILES'},
            stats={'action': {'q01': [-1.] * 6 + [0.], 'q99': [1.] * 7}},
        )
        state = torch.tensor([[1., 2., 3., 4., 5., 6., .8, 1.]])
        absolute = torch.tensor([[[1.25, 1.5, 3., 3.75, 5.5, 5.5, .75],
                                  [.5, 2.25, 3.25, 4., 5.5, 6.25, 1.]]])
        saved_state, saved_actions = state.clone(), absolute.clone()
        expected = torch.tensor([[[.25, -.5, 0., -.25, .5, -.5, .5],
                                  [-.5, .25, .25, 0., .5, .25, 1.]]])

        result = normalize_absolute_prefix(absolute, state, relative, normalizer)

        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(normalize_absolute_prefix(absolute[0], state[0], relative, normalizer),
                                   expected[0])
        self.assertTrue(torch.equal(absolute, saved_actions))
        self.assertTrue(torch.equal(state, saved_state))
        for invalid_state in (state[0], state.double(), torch.empty((1, 8), device='meta')):
            with self.subTest(shape=invalid_state.shape, dtype=invalid_state.dtype, device=invalid_state.device):
                with self.assertRaises(ValueError):
                    normalize_absolute_prefix(absolute, invalid_state, relative, normalizer)


if __name__ == '__main__':
    unittest.main()
