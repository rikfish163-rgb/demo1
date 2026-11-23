"""Verify that the integration probe preserves the observation and initial noise."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'diagnostics'))
import probe_cup_denoising_steps as probe


class Model:
    def __init__(self, change_image=False):
        self.torch = torch
        self.policy = SimpleNamespace(config=SimpleNamespace(num_inference_steps=7),
                                      model=SimpleNamespace(sample_noise=lambda shape, device: torch.randn(shape)))
        self.training = SimpleNamespace(predict_action_chunk=lambda policy, batch:
            self.policy.model.sample_noise((1, 50, 32), 'cpu'))
        self.change_image = change_image
        self.observed_counts = []

    def reset(self):
        pass

    def predict(self, request):
        count = self.policy.config.num_inference_steps
        self.observed_counts.append(count)
        image = torch.zeros(1, 3, 2, 2)
        if self.change_image and count == 20:
            image[0, 0, 0, 0] = 1.
        batch = {probe.CLOCK: torch.zeros(1, 25), probe.FIXED: True,
                 'observation.images.front': image,
                 'observation.state': torch.tensor([request['cup_relative_state'] + [0.] * 24])}
        state = torch.get_rng_state().tolist()
        noise = self.training.predict_action_chunk(self.policy, batch)
        return {'actions': (noise[0, :, :7] / count).tolist(), 'inference_seconds': .01,
                'sampling': {'cuda_rng_state': state, 'denoising_steps': count}}


class ProbeTests(unittest.TestCase):
    def test_pairs_exact_observation_and_noise_while_changing_integration_count(self):
        model = Model()
        original_predict, original_noise = model.training.predict_action_chunk, model.policy.model.sample_noise
        request = {'cup_relative_state': [0.] * 6 + [1., 1.]}
        replies, elapsed, audit = probe.paired_predict(model, request, 20261105, [10, 20])
        self.assertEqual(model.observed_counts, [10, 20])
        self.assertEqual(model.policy.config.num_inference_steps, 7)
        self.assertIs(model.training.predict_action_chunk, original_predict)
        self.assertIs(model.policy.model.sample_noise, original_noise)
        self.assertEqual(len(elapsed), 2)
        self.assertTrue(audit['initial_noise_elementwise_equal'])
        self.assertTrue(audit['preprocessed_tensors_elementwise_equal'])
        self.assertEqual(torch.tensor(replies[0]['actions']).shape, (50, 7))
        self.assertTrue(torch.equal(torch.tensor(replies[0]['actions']) / 2,
                                    torch.tensor(replies[1]['actions'])))

    def test_rejects_observation_changes_and_restores_sampling_configuration(self):
        model = Model(change_image=True)
        original_predict, original_noise = model.training.predict_action_chunk, model.policy.model.sample_noise
        with self.assertRaisesRegex(ValueError, 'preprocessed observations differ'):
            probe.paired_predict(model, {'cup_relative_state': [0.] * 8}, 1, [10, 20])
        self.assertEqual(model.policy.config.num_inference_steps, 7)
        self.assertIs(model.training.predict_action_chunk, original_predict)
        self.assertIs(model.policy.model.sample_noise, original_noise)

    def test_all_open_static_drift_is_distinct_from_a_predicted_close(self):
        case = {'case_id': 'original', 'source_index': 0, 'row_index': 243,
                'source_frame_index': 245, 'kind': 'first_close_offset',
                'targets_camera_aligned': [{'cup_relative_action': [0.] * 6 + [1.]} for _ in range(3)]}
        request = {'cup_relative_state': [0.] * 6 + [1., 1.], 'cup_timing': {'phase': 'approach'}}
        reply = {'actions': [[.003, 0., 0., 0., 0., 0., 1.] for _ in range(50)]}
        value = probe.score(case, request, reply)
        self.assertTrue(value['steady_full_open_drift_over_2mm'])
        changed = deepcopy(reply)
        changed['actions'][0][6] = .8
        value = probe.score(case, request, changed)
        self.assertTrue(value['steady_drift_over_2mm'])
        self.assertFalse(value['steady_full_open_drift_over_2mm'])
        self.assertTrue(value['early_close_against_recorded_labels'])

    def test_aliases_do_not_double_count_original_observations(self):
        metrics = {'source': 5, 'row_index': 2780, 'kind': 'nearest_valid_open_before_first_close',
                   'actual_move_mm': 0., 'pred_move_mm': 3.,
                   'steady_drift_over_2mm': True, 'steady_full_open_drift_over_2mm': True,
                   'opposed_recorded_move': False, 'early_close_against_recorded_labels': False,
                   'endpoint_error_mm': 3.}
        row = {'case_id': 'first', 'seed': 1, 'metrics': metrics,
               'reply': {'actions': [[0.] * 7] * 50, 'inference_seconds': .3},
               'predict_elapsed_seconds': .4}
        alias = deepcopy(row)
        alias['case_id'] = 'second'
        alias['metrics']['kind'] = 'first_close_offset'
        summary = probe.summarize([row, alias])
        self.assertEqual(summary['draws_with_aliases'], 2)
        self.assertEqual(summary['unique_source_row_seed_draws'], 1)
        self.assertEqual(summary['steady_full_open_drift_over_2mm_draws'], 1)
        self.assertEqual(summary['selector_status'], 'not_applied')


if __name__ == '__main__':
    unittest.main()
