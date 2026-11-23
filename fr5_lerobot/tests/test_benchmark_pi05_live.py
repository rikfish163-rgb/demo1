"""CPU-only checks for live-request replay and action comparison."""
from contextlib import nullcontext
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from fr5_lerobot.diagnostics import benchmark_pi05_live as benchmark


class LinearKinematics:
    def points(self, joints):
        return np.asarray(joints)[:, :3] * 1000.


def archive_rows():
    digest = 'a' * 64
    rows = [{'type': 'session', 'checkpoint_sha256': digest}]
    for i in range(5):
        identifier = f'request-{i}'
        rows.append({'type': 'prediction', 'index': i, 'request': {'id': identifier},
                     'reply': {'id': identifier, 'checkpoint_sha256': digest,
                               'wire_action_representation': 'absolute-joint-position-v1',
                               'model_action_representation': 'joint-delta-from-anchor-v1',
                               'sampling': {'denoising_steps': 10, 'cuda_rng_state': [1, 2, 3]},
                               'actions': [[0.] * 7 for _ in range(50)]}})
    rows.append({'type': 'complete', 'predictions': 5})
    return rows


class BenchmarkTests(unittest.TestCase):
    def test_action_difference_uses_same_step_and_reports_prefix_and_full(self):
        baseline = np.zeros((50, 7))
        candidate = baseline.copy()
        candidate[:10, 0] = .002
        candidate[:10, 6] = .1
        result = benchmark.action_difference(baseline, candidate, LinearKinematics())
        self.assertAlmostEqual(result['prefix10_grasp_mean_mm'], 2.)
        self.assertAlmostEqual(result['prefix10_grasp_max_mm'], 2.)
        self.assertAlmostEqual(result['full50_grasp_mean_mm'], .4)
        self.assertAlmostEqual(result['prefix10_gripper_max_fraction'], .1)
        for bad in (candidate[:49], np.full((50, 7), np.nan)):
            with self.assertRaisesRegex(ValueError, 'finite 50 by 7'):
                benchmark.action_difference(baseline, bad, LinearKinematics())

    def test_archived_request_id_checkpoint_rng_and_complete_are_required(self):
        rows = archive_rows()
        def load(items):
            payload = ''.join(json.dumps(row) + '\n' for row in items)
            with patch('pathlib.Path.open', return_value=io.StringIO(payload)):
                return benchmark.load_requests('/data/archive.jsonl', 'a' * 64)
        self.assertEqual(len(load(rows)), 5)
        with self.assertRaisesRegex(ValueError, 'complete'):
            load(rows[:-1])
        rows[1]['reply']['id'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'request, model output or RNG'):
            load(rows)

    def test_timed_inference_restores_saved_rng_and_preserves_wire_actions(self):
        state = [float(i) for i in range(6)] + [.7, 1.]
        request = {'id': 'frame', 'state': state, 'observation_history': [None] * 6}
        cuda = SimpleNamespace(synchronize=Mock(), set_rng_state=Mock())
        fake_torch = SimpleNamespace(cuda=cuda, tensor=torch.tensor, uint8=torch.uint8,
                                     inference_mode=torch.inference_mode, autocast=lambda *a, **k: nullcontext(),
                                     bfloat16=torch.bfloat16, isfinite=torch.isfinite)
        config = SimpleNamespace(num_inference_steps=10)
        normalized = torch.arange(350, dtype=torch.float32).reshape(1, 50, 7)
        policy = SimpleNamespace(config=config, predict_action_chunk=Mock(return_value=normalized))
        model = SimpleNamespace(torch=fake_torch, policy=policy, training=object(), pre=object(),
                                post=lambda x: x, relative=object(),
                                info={'image_profile': 'full-and-workspace-v1',
                                      'observation_context_profile': 'state-history-physical-v1',
                                      'action_representation': 'joint-delta-from-anchor-v1'})
        service = SimpleNamespace(
            validate_request=Mock(return_value=('frame', 'task', state, object())),
            prepare_request_batch=Mock(return_value={'observation.state': torch.zeros((1, 32))}),
            actions_for_wire=Mock(side_effect=lambda values, anchor, relative, **kw: values + 1),
        )
        actions, times = benchmark.timed_inference(model, service, request, [4, 5, 6], 5)
        self.assertEqual(config.num_inference_steps, 10)
        self.assertEqual(actions, (normalized[0] + 1).tolist())
        self.assertEqual(cuda.set_rng_state.call_args.args[0].tolist(), [4, 5, 6])
        self.assertEqual(service.actions_for_wire.call_args.args[1], state)
        self.assertEqual(service.prepare_request_batch.call_args.args[-1], request['observation_history'])
        self.assertTrue(all(value >= 0 for value in times.values()))


if __name__ == '__main__':
    unittest.main()
