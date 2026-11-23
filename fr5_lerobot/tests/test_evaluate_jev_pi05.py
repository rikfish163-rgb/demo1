"""CPU tests for offline source selection, causal inputs and label metrics."""
import contextlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('evaluate_jev_pi05', HERE.parent / 'training/evaluate_jev_pi05.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def fixtures():
    rows, episodes = [], {0: list(range(6)), 1: list(range(6, 9))}
    targets = (1., 1., .3, .3, 1., 1., 1., 1., 1.)
    for i, gripper in enumerate(targets):
        source = 0 if i < 6 else 1
        start = 0 if source == 0 else 6
        ref = {'kind': 'gripper', 'executed': True, 'source': 'wire_command',
               'sequence': i, 'target_degrees_percent': [0] * 6 + [gripper * 100]}
        rows.append({'_physical_source': str(source), '_split': 'train' if source == 0 else 'validation',
                     '_grid_time': float(100 * (source + 1) + i - start),
                     '_camera_time': float(100 * (source + 1) + i - start - .01),
                     '_source_capture_frame': i + 10, '_gripper_target_ref': ref,
                     '_physical_observation_history': [None] * 6,
                     '_recorded_hold_remaining_s': 999.,
                     'observation.state': [0.] * 6 + [1., 1.], 'action': [0.] * 6 + [gripper]})
    proof = {'sources': [
        {'source_index': 0, 'split': 'train', 'task_instruction': 'pick',
         'hold_records': [{'start_monotonic': 102., 'end_monotonic': 103.}]},
        {'source_index': 1, 'split': 'validation', 'task_instruction': 'pick', 'hold_records': []}]}
    return rows, episodes, proof


class SelectionTests(unittest.TestCase):
    def test_uniform_events_holds_and_episode_tail(self):
        rows, episodes, proof = fixtures()
        cases, missing = evaluation.select_cases(rows, episodes, proof, 3)
        self.assertEqual(missing, [])
        uniform = {c['row_index'] for c in cases if 'uniform' in c['tags']}
        self.assertEqual(uniform, {0, 2, 5, 6, 7, 8})
        by_index = {c['row_index']: c for c in cases}
        for index, tag in ((1, 'close_before'), (2, 'close_at'), (3, 'open_before'), (4, 'open_at'),
                           (2, 'hold_start'), (3, 'hold_end')):
            self.assertIn(tag, by_index[index]['tags'])
        target, mask = evaluation.training.chunk_for_anchor(rows, episodes[0], 5)
        self.assertEqual(sum(mask), 1)
        self.assertEqual(target[0], rows[5]['action'])
        self.assertNotEqual(target[0], [9] * 7)
        self.assertEqual(target[1], target[0])
        self.assertEqual(by_index[8]['split'], 'validation')

    def test_missing_hold_is_reported_and_short_source_not_duplicated(self):
        rows, episodes, proof = fixtures()
        proof['sources'][1]['hold_records'] = [{'start_monotonic': 300, 'end_monotonic': 301}]
        cases, missing = evaluation.select_cases(rows, episodes, proof, 12)
        self.assertEqual(missing, [{'source_index': 1, 'hold_index': 0}])
        self.assertEqual(sum('uniform' in case['tags'] for case in cases), 9)


class MetricTests(unittest.TestCase):
    @staticmethod
    def record(row, count, *, split='train', tag='uniform'):
        target = np.zeros((50, 7))
        target[0, :6], target[1, :6] = math.radians(1), math.radians(3)
        target[:, 6] = .1
        target[1, 6] = .9
        target[count:, :] = 1e6
        prediction = np.zeros((50, 7))
        prediction[:, 6] = .2
        return {'source_index': 0, 'episode': 0, 'row_index': row, 'split': split,
                'tags': [tag], 'events': [], 'state': [0.] * 6 + [1., 1.],
                'absolute_actions': prediction.tolist(), 'target_absolute': target.tolist(),
                'valid_mask': [True] * count + [False] * (50-count)}

    def test_micro_averages_use_only_valid_labels_and_keep_splits(self):
        summary = evaluation.Summary(prefix=2)
        summary.add(self.record(0, 1))
        summary.add(self.record(1, 2))
        summary.add(self.record(2, 1, split='validation'))
        result = summary.result()
        train = result['train']['uniform']
        self.assertEqual(train['unique_anchors'], 2)
        self.assertEqual(train['full']['valid_steps'], 3)
        self.assertEqual(train['full']['joint_values'], 18)
        self.assertAlmostEqual(train['full']['joint_mae_deg'], 5 / 3)
        self.assertAlmostEqual(train['full']['gripper_mae_percentage_points'], 30)
        self.assertAlmostEqual(train['first']['joint_mae_deg'], 1)
        self.assertEqual(result['validation']['uniform']['full']['valid_steps'], 1)
        self.assertEqual(train['full']['near_anchor_stationary']['valid_steps'], 0)
        self.assertEqual(train['full']['moving_from_anchor']['valid_steps'], 3)

    def test_switch_accuracy_is_conditioned_on_real_labels_and_ties_count_unmatched(self):
        record = self.record(0, 2, tag='close_at')
        record['events'] = [{'kind': 'close', 'where': 'at', 'event_row_index': 0,
                             'old_target': .9, 'new_target': .1}]
        record['absolute_actions'][0][6] = .1
        record['absolute_actions'][1][6] = .5
        summary = evaluation.Summary(prefix=2)
        summary.add(record)
        metrics = summary.result()['train']['close_at']['full']['switch']
        self.assertEqual(metrics, {'eligible_valid_steps': 2, 'matched_steps': 1,
                                   'ties': 1, 'match_fraction': .5})

    def test_stationary_drift_compares_prediction_and_target_on_identical_steps(self):
        record = self.record(0, 1)
        record['target_absolute'][0][:6] = [math.radians(.2)] * 6
        record['absolute_actions'][0][:6] = [math.radians(2)] * 6
        summary = evaluation.Summary(prefix=3)
        summary.add(record)
        still = summary.result()['train']['uniform']['full']['near_anchor_stationary']
        self.assertEqual(still['valid_steps'], 1)
        self.assertAlmostEqual(still['predicted_abs_from_state_deg'], 2.)
        self.assertAlmostEqual(still['target_abs_from_state_deg'], .2)

    def test_invalid_gripper_feedback_is_excluded_from_state_displacement(self):
        record = self.record(0, 1)
        record['state'][7] = 0.
        summary = evaluation.Summary(prefix=3)
        summary.add(record)
        metrics = summary.result()['train']['uniform']['first']
        self.assertEqual(metrics['valid_steps'], 1)
        self.assertAlmostEqual(metrics['gripper_mae_percentage_points'], 10.)
        self.assertEqual(metrics['gripper_state_valid_steps'], 0)
        self.assertIsNone(metrics['predicted_gripper_abs_from_state_percentage_points'])


class ProvenanceAndOutputTests(unittest.TestCase):
    def test_synthetic_proof_and_robot_marker_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'synthetic'):
            evaluation.require_captured_jev({'training_data_kind': 'synthetic_diagnostic', 'sources': [{}]})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'continuous').mkdir()
            (root / 'scope.json').write_text('{}')
            (root / 'continuous/capture.jsonl').write_text(json.dumps({'robot': {'simulated': True}}) + '\n')
            source = {'trial': str(root), 'data_provenance': {'training_data_kind': 'captured',
                      'scope_contract': 'fr5-whole-task-collection-v1', 'scope_markers': {},
                      'camera_markers': [], 'camera_serials': ['camera']}}
            with self.assertRaisesRegex(ValueError, 'Synthetic or simulated'):
                evaluation.require_captured_jev({'training_data_kind': 'captured', 'sources': [source]})

    def test_output_must_be_new_and_never_overlap_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(evaluation, 'EVALUATIONS_ROOT', root / 'evaluations'):
                good = root / 'evaluations' / 'new'
                self.assertEqual(evaluation.output_directory(good, root / 'data', root / 'ckpt'), good)
                with self.assertRaises(ValueError):
                    evaluation.output_directory(root / 'elsewhere', root / 'data', root / 'ckpt')
                good.mkdir(parents=True)
                with self.assertRaises(FileExistsError):
                    evaluation.output_directory(good, root / 'data', root / 'ckpt')
                with self.assertRaises(ValueError):
                    evaluation.output_directory(good / 'nested', good, root / 'ckpt')


class InputAndInferenceTests(unittest.TestCase):
    def test_actual_batch_builder_uses_past_history_and_never_passes_future_labels(self):
        rows, episodes, _ = fixtures()
        rows[0]['_physical_observation_history'][0] = {
            'age_s': .5, 'state': [0.123] * 6 + [1., 1.]}
        image = torch.zeros((3, 4, 4))
        dataset = {0: {evaluation.training.IMAGE_KEY: image, 'task': 'pick',
                       'action': torch.full((7,), 999.), '_recorded_hold_remaining_s': 999.}}
        case = {'row_index': 0}
        def processor(pre, batch):
            return {**batch, 'action': None, 'next.reward': 0., 'next.done': False, 'info': {}}

        with patch.object(evaluation.training.prompt_integrity, 'preprocess', side_effect=processor):
            for profile in ('none', 'state-history-physical-v1'):
                info = {'image_profile': 'full-v1', 'observation_context_profile': profile}
                batch, task = evaluation.model_input(case, rows, episodes, dataset, None, info)
                self.assertEqual(task, 'pick')
                self.assertNotIn('action', batch)
                self.assertNotIn('next.reward', batch)
                self.assertNotIn('info', batch)
                self.assertFalse(any('hold' in key or 'future' in key for key in batch))
                self.assertNotIn('999', batch['task'][0])
                self.assertEqual('History(' in batch['task'][0], profile != 'none')
                if profile != 'none':
                    self.assertIn('0.5:123,123,123,123,123,123,100,1', batch['task'][0])

    def test_evaluation_writes_each_full_draw_and_truncates_targets_at_episode_end(self):
        rows, episodes, proof = fixtures()
        rows[5]['observation.state'] = [1., 2., 3., 4., 5., 6., 1., 1.]
        case = {'case_index': 0, 'row_index': 5, 'episode': 0, 'position': 5,
                'source_index': 0, 'split': 'train', 'tags': ['uniform'], 'events': [], 'holds': []}
        image = torch.zeros((3, 4, 4))
        dataset = {5: {evaluation.training.IMAGE_KEY: image, 'task': 'pick', 'action': torch.ones(7)}}
        info = {'image_profile': 'full-v1', 'observation_context_profile': 'none',
                'n_action_steps': 3, 'source_tasks': {'0': 'pick'}}

        class Policy:
            def reset(self):
                pass
            def predict_action_chunk(self, batch):
                assert 'action' not in batch
                return torch.zeros((1, 50, 7))

        with tempfile.TemporaryDirectory() as temp, \
                patch.object(evaluation.training.prompt_integrity, 'preprocess', side_effect=lambda pre, batch: batch), \
                patch.object(torch, 'autocast', side_effect=lambda *a, **k: contextlib.nullcontext()):
            path = Path(temp) / 'draws.jsonl'
            summary = evaluation.evaluate_model('fine_tuned', Policy(), None, lambda x: x, info, rows,
                                                 episodes, dataset, [case], 3, 42, path)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r['seed'] for r in records], [42, 43, 44])
            self.assertTrue(all(len(r['native_normalized_actions']) == 50 for r in records))
            self.assertTrue(all(len(r['native_normalized_actions'][0]) == 7 for r in records))
            self.assertTrue(all(sum(r['valid_mask']) == 1 for r in records))
            self.assertEqual(records[0]['source_capture_frame'], 15)
            self.assertEqual(records[0]['target_absolute'][0], rows[5]['action'])
            self.assertEqual(records[0]['absolute_actions'][0][:6], rows[5]['observation.state'][:6])
            self.assertEqual(summary['train']['uniform']['full']['valid_steps'], 3)
            with self.assertRaises(FileExistsError):
                evaluation.evaluate_model('fine_tuned', Policy(), None, lambda x: x, info, rows,
                                          episodes, dataset, [case], 1, 42, path)

    def test_base_control_loads_only_strict_base_parameters(self):
        module = types.ModuleType('lerobot.policies.pi05.modeling_pi05')

        class BasePolicy:
            def __init__(self, config):
                self.config = config
            def _fix_pytorch_state_dict_keys(self, original, config):
                self.original = original
                return {'weight': 1, 'model.bias': 2}
            def load_state_dict(self, mapped, strict):
                self.mapped, self.strict = mapped, strict

        module.PI05Policy = BasePolicy
        with patch.dict(sys.modules, {'lerobot.policies.pi05.modeling_pi05': module}), \
                patch('safetensors.torch.load_file', return_value={'base': 3}) as load, \
                patch.object(evaluation.training, 'reload_parameters', side_effect=AssertionError):
            policy = evaluation.load_base_policy('config', '/base/model.safetensors')
        load.assert_called_once_with('/base/model.safetensors', device='cpu')
        self.assertTrue(policy.strict)
        self.assertEqual(policy.mapped, {'model.weight': 1, 'model.bias': 2})


if __name__ == '__main__':
    unittest.main()
