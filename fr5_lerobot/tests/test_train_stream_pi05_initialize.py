"""CPU checks for warm-starting with reviewed physical sources."""
import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from lerobot.policies.pi05.configuration_pi05 import PI05Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import train_stream_pi05 as training


def source(number, split):
    return {'source_scope_sha256': f'scope-{number}', 'user_review_sha256': f'review-{number}',
            'session': f'session-{number}', 'source_stream_sha256': f'stream-{number}', 'split': split}


def proof(sources, kind='captured'):
    return {'schema': 'fr5-stream-retarget-export-v1', 'training_data_kind': kind,
            'sources': copy.deepcopy(sources)}


def original_sources():
    return [source(i, 'train' if i < 7 else 'validation') for i in range(8)]


def parent_report(sources):
    return {'rank': 16, 'source_provenance': proof(sources), 'steps_completed': 5000,
            'sampling': training.TransitionSampler.VERSION,
            'sampling_profile': training.DEFAULT_SAMPLING_PROFILE,
            'batch_size': 1, 'gradient_accumulation': 4, 'learning_rate': 1e-5, 'seed': 4,
            'first_action_weight': 8., 'gripper_weight': 3.,
            'adaptation_mode': training.VISION_EXPERT_ADAPTATION,
            'action_representation': training.RELATIVE_REPRESENTATION,
            'observation_context_profile': training.observation_history.PHYSICAL_HISTORY_PROFILE,
            'normalization_fallbacks': [{'feature': 'observation.state', 'index': 0, 'reason': 'parent'}]}


def parent_args():
    return SimpleNamespace(rank=16, adaptation=training.VISION_EXPERT_ADAPTATION,
                           action_representation=training.RELATIVE_REPRESENTATION,
                           observation_context_profile=training.observation_history.PHYSICAL_HISTORY_PROFILE,
                           batch_size=1, gradient_accumulation=4, learning_rate=1e-5, seed=4,
                           first_action_weight=8., gripper_weight=3.)


class ParentSourcesTests(unittest.TestCase):
    def setUp(self):
        self.original = original_sources()
        self.parent = parent_report(self.original)
        self.args = parent_args()

    def test_same_sources_initialize_and_exact_resume(self):
        current = proof(self.original)
        self.assertFalse(training.validate_parent(self.parent, current, self.args, resume=False))
        self.assertFalse(training.validate_parent(self.parent, current, self.args, resume=True))

    def test_added_sources_and_prepare_order(self):
        current = self.original + [source(8, 'train'), source(9, 'validation')]
        self.assertTrue(training.validate_parent(self.parent, proof(current), self.args, resume=False))
        # prepare_jev_pi05_dataset places all new train sources before new validation sources.
        parent = parent_report(current)
        later = self.original + [current[8], source(10, 'train'), current[9], source(11, 'validation')]
        self.assertTrue(training.validate_parent(parent, proof(later), self.args, resume=False))

    def test_rejects_reassigned_holdout_changed_review_and_missing_source(self):
        changed_split = copy.deepcopy(self.original) + [source(8, 'train')]
        changed_split[7]['split'] = 'train'
        changed_review = copy.deepcopy(self.original) + [source(8, 'train')]
        changed_review[0]['user_review_sha256'] = 'different-review'
        changed_holdout_review = copy.deepcopy(self.original) + [source(8, 'train')]
        changed_holdout_review[7]['user_review_sha256'] = 'different-review'
        missing = copy.deepcopy(self.original[1:]) + [source(8, 'train'), source(9, 'validation')]
        missing_holdout = copy.deepcopy(self.original[:7]) + [source(8, 'train'), source(9, 'validation')]
        for current in (changed_split, changed_review, changed_holdout_review, missing, missing_holdout):
            with self.subTest(current=current), self.assertRaisesRegex(ValueError, 'physical source split'):
                training.validate_parent(self.parent, proof(current), self.args, resume=False)

    def test_rejects_repeated_holdout_or_changed_recording(self):
        repeated_scope = copy.deepcopy(self.original) + [source(8, 'train')]
        repeated_scope[-1]['source_scope_sha256'] = self.original[7]['source_scope_sha256']
        repeated_recording = copy.deepcopy(self.original) + [source(8, 'train')]
        repeated_recording[-1]['session'] = self.original[7]['session']
        repeated_stream = copy.deepcopy(self.original) + [source(8, 'train')]
        repeated_stream[-1]['source_stream_sha256'] = self.original[7]['source_stream_sha256']
        changed_recording = copy.deepcopy(self.original) + [source(8, 'train')]
        changed_recording[0]['source_stream_sha256'] = 'different-stream'
        for current in (repeated_scope, repeated_recording, repeated_stream, changed_recording):
            with self.subTest(current=current), self.assertRaisesRegex(ValueError, 'physical source split'):
                training.validate_parent(self.parent, proof(current), self.args, resume=False)

    def test_rejects_synthetic_extension_and_resume_extension(self):
        extended = self.original + [source(8, 'train')]
        for current, resume in ((proof(extended, 'synthetic_diagnostic'), False), (proof(extended), True)):
            with self.subTest(resume=resume), self.assertRaisesRegex(ValueError, 'physical source split'):
                training.validate_parent(self.parent, current, self.args, resume=resume)


class TinyPolicy(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = torch.nn.Parameter(torch.tensor(1.5))

    def to(self, device):
        return self


class TinySampler:
    VERSION = training.TransitionSampler.VERSION

    def draw_balanced(self, count):
        return [(0, {})] * count

    def report(self):
        return {}


def sample_rows(sources, holdout_offset):
    rows = []
    for episode, item in enumerate(sources):
        is_train = item['split'] == 'train'
        base = episode * .1 if is_train else holdout_offset
        movement = .4 if item['source_scope_sha256'] == 'scope-8' else .02
        if not is_train:
            movement = 10.
        for frame in range(3):
            state = [base + frame * .02 + joint * .01 for joint in range(6)] + [.4 + frame * .05, 1.]
            action = [value + movement * (frame + 1) for value in state[:6]] + [.5 + frame * .05]
            rows.append({'index': len(rows), 'episode_index': episode, 'frame_index': frame,
                         'timestamp': frame / training.FPS, '_physical_source': str(episode),
                         '_split': item['split'], 'observation.state': state, 'action': action})
    return rows


class InitializeRunTests(unittest.TestCase):
    def run_initialize(self, additions=(), *, holdout_offset=100., representation=training.RELATIVE_REPRESENTATION):
        original = original_sources()
        sources = original + list(additions)
        rows = sample_rows(sources, holdout_offset)
        episodes = training.validate_rows(rows)
        train_eps, val_eps, _ = training.split_sources(rows, episodes)
        train_rows = [rows[index] for indices in train_eps.values() for index in indices]
        exported = {}
        for feature in ('observation.state', 'action'):
            values = np.asarray([row[feature] for row in train_rows], dtype=np.float64)
            exported[feature] = {name: np.quantile(values, quantile, axis=0).tolist()
                                 for name, quantile in (('q01', .01), ('q99', .99))}
        current_proof = proof(sources)
        old_report = parent_report(original)
        old_report['action_representation'] = representation
        saved_stats = {name: {'q01': [-7.] * size, 'q99': [7.] * size}
                       for name, size in (('observation.state', 8), ('action', 7))}
        contract = {'name': 'test-stream'}
        bundle = {'manifest': {'policy_contract': contract, 'dataset_fingerprint': 'parent-dataset'},
                  'checkpoint_sha256': 'parent-checkpoint', 'n_action_steps': 10,
                  'observation_context_profile': training.observation_history.PHYSICAL_HISTORY_PROFILE,
                  'action_representation': representation, 'adaptation_mode': training.VISION_EXPERT_ADAPTATION,
                  'image_profile': training.checkpoints.WORKSPACE_IMAGE_PROFILE}
        relative = training.load_relative_helper()
        captured = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, parent, base = root / 'dataset', root / 'parent', root / 'base'
            (dataset / 'meta').mkdir(parents=True)
            (dataset / 'meta/stream_stats.json').write_text(json.dumps({'stats': exported}))
            parent.mkdir()
            (parent / 'training-report.json').write_text(json.dumps(old_report))
            (parent / 'normalization.json').write_text(json.dumps(saved_stats))
            base.mkdir()
            (base / 'model.safetensors').touch()
            args = SimpleNamespace(dataset=dataset, run=root / 'checkpoints/run', base=base, gpu=None,
                resume=None, initialize=parent, n_action_steps=None, image_profile=None,
                sampling_profile=None, observation_context_profile=None, rank=16, seed=4,
                adaptation=training.VISION_EXPERT_ADAPTATION, action_representation=representation,
                steps=1, batch_size=1, gradient_accumulation=4, learning_rate=1e-5, lr_schedule='constant',
                first_action_weight=8., gripper_weight=3., transition_eval_draws=1, save_every=1, eval_samples=1)
            config = PI05Config(device='cpu', n_action_steps=10)
            policy = TinyPolicy(config)
            peft = types.ModuleType('peft')
            peft.get_peft_model_state_dict = lambda unused: {}

            def save_checkpoint(_root, step, saved_policy, _optimizer, _pre, _post, stats, report,
                                _sampler, _seen, _base, _fingerprint, *, scheduler):
                captured.update(step=step, weight=float(saved_policy.weight.detach()),
                                stats=copy.deepcopy(stats), report=copy.deepcopy(report))

            with contextlib.ExitStack() as stack:
                for context in (
                    patch.dict(sys.modules, {'peft': peft}), patch.dict(os.environ),
                    patch.object(training.legacy, 'ROOT', root),
                    patch.object(training.legacy, 'require_runtime_versions'),
                    patch.object(training.legacy, 'configure_environment'),
                    patch.object(training.legacy, 'check_data4', side_effect=lambda path: Path(path).resolve()),
                    patch.object(training.legacy, 'sha', return_value=training.legacy.EXPECTED_WEIGHTS_SHA256),
                    patch.object(training.legacy, 'gpu_status', return_value={'index': 0}),
                    patch.object(training.legacy, 'tokenizer_path', return_value=root),
                    patch.object(training.legacy, 'make_processors', return_value=(None, None)),
                    patch.object(torch.cuda, 'device_count', return_value=1),
                    patch.object(torch.cuda, 'manual_seed_all'),
                    patch.object(torch.cuda, 'max_memory_allocated', return_value=0),
                    patch.object(torch, 'autocast', side_effect=lambda *a, **k: contextlib.nullcontext()),
                    patch.object(PI05Config, 'from_pretrained', return_value=config),
                    patch.object(training.checkpoints, 'verify_bundle', return_value=bundle),
                    patch.object(training, 'load_contract', return_value=(SimpleNamespace(contract=lambda: contract), root)),
                    patch.object(training, 'read_dataset', return_value=(rows, episodes, None, current_proof, 'current-dataset')),
                    patch.object(training, 'load_policy', return_value=policy),
                    patch.object(training, 'trainable_parameters', return_value=[policy.weight]),
                    patch.object(training, 'vision_adapter_report', return_value=None),
                    patch.object(training, 'make_sampler', return_value=TinySampler()),
                    patch.object(relative, 'assert_manual_pipeline'),
                    patch.object(training, 'batch_for', return_value=(None, None)),
                    patch.object(training, 'masked_flow_loss', side_effect=lambda model, *a, **k: model.weight.square()),
                    patch.object(training, 'gradient_summary', return_value={
                        'action_expert': {'norm': 1.}, 'language_lora': {'norm': 1.},
                        'vision_lora': {'norm': 1.}}),
                    patch.object(training, 'evaluate', return_value={}),
                    patch.object(training, 'evaluate_transitions', return_value={}),
                    patch.object(training, 'save_checkpoint', side_effect=save_checkpoint),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    stack.enter_context(context)
                training.run(args)
                processors = training.legacy.make_processors.call_args.args
                self.assertEqual(training.load_policy.call_args.kwargs['checkpoint'], parent)
                self.assertEqual(training.make_sampler.call_args.args[2], train_eps)
                self.assertEqual(training.batch_for.call_args.args[2], train_eps)
                for feature in saved_stats:
                    for key in ('q01', 'q99'):
                        self.assertTrue(torch.equal(processors[1][feature][key], captured['stats'][feature][key]))
        self.assertEqual(captured['step'], 1)
        self.assertLess(captured['weight'], 1.5)
        self.assertEqual(captured['report']['initialized_from_steps'], 5000)
        self.assertEqual(captured['report']['n_action_steps'], 10)
        self.assertEqual(captured['report']['image_profile'], training.checkpoints.WORKSPACE_IMAGE_PROFILE)
        self.assertEqual(captured['report']['observation_context_profile'],
                         training.observation_history.PHYSICAL_HISTORY_PROFILE)
        self.assertEqual(captured['report']['adaptation_mode'], training.VISION_EXPERT_ADAPTATION)
        self.assertFalse(captured['report']['optimizer_restored'])
        self.assertEqual(captured['report']['heldout_episodes'], list(val_eps))
        self.assertEqual(captured['report']['train_episodes'], list(train_eps))
        return captured, saved_stats, exported

    def test_same_source_initialize_uses_parent_normalization(self):
        result, saved, _ = self.run_initialize()
        for feature, values in saved.items():
            for key, expected in values.items():
                self.assertTrue(torch.equal(result['stats'][feature][key], torch.tensor(expected)))
        self.assertEqual(result['report']['normalization_from'], 'parent_checkpoint_unchanged')
        self.assertEqual(result['report']['normalization_checkpoint_sha256'], 'parent-checkpoint')
        self.assertEqual(result['report']['normalization_fallbacks'][0]['reason'], 'parent')

    def test_added_sources_use_current_train_stats_and_exclude_holdouts(self):
        additions = [source(8, 'train'), source(9, 'validation')]
        result, saved, exported = self.run_initialize(additions)
        changed_holdouts, _, _ = self.run_initialize(additions, holdout_offset=-1000.)
        for feature in ('observation.state', 'action'):
            for key in ('q01', 'q99'):
                actual = result['stats'][feature][key]
                self.assertTrue(torch.equal(actual, changed_holdouts['stats'][feature][key]))
                self.assertFalse(torch.equal(actual, torch.tensor(saved[feature][key])))
        for key in ('q01', 'q99'):
            self.assertTrue(torch.allclose(result['stats']['observation.state'][key][:7],
                                           torch.tensor(exported['observation.state'][key][:7])))
            self.assertEqual(float(result['stats']['observation.state'][key][7]), 0. if key == 'q01' else 1.)
        deltas = 7 * [.02, .06, .10, .04, .08, .06] + [.4, .82, 1.24, .8, 1.22, 1.2]
        self.assertAlmostEqual(float(result['stats']['action']['q99'][0]), float(np.quantile(deltas, .99)), places=5)
        report = result['report']
        self.assertEqual(report['normalization_from'], 'train_source_state_and_all_valid_relative_chunks')
        self.assertIsNone(report['normalization_checkpoint_sha256'])
        self.assertEqual(report['normalization_fallbacks'], report['dataset_suggested_normalization_fallbacks'])
        self.assertEqual(report['relative_action_statistics']['train_physical_sources'],
                         [str(i) for i in range(7)] + ['8'])
        self.assertFalse(report['relative_action_statistics']['heldout_included'])
        self.assertEqual(report['heldout_physical_source'], ['7', '9'])
        for feature in ('observation.state', 'action'):
            for key in ('q01', 'q99'):
                self.assertEqual(report['current_train_stats_minus_used_stats'][feature][key],
                                 [0.] * len(result['stats'][feature][key]))

    def test_validation_only_addition_still_reports_current_stats(self):
        result, _, _ = self.run_initialize([source(8, 'validation')])
        self.assertEqual(result['report']['normalization_from'], 'train_source_state_and_all_valid_relative_chunks')
        self.assertIsNone(result['report']['normalization_checkpoint_sha256'])
        self.assertEqual(result['report']['heldout_physical_source'], ['7', '8'])

    def test_absolute_representation_reports_current_exported_stats(self):
        result, _, exported = self.run_initialize([source(8, 'train')],
                                                  representation=training.ABSOLUTE_REPRESENTATION)
        self.assertEqual(result['report']['normalization_from'],
                         'exported_training_sources_only_with_near_constant_fallbacks')
        for feature in ('observation.state', 'action'):
            for key in ('q01', 'q99'):
                self.assertTrue(torch.allclose(result['stats'][feature][key][:7],
                                               torch.tensor(exported[feature][key][:7])))
        self.assertEqual(float(result['stats']['observation.state']['q01'][7]), 0.)
        self.assertEqual(float(result['stats']['observation.state']['q99'][7]), 1.)


if __name__ == '__main__':
    unittest.main()
