"""CPU checks for metrics history when stream training resumes."""
import contextlib
import io
import json
import os
import random
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


def lines(steps):
    return b''.join((json.dumps({'step': step, 'loss': step / 10}) + '\n').encode() for step in steps)


class MetricsHistoryTests(unittest.TestCase):
    def test_no_tail_leaves_the_history_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = lines(range(1, 4))
            (root / 'metrics.jsonl').write_bytes(original)
            lineage = training.prepare_resume_metrics(root, 3)
            self.assertEqual(lineage['preserved_unsaved_tails'], [])
            self.assertEqual(lineage['checkpoint_history']['available_step_range'], [1, 3])
            self.assertEqual(lineage['checkpoint_history']['missing_step_ranges'], [])
            self.assertEqual((root / 'metrics.jsonl').read_bytes(), original)
            self.assertEqual(list(root.glob('metrics.unsaved-after-step-*.jsonl')), [])

    def test_repeated_resumes_preserve_each_tail_exactly(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            metrics = root / 'metrics.jsonl'
            first_tail = lines(range(4, 6))
            metrics.write_bytes(lines(range(1, 4)) + first_tail)
            first = Path(training.prepare_resume_metrics(root, 3)['preserved_unsaved_tails'][0])
            second_tail = lines(range(4, 7)) + b'{"step": 7, "loss":'
            with metrics.open('ab') as log:
                log.write(second_tail)
            archives = [Path(path) for path in training.prepare_resume_metrics(root, 3)['preserved_unsaved_tails']]
            self.assertEqual(archives[0], first)
            self.assertEqual(first.read_bytes(), first_tail)
            self.assertEqual(archives[1].read_bytes(), second_tail)
            self.assertEqual(metrics.read_bytes(), lines(range(1, 4)))

    def test_interleaved_history_and_tail_keep_all_existing_records(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            metrics = root / 'metrics.jsonl'
            metrics.write_bytes(lines((1, 3, 5, 2, 6)))
            lineage = training.prepare_resume_metrics(root, 4)
            self.assertEqual(metrics.read_bytes(), lines((1, 3, 2)))
            self.assertEqual(lineage['checkpoint_history']['available_step_range'], [1, 3])
            self.assertEqual(lineage['checkpoint_history']['missing_step_ranges'], [[4, 4]])
            self.assertEqual(Path(lineage['preserved_unsaved_tails'][0]).read_bytes(), lines((5, 6)))

    def test_unfinished_record_is_preserved_without_blocking_resume(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            metrics = root / 'metrics.jsonl'
            unfinished = b'{"step": 3, "loss":'
            metrics.write_bytes(lines((1, 2)) + unfinished)
            lineage = training.prepare_resume_metrics(root, 3)
            self.assertEqual(metrics.read_bytes(), lines((1, 2)))
            self.assertEqual(lineage['checkpoint_history']['missing_step_ranges'], [[3, 3]])
            archive = Path(lineage['preserved_unparsed_tails'][0])
            self.assertEqual(archive.read_bytes(), unfinished)

    def test_interrupted_training_resumes_without_repeating_metrics(self):
        self.check_interrupted_resume('complete')

    def test_resume_accepts_missing_or_incomplete_metrics(self):
        for mode in ('missing', 'incomplete'):
            with self.subTest(mode=mode):
                self.check_interrupted_resume(mode)

    def check_interrupted_resume(self, mode):
        class Policy(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.weight = torch.nn.Parameter(torch.tensor(1.))

            def to(self, device):
                return self

        class Sampler:
            VERSION = training.TransitionSampler.VERSION

            def __init__(self, stop_after=None):
                self.count = 0
                self.rng = random.Random(4)
                self.stop_after = stop_after

            def draw_balanced(self, count):
                if self.count == self.stop_after:
                    raise InterruptedError('training stopped')
                self.count += 1
                return [(0, {'draw': self.rng.randrange(1000000)}) for _ in range(count)]

            def getstate(self):
                return {'count': self.count, 'rng': self.rng.getstate()}

            def setstate(self, state):
                self.count = state['count']
                self.rng.setstate(state['rng'])

            def report(self):
                return {}

        def save_checkpoint(root, step, policy, optimizer, pre, post, stats, report, sampler,
                            seen, base, fingerprint, *, scheduler):
            checkpoint = root / f'step-{step:06d}'
            checkpoint.mkdir()
            torch.save({'steps_completed': step, 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict() if scheduler else None,
                        'python_rng': sampler.getstate(), 'numpy_rng': np.random.get_state(),
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.zeros(1, dtype=torch.uint8),
                        'seen_anchors': sorted(seen), 'weight': policy.weight.detach().clone()},
                       checkpoint / 'optimizer.pt')
            (checkpoint / 'training-report.json').write_text(json.dumps({**report, 'steps_completed': step}))
            (checkpoint / 'normalization.json').write_text(json.dumps({
                name: {key: value.tolist() for key, value in values.items()}
                for name, values in stats.items()}))

        def load_policy(config, base, *, checkpoint=None, **kwargs):
            policy = Policy(config)
            if checkpoint:
                saved = torch.load(checkpoint / 'optimizer.pt', map_location='cpu', weights_only=False)
                policy.weight.data.copy_(saved['weight'])
            return policy

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'dataset'
            (dataset / 'meta').mkdir(parents=True)
            (dataset / 'meta/stream_stats.json').write_text('{"stats": {}}')
            base = root / 'base'
            base.mkdir()
            (base / 'model.safetensors').touch()
            run = root / 'checkpoints' / 'run'
            config = PI05Config(device='cpu', n_action_steps=50)
            stats = {name: {'q01': torch.zeros(size), 'q99': torch.ones(size)}
                     for name, size in (('observation.state', 8), ('action', 7))}
            proof = {'schema': 'test', 'training_data_kind': 'test', 'sources': []}
            args = SimpleNamespace(dataset=dataset, run=run, base=base, steps=550, gpu=None,
                rank=32, adaptation=training.LORA_ADAPTATION,
                action_representation=training.ABSOLUTE_REPRESENTATION, n_action_steps=50,
                image_profile=training.FULL_IMAGE_PROFILE, sampling_profile=None,
                observation_context_profile=None, batch_size=1,
                gradient_accumulation=4, resume=None, initialize=None, learning_rate=1e-5,
                lr_schedule='constant', first_action_weight=8., gripper_weight=3.,
                transition_eval_draws=1, save_every=500, eval_samples=1, seed=1)
            bundle = {'manifest': {'policy_contract': {}, 'dataset_fingerprint': 'dataset'},
                      'checkpoint_sha256': 'test', 'n_action_steps': 50,
                      'observation_context_profile': 'none', 'image_profile': training.FULL_IMAGE_PROFILE,
                      'action_representation': training.ABSOLUTE_REPRESENTATION,
                      'adaptation_mode': training.LORA_ADAPTATION}
            peft = types.ModuleType('peft')
            peft.get_peft_model_state_dict = lambda policy: {}
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules, {'peft': peft}))
                stack.enter_context(patch.dict(os.environ))
                stack.enter_context(patch.object(training.legacy, 'ROOT', root))
                stack.enter_context(patch.object(training.legacy, 'require_runtime_versions'))
                stack.enter_context(patch.object(training.legacy, 'configure_environment'))
                stack.enter_context(patch.object(training.legacy, 'check_data4', side_effect=lambda p: Path(p).resolve()))
                stack.enter_context(patch.object(training.legacy, 'sha', return_value=training.legacy.EXPECTED_WEIGHTS_SHA256))
                stack.enter_context(patch.object(training.legacy, 'gpu_status', return_value={'index': 0}))
                stack.enter_context(patch.object(training.legacy, 'tokenizer_path', return_value=root))
                stack.enter_context(patch.object(training.legacy, 'make_processors', return_value=(None, None)))
                stack.enter_context(patch.object(torch.cuda, 'device_count', return_value=1))
                stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=False))
                stack.enter_context(patch.object(torch.cuda, 'manual_seed_all'))
                stack.enter_context(patch.object(torch.cuda, 'set_rng_state'))
                stack.enter_context(patch.object(torch.cuda, 'max_memory_allocated', return_value=0))
                stack.enter_context(patch.object(torch, 'autocast', side_effect=lambda *a, **k: contextlib.nullcontext()))
                stack.enter_context(patch.object(PI05Config, 'from_pretrained', return_value=config))
                stack.enter_context(patch.object(training.checkpoints, 'verify_bundle', return_value=bundle))
                stack.enter_context(patch.object(training, 'load_contract', return_value=(SimpleNamespace(contract=lambda: {}), root)))
                stack.enter_context(patch.object(training, 'read_dataset', return_value=([{}, {}], {}, None, proof, 'dataset')))
                stack.enter_context(patch.object(training, 'split_sources', return_value=({0: [0]}, {1: [1]}, 'heldout')))
                stack.enter_context(patch.object(training, 'calculate_stats', return_value=(stats, [])))
                stack.enter_context(patch.object(training, 'make_config', return_value=config))
                stack.enter_context(patch.object(training, 'load_policy', side_effect=load_policy))
                stack.enter_context(patch.object(training, 'trainable_parameters', side_effect=lambda policy, mode: [policy.weight]))
                stack.enter_context(patch.object(training, 'make_sampler', side_effect=[Sampler(540), Sampler()]))
                stack.enter_context(patch.object(training, 'batch_for', return_value=(None, None)))
                stack.enter_context(patch.object(training, 'masked_flow_loss', side_effect=lambda policy, *a, **k: policy.weight.square()))
                stack.enter_context(patch.object(training, 'gradient_summary', return_value={}))
                stack.enter_context(patch.object(training, 'evaluate', return_value={}))
                stack.enter_context(patch.object(training, 'evaluate_transitions', return_value={}))
                stack.enter_context(patch.object(training, 'anchor_coverage', return_value={}))
                stack.enter_context(patch.object(training, 'save_checkpoint', side_effect=save_checkpoint))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                with self.assertRaises(InterruptedError):
                    training.run(args)
                original = (run / 'metrics.jsonl').read_bytes().splitlines(keepends=True)
                self.assertEqual(len(original), 540)
                if mode == 'missing':
                    (run / 'metrics.jsonl').unlink()
                    expected_history = []
                elif mode == 'incomplete':
                    expected_history = [line for number, line in enumerate(original[:500], 1)
                                        if number not in (100, 250, 500)]
                    (run / 'metrics.jsonl').write_bytes(b''.join(expected_history + original[500:]))
                else:
                    expected_history = original[:500]
                args.resume = run / 'step-000500'
                training.run(args)

            active_lines = (run / 'metrics.jsonl').read_bytes().splitlines(keepends=True)
            active = [json.loads(line) for line in active_lines]
            expected_steps = [json.loads(line)['step'] for line in expected_history] + list(range(501, 551))
            self.assertEqual([item['step'] for item in active], expected_steps)
            self.assertEqual(active_lines[:len(expected_history)], expected_history)
            report = json.loads((run / 'step-000550' / 'training-report.json').read_text())
            self.assertEqual(report['metrics_lineage']['resumed_from_step'], 500)
            self.assertEqual(report['metrics_lineage']['active_metrics'], str(run / 'metrics.jsonl'))
            history = report['metrics_lineage']['checkpoint_history']
            self.assertEqual(history['file_present'], mode != 'missing')
            self.assertEqual(history['available_step_count'], len(expected_history))
            self.assertEqual(history['available_step_range'],
                             None if mode == 'missing' else [1, 499 if mode == 'incomplete' else 500])
            expected_missing = ([[1, 500]] if mode == 'missing' else
                                [[100, 100], [250, 250], [500, 500]] if mode == 'incomplete' else [])
            self.assertEqual(history['missing_step_ranges'], expected_missing)
            archived = [Path(path) for path in report['metrics_lineage']['preserved_unsaved_tails']]
            self.assertEqual(len(archived), 0 if mode == 'missing' else 1)
            if archived:
                self.assertEqual(archived[0].read_bytes(), b''.join(original[500:]))
            start = len(expected_history)
            for previous, resumed in zip(original[500:], active[start:start + 40]):
                previous = json.loads(previous)
                for key in ('step', 'loss', 'gradient_norm', 'learning_rate', 'sampling'):
                    self.assertEqual(previous[key], resumed[key])


if __name__ == '__main__':
    unittest.main()
