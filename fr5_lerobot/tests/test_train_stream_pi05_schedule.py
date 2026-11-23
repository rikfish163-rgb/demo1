"""CPU checks for the stream trainer's learning-rate schedules."""
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
from unittest.mock import Mock, patch

import torch
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import train_stream_pi05 as training


PEAK_LR = 1e-5


def optimizer_for(value=1.):
    parameter = torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter], lr=PEAK_LR, betas=(.9, .95), weight_decay=.01)
    return parameter, optimizer


def take_steps(parameter, optimizer, scheduler, count):
    rates = []
    for _ in range(count):
        rates.append(optimizer.param_groups[0]['lr'])
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return rates


def fake_peft():
    module = types.ModuleType('peft')
    module.get_peft_model_state_dict = lambda policy: {'adapter': torch.ones(1)}
    return module


class LearningRateScheduleTests(unittest.TestCase):
    def setUp(self):
        self.config = PI05Config(device='cpu')

    def test_cli_defaults_to_constant_and_accepts_cosine(self):
        argv = ['train_stream_pi05.py', '--dataset', '/dataset', '--run', '/run', '--steps', '1000']
        with patch.object(sys, 'argv', argv), patch.object(training, 'run') as run:
            training.main()
            self.assertEqual(run.call_args.args[0].lr_schedule, 'constant')
        with patch.object(sys, 'argv', argv + ['--lr-schedule', 'cosine']), patch.object(training, 'run') as run:
            training.main()
            self.assertEqual(run.call_args.args[0].lr_schedule, 'cosine')

    def test_legacy_constant_resume_from_300_to_600(self):
        parameter, optimizer = optimizer_for()
        scheduler, plan = training.build_lr_schedule(self.config, optimizer, 'constant', PEAK_LR, 300)
        self.assertIsNone(scheduler)
        self.assertEqual(plan, {'name': 'constant'})
        take_steps(parameter, optimizer, scheduler, 300)
        checkpoint = io.BytesIO()
        torch.save({'steps_completed': 300, 'optimizer': optimizer.state_dict()}, checkpoint)
        checkpoint.seek(0)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        resumed_parameter, resumed_optimizer = optimizer_for(parameter.item())
        resumed_scheduler, resumed_plan = training.build_lr_schedule(
            self.config, resumed_optimizer, 'constant', PEAK_LR, 600, parent_report={})
        training.restore_optimizer_and_scheduler(resumed_optimizer, resumed_scheduler, saved)
        resumed_rates = take_steps(resumed_parameter, resumed_optimizer, resumed_scheduler, 300)
        uninterrupted_rates = take_steps(parameter, optimizer, scheduler, 300)
        self.assertEqual(resumed_plan, plan)
        self.assertEqual(resumed_rates, [PEAK_LR] * 300)
        self.assertEqual(resumed_rates, uninterrupted_rates)
        self.assertTrue(torch.equal(resumed_parameter, parameter))
        self.assertEqual(int(resumed_optimizer.state[resumed_parameter]['step']), 600)

    def test_explicit_cosine_decay_learning_rate(self):
        argv = ['train_stream_pi05.py', '--dataset', '/dataset', '--run', '/run',
                '--steps', '1000', '--lr-schedule', 'cosine', '--learning-rate', '0.0001',
                '--decay-learning-rate', '0.00001']
        with patch.object(sys, 'argv', argv), patch.object(training, 'run') as run:
            training.main()
            self.assertEqual(run.call_args.args[0].decay_learning_rate, 1e-5)
        parameter = torch.nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        config = PI05Config(device='cpu', scheduler_decay_lr=1e-5)
        scheduler, plan = training.build_lr_schedule(config, optimizer, 'cosine', 1e-4, 1000)
        rates = take_steps(parameter, optimizer, scheduler, 1000)
        self.assertEqual(plan['decay_lr'], 1e-5)
        self.assertAlmostEqual(rates[0], 1e-4 / 34, places=18)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1e-5, places=18)

    def test_cosine_first_and_last_optimizer_step_use_native_schedule(self):
        self.assertIsInstance(self.config.get_scheduler_preset(), CosineDecayWithWarmupSchedulerConfig)
        parameter, optimizer = optimizer_for()
        scheduler, plan = training.build_lr_schedule(self.config, optimizer, 'cosine', PEAK_LR, 1000)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
        self.assertEqual(plan['effective_warmup_steps'], 33)
        self.assertEqual(plan['effective_decay_steps'], 1000)
        self.assertEqual(plan['peak_lr'], PEAK_LR)
        rates = take_steps(parameter, optimizer, scheduler, 1000)
        self.assertAlmostEqual(rates[0], 2.9411764705882356e-7, places=18)
        self.assertAlmostEqual(rates[-1], 2.500018505493032e-6, places=18)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 2.5e-6, places=18)

    def test_cosine_checkpoint_restores_original_plan_and_lr_sequence(self):
        parameter, optimizer = optimizer_for()
        scheduler, plan = training.build_lr_schedule(self.config, optimizer, 'cosine', PEAK_LR, 1000)
        take_steps(parameter, optimizer, scheduler, 300)
        snapshot = parameter.item()
        policy = SimpleNamespace(config=Mock(), peft_config={'default': Mock()})
        processor = Mock()
        contract = SimpleNamespace(contract=lambda: {})
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {'peft': fake_peft()}), \
                patch.object(torch.cuda, 'get_rng_state', return_value=torch.zeros(1, dtype=torch.uint8)), \
                patch.object(training, 'projection_state', return_value={'weight': parameter.detach().clone()}), \
                patch.object(training, 'load_contract', return_value=(contract, Path(directory) / 'contract.py')), \
                patch.object(training.legacy, 'tokenizer_path', return_value=Path(directory)), \
                patch.object(training.checkpoints, 'create_bundle', return_value={'checkpoint_sha256': 'test'}), \
                contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            training.save_checkpoint(root, 300, policy, optimizer, processor, processor, {},
                                     {'lr_schedule': plan}, random.Random(1), set(), root / 'base', 'dataset',
                                     scheduler=scheduler)
            checkpoint = root / 'step-000300'
            saved = torch.load(checkpoint / 'optimizer.pt', map_location='cpu', weights_only=False)
            report = json.loads((checkpoint / 'training-report.json').read_text())
            self.assertEqual(saved['steps_completed'], 300)
            self.assertEqual(float(load_file(str(checkpoint / 'projections.safetensors'))['weight']), snapshot)
            resumed_parameter, resumed_optimizer = optimizer_for(snapshot)
            resumed_scheduler, resumed_plan = training.build_lr_schedule(
                self.config, resumed_optimizer, 'cosine', PEAK_LR, 600, parent_report=report)
            training.restore_optimizer_and_scheduler(resumed_optimizer, resumed_scheduler, saved)
            self.assertEqual(resumed_plan, plan)
            self.assertEqual(resumed_plan['num_training_steps'], 1000)
            resumed_rates = take_steps(resumed_parameter, resumed_optimizer, resumed_scheduler, 300)
            uninterrupted_rates = take_steps(parameter, optimizer, scheduler, 300)
            self.assertEqual(resumed_rates, uninterrupted_rates)
            self.assertTrue(torch.equal(resumed_parameter, parameter))
            with self.assertRaisesRegex(ValueError, 'same learning-rate schedule'):
                training.build_lr_schedule(self.config, resumed_optimizer, 'constant', PEAK_LR, 600,
                                           parent_report=report)
            changed_config = PI05Config(device='cpu', scheduler_decay_lr=3e-6)
            with self.assertRaisesRegex(ValueError, 'sealed cosine schedule parameters'):
                training.build_lr_schedule(changed_config, resumed_optimizer, 'cosine', PEAK_LR, 600,
                                           parent_report=report)
            with self.assertRaisesRegex(ValueError, 'saved scheduler state'):
                training.restore_optimizer_and_scheduler(resumed_optimizer, resumed_scheduler,
                                                         {'optimizer': saved['optimizer']})

    def test_training_metrics_record_the_lr_used_by_each_update(self):
        class Policy(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.weight = torch.nn.Parameter(torch.tensor(1.))

            def to(self, device):
                return self

        class Sampler:
            VERSION = training.TransitionSampler.VERSION

            def draw_balanced(self, count):
                return [(0, {})] * count

            def report(self):
                return {}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'dataset'
            (dataset / 'meta').mkdir(parents=True)
            (dataset / 'meta/stream_stats.json').write_text('{"stats": {}}')
            base = root / 'base'
            base.mkdir()
            (base / 'model.safetensors').touch()
            run = root / 'checkpoints' / 'run'
            stats = {name: {'q01': torch.zeros(size), 'q99': torch.ones(size)}
                     for name, size in (('observation.state', 8), ('action', 7))}
            proof = {'schema': 'test', 'training_data_kind': 'test'}
            args = SimpleNamespace(dataset=dataset, run=run, base=base, steps=1000, gpu=None,
                rank=32, adaptation=training.LORA_ADAPTATION,
                action_representation=training.ABSOLUTE_REPRESENTATION, n_action_steps=50,
                image_profile=training.FULL_IMAGE_PROFILE, sampling_profile=None,
                observation_context_profile=None, batch_size=1,
                gradient_accumulation=4, resume=None, initialize=None, learning_rate=PEAK_LR,
                lr_schedule='cosine', first_action_weight=8., gripper_weight=3.,
                transition_eval_draws=1, save_every=1000, eval_samples=1, seed=1)
            saved_reports = []
            def capture_checkpoint(root, step, policy, optimizer, pre, post, stats, report, sampler,
                                   seen, base, fingerprint, *, scheduler):
                self.assertEqual(policy.config.optimizer_lr, args.learning_rate)
                saved_reports.append((step, report.copy(), optimizer.param_groups[0]['lr'], scheduler.last_epoch))

            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules, {'peft': fake_peft()}))
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
                stack.enter_context(patch.object(torch.cuda, 'manual_seed_all'))
                stack.enter_context(patch.object(torch.cuda, 'max_memory_allocated', return_value=0))
                stack.enter_context(patch.object(torch, 'autocast', side_effect=lambda *a, **k: contextlib.nullcontext()))
                stack.enter_context(patch.object(training, 'read_dataset', return_value=([{}, {}], {}, None, proof, 'dataset')))
                stack.enter_context(patch.object(training, 'split_sources', return_value=({0: [0]}, {1: [1]}, 'heldout')))
                stack.enter_context(patch.object(training, 'calculate_stats', return_value=(stats, [])))
                stack.enter_context(patch.object(training, 'make_config', return_value=self.config))
                stack.enter_context(patch.object(training, 'load_policy', side_effect=lambda config, *a, **k: Policy(config)))
                stack.enter_context(patch.object(training, 'trainable_parameters', side_effect=lambda policy, mode: [policy.weight]))
                stack.enter_context(patch.object(training, 'make_sampler', return_value=Sampler()))
                stack.enter_context(patch.object(training, 'batch_for', return_value=(None, None)))
                stack.enter_context(patch.object(training, 'masked_flow_loss', side_effect=lambda policy, *a, **k: policy.weight.square()))
                stack.enter_context(patch.object(training, 'gradient_summary', return_value={}))
                stack.enter_context(patch.object(training, 'evaluate', return_value={}))
                stack.enter_context(patch.object(training, 'evaluate_transitions', return_value={}))
                stack.enter_context(patch.object(training, 'anchor_coverage', return_value={}))
                stack.enter_context(patch.object(training, 'save_checkpoint', side_effect=capture_checkpoint))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                training.run(args)
            metrics = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(len(metrics), 1000)
            self.assertAlmostEqual(metrics[0]['learning_rate'], 2.9411764705882356e-7, places=18)
            self.assertAlmostEqual(metrics[-1]['learning_rate'], 2.500018505493032e-6, places=18)
            self.assertEqual(saved_reports[0][1]['lr_schedule']['effective_warmup_steps'], 33)
            self.assertEqual(saved_reports[0][1]['target_steps'], 1000)
            self.assertAlmostEqual(saved_reports[0][2], 2.5e-6, places=18)
            self.assertEqual(saved_reports[0][3], 1000)


if __name__ == '__main__':
    unittest.main()
