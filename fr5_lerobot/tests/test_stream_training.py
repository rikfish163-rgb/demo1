"""CPU-only checks for stream alignment, source holdout, and padding."""
import sys
import json
import copy
from contextlib import ExitStack
import hashlib
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import train_stream_pi05 as training


class ActionClockConditionTests(unittest.TestCase):
    def test_future_ticks_only_condition_padding_channels(self):
        noisy = torch.arange(2 * 10 * 32, dtype=torch.float32).reshape(2, 10, 32)
        original = noisy.clone()
        fake = SimpleNamespace(model=SimpleNamespace(embed_suffix=lambda actions, _: actions))
        clock = torch.tensor([[1., 1., 1., 0.], [0., 0., 0., 1.]])
        with training.action_clock_condition(fake, {'observation.cup_action_clock': clock}):
            actions = fake.model.embed_suffix(noisy, None)
        self.assertTrue(torch.equal(actions[..., :7], original[..., :7]))
        self.assertTrue(torch.equal(actions[..., 11:], original[..., 11:]))
        delta = actions - original
        self.assertGreater(float(delta[0, 0, 7]), 0.)
        self.assertLess(float(delta[0, 9, 7]), 0.)
        self.assertTrue(torch.equal(delta[1, :, 7], torch.zeros(10)))
        self.assertTrue(torch.equal(noisy, original))
        self.assertIs(fake.model.embed_suffix(noisy, None), noisy)


def sample_rows(lengths=(4, 3), offset=100.):
    rows = []
    for ep, count in enumerate(lengths):
        for n in range(count):
            v = float(n) + (offset if ep == len(lengths) - 1 else 0.)
            rows.append({'index': len(rows), 'episode_index': ep, 'frame_index': n,
                         'timestamp': n / 6, '_physical_source': str(ep),
                         '_split': 'validation' if ep == len(lengths) - 1 else 'train',
                         'observation.state': [v] * 6 + [1., 1.],
                         'action': [v + .01] * 6 + [.7 if n == count - 1 else 1.]})
    return rows


def transition_rows():
    rows, sources = [], []
    for ep in range(3):
        start = ep * 100.
        sources.append({'source_index': ep, 'split': 'validation' if ep == 2 else 'train',
                        'source_scope_sha256': f'scope-{ep}', 'user_review_sha256': f'review-{ep}',
                        'hold_records': [{'start_monotonic': start + 6/6,
                                          'end_monotonic': start + 7/6}]})
        for n in range(14):
            target = 1. if n < 4 or n >= 9 else .76
            rows.append({'index': len(rows), 'episode_index': ep, 'frame_index': n,
                         'timestamp': n/6, '_physical_source': str(ep),
                         '_split': sources[-1]['split'], '_grid_time': start + n/6,
                         '_gripper_target_ref': {'kind': 'gripper', 'executed': True,
                             'source': 'wire_command', 'sequence': n,
                             'target_degrees_percent': [0.] * 6 + [100 * target]},
                         'observation.state': [n*.01]*6+[target, 1.],
                         'action': [n*.01 + .01]*6+[target]})
    proof = {'sources': sources}
    episodes = training.validate_rows(rows)
    train, val, _ = training.split_sources(rows, episodes)
    return rows, proof, train, val


def compact_clock_fixture():
    times = [100., 100.+1/6, 100.+4/6, 100.+5/6]
    source_frames = [10, 11, 14, 15]
    maps = []
    for i, (t, source_frame) in enumerate(zip(times, source_frames)):
        maps.append({'frame': i, 'episode_index': 0, 'source_index': 0, 'split': 'train', 'step': i,
                     'source_v1_frame': source_frame, 'grid_time_monotonic': t,
                     'target_time_monotonic': times[i+1] if i+1 < len(times) else t+1/6,
                     'task_time_seconds': i/6, 'target_task_time_seconds': (i+1)/6,
                     'cut_after': i == 1, 'cut_id': 'c1' if i == 1 else None})
    proof = {'sources': [{'hold_records': [{'start_monotonic': 102., 'end_monotonic': 103.}]}]}
    cut = {'cut_id': 'c1', 'episode_index': 0, 'source_index': 0, 'split': 'train',
           'left_frame': 1, 'right_frame': 2, 'left_task_step': 1, 'right_task_step': 2,
           'left_source_v1_frame': 11, 'right_source_v1_frame': 14,
           'removed_source_v1_frames': [12, 13], 'left_task_time_seconds': 1/6,
           'right_task_time_seconds': 2/6, 'left_source_time_monotonic': times[1],
           'right_source_time_monotonic': times[2], 'source_elapsed_seconds': times[2]-times[1],
           'removed_task_seconds': 2/6, 'previous_command': {'completed_monotonic': times[1]},
           'next_command': {'issued_monotonic': times[2]}, 'hold_intersections': [],
           'feedback': {'all_gripper_feedback_valid': True, 'joint_span_deg': [.01]*6,
                        'feedback_samples': 5, 'lower_time_monotonic': times[1],
                        'upper_time_monotonic': times[2], 'gripper_min_percent': 75.,
                        'gripper_max_percent': 75.5}}
    def segment(a, b):
        return {'episode_index': 0, 'source_index': 0, 'task_start_step': a, 'task_end_step': b,
                'source_v1_start_frame': source_frames[a], 'source_v1_end_frame': source_frames[b],
                'source_start_monotonic': times[a], 'source_end_monotonic': times[b]}
    clock = {'schema': 'fr5-stream-clock-map-v2', 'task_fps': 6,
             'episodes': [{'episode_index': 0, 'source_index': 0, 'task_fps': 6,
                           'segments': [segment(0, 1), segment(2, 3)]}]}
    return maps, proof, [cut], clock


class ToyFlow(torch.nn.Module):
    def embed_suffix(self, actions, timestep):
        pad = torch.ones(actions.shape[:2], dtype=torch.bool)
        return actions, pad, torch.zeros_like(pad), None

    def sample_noise(self, shape, device):
        return torch.zeros(shape, device=device)

    def sample_time(self, size, device):
        return torch.zeros(size, device=device)

    def forward(self, images, image_masks, tokens, masks, actions, noise, sample_time):
        embeddings, valid, _, _ = self.embed_suffix(actions, sample_time)
        weights = valid.to(embeddings.dtype).unsqueeze(-1)
        average = (embeddings * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True)
        return (embeddings - average).square()


class ToyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = torch.nn.Linear(3, 3).to(torch.bfloat16)
        self.lora_A = torch.nn.Linear(3, 2, bias=False).to(torch.bfloat16)
        self.lora_B = torch.nn.Linear(2, 3, bias=False).to(torch.bfloat16)


class ToyExpertPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.action_in_proj = torch.nn.Linear(3, 3).to(torch.bfloat16)
        self.model.action_out_proj = torch.nn.Linear(3, 3).to(torch.bfloat16)
        container = torch.nn.Module()
        self.model.paligemma_with_expert = container
        container.gemma_expert = torch.nn.Module()
        container.gemma_expert.model = torch.nn.Module()
        container.gemma_expert.model.proj = ToyAdapter()
        container.gemma_expert.model.norm = torch.nn.Linear(3, 3).to(torch.bfloat16)
        container.gemma_expert.lm_head = torch.nn.Linear(3, 3).to(torch.bfloat16)
        container.paligemma = torch.nn.Module()
        container.paligemma.model = torch.nn.Module()
        container.paligemma.model.language_model = torch.nn.Module()
        container.paligemma.model.language_model.proj = ToyAdapter()


class StreamTrainingTests(unittest.TestCase):
    def test_cli_action_prefix_defaults_to_50_and_accepts_10(self):
        common = ['train', '--dataset', '/dataset', '--run', '/run', '--steps', '1']
        for extra, expected in (([], 50), (['--n-action-steps', '10'], 10)):
            with self.subTest(expected=expected), patch.object(sys, 'argv', common + extra), \
                    patch.object(training, 'run') as run:
                training.main()
                self.assertEqual(run.call_args.args[0].n_action_steps, expected)
        for value in ('0', '51'):
            with self.subTest(value=value), patch.object(sys, 'argv', common + ['--n-action-steps', value]), \
                    patch('sys.stderr'), self.assertRaises(SystemExit):
                training.main()
        for parent in ('--resume', '--initialize'):
            with self.subTest(parent=parent), patch.object(sys, 'argv', common + [parent, '/parent']), \
                    patch.object(training, 'run') as run:
                training.main()
                self.assertIsNone(run.call_args.args[0].n_action_steps)

    def test_action_prefix_inherits_resume_and_requires_explicit_new_run_to_change(self):
        resolve = training.resolve_action_steps
        self.assertEqual(resolve(None), 50)
        self.assertEqual(resolve(10), 10)
        self.assertEqual(resolve(None, parent_steps=50, resume=True), 50)
        self.assertEqual(resolve(None, parent_steps=10, resume=True), 10)
        self.assertEqual(resolve(50, parent_steps=50, resume=True), 50)
        self.assertEqual(resolve(10, parent_steps=50, resume=False), 10)
        self.assertEqual(resolve(None, parent_steps=10, resume=False), 10)
        with self.assertRaisesRegex(ValueError, 'sealed parent'):
            resolve(10, parent_steps=50, resume=True)
        with self.assertRaisesRegex(ValueError, 'sealed parent'):
            resolve(None, resume=True)
        for value in (0, 51, True, 10.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve(value)

    def test_image_profile_cli_and_resume_preserve_sealed_views(self):
        resolve = training.resolve_image_profile
        full = training.checkpoints.FULL_IMAGE_PROFILE
        cropped = training.checkpoints.WORKSPACE_IMAGE_PROFILE
        self.assertEqual(resolve(None), full)
        self.assertEqual(resolve(None, parent_profile=cropped, resume=True), cropped)
        self.assertEqual(resolve(cropped, parent_profile=full, resume=False), cropped)
        with self.assertRaisesRegex(ValueError, 'sealed parent'):
            resolve(full, parent_profile=cropped, resume=True)
        with self.assertRaises(ValueError):
            resolve('unknown')
        common = ['train', '--dataset', '/dataset', '--run', '/run', '--steps', '1']
        for extra, expected in (([], full), (['--image-profile', cropped], cropped),
                                (['--resume', '/parent'], None)):
            with self.subTest(extra=extra), patch.object(sys, 'argv', common + extra), \
                    patch.object(training, 'run') as run:
                training.main()
                self.assertEqual(run.call_args.args[0].image_profile, expected)

    def test_sampling_profile_inherits_parent_and_cannot_change_on_resume(self):
        old = {'sampling': training.TransitionSampler.VERSION}
        new = {'sampling_profile': training.APPROACH_SAMPLING_PROFILE,
               'sampling': training.conditioned_sampling.APPROACH_OPEN_VERSION}
        hold = {'sampling_profile': training.UNIFORM_SWITCH_HOLD_PROFILE,
                'sampling': training.UniformSwitchHoldSampler.VERSION}
        switch = {'sampling': training.UniformSwitchSampler.VERSION}
        resolve = training.resolve_sampling_profile
        self.assertEqual(resolve(None), training.DEFAULT_SAMPLING_PROFILE)
        self.assertEqual(resolve(None, parent_report=old, resume=True),
                         training.DEFAULT_SAMPLING_PROFILE)
        self.assertEqual(resolve(None, parent_report=new, resume=True),
                         training.APPROACH_SAMPLING_PROFILE)
        self.assertEqual(resolve(None, parent_report=hold, resume=True), training.UNIFORM_SWITCH_HOLD_PROFILE)
        self.assertEqual(resolve(None, parent_report=switch, resume=True), training.UNIFORM_SWITCH_PROFILE)
        self.assertEqual(resolve(training.APPROACH_SAMPLING_PROFILE, parent_report=old),
                         training.APPROACH_SAMPLING_PROFILE)
        with self.assertRaisesRegex(ValueError, 'same sampler'):
            resolve(training.APPROACH_SAMPLING_PROFILE, parent_report=old, resume=True)
        with self.assertRaisesRegex(ValueError, 'same sampler'):
            resolve(None, parent_report={**new, 'sampling': 'mismatched'}, resume=True)
        with self.assertRaisesRegex(ValueError, 'same sampler'):
            resolve(training.UNIFORM_SWITCH_HOLD_PROFILE, parent_report=switch, resume=True)
        argv = ['train', '--dataset', '/dataset', '--run', '/run', '--steps', '1',
                '--sampling-profile', training.UNIFORM_SWITCH_HOLD_PROFILE]
        with patch.object(sys, 'argv', argv), patch.object(training, 'run') as run:
            training.main()
            self.assertEqual(run.call_args.args[0].sampling_profile, training.UNIFORM_SWITCH_HOLD_PROFILE)

    def test_observation_context_profile_inherits_and_resume_cannot_change(self):
        resolve = training.resolve_observation_context_profile
        self.assertEqual(resolve(None), 'none')
        self.assertEqual(resolve(None, parent_profile='state-history-v1', resume=True), 'state-history-v1')
        self.assertEqual(resolve('state-history-v1', parent_profile='none'), 'state-history-v1')
        with self.assertRaisesRegex(ValueError, 'same observation context'):
            resolve('none', parent_profile='state-history-v1', resume=True)
        with self.assertRaises(ValueError):
            resolve('unknown')
        argv = ['train', '--dataset', '/dataset', '--run', '/run', '--steps', '1',
                '--initialize', '/parent', '--observation-context-profile', 'state-history-v1']
        with patch.object(sys, 'argv', argv), patch.object(training, 'run') as run:
            training.main()
            self.assertEqual(run.call_args.args[0].observation_context_profile, 'state-history-v1')

    def test_fixed_setdown_profile_supports_warmstart_and_sealed_resume(self):
        history = training.observation_history
        profile = history.CUP_SETDOWN_FIXED_PROFILE
        resolve = training.resolve_observation_context_profile
        self.assertEqual(resolve(profile, parent_profile=history.CUP_SETDOWN_PROFILE), profile)
        self.assertEqual(resolve(None, parent_profile=profile, resume=True), profile)
        self.assertEqual(resolve(profile, parent_profile=profile, resume=True), profile)
        with self.assertRaisesRegex(ValueError, 'same observation context'):
            resolve(profile, parent_profile=history.CUP_SETDOWN_PROFILE, resume=True)
        argv = ['train', '--dataset', '/dataset', '--run', '/run', '--steps', '128',
                '--initialize', '/parent', '--observation-context-profile', profile]
        with patch.object(sys, 'argv', argv), patch.object(training, 'run') as run:
            training.main()
            self.assertEqual(run.call_args.args[0].observation_context_profile, profile)
            self.assertEqual(run.call_args.args[0].initialize, Path('/parent'))

    def test_fixed_setdown_warmstart_keeps_original_normalization_and_wait_conditions(self):
        history = training.observation_history
        rows = sample_rows((1, 1))
        episodes = training.validate_rows(rows)
        proof = {'sources': [{'source_scope_sha256': f'scope-{i}', 'user_review_sha256': f'review-{i}',
                              'split': 'train' if i == 0 else 'validation'} for i in range(2)]}
        report = {'rank': 32, 'source_provenance': proof, 'sampling_profile': training.CUP_COUPLED_SAMPLING_PROFILE}
        saved_stats = {name: {'q01': [0.] * size, 'q99': [1.] * size}
                       for name, size in (('observation.state', 8), ('action', 7))}
        current_stats = {name: {'q01': torch.zeros(size), 'q99': torch.full((size,), 2.)}
                         for name, size in (('observation.state', 8), ('action', 7))}
        contract = {'name': 'test-stream'}
        parent = {'manifest': {'policy_contract': contract}, 'observation_context_profile': history.CUP_SETDOWN_PROFILE,
                  'action_representation': training.checkpoints.CUP_TICK_DELTA_REPRESENTATION,
                  'adaptation_mode': training.LORA_ADAPTATION, 'n_action_steps': 3,
                  'image_profile': training.FULL_IMAGE_PROFILE}
        texts = {'/unit/dataset/meta/stream_stats.json': json.dumps({'stats': {}}),
                 '/unit/parent/training-report.json': json.dumps(report),
                 '/unit/parent/normalization.json': json.dumps(saved_stats)}
        legacy = SimpleNamespace(ROOT=Path('/unit'), require_runtime_versions=lambda: None,
            configure_environment=lambda: None, check_data4=Path, gpu_status=Mock(return_value={'index': 0}),
            sha=lambda _path: 'base', EXPECTED_WEIGHTS_SHA256='base')
        relative = SimpleNamespace(compute_relative_action_stats=lambda *_a, **_k:
                                   (current_stats, {'fallbacks': []}))
        for profile in (history.CUP_SETDOWN_PROFILE, history.CUP_SETDOWN_FIXED_PROFILE,
                        history.CUP_SETDOWN_CLEAN_PROFILE):
            config = SimpleNamespace(n_action_steps=3, tokenizer_max_length=200)
            data = SimpleNamespace(attach_cup_relative=Mock(), attach_camera_aligned_actions=Mock(),
                attach_cup_tracking=Mock(side_effect=RuntimeError('observation preparation complete')))
            roll = Mock(side_effect=lambda stats, *_a, **_k: (stats, None, None, False))
            args = SimpleNamespace(n_action_steps=3, dataset='/unit/dataset', run='/unit/checkpoints/run',
                base='/unit/base', gpu=0, resume=None, initialize='/unit/parent', observation_context_profile=profile,
                action_representation=parent['action_representation'], adaptation=training.LORA_ADAPTATION,
                rank=32, seed=25, image_profile=training.FULL_IMAGE_PROFILE,
                sampling_profile=training.CUP_COUPLED_SAMPLING_PROFILE)
            modules = {'peft': SimpleNamespace(get_peft_model_state_dict=lambda _: {}),
                       'lerobot.policies.pi05.configuration_pi05': SimpleNamespace(
                           PI05Config=SimpleNamespace(from_pretrained=lambda *_a, **_k: config)),
                       'cup_relative_stream_data': data}
            with self.subTest(profile=profile), ExitStack() as stack:
                for context in (
                        patch.object(training, 'legacy', legacy), patch.dict(sys.modules, modules),
                        patch.dict('os.environ'), patch.object(Path, 'is_file', return_value=True),
                        patch.object(Path, 'read_text', autospec=True, side_effect=lambda path, *_a, **_k: texts[str(path)]),
                        patch.object(torch.cuda, 'device_count', return_value=1), patch.object(torch, 'set_num_threads'),
                        patch.object(torch, 'manual_seed'), patch.object(training.random, 'seed'), patch('numpy.random.seed'),
                        patch.object(training, 'read_dataset', return_value=(rows, episodes, object(), proof, 'dataset')),
                        patch.object(training.checkpoints, 'verify_bundle', return_value=parent),
                        patch.object(training, 'calculate_stats', side_effect=lambda *_a, **_k: (copy.deepcopy(current_stats), [])),
                        patch.object(training, 'load_relative_helper', return_value=relative),
                        patch.object(training, 'load_contract', return_value=(SimpleNamespace(contract=lambda: contract), None)),
                        patch.object(training, 'load_policy'), patch.object(training, 'configure_cup_roll_state_stats', roll),
                        patch.object(training, 'attach_initial_cup')):
                    stack.enter_context(context)
                with self.assertRaisesRegex(RuntimeError, 'observation preparation complete'):
                    training.run(args)
            self.assertEqual(data.attach_cup_tracking.call_args.kwargs, {
                'with_timing': True, 'aligned': True, 'setdown': True, 'execution_steps': 3,
                'mask_settled_wait_in_prefix': profile == history.CUP_SETDOWN_CLEAN_PROFILE,
                'mask_pickup_wait': True})
            self.assertFalse(roll.call_args.kwargs['rebase_stats'])
            for name, values in saved_stats.items():
                for key, expected in values.items():
                    self.assertTrue(torch.equal(roll.call_args.args[0][name][key], torch.tensor(expected)))
            self.assertEqual(config.tokenizer_max_length, 512)

    def test_history_context_fixes_tokenizer_width_from_first_batch(self):
        from fr5_lerobot.tests.test_stream_prompt_integrity import Pipeline, batch
        config = SimpleNamespace(tokenizer_max_length=200)
        training.configure_observation_context(config, 'none')
        self.assertEqual(config.tokenizer_max_length, 200)
        training.configure_observation_context(config, 'state-history-v1')
        self.assertEqual(config.tokenizer_max_length, 512)
        pre = Pipeline()
        pre.steps[1].max_length = config.tokenizer_max_length
        history = [{'age_s': age, 'state': [1., -1., 1., -1., 1., -1., .78, 1.]}
                   for age in training.observation_history.HISTORY_OFFSETS_S]
        early = training.observation_history.render_task('抓杯', [None] * 6, 'state-history-v1')
        later = training.observation_history.render_task('抓杯', history, 'state-history-v1')
        for task in (early, later):
            value = training.prompt_integrity.preprocess(pre, batch(task))
            self.assertEqual(value['observation']['observation.language.tokens'].shape, (1, 512))
            self.assertEqual(pre.steps[1].max_length, 512)
        with self.assertRaisesRegex(ValueError, 'maximum is 512'):
            training.prompt_integrity.preprocess(pre, batch(later + '字' * 400))

    def test_history_uses_each_episode_task_clock_and_preserves_missing_feedback(self):
        rows = sample_rows((13, 13))
        rows[6]['observation.state'][6:] = [0., 0.]
        episodes = training.validate_rows(rows)
        training.attach_observation_history(rows, episodes)
        history = rows[12]['_observation_history']
        self.assertEqual(history[1], {'age_s': 1., 'state': rows[6]['observation.state']})
        self.assertIsNone(history[3])
        self.assertTrue(all(item is None for item in rows[13]['_observation_history']))
        self.assertEqual(rows[25]['_observation_history'][1]['state'], rows[19]['observation.state'])

    def test_physical_history_uses_removed_source_frames_and_actual_camera_age(self):
        import math
        from fr5_lerobot.tests.test_stream_history_runtime import recorder, sample
        from lerobot_robot_fairino.pi05_bridge import observation_state
        profile = training.observation_history.PHYSICAL_HISTORY_PROFILE
        base = 1000.
        live = recorder(profile)
        captures = []
        for index in range(13):
            stamp = base + index * .5 + (.13 if index == 12 else 0.)
            observed = sample(index, stamp)
            if index == 10:
                observed['robot_state']['gripper_motiondone'] = 0
            live._sample(observed)
            captures.append({'frame_index': index, 'camera': observed['camera']['metadata'],
                             'robot': observed['robot_state']})
        rows = sample_rows((3,))
        kept = (0, 1, 12)
        for row, source_frame in zip(rows, kept):
            capture = captures[source_frame]
            row['_source_capture_frame'] = source_frame
            row['_camera_time'] = capture['camera']['source_timestamp_monotonic']
            row['observation.state'] = observation_state(capture['robot'])
        episodes = training.validate_rows(rows)
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory)
            (trial / 'continuous').mkdir()
            (trial / 'continuous/capture.jsonl').write_text(
                ''.join(json.dumps(value) + '\n' for value in captures))
            proof = {'training_data_kind': 'synthetic_diagnostic', 'sources': [{
                'trial': str(trial), 'data_provenance': {'training_data_kind': 'synthetic_diagnostic'}}]}
            training.attach_observation_history(rows, episodes)
            training.attach_physical_observation_history(rows, episodes, proof)
        physical = rows[-1]['_physical_observation_history']
        self.assertEqual(physical, live._observation_with_history(sample(12, base + 6.13))['observation_history'])
        self.assertAlmostEqual(physical[0]['age_s'], .63)
        self.assertAlmostEqual(physical[0]['state'][0], math.radians(12.))
        self.assertAlmostEqual(physical[1]['age_s'], 1.13)
        self.assertEqual(physical[1]['state'][6:], [0., 0.])
        self.assertTrue(all(value is None for value in rows[-1]['_observation_history']))
        self.assertTrue(all(value is None for value in rows[0]['_physical_observation_history']))

    def test_physical_history_requires_and_checks_capture_receipt(self):
        from fr5_lerobot.tests.test_stream_history_runtime import sample
        captures = []
        for index in range(4):
            observed = sample(index, 1000. + index * .5)
            captures.append({'frame_index': index, 'camera': observed['camera']['metadata'],
                             'robot': observed['robot_state']})
        rows = [{'_physical_source': '0', '_source_capture_frame': index,
                 '_camera_time': 1000. + index * .5} for index in (0, 3)]
        episodes = {0: [0, 1]}
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory)
            (trial / 'continuous').mkdir()
            journal = trial / 'continuous/capture.jsonl'
            raw = ''.join(json.dumps(value) + '\n' for value in captures).encode('utf-8')
            journal.write_bytes(raw)
            source = {'trial': str(trial), 'capture_journal_sha256': hashlib.sha256(raw).hexdigest()}
            proof = {'training_data_kind': 'captured', 'sources': [source]}
            training.attach_physical_observation_history(rows, episodes, proof)
            self.assertAlmostEqual(rows[1]['_physical_observation_history'][0]['age_s'], .5)

            changed = copy.deepcopy(captures)
            changed[2]['robot']['joints'][0] += 1.
            journal.write_text(''.join(json.dumps(value) + '\n' for value in changed))
            with self.assertRaisesRegex(ValueError, 'capture journal differs from provenance'):
                training.attach_physical_observation_history(rows, episodes, proof)
            proof['training_data_kind'] = 'synthetic_diagnostic'
            with self.assertRaisesRegex(ValueError, 'capture journal differs from provenance'):
                training.attach_physical_observation_history(rows, episodes, proof)

            journal.write_bytes(raw)
            del source['capture_journal_sha256']
            proof['training_data_kind'] = 'captured'
            with self.assertRaisesRegex(ValueError, 'requires a capture journal receipt'):
                training.attach_physical_observation_history(rows, episodes, proof)
            proof['training_data_kind'] = 'synthetic_diagnostic'
            source['data_provenance'] = {'training_data_kind': 'captured'}
            with self.assertRaisesRegex(ValueError, 'requires a capture journal receipt'):
                training.attach_physical_observation_history(rows, episodes, proof)
            source['data_provenance']['training_data_kind'] = 'synthetic_diagnostic'
            training.attach_physical_observation_history(rows, episodes, proof)

    @patch.object(training.prompt_integrity, 'preprocess', side_effect=lambda pre, value: pre(value))
    def test_history_training_and_service_inputs_match_and_keep_raw_task(self, _guard):
        import numpy as np
        import serve_stream_pi05 as service
        rows = sample_rows((13,))
        episodes = training.validate_rows(rows)
        rows[6]['observation.state'][6:] = [0., 0.]
        training.attach_observation_history(rows, episodes)
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        task = '抓起杯子，悬停后放下'
        class Samples:
            def __getitem__(self, index):
                return {training.IMAGE_KEY: torch.from_numpy(image).permute(2, 0, 1).float() / 255,
                        'task': task}
        raw_history = copy.deepcopy(rows[12]['_observation_history'])
        batch, _ = training.batch_for([12], rows, episodes, Samples(), lambda value: value, labels=False,
                                      observation_context_profile='state-history-v1')
        served = service.prepare_request_batch(training, lambda value: value, rows[12]['observation.state'],
                                               image, task, training.FULL_IMAGE_PROFILE,
                                               'state-history-v1', raw_history)
        self.assertEqual(batch['task'], served['task'])
        self.assertTrue(batch['task'][0].startswith(task))
        self.assertIn('0,0', batch['task'][0])
        self.assertEqual(raw_history, rows[12]['_observation_history'])
        self.assertEqual(task, '抓起杯子，悬停后放下')
        self.assertTrue(torch.equal(batch['observation.state'], served['observation.state']))

    @patch.object(training.prompt_integrity, 'preprocess', side_effect=lambda pre, value: pre(value))
    def test_physical_history_training_and_service_use_the_same_input(self, _guard):
        import numpy as np
        import serve_stream_pi05 as service
        profile = training.observation_history.PHYSICAL_HISTORY_PROFILE
        rows = sample_rows((1,))
        physical = [{'age_s': .63, 'state': [1., 2., 3., 4., 5., 6., 0., 0.]}] + [None] * 5
        rows[0]['_physical_observation_history'] = physical
        rows[0]['_observation_history'] = [None] * 6
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        task = '抓取杯子，停留5秒后放下'
        class Samples:
            def __getitem__(self, index):
                return {training.IMAGE_KEY: torch.from_numpy(image).permute(2, 0, 1).float() / 255,
                        'task': task}
        batch, _ = training.batch_for([0], rows, {0: [0]}, Samples(), lambda value: value,
                                      labels=False, observation_context_profile=profile)
        served = service.prepare_request_batch(training, lambda value: value, rows[0]['observation.state'],
                                               image, task, training.FULL_IMAGE_PROFILE, profile, physical)
        self.assertEqual(batch['task'], served['task'])
        self.assertIn('0.63:1000,2000,3000,4000,5000,6000,0,0', batch['task'][0])
        self.assertEqual(training.observation_history.context_metadata(profile)['training_clock'],
                         'host_monotonic_camera_time')
        self.assertTrue(torch.equal(batch['observation.state'], served['observation.state']))

    @patch.object(training.prompt_integrity, 'preprocess', side_effect=lambda pre, value: pre(value))
    def test_fixed_setdown_train_and_serve_change_only_reserved_action_noise(self, _guard):
        import numpy as np
        import serve_stream_pi05 as service
        history = training.observation_history
        rows = sample_rows((3,))
        rgb = np.arange(360 * 640 * 3, dtype=np.uint8).reshape(360, 640, 3)
        task = '抓起杯子，在空中维持5秒，然后放下'
        for index, phase in enumerate(('at_pickup', 'hover', 'returning')):
            state = [.01 * index, -.02, .1, 1.617, -.02, -1.17, 1. if index == 0 else .76, 1.]
            rows[index].update({
                '_cup_relative_state': state,
                '_cup_relative_history': [{'age_s': .63, 'state': state[:6] + [0., 0.]}] + [None] * 5,
                '_physical_observation_history': [None] * 6,
                '_initial_cup_uv': [1429.53, 696.82],
                '_cup_tracking': {'rim_minus_jaw_mm': [12., -5., 3.] if index != 2 else None,
                                  'native_contact': 0 if index == 0 else 1},
                '_cup_timing': {'phase': phase, 'hover_seconds': 4.6 if index == 1 else None,
                                'visual': 'table_height' if index == 0 else 'raised' if index == 1 else 'missing',
                                'support_gap_mm': 50. if index == 1 else 0.,
                                'setdown_age_s': .15 if index == 2 else None},
                '_cup_relative_action': [9.] * 6 + [.76],
                '_camera_aligned_actions': [[state[0] + tick * .0005, *state[1:6],
                                             1. if index == 0 and tick <= 2 else .76] for tick in range(1, 5)],
                '_cup_table_normal': torch.tensor([0., 0., 1.]),
                '_recorded_hold_remaining_s': .4 if index == 1 else None,
                '_settled_wait_mask': [False, False, False, index == 2],
                '_before_setdown_mask': [index == 2, False, False, False],
                '_pickup_wait_mask': [index == 0, index == 0, False, False],
            })
        class Samples:
            def __getitem__(self, index):
                return {training.IMAGE_KEY: torch.from_numpy(rgb).permute(2, 0, 1).float() / 255, 'task': task}
        episodes = {0: [0, 1, 2]}
        batches, embedded = {}, {}
        image_profile = training.checkpoints.CUP_CENTERED_IMAGE_PROFILE
        noisy = torch.arange(3 * 50 * 32, dtype=torch.float32).reshape(3, 50, 32) / 100.
        other_noise = noisy.clone()
        other_noise[..., 7:] += 2.
        policy = SimpleNamespace(model=SimpleNamespace(embed_suffix=lambda xs, _time: xs))
        rng = torch.get_rng_state().clone()
        for profile in (history.CUP_SETDOWN_PROFILE, history.CUP_SETDOWN_FIXED_PROFILE,
                        history.CUP_SETDOWN_CLEAN_PROFILE):
            with self.subTest(profile=profile):
                batch, valid = training.batch_for([0, 1, 2], rows, episodes, Samples(), lambda value: value,
                    labels=True, valid_device='cpu', image_profile=image_profile,
                    action_representation=training.checkpoints.CUP_TICK_DELTA_REPRESENTATION,
                    observation_context_profile=profile, cup_boundary_steps=3)
                prediction, _ = training.batch_for([0, 1, 2], rows, episodes, Samples(), lambda value: value,
                    labels=False, valid_device='cpu', image_profile=image_profile, observation_context_profile=profile)
                batches[profile] = batch, valid
                fixed = profile != history.CUP_SETDOWN_PROFILE
                self.assertEqual(batch.get('observation.fixed_cup_action_clock', False), fixed)
                self.assertEqual(prediction.get('observation.fixed_cup_action_clock', False), fixed)
                self.assertEqual(tuple(batch['action'].shape), (3, 50, 7))
                self.assertEqual(valid.sum(dim=1).tolist(), [4, 4, 4])
                with training.action_clock_condition(policy, batch):
                    embedded[profile] = policy.model.embed_suffix(noisy, None)
                    changed = policy.model.embed_suffix(other_noise, None)
                self.assertTrue(torch.equal(embedded[profile][..., :7], noisy[..., :7]))
                if fixed:
                    self.assertTrue(torch.equal(embedded[profile], changed))
                for index, row in enumerate(rows):
                    served = service.prepare_request_batch(training, lambda value: value, row['_cup_relative_state'],
                        rgb, task, image_profile, profile, row['_cup_relative_history'],
                        initial_cup_uv=row['_initial_cup_uv'], cup_tracking=row['_cup_tracking'], cup_timing=row['_cup_timing'])
                    self.assertEqual(served.get('observation.fixed_cup_action_clock', False), fixed)
                    self.assertEqual(served['task'], batch['task'][index:index + 1])
                    for key in ('observation.state', training.IMAGE_KEY, 'observation.cup_action_clock'):
                        self.assertTrue(torch.equal(served[key], batch[key][index:index + 1]), key)
                        self.assertTrue(torch.equal(served[key], prediction[key][index:index + 1]), key)
                    with training.action_clock_condition(policy, served):
                        conditioned = policy.model.embed_suffix(noisy[index:index + 1], None)
                    self.assertTrue(torch.equal(conditioned, embedded[profile][index:index + 1]))
        original, original_valid = batches[history.CUP_SETDOWN_PROFILE]
        fixed, fixed_valid = batches[history.CUP_SETDOWN_FIXED_PROFILE]
        self.assertTrue(torch.equal(original_valid, fixed_valid))
        self.assertEqual(set(fixed) - {'observation.fixed_cup_action_clock'}, set(original))
        for key, value in original.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, fixed[key]), key)
            elif key == 'cup_boundary':
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(value, fixed[key])))
            else:
                self.assertEqual(value, fixed[key])
        original_input = embedded[history.CUP_SETDOWN_PROFILE]
        fixed_input = embedded[history.CUP_SETDOWN_FIXED_PROFILE]
        self.assertTrue(torch.equal(original_input[..., 7:], noisy[..., 7:] + fixed_input[..., 7:]))
        clean = batches[history.CUP_SETDOWN_CLEAN_PROFILE][0]
        self.assertAlmostEqual(float(fixed['observation.cup_action_clock'][2, -1]), .15)
        self.assertAlmostEqual(float(clean['observation.cup_action_clock'][2, -1]), .5)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))


    @patch.object(training.prompt_integrity, 'preprocess', side_effect=lambda pre, value: pre(value))
    def test_training_batch_supplies_full_and_workspace_before_processor(self, _guard):
        image = torch.arange(3 * 360 * 640, dtype=torch.float32).reshape(3, 360, 640)
        rows = sample_rows((2,))
        class Samples:
            def __getitem__(self, index):
                return {training.IMAGE_KEY: image, 'task': 'cup'}
        observed = []
        batch, _ = training.batch_for([0], rows, {0: [0, 1]}, Samples(),
                                      lambda value: observed.append(value) or value, labels=False,
                                      image_profile=training.checkpoints.WORKSPACE_IMAGE_PROFILE)
        self.assertIs(batch, observed[0])
        self.assertTrue(torch.equal(batch[training.IMAGE_KEY][0], image))
        self.assertTrue(torch.equal(batch['observation.images.left_wrist_0_rgb'][0], image[:, 100:360, 240:640]))

    @patch.object(training.prompt_integrity, 'preprocess', side_effect=lambda pre, value: pre(value))
    def test_training_and_serving_use_same_profile_before_processor(self, _guard):
        import numpy as np
        import serve_stream_pi05 as service
        rgb = np.arange(360 * 640 * 3, dtype=np.uint8).reshape(360, 640, 3)
        image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
        rows = sample_rows((2,))
        class Samples:
            def __getitem__(self, index):
                return {training.IMAGE_KEY: image, 'task': 'cup'}
        profile = training.checkpoints.WORKSPACE_IMAGE_PROFILE
        training_batch, _ = training.batch_for([0], rows, {0: [0, 1]}, Samples(), lambda x: x,
                                               labels=False, image_profile=profile)
        serving_batch = service.prepare_request_batch(training, lambda x: x, rows[0]['observation.state'],
                                                      rgb, 'cup', profile)
        for key in (training.IMAGE_KEY, 'observation.images.left_wrist_0_rgb', 'observation.state'):
            self.assertTrue(torch.equal(training_batch[key], serving_batch[key]))

    def test_unweighted_loss_is_mean_over_valid_targets_and_dimensions(self):
        model = ToyFlow()
        policy = SimpleNamespace(model=model, _preprocess_images=lambda batch: ([], []),
                                 prepare_action=lambda batch: batch['action'])
        valid = torch.tensor([[True, True, True] + [False] * 47])
        actions = torch.arange(350, dtype=torch.float32).reshape(1, 50, 7) / 10
        batch = {'action': actions, 'observation.language.tokens': torch.ones(1, 2),
                 'observation.language.attention_mask': torch.ones(1, 2)}
        expected = (actions[:, :3, :] - actions[:, :3, :].mean(dim=1, keepdim=True)).square().mean()
        actual = training.masked_flow_loss(policy, batch, valid, first_action_weight=1, gripper_weight=1)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertEqual(training.loss_normalizer([3], 1, 1), 21)

    def test_source_helpers_load_without_project_on_pythonpath(self):
        with patch.object(sys, 'path', [p for p in sys.path if '/fr5_lerobot/src' not in p]):
            timing, feedback = training.load_record_helpers()
        self.assertEqual(timing.camera_clock({'source_timestamp_monotonic': 10.}), (10., timing.MONOTONIC))
        self.assertFalse(feedback.gripper_feedback_valid({}))

    def test_full_expert_is_trainable_and_roundtrips_every_tensor(self):
        from safetensors.torch import save_file
        model = ToyExpertPolicy()
        training.trainable_parameters(model, training.FULL_EXPERT_ADAPTATION)
        values = dict(model.named_parameters())
        expert_names = {name for name in values if training.is_expert_base(name)}
        self.assertTrue(expert_names)
        self.assertTrue(all(values[name].requires_grad and values[name].dtype == torch.float32
                            for name in expert_names))
        frozen_expert_adapters = [p for name, p in values.items()
                                  if '.gemma_expert.model.' in name and '.lora_' in name]
        self.assertTrue(frozen_expert_adapters)
        self.assertTrue(all(not p.requires_grad and p.dtype == torch.float32 for p in frozen_expert_adapters))
        language_adapters = [p for name, p in values.items()
                             if '.paligemma.model.language_model.' in name and '.lora_' in name]
        self.assertTrue(all(p.requires_grad and p.dtype == torch.float32 for p in language_adapters))
        self.assertFalse(model.model.paligemma_with_expert.gemma_expert.lm_head.weight.requires_grad)
        saved = {name: value.clone() for name, value in training.expert_state(model).items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'expert.safetensors'
            save_file(saved, str(path))
            with torch.no_grad():
                for name in expert_names:
                    values[name].add_(10.)
            training.restore_expert(model, path)
            self.assertTrue(all(torch.equal(values[name].detach(), value) for name, value in saved.items()))
            save_file({k: v for k, v in saved.items() if k != next(iter(saved))}, str(path))
            with self.assertRaisesRegex(ValueError, 'do not match'):
                training.restore_expert(model, path)

    def test_lora_mode_keeps_expert_base_frozen(self):
        model = ToyExpertPolicy()
        training.trainable_parameters(model)
        values = dict(model.named_parameters())
        self.assertTrue(all(not p.requires_grad for name, p in values.items() if training.is_expert_base(name)))
        self.assertTrue(all(p.requires_grad for name, p in values.items()
                            if '.gemma_expert.model.' in name and '.lora_' in name))

    def test_each_tail_anchor_keeps_real_release_and_never_crosses_episode(self):
        rows = sample_rows()
        episodes = training.validate_rows(rows)
        targets, valid = training.chunk_for_anchor(rows, episodes[0], 2)
        self.assertEqual(valid, [True, True] + [False] * 48)
        self.assertEqual(targets[1], rows[3]['action'])
        self.assertNotEqual(targets[1], rows[4]['action'])
        self.assertTrue(all(x == targets[1] for x in targets[2:]))
        last, last_valid = training.chunk_for_anchor(rows, episodes[0], 3)
        self.assertEqual(sum(last_valid), 1)
        self.assertEqual(last[0][6], .7)

    @patch.object(training.prompt_integrity, 'preprocess', side_effect=lambda pre, value: pre(value))
    def test_relative_batch_and_wire_roundtrip_use_each_raw_anchor(self, _guard):
        import serve_stream_pi05 as service
        rows = sample_rows((4, 3))
        episodes = training.validate_rows(rows)
        class Samples:
            def __getitem__(self, index):
                return {training.IMAGE_KEY: torch.zeros(3, 2, 2), 'task': 'cup'}
        batch, valid = training.batch_for([0, 2], rows, {0: episodes[0]}, Samples(), lambda x: x,
            labels=True, action_representation=training.RELATIVE_REPRESENTATION, valid_device='cpu')
        self.assertEqual(batch['action'].shape, (2, 50, 7))
        self.assertEqual(valid.sum(dim=1).tolist(), [4, 2])
        helper = training.load_relative_helper()
        for b, index in enumerate((0, 2)):
            absolute = service.actions_for_wire(batch['action'][b:b+1], rows[index]['observation.state'], helper)
            targets, _ = training.chunk_for_anchor(rows, episodes[0], index)
            torch.testing.assert_close(absolute[0], torch.tensor(targets, dtype=torch.float32))
            self.assertTrue(torch.equal(absolute[0, :, 6], batch['action'][b, :, 6]))

    def test_representation_metadata_keeps_wire_absolute(self):
        metadata = training.action_representation_metadata(training.RELATIVE_REPRESENTATION)
        self.assertEqual(metadata['mode'], training.RELATIVE_REPRESENTATION)
        self.assertEqual(metadata['wire_mode'], training.ABSOLUTE_REPRESENTATION)
        self.assertFalse(metadata['official_relative_processor_enabled'])

    def test_heldout_source_never_changes_training_normalization(self):
        first = sample_rows(offset=100.)
        episodes = training.validate_rows(first)
        train, val, held = training.split_sources(first, episodes)
        self.assertEqual(held, ['1'])
        self.assertEqual(set(train), {0})
        self.assertEqual(set(val), {1})
        stats_a, _ = training.calculate_stats(first, train)
        second = sample_rows(offset=10000.)
        stats_b, _ = training.calculate_stats(second, train)
        for feature in stats_a:
            for key in stats_a[feature]:
                self.assertTrue(torch.equal(stats_a[feature][key], stats_b[feature][key]))
        self.assertGreater(float(stats_a['action']['q01'][6]), .7)
        self.assertEqual(float(stats_a['action']['q99'][6]), 1.)

    def test_cup_roll_span_uses_only_training_extrema_and_preserves_other_stats(self):
        rows = [{'_split': 'train', '_physical_source': str(i),
                 '_cup_relative_state': [0., 0., 0., 1.618 if i == 0 else 1.620, 0., 0., 1., 1.]}
                for i in range(7)]
        rows.append({'_split': 'validation', '_physical_source': '7',
                     '_cup_relative_state': [0., 0., 0., 4., 0., 0., 1., 1.]})
        stats = {'observation.state': {'q01': torch.tensor([0., 1., 2., 1.618, 4., 5., 6., 7.]),
                                       'q99': torch.tensor([1., 2., 3., 1.620, 5., 6., 7., 8.])},
                 'action': {'q01': torch.arange(7.), 'q99': torch.arange(7.) + 1.}}
        original = copy.deepcopy(stats)
        widened, span, evidence, applied = training.configure_cup_roll_state_stats(
            stats, rows, {i: [i] for i in range(7)}, .010, parent_report={}, cup_relative=True)
        self.assertEqual(span, .010)
        self.assertTrue(applied)
        self.assertAlmostEqual(float(widened['observation.state']['q01'][3]), 1.614, places=6)
        self.assertAlmostEqual(float(widened['observation.state']['q99'][3]), 1.624, places=6)
        self.assertGreaterEqual(float(widened['observation.state']['q99'][3])
                                - float(widened['observation.state']['q01'][3]), .010)
        self.assertEqual((evidence['training_min_rad'], evidence['training_max_rad']), (1.618, 1.620))
        for key in ('q01', 'q99'):
            self.assertTrue(torch.equal(widened['action'][key], original['action'][key]))
            self.assertTrue(torch.equal(stats['observation.state'][key], original['observation.state'][key]))
            self.assertTrue(torch.equal(widened['observation.state'][key][[0, 1, 2, 4, 5, 6, 7]],
                                        original['observation.state'][key][[0, 1, 2, 4, 5, 6, 7]]))
        with self.assertRaisesRegex(ValueError, 'training-source'):
            training.apply_cup_roll_state_min_span(stats, rows, {7: [7]}, .010)
        with self.assertRaisesRegex(ValueError, 'sources 0–6'):
            training.apply_cup_roll_state_min_span(stats, rows, {0: [0]}, .010)
        low, high = training.cup_roll_state_bounds(1., 1.002, .010, (.98, 1.02))
        self.assertAlmostEqual(low, .981, places=6)
        self.assertAlmostEqual(high, 1.021, places=6)
        self.assertGreaterEqual(high - low, .04)

    def test_cup_roll_span_inherits_parent_and_resume_cannot_change_it(self):
        resolve = training.resolve_cup_roll_state_min_span
        self.assertIsNone(resolve(None, cup_relative=True))
        self.assertIsNone(resolve(None, parent_report={}, resume=True, cup_relative=True))
        parent = {'cup_roll_state_min_span_rad': .010}
        self.assertEqual(resolve(None, parent_report=parent, resume=True, cup_relative=True), .010)
        self.assertEqual(resolve(.010, parent_report=parent, resume=True, cup_relative=True), .010)
        self.assertEqual(resolve(.020, parent_report=parent, cup_relative=True), .020)
        with self.assertRaisesRegex(ValueError, 'cannot change'):
            resolve(.020, parent_report=parent, resume=True, cup_relative=True)
        with self.assertRaisesRegex(ValueError, 'cannot change'):
            resolve(.010, parent_report={}, resume=True, cup_relative=True)
        with self.assertRaisesRegex(ValueError, 'cup-relative'):
            resolve(.010, cup_relative=False)
        for value in (0., -1., float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve(value, cup_relative=True)
        common = ['train', '--dataset', '/dataset', '--run', '/run', '--steps', '1']
        for extra, expected in (([], None), (['--cup-roll-state-min-span-rad', '.010'], .010)):
            with patch.object(sys, 'argv', common + extra), patch.object(training, 'run') as run:
                training.main()
                self.assertEqual(run.call_args.args[0].cup_roll_state_min_span_rad, expected)
        stats = {'observation.state': {'q01': torch.arange(8.), 'q99': torch.arange(8.) + 1.},
                 'action': {'q01': torch.arange(7.), 'q99': torch.arange(7.) + 1.}}
        for parent_report, requested, resume, expected in (
                ({}, None, False, None), (parent, None, False, .010),
                (parent, None, True, .010), (parent, .010, True, .010)):
            with self.subTest(parent_report=parent_report, requested=requested, resume=resume):
                result, span, _, applied = training.configure_cup_roll_state_stats(
                    stats, [], {}, requested, parent_report=parent_report, resume=resume, cup_relative=True)
                self.assertIs(result, stats)
                self.assertEqual(span, expected)
                self.assertFalse(applied)
        with self.assertRaisesRegex(ValueError, 'parent action and state normalization'):
            training.configure_cup_roll_state_stats(stats, [], {}, .010, parent_report=parent,
                                                    cup_relative=True, rebase_stats=True)



    def test_source_cannot_be_split_between_train_and_validation(self):
        rows = sample_rows()
        rows[1]['_split'] = 'validation'
        episodes = training.validate_rows(rows)
        with self.assertRaisesRegex(ValueError, 'exactly one split'):
            training.split_sources(rows, episodes)

    def test_attention_and_loss_ignore_changed_tail_values_and_gradients(self):
        model = ToyFlow()
        policy = SimpleNamespace(model=model, _preprocess_images=lambda batch: ([], []),
                                 prepare_action=lambda batch: batch['action'])
        valid = torch.tensor([[True, True] + [False] * 48])
        actions = torch.zeros(1, 50, 7, requires_grad=True)
        actions.data[0, 0, :] = 1.
        actions.data[0, 1, :] = 3.
        batch = {'action': actions, 'observation.language.tokens': torch.ones(1, 2),
                 'observation.language.attention_mask': torch.ones(1, 2)}
        first = training.masked_flow_loss(policy, batch, valid, first_action_weight=8, gripper_weight=3)
        first.backward()
        grads = actions.grad.detach().clone()
        self.assertTrue(torch.equal(grads[:, 2:], torch.zeros_like(grads[:, 2:])))
        altered = actions.detach().clone()
        altered[:, 2:] = 100000.
        altered.requires_grad_()
        batch['action'] = altered
        second = training.masked_flow_loss(policy, batch, valid, first_action_weight=8, gripper_weight=3)
        second.backward()
        self.assertTrue(torch.allclose(first, second))
        self.assertTrue(torch.allclose(grads[:, :2], altered.grad[:, :2]))
        _, pad, _, _ = model.embed_suffix(altered, torch.zeros(1))
        self.assertTrue(bool(pad.all()), 'temporary patch must be restored')

    def test_mask_requires_real_prefix_and_rejects_holes(self):
        policy = SimpleNamespace()
        for mask in ([False]*50, [True, False, True] + [False]*47):
            with self.assertRaises(ValueError):
                training.masked_flow_loss(policy, {}, torch.tensor([mask]))

    def test_pickup_wait_mask_keeps_pose_and_actual_close_supervision(self):
        model = ToyFlow()
        policy = SimpleNamespace(model=model, _preprocess_images=lambda batch: ([], []),
                                 prepare_action=lambda batch: batch['action'])
        valid = torch.tensor([[True, True, True] + [False] * 47])
        actions = torch.zeros(1, 50, 7, requires_grad=True)
        actions.data[0, :3, 0] = torch.tensor([1., 2., 3.])
        actions.data[0, :3, 6] = torch.tensor([1., 1., .76])
        batch = {'action': actions, 'observation.language.tokens': torch.ones(1, 2),
                 'observation.language.attention_mask': torch.ones(1, 2),
                 'cup_pickup_wait_mask': torch.tensor([[True, True] + [False] * 48])}
        with patch.object(model, 'forward', side_effect=lambda _i, _m, _t, _a, values, *_: values.square()):
            training.masked_flow_loss(policy, batch, valid).backward()
        self.assertEqual(actions.grad[0, :2, 6].abs().sum(), 0.)
        self.assertGreater(actions.grad[0, 2, 6], 0.)
        self.assertTrue(bool((actions.grad[0, :3, 0] > 0.).all()))
        self.assertEqual(actions.grad[0, 3:].abs().sum(), 0.)

    def test_fixed_cup_clock_does_not_change_motor_noise_or_mix_in_reserved_noise(self):
        state = SimpleNamespace(model=SimpleNamespace(embed_suffix=lambda xs, _time: xs))
        clock = torch.zeros(1, 25)
        clock[0, :2] = torch.tensor([.4, 1.])
        clock[0, -1] = 1.
        x = torch.arange(32.).expand(1, 3, 32).clone()
        y = x.clone()
        y[..., 7:] += 2.
        batch = {'observation.cup_action_clock': clock, 'observation.fixed_cup_action_clock': True}
        with training.action_clock_condition(state, batch):
            a = state.model.embed_suffix(x, torch.zeros(1))
            b = state.model.embed_suffix(y, torch.zeros(1))
        torch.testing.assert_close(a[..., :7], x[..., :7])
        torch.testing.assert_close(a[..., 7:], b[..., 7:])
        torch.testing.assert_close(a[0, :, 7], 3. * (.4 - torch.arange(1, 4) / 6.) / 2.)
        self.assertTrue(bool((a[0, :, -1] == 3).all()))
        self.assertIs(state.model.embed_suffix(x, None), x)

    def test_camera_clock_boundary_loss_does_not_delay_recorded_switches(self):
        # Hover deadline between the first two ticks. The return reaches the
        # recorded grasp height before its target changes from 76% to 100%.
        normal = torch.tensor([[0., 0., 1.]] * 3)
        boundary = (
            torch.tensor([[0., 0., 0.], [-2., -5., -5.], [0., 0., 0.]]),
            torch.tensor([[.76, .76, .76], [.76, 1., 1.], [1., .76, .76]]),
            normal, torch.tensor([50., 5., 0.]),
            torch.tensor([True, False, False]), torch.tensor([False, True, False]),
            torch.tensor([True, True, False]), torch.tensor([False, False, True]),
            torch.tensor([.2, 0., 0.]),
        )
        clock = torch.tensor([[.2, 1.], [0., 0.], [0., 0.]])
        error = torch.zeros(3, 3, 7, requires_grad=True)
        with torch.no_grad():
            error[0, :2, 2] = -2.  # premature descent only at tick one
            error[1, 1, 2] = 1.  # remains above table at release
            error[1, :2, 6] = torch.tensor([1., -1.])  # early opening, then missed release
            error[2, :, 6] = torch.tensor([-1., 1., 1.])  # premature close, then two missed closes
        weights = torch.ones_like(error)
        weights[2, 2] = 0.  # padding
        scale = torch.tensor([.001] * 6 + [.12])
        penalty = training.cup_boundary_penalty(error, boundary, clock, scale, weights)
        penalty.backward()
        self.assertLess(error.grad[0, 0, 2], 0.)
        self.assertEqual(error.grad[0, 1, 2], 0.)
        self.assertGreater(error.grad[1, 1, 2], 0.)
        self.assertGreater(error.grad[1, 0, 6], 0.)
        self.assertLess(error.grad[1, 1, 6], 0.)
        self.assertLess(error.grad[2, 0, 6], 0.)
        self.assertGreater(error.grad[2, 1, 6], 0.)
        self.assertEqual(error.grad[2, 2].abs().sum(), 0.)
        error.grad = None
        # The observed clock overestimates the elapsed time; the supervised
        # window must still cover the remaining time in the original hold log.
        clock[0, 0] = 0.
        training.cup_boundary_penalty(error, boundary, clock, scale, weights, support_focus=True).backward()
        self.assertLess(error.grad[0, 0, 2], 0.)
        self.assertEqual(error.grad[0, 1, 2], 0.)
        self.assertGreater(error.grad[1, 0, 6], 0.)
        self.assertLess(error.grad[1, 1, 6], 0.)

    def test_coupled_support_penalty_corrects_gripper_without_rewarding_an_early_descent(self):
        boundary = (torch.tensor([[-1.]]), torch.tensor([[.76]]), torch.tensor([[0., 0., 1.]]),
                    torch.tensor([5.]), torch.tensor([False]), torch.tensor([True]),
                    torch.tensor([True]), torch.tensor([False]), torch.tensor([0.]))
        clock = torch.zeros(1, 2)
        error = torch.tensor([[[0., 0., 0., 0., 0., 0., 1.]]], requires_grad=True)
        scale, weights = torch.tensor([.001] * 6 + [.12]), torch.ones_like(error)
        support = training.cup_boundary_penalty(error, boundary, clock, scale, weights, support_focus=True)
        coupled = training.cup_boundary_penalty(error, boundary, clock, scale, weights,
                                                support_focus=True, coupled_support=True)
        extra_gradient, = torch.autograd.grad(coupled - support, error)
        self.assertGreater(extra_gradient[0, 0, 6], 0.)
        self.assertEqual(extra_gradient[0, 0, 2], 0.)

    def test_coupled_xyz_prefix_penalty_corrects_horizontal_hold_drift(self):
        boundary = (torch.zeros(1, 3), torch.ones(1, 3), torch.tensor([[0., 0., 1.]]),
                    torch.zeros(1), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool),
                    torch.zeros(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool), torch.zeros(1))
        error = torch.zeros(1, 3, 7, requires_grad=True)
        error.data[0, :2, 0] = torch.tensor([1., -1.])
        error.data[0, 2, 1] = 1.
        scale, weights = torch.tensor([.001] * 6 + [.12]), torch.ones_like(error)
        weights[0, 2] = 0.
        training.cup_boundary_penalty(error, boundary, torch.zeros(1, 2), scale, weights,
                                      support_focus=True, coupled_support=True).backward()
        self.assertGreater(error.grad[0, 0, 0], 0.)
        self.assertLess(error.grad[0, 1, 0], 0.)
        self.assertEqual(error.grad[0, 2].abs().sum(), 0.)
        self.assertEqual(error.grad[..., 6].abs().sum(), 0.)

    def test_coupled_support_strengthens_only_observed_open_gripper_targets(self):
        boundary = (
            torch.zeros(5, 1), torch.tensor([[1.], [.76], [.76], [1.], [1.]]),
            torch.tensor([[0., 0., 1.]] * 5), torch.zeros(5),
            torch.zeros(5, dtype=torch.bool), torch.zeros(5, dtype=torch.bool),
            torch.zeros(5, dtype=torch.bool), torch.tensor([True, True, True, False, True]),
            torch.zeros(5),
        )
        error = torch.zeros(5, 1, 7, requires_grad=True)
        with torch.no_grad():
            error[:, 0, 6] = torch.tensor([-1., 1., -1., -1., -1.])
        scale = torch.tensor([.001] * 6 + [.12])
        weights = torch.ones_like(error)
        weights[4, 0, 6] = 0.
        clock = torch.zeros(5, 2)

        support = training.cup_boundary_penalty(error, boundary, clock, scale, weights, support_focus=True)
        coupled = training.cup_boundary_penalty(error, boundary, clock, scale, weights,
                                                support_focus=True, coupled_support=True)
        support_grad, = torch.autograd.grad(support, error, retain_graph=True)
        coupled_grad, = torch.autograd.grad(coupled, error)

        self.assertLess(support_grad[0, 0, 6], 0.)
        torch.testing.assert_close(coupled_grad[0, 0, 6], 3. * support_grad[0, 0, 6])
        self.assertGreater(support_grad[1, 0, 6], 0.)
        torch.testing.assert_close(coupled_grad[1, 0, 6], 5. * support_grad[1, 0, 6])
        torch.testing.assert_close(coupled_grad[2, 0, 6], support_grad[2, 0, 6])
        self.assertEqual(coupled_grad[3, 0, 6], 0.)
        self.assertEqual(coupled_grad[4, 0, 6], 0.)
        self.assertTrue(bool((coupled_grad[..., :6] == 0).all()))

    def test_setdown_loss_separates_prompt_opening_from_keeping_the_cup_held(self):
        boundary = (
            torch.zeros(3, 1), torch.tensor([[1.], [1.], [.76]]), torch.tensor([[0., 0., 1.]] * 3),
            torch.tensor([0., 0., 5.]), torch.zeros(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool),
            torch.ones(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool), torch.zeros(3),
        )
        error = torch.tensor([[[1.2, 0., 0., 0., 0., 0., -1.]],
                              [[1.2, 0., 0., 0., 0., 0., -1.]],
                              [[0., 0., 0., 0., 0., 0., 1.]]], requires_grad=True)
        clock, scale, weights = torch.zeros(3, 2), torch.tensor([.001] * 6 + [.12]), torch.ones_like(error)
        options = dict(support_focus=True, coupled_support=True)
        base = training.cup_boundary_penalty(error, boundary, clock, scale, weights, **options)
        setdown = training.cup_boundary_penalty(
            error, boundary, clock, scale, weights, setdown_age=torch.tensor([.5, 0., 0.]),
            before_setdown=torch.tensor([[False], [False], [True]]), **options,
        )
        extra, = torch.autograd.grad(setdown - base, error)
        self.assertGreater(extra[0, 0, 0], 0.)  # Match the demonstrated horizontal stop.
        self.assertLess(extra[0, 0, 6], 0.)  # Open toward the recorded command.
        self.assertEqual(extra[1].abs().sum(), 0.)
        self.assertGreater(extra[2, 0, 6], 0.)  # Stay closed while still above the table.
        self.assertEqual(extra[2, 0, :6].abs().sum(), 0.)

    def test_transition_sampler_covers_train_sources_and_preserves_other_samples(self):
        rows, proof, train, val = transition_rows()
        sampler = training.TransitionSampler(rows, train, proof, seed=9)
        sampled = [sampler.draw() for _ in range(40)]
        report = sampler.report()
        self.assertEqual(report['category_counts'], {'close': 10, 'open': 10, 'motion': 10, 'hold': 6, 'all': 4})
        self.assertTrue(all(i in set(train[0] + train[1]) for i, _ in sampled))
        self.assertFalse(any(i in set(val[2]) for i, _ in sampled))
        for kind in ('close', 'open'):
            self.assertEqual({event['source'] for event in report['events'][kind]}, {'0', '1'})
            self.assertTrue(all(c['total'] and c['at'] for c in report['event_counts'][kind].values()))
        for index, detail in sampled:
            self.assertEqual(rows[index]['_physical_source'], detail['source'])
            if detail['category'] == 'hold':
                self.assertIn(rows[index]['frame_index'], (6, 7))
            if detail['category'] in ('close', 'open'):
                self.assertEqual(rows[index]['episode_index'], rows[detail['event_index']]['episode_index'])

    def test_sampler_state_restores_exact_draw_sequence(self):
        rows, proof, train, _ = transition_rows()
        original = training.TransitionSampler(rows, train, proof, seed=17)
        for _ in range(13):
            original.draw_balanced(4)
        state = original.getstate()
        expected = [original.draw_balanced(4) for _ in range(20)]
        restored = training.TransitionSampler(rows, train, proof, seed=44)
        restored.setstate(state)
        self.assertEqual(expected, [restored.draw_balanced(4) for _ in range(20)])
        self.assertEqual(original.report()['event_counts'], restored.report()['event_counts'])

    def test_each_accumulated_update_balances_events_motion_and_full_trajectory(self):
        rows, proof, train, _ = transition_rows()
        sampler = training.TransitionSampler(rows, train, proof, seed=37)
        groups = [sampler.draw_balanced(4) for _ in range(16)]
        for group in groups:
            categories = [detail['category'] for _, detail in group]
            self.assertEqual(categories.count('close'), 1)
            self.assertEqual(categories.count('open'), 1)
            self.assertEqual(categories.count('motion'), 1)
            self.assertEqual(categories.count('all') + categories.count('hold'), 1)
        fourth = [detail['category'] for group in groups[:5] for _, detail in group
                  if detail['category'] in ('all', 'hold')]
        self.assertEqual((fourth.count('all'), fourth.count('hold')), (3, 2))
        for kind in ('close', 'open'):
            self.assertTrue(all(count['at'] for count in sampler.report()['event_counts'][kind].values()))
        with self.assertRaisesRegex(ValueError, 'multiple of four'):
            sampler.draw_balanced(3)

    def test_uniform_switch_uses_real_events_and_restores_both_queues(self):
        rows, episodes = [], {}
        for ep in range(9):
            episodes[ep] = []
            for pos, grip in enumerate((1., .76, 1.)):
                index = len(rows)
                episodes[ep].append(index)
                rows.append({'action': [0.]*6+[grip], '_physical_source': str(ep),
                    '_gripper_target_ref': {'kind': 'gripper', 'executed': True, 'source': 'wire_command',
                        'sequence': index, 'target_degrees_percent': [0.]*6+[100*grip]}})
        original_rows = copy.deepcopy(rows)
        train = {ep: episodes[ep] for ep in range(8)}
        sampler = training.make_sampler(training.UNIFORM_SWITCH_PROFILE, rows, train, {}, 43)
        first = [sampler.draw_balanced(4) for _ in range(5)]
        state = sampler.getstate()
        expected = [sampler.draw_balanced(4) for _ in range(7)]
        restored = training.make_sampler(training.UNIFORM_SWITCH_PROFILE, rows, train, {}, 1)
        restored.setstate(state)
        self.assertEqual(expected, [restored.draw_balanced(4) for _ in range(7)])
        groups = first+expected
        for group in groups:
            self.assertEqual(sorted(item['category'] for _, item in group), ['close', 'open', 'uniform', 'uniform'])
            for index, item in group:
                if item['category'] in ('close', 'open'):
                    self.assertEqual(index, item['event_index'])
                    self.assertEqual(item['offset_frames'], 0)
                    self.assertEqual(index % 3, 1 if item['category'] == 'close' else 2)
                self.assertLess(index, 24)
        for kind in ('close', 'open'):
            self.assertEqual(len({index for group in groups[:8] for index, item in group
                                  if item['category'] == kind}), 8)
        uniform = [index for group in groups for index, item in group if item['category'] == 'uniform']
        self.assertEqual(set(uniform), set(range(24)))
        self.assertEqual(len(set(uniform)), len(uniform))
        self.assertEqual(rows, original_rows)
        with self.assertRaisesRegex(ValueError, 'multiple of four'):
            sampler.draw_balanced(2)

    def test_uniform_switch_keeps_existing_seeded_sequence(self):
        rows, proof, train, _ = transition_rows()
        sampler = training.make_sampler(training.UNIFORM_SWITCH_PROFILE, rows, train, proof, 43)
        groups = [[(index, detail['category']) for index, detail in sampler.draw_balanced(4)]
                  for _ in range(3)]
        self.assertEqual(groups, [[(27, 'uniform'), (4, 'close'), (23, 'open'), (21, 'uniform')],
                                  [(9, 'open'), (18, 'close'), (20, 'uniform'), (23, 'uniform')],
                                  [(18, 'uniform'), (7, 'uniform'), (23, 'open'), (4, 'close')]])

    def test_uniform_switch_hold_uses_recorded_boundaries_and_resumes(self):
        rows, proof, train, val = transition_rows()
        physical_times = [1000., 1001., 1002., 1003., 1004., 1007., 1008.,
                          1009., 1010., 1011., 1012., 1013., 1014., 1015.]
        for index, timestamp in zip(train[0], physical_times):
            rows[index]['_grid_time'] = timestamp
        for index, timestamp in zip(val[2], physical_times):
            rows[index]['_grid_time'] = timestamp
        proof['sources'][0]['hold_records'] = [
            {'start_monotonic': 1002.25, 'end_monotonic': 1007.5},
            {'start_monotonic': 1008.1, 'end_monotonic': 1010.}]
        proof['sources'][1]['hold_records'] = [{'start_monotonic': 100.51, 'end_monotonic': 100.99}]
        proof['sources'][2]['hold_records'] = copy.deepcopy(proof['sources'][0]['hold_records'])
        original_rows = copy.deepcopy(rows)
        sampler = training.make_sampler(training.UNIFORM_SWITCH_HOLD_PROFILE, rows, train, proof, 17)
        boundaries = sampler.report()['hold_boundaries']
        self.assertEqual([(h['source'], h['episode'], h['hold_index'], h['start'], h['end'])
                          for h in boundaries], [('0', 0, 0, 3, 5), ('0', 0, 1, 7, 8),
                                                  ('1', 1, 0, 18, 19)])
        first = [sampler.draw_balanced(4) for _ in range(5)]
        state = sampler.getstate()
        expected = [sampler.draw_balanced(4) for _ in range(13)]
        restored = training.make_sampler(training.UNIFORM_SWITCH_HOLD_PROFILE, rows, train, proof, 99)
        restored.setstate(state)
        self.assertEqual(expected, [restored.draw_balanced(4) for _ in range(13)])
        self.assertEqual(sampler.report(), restored.report())
        groups = first + expected
        chosen = []
        for group in groups:
            self.assertEqual(sorted(detail['category'] for _, detail in group),
                             ['close', 'open', 'recorded_hold_boundary', 'uniform'])
            self.assertTrue(all(index not in val[2] for index, _ in group))
            index, detail = next(item for item in group if item[1]['category'] == 'recorded_hold_boundary')
            chosen.append((index, detail['boundary']))
            self.assertEqual(rows[index]['_physical_source'], detail['source'])
            self.assertEqual(rows[index]['episode_index'], detail['episode'])
            self.assertLessEqual(detail['start_monotonic'], rows[index]['_grid_time'])
            self.assertLessEqual(rows[index]['_grid_time'], detail['end_monotonic'])
            episode = train[detail['episode']]
            targets, valid = training.chunk_for_anchor(rows, episode, episode.index(index))
            self.assertEqual(len(targets), 50)
            self.assertEqual(targets[:sum(valid)], [rows[i]['action'] for i in episode[episode.index(index):]])
        self.assertEqual([kind for _, kind in chosen], ['start', 'end'] * 9)
        self.assertEqual(set(chosen[:6]), {(3, 'start'), (5, 'end'), (7, 'start'),
                                          (8, 'end'), (18, 'start'), (19, 'end')})
        report = sampler.report()
        self.assertEqual(report['hold_boundary_counts'], {'start': [3, 3, 3], 'end': [3, 3, 3]})
        self.assertEqual(report['uniform']['epoch_samples'], 18)
        self.assertEqual(sum(report['event_counts']['close'].values()), 18)
        self.assertEqual(sum(report['event_counts']['open'].values()), 18)
        self.assertEqual(rows, original_rows)
        with self.assertRaisesRegex(ValueError, 'multiple of four'):
            sampler.draw_balanced(2)
        with self.assertRaisesRegex(ValueError, 'training sources only'):
            training.make_sampler(training.UNIFORM_SWITCH_HOLD_PROFILE, rows, {**train, **val}, proof, 17)
        changed = copy.deepcopy(proof)
        changed['sources'][0]['hold_records'][0]['start_monotonic'] = 1003.25
        with self.assertRaisesRegex(ValueError, 'different recorded hold'):
            training.make_sampler(training.UNIFORM_SWITCH_HOLD_PROFILE, rows, train, changed, 17).setstate(state)
        changed['sources'][0]['hold_records'][0] = {'start_monotonic': 2000., 'end_monotonic': 2001.}
        with self.assertRaisesRegex(ValueError, 'one training episode'):
            training.make_sampler(training.UNIFORM_SWITCH_HOLD_PROFILE, rows, train, changed, 17)


    def test_accumulated_weighted_gradients_match_full_batch_with_padding(self):
        model = ToyFlow()
        policy = SimpleNamespace(model=model, _preprocess_images=lambda batch: ([], []),
                                 prepare_action=lambda batch: batch['action'])
        valid = torch.tensor([[True]*2 + [False]*48, [True]*17 + [False]*33])
        values = torch.arange(700, dtype=torch.float32).reshape(2, 50, 7) / 100
        def batch(actions):
            return {'action': actions, 'observation.language.tokens': torch.ones(len(actions), 2),
                    'observation.language.attention_mask': torch.ones(len(actions), 2)}
        full = values.clone().requires_grad_()
        options = {'prefix_steps': 10, 'prefix_weight': 6., 'vertical_weight': 4.}
        full_loss = training.masked_flow_loss(policy, batch(full), valid, first_action_weight=8, gripper_weight=3,
                                              **options)
        full_loss.backward()
        accumulated = values.clone().requires_grad_()
        weights = [training.loss_normalizer([n], 8, 3, **options) for n in (2, 17)]
        for i, weight in enumerate(weights):
            part = training.masked_flow_loss(policy, batch(accumulated[i:i+1]), valid[i:i+1],
                                             first_action_weight=8, gripper_weight=3, **options)
            (part * (weight/sum(weights))).backward()
        self.assertTrue(torch.allclose(full.grad, accumulated.grad, atol=1e-7))

    def test_transition_requires_matching_real_command(self):
        rows, _, train, _ = transition_rows()
        rows[4]['_gripper_target_ref']['source'] = 'invented'
        with self.assertRaisesRegex(ValueError, 'provenance'):
            training.transition_events(rows, train)

    def test_resume_requires_matching_source_split_and_sampler(self):
        _, proof, _, _ = transition_rows()
        args = SimpleNamespace(rank=32, batch_size=1, gradient_accumulation=4, learning_rate=1e-5, seed=2,
                               adaptation=training.LORA_ADAPTATION,
                               action_representation=training.ABSOLUTE_REPRESENTATION,
                               first_action_weight=8, gripper_weight=3)
        parent = {'rank': 32, 'source_provenance': proof, 'sampling': training.TransitionSampler.VERSION,
                  'batch_size': 1, 'gradient_accumulation': 4, 'learning_rate': 1e-5, 'seed': 2,
                  'first_action_weight': 8, 'gripper_weight': 3}
        training.validate_parent(parent, proof, args, resume=True)
        with self.assertRaisesRegex(ValueError, 'same sampler'):
            training.validate_parent({**parent, 'sampling': 'old'}, proof, args, resume=True)
        training.validate_parent({**parent, 'sampling': 'old'}, proof, args, resume=False)
        with self.assertRaisesRegex(ValueError, 'parameters must match'):
            training.validate_parent({**parent, 'learning_rate': 2e-5}, proof, args, resume=True)
        bad_proof = {'sources': [{**source} for source in proof['sources']]}
        bad_proof['sources'][0]['split'] = 'validation'
        with self.assertRaisesRegex(ValueError, 'same physical source split'):
            training.validate_parent(parent, bad_proof, args, resume=False)

    def test_compact_dual_clock_requires_evidenced_cut_and_preserves_holds(self):
        maps, proof, cuts, clock = compact_clock_fixture()
        training.validate_compact_clocks(maps, proof, cuts, clock)
        with self.assertRaisesRegex(ValueError, 'evidence'):
            training.validate_compact_clocks(maps, proof, [], clock)
        bad = copy.deepcopy(maps)
        bad[1]['target_task_time_seconds'] += .1
        with self.assertRaisesRegex(ValueError, 'task clock'):
            training.validate_compact_clocks(bad, proof, cuts, clock)
        bad = copy.deepcopy(cuts)
        bad[0]['previous_command']['completed_monotonic'] += .1
        with self.assertRaisesRegex(ValueError, 'command or recorded hold'):
            training.validate_compact_clocks(maps, proof, bad, clock)
        bad = copy.deepcopy(proof)
        bad['sources'][0]['hold_records'][0] = {'start_monotonic': 100.3, 'end_monotonic': 100.4}
        with self.assertRaisesRegex(ValueError, 'command or recorded hold'):
            training.validate_compact_clocks(maps, bad, cuts, clock)
        bad = copy.deepcopy(cuts)
        bad[0]['feedback']['joint_span_deg'][0] = .2
        with self.assertRaisesRegex(ValueError, 'stationary feedback'):
            training.validate_compact_clocks(maps, proof, bad, clock)

    def test_compact_dual_clock_rejects_unmapped_gaps_and_segments(self):
        maps, proof, cuts, clock = compact_clock_fixture()
        bad = copy.deepcopy(maps)
        bad[0]['target_time_monotonic'] += .1
        with self.assertRaisesRegex(ValueError, 'next source observation'):
            training.validate_compact_clocks(bad, proof, cuts, clock)
        bad = copy.deepcopy(clock)
        bad['episodes'][0]['segments'].pop()
        with self.assertRaisesRegex(ValueError, 'cover'):
            training.validate_compact_clocks(maps, proof, cuts, bad)

    def test_diagnostics_separately_measure_closing_and_motion(self):
        state = [0.] * 6 + [1., 1.]
        result = training.validation_metrics([(state, [.1]*6+[.7], [.2]*6+[.8]),
                                              (state, [0.]*6+[1.], [0.]*6+[1.])])
        self.assertEqual(result['closing_target_count'], 1)
        self.assertAlmostEqual(result['closing_gripper_target_mae'], .1)
        self.assertIn('not_task_success', result['evaluation_kind'])

    def test_coverage_reports_tail_and_real_target_transitions(self):
        rows = sample_rows((60, 3))
        episodes = training.validate_rows(rows)
        rows[59]['action'][6] = 1.
        rows[58]['action'][6] = .7
        coverage = training.anchor_coverage(rows, {0: episodes[0]}, {0, 58, 59})
        self.assertEqual(coverage['eligible_anchors'], 60)
        self.assertEqual(coverage['eligible_tail_anchors'], 49)
        self.assertEqual(coverage['sampled_unique_tail_anchors'], 2)
        self.assertEqual(coverage['sampled_opening_target_changes'], 1)
        self.assertEqual(coverage['sampled_closing_target_changes'], 1)
        chunk_only = training.anchor_coverage(rows, {0: episodes[0]}, {10})
        self.assertEqual(chunk_only['sampled_opening_target_changes'], 0)
        self.assertEqual(chunk_only['supervised_opening_target_changes'], 1)
        self.assertEqual(chunk_only['supervised_closing_target_changes'], 1)

    def test_rejects_broken_timebase(self):
        rows = sample_rows()
        rows[2]['timestamp'] += .1
        with self.assertRaisesRegex(ValueError, '6 Hz'):
            training.validate_rows(rows)

    def test_actual_lerobot_export_loads_and_checks_frame_provenance(self):
        from . import test_stream_export as fixture
        from lerobot_robot_fairino import export_stream_policy as exporter
        with tempfile.TemporaryDirectory(prefix='stream-training-test-') as directory:
            root = Path(directory)
            make_source = fixture.fixture_module().make_source
            source_a = make_source(root / 'a', trial=0)
            source_b = make_source(root / 'b', trial=1)
            output = root / 'dataset'
            with patch.object(exporter, 'DATA_ROOT', root):
                exporter.export([source_a, source_b], output, allow_initial_hold=True)
            rows, episodes, dataset, proof, fingerprint = training.read_dataset(output)
            train, heldout, sources = training.split_sources(rows, episodes)
            self.assertEqual(sources, ['1'])
            self.assertEqual(len(dataset), len(rows))
            self.assertEqual(list(train), proof['train_episode_indices'])
            self.assertEqual(list(heldout), proof['validation_episode_indices'])
            self.assertEqual(len(fingerprint), 64)
            mapping = output / 'meta/stream_source_frames.jsonl'
            frames = mapping.read_text().splitlines()
            first = json.loads(frames[0])
            first['derived_action_radians_fraction'][0] += 1
            frames[0] = json.dumps(first)
            mapping.write_text('\n'.join(frames)+'\n')
            with self.assertRaisesRegex(ValueError, 'derived source action'):
                training.read_dataset(output)



if __name__ == '__main__':
    unittest.main()
