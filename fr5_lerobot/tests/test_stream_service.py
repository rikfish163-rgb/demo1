"""CPU-only checks for the stream protocol and checkpoint integrity."""
import base64
import copy
from contextlib import nullcontext
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import serve_stream_pi05 as service
import stream_checkpoint as checkpoints


def request(identifier='sample'):
    data = io.BytesIO()
    Image.new('RGB', (16, 12), 'white').save(data, format='JPEG')
    return {'schema': 'fr5-stream-request-v1', 'id': identifier, 'task': 'grab cup',
            'state': [0.] * 6 + [1., 1.], 'source_time': 123., 'source_clock': 'host_monotonic',
            'rgb_jpeg_base64': base64.b64encode(data.getvalue()).decode()}


class FakeInference:
    def __init__(self):
        self.info = {'contract': service.load_contract(), 'checkpoint_sha256': 'abc', 'n_action_steps': 50}
        self.resets = 0

    def reset(self):
        self.resets += 1

    def predict(self, payload):
        identifier, _, _, _ = service.validate_request(payload)
        return {'schema': 'fr5-stream-reply-v1', 'id': identifier, 'checkpoint_sha256': 'abc',
                'policy_contract': self.info['contract']['name'], 'actions': [[0.] * 7] * 50,
                'inference_seconds': .01}


class StreamServiceTests(unittest.TestCase):


    def test_request_requires_finite_state_timestamp_and_real_jpeg(self):
        item = request()
        self.assertEqual(service.validate_request(item)[3].shape, (12, 16, 3))
        for field, value in (('state', [float('nan')] * 8), ('source_time', float('nan')),
                             ('rgb_jpeg_base64', 'not base64'), ('source_clock', '')):
            invalid = {**item, field: value}
            with self.assertRaises((ValueError, OSError)):
                service.validate_request(invalid)

    def test_multiple_predictions_keep_queue_until_explicit_reset(self):
        model = FakeInference()
        messages = [request('one'), request('two'), {'op': 'reset', 'id': 'episode'}, request('three'),
                    {'op': 'shutdown'}]
        source = io.StringIO(''.join(json.dumps(m) + '\n' for m in messages))
        sink = io.StringIO()
        service.serve_stream(model, source, sink)
        reply = [json.loads(x) for x in sink.getvalue().splitlines()]
        self.assertEqual(reply[0]['schema'], 'fr5-stream-ready-v1')
        self.assertEqual(reply[0]['policy_contract'], 'fr5-joint-position-stream-v1')
        self.assertEqual(reply[0]['action_shape'], [50, 7])
        self.assertEqual(reply[0]['n_action_steps'], 50)
        self.assertEqual(reply[0]['image_profile'], checkpoints.FULL_IMAGE_PROFILE)
        self.assertEqual([reply[i]['id'] for i in (1, 2, 4)], ['one', 'two', 'three'])
        self.assertEqual(len(reply[1]['actions']), 50)
        self.assertEqual(model.resets, 1)
        self.assertEqual(reply[3]['schema'], 'fr5-stream-reset-v1')

    def test_ready_advertises_configured_prefix_without_changing_the_wire_chunk(self):
        model = FakeInference()
        model.info['n_action_steps'] = 10
        model.info['image_profile'] = checkpoints.WORKSPACE_IMAGE_PROFILE
        sink = io.StringIO()
        source = io.StringIO(json.dumps(request()) + '\n' + json.dumps({'op': 'shutdown'}) + '\n')
        service.serve_stream(model, source, sink)
        ready, reply = [json.loads(line) for line in sink.getvalue().splitlines()]
        self.assertEqual(ready['n_action_steps'], 10)
        self.assertEqual(ready['image_profile'], checkpoints.WORKSPACE_IMAGE_PROFILE)
        self.assertEqual(ready['action_shape'], [50, 7])
        self.assertEqual(len(reply['actions']), 50)

    def test_ready_requires_verified_action_prefix(self):
        model = FakeInference()
        del model.info['n_action_steps']
        with self.assertRaisesRegex(ValueError, 'sealed checkpoint n_action_steps'):
            service.serve_stream(model, io.StringIO(), io.StringIO())

    def test_ready_identifies_vision_adaptation(self):
        model = FakeInference()
        model.info['adaptation_mode'] = checkpoints.VISION_EXPERT_ADAPTATION
        sink = io.StringIO()
        service.serve_stream(model, io.StringIO('{"op":"shutdown"}\n'), sink)
        self.assertEqual(json.loads(sink.getvalue())['adaptation_mode'], checkpoints.VISION_EXPERT_ADAPTATION)

    def test_history_ready_and_prediction_preserve_raw_request(self):
        import torch
        model = FakeInference()
        model.info['observation_context_profile'] = 'state-history-v1'
        sink = io.StringIO()
        service.serve_stream(model, io.StringIO('{"op":"shutdown"}\n'), sink)
        self.assertEqual(json.loads(sink.getvalue())['observation_context_profile'], 'state-history-v1')

        inference = service.Inference.__new__(service.Inference)
        inference.info = {'observation_context_profile': 'state-history-v1', 'image_profile': checkpoints.FULL_IMAGE_PROFILE,
                          'checkpoint_sha256': 'abc', 'contract': service.load_contract(),
                          'action_representation': checkpoints.ABSOLUTE_REPRESENTATION,
                          'training_data_kind': 'captured'}
        inference.training = SimpleNamespace()
        inference.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None,
            get_rng_state=lambda: torch.zeros(2, dtype=torch.uint8)),
            inference_mode=torch.inference_mode, autocast=lambda *a, **k: nullcontext(),
            bfloat16=torch.bfloat16, isfinite=torch.isfinite)
        inference.pre = lambda value: {**value, 'observation.state': torch.zeros((1, 32))}
        inference.post = lambda value: value
        inference.policy = SimpleNamespace(predict_action_chunk=lambda batch: torch.zeros((1, 50, 7)),
                                           config=SimpleNamespace(num_inference_steps=10))
        inference.relative = None
        item = request()
        item['observation_history'] = [{'age_s': .5, 'state': [0.] * 8}] + [None] * 5
        original = copy.deepcopy(item)
        with patch.object(service.prompt_integrity, 'preprocess', side_effect=lambda pre, batch: pre(batch)):
            reply = inference.predict(item)
            self.assertEqual(reply['observation_context_profile'], 'state-history-v1')
            self.assertEqual(item, original)
            with self.assertRaisesRegex(ValueError, 'six entries'):
                inference.predict({k: v for k, v in item.items() if k != 'observation_history'})

    def test_fixed_setdown_prediction_uses_fixed_clock_at_each_denoising_step(self):
        import torch
        import train_stream_pi05 as training
        profile = checkpoints.observation_history.CUP_SETDOWN_FIXED_PROFILE
        inference = service.Inference.__new__(service.Inference)
        inference.info = {'observation_context_profile': profile, 'image_profile': checkpoints.FULL_IMAGE_PROFILE,
                          'checkpoint_sha256': 'abc', 'contract': service.load_contract(), 'n_action_steps': 3,
                          'action_representation': checkpoints.CUP_TICK_DELTA_REPRESENTATION,
                          'training_data_kind': 'captured'}
        inference.training = training
        inference.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None,
            get_rng_state=lambda: torch.zeros(2, dtype=torch.uint8)), inference_mode=torch.inference_mode,
            autocast=lambda *a, **k: nullcontext(), bfloat16=torch.bfloat16, isfinite=torch.isfinite)
        inference.pre = lambda value: {**value, 'observation.state': torch.nn.functional.pad(value['observation.state'], (0, 24))}
        inference.post = lambda value: value
        inference.relative = training.load_relative_helper()
        model = SimpleNamespace(embed_suffix=lambda actions, _time: actions)
        conditions = []
        def predict(batch):
            self.assertIs(batch['observation.fixed_cup_action_clock'], True)
            noisy = torch.arange(50 * 32, dtype=torch.float32).reshape(1, 50, 32)
            for step in (.9, .5, .1):
                actions = noisy + step
                embedded = model.embed_suffix(actions, torch.tensor([step]))
                self.assertTrue(torch.equal(embedded[..., :7], actions[..., :7]))
                conditions.append(embedded[..., 7:])
            output = torch.zeros(1, 50, 7)
            output[..., 6] = .76
            return output
        inference.policy = SimpleNamespace(model=model, predict_action_chunk=predict,
                                           config=SimpleNamespace(num_inference_steps=3))
        item = request()
        item.update(task='抓起杯子，在空中维持5秒，然后放下', state=[0.] * 6 + [.76, 1.],
                    cup_relative_state=[.01, -.02, .1, 1.617, -.02, -1.17, .76, 1.],
                    observation_history=[None] * 6, initial_cup_uv=[1429.53, 696.82],
                    cup_tracking={'rim_minus_jaw_mm': None, 'native_contact': 1},
                    cup_timing={'phase': 'returning', 'hover_seconds': None, 'visual': 'missing',
                                'support_gap_mm': 0., 'setdown_age_s': .15})
        original = copy.deepcopy(item)
        source = io.StringIO(json.dumps(item) + '\n{"op":"shutdown"}\n')
        sink = io.StringIO()
        with patch.object(service.prompt_integrity, 'preprocess', side_effect=lambda pre, value: pre(value)):
            service.serve_stream(inference, source, sink)
        ready, reply = [json.loads(line) for line in sink.getvalue().splitlines()]
        self.assertEqual(ready['observation_context_profile'], profile)
        self.assertEqual(ready['action_shape'], [50, 7])
        self.assertEqual(ready['n_action_steps'], 3)
        self.assertEqual(reply['observation_context_profile'], profile)
        self.assertEqual(np.asarray(reply['actions']).shape, (50, 7))
        self.assertTrue(all(torch.equal(conditions[0], value) for value in conditions[1:]))
        self.assertAlmostEqual(float(conditions[0][0, 0, -1]), .45)
        self.assertEqual(item, original)

    def test_fixed_setdown_is_allowed_in_both_offline_replay_suites(self):
        from fr5_lerobot.diagnostics import evaluate_cup_timing_policy as timing
        from fr5_lerobot.diagnostics import replay_true_five_second_pi05 as replay
        history = checkpoints.observation_history
        profile = history.CUP_SETDOWN_FIXED_PROFILE
        self.assertIn(profile, timing.allowed_profiles('candidate'))
        self.assertIn(profile, replay.SETDOWN_PROFILES)
        state = {'phase': 'returning', 'hover_seconds': None, 'visual': 'missing',
                 'support_gap_mm': 0., 'setdown_age_s': .15}
        tracking = {'rim_minus_jaw_mm': None, 'native_contact': 1}
        case = {'case_id': 'setdown', 'request': {'task': '放杯', 'source_time': 123.}, 'tracking': tracking,
                'timing_setdown': {'state': state}, 'timing_aligned': {'state': {'phase': 'hover'}},
                'observers': {'setdown': {'tracking': tracking, 'model_timing': state}}}
        original_case = copy.deepcopy(case)
        for make_request in (lambda selected: timing.model_request(case, 'candidate', 'native', profile=selected),
                             lambda selected: replay.model_request(case, 'setdown', selected)):
            original = make_request(history.CUP_SETDOWN_PROFILE)
            fixed = make_request(profile)
            self.assertEqual(fixed['observation_context_profile'], profile)
            self.assertEqual(fixed['cup_timing'], state)
            fixed['observation_context_profile'] = original['observation_context_profile']
            self.assertEqual(fixed, original)
        self.assertEqual(case, original_case)

    def test_history_bundle_profile_and_legacy_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base.safetensors'
            base.write_bytes(b'base')
            tok = root / 'revision'
            tok.mkdir()
            (tok / 'tokenizer.json').write_text('{}')
            bundle = root / 'bundle'
            bundle.mkdir()
            for name in checkpoints.REQUIRED - {'stream_contract.py', 'train_stream_pi05.py',
                    'serve_stream_pi05.py', 'stream_checkpoint.py', 'train_pi05_fr5.py',
                    'config.json', 'training-report.json'}:
                (bundle / name).write_bytes(b'test')
            (bundle / 'policy_preprocessor_step_2_normalizer_processor.safetensors').write_bytes(b'stats')
            config = {'type': 'pi05', 'chunk_size': 50, 'use_relative_actions': False,
                      'input_features': {'observation.state': {'shape': [8]}},
                      'output_features': {'action': {'shape': [7]}}, 'tokenizer_max_length': 512}
            checkpoints.write_json(bundle / 'config.json', config)
            checkpoints.write_json(bundle / 'policy_preprocessor.json', {'steps': [
                {'registry_name': 'tokenizer_processor', 'config': {'max_length': 512, 'padding': 'max_length'}}]})
            report_path = bundle / 'training-report.json'
            checkpoints.write_json(report_path, {'steps_completed': 100, 'dataset_fingerprint': 'f',
                                                 'observation_context_profile': 'state-history-v1'})
            contract_file = Path(__file__).resolve().parents[1] / 'src/lerobot_robot_fairino/stream_contract.py'
            checkpoints.create_bundle(bundle, contract=service.load_contract(), base_weights=base,
                                      base_revision='base', base_sha256=checkpoints.digest(base),
                                      tokenizer_source=tok, tokenizer_revision='revision',
                                      contract_source=contract_file, dataset_fingerprint='f')
            info = service.inspect_checkpoint(bundle)
            self.assertEqual(info['observation_context_profile'], 'state-history-v1')
            self.assertEqual(info['manifest']['observation_context_profile'], info['observation_context_profile'])
            self.assertIn('stream_observation_history.py', info['manifest']['files'])
            metadata = json.loads((bundle / 'observation_context.json').read_text())
            self.assertEqual(metadata['offsets_s'], [0.5, 1, 2, 4, 6, 8])
            self.assertEqual(metadata['training_clock'], 'retimed_episode_task_clock')
            self.assertEqual(metadata['runtime_clock'], 'host_monotonic_camera_time')
            with self.assertRaisesRegex(ValueError, 'fixed 512-token'):
                checkpoints.validate_context_tokenizer(bundle, {**config, 'tokenizer_max_length': 200},
                                                       'state-history-v1')
            checkpoints.write_json(bundle / 'policy_preprocessor.json', {'steps': [
                {'registry_name': 'tokenizer_processor', 'config': {'max_length': 200, 'padding': 'max_length'}}]})
            with self.assertRaisesRegex(ValueError, 'fixed 512-token'):
                checkpoints.validate_context_tokenizer(bundle, config, 'state-history-v1')
            checkpoints.write_json(bundle / 'policy_preprocessor.json', {'steps': [
                {'registry_name': 'tokenizer_processor', 'config': {'max_length': 512, 'padding': 'max_length'}}]})
            manifest = info['manifest']
            manifest['observation_context_profile'] = 'none'
            checkpoints.write_json(bundle / checkpoints.MANIFEST, manifest)
            with self.assertRaisesRegex(ValueError, 'disagrees with manifest'):
                service.inspect_checkpoint(bundle)
            report = json.loads(report_path.read_text())
            report.pop('observation_context_profile')
            checkpoints.write_json(report_path, report)
            manifest.pop('observation_context_profile')
            manifest['files'].pop('observation_context.json')
            manifest['files'].pop('stream_observation_history.py')
            (bundle / 'observation_context.json').unlink()
            (bundle / 'stream_observation_history.py').unlink()
            manifest['files']['training-report.json'] = {'sha256': checkpoints.digest(report_path),
                                                         'bytes': report_path.stat().st_size}
            checkpoints.write_json(bundle / checkpoints.MANIFEST, manifest)
            self.assertEqual(service.inspect_checkpoint(bundle)['observation_context_profile'], 'none')

    def test_action_prefix_validation_and_legacy_default(self):
        self.assertEqual(checkpoints.configured_action_steps({}), 50)
        for value in (1, 10, 50):
            self.assertEqual(checkpoints.configured_action_steps({'n_action_steps': value}), value)
        for value in (0, 51, True, 10.0, None, '10'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'n_action_steps'):
                checkpoints.configured_action_steps({'n_action_steps': value})

    def test_invalid_request_does_not_kill_service(self):
        messages = [{'schema': 'wrong', 'id': 'bad'}, request('good'), {'op': 'shutdown'}]
        sink = io.StringIO()
        service.serve_stream(FakeInference(), io.StringIO(''.join(json.dumps(x)+'\n' for x in messages)), sink)
        replies = [json.loads(x) for x in sink.getvalue().splitlines()]
        self.assertEqual(replies[1]['status'], 'rejected')
        self.assertEqual(replies[2]['id'], 'good')

    def test_bundle_includes_projection_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base.safetensors'
            base.write_bytes(b'base')
            tok = root / 'revision'
            tok.mkdir()
            (tok / 'tokenizer.json').write_text('{}')
            bundle = root / 'bundle'
            bundle.mkdir()
            contract = service.load_contract()
            config = {'type': 'pi05', 'chunk_size': 50, 'n_action_steps': 10, 'use_relative_actions': False,
                      'input_features': {'observation.state': {'shape': [8]}},
                      'output_features': {'action': {'shape': [7]}}}
            for name in checkpoints.REQUIRED - {'stream_contract.py', 'train_stream_pi05.py',
                                                'serve_stream_pi05.py', 'stream_checkpoint.py',
                                                'train_pi05_fr5.py', 'config.json', 'training-report.json'}:
                (bundle / name).write_bytes(b'test')
            (bundle / 'policy_preprocessor_step_2_normalizer_processor.safetensors').write_bytes(b'stats')
            (bundle / 'policy_postprocessor_step_0_unnormalizer_processor.safetensors').write_bytes(b'stats')
            checkpoints.write_json(bundle / 'config.json', config)
            checkpoints.write_json(bundle / 'training-report.json', {'steps_completed': 500, 'dataset_fingerprint': 'f',
                                                                      'n_action_steps': 10})
            contract_file = Path(__file__).resolve().parents[1] / 'src/lerobot_robot_fairino/stream_contract.py'
            checkpoints.create_bundle(bundle, contract=contract, base_weights=base, base_revision='base-rev',
                                      base_sha256=checkpoints.digest(base), tokenizer_source=tok,
                                      tokenizer_revision='revision', contract_source=contract_file,
                                      dataset_fingerprint='f')
            info = service.inspect_checkpoint(bundle)
            self.assertEqual(info['action_shape'], [50, 7])
            self.assertEqual(info['n_action_steps'], 10)
            self.assertEqual(info['manifest']['n_action_steps'], 10)
            self.assertEqual(info['adaptation_mode'], 'lora')
            self.assertIn('record_helpers/camera_timing.py', info['manifest']['files'])
            self.assertIn('record_helpers/gripper_feedback.py', info['manifest']['files'])
            self.assertEqual(info['image_profile'], checkpoints.FULL_IMAGE_PROFILE)
            self.assertIn('image_profile.json', info['manifest']['files'])
            self.assertIn('stream_image_profile.py', info['manifest']['files'])
            self.assertEqual(info['manifest']['prompt_integrity'], checkpoints.prompt_integrity.PROMPT_INTEGRITY)
            self.assertIn('stream_prompt_integrity.py', info['manifest']['files'])
            self.assertIn('stream_conditioned_sampling.py', info['manifest']['files'])
            bundled_training = service.load_bundle_module('bundled_stream_training_test', bundle / 'train_stream_pi05.py')
            timing, feedback = bundled_training.load_record_helpers()
            self.assertEqual(timing.camera_clock({'source_timestamp_monotonic': 1.}), (1., timing.MONOTONIC))
            self.assertFalse(feedback.gripper_feedback_valid({}))
            manifest_path = bundle / checkpoints.MANIFEST
            manifest = json.loads(manifest_path.read_text())
            helper_entry = manifest['files'].pop('stream_prompt_integrity.py')
            checkpoints.write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, 'prompt integrity helper'):
                service.inspect_checkpoint(bundle)
            manifest['files']['stream_prompt_integrity.py'] = helper_entry
            checkpoints.write_json(manifest_path, manifest)
            manifest['n_action_steps'] = 50
            checkpoints.write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, 'disagree with manifest'):
                service.inspect_checkpoint(bundle)
            manifest['n_action_steps'] = 10
            checkpoints.write_json(manifest_path, manifest)
            manifest['image_profile'] = checkpoints.WORKSPACE_IMAGE_PROFILE
            checkpoints.write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, 'image profile disagrees with manifest'):
                service.inspect_checkpoint(bundle)
            manifest['image_profile'] = checkpoints.FULL_IMAGE_PROFILE
            checkpoints.write_json(manifest_path, manifest)
            (bundle / 'projections.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash/size mismatch'):
                service.inspect_checkpoint(bundle)

    def test_workspace_profile_is_sealed_and_legacy_bundle_defaults_to_full(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base.safetensors'
            base.write_bytes(b'base')
            tok = root / 'revision'
            tok.mkdir()
            (tok / 'tokenizer.json').write_text('{}')
            bundle = root / 'bundle'
            bundle.mkdir()
            contract = service.load_contract()
            config = {'type': 'pi05', 'chunk_size': 50, 'use_relative_actions': False,
                      'input_features': {'observation.state': {'shape': [8]}},
                      'output_features': {'action': {'shape': [7]}}}
            for name in checkpoints.REQUIRED - {'stream_contract.py', 'train_stream_pi05.py',
                                                'serve_stream_pi05.py', 'stream_checkpoint.py',
                                                'train_pi05_fr5.py', 'config.json', 'training-report.json'}:
                (bundle / name).write_bytes(b'test')
            (bundle / 'policy_preprocessor_step_2_normalizer_processor.safetensors').write_bytes(b'stats')
            checkpoints.write_json(bundle / 'config.json', config)
            checkpoints.write_json(bundle / 'training-report.json', {'steps_completed': 100,
                                   'dataset_fingerprint': 'f', 'image_profile': checkpoints.WORKSPACE_IMAGE_PROFILE})
            contract_file = Path(__file__).resolve().parents[1] / 'src/lerobot_robot_fairino/stream_contract.py'
            checkpoints.create_bundle(bundle, contract=contract, base_weights=base, base_revision='base-rev',
                                      base_sha256=checkpoints.digest(base), tokenizer_source=tok,
                                      tokenizer_revision='revision', contract_source=contract_file,
                                      dataset_fingerprint='f')
            info = service.inspect_checkpoint(bundle)
            self.assertEqual(info['image_profile'], checkpoints.WORKSPACE_IMAGE_PROFILE)
            metadata = json.loads((bundle / 'image_profile.json').read_text())
            self.assertFalse(metadata['workspace_is_physical_wrist_camera'])
            self.assertEqual(metadata['workspace_crop_xyxy'], [240, 100, 640, 360])
            manifest_path = bundle / checkpoints.MANIFEST
            manifest = json.loads(manifest_path.read_text())
            del manifest['files']['stream_image_profile.py']
            checkpoints.write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, 'Unsealed checkpoint image profile'):
                service.inspect_checkpoint(bundle)

            # A pre-profile checkpoint has neither metadata nor helper and stays full-view.
            report_path = bundle / 'training-report.json'
            report = json.loads(report_path.read_text())
            report.pop('image_profile')
            checkpoints.write_json(report_path, report)
            manifest.pop('image_profile')
            manifest['files'].pop('image_profile.json')
            manifest['files']['training-report.json'] = {'sha256': checkpoints.digest(report_path),
                                                          'bytes': report_path.stat().st_size}
            (bundle / 'image_profile.json').unlink()
            (bundle / 'stream_image_profile.py').unlink()
            checkpoints.write_json(manifest_path, manifest)
            self.assertEqual(service.inspect_checkpoint(bundle)['image_profile'], checkpoints.FULL_IMAGE_PROFILE)

    def test_full_expert_bundle_requires_sealed_expert_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base.safetensors'
            base.write_bytes(b'base')
            tok = root / 'revision'
            tok.mkdir()
            (tok / 'tokenizer.json').write_text('{}')
            bundle = root / 'bundle'
            bundle.mkdir()
            contract = service.load_contract()
            config = {'type': 'pi05', 'chunk_size': 50, 'use_relative_actions': False,
                      'input_features': {'observation.state': {'shape': [8]}},
                      'output_features': {'action': {'shape': [7]}}}
            for name in checkpoints.REQUIRED - {'stream_contract.py', 'train_stream_pi05.py',
                                                'serve_stream_pi05.py', 'stream_checkpoint.py',
                                                'train_pi05_fr5.py', 'config.json', 'training-report.json'}:
                (bundle / name).write_bytes(b'test')
            (bundle / 'policy_preprocessor_step_2_normalizer_processor.safetensors').write_bytes(b'stats')
            (bundle / 'policy_postprocessor_step_0_unnormalizer_processor.safetensors').write_bytes(b'stats')
            (bundle / 'expert_model.safetensors').write_bytes(b'expert')
            checkpoints.write_json(bundle / 'adaptation.json', {'schema': 'fr5-stream-adaptation-v1',
                                                                'mode': 'full-expert-language-lora'})
            checkpoints.write_json(bundle / 'config.json', config)
            checkpoints.write_json(bundle / 'training-report.json', {'steps_completed': 100,
                                    'dataset_fingerprint': 'f', 'adaptation_mode': 'full-expert-language-lora'})
            contract_file = Path(__file__).resolve().parents[1] / 'src/lerobot_robot_fairino/stream_contract.py'
            checkpoints.create_bundle(bundle, contract=contract, base_weights=base, base_revision='base-rev',
                                      base_sha256=checkpoints.digest(base), tokenizer_source=tok,
                                      tokenizer_revision='revision', contract_source=contract_file,
                                      dataset_fingerprint='f')
            info = service.inspect_checkpoint(bundle)
            self.assertEqual(info['adaptation_mode'], 'full-expert-language-lora')
            self.assertEqual(info['n_action_steps'], 50)
            self.assertIn('expert_model.safetensors', info['manifest']['files'])
            manifest_path = bundle / checkpoints.MANIFEST
            manifest = json.loads(manifest_path.read_text())
            del manifest['n_action_steps']
            checkpoints.write_json(manifest_path, manifest)
            self.assertEqual(service.inspect_checkpoint(bundle)['n_action_steps'], 50)
            (bundle / 'expert_model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash/size mismatch'):
                service.inspect_checkpoint(bundle)
            (bundle / 'expert_model.safetensors').write_bytes(b'expert')
            manifest = json.loads((bundle / checkpoints.MANIFEST).read_text())
            del manifest['files']['expert_model.safetensors']
            checkpoints.write_json(bundle / checkpoints.MANIFEST, manifest)
            with self.assertRaisesRegex(ValueError, 'missing sealed expert'):
                service.inspect_checkpoint(bundle)

    def test_vision_bundle_requires_matching_sealed_metadata_and_targets(self):
        import train_stream_pi05 as training
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base.safetensors'
            base.write_bytes(b'base')
            tok = root / 'revision'
            tok.mkdir()
            (tok / 'tokenizer.json').write_text('{}')
            bundle = root / 'bundle'
            bundle.mkdir()
            contract = service.load_contract()
            config = {'type': 'pi05', 'chunk_size': 50, 'use_relative_actions': False,
                      'freeze_vision_encoder': False, 'train_expert_only': False,
                      'input_features': {'observation.state': {'shape': [8]}},
                      'output_features': {'action': {'shape': [7]}}}
            for name in checkpoints.REQUIRED - {'stream_contract.py', 'train_stream_pi05.py',
                                                'serve_stream_pi05.py', 'stream_checkpoint.py',
                                                'train_pi05_fr5.py', 'config.json', 'training-report.json',
                                                'adapter_config.json'}:
                (bundle / name).write_bytes(b'test')
            (bundle / 'policy_preprocessor_step_2_normalizer_processor.safetensors').write_bytes(b'stats')
            (bundle / 'expert_model.safetensors').write_bytes(b'expert')
            checkpoints.write_json(bundle / 'adapter_config.json', {'target_modules': checkpoints.VISION_EXPERT_LORA_TARGETS})
            checkpoints.write_json(bundle / 'adaptation.json', training.adaptation_metadata(training.VISION_EXPERT_ADAPTATION))
            checkpoints.write_json(bundle / 'config.json', config)
            checkpoints.write_json(bundle / 'training-report.json', {'steps_completed': 100, 'dataset_fingerprint': 'f',
                                    'adaptation_mode': checkpoints.VISION_EXPERT_ADAPTATION})
            contract_file = Path(__file__).resolve().parents[1] / 'src/lerobot_robot_fairino/stream_contract.py'
            checkpoints.create_bundle(bundle, contract=contract, base_weights=base, base_revision='base-rev',
                                      base_sha256=checkpoints.digest(base), tokenizer_source=tok,
                                      tokenizer_revision='revision', contract_source=contract_file,
                                      dataset_fingerprint='f')
            info = service.inspect_checkpoint(bundle)
            self.assertEqual(info['adaptation_mode'], checkpoints.VISION_EXPERT_ADAPTATION)
            manifest_path = bundle / checkpoints.MANIFEST
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest['adaptation_mode'], checkpoints.VISION_EXPERT_ADAPTATION)
            del manifest['adaptation_mode']
            checkpoints.write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, 'vision adaptation disagrees with manifest'):
                service.inspect_checkpoint(bundle)
            manifest['adaptation_mode'] = checkpoints.VISION_EXPERT_ADAPTATION
            adapter_path = bundle / 'adapter_config.json'
            checkpoints.write_json(adapter_path, {'target_modules': checkpoints.BASE_LORA_TARGETS})
            manifest['files']['adapter_config.json'] = {'sha256': checkpoints.digest(adapter_path),
                                                        'bytes': adapter_path.stat().st_size}
            checkpoints.write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, 'adapter targets'):
                service.inspect_checkpoint(bundle)
            checkpoints.write_json(adapter_path, {'target_modules': checkpoints.VISION_EXPERT_LORA_TARGETS})
            manifest['files']['adapter_config.json'] = {'sha256': checkpoints.digest(adapter_path),
                                                        'bytes': adapter_path.stat().st_size}
            config['freeze_vision_encoder'] = True
            config_path = bundle / 'config.json'
            checkpoints.write_json(config_path, config)
            manifest['files']['config.json'] = {'sha256': checkpoints.digest(config_path),
                                               'bytes': config_path.stat().st_size}
            checkpoints.write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, 'enable the vision path'):
                service.inspect_checkpoint(bundle)

    def test_relative_bundle_rejects_old_schema_and_preserves_absolute_wire(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base.safetensors'
            base.write_bytes(b'base')
            tok = root / 'revision'
            tok.mkdir()
            (tok / 'tokenizer.json').write_text('{}')
            bundle = root / 'bundle'
            bundle.mkdir()
            contract = service.load_contract()
            config = {'type': 'pi05', 'chunk_size': 50, 'use_relative_actions': False,
                      'input_features': {'observation.state': {'shape': [8]}},
                      'output_features': {'action': {'shape': [7]}}}
            for name in checkpoints.REQUIRED - {'stream_contract.py', 'train_stream_pi05.py',
                                                'serve_stream_pi05.py', 'stream_checkpoint.py',
                                                'train_pi05_fr5.py', 'config.json', 'training-report.json'}:
                (bundle / name).write_bytes(b'test')
            (bundle / 'policy_preprocessor_step_2_normalizer_processor.safetensors').write_bytes(b'stats')
            (bundle / 'policy_postprocessor_step_0_unnormalizer_processor.safetensors').write_bytes(b'stats')
            checkpoints.write_json(bundle / 'action_representation.json',
                {'schema': 'fr5-stream-action-representation-v1', 'mode': checkpoints.RELATIVE_REPRESENTATION,
                 'wire_mode': checkpoints.ABSOLUTE_REPRESENTATION, 'joint_dimensions': 6,
                 'gripper_representation': 'absolute', 'anchor': 'raw_request_state',
                 'official_relative_processor_enabled': False})
            checkpoints.write_json(bundle / 'config.json', config)
            checkpoints.write_json(bundle / 'training-report.json', {'steps_completed': 100,
                'dataset_fingerprint': 'f', 'action_representation': checkpoints.RELATIVE_REPRESENTATION})
            contract_file = Path(__file__).resolve().parents[1] / 'src/lerobot_robot_fairino/stream_contract.py'
            checkpoints.create_bundle(bundle, contract=contract, base_weights=base, base_revision='base-rev',
                                      base_sha256=checkpoints.digest(base), tokenizer_source=tok,
                                      tokenizer_revision='revision', contract_source=contract_file,
                                      dataset_fingerprint='f')
            info = service.inspect_checkpoint(bundle)
            self.assertEqual(info['action_representation'], checkpoints.RELATIVE_REPRESENTATION)
            self.assertEqual(info['manifest']['schema'], checkpoints.RELATIVE_SCHEMA)
            self.assertNotEqual(info['manifest']['schema'], checkpoints.SCHEMA)
            self.assertIn('relative_stream_actions.py', info['manifest']['files'])
            manifest = info['manifest']
            manifest['schema'] = checkpoints.SCHEMA
            checkpoints.write_json(bundle / checkpoints.MANIFEST, manifest)
            with self.assertRaisesRegex(ValueError, 'schema does not match'):
                service.inspect_checkpoint(bundle)

    def test_relative_wire_conversion_uses_each_request_anchor(self):
        import torch
        import relative_stream_actions as relative
        actions = torch.zeros(1, 50, 7)
        actions[:, :, :6] = .25
        actions[:, :, 6] = .75
        first = [1., 2., 3., 4., 5., 6., 1., 1.]
        second = [-1., -2., -3., -4., -5., -6., .5, 1.]
        one = service.actions_for_wire(actions, first, relative)
        two = service.actions_for_wire(actions, second, relative)
        torch.testing.assert_close(one[0, 0, :6], torch.tensor(first[:6]) + .25)
        torch.testing.assert_close(two[0, 0, :6], torch.tensor(second[:6]) + .25)
        self.assertTrue(torch.equal(one[:, :, 6], actions[:, :, 6]))
        self.assertTrue(torch.equal(two[:, :, 6], actions[:, :, 6]))

    def test_cup_delta_metadata_and_bundle_seal_both_helpers(self):
        history = checkpoints.observation_history
        cases = [(mode, history.CUP_TCP_HISTORY_PROFILE) for mode in (checkpoints.CUP_REPRESENTATION,
                 checkpoints.CUP_DELTA_REPRESENTATION, checkpoints.CUP_TICK_DELTA_REPRESENTATION)]
        cases += [(checkpoints.CUP_TICK_DELTA_REPRESENTATION, profile) for profile in (
            history.CUP_SETDOWN_PROFILE, history.CUP_SETDOWN_FIXED_PROFILE, history.CUP_SETDOWN_CLEAN_PROFILE)]
        for mode, profile in cases:
            with self.subTest(mode=mode, profile=profile), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base = root / 'base.safetensors'
                base.write_bytes(b'base')
                tokenizer = root / 'revision'
                tokenizer.mkdir()
                (tokenizer / 'tokenizer.json').write_text('{}')
                bundle = root / 'bundle'
                bundle.mkdir()
                for name in checkpoints.REQUIRED - {'stream_contract.py', 'train_stream_pi05.py',
                        'serve_stream_pi05.py', 'stream_checkpoint.py', 'train_pi05_fr5.py',
                        'config.json', 'training-report.json'}:
                    (bundle / name).write_bytes(b'test')
                (bundle / 'policy_preprocessor_step_2_normalizer_processor.safetensors').write_bytes(b'stats')
                (bundle / 'policy_postprocessor_step_0_unnormalizer_processor.safetensors').write_bytes(b'stats')
                config = {'type': 'pi05', 'chunk_size': 50, 'n_action_steps': 10,
                          'use_relative_actions': False, 'tokenizer_max_length': 512,
                          'input_features': {'observation.state': {'shape': [8]}},
                          'output_features': {'action': {'shape': [7]}}}
                checkpoints.write_json(bundle / 'config.json', config)
                checkpoints.write_json(bundle / 'policy_preprocessor.json', {'steps': [
                    {'registry_name': 'tokenizer_processor', 'config': {'max_length': 512, 'padding': 'max_length'}}]})
                report = {'steps_completed': 100,
                    'n_action_steps': 10, 'dataset_fingerprint': 'f', 'action_representation': mode,
                    'observation_context_profile': profile}
                if profile == history.CUP_SETDOWN_FIXED_PROFILE:
                    report['xyz_loss_mode'] = 'terminal'
                if profile in history.CUP_TRACKING_PROFILES:
                    report['cup_tracking_data'] = {'rows': 1, 'sources': [
                        {'source_index': 0, 'rows': 1, 'rows_observed': 1}]}
                checkpoints.write_json(bundle / 'training-report.json', report)
                metadata = checkpoints.cup_action_metadata(mode)
                if mode in (checkpoints.CUP_DELTA_REPRESENTATION, checkpoints.CUP_TICK_DELTA_REPRESENTATION):
                    checkpoints.write_json(bundle / 'action_representation.json',
                                           {**metadata, 'anchor': 'raw_request_state'})
                    with self.assertRaisesRegex(ValueError, 'Cup action representation'):
                        checkpoints.read_action_representation(bundle)
                if mode == checkpoints.CUP_TICK_DELTA_REPRESENTATION:
                    self.assertEqual(metadata['horizon_divisor'], 'one_based_action_index')
                    checkpoints.write_json(bundle / 'action_representation.json',
                                           {**metadata, 'horizon_divisor': 'zero_based_action_index'})
                    with self.assertRaisesRegex(ValueError, 'Cup action representation'):
                        checkpoints.read_action_representation(bundle)
                checkpoints.write_json(bundle / 'action_representation.json', metadata)
                contract_file = Path(__file__).resolve().parents[1] / 'src/lerobot_robot_fairino/stream_contract.py'
                checkpoints.create_bundle(bundle, contract=service.load_contract(), base_weights=base,
                    base_revision='base', base_sha256=checkpoints.digest(base), tokenizer_source=tokenizer,
                    tokenizer_revision='revision', contract_source=contract_file, dataset_fingerprint='f')
                info = service.inspect_checkpoint(bundle)
                if profile in history.CUP_TRACKING_PROFILES:
                    self.assertIn('cup_rim_geometry.py', info['manifest']['files'])
                    self.assertEqual((bundle / 'cup_rim_geometry.py').read_bytes(),
                                     (Path(__file__).resolve().parents[2] / 'fr5_jev/cup_rim_geometry.py').read_bytes())
                if profile == history.CUP_SETDOWN_FIXED_PROFILE:
                    self.assertIn('stream_terminal_loss.py', info['manifest']['files'])
                    manifest_path = bundle / checkpoints.MANIFEST
                    manifest = json.loads(manifest_path.read_text())
                    entry = manifest['files'].pop('stream_terminal_loss.py')
                    checkpoints.write_json(manifest_path, manifest)
                    with self.assertRaisesRegex(ValueError, 'sealed sampling helper'):
                        checkpoints.verify_bundle(bundle)
                    manifest['files']['stream_terminal_loss.py'] = entry
                    checkpoints.write_json(manifest_path, manifest)
                self.assertEqual(info['action_representation'], mode)
                self.assertEqual(info['observation_context_profile'], profile)
                self.assertEqual(info['action_shape'], [50, 7])
                self.assertEqual(json.loads((bundle / 'observation_context.json').read_text()),
                                 history.context_metadata(profile))
                self.assertEqual(info['wire_action_representation'], checkpoints.CUP_REPRESENTATION)
                self.assertIn('cup_relative_stream_data.py', info['manifest']['files'])
                self.assertEqual(metadata['wire_mode'], checkpoints.CUP_REPRESENTATION)
                if mode == checkpoints.CUP_REPRESENTATION:
                    self.assertEqual(metadata, checkpoints.cup_action_metadata())
                    self.assertEqual(info['manifest']['schema'], checkpoints.SCHEMA)
                    self.assertNotIn('relative_stream_actions.py', info['manifest']['files'])
                    continue
                self.assertEqual(metadata['anchor'], 'raw_request_cup_relative_state')
                self.assertEqual(metadata['gripper_representation'], 'absolute')
                self.assertEqual(info['manifest']['schema'], checkpoints.RELATIVE_SCHEMA)
                self.assertIn('relative_stream_actions.py', info['manifest']['files'])
                helper = service.load_bundle_module('test_cup_delta_sealed_helper', bundle / 'relative_stream_actions.py')
                helper.assert_manual_pipeline(config)
                import torch
                action = torch.zeros((1, 50, 7)); action[:, :, :6] = .01; action[:, :, 6] = .7
                anchor = torch.tensor([[.1, .2, .3, .4, .5, .6, .8, 1.]])
                converted = helper.to_absolute_actions(action, anchor, input_representation=mode)
                multiplier = (torch.arange(1, 51, dtype=torch.float32).unsqueeze(1)
                              if mode == checkpoints.CUP_TICK_DELTA_REPRESENTATION else torch.ones((50, 1)))
                torch.testing.assert_close(converted[0, :, :6], anchor[0, :6] + .01 * multiplier)
                torch.testing.assert_close(converted[0, :, 6], action[0, :, 6])

                manifest_path = bundle / checkpoints.MANIFEST
                manifest = info['manifest']
                for name in ('relative_stream_actions.py', 'cup_relative_stream_data.py'):
                    entry = manifest['files'].pop(name)
                    checkpoints.write_json(manifest_path, manifest)
                    with self.assertRaisesRegex(ValueError, 'sealed action converter|data reader'):
                        checkpoints.verify_bundle(bundle)
                    manifest['files'][name] = entry
                history = checkpoints.observation_history
                context_path = bundle / 'observation_context.json'
                report_path = bundle / 'training-report.json'
                original_context = json.loads(context_path.read_text())
                original_report = json.loads(report_path.read_text())
                checkpoints.write_json(context_path, history.context_metadata(history.PHYSICAL_HISTORY_PROFILE))
                checkpoints.write_json(report_path, {**original_report,
                    'observation_context_profile': history.PHYSICAL_HISTORY_PROFILE})
                manifest['observation_context_profile'] = history.PHYSICAL_HISTORY_PROFILE
                for path in (context_path, report_path):
                    manifest['files'][path.name] = {'sha256': checkpoints.digest(path), 'bytes': path.stat().st_size}
                checkpoints.write_json(manifest_path, manifest)
                with self.assertRaisesRegex(ValueError, 'matching measured TCP context'):
                    checkpoints.verify_bundle(bundle)
                checkpoints.write_json(context_path, original_context)
                checkpoints.write_json(report_path, original_report)
                manifest['observation_context_profile'] = profile
                for path in (context_path, report_path):
                    manifest['files'][path.name] = {'sha256': checkpoints.digest(path), 'bytes': path.stat().st_size}
                config['use_relative_actions'] = True
                checkpoints.write_json(bundle / 'config.json', config)
                manifest['files']['config.json'] = {'sha256': checkpoints.digest(bundle / 'config.json'),
                                                     'bytes': (bundle / 'config.json').stat().st_size}
                checkpoints.write_json(manifest_path, manifest)
                with self.assertRaisesRegex(ValueError, 'use_relative_actions=False'):
                    checkpoints.verify_bundle(bundle)

    def test_cup_delta_uses_raw_same_request_cup_state_after_postprocessing(self):
        import torch
        import relative_stream_actions as relative
        inference = service.Inference.__new__(service.Inference)
        inference.info = {'observation_context_profile': checkpoints.observation_history.CUP_TCP_HISTORY_PROFILE,
                          'image_profile': checkpoints.FULL_IMAGE_PROFILE, 'checkpoint_sha256': 'abc',
                          'contract': service.load_contract(), 'action_representation': checkpoints.CUP_DELTA_REPRESENTATION,
                          'training_data_kind': 'captured'}
        inference.training = SimpleNamespace()
        inference.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None,
            get_rng_state=lambda: torch.zeros(2, dtype=torch.uint8)),
            inference_mode=torch.inference_mode, autocast=lambda *a, **k: nullcontext(),
            bfloat16=torch.bfloat16, isfinite=torch.isfinite)
        seen = []
        def pre(batch):
            seen.append(batch['observation.state'].clone())
            return {**batch, 'observation.state': torch.zeros((1, 32))}
        inference.pre = pre
        inference.post = lambda value: value + torch.tensor([.01, -.02, .03, .04, -.05, .06, .7])
        inference.policy = SimpleNamespace(predict_action_chunk=lambda batch: torch.zeros((1, 50, 7)),
                                           config=SimpleNamespace(num_inference_steps=10))
        inference.relative = relative
        item = request('cup-one')
        item['state'] = [9., 8., 7., 6., 5., 4., .8, 1.]
        item['cup_relative_state'] = [.1, .2, .3, .4, .5, .6, .8, 1.]
        item['initial_cup_uv'] = [1200., 700.]
        item['observation_history'] = [None] * 6
        original = copy.deepcopy(item)
        with patch.object(service.prompt_integrity, 'preprocess', side_effect=lambda pre, batch: pre(batch)):
            first = inference.predict(item)
            second_item = copy.deepcopy(item)
            second_item['id'] = 'cup-two'
            second_item['cup_relative_state'][:6] = [-.1, -.2, -.3, -.4, -.5, -.6]
            second = inference.predict(second_item)
        self.assertEqual(item, original)
        self.assertEqual(first['model_action_representation'], checkpoints.CUP_DELTA_REPRESENTATION)
        self.assertEqual(first['wire_action_representation'], checkpoints.CUP_REPRESENTATION)
        torch.testing.assert_close(seen[0][0], torch.tensor(item['cup_relative_state']))
        delta = np.array([.01, -.02, .03, .04, -.05, .06])
        first_expected = np.tile(np.array(item['cup_relative_state'][:6])+delta, (50, 1))
        second_expected = np.tile(np.array(second_item['cup_relative_state'][:6])+delta, (50, 1))
        np.testing.assert_allclose(np.asarray(first['actions'])[:, :6], first_expected, atol=1e-6)
        np.testing.assert_allclose(np.asarray(second['actions'])[:, :6], second_expected, atol=1e-6)
        self.assertAlmostEqual(first['actions'][0][6], .7, places=6)
        self.assertAlmostEqual(second['actions'][0][6], .7, places=6)

        inference.info['action_representation'] = checkpoints.CUP_REPRESENTATION
        inference.relative = None
        with patch.object(service.prompt_integrity, 'preprocess', side_effect=lambda pre, batch: pre(batch)):
            legacy_cup = inference.predict(item)
        self.assertEqual(legacy_cup['wire_action_representation'], checkpoints.CUP_REPRESENTATION)
        np.testing.assert_allclose(legacy_cup['actions'][0][:6], delta, atol=1e-6)

        model = FakeInference()
        model.info['action_representation'] = checkpoints.CUP_DELTA_REPRESENTATION
        sink = io.StringIO()
        service.serve_stream(model, io.StringIO('{"op":"shutdown"}\n'), sink)
        ready = json.loads(sink.getvalue())
        self.assertEqual(ready['model_action_representation'], checkpoints.CUP_DELTA_REPRESENTATION)
        self.assertEqual(ready['wire_action_representation'], checkpoints.CUP_REPRESENTATION)

    def test_cup_delta_rejects_missing_converter_and_mismatched_gripper_state(self):
        import torch
        for mode in (checkpoints.CUP_DELTA_REPRESENTATION, checkpoints.CUP_TICK_DELTA_REPRESENTATION):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'lacks its loaded action converter'):
                service.actions_for_wire(torch.zeros((1, 50, 7)), [0.] * 8, None, input_representation=mode)
            inference = service.Inference.__new__(service.Inference)
            inference.info = {'action_representation': mode}
            item = request()
            for cup_state in (None, [0.] * 6 + [.5, 1.], [float('nan')] * 6 + [1., 1.], [0.] * 7):
                item['cup_relative_state'] = cup_state
                with self.subTest(mode=mode, cup_state=cup_state), self.assertRaisesRegex(ValueError, 'unchanged gripper'):
                    inference.predict(item)

    def test_cup_tick_delta_is_restored_once_with_the_request_cup_anchor(self):
        import torch
        import relative_stream_actions as relative
        inference = service.Inference.__new__(service.Inference)
        inference.info = {'observation_context_profile': checkpoints.observation_history.CUP_TCP_HISTORY_PROFILE,
                          'image_profile': checkpoints.FULL_IMAGE_PROFILE, 'checkpoint_sha256': 'abc',
                          'contract': service.load_contract(),
                          'action_representation': checkpoints.CUP_TICK_DELTA_REPRESENTATION,
                          'training_data_kind': 'captured'}
        inference.training = SimpleNamespace()
        inference.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None,
            get_rng_state=lambda: torch.zeros(2, dtype=torch.uint8)),
            inference_mode=torch.inference_mode, autocast=lambda *a, **k: nullcontext(),
            bfloat16=torch.bfloat16, isfinite=torch.isfinite)
        inference.pre = lambda batch: {**batch, 'observation.state': torch.zeros((1, 32))}
        tick = torch.tensor([.001, -.002, .003, .004, -.005, .006, .7])
        inference.post = lambda value: value + tick
        inference.policy = SimpleNamespace(predict_action_chunk=lambda batch: torch.zeros((1, 50, 7)),
                                           config=SimpleNamespace(num_inference_steps=10))
        inference.relative = relative
        item = request('tick-one')
        item['state'] = [9., 8., 7., 6., 5., 4., .8, 1.]
        item['cup_relative_state'] = [.1, .2, .3, .4, .5, .6, .8, 1.]
        item['initial_cup_uv'] = [1200., 700.]
        item['observation_history'] = [None] * 6
        with patch.object(service.prompt_integrity, 'preprocess', side_effect=lambda pre, batch: pre(batch)), \
             patch.object(relative, 'to_absolute_actions', wraps=relative.to_absolute_actions) as convert:
            reply = inference.predict(item)
            convert.assert_called_once()
            second_item = copy.deepcopy(item)
            second_item['id'] = 'tick-two'
            second_item['cup_relative_state'][:6] = [-.1, -.2, -.3, -.4, -.5, -.6]
            second_reply = inference.predict(second_item)
        self.assertEqual(convert.call_count, 2)
        self.assertEqual(convert.call_args.kwargs['input_representation'], checkpoints.CUP_TICK_DELTA_REPRESENTATION)
        self.assertEqual(reply['wire_action_representation'], checkpoints.CUP_REPRESENTATION)
        self.assertEqual(reply['model_action_representation'], checkpoints.CUP_TICK_DELTA_REPRESENTATION)
        for index in (0, 1, 9, 49):
            expected = np.asarray(item['cup_relative_state'][:6]) + tick[:6].numpy() * (index + 1)
            np.testing.assert_allclose(reply['actions'][index][:6], expected, atol=1e-6)
            second_expected = np.asarray(second_item['cup_relative_state'][:6]) + tick[:6].numpy() * (index + 1)
            np.testing.assert_allclose(second_reply['actions'][index][:6], second_expected, atol=1e-6)
            self.assertAlmostEqual(reply['actions'][index][6], .7, places=6)
            self.assertAlmostEqual(second_reply['actions'][index][6], .7, places=6)
        model = FakeInference()
        model.info['action_representation'] = checkpoints.CUP_TICK_DELTA_REPRESENTATION
        sink = io.StringIO()
        service.serve_stream(model, io.StringIO('{"op":"shutdown"}\n'), sink)
        ready = json.loads(sink.getvalue())
        self.assertEqual(ready['model_action_representation'], checkpoints.CUP_TICK_DELTA_REPRESENTATION)
        self.assertEqual(ready['wire_action_representation'], checkpoints.CUP_REPRESENTATION)


if __name__ == '__main__':
    unittest.main()
