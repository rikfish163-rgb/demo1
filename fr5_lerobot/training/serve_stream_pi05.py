"""Persistent stdio inference for the seven-dimensional FR5 stream policy.

This process does not connect to the robot. One JSON line returns a native
50-step chunk, and an explicit reset operation starts a new episode.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import redirect_stdout
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import time

import stream_checkpoint as checkpoints
import stream_prompt_integrity as prompt_integrity
import stream_observation_history as observation_history

MAX_LINE = 4 * 1024 * 1024


def load_bundle_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_contract(root=None):
    candidates = ([Path(root) / 'stream_contract.py'] if root else
                  [Path(__file__).with_name('stream_contract.py'),
                   Path(__file__).resolve().parents[1] / 'src/lerobot_robot_fairino/stream_contract.py'])
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location('fr5_stream_contract', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.contract()
    raise FileNotFoundError('FR5 stream contract not found')


def inspect_checkpoint(path, *, verify_base=True):
    root = Path(path).resolve()
    result = checkpoints.verify_bundle(root, verify_base=verify_base)
    contract = load_contract(root)
    if contract != result['manifest'].get('policy_contract') or contract != load_contract():
        raise ValueError('Stream checkpoint and runtime contracts differ')
    config = json.loads((root / 'config.json').read_text())
    if (config.get('type') != 'pi05' or config.get('chunk_size') != contract['chunk_size']
            or config.get('use_relative_actions') is not False
            or config.get('input_features', {}).get('observation.state', {}).get('shape') != [8]
            or config.get('output_features', {}).get('action', {}).get('shape') != [7]):
        raise ValueError('Stream model configuration does not match contract')
    result.update(contract=contract, path=str(root), state_shape=[8], action_shape=[50, 7])
    return result


def validate_request(request):
    if not isinstance(request, dict) or request.get('schema') != 'fr5-stream-request-v1':
        raise ValueError('Expected fr5-stream-request-v1')
    if request.get('policy_contract', 'fr5-joint-position-stream-v1') != 'fr5-joint-position-stream-v1':
        raise ValueError('Incorrect stream policy contract')
    identifier, task, state = request.get('id'), request.get('task'), request.get('state')
    if (not isinstance(identifier, str) or not 1 <= len(identifier) <= 128
            or not isinstance(task, str) or not task.strip() or len(task) > 512
            or not isinstance(state, list) or len(state) != 8
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in state)):
        raise ValueError('Invalid id, instruction or state')
    if not 0 <= state[6] <= 1 or state[7] not in (0, 1):
        raise ValueError('Invalid gripper fraction or validity flag')
    stamp, clock = request.get('source_time'), request.get('source_clock')
    if (type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp < 0
            or not isinstance(clock, str) or not clock.strip() or len(clock) > 128):
        raise ValueError('Invalid observation timestamp or clock')
    try:
        raw = base64.b64decode(request.get('rgb_jpeg_base64', ''), validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError('Invalid JPEG encoding') from exc
    if not 1 <= len(raw) <= 3 * 1024 * 1024:
        raise ValueError('Invalid JPEG size')
    import numpy as np
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != 'JPEG' or image.width * image.height > 4096 * 2160:
            raise ValueError('Expected a camera JPEG')
        rgb = np.asarray(image.convert('RGB')).copy()
    return identifier, task, state, rgb


def validate_normalization(root):
    import torch
    from safetensors.torch import load_file
    expected = json.loads((Path(root) / 'normalization.json').read_text())
    states = sorted(Path(root).glob('policy_*processor_step_*.safetensors'))
    if len(states) < 2:
        raise ValueError('Missing processor statistics')
    for path in states:
        actual = load_file(str(path), device='cpu')
        for feature, size in (('observation.state', 8), ('action', 7)):
            for key in ('q01', 'q99'):
                wanted = torch.tensor(expected[feature][key], dtype=torch.float32)
                got = actual.get(f'{feature}.{key}')
                if wanted.shape != (size,) or got is None or not torch.equal(wanted, got.cpu().float()):
                    raise ValueError(f'Processor statistics differ from bundle: {path.name}')


def actions_for_wire(actions, state, relative, *, input_representation=checkpoints.RELATIVE_REPRESENTATION):
    """Convert each request's postprocessed model actions to absolute wire targets."""
    if relative is None:
        if input_representation in (checkpoints.CUP_DELTA_REPRESENTATION, checkpoints.CUP_TICK_DELTA_REPRESENTATION):
            raise ValueError('Cup-relative delta checkpoint lacks its loaded action converter')
        return actions
    import torch
    anchor = torch.tensor([state], dtype=torch.float32)
    return relative.to_absolute_actions(actions, anchor, input_representation=input_representation)


def prepare_request_batch(training, pre, state, rgb, task, image_profile,
                          observation_context_profile=observation_history.NONE_PROFILE, history=None,
                          *, initial_cup_uv=None, cup_tracking=None, cup_timing=None):
    """Use the bundle's training image path before its saved processor."""
    import torch
    raw = {'observation.state': torch.tensor([state], dtype=torch.float32),
           'observation.images.front': torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255,
           'task': [observation_history.render_task(task, history, observation_context_profile,
                                                    initial_cup_uv=initial_cup_uv, cup_tracking=cup_tracking,
                                                    cup_timing=cup_timing)]}
    if hasattr(training, 'apply_image_profile'):
        if image_profile == checkpoints.CUP_CENTERED_IMAGE_PROFILE:
            raw = training.apply_image_profile(raw, image_profile, initial_cup_uv=[initial_cup_uv])
        else:
            raw = training.apply_image_profile(raw, image_profile)
    elif image_profile != checkpoints.FULL_IMAGE_PROFILE:
        raise ValueError('Checkpoint lacks its image-profile preprocessing implementation')
    batch = prompt_integrity.preprocess(pre, raw)
    if observation_context_profile in observation_history.CUP_ACTION_PROFILES:
        batch['observation.cup_action_clock'] = torch.tensor([observation_history.action_clock(
            task, cup_timing, cup_tracking=cup_tracking, profile=observation_context_profile)],
                                                            dtype=torch.float32, device=batch['observation.state'].device)
        if observation_context_profile in (observation_history.CUP_SETDOWN_CLEAN_PROFILE,
                                           observation_history.CUP_SETDOWN_FIXED_PROFILE):
            batch['observation.fixed_cup_action_clock'] = True
    return batch


class Inference:
    def __init__(self, checkpoint, gpu=None):
        self.root = Path(checkpoint).resolve()
        self.info = inspect_checkpoint(self.root)
        self.info['training_data_kind'] = json.loads((self.root / 'training-report.json').read_text())['training_data_kind']
        legacy = load_bundle_module('fr5_stream_bundled_legacy', self.root / 'train_pi05_fr5.py')
        legacy.require_runtime_versions()
        legacy.configure_environment()
        selected = legacy.gpu_status(gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(selected['index'])
        import torch
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05 import processor_pi05  # noqa: F401
        from lerobot.processor import DataProcessorPipeline
        from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
        training = load_bundle_module('fr5_stream_bundled_training', self.root / 'train_stream_pi05.py')
        training.legacy = legacy
        self.training = training
        if (self.info['adaptation_mode'] == checkpoints.VISION_EXPERT_ADAPTATION
                and getattr(training, 'VISION_EXPERT_ADAPTATION', None) != checkpoints.VISION_EXPERT_ADAPTATION):
            raise ValueError('Bundled training code does not support the sealed vision adaptation')
        if (self.info['image_profile'] != checkpoints.FULL_IMAGE_PROFILE
                and not hasattr(training, 'apply_image_profile')):
            raise ValueError('Checkpoint lacks its image-profile preprocessing implementation')
        self.torch = torch
        validate_normalization(self.root)
        config = PI05Config.from_pretrained(self.root, local_files_only=True, cli_overrides=['--device=cpu'])
        self.policy = training.load_policy(config, Path(self.info['manifest']['base']['weights_path']),
                                           checkpoint=self.root)
        legacy.gpu_status(selected['index'])
        self.policy.to('cuda').eval().requires_grad_(False)
        config.device = 'cuda'
        legacy.register_padding_step()
        self.pre = DataProcessorPipeline.from_pretrained(
            self.root, config_filename='policy_preprocessor.json', local_files_only=True,
            overrides={'tokenizer_processor': {'tokenizer_name': str(self.root / 'tokenizer')},
                       'device_processor': {'device': 'cuda'}})
        self.post = DataProcessorPipeline.from_pretrained(
            self.root, config_filename='policy_postprocessor.json', local_files_only=True,
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        self.relative = None
        if self.info['action_representation'] in checkpoints.RELATIVE_REPRESENTATIONS:
            self.relative = load_bundle_module('fr5_stream_bundled_relative_actions', self.root / 'relative_stream_actions.py')
            self.relative.assert_manual_pipeline(config, preprocessor=self.pre, postprocessor=self.post)

    def reset(self):
        self.policy.reset()

    def predict(self, request):
        identifier, task, state, rgb = validate_request(request)
        representation = self.info['action_representation']
        wire_mode = (checkpoints.CUP_REPRESENTATION if representation in checkpoints.CUP_REPRESENTATIONS
                     else checkpoints.ABSOLUTE_REPRESENTATION)
        model_state = state
        if wire_mode == checkpoints.CUP_REPRESENTATION:
            model_state = request.get('cup_relative_state')
            if (not isinstance(model_state, list) or len(model_state) != 8
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in model_state)
                    or model_state[6:] != state[6:]):
                raise ValueError('Expected measured cup-relative TCP state and unchanged gripper feedback')
        t = self.torch
        batch = prepare_request_batch(self.training, self.pre, model_state, rgb, task, self.info['image_profile'],
                                      self.info['observation_context_profile'], request.get('observation_history'),
                                      initial_cup_uv=request.get('initial_cup_uv'), cup_tracking=request.get('cup_tracking'),
                                      cup_timing=request.get('cup_timing'))
        if tuple(batch['observation.state'].shape) != (1, 32) or batch.get('action') is not None:
            raise ValueError('Invalid model preprocessor output')
        t.cuda.synchronize()
        cuda_rng_state = t.cuda.get_rng_state().tolist()
        start = time.monotonic()
        with t.inference_mode(), t.autocast('cuda', dtype=t.bfloat16):
            if 'observation.cup_action_clock' in batch:
                normalized = self.training.predict_action_chunk(self.policy, batch)
            else:
                normalized = self.policy.predict_action_chunk(batch)
            actions = self.post(normalized.detach().clone()).cpu().float()
            anchor = (model_state if representation in (checkpoints.CUP_DELTA_REPRESENTATION,
                                                       checkpoints.CUP_TICK_DELTA_REPRESENTATION) else state)
            actions = actions_for_wire(actions, anchor, self.relative, input_representation=representation)
        t.cuda.synchronize()
        if tuple(actions.shape) != (1, 50, 7) or not bool(t.isfinite(actions).all()):
            raise ValueError('Invalid model action chunk')
        return {'schema': 'fr5-stream-reply-v1', 'id': identifier,
                'checkpoint_sha256': self.info['checkpoint_sha256'],
                'policy_contract': self.info['contract']['name'], 'actions': actions[0].tolist(),
                'inference_seconds': time.monotonic() - start,
                'model_action_representation': self.info['action_representation'],
                'wire_action_representation': wire_mode,
                'image_profile': self.info['image_profile'],
                'observation_context_profile': self.info['observation_context_profile'],
                'training_data_kind': self.info['training_data_kind'],
                'sampling': {'cuda_rng_state': cuda_rng_state,
                             'denoising_steps': self.policy.config.num_inference_steps},
                'runtime_source': str(Path(__file__).resolve())}


def serve_stream(model, source, sink):
    def emit(value):
        sink.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        sink.flush()
    if 'n_action_steps' not in model.info:
        raise ValueError('Missing sealed checkpoint n_action_steps')
    n_action_steps = checkpoints.configured_action_steps({'n_action_steps': model.info['n_action_steps']}, 50)
    ready = {'status': 'ready', 'schema': 'fr5-stream-ready-v1',
          'checkpoint_sha256': model.info['checkpoint_sha256'],
          'policy_contract': model.info['contract']['name'], 'state_shape': [8], 'action_shape': [50, 7],
          'n_action_steps': n_action_steps,
          'image_profile': model.info.get('image_profile', checkpoints.FULL_IMAGE_PROFILE),
          'observation_context_profile': model.info.get('observation_context_profile', observation_history.NONE_PROFILE),
          'model_action_representation': model.info.get('action_representation', checkpoints.ABSOLUTE_REPRESENTATION),
          'wire_action_representation': (checkpoints.CUP_REPRESENTATION
              if model.info.get('action_representation') in checkpoints.CUP_REPRESENTATIONS
              else checkpoints.ABSOLUTE_REPRESENTATION),
          'training_data_kind': model.info.get('training_data_kind')}
    if model.info.get('adaptation_mode') == checkpoints.VISION_EXPERT_ADAPTATION:
        ready['adaptation_mode'] = checkpoints.VISION_EXPERT_ADAPTATION
    emit(ready)
    for line in source:
        request = None
        try:
            if len(line) > MAX_LINE:
                raise ValueError('Request exceeds protocol limit')
            request = json.loads(line)
            if isinstance(request, dict) and request.get('op') == 'shutdown':
                break
            if isinstance(request, dict) and request.get('op') == 'reset':
                model.reset()
                emit({'schema': 'fr5-stream-reset-v1', 'id': request.get('id'), 'status': 'reset'})
                continue
            with redirect_stdout(sys.stderr):
                result = model.predict(request)
            emit(result)
        except Exception as exc:
            emit({'status': 'rejected', 'id': request.get('id') if isinstance(request, dict) else None,
                  'error': f'{type(exc).__name__}: {exc}'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    if args.inspect:
        print(json.dumps(inspect_checkpoint(args.checkpoint), ensure_ascii=False, indent=2))
    else:
        with redirect_stdout(sys.stderr):
            model = Inference(args.checkpoint, args.gpu)
        serve_stream(model, sys.stdin, sys.stdout)


if __name__ == '__main__':
    main()
