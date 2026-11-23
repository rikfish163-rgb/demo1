#!/usr/bin/env python3
"""Check native terminal sampling and cup XYZ gradients on recorded inputs."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext, redirect_stdout
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


ROOT = Path('/home/data4/hts/demo1/fr5-learning')
AUDITS = ROOT / 'audits'
INPUT_SCHEMA = 'fr5-cup-timing-replay-inputs-v1'
FIXED_PROFILE = 'state-history-physical-cup-setdown-fixed-v1'
CLOCK = 'observation.cup_action_clock'
FIXED = 'observation.fixed_cup_action_clock'
MEMORY_LIMIT_BYTES = 24 * 1024**3
DEFAULT_CASE_IDS = (
    'original-source0-first-close-offset-6-row-249',
    'original-source2-first-close-offset-6-row-1207',
    'original-neighbor-chunk-00027-row-2780',
)
GRADIENT_GROUPS = ('vision_lora', 'language_lora', 'full_expert', 'action_projections', 'expert_lora')
sys.dont_write_bytecode = True
training_path = Path(__file__).resolve().parents[1] / 'training'
if training_path.is_dir():
    sys.path.insert(0, str(training_path))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def select_cases(inputs, indices=None):
    if inputs.get('schema') != INPUT_SCHEMA or inputs.get('timing_ready') is not True:
        raise ValueError('Expected recorded training-support inputs with timing evidence')
    cases = inputs['cases']
    if indices is None:
        indices = [next(i for i, case in enumerate(cases) if case['case_id'] == identifier)
                   for identifier in DEFAULT_CASE_IDS]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('Expected distinct case indices')
    result = []
    for index in indices:
        if not 0 <= index < len(cases):
            raise ValueError(f'Case index outside inputs: {index}')
        case = cases[index]
        if (case.get('split') != 'train' or case.get('group') != 'train_support'
                or case.get('source_index') not in range(7)
                or not case.get('case_id', '').startswith('original-')
                or 'task_override' in case or len(case.get('targets_camera_aligned', [])) < 3):
            raise ValueError(f'Case {index} is not an original training-support observation')
        result.append((index, case))
    return result


class _BatchPrepared(Exception):
    pass


def capture_request_batch(service, model, request):
    """Capture the service batch before its inference sampler starts."""
    original = service.prepare_request_batch
    captured = []

    def capture(*args, **kwargs):
        captured.append(original(*args, **kwargs))
        raise _BatchPrepared

    service.prepare_request_batch = capture
    try:
        service.Inference.predict(model, request)
    except _BatchPrepared:
        return captured[0]
    finally:
        service.prepare_request_batch = original
    raise RuntimeError('Service did not call prepare_request_batch')


def snapshot_batch(torch, batch):
    return {key: value.detach().cpu().clone() if torch.is_tensor(value) else deepcopy(value)
            for key, value in batch.items()}


def batch_equal(torch, batch, saved):
    if batch.keys() != saved.keys():
        return False
    return all((torch.is_tensor(value) and value.dtype == saved[key].dtype
                and torch.equal(value.detach().cpu(), saved[key])) if torch.is_tensor(saved[key])
               else value == saved[key] for key, value in batch.items())


def mode_snapshot(policy):
    return {name: module.training for name, module in policy.named_modules()}


def mode_report(policy, before, suffix):
    after = mode_snapshot(policy)
    current = policy.model.embed_suffix
    same_function = (getattr(current, '__func__', current) is getattr(suffix, '__func__', suffix)
                     and getattr(current, '__self__', None) is getattr(suffix, '__self__', None))
    return {'policy_training_before': before[''], 'policy_training_after': after[''],
            'training_modules_before': sum(before.values()), 'training_modules_after': sum(after.values()),
            'module_modes_restored': before == after,
            'changed_modules': [name for name in before if before[name] != after.get(name)],
            'suffix_function_object_restored': same_function,
            'suffix_callable_object_restored': current is suffix}


def rng_snapshot(torch, device):
    states = {'cpu': torch.get_rng_state().clone()}
    if device.type == 'cuda':
        states['cuda'] = torch.cuda.get_rng_state(device).clone()
    return states


def rng_equal(torch, device, before):
    after = rng_snapshot(torch, device)
    return {key + '_unchanged': torch.equal(state, after[key]) for key, state in before.items()}


class MemoryLimitExceeded(RuntimeError):
    pass


@contextmanager
def measure(torch, device, evidence, name):
    cuda = device.type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    completed = False
    try:
        yield
        completed = True
    finally:
        if cuda:
            torch.cuda.synchronize(device)
        stats = {'seconds': time.monotonic() - start,
                 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if cuda else None,
                 'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if cuda else None}
        evidence[name] = stats
        if completed and cuda and max(stats['peak_allocated_bytes'], stats['peak_reserved_bytes']) > MEMORY_LIMIT_BYTES:
            raise MemoryLimitExceeded(f'{name} exceeded the 24 GiB memory budget')


def finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def compare_outputs(torch, native, terminal):
    left, right = native.detach().cpu(), terminal.detach().cpu()
    shapes = tuple(left.shape) == tuple(right.shape) == (1, 50, 7)
    result = {'native_shape': list(left.shape), 'terminal_shape': list(right.shape),
              'native_dtype': str(left.dtype), 'terminal_dtype': str(right.dtype),
              'full_50x7': shapes, 'dtype_equal': left.dtype == right.dtype,
              'native_finite': bool(torch.isfinite(left).all()),
              'terminal_finite': bool(torch.isfinite(right).all()),
              'elementwise_equal': shapes and torch.equal(left, right)}
    if shapes:
        different = left != right
        result.update(compared_elements=left.numel(), unequal_elements=int(different.sum()),
                      max_abs_difference=finite_number((left.double() - right.double()).abs().max()),
                      first_differences=[{'index': index, 'native': finite_number(left[tuple(index)]),
                                          'terminal': finite_number(right[tuple(index)])}
                                         for index in different.nonzero().tolist()[:8]])
    return result


def gradient_statistics(torch, training, policy):
    groups = {name: {'parameter_tensors': 0, 'trainable_tensors': 0, 'trainable_elements': 0,
                     'gradient_tensors': 0, 'missing_gradient_tensors': 0, 'zero_gradient_tensors': 0,
                     'nonfinite_gradient_tensors': 0, 'finite_nonzero_gradient_tensors': 0,
                     'finite_nonzero_gradient_elements': 0, 'gradient_l2_norm': 0.}
              for name in GRADIENT_GROUPS}
    unexpected_trainable, unexpected_gradients = [], []
    for name, parameter in policy.named_parameters():
        if training.is_vision_adapter(name):
            kind = 'vision_lora'
        elif '.paligemma.model.language_model.' in name and '.lora_' in name:
            kind = 'language_lora'
        elif training.is_expert_base(name):
            kind = 'full_expert'
        elif name.startswith(training.PROJECTION_PREFIXES):
            kind = 'action_projections'
        elif '.gemma_expert.model.' in name and '.lora_' in name:
            kind = 'expert_lora'
        else:
            if parameter.requires_grad:
                unexpected_trainable.append(name)
            if parameter.grad is not None:
                unexpected_gradients.append(name)
            continue
        value = groups[kind]
        value['parameter_tensors'] += 1
        value['trainable_tensors'] += int(parameter.requires_grad)
        value['trainable_elements'] += parameter.numel() if parameter.requires_grad else 0
        grad = parameter.grad
        if grad is None:
            value['missing_gradient_tensors'] += int(parameter.requires_grad)
            continue
        value['gradient_tensors'] += 1
        grad = grad.detach()
        if not bool(torch.isfinite(grad).all()):
            value['nonfinite_gradient_tensors'] += 1
            value['gradient_l2_norm'] = None
            continue
        nonzero = int(torch.count_nonzero(grad))
        value['zero_gradient_tensors'] += int(nonzero == 0)
        value['finite_nonzero_gradient_tensors'] += int(nonzero > 0)
        value['finite_nonzero_gradient_elements'] += nonzero
        if value['gradient_l2_norm'] is not None:
            value['gradient_l2_norm'] = finite_number(math.hypot(value['gradient_l2_norm'], float(grad.float().norm())))
    required = all(groups[name]['trainable_tensors'] > 0
                   and groups[name]['finite_nonzero_gradient_tensors'] > 0
                   and groups[name]['nonfinite_gradient_tensors'] == 0 for name in GRADIENT_GROUPS[:-1])
    expert = groups['expert_lora']
    frozen = expert['parameter_tensors'] > 0 and expert['trainable_tensors'] == expert['gradient_tensors'] == 0
    return {'groups': groups, 'required_groups_finite_nonzero': required,
            'expert_lora_frozen_without_gradients': frozen,
            'unexpected_trainable': unexpected_trainable, 'unexpected_gradients': unexpected_gradients,
            'passed': required and frozen and not unexpected_trainable and not unexpected_gradients}


def terminal_targets(torch, relative, case, normalization, device):
    targets = case['targets_camera_aligned'][:3]
    if [point['tick'] for point in targets] != [1, 2, 3]:
        raise ValueError('Expected the original first three camera-aligned ticks')
    absolute = torch.tensor([[point['cup_relative_action'] for point in targets]], dtype=torch.float32, device=device)
    anchor = torch.tensor([case['request']['cup_relative_state']], dtype=torch.float32, device=device)
    delta = relative.to_relative_actions(absolute, anchor, input_representation=relative.CUP_ABSOLUTE, per_tick=True)
    q01 = torch.tensor(normalization['action']['q01'], dtype=torch.float32, device=device)
    q99 = torch.tensor(normalization['action']['q99'], dtype=torch.float32, device=device)
    if not bool((q99 > q01).all()):
        raise ValueError('Expected positive saved action quantile spans')
    return 2. * (delta - q01) / (q99 - q01) - 1., (q99 - q01) / 2.


def probe_batch(torch, training, policy, batch, noise, targets, action_scale, weights, normalizer):
    """Compare both complete samplers, then backpropagate only terminal XYZ loss."""
    from stream_terminal_loss import sample_terminal_actions, sampling_mode, terminal_xyz_penalty

    device = noise.device
    evidence = {'status': 'failed', 'measurements': {}, 'rng': {}, 'sampler_calls': [],
                'native_actions_normalized': None, 'terminal_actions_normalized': None}
    before = mode_snapshot(policy)
    suffix = policy.model.embed_suffix
    flags = {name: parameter.requires_grad for name, parameter in policy.named_parameters()}
    saved_batch, saved_noise = snapshot_batch(torch, batch), noise.detach().clone()
    initial_rng = rng_snapshot(torch, device)
    original_sample = policy.model.sample_actions
    native_sample = original_sample.__wrapped__
    stage = 'native_predict'

    def record(received):
        evidence['sampler_calls'].append({'stage': stage, 'noise_shape': list(received.shape),
            'noise_dtype': str(received.dtype), 'noise_elementwise_equal': torch.equal(received.detach(), saved_noise),
            'grad_enabled': torch.is_grad_enabled(), 'all_modules_eval': not any(mode_snapshot(policy).values())})

    def native(*args, **kwargs):
        record(kwargs['noise'])
        return original_sample(*args, **kwargs)

    def differentiable(model, *args, **kwargs):
        record(kwargs['noise'])
        return native_sample(model, *args, **kwargs)

    native.__wrapped__ = differentiable
    policy.model.sample_actions = native
    policy.zero_grad(set_to_none=True)
    try:
        if batch.get(FIXED) is not True or tuple(batch[CLOCK].shape) != (1, 25):
            raise ValueError('Expected the original fixed 25-channel cup clock')
        autocast = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        rng = rng_snapshot(torch, device)
        with measure(torch, device, evidence['measurements'], stage):
            with sampling_mode(policy), training.action_clock_condition(policy, batch), torch.no_grad(), autocast():
                native_output = policy.predict_action_chunk(batch, noise=noise.detach().clone())
        evidence['rng'][stage] = rng_equal(torch, device, rng)
        evidence['native_restoration'] = mode_report(policy, before, suffix)
        evidence['native_requires_grad'] = native_output.requires_grad
        evidence['native_has_grad_fn'] = native_output.grad_fn is not None
        evidence['native_actions_normalized'] = [
            [finite_number(value) for value in row] for row in native_output.detach().cpu()[0]]
        suffix = policy.model.embed_suffix
        stage = 'terminal_forward'
        rng = rng_snapshot(torch, device)
        with measure(torch, device, evidence['measurements'], stage), autocast():
            terminal = sample_terminal_actions(policy, batch, noise)
        evidence['rng'][stage] = rng_equal(torch, device, rng)
        evidence['terminal_forward_restoration'] = mode_report(policy, before, suffix)
        evidence['comparison'] = compare_outputs(torch, native_output, terminal)
        evidence['terminal_requires_grad'] = terminal.requires_grad
        evidence['terminal_has_grad_fn'] = terminal.grad_fn is not None
        evidence['terminal_actions_normalized'] = [
            [finite_number(value) for value in row] for row in terminal.detach().cpu()[0]]
        if (not evidence['comparison']['full_50x7'] or not evidence['comparison']['terminal_finite']
                or not evidence['comparison']['native_finite']):
            raise ValueError('Samplers did not return finite full 50x7 actions')
        error = terminal[:, :3] - targets
        loss = terminal_xyz_penalty(error, action_scale, weights) / normalizer
        evidence['terminal_loss'] = finite_number(loss.detach())
        evidence['target_prefix_normalized'] = targets.detach().cpu()[0].tolist()
        stage = 'terminal_backward'
        rng = rng_snapshot(torch, device)
        with measure(torch, device, evidence['measurements'], stage), autocast():
            loss.backward()
        evidence['rng'][stage] = rng_equal(torch, device, rng)
        with measure(torch, device, evidence['measurements'], 'gradient_statistics'):
            evidence['gradients'] = gradient_statistics(torch, training, policy)
        comparison = evidence['comparison']
        if (comparison['elementwise_equal'] and comparison['dtype_equal'] and comparison['native_finite']
                and not evidence['native_requires_grad'] and not evidence['native_has_grad_fn']
                and evidence['terminal_requires_grad'] and evidence['terminal_has_grad_fn']
                and evidence['terminal_loss'] is not None
                and evidence['gradients']['passed']):
            evidence['status'] = 'passed'
    except Exception as exc:
        evidence['status'] = ('failed_oom' if isinstance(exc, torch.cuda.OutOfMemoryError)
                              else 'failed_memory_limit' if isinstance(exc, MemoryLimitExceeded) else 'failed')
        evidence['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        policy.model.sample_actions = original_sample
        evidence['restoration'] = mode_report(policy, before, suffix)
        evidence['requires_grad_preserved'] = flags == {
            name: parameter.requires_grad for name, parameter in policy.named_parameters()}
        evidence['batch_elementwise_unchanged'] = batch_equal(torch, batch, saved_batch)
        evidence['explicit_noise_unchanged'] = torch.equal(noise.detach(), saved_noise)
        evidence['sampler_object_restored'] = policy.model.sample_actions is original_sample
        evidence['rng']['overall'] = rng_equal(torch, device, initial_rng)
        policy.zero_grad(set_to_none=True)
    restorations = [value for key, value in evidence.items() if key.endswith('restoration')]
    evidence['invariants_passed'] = (all(value['module_modes_restored'] and value['suffix_function_object_restored']
                                       for value in restorations)
        and evidence['requires_grad_preserved'] and evidence['batch_elementwise_unchanged']
        and evidence['explicit_noise_unchanged'] and evidence['sampler_object_restored']
        and bool(evidence['sampler_calls'])
        and all(call['noise_elementwise_equal'] and call['all_modules_eval']
                and call['grad_enabled'] == (call['stage'] != 'native_predict') for call in evidence['sampler_calls'])
        and all(all(value.values()) for value in evidence['rng'].values()))
    if evidence['status'] == 'passed' and not evidence['invariants_passed']:
        evidence['status'] = 'failed'
    return evidence


def load_model(checkpoint, gpu, report):
    import train_stream_pi05 as training
    import serve_stream_pi05 as service
    import relative_stream_actions as relative

    training.legacy.require_runtime_versions()
    training.legacy.configure_environment()
    selected = training.legacy.gpu_status(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(selected['index'])
    import torch
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05 import processor_pi05  # noqa: F401
    from lerobot.processor import DataProcessorPipeline

    torch.set_num_threads(4)
    device = torch.device('cuda')
    properties = torch.cuda.get_device_properties(device)
    report['gpu'] = {**selected, 'name': properties.name, 'total_memory_bytes': properties.total_memory}
    config = PI05Config.from_pretrained(checkpoint, local_files_only=True, cli_overrides=['--device=cpu'])
    saved = read_json(checkpoint / 'training-report.json')
    adaptation = read_json(checkpoint / 'adaptation.json')
    if (config.chunk_size != 50 or config.n_action_steps != 3 or config.num_inference_steps != 10
            or config.use_relative_actions is not False or config.freeze_vision_encoder is not False
            or config.train_expert_only is not False
            or adaptation['mode'] != training.VISION_EXPERT_ADAPTATION
            or saved['action_representation'] != relative.CUP_TICK_RELATIVE
            or saved['observation_context_profile'] != FIXED_PROFILE):
        raise ValueError('Expected the uncached fixed-clock n3 vision/language/full-expert checkpoint at 10 steps')
    report['runtime'] = {'python': sys.executable, 'torch': torch.__version__,
                         'training_source': training.__file__, 'service_source': service.__file__,
                         'terminal_source': str(Path(training.__file__).with_name('stream_terminal_loss.py')),
                         'num_inference_steps': config.num_inference_steps,
                         'chunk_size': config.chunk_size, 'n_action_steps': config.n_action_steps,
                         'adaptation_mode': adaptation['mode']}
    with measure(torch, device, report['measurements'], 'load_policy'):
        base = Path(read_json(checkpoint / 'stream-manifest.json')['base']['weights_path'])
        policy = training.load_policy(config, base, checkpoint=checkpoint)
        training.legacy.gpu_status(selected['index'])
        policy.to(device).train()
        config.device = 'cuda'
        training.legacy.register_padding_step()
        pre = DataProcessorPipeline.from_pretrained(checkpoint, config_filename='policy_preprocessor.json',
            local_files_only=True, overrides={'tokenizer_processor': {'tokenizer_name': str(checkpoint / 'tokenizer')},
                                               'device_processor': {'device': 'cuda'}})
    model = SimpleNamespace(policy=policy, training=training, pre=pre, torch=torch, info=saved)
    return model, service, relative, read_json(checkpoint / 'normalization.json')


def run(args):
    output = args.output.resolve()
    if not output.is_relative_to(AUDITS) or output.exists():
        raise ValueError('Output must be a new file under the server audits directory')
    inputs = read_json(args.inputs)
    selected = select_cases(inputs, args.case_index)
    checkpoint = args.checkpoint.resolve()
    started = time.monotonic()
    report = {'schema': 'fr5-cup-terminal-gradients-v1', 'status': 'failed',
              'checkpoint': str(checkpoint), 'inputs': str(args.inputs.resolve()), 'seed': args.seed,
              'case_indices': [index for index, _ in selected], 'memory_limit_bytes': MEMORY_LIMIT_BYTES,
              'robot_commands': 0, 'optimizer_steps': 0, 'measurements': {}, 'cases': []}
    try:
        with redirect_stdout(sys.stderr):
            model, service, relative, normalization = load_model(checkpoint, args.gpu, report)
        torch, policy = model.torch, model.policy
        saved = model.info
        if str(Path(inputs['dataset']).resolve()) != str(Path(saved['dataset_path']).resolve()):
            raise ValueError('Training-support inputs and checkpoint dataset differ')
        train_sources = {row['source_index'] for row in saved['source_provenance']['episodes'] if row['split'] == 'train'}
        if any(case['source_index'] not in train_sources for _, case in selected):
            raise ValueError('Selected observation is outside the checkpoint training sources')
        for index, case in selected:
            request = deepcopy(case['request'])
            batch = capture_request_batch(service, model, request)
            seed = args.seed + index
            generator = torch.Generator(device=batch['observation.state'].device).manual_seed(seed)
            noise = torch.randn((1, 50, policy.config.max_action_dim), dtype=torch.float32,
                                device=batch['observation.state'].device, generator=generator)
            targets, scale = terminal_targets(torch, relative, case, normalization, noise.device)
            weights = torch.ones_like(targets)
            weights[:, 0] = saved['first_action_weight']
            weights[:, 1:] = saved['executed_prefix_weight']
            normalizer = model.training.loss_normalizer([50], saved['first_action_weight'], saved['gripper_weight'],
                prefix_steps=3, prefix_weight=saved['executed_prefix_weight'], vertical_weight=saved['vertical_weight'])
            evidence = probe_batch(torch, model.training, policy, batch, noise, targets, scale, weights, normalizer)
            evidence.update(case_index=index, case_id=case['case_id'], source_index=case['source_index'],
                row_index=case['row_index'], source_frame_index=case['source_frame_index'], kind=case['kind'], seed=seed,
                request_unchanged=request == case['request'], task=case['request']['task'],
                state=case['request']['state'], cup_relative_state=case['request']['cup_relative_state'],
                observation_history=case['request']['observation_history'],
                cup_tracking=case['request']['cup_tracking'], cup_timing=case['request']['cup_timing'],
                effective_observation_context_profile=saved['observation_context_profile'],
                preprocessed_tensor_shapes={key: list(value.shape) for key, value in batch.items() if torch.is_tensor(value)},
                cup_action_clock=batch[CLOCK].detach().cpu().tolist(), fixed_cup_action_clock=batch[FIXED],
                action_scale=scale.detach().cpu().tolist(), temporal_weights=weights[0, :, 0].detach().cpu().tolist(),
                loss_normalizer=normalizer, targets_camera_aligned=case['targets_camera_aligned'][:3])
            if evidence['status'] == 'passed' and not evidence['request_unchanged']:
                evidence['status'] = 'failed'
            report['cases'].append(evidence)
            print(json.dumps({'case_index': index, 'source_index': case['source_index'],
                              'status': evidence['status'], 'terminal_loss': evidence.get('terminal_loss')},
                             ensure_ascii=False), file=sys.stderr, flush=True)
            if evidence['status'] in ('failed_oom', 'failed_memory_limit') or 'error' in evidence:
                report['status'] = evidence['status']
                break
        else:
            report['status'] = 'passed' if all(case['status'] == 'passed' for case in report['cases']) else 'failed'
    except Exception as exc:
        report['status'] = ('failed_oom' if type(exc).__name__ == 'OutOfMemoryError'
                            else 'failed_memory_limit' if isinstance(exc, MemoryLimitExceeded) else 'failed')
        report['error'] = f'{type(exc).__name__}: {exc}'
    report['seconds'] = time.monotonic() - started
    measurements = list(report['measurements'].values()) + [stats for case in report['cases']
                                                           for stats in case['measurements'].values()]
    for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
        values = [measurement[key] for measurement in measurements if measurement[key] is not None]
        report[key] = max(values) if values else None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(output), 'status': report['status'], 'cases_completed': len(report['cases']),
                      'peak_allocated_bytes': report['peak_allocated_bytes'],
                      'peak_reserved_bytes': report['peak_reserved_bytes'], 'seconds': report['seconds']}, ensure_ascii=False))
    return 0 if report['status'] == 'passed' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-index', type=int, action='append')
    parser.add_argument('--seed', type=int, default=20261116)
    return run(parser.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
